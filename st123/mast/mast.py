"""MAST query, filter, and download helpers for HST and JWST imaging."""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from typing import Iterable, Optional, Sequence

import numpy as np
import shapely
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Column, Table
from astroquery.mast import Observations
from astroquery.vizier import Vizier

from st123.utils.logging import capture_output
from st123.utils.settings import (
    DEFAULT_DOWNLOAD_LAYOUT,
    DEFAULT_HST_FILTERS,
    DEFAULT_HST_INSTRUMENTS,
    DEFAULT_JWST_INSTRUMENTS,
    HST_PRODUCT_RULES,
)

logger = logging.getLogger(__name__)

# Token successfully passed to ``Observations.login`` in this process.
# Avoids re-authing on every query/download stage in one ``download`` CLI run.
_MAST_SESSION_TOKEN: str | None = None
_MAST_PUBLIC_NOTICE: bool = False

# MAST product-list / download retries (issue #4: silent partial inventory).
_HST_MAST_MAX_ATTEMPTS = 3
_HST_MAST_RETRY_DELAY_SEC = 5.0


def _is_transient_mast_error(exc: BaseException) -> bool:
    """True for timeouts / connection failures worth retrying against MAST."""
    if isinstance(
        exc,
        (
            TimeoutError,
            ConnectionError,
            ConnectionResetError,
            BrokenPipeError,
            OSError,
        ),
    ):
        return True
    text = f'{type(exc).__name__}: {exc}'.lower()
    needles = (
        'timeout',
        'timed out',
        'time out',
        'connection reset',
        'connection aborted',
        'connection refused',
        'temporarily unavailable',
        '503',
        '502',
        '504',
        '429',
        'remote end closed',
        'broken pipe',
    )
    return any(n in text for n in needles)


def _mast_call_with_retries(label: str, fn, *args, **kwargs):
    """
    Call ``fn`` with retries on transient MAST errors.

    Returns
    -------
    tuple
        ``(result, None)`` on success, or ``(None, last_exc)`` after exhausting
        attempts (non-transient errors fail immediately).
    """
    last_exc: Exception | None = None
    for attempt in range(1, _HST_MAST_MAX_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs), None
        except Exception as exc:
            last_exc = exc
            if attempt < _HST_MAST_MAX_ATTEMPTS and _is_transient_mast_error(exc):
                delay = _HST_MAST_RETRY_DELAY_SEC * attempt
                logger.warning(
                    'Transient MAST error for %s (attempt %d/%d): %s; '
                    'retrying in %.1fs',
                    label,
                    attempt,
                    _HST_MAST_MAX_ATTEMPTS,
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue
            break
    return None, last_exc


def resolve_mast_token(token: Optional[str] = None) -> Optional[str]:
    """Resolve a MAST API token from an explicit value or the environment.

    Checks ``token``, then ``MAST_API_TOKEN``, then ``MAST_TOKEN``.
    Create a token at https://auth.mast.stsci.edu/info

    Parameters
    ----------
    token : str or None, optional
        Explicit API token. When ``None`` or empty, environment variables are
        consulted.

    Returns
    -------
    str or None
        Resolved token string, or ``None`` when no token is available.
    """
    for candidate in (token, os.environ.get('MAST_API_TOKEN'), os.environ.get('MAST_TOKEN')):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return None


def reset_mast_login_state() -> None:
    """Clear the in-process MAST login cache (tests / rare re-auth)."""
    global _MAST_SESSION_TOKEN, _MAST_PUBLIC_NOTICE
    _MAST_SESSION_TOKEN = None
    _MAST_PUBLIC_NOTICE = False


def prepare_mast_auth(token: Optional[str] = None) -> bool:
    """
    Optionally authenticate to MAST; never abort a public-data download.

    * No token → log once that only public data will be queried; return False.
    * Token present → login once per process (cached). Login failure warns and
      continues with public data only (return False).

    Returns
    -------
    bool
        True when authenticated for proprietary access.
    """
    global _MAST_PUBLIC_NOTICE
    resolved = resolve_mast_token(token)
    if not resolved:
        if not _MAST_PUBLIC_NOTICE:
            logger.info(
                'No MAST token set (MAST_API_TOKEN / --token); '
                'continuing with public data only'
            )
            _MAST_PUBLIC_NOTICE = True
        return False
    if mast_login(resolved, required=False):
        return True
    logger.warning(
        'MAST token was provided but login failed; '
        'continuing with public data only'
    )
    return False


def mast_login(token: Optional[str] = None, *, required: bool = False) -> bool:
    """
    Authenticate to MAST with an authorization token.

    Uses ``Observations.login(token=...)`` so proprietary products available to
    the token holder can be queried and downloaded (same pattern as hst123).
    See https://auth.mast.stsci.edu/info

    Successful logins are cached for this process so query + download stages
    do not call ``Observations.login`` twice.

    Parameters
    ----------
    token : str or None
        MAST API token. If None/empty, no login is attempted unless
        ``MAST_API_TOKEN`` / ``MAST_TOKEN`` is set in the environment.
    required : bool
        If True, raise ``RuntimeError`` when a token is expected but login fails.

    Returns
    -------
    bool
        True if login was attempted and succeeded (or already cached), else False.
    """
    global _MAST_SESSION_TOKEN
    token = resolve_mast_token(token)
    if not token:
        if required:
            raise RuntimeError(
                'A MAST token is required but none was provided. '
                'Pass token=... / --token, or set MAST_API_TOKEN.'
            )
        return False
    if _MAST_SESSION_TOKEN == token:
        logger.debug('MAST session already authenticated; skipping re-login')
        return True
    try:
        logger.info('Logging in to MAST with token...')
        Observations.login(token=token)
        _MAST_SESSION_TOKEN = token
        logger.info('MAST login successful.')
        return True
    except Exception as exc:
        message = f'could not log in to MAST with input token: {exc}'
        if required:
            raise RuntimeError(message) from exc
        logger.warning('could not log in to MAST with input token: %s', exc)
        return False


def _combine_masks(masks: Sequence[Sequence[bool]]) -> list[bool]:
    return [all(row) for row in zip(*masks)]


def _instrument_contains(obs_table: Table, detectors: Sequence[str]) -> list[bool]:
    return [
        any(det in str(inst).upper() for det in detectors)
        for inst in obs_table['instrument_name']
    ]


def has_supported_hst_science_products(instrument_name: object) -> bool:
    """
    Return True when *instrument_name* has a native product rule.

    Used to drop MAST detectors with no downloadable science files under
    :data:`HST_PRODUCT_RULES` (e.g. unsupported instruments).
    """
    text = str(instrument_name)
    return any(tag in text for _suffix, tag in HST_PRODUCT_RULES)


def galaxy_query_radius(
    coord: SkyCoord,
    vizier_radius: u.Quantity = 5 * u.arcsec,
    vizier_cat: str = 'VII/237/pgc',
    floor: Optional[u.Quantity] = None,
) -> u.Quantity:
    """Estimate a search radius from the PGC logD25 size, with an optional floor.

    Parameters
    ----------
    coord : SkyCoord
        Target sky position used for the VizieR lookup.
    vizier_radius : Quantity, optional
        Cone-search radius for the PGC catalog query (default 5 arcsec).
    vizier_cat : str, optional
        VizieR catalog identifier (default ``'VII/237/pgc'``).
    floor : Quantity or None, optional
        Minimum returned radius. When ``None``, no floor is applied.

    Returns
    -------
    Quantity
        Estimated search radius as an angular :class:`~astropy.units.Quantity`.
    """
    vizier = Vizier(columns=['logD25'])
    result = vizier.query_region(coord, radius=vizier_radius, catalog=vizier_cat)
    logd = result[0]['logD25'][0]
    radius = (10 ** logd * 0.1 * u.arcmin)
    if floor is not None:
        return max(floor, radius)
    return radius


def query_region(coord: SkyCoord, radius: u.Quantity) -> Table:
    """Query MAST Observations around ``coord`` and fill masked values.

    Parameters
    ----------
    coord : SkyCoord
        Center of the cone search.
    radius : Quantity
        Search radius (angular :class:`~astropy.units.Quantity`).

    Returns
    -------
    Table
        MAST observation table with masked cells filled.
    """
    with capture_output():
        obs_table = Observations.query_region(coord, radius=radius)
    return obs_table.filled()


def filter_hst_observations(
    obs_table: Table,
    filters: Optional[Sequence[str]] = None,
    instruments: Sequence[str] = DEFAULT_HST_INSTRUMENTS,
    public_only: bool = True,
    *,
    pipeline_only: bool = True,
) -> Table:
    """Apply standard HST imaging masks to a MAST observation table.

    Parameters
    ----------
    obs_table : Table
        Raw MAST observation table from :func:`query_region`.
    filters : sequence of str or None, optional
        Allowed filter names (e.g. ``'F814W'``). When ``None``, no filter
        restriction is applied.
    instruments : sequence of str, optional
        Instrument substrings matched against ``instrument_name``
        (default :data:`DEFAULT_HST_INSTRUMENTS`). Bare ``ACS`` matches
        ``ACS/WFC``, ``ACS/HRC``, and ``ACS/SBC``.
    public_only : bool, optional
        When ``True``, keep only rows with ``dataRights == 'PUBLIC'``.
    pipeline_only : bool, optional
        When ``True`` (default), keep only ``project == 'HST'`` rows so HAP
        visit / skycell reprocessings (and HLA) are excluded. Those duplicates
        re-list the same ``flc``/``flt`` files or expose only derived products.

    Returns
    -------
    Table
        Filtered table sorted by ``t_min``.
    """
    masks = [
        [str(t).upper() == 'HST' for t in obs_table['obs_collection']],
        [str(p).upper() == 'IMAGE' for p in obs_table['dataproduct_type']],
        _instrument_contains(obs_table, instruments),
        # Keep only detectors with a native product rule (ACS/WFC flc,
        # ACS/HRC+SBC flt, WFC3 UVIS/IR, WFPC2).
        [
            has_supported_hst_science_products(inst)
            for inst in obs_table['instrument_name']
        ],
        [str(f).upper() != 'DETECTION' for f in obs_table['filters']],
        [str(i).upper() != 'CALIBRATION' for i in obs_table['intentType']],
    ]
    if filters is not None:
        allowed = {f.upper() for f in filters}
        masks.append([str(f).upper() in allowed for f in obs_table['filters']])
    if public_only:
        masks.append([str(r).upper() == 'PUBLIC' for r in obs_table['dataRights']])
    # Native calwf3/calacs/calwp2 associations only (not HAP/HLA mosaics).
    if pipeline_only and 'project' in obs_table.colnames:
        masks.append([str(p).upper() == 'HST' for p in obs_table['project']])

    out = obs_table[_combine_masks(masks)]
    out.sort('t_min')
    return out


def filter_jwst_observations(
    obs_table: Table,
    instruments: Sequence[str] = DEFAULT_JWST_INSTRUMENTS,
    public_only: bool = True,
) -> Table:
    """Apply standard JWST imaging masks to a MAST observation table.

    Parameters
    ----------
    obs_table : Table
        Raw MAST observation table from :func:`query_region`.
    instruments : sequence of str, optional
        Instrument substrings matched against ``instrument_name``
        (default :data:`DEFAULT_JWST_INSTRUMENTS`).
    public_only : bool, optional
        When ``True``, keep only rows with ``dataRights == 'PUBLIC'``.

    Returns
    -------
    Table
        Filtered table sorted by ``t_min``.
    """
    masks = [
        [str(t).upper() == 'JWST' for t in obs_table['obs_collection']],
        _instrument_contains(obs_table, instruments),
        [str(f).upper() != 'DETECTION' for f in obs_table['filters']],
        [str(i).upper() != 'CALIBRATION' for i in obs_table['intentType']],
        [str(t).upper() == 'IMAGE' for t in obs_table['dataproduct_type']],
    ]
    if public_only:
        masks.append([str(d).upper() == 'PUBLIC' for d in obs_table['dataRights']])

    out = obs_table[_combine_masks(masks)]
    out.sort('t_min')
    return out


def observation_matches_calib_stage(calib_level: object, stage: int) -> bool:
    """Return True if a MAST ``calib_level`` can satisfy ``stage``.

    MAST uses ``calib_level=-1`` (or missing) when no calibrated products exist
    yet — e.g. APT placeholders or unexecuted visits. Observations that only
    reach a lower level than requested are also not relevant for download.
    Unknown / unparseable levels are treated as potentially relevant so we do
    not silently drop rows that might still have products.

    Parameters
    ----------
    calib_level : object
        MAST ``calib_level`` value (may be masked, ``None``, or non-numeric).
    stage : int
        Minimum calibration stage required (e.g. 2 for CAL, 3 for I2D).

    Returns
    -------
    bool
        ``True`` when the observation may have products at ``stage`` or higher.
    """
    if calib_level is None:
        return True
    try:
        if np.ma.is_masked(calib_level):
            return True
    except Exception:
        pass
    try:
        level = int(calib_level)
    except (TypeError, ValueError):
        return True
    if level < 0:
        return False
    return level >= int(stage)


def filter_jwst_observations_by_stage(obs_table: Table, stage: int) -> Table:
    """Keep JWST observations expected to have products at ``stage`` or higher.

    Rows with ``calib_level=-1`` or otherwise below ``stage`` are dropped so
    download does not report attempts on programs with no matching products.

    Parameters
    ----------
    obs_table : Table
        JWST observation table (may be empty or ``None``).
    stage : int
        Minimum calibration stage required (e.g. 2 for CAL, 3 for I2D).

    Returns
    -------
    Table
        Filtered table, or the input unchanged when empty / missing
        ``calib_level``.
    """
    if obs_table is None or len(obs_table) == 0:
        return obs_table
    if 'calib_level' not in obs_table.colnames:
        return obs_table
    keep = [
        observation_matches_calib_stage(level, stage)
        for level in obs_table['calib_level']
    ]
    return obs_table[keep]


def query_hst(
    coord: SkyCoord,
    radius: Optional[u.Quantity] = None,
    filters: Optional[Sequence[str]] = DEFAULT_HST_FILTERS,
    instruments: Sequence[str] = DEFAULT_HST_INSTRUMENTS,
    use_galaxy_size: bool = False,
    public_only: bool = True,
    token: Optional[str] = None,
) -> Table:
    """Query and filter HST imaging observations near ``coord``.

    If a MAST token is provided (argument or environment), authenticate once and
    include proprietary observations (``public_only`` is forced ``False``).
    Without a token (or if login fails), only public data are queried.
    By default only native pipeline rows (``project == 'HST'``) are kept so
    HAP/HLA reprocessings of the same visit are excluded.

    Parameters
    ----------
    coord : SkyCoord
        Target sky position.
    radius : Quantity or None, optional
        Search radius. When ``None``, uses :func:`galaxy_query_radius` if
        ``use_galaxy_size`` is ``True``, otherwise 5 arcmin.
    filters : sequence of str or None, optional
        Allowed HST filters (default :data:`DEFAULT_HST_FILTERS`).
    instruments : sequence of str, optional
        Instrument substrings (default :data:`DEFAULT_HST_INSTRUMENTS`).
    use_galaxy_size : bool, optional
        When ``True`` and ``radius`` is ``None``, derive radius from PGC size.
    public_only : bool, optional
        When ``True``, exclude proprietary rows unless a token is resolved.
    token : str or None, optional
        MAST API token; also read from ``MAST_API_TOKEN`` / ``MAST_TOKEN``.

    Returns
    -------
    Table
        Filtered HST imaging observation table.
    """
    if prepare_mast_auth(token):
        public_only = False
    if radius is None:
        if use_galaxy_size:
            radius = galaxy_query_radius(coord)
        else:
            radius = 5 * u.arcmin
    logger.info(
        'MAST Observations.query_region center=%s radius=%s',
        coord.to_string('decimal'),
        radius,
    )
    obs_table = query_region(coord, radius)
    return filter_hst_observations(
        obs_table, filters=filters, instruments=instruments, public_only=public_only
    )


def query_jwst(
    coord: SkyCoord,
    radius: Optional[u.Quantity] = None,
    instruments: Sequence[str] = DEFAULT_JWST_INSTRUMENTS,
    use_galaxy_size: bool = False,
    public_only: bool = True,
    token: Optional[str] = None,
) -> Table:
    """Query and filter JWST imaging observations near ``coord``.

    If a MAST token is provided (argument or ``MAST_API_TOKEN`` / ``MAST_TOKEN``),
    authenticate first and include proprietary observations (``public_only`` is
    forced ``False``). Without a token, only public data are returned by default.
    A missing or failed token never aborts the query.

    Parameters
    ----------
    coord : SkyCoord
        Target sky position.
    radius : Quantity or None, optional
        Search radius. When ``None``, uses :func:`galaxy_query_radius` with a
        5 arcmin floor if ``use_galaxy_size`` is ``True``, otherwise 5 arcmin.
    instruments : sequence of str, optional
        Instrument substrings (default :data:`DEFAULT_JWST_INSTRUMENTS`).
    use_galaxy_size : bool, optional
        When ``True`` and ``radius`` is ``None``, derive radius from PGC size.
    public_only : bool, optional
        When ``True``, exclude proprietary rows unless a token is resolved.
    token : str or None, optional
        MAST API token; also read from ``MAST_API_TOKEN`` / ``MAST_TOKEN``.

    Returns
    -------
    Table
        Filtered JWST imaging observation table.
    """
    if prepare_mast_auth(token):
        public_only = False
    if radius is None:
        if use_galaxy_size:
            radius = galaxy_query_radius(coord, floor=5 * u.arcmin)
        else:
            radius = 5 * u.arcmin
    logger.info(
        'MAST Observations.query_region center=%s radius=%s',
        coord.to_string('decimal'),
        radius,
    )
    obs_table = query_region(coord, radius)
    return filter_jwst_observations(
        obs_table, instruments=instruments, public_only=public_only
    )


def is_hst_science_product(filename: str, instrument: str) -> bool:
    """Return True if ``filename`` is a usable HST science product for ``instrument``.

    Parameters
    ----------
    filename : str
        MAST product filename (e.g. ``*.flc.fits``).
    instrument : str
        MAST ``instrument_name`` string for the parent observation.

    Returns
    -------
    bool
        ``True`` when ``filename`` matches :data:`HST_PRODUCT_RULES` for the
        instrument.
    """
    return any(
        suffix in filename and tag in instrument
        for suffix, tag in HST_PRODUCT_RULES
    )


def collect_hst_products(obs_table: Table) -> Optional[Table]:
    """Build a product table of HST flt/flc/c0m/c1m science files from observations.

    Parameters
    ----------
    obs_table : Table
        HST observation table (rows from :func:`query_hst` or similar).

    Returns
    -------
    Table or None
        Combined science-product table, or ``None`` when no products were found.
    """
    productlist = None
    for obs in obs_table:
        try:
            product_list = Observations.get_product_list(obs)
            product_list = product_list[product_list['type'] == 'S']
        except Exception:
            logger.error('MAST is not working currently\nTry again later...')
            continue

        instrument = obs['instrument_name']
        product_list.add_column(Column([instrument] * len(product_list), name='instrument_name'))
        product_list.add_column(Column([obs['s_ra']] * len(product_list), name='ra'))
        product_list.add_column(Column([obs['s_dec']] * len(product_list), name='dec'))

        for prod in product_list:
            if not is_hst_science_product(prod['productFilename'], instrument):
                continue
            if productlist is None:
                productlist = Table(prod)
            else:
                productlist.add_row(prod)
    return productlist


def normalize_filter_name(filt: str) -> str:
    """Turn MAST filter strings like ``F560W;CLEAR`` into a directory name.

    Parameters
    ----------
    filt : str
        MAST ``filters`` field (may contain semicolon-separated components).

    Returns
    -------
    str
        Uppercase, filesystem-safe filter directory name.
    """
    name = str(filt).split(';')[0].strip().upper()
    name = re.sub(r'[^A-Z0-9_\-]+', '_', name)
    return name or 'UNKNOWN'


def normalize_telescope_dirname(
    obs_collection: object | None = None,
    *,
    default: str = 'JWST',
) -> str:
    """Return a telescope directory name (e.g. ``JWST``, ``HST``, ``Roman``).

    Parameters
    ----------
    obs_collection : object or None, optional
        MAST ``obs_collection`` value (or similar telescope label).
    default : str, optional
        Fallback name when ``obs_collection`` is empty (default ``'JWST'``).

    Returns
    -------
    str
        Normalized telescope directory name.
    """
    text = str(obs_collection or default).strip().upper()
    if 'JWST' in text or text in ('JWST', 'JW'):
        return 'JWST'
    if text.startswith('HST') or 'HST' in text:
        return 'HST'
    if 'ROMAN' in text:
        return 'Roman'
    if 'EUCLID' in text:
        return 'Euclid'
    return (str(obs_collection or default).split('/')[0].strip() or default)


def normalize_instrument_dirname(instrument_name: object | None = None) -> str:
    """Return an instrument directory name under the telescope root.

    Examples: ``MIRI``, ``NIRCam``, ``NIRISS``, ``ACS``, ``WFC3``, ``WFI``.

    Parameters
    ----------
    instrument_name : object or None, optional
        MAST ``instrument_name`` string (may include slash-separated components).

    Returns
    -------
    str
        Normalized instrument directory name.
    """
    text = str(instrument_name or 'UNKNOWN').upper()
    if 'NIRCAM' in text:
        return 'NIRCam'
    if 'NIRISS' in text:
        return 'NIRISS'
    if 'MIRI' in text:
        return 'MIRI'
    if 'WFC3' in text:
        return 'WFC3'
    if 'WFPC2' in text:
        return 'WFPC2'
    if 'ACS' in text:
        return 'ACS'
    if 'WFI' in text or 'ROMAN' in text:
        return 'WFI'
    if 'VIS' in text and 'EUCLID' in text:
        return 'VIS'
    if 'NISP' in text:
        return 'NISP'
    return str(instrument_name).split('/')[0].strip() or 'UNKNOWN'


def filter_jwst_products(
    product_list: Table,
    stage: int = 2,
    *,
    mirimage_only: bool = False,
) -> Table:
    """Filter JWST products to science CAL (stage 2) or I2D (stage 3) files.

    Parameters
    ----------
    product_list : Table
        Raw MAST product list from ``Observations.get_product_list``.
    stage : int, optional
        Calibration stage to keep: ``2`` for CAL, ``3`` for I2D (default 2).
    mirimage_only : bool, optional
        When ``True``, keep only filenames containing ``mirimage``.

    Returns
    -------
    Table
        Filtered product table.

    Raises
    ------
    ValueError
        If ``stage`` is not ``2`` or ``3``.
    """
    masks = [[str(p).upper() == 'SCIENCE' for p in product_list['productType']]]
    if mirimage_only:
        masks.append(
            ['mirimage' in str(name).lower() for name in product_list['productFilename']]
        )
    if stage == 2:
        masks.append([t == 'CAL' for t in product_list['productSubGroupDescription']])
        masks.append([c == 2 for c in product_list['calib_level']])
    elif stage == 3:
        masks.append([t == 'I2D' for t in product_list['productSubGroupDescription']])
        masks.append([c == 3 for c in product_list['calib_level']])
    else:
        raise ValueError(f'Unsupported JWST stage: {stage}')
    return product_list[_combine_masks(masks)]


def flatten_mast_download_dir(download_dir: str | os.PathLike) -> list[str]:
    """
    Move nested MAST products up to *download_dir* and prune empty folders.

    ``Observations.download_products`` writes
    ``<download_dir>/mastDownload/.../<filename>``. This relocates every
    ``*.fits`` / ``*.fits.gz`` to ``<download_dir>/<filename>`` so the
    on-disk path is
    ``download/<telescope>/<instrument>/<filter>/<obsid>/<filename>``.
    """
    root = os.path.abspath(str(download_dir))
    if not os.path.isdir(root):
        return []
    moved: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            if not (lower.endswith('.fits') or lower.endswith('.fits.gz')):
                continue
            src = os.path.join(dirpath, name)
            if os.path.dirname(src) == root:
                continue
            dest = os.path.join(root, name)
            if os.path.exists(dest):
                if os.path.samefile(src, dest):
                    continue
                # Prefer the already-flattened file; drop the nested copy.
                try:
                    os.remove(src)
                except OSError:
                    pass
                continue
            os.rename(src, dest)
            moved.append(dest)
    # Drop the nested MAST staging tree (non-FITS leftovers and all).
    mast_tree = os.path.join(root, 'mastDownload')
    if os.path.isdir(mast_tree):
        shutil.rmtree(mast_tree, ignore_errors=True)
    # Remove any other empty nested directories.
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        if dirpath == root:
            continue
        try:
            os.rmdir(dirpath)
        except OSError:
            pass
    return moved


def observation_download_subdir(
    filt: str,
    obsid: object,
    layout: str = DEFAULT_DOWNLOAD_LAYOUT,
    *,
    telescope: object = 'JWST',
    instrument: object | None = None,
) -> str:
    """Return the per-observation subdirectory under the download root.

    Parameters
    ----------
    filt : str
        MAST ``filters`` field for the observation.
    obsid : object
        MAST observation ID (coerced to ``str``).
    layout : str, optional
        Directory layout name (default :data:`DEFAULT_DOWNLOAD_LAYOUT`).
    telescope : object, optional
        MAST ``obs_collection`` value (default ``'JWST'``).
    instrument : object or None, optional
        MAST ``instrument_name`` value.

    Returns
    -------
    str
        Relative subdirectory path for this observation.

    Raises
    ------
    ValueError
        If ``layout`` is not a supported layout name.

    Notes
    -----
    Supported layouts:

    ``telescope/instrument/filter/obsid`` (default)
        e.g. ``JWST/MIRI/F560W/<obsid>``
    ``filter/obsid``
        e.g. ``<FILTER>/<obsid>`` (older download layout)
    ``filter_obsid``
        e.g. ``<FILTER>_<obsid>`` (legacy flat name)
    """
    filt_name = normalize_filter_name(filt)
    obsid_s = str(obsid)
    if layout in (
        'telescope/instrument/filter/obsid',
        'tel/inst/filter/obsid',
        'canonical',
    ):
        tel = normalize_telescope_dirname(telescope)
        inst = normalize_instrument_dirname(instrument)
        return os.path.join(tel, inst, filt_name, obsid_s)
    if layout in ('filter/obsid', 'filter_dir'):
        return os.path.join(filt_name, obsid_s)
    if layout in ('filter_obsid', 'legacy'):
        return f'{filt_name}_{obsid_s}'
    raise ValueError(
        f'Unsupported download layout {layout!r}; '
        "use 'telescope/instrument/filter/obsid', 'filter/obsid', or 'filter_obsid'"
    )


def prune_non_full_frame_miri(
    root: str | os.PathLike,
    *,
    remove: bool = True,
) -> list[str]:
    """
    Remove (or list) non-full-frame MIRI ``*mirimage*`` FITS under *root*.

    Subarrays and cutouts are unsupported by alignment / ``mirimask``. Called
    after MAST download so they never remain in the science tree.

    Parameters
    ----------
    root : str or path-like
        Directory to scan recursively.
    remove : bool, optional
        If True (default), delete matching files; otherwise only return paths.

    Returns
    -------
    list of str
        Paths that were non-full-frame (deleted when ``remove`` is True).
    """
    from pathlib import Path

    from st123.utils.helpers import is_full_frame_miri

    root_path = Path(root)
    if not root_path.is_dir():
        return []
    rejected: list[str] = []
    for path in root_path.rglob('*mirimage*.fits'):
        name = path.name.lower()
        if name.endswith('.sky.fits'):
            continue
        if not any(name.endswith(suf) for suf in ('_cal.fits', '_rate.fits', '_jhat.fits')):
            # Still check other mirimage products (e.g. calints) by shape.
            if '_cal' not in name and '_rate' not in name and '_jhat' not in name:
                continue
        if is_full_frame_miri(path):
            continue
        rejected.append(str(path))
        logger.warning(
            'Rejecting non-full-frame MIRI product (unsupported subarray/cutout): %s',
            path,
        )
        if remove:
            try:
                path.unlink()
            except OSError as exc:
                logger.warning('could not remove %s: %s', path, exc)
    return rejected


def download_jwst_observations(
    obs_table: Table,
    outdir: str,
    stage: int = 2,
    extension: str = 'fits',
    token: Optional[str] = None,
    *,
    layout: str = DEFAULT_DOWNLOAD_LAYOUT,
    mirimage_only: bool = False,
    dry_run: bool = False,
) -> int:
    """Download filtered JWST products for each observation into ``outdir``.

    Pass ``token`` (or set ``MAST_API_TOKEN``) to authenticate before downloading
    proprietary products, matching the hst123 ``Observations.login`` flow.

    Parameters
    ----------
    obs_table : Table
        JWST observation table (from :func:`query_jwst` or similar).
    outdir : str
        Download tree root (typically ``<base-dir>/download``).
    stage : int, optional
        Calibration stage to download: ``2`` for CAL, ``3`` for I2D (default 2).
    extension : str, optional
        File extension passed to ``Observations.download_products`` (default
        ``'fits'``).
    token : str or None, optional
        MAST API token; also read from ``MAST_API_TOKEN`` / ``MAST_TOKEN``.
    layout : str, optional
        Per-observation subdirectory layout (default
        :data:`DEFAULT_DOWNLOAD_LAYOUT`). See :func:`observation_download_subdir`.
    mirimage_only : bool, optional
        When ``True``, keep only ``*mirimage*`` product filenames (MIRI imager).
    dry_run : bool, optional
        When ``True``, list products without downloading.

    Returns
    -------
    int
        Number of observation product sets downloaded (or listed in dry-run).
    """
    # Auth is normally done in query_jwst; keep a cached no-op for standalone use.
    prepare_mast_auth(token)

    if obs_table is None or len(obs_table) == 0:
        logger.error('observation table is empty. Cannot download files.')
        return 0

    n_before = len(obs_table)
    obs_table = filter_jwst_observations_by_stage(obs_table, stage)
    n_skipped = n_before - len(obs_table)
    if n_skipped:
        # Observations with calib_level=-1 / below the requested stage are not
        # expected to have matching products (e.g. APT placeholders). Skip
        # silently aside from a one-line tally — do not per-obs "tried".
        logger.info(
            'Skipping %d observation(s) with no calib_level>=%d products available',
            n_skipped,
            stage,
        )

    if len(obs_table) == 0:
        logger.error(
            'no observations with calib_level>=%d. Cannot download files.',
            stage,
        )
        return 0

    os.makedirs(outdir, exist_ok=True)
    n_obs = len(obs_table)
    n_downloaded = 0
    logger.info('Downloading JWST products for %d observation(s) into %s', n_obs, outdir)
    if dry_run:
        logger.info('Dry run: no files will be downloaded')

    has_collection = 'obs_collection' in obs_table.colnames
    has_instrument = 'instrument_name' in obs_table.colnames
    for i, obs in enumerate(obs_table, start=1):
        filt = obs['filters']
        obsid = obs['obsid']
        telescope = obs['obs_collection'] if has_collection else 'JWST'
        instrument = obs['instrument_name'] if has_instrument else None
        subdir = observation_download_subdir(
            filt,
            obsid,
            layout=layout,
            telescope=telescope,
            instrument=instrument,
        )
        try:
            with capture_output():
                raw_products = Observations.get_product_list(obs)
            product_list = filter_jwst_products(
                raw_products,
                stage=stage,
                mirimage_only=mirimage_only,
            )
        except Exception as exc:
            logger.warning('could not get products for obsid=%s: %s', obsid, exc)
            continue
        if len(product_list) == 0:
            # Only reached for observations that looked stage-relevant.
            logger.info('[%d/%d] %s: no stage-%d science products', i, n_obs, subdir, stage)
            continue
        download_dir = os.path.join(outdir, subdir)
        os.makedirs(download_dir, exist_ok=True)
        logger.info(
            '[%d/%d] %s: %s %d product(s)...',
            i,
            n_obs,
            subdir,
            'listing' if dry_run else 'downloading',
            len(product_list),
        )
        for row in product_list:
            logger.info('    %s', row['productFilename'])
        if dry_run:
            n_downloaded += 1
            continue
        try:
            with capture_output():
                Observations.download_products(
                    product_list, download_dir=download_dir, extension=extension
                )
            flat = flatten_mast_download_dir(download_dir)
            if flat:
                logger.debug(
                    '[%d/%d] %s: flattened %d nested MAST product(s)',
                    i,
                    n_obs,
                    subdir,
                    len(flat),
                )
            # Drop MIRI subarrays/cutouts immediately so they never enter
            # alignment or DOLPHOT staging.
            pruned = prune_non_full_frame_miri(download_dir, remove=True)
            if pruned:
                logger.info(
                    '[%d/%d] %s: removed %d non-full-frame MIRI product(s)',
                    i,
                    n_obs,
                    subdir,
                    len(pruned),
                )
            n_downloaded += 1
        except Exception as exc:
            logger.warning('download failed for obsid=%s: %s', obsid, exc)

    logger.info(
        'Finished: downloaded products for %d/%d observation(s).',
        n_downloaded,
        n_obs,
    )
    return n_downloaded


def filter_hst_products(
    product_list: Table,
    instrument_name: object,
) -> Table:
    """Keep HST flt/flc/c0m/c1m SCIENCE products for *instrument*.

    WFPC2 ``c1m`` DQ companions are always retained — they are required for
    AstroDrizzle and ``wfpc2mask``.

    Parameters
    ----------
    product_list : Table
        Raw MAST product list.
    instrument_name : object
        MAST ``instrument_name`` for the parent observation.

    Returns
    -------
    Table
        Filtered product table.
    """
    inst = str(instrument_name)
    mask = [
        is_hst_science_product(str(row['productFilename']), inst)
        for row in product_list
    ]
    return product_list[np.asarray(mask, dtype=bool)]


def _existing_hst_basenames(outdir: str) -> set[str]:
    """Basenames of FITS files already present under *outdir* (recursive)."""
    found: set[str] = set()
    root = os.path.abspath(outdir)
    if not os.path.isdir(root):
        return found
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            if lower.endswith(('.fits', '.fits.gz')):
                found.add(name)
    return found


def _local_hst_science_basenames(
    download_dir: str,
    instrument_name: object,
) -> list[str]:
    """Science FITS basenames already present directly under *download_dir*."""
    if not os.path.isdir(download_dir):
        return []
    out: list[str] = []
    for name in sorted(os.listdir(download_dir)):
        lower = name.lower()
        if not lower.endswith(('.fits', '.fits.gz')):
            continue
        if is_hst_science_product(name, str(instrument_name)):
            out.append(name)
    return out


def _hst_obs_locally_complete(
    download_dir: str,
    instrument_name: object,
) -> list[str]:
    """
    Return local science basenames when the obs subdir looks complete enough
    to skip a MAST ``get_product_list`` round-trip.

    ACS/WFC3: any matching flc/flt. WFPC2: both ``c0m`` and ``c1m`` present.
    """
    local = _local_hst_science_basenames(download_dir, instrument_name)
    if not local:
        return []
    inst = str(instrument_name).upper()
    if 'WFPC2' in inst or 'PC/WFC' in inst:
        has_c0m = any(n.lower().endswith('c0m.fits') for n in local)
        has_c1m = any(n.lower().endswith('c1m.fits') for n in local)
        return local if (has_c0m and has_c1m) else []
    return local


def download_hst_observations(
    obs_table: Table,
    outdir: str,
    token: Optional[str] = None,
    *,
    layout: str = DEFAULT_DOWNLOAD_LAYOUT,
    dry_run: bool = False,
    extension: str = 'fits',
):
    """Download filtered HST science products for each observation into ``outdir``.

    Products are restricted to native pipeline science files (WFPC2 ``c0m``/
    ``c1m``, ACS/WFC ``flc``, ACS/HRC+SBC ``flt``, WFC3/UVIS ``flc``,
    WFC3/IR ``flt``) with product ``type == 'S'``. Each unique
    ``productFilename`` is downloaded **once** (later obsids that re-list
    the same file are skipped), and files already present under *outdir*
    are not re-fetched. Observation directories that already contain local
    science products skip the MAST product-list API call.

    Transient MAST timeouts / connection errors on product-list and download
    are retried; observations that still fail are counted in
    ``MastDownloadResult.n_failed`` so callers can refuse a silent partial
    inventory (issue #4).

    Parameters
    ----------
    obs_table : Table
        HST observation table from :func:`query_hst`.
    outdir : str
        Download tree root (typically ``<base-dir>/download``).
    token : str or None, optional
        MAST API token.
    layout : str, optional
        Per-observation subdirectory layout.
    dry_run : bool, optional
        List products without downloading.
    extension : str, optional
        File extension for ``download_products``.

    Returns
    -------
    MastDownloadResult
        ``n_observations`` ready (newly downloaded, listed in dry-run, or
        already present). ``n_failed`` counts product-list / download failures
        after retries. ``0`` ready only when nothing usable was found.
    """
    # Lazy import avoids circular import with st123.mast.download.
    from st123.mast.download import MastDownloadResult

    # Auth is normally done in query_hst; keep a cached no-op for standalone use.
    prepare_mast_auth(token)

    if obs_table is None or len(obs_table) == 0:
        logger.error('observation table is empty. Cannot download files.')
        return MastDownloadResult(0)

    os.makedirs(outdir, exist_ok=True)
    n_obs = len(obs_table)
    n_downloaded = 0
    n_ready = 0
    n_failed = 0
    n_unique = 0
    n_skipped_dup = 0
    n_skipped_existing = 0
    n_skipped_local = 0
    seen_filenames: set[str] = set()
    existing = _existing_hst_basenames(outdir)
    logger.info(
        'Downloading HST products for %d observation(s) into %s '
        '(%d existing FITS basename(s) under tree)',
        n_obs,
        outdir,
        len(existing),
    )
    if dry_run:
        logger.info('Dry run: no files will be downloaded')

    has_collection = 'obs_collection' in obs_table.colnames
    has_instrument = 'instrument_name' in obs_table.colnames
    for i, obs in enumerate(obs_table, start=1):
        filt = obs['filters']
        obsid = obs['obsid']
        telescope = obs['obs_collection'] if has_collection else 'HST'
        instrument = obs['instrument_name'] if has_instrument else None
        subdir = observation_download_subdir(
            filt,
            obsid,
            layout=layout,
            telescope=telescope,
            instrument=instrument,
        )
        download_dir = os.path.join(outdir, subdir)

        # Fast path: avoid MAST get_product_list when the obs subdir already
        # has local science products (dominant cost on re-runs).
        if not dry_run:
            local = _hst_obs_locally_complete(download_dir, instrument)
            if local:
                for name in local:
                    seen_filenames.add(name)
                n_ready += 1
                n_skipped_local += 1
                n_skipped_existing += len(local)
                logger.info(
                    '[%d/%d] %s: %d science product(s) already on disk — '
                    'skipping MAST product list',
                    i,
                    n_obs,
                    subdir,
                    len(local),
                )
                continue

        logger.info(
            '[%d/%d] %s: fetching MAST product list (obsid=%s)...',
            i,
            n_obs,
            subdir,
            obsid,
        )
        t0 = time.monotonic()

        def _fetch_products():
            with capture_output():
                return Observations.get_product_list(obs)

        raw_products, plist_exc = _mast_call_with_retries(
            f'product list obsid={obsid}',
            _fetch_products,
        )
        if raw_products is None:
            logger.error(
                'could not get products for obsid=%s after retries: %s',
                obsid,
                plist_exc,
            )
            n_failed += 1
            continue

        try:
            dt = time.monotonic() - t0
            logger.info(
                '[%d/%d] %s: product list returned %d row(s) in %.1fs',
                i,
                n_obs,
                subdir,
                len(raw_products) if raw_products is not None else 0,
                dt,
            )
            # Simple exposures only (associations / HAP derived are type C/D).
            if 'type' in raw_products.colnames:
                raw_products = raw_products[raw_products['type'] == 'S']
            product_list = filter_hst_products(raw_products, instrument)
        except Exception as exc:
            logger.error(
                'could not filter products for obsid=%s: %s', obsid, exc
            )
            n_failed += 1
            continue
        if len(product_list) == 0:
            # Should be rare after filter_hst_observations drops unsupported
            # detectors; keep as debug noise, not a failure.
            logger.debug(
                '[%d/%d] %s: no supported native HST products '
                '(instrument=%s; expect flc/flt/c0m per HST_PRODUCT_RULES)',
                i,
                n_obs,
                subdir,
                instrument,
            )
            continue

        # Dedupe across obsids; skip files already on disk under outdir.
        keep_idx: list[int] = []
        for j, row in enumerate(product_list):
            fname = str(row['productFilename'])
            if fname in seen_filenames:
                n_skipped_dup += 1
                continue
            if fname in existing:
                n_skipped_existing += 1
                seen_filenames.add(fname)
                continue
            seen_filenames.add(fname)
            keep_idx.append(j)
        if not keep_idx:
            # Already on disk (or duplicate of another obsid) — treat as success.
            n_ready += 1
            logger.info(
                '[%d/%d] %s: all science products already downloaded or duplicated',
                i,
                n_obs,
                subdir,
            )
            continue
        product_list = product_list[np.asarray(keep_idx, dtype=int)]
        n_unique += len(product_list)

        os.makedirs(download_dir, exist_ok=True)
        logger.info(
            '[%d/%d] %s: %s %d unique product(s)...',
            i,
            n_obs,
            subdir,
            'listing' if dry_run else 'downloading',
            len(product_list),
        )
        for row in product_list:
            logger.info('    %s', row['productFilename'])
        if dry_run:
            n_downloaded += 1
            n_ready += 1
            continue

        def _download():
            with capture_output():
                Observations.download_products(
                    product_list, download_dir=download_dir, extension=extension
                )

        _, dl_exc = _mast_call_with_retries(
            f'download obsid={obsid}',
            _download,
        )
        if dl_exc is not None:
            logger.error(
                'download failed for obsid=%s after retries: %s',
                obsid,
                dl_exc,
            )
            n_failed += 1
            continue

        flat = flatten_mast_download_dir(download_dir)
        if flat:
            logger.debug(
                '[%d/%d] %s: flattened %d nested MAST product(s)',
                i,
                n_obs,
                subdir,
                len(flat),
            )
        n_downloaded += 1
        n_ready += 1
        for row in product_list:
            existing.add(str(row['productFilename']))

    logger.info(
        'Finished: downloaded HST products for %d/%d observation(s) '
        '(%d unique file(s); skipped %d duplicate listing(s), '
        '%d already on disk, %d local obs skip(s); %d observation(s) ready, '
        '%d failed).',
        n_downloaded,
        n_obs,
        n_unique,
        n_skipped_dup,
        n_skipped_existing,
        n_skipped_local,
        n_ready,
        n_failed,
    )
    if n_failed:
        logger.error(
            'HST download incomplete: %d/%d observation(s) failed after '
            'retries; re-run download or check MAST before mosaicking',
            n_failed,
            n_obs,
        )
    return MastDownloadResult(n_ready, n_failed=n_failed)


def parse_s_region(region: str) -> shapely.Polygon:
    """Parse a MAST/FITS ``S_REGION`` POLYGON string into a shapely Polygon.

    Parameters
    ----------
    region : str
        ``S_REGION`` header value (e.g. ``'POLYGON ICRS ra dec ...'``).

    Returns
    -------
    Polygon
        Sky polygon with vertices in degrees (RA, Dec).

    Raises
    ------
    ValueError
        If the string format is unrecognized or has an odd coordinate count.
    """
    text = str(region)
    for prefix in ('POLYGON ICRS  ', 'POLYGON ICRS ', 'POLYGON '):
        if prefix in text:
            coords = np.array(text.split(prefix, 1)[1].split(), dtype=float)
            break
    else:
        raise ValueError(f'Unrecognized S_REGION format: {region!r}')
    if coords.size % 2 != 0:
        raise ValueError(f'Odd number of S_REGION coordinates: {region!r}')
    return shapely.Polygon(coords.reshape(-1, 2))


def polygons_from_obs_table(obs_table: Table) -> tuple[np.ndarray, list[str]]:
    """Return sky polygons and filter names for each row in ``obs_table``.

    Parameters
    ----------
    obs_table : Table
        MAST observation table with ``s_region`` and ``filters`` columns.

    Returns
    -------
    pgons : ndarray
        Object array of :class:`shapely.Polygon` footprints, one per row.
    filters : list of str
        Filter name for each row (same order as ``pgons``).
    """
    pgons = [parse_s_region(region) for region in obs_table['s_region']]
    filters = list(obs_table['filters'])
    return np.array(pgons, dtype=object), filters


def coverage_fraction(
    pgons: Iterable[shapely.geometry.base.BaseGeometry],
    filter_mask: Sequence[bool],
) -> float:
    """Fraction of the union footprint covered by the masked subset of polygons.

    Parameters
    ----------
    pgons : iterable of BaseGeometry
        Footprint polygons (typically from :func:`polygons_from_obs_table`).
    filter_mask : sequence of bool
        Boolean mask selecting which polygons contribute to the subset union.

    Returns
    -------
    float
        Intersection area divided by the full union area (0.0 when union area
        is zero).
    """
    pgons = np.asarray(list(pgons), dtype=object)
    net_field = shapely.unary_union(pgons)
    subset = shapely.unary_union(pgons[np.asarray(filter_mask)])
    if net_field.area == 0:
        return 0.0
    return shapely.intersection(subset, net_field).area / net_field.area
