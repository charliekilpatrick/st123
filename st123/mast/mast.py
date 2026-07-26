"""MAST query, filter, and download helpers for HST and JWST imaging."""

from __future__ import annotations

import logging
import os
import re
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


def mast_login(token: Optional[str] = None, *, required: bool = False) -> bool:
    """
    Authenticate to MAST with an authorization token.

    Uses ``Observations.login(token=...)`` so proprietary products available to
    the token holder can be queried and downloaded (same pattern as hst123).
    See https://auth.mast.stsci.edu/info

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
        True if login was attempted and succeeded, False otherwise.
    """
    token = resolve_mast_token(token)
    if not token:
        if required:
            raise RuntimeError(
                'A MAST token is required but none was provided. '
                'Pass token=... / --token, or set MAST_API_TOKEN.'
            )
        return False
    try:
        logger.info('Logging in to MAST with token...')
        Observations.login(token=token)
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
        (default :data:`DEFAULT_HST_INSTRUMENTS`).
    public_only : bool, optional
        When ``True``, keep only rows with ``dataRights == 'PUBLIC'``.

    Returns
    -------
    Table
        Filtered table sorted by ``t_min``.
    """
    masks = [
        [str(t).upper() == 'HST' for t in obs_table['obs_collection']],
        [str(p).upper() == 'IMAGE' for p in obs_table['dataproduct_type']],
        _instrument_contains(obs_table, instruments),
        [str(f).upper() != 'DETECTION' for f in obs_table['filters']],
        [str(i).upper() != 'CALIBRATION' for i in obs_table['intentType']],
    ]
    if filters is not None:
        allowed = {f.upper() for f in filters}
        masks.append([str(f).upper() in allowed for f in obs_table['filters']])
    if public_only:
        masks.append([str(r).upper() == 'PUBLIC' for r in obs_table['dataRights']])

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

    If a MAST token is provided (argument or environment), authenticate first and
    include proprietary observations (``public_only`` is forced ``False``).

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
    resolved = resolve_mast_token(token)
    if resolved:
        mast_login(resolved, required=True)
        public_only = False
    if radius is None:
        if use_galaxy_size:
            radius = galaxy_query_radius(coord)
        else:
            radius = 5 * u.arcmin
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
    resolved = resolve_mast_token(token)
    if resolved:
        mast_login(resolved, required=True)
        public_only = False
    if radius is None:
        if use_galaxy_size:
            radius = galaxy_query_radius(coord, floor=5 * u.arcmin)
        else:
            radius = 5 * u.arcmin
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
        Root download directory.
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
    resolved = resolve_mast_token(token)
    if resolved:
        mast_login(resolved, required=True)

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
            n_downloaded += 1
        except Exception as exc:
            logger.warning('download failed for obsid=%s: %s', obsid, exc)

    logger.info(
        'Finished: downloaded products for %d/%d observation(s).',
        n_downloaded,
        n_obs,
    )
    return n_downloaded


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
