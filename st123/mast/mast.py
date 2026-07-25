"""MAST query, filter, and download helpers for HST and JWST imaging."""

from __future__ import annotations

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

# Default HST science products used by download helpers / notebooks
HST_PRODUCT_RULES = (
    ('c0m.fits', 'WFPC2'),
    ('c1m.fits', 'WFPC2'),
    ('c0m.fits', 'PC/WFC'),
    ('c1m.fits', 'PC/WFC'),
    ('flc.fits', 'ACS/WFC'),
    ('flt.fits', 'ACS/HRC'),
    ('flc.fits', 'WFC3/UVIS'),
    ('flt.fits', 'WFC3/IR'),
)

DEFAULT_HST_FILTERS = ('F275W', 'F555W', 'F814W')
DEFAULT_HST_INSTRUMENTS = ('ACS', 'WFC', 'WFPC2')
DEFAULT_JWST_INSTRUMENTS = ('NIRCAM', 'MIRI')


def resolve_mast_token(token: Optional[str] = None) -> Optional[str]:
    """
    Resolve a MAST API token from an explicit value or the environment.

    Checks ``token``, then ``MAST_API_TOKEN``, then ``MAST_TOKEN``.
    Create a token at https://auth.mast.stsci.edu/info
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
        print('Logging in to MAST with token...')
        Observations.login(token=token)
        print('MAST login successful.')
        return True
    except Exception as exc:
        message = f'could not log in to MAST with input token: {exc}'
        if required:
            raise RuntimeError(message) from exc
        print(f'WARNING: {message}')
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
    vizier_radius=5 * u.arcsec,
    vizier_cat: str = 'VII/237/pgc',
    floor: Optional[u.Quantity] = None,
) -> u.Quantity:
    """Estimate a search radius from the PGC logD25 size, with an optional floor."""
    vizier = Vizier(columns=['logD25'])
    result = vizier.query_region(coord, radius=vizier_radius, catalog=vizier_cat)
    logd = result[0]['logD25'][0]
    radius = (10 ** logd * 0.1 * u.arcmin)
    if floor is not None:
        return max(floor, radius)
    return radius


def query_region(coord: SkyCoord, radius: u.Quantity) -> Table:
    """Query MAST Observations around ``coord`` and fill masked values."""
    obs_table = Observations.query_region(coord, radius=radius)
    return obs_table.filled()


def filter_hst_observations(
    obs_table: Table,
    filters: Optional[Sequence[str]] = None,
    instruments: Sequence[str] = DEFAULT_HST_INSTRUMENTS,
    public_only: bool = True,
) -> Table:
    """Apply standard HST imaging masks to a MAST observation table."""
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
    """Apply standard JWST NIRCam imaging masks to a MAST observation table."""
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


def query_hst(
    coord: SkyCoord,
    radius: Optional[u.Quantity] = None,
    filters: Optional[Sequence[str]] = DEFAULT_HST_FILTERS,
    instruments: Sequence[str] = DEFAULT_HST_INSTRUMENTS,
    use_galaxy_size: bool = False,
    public_only: bool = True,
    token: Optional[str] = None,
) -> Table:
    """
    Query and filter HST imaging observations near ``coord``.

    If a MAST token is provided (argument or environment), authenticate first and
    include proprietary observations (``public_only`` is forced False).
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
    """
    Query and filter JWST imaging observations near ``coord``.

    If a MAST token is provided (argument or ``MAST_API_TOKEN`` / ``MAST_TOKEN``),
    authenticate first and include proprietary observations (``public_only`` is
    forced False). Without a token, only public data are returned by default.
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
    """Return True if ``filename`` is a usable HST science product for ``instrument``."""
    return any(
        suffix in filename and tag in instrument
        for suffix, tag in HST_PRODUCT_RULES
    )


def collect_hst_products(obs_table: Table) -> Optional[Table]:
    """Build a product table of HST flt/flc/c0m/c1m science files from observations."""
    productlist = None
    for obs in obs_table:
        try:
            product_list = Observations.get_product_list(obs)
            product_list = product_list[product_list['type'] == 'S']
        except Exception:
            print('ERROR: MAST is not working currently\nTry again later...')
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
    """Turn MAST filter strings like ``F560W;CLEAR`` into a directory name."""
    name = str(filt).split(';')[0].strip().upper()
    name = re.sub(r'[^A-Z0-9_\-]+', '_', name)
    return name or 'UNKNOWN'


def normalize_telescope_dirname(obs_collection: object = None, *, default: str = 'JWST') -> str:
    """Return a telescope directory name (e.g. ``JWST``, ``HST``, ``Roman``)."""
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


def normalize_instrument_dirname(instrument_name: object = None) -> str:
    """
    Return an instrument directory name under the telescope root.

    Examples: ``MIRI``, ``NIRCam``, ``NIRISS``, ``ACS``, ``WFC3``, ``WFI``.
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
    """Filter JWST products to science CAL (stage 2) or I2D (stage 3) files."""
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


# Canonical layout used for raw (and aligned-beside-raw) JWST/HST products.
DEFAULT_DOWNLOAD_LAYOUT = 'telescope/instrument/filter/obsid'


def observation_download_subdir(
    filt: str,
    obsid: object,
    layout: str = DEFAULT_DOWNLOAD_LAYOUT,
    *,
    telescope: object = 'JWST',
    instrument: object = None,
) -> str:
    """
    Return the per-observation subdirectory under the download root.

    Layouts
    -------
    ``telescope/instrument/filter/obsid`` (default / preferred)
        ``JWST/MIRI/F560W/<obsid>``
    ``filter/obsid``
        ``<FILTER>/<obsid>`` (older alignment_wrap layout)
    ``filter_obsid``
        ``<FILTER>_<obsid>`` (legacy flat name)
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
    """
    Download filtered JWST products for each observation into ``outdir``.

    Pass ``token`` (or set ``MAST_API_TOKEN``) to authenticate before downloading
    proprietary products, matching the hst123 ``Observations.login`` flow.

    Parameters
    ----------
    layout : str
        ``telescope/instrument/filter/obsid`` →
            ``<outdir>/JWST/MIRI/<FILTER>/<obsid>/mastDownload/...``
        ``filter/obsid`` → ``<outdir>/<FILTER>/<obsid>/mastDownload/...``
        ``filter_obsid`` → ``<outdir>/<FILTER>_<obsid>/mastDownload/...``
    mirimage_only : bool
        If True, keep only ``*mirimage*`` product filenames (MIRI imager).
    dry_run : bool
        If True, list products without downloading.

    Returns
    -------
    int
        Number of observation product sets downloaded (or listed in dry-run).
    """
    resolved = resolve_mast_token(token)
    if resolved:
        mast_login(resolved, required=True)

    if obs_table is None or len(obs_table) == 0:
        print('ERROR: observation table is empty. Cannot download files.')
        return 0

    os.makedirs(outdir, exist_ok=True)
    n_obs = len(obs_table)
    n_downloaded = 0
    print(f'Downloading JWST products for {n_obs} observation(s) into {outdir}')
    if dry_run:
        print('Dry run: no files will be downloaded')

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
            product_list = filter_jwst_products(
                Observations.get_product_list(obs),
                stage=stage,
                mirimage_only=mirimage_only,
            )
        except Exception as exc:
            print(f'WARNING: could not get products for obsid={obsid}: {exc}')
            continue
        if len(product_list) == 0:
            print(f'[{i}/{n_obs}] {subdir}: no stage-{stage} science products')
            continue
        download_dir = os.path.join(outdir, subdir)
        os.makedirs(download_dir, exist_ok=True)
        print(
            f'[{i}/{n_obs}] {subdir}: '
            f'{"listing" if dry_run else "downloading"} '
            f'{len(product_list)} product(s)...'
        )
        for row in product_list:
            print(f'    {row["productFilename"]}')
        if dry_run:
            n_downloaded += 1
            continue
        try:
            Observations.download_products(
                product_list, download_dir=download_dir, extension=extension
            )
            n_downloaded += 1
        except Exception as exc:
            print(f'WARNING: download failed for obsid={obsid}: {exc}')

    print(f'Finished: downloaded products for {n_downloaded}/{n_obs} observation(s).')
    return n_downloaded


def parse_s_region(region: str) -> shapely.Polygon:
    """Parse a MAST/FITS ``S_REGION`` POLYGON string into a shapely Polygon."""
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


def polygons_from_obs_table(obs_table: Table) -> tuple[np.ndarray, list]:
    """Return sky polygons and filter names for each row in ``obs_table``."""
    pgons = [parse_s_region(region) for region in obs_table['s_region']]
    filters = list(obs_table['filters'])
    return np.array(pgons, dtype=object), filters


def coverage_fraction(
    pgons: Iterable[shapely.geometry.base.BaseGeometry],
    filter_mask: Sequence[bool],
) -> float:
    """Fraction of the union footprint covered by the masked subset of polygons."""
    pgons = np.asarray(list(pgons), dtype=object)
    net_field = shapely.unary_union(pgons)
    subset = shapely.unary_union(pgons[np.asarray(filter_mask)])
    if net_field.area == 0:
        return 0.0
    return shapely.intersection(subset, net_field).area / net_field.area
