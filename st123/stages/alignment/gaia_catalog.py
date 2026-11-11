"""Gaia catalog queries for alignment (Gaia@AIP, ESA TAP, Vizier).

Cone queries try the fast Gaia@AIP IVOA cone-search first, then Gaia@AIP TAP,
ESA Archive TAP, and finally CDS VizieR (slow when mirrors are down). Results
are cached under ``<reduction>/gaia/``. JHAT's ``get_GAIA_sources`` is patched
to the same path via :func:`install_jhat_gaia_vizier_patch`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from st123.datamodels import as_datamodel
from astropy.table import Table

logger = logging.getLogger(__name__)

# VizieR Gaia DR3 main source table (I/355).
VIZIER_GAIA_DR3 = 'I/355/gaiadr3'
VIZIER_GAIA_DR2 = 'I/345/gaia2'

# ESA Gaia Archive TAP source tables.
TAP_GAIA_DR3 = 'gaiadr3.gaia_source'
TAP_GAIA_DR2 = 'gaiadr2.gaia_source'

GaiaBackend = Literal['auto', 'vizier', 'tap']

# Shared field catalog under ``<reduction>/gaia/``.
GAIA_CACHE_DIRNAME = 'gaia'

# Floor on field-cone radius so sparse / small-FOV HST visits still pull enough
# Gaia for gaia_simple and absolute checks (NGC 3913-like fields need ~0.1 deg).
GAIA_MIN_CONE_RADIUS_DEG = 0.1

# Map common VizieR column names -> TAP / JHAT-style names used elsewhere.
_VIZIER_COLMAP = (
    ('RA_ICRS', 'ra'),
    ('DE_ICRS', 'dec'),
    ('RAJ2000', 'ra'),
    ('DEJ2000', 'dec'),
    ('pmRA', 'pmra'),
    ('pmDE', 'pmdec'),
    ('e_pmRA', 'pmra_error'),
    ('e_pmDE', 'pmdec_error'),
    ('Gmag', 'phot_g_mean_mag'),
    ('Source', 'source_id'),
    ('DR3Name', 'designation'),
)

PathLike = str | Path


def cut_gaia_sources(image: str, table_gaia: Table) -> Table:
    """
    Drop Gaia sources that fall outside an image.

    Parameters
    ----------
    image : str
        Image file name.
    table_gaia : astropy.table.Table
        Gaia table with ``ra`` / ``dec`` columns.

    Returns
    -------
    astropy.table.Table
        Sources inside the image bounds.
    """
    if len(table_gaia) == 0:
        return table_gaia
    dm = as_datamodel(image)
    with dm.open(memmap=True) as im:
        hdr = im['SCI'].header
        nx, ny = hdr['NAXIS1'], hdr['NAXIS2']
    # sci_wcs passes fobj so ACS/WFC3 lookup-table distortion resolves.
    w = dm.sci_wcs()

    pix_coords = w.all_world2pix(
        np.array(table_gaia['ra'], dtype=float),
        np.array(table_gaia['dec'], dtype=float),
        0,
    )
    im_x, im_y = pix_coords[0], pix_coords[1]
    mask = (im_x > 0) & (im_x < nx) & (im_y > 0) & (im_y < ny)
    return table_gaia[mask]


def _normalize_vizier_gaia(table: Table) -> Table:
    """Rename Vizier columns to TAP-style ``ra`` / ``dec`` / ... names."""
    out = table.copy()
    for src, dest in _VIZIER_COLMAP:
        if src in out.colnames and dest not in out.colnames:
            out.rename_column(src, dest)
    return out


def _normalize_tap_gaia(table: Table) -> Table:
    """
    Normalize ESA TAP Gaia columns to the shared st123 / JHAT schema.

    Adds Vizier-style ``BPmag`` / ``RPmag`` aliases when only ``phot_*``
    columns are present so :func:`jhat_get_gaia_sources` can stay backend-agnostic.
    """
    out = table.copy()
    aliases = (
        ('phot_bp_mean_mag', 'BPmag'),
        ('phot_rp_mean_mag', 'RPmag'),
        ('ref_epoch', 'Epoch'),
    )
    for src, dest in aliases:
        if src in out.colnames and dest not in out.colnames:
            out[dest] = out[src]
    return out


def _tap_table_for_dr(dr: str) -> str:
    key = str(dr).strip().lower().replace('_', '').replace('.', '')
    if key in ('gaiadr2', 'dr2', 'gaia2'):
        return TAP_GAIA_DR2
    return TAP_GAIA_DR3


def _add_pm_ratio(table: Table) -> Table:
    pm_cols = ('pmra', 'pmdec', 'pmra_error', 'pmdec_error')
    if all(col in table.colnames for col in pm_cols) and 'pm/pmerr' not in table.colnames:
        table = table.copy()
        table['pm/pmerr'] = (table['pmra'] ** 2 + table['pmdec'] ** 2) / (
            table['pmra_error'] ** 2 + table['pmdec_error'] ** 2
        )
    return table


def _image_cone(
    image: str,
    *,
    telescope: str = 'jwst',
) -> tuple[SkyCoord, u.Quantity]:
    """Return cone center and radius covering the science footprint."""
    dm = as_datamodel(image)
    with dm.open(memmap=True) as im:
        hdr = im['SCI'].header
        nx = int(hdr['NAXIS1'])
        ny = int(hdr['NAXIS2'])

        if telescope == 'jwst':
            # Lazy: JWST GWCS stack is heavy; HST callers never enter here.
            from jwst.datamodels import ImageModel

            image_model = ImageModel(im)

            def pix_to_world(x, y):
                return image_model.meta.wcs(x, y)

            ra0, dec0 = pix_to_world(nx / 2.0 - 1, ny / 2.0 - 1)
        elif telescope == 'hst':
            # Bare WCS(hdr) fails on ACS/WFC3 flc when CPDIS lookup tables
            # need the open HDUList (sn2004g field-cache abort).
            try:
                w = dm.sci_wcs()

                def pix_to_world(x, y):
                    return w.pixel_to_world_values(x, y)

                ra0, dec0 = pix_to_world(nx / 2.0 - 1, ny / 2.0 - 1)
            except Exception as wcs_exc:
                # Gaia only needs an approximate covering cone -- CRVAL +
                # plate scale is enough when distortion tables are unavailable.
                logger.warning(
                    'Gaia cone: sci_wcs failed for %s (%s); using CRVAL/CD',
                    Path(image).name,
                    wcs_exc,
                )
                ra0 = float(hdr.get('CRVAL1') or im[0].header.get('CRVAL1'))
                dec0 = float(hdr.get('CRVAL2') or im[0].header.get('CRVAL2'))
                scale = _approx_plate_scale_deg(hdr)
                half = scale * 0.5 * float(np.hypot(nx, ny))
                coord0 = SkyCoord(ra0, dec0, unit=(u.deg, u.deg), frame='icrs')
                return coord0, float(half * 1.1) * u.deg
        else:
            raise ValueError(f'Unsupported telescope: {telescope}')

    coord0 = SkyCoord(ra0, dec0, unit=(u.deg, u.deg), frame='icrs')
    radius_deg = []
    for x in [0, nx - 1]:
        for y in [0, ny - 1]:
            ra, dec = pix_to_world(x, y)
            radius_deg.append(
                coord0.separation(
                    SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
                ).deg
            )
    radius = float(np.amax(radius_deg) * 1.1) * u.deg
    return coord0, radius


def _approx_plate_scale_deg(hdr) -> float:
    """Absolute plate scale in degrees/pixel from CD / CDELT (Gaia cone only)."""
    try:
        cd11 = float(hdr.get('CD1_1', 0.0) or 0.0)
        cd12 = float(hdr.get('CD1_2', 0.0) or 0.0)
        cd21 = float(hdr.get('CD2_1', 0.0) or 0.0)
        cd22 = float(hdr.get('CD2_2', 0.0) or 0.0)
        det = abs(cd11 * cd22 - cd12 * cd21)
        if det > 0.0:
            return float(np.sqrt(det))
    except (TypeError, ValueError):
        pass
    for key in ('CDELT1', 'CDELT2'):
        try:
            v = abs(float(hdr.get(key) or 0.0))
            if v > 0.0:
                return v
        except (TypeError, ValueError):
            continue
    return 1.0 / 3600.0  # 1"/pix fallback


def _apply_min_cone_radius(
    coord: SkyCoord,
    radius: u.Quantity,
    *,
    min_radius_deg: float = GAIA_MIN_CONE_RADIUS_DEG,
) -> tuple[SkyCoord, u.Quantity]:
    """Enforce a minimum cone radius (degrees)."""
    r = float(radius.to_value(u.deg))
    floor = float(min_radius_deg)
    if r < floor:
        logger.info(
            'Gaia cone radius %.4f deg raised to minimum %.3f deg',
            r,
            floor,
        )
        r = floor
    return coord, r * u.deg


def _union_cone(
    images: Sequence[PathLike],
    *,
    telescope: str = 'hst',
    min_radius_deg: float = GAIA_MIN_CONE_RADIUS_DEG,
) -> tuple[SkyCoord, u.Quantity]:
    """Bounding cone covering every image footprint (with min-radius floor)."""
    centers: list[SkyCoord] = []
    radii: list[float] = []
    for image in images:
        try:
            c, r = _image_cone(str(image), telescope=telescope)
        except Exception as exc:
            logger.warning(
                'Gaia union cone: skipping %s (%s)',
                Path(image).name,
                exc,
            )
            continue
        centers.append(c)
        radii.append(float(r.to_value(u.deg)))
    if not centers:
        raise ValueError('no images for Gaia cone')
    if len(centers) == 1:
        return _apply_min_cone_radius(
            centers[0], radii[0] * u.deg, min_radius_deg=min_radius_deg
        )

    ras = np.array([c.ra.degree for c in centers], dtype=float)
    decs = np.array([c.dec.degree for c in centers], dtype=float)
    # Mean on the sphere (adequate for HST/JWST fields of a few arcmin).
    lon = np.deg2rad(ras)
    lat = np.deg2rad(decs)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    mean = SkyCoord(
        ra=np.rad2deg(np.arctan2(y.mean(), x.mean())) * u.deg,
        dec=np.rad2deg(np.arcsin(np.clip(z.mean(), -1.0, 1.0))) * u.deg,
        frame='icrs',
    )
    # Radius = max(distance to corner of each image) ~ center sep + image radius.
    need = 0.0
    for c, r in zip(centers, radii):
        need = max(need, float(mean.separation(c).deg) + r)
    return _apply_min_cone_radius(
        mean, (need * 1.05) * u.deg, min_radius_deg=min_radius_deg
    )


def _vizier_catalog_for_dr(dr: str) -> str:
    key = str(dr).strip().lower().replace('_', '').replace('.', '')
    if key in ('gaiadr2', 'dr2', 'gaia2'):
        return VIZIER_GAIA_DR2
    return VIZIER_GAIA_DR3


def gaia_cache_paths(
    cache_dir: PathLike,
    *,
    dr: str = 'gaiadr3',
) -> tuple[Path, Path, Path]:
    """
    Return ``(ecsv, meta_json, radec_txt)`` under a Gaia cache directory.

    Canonical location: ``<reduction>/gaia/gaiadr3.ecsv`` (+ ``.json``,
    ``_radec.txt``).
    """
    root = Path(cache_dir).expanduser()
    stem = str(dr).strip().lower().replace('.', '')
    if not stem.startswith('gaia'):
        stem = f'gaia{stem}'
    return (
        root / f'{stem}.ecsv',
        root / f'{stem}.json',
        root / f'{stem}_radec.txt',
    )


def find_reduction_dir(path: PathLike) -> Path | None:
    """
    Walk parents of *path* to find a reduction workdir.

    Recognizes a directory named ``reduction``, or one containing the usual
    markers (``raw/``, ``jhat/``, ``reference/``, ``reference_prelim/``,
    ``gaia/``).
    """
    p = Path(path).expanduser().resolve()
    if p.is_file():
        p = p.parent
    markers = ('raw', 'jhat', 'reference', 'reference_prelim', GAIA_CACHE_DIRNAME, 'align')
    for cand in (p, *p.parents):
        if cand.name == 'reduction':
            return cand
        if any((cand / m).exists() for m in markers):
            # Prefer an explicit reduction/ child when sitting at project root.
            if (cand / 'reduction').is_dir():
                return (cand / 'reduction').resolve()
            return cand
    return None


def default_gaia_cache_dir(path: PathLike) -> Path | None:
    """Return ``<reduction>/gaia`` for *path*, or ``None`` if unknown."""
    reduction = find_reduction_dir(path)
    if reduction is None:
        return None
    return reduction / GAIA_CACHE_DIRNAME


def _load_cache_meta(meta_path: Path) -> dict | None:
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _cache_covers(
    meta: dict,
    coord: SkyCoord,
    radius: u.Quantity,
    *,
    slack: float = 1.02,
) -> bool:
    try:
        cra = float(meta['ra'])
        cdec = float(meta['dec'])
        cr = float(meta['radius_deg'])
    except (KeyError, TypeError, ValueError):
        return False
    center = SkyCoord(cra, cdec, unit='deg', frame='icrs')
    sep = float(center.separation(coord).deg)
    need = float(radius.to_value(u.deg))
    return (sep + need) <= (cr * slack)


def _save_gaia_cache(
    table: Table,
    *,
    cache_dir: PathLike,
    coord: SkyCoord,
    radius: u.Quantity,
    dr: str,
    backend: str,
) -> Path:
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    ecsv, meta_path, radec_path = gaia_cache_paths(cache_dir, dr=dr)
    table.write(ecsv, format='ascii.ecsv', overwrite=True)
    meta = {
        'ra': float(coord.ra.degree),
        'dec': float(coord.dec.degree),
        'radius_deg': float(radius.to_value(u.deg)),
        'dr': dr,
        'backend': backend,
        'n': int(len(table)),
        'ecsv': ecsv.name,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + '\n')
    if 'ra' in table.colnames and 'dec' in table.colnames and len(table):
        np.savetxt(
            radec_path,
            np.column_stack(
                [
                    np.asarray(table['ra'], dtype=float),
                    np.asarray(table['dec'], dtype=float),
                ]
            ),
            fmt='%.10f',
            header='ra dec',
        )
    logger.info(
        'Wrote field Gaia cache %s (%d sources, r=%.3f deg) -> %s',
        ecsv.name,
        len(table),
        radius.to_value(u.deg),
        cache_dir,
    )
    return ecsv


def load_gaia_cache(
    cache_dir: PathLike,
    *,
    dr: str = 'gaiadr3',
) -> tuple[Table | None, dict | None]:
    """Load a previously saved field Gaia catalog, if present."""
    ecsv, meta_path, _radec = gaia_cache_paths(cache_dir, dr=dr)
    meta = _load_cache_meta(meta_path)
    if not ecsv.is_file():
        return None, meta
    try:
        table = Table.read(ecsv, format='ascii.ecsv')
    except Exception as exc:
        logger.warning('Failed to read Gaia cache %s: %s', ecsv, exc)
        return None, meta
    return table, meta


# Prefer CDS, then common mirrors (astroquery Conf defaults).
VIZIER_MIRRORS: tuple[str, ...] = (
    'vizier.cds.unistra.fr',
    'vizier.cfa.harvard.edu',
    'vizier.nao.ac.jp',
    'vizier.ast.cam.ac.uk',
    'vizier.china-vo.org',
)

# Fail fast: 5 mirrors x 2 tries x 60s was burning ~10 min before TAP.
VIZIER_TIMEOUT_SEC = 15.0
VIZIER_RETRIES_PER_MIRROR = 1
# In auto mode only probe this many mirrors before falling through to TAP.
VIZIER_MAX_MIRRORS_AUTO = 2

# Gaia@AIP IVOA cone search (stdlib urllib; independent of ESA/Vizier).
AIP_GAIA_CONE_URL = {
    'gaiadr3': 'https://gaia.aip.de/conesearch/api/gaiadr3.gaia_source/',
    'gaiadr2': 'https://gaia.aip.de/conesearch/api/gaiadr2.gaia_source/',
}
AIP_GAIA_TAP_URL = 'https://gaia.aip.de/tap'

_JHAT_GAIA_PATCH_MARKER = '_st123_get_GAIA_sources_vizier'


def _query_gaia_vizier(
    coord: SkyCoord,
    radius: u.Quantity,
    *,
    dr: str = 'gaiadr3',
    retries_per_mirror: int = VIZIER_RETRIES_PER_MIRROR,
    max_mirrors: int | None = None,
    timeout_sec: float = VIZIER_TIMEOUT_SEC,
) -> Table:
    """Query Gaia via CDS VizieR, retrying and rotating mirrors on failure."""
    import time

    from astroquery.vizier import Vizier
    from astroquery.vizier import conf as vizier_conf

    catalog = _vizier_catalog_for_dr(dr)
    columns = [
        'RA_ICRS',
        'DE_ICRS',
        'pmRA',
        'pmDE',
        'e_pmRA',
        'e_pmDE',
        'Gmag',
        'BPmag',
        'RPmag',
        'Source',
        'Epoch',
    ]
    mirrors = list(VIZIER_MIRRORS)
    if max_mirrors is not None:
        mirrors = mirrors[: max(1, int(max_mirrors))]
    errors: list[str] = []
    prev_cls_timeout = getattr(Vizier, 'TIMEOUT', None)
    prev_conf_timeout = getattr(vizier_conf, 'timeout', None)
    try:
        # Both class and conf timeouts must be set -- astroquery requests
        # use conf.timeout for connect/read (default 60s). conf expects int.
        timeout_i = max(1, int(round(float(timeout_sec))))
        Vizier.TIMEOUT = timeout_i
        vizier_conf.timeout = timeout_i
        for server in mirrors:
            for attempt in range(1, retries_per_mirror + 1):
                try:
                    viz = Vizier(
                        columns=columns,
                        row_limit=-1,
                        vizier_server=server,
                    )
                    viz.TIMEOUT = timeout_i
                    logger.info(
                        'Gaia Vizier query %s @ %s around %s (r=%.3f deg) '
                        '[try %d/%d timeout=%.0fs]',
                        catalog,
                        server,
                        coord.to_string('decimal'),
                        radius.to_value(u.deg),
                        attempt,
                        retries_per_mirror,
                        float(timeout_sec),
                    )
                    result = viz.query_region(
                        coord, radius=radius, catalog=catalog
                    )
                    # Empty TableList = no sources in cone (success).
                    if result is None or len(result) == 0:
                        logger.info(
                            'Vizier %s: no Gaia sources in cone',
                            server,
                        )
                        return Table()
                    return _normalize_vizier_gaia(result[0])
                except Exception as exc:
                    msg = f'{server} try {attempt}: {type(exc).__name__}: {exc}'
                    errors.append(msg)
                    logger.warning('Vizier Gaia query failed (%s)', msg)
                    time.sleep(min(1.0 * attempt, 2.0))
    finally:
        if prev_cls_timeout is not None:
            Vizier.TIMEOUT = prev_cls_timeout
        if prev_conf_timeout is not None:
            vizier_conf.timeout = prev_conf_timeout
    raise RuntimeError(
        'All Vizier mirrors failed for Gaia cone query. Tried: '
        + '; '.join(errors)
    )


def _gaia_adql_cone(coord: SkyCoord, radius: u.Quantity, *, dr: str) -> str:
    table_name = _tap_table_for_dr(dr)
    ra = float(coord.ra.degree)
    dec = float(coord.dec.degree)
    r_deg = float(radius.to_value(u.deg))
    return (
        'SELECT source_id, ra, dec, pmra, pmdec, pmra_error, pmdec_error, '
        'phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, ref_epoch '
        f'FROM {table_name} '
        "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra}, {dec}, {r_deg}))"
    )


def _query_gaia_esa_tap(
    coord: SkyCoord,
    radius: u.Quantity,
    *,
    dr: str = 'gaiadr3',
) -> Table:
    """
    Query Gaia via ESA Archive TAP (``astroquery.gaia``).

    Prefer async jobs so large cones are not rejected by the synchronous
    queue limit; fall back to sync if async fails.
    """
    from st123.utils.logging import capture_output

    with capture_output(discard=True):
        from astroquery.gaia import Gaia

    query = _gaia_adql_cone(coord, radius, dr=dr)
    logger.info(
        'Gaia ESA TAP query %s around %s (r=%.3f deg)',
        _tap_table_for_dr(dr),
        coord.to_string('decimal'),
        radius.to_value(u.deg),
    )
    # Single async->sync attempt: the TAP chain falls through to Gaia@AIP
    # quickly when ESA is 5xx / queue-saturated (sn2004g case).
    try:
        job = Gaia.launch_job_async(query)
        result = job.get_results()
    except Exception as async_exc:
        logger.warning(
            'Gaia ESA TAP async failed (%s: %s); trying synchronous',
            type(async_exc).__name__,
            async_exc,
        )
        try:
            job = Gaia.launch_job(query)
            result = job.get_results()
        except Exception as sync_exc:
            raise RuntimeError(
                f'ESA TAP failed (async={type(async_exc).__name__}: {async_exc}; '
                f'sync={type(sync_exc).__name__}: {sync_exc})'
            ) from sync_exc
    if result is None:
        return Table()
    return _normalize_tap_gaia(Table(result))


def _query_gaia_aip_cone(
    coord: SkyCoord,
    radius: u.Quantity,
    *,
    dr: str = 'gaiadr3',
    timeout_sec: float = 60.0,
) -> Table:
    """
    Query Gaia via Gaia@AIP IVOA cone-search HTTP (no ESA / Vizier dependency).

    Returns the shared TAP-style column schema used elsewhere in st123.
    """
    from io import BytesIO
    from urllib.error import HTTPError, URLError
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    key = str(dr).strip().lower().replace('_', '').replace('.', '')
    if key in ('gaiadr2', 'dr2', 'gaia2'):
        base = AIP_GAIA_CONE_URL['gaiadr2']
    else:
        base = AIP_GAIA_CONE_URL['gaiadr3']
    params = urlencode(
        {
            'RA': f'{float(coord.ra.degree):.8f}',
            'DEC': f'{float(coord.dec.degree):.8f}',
            'SR': f'{float(radius.to_value(u.deg)):.8f}',
            'VERB': '3',
        }
    )
    url = f'{base}?{params}'
    logger.info(
        'Gaia AIP cone query around %s (r=%.3f deg)',
        coord.to_string('decimal'),
        radius.to_value(u.deg),
    )
    req = Request(url, headers={'Accept': 'application/x-votable+xml,application/xml'})
    try:
        with urlopen(req, timeout=float(timeout_sec)) as resp:
            payload = resp.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f'AIP Gaia cone failed: {type(exc).__name__}: {exc}') from exc
    table = Table.read(BytesIO(payload), format='votable')
    return _normalize_tap_gaia(table)


def _query_gaia_aip_tap(
    coord: SkyCoord,
    radius: u.Quantity,
    *,
    dr: str = 'gaiadr3',
) -> Table:
    """Query Gaia via Gaia@AIP TAP (``pyvo`` when installed)."""
    try:
        import pyvo
    except ImportError as exc:
        raise RuntimeError('pyvo is not installed for AIP TAP') from exc

    query = _gaia_adql_cone(coord, radius, dr=dr)
    logger.info(
        'Gaia AIP TAP query %s around %s (r=%.3f deg)',
        _tap_table_for_dr(dr),
        coord.to_string('decimal'),
        radius.to_value(u.deg),
    )
    service = pyvo.dal.TAPService(AIP_GAIA_TAP_URL)
    result = service.search(query)
    return _normalize_tap_gaia(result.to_table())


def _vizier_auto_query(
    coord: SkyCoord,
    radius: u.Quantity,
    *,
    dr: str = 'gaiadr3',
) -> Table:
    """Fail-fast Vizier probe used only as the last auto backend."""
    return _query_gaia_vizier(
        coord,
        radius,
        dr=dr,
        retries_per_mirror=VIZIER_RETRIES_PER_MIRROR,
        max_mirrors=VIZIER_MAX_MIRRORS_AUTO,
        timeout_sec=VIZIER_TIMEOUT_SEC,
    )


# Labels only -- resolve callables at query time so tests can patch them.
# Order: fast/reliable first; Vizier last (slow when mirrors are down).
_AUTO_GAIA_BACKEND_NAMES: tuple[str, ...] = (
    'tap_aip_cone',
    'tap_aip',
    'tap',
    'vizier',
)
_TAP_GAIA_BACKEND_NAMES: tuple[str, ...] = (
    'tap_aip_cone',
    'tap_aip',
    'tap',
)


def _gaia_backend_callable(label: str):
    """Map backend label -> query function (looked up live for patchability)."""
    if label == 'tap_aip_cone':
        return _query_gaia_aip_cone
    if label == 'tap_aip':
        return _query_gaia_aip_tap
    if label == 'tap':
        return _query_gaia_esa_tap
    if label == 'vizier':
        return _vizier_auto_query
    raise ValueError(f'unknown Gaia backend label {label!r}')


def _query_gaia_backend_chain(
    coord: SkyCoord,
    radius: u.Quantity,
    *,
    dr: str,
    backend_names: tuple[str, ...],
) -> tuple[Table, str]:
    """Try *backend_names* in order; return ``(table, label)`` or raise."""
    errors: list[str] = []
    for label in backend_names:
        fn = _gaia_backend_callable(label)
        try:
            table = fn(coord, radius, dr=dr)
            return table, label
        except Exception as exc:
            msg = f'{label}: {type(exc).__name__}: {exc}'
            errors.append(msg)
            logger.warning('Gaia %s failed (%s)', label, msg)
    raise RuntimeError(
        'All Gaia backends failed. Tried: ' + '; '.join(errors)
    )


def _query_gaia_tap_chain(
    coord: SkyCoord,
    radius: u.Quantity,
    *,
    dr: str = 'gaiadr3',
) -> tuple[Table, str]:
    """Gaia@AIP cone -> Gaia@AIP TAP -> ESA TAP (no Vizier)."""
    return _query_gaia_backend_chain(
        coord, radius, dr=dr, backend_names=_TAP_GAIA_BACKEND_NAMES
    )


def fetch_gaia_cone(
    coord: SkyCoord,
    radius: u.Quantity,
    *,
    dr: str = 'gaiadr3',
    backend: GaiaBackend = 'auto',
) -> Table:
    """
    Download Gaia sources in a cone (no image cut).

    Parameters
    ----------
    backend : {'auto', 'vizier', 'tap'}, optional
        ``auto`` (default) tries Gaia@AIP cone, Gaia@AIP TAP, ESA TAP, then
        Vizier last. ``tap`` skips Vizier. ``vizier`` forces CDS only.
    """
    be = str(backend or 'auto').strip().lower()
    if be not in ('auto', 'vizier', 'tap'):
        raise ValueError(
            f"Gaia backend {backend!r} is not allowed; use 'auto', 'vizier', or 'tap'"
        )

    if be == 'vizier':
        table = _query_gaia_vizier(coord, radius, dr=dr)
        used = 'vizier'
    elif be == 'tap':
        table, used = _query_gaia_tap_chain(coord, radius, dr=dr)
    else:
        table, used = _query_gaia_backend_chain(
            coord, radius, dr=dr, backend_names=_AUTO_GAIA_BACKEND_NAMES
        )

    out = _add_pm_ratio(table)
    # Record which path produced this catalog (field-cache metadata).
    try:
        out.meta['st123_gaia_backend'] = used
    except Exception:
        pass
    logger.info(
        'Gaia cone via %s: %d source(s) (r=%.3f deg)',
        used,
        len(out),
        radius.to_value(u.deg),
    )
    return out


def jhat_get_gaia_sources(
    ra0,
    dec0,
    radius_deg,
    radius_factor=1.1,
    mjd=None,
    pm_median=False,
    datarelease='dr3',
    calc_mag_errors=True,
    rename_mag_colnames=True,
    remove_null=True,
    columns=None,
):
    """
    Drop-in for JHAT ``get_GAIA_sources`` using :func:`fetch_gaia_cone`.

    Prefers Vizier, falls back to ESA TAP when Vizier is down. Returns
    ``(dataframe, racol, deccol)`` matching JHAT's expected shape.
    """
    import pandas as pd
    from astropy.time import Time

    del calc_mag_errors, rename_mag_colnames  # Gmag / phot_g supplied by backends
    dr = datarelease or 'gaiadr3'
    if str(dr).lower() in ('dr2', 'gaiadr2'):
        dr_key = 'gaiadr2'
    else:
        dr_key = 'gaiadr3'
    r_deg = float(radius_deg) * float(radius_factor)
    coord = SkyCoord(float(ra0), float(dec0), unit='deg', frame='icrs')
    tb = fetch_gaia_cone(coord, r_deg * u.deg, dr=dr_key, backend='auto')
    used = tb.meta.get('st123_gaia_backend', 'auto') if hasattr(tb, 'meta') else 'auto'
    logger.info(
        'JHAT Gaia via %s: %d sources (r=%.4f deg)', used, len(tb), r_deg
    )

    n = len(tb)
    ra = np.asarray(tb['ra'], dtype=float) if n else np.array([])
    dec = np.asarray(tb['dec'], dtype=float) if n else np.array([])
    pmra = (
        np.asarray(tb['pmra'], dtype=float)
        if n and 'pmra' in tb.colnames
        else np.zeros(n)
    )
    pmdec = (
        np.asarray(tb['pmdec'], dtype=float)
        if n and 'pmdec' in tb.colnames
        else np.zeros(n)
    )
    e_pmra = (
        np.asarray(tb['pmra_error'], dtype=float)
        if n and 'pmra_error' in tb.colnames
        else np.full(n, np.nan)
    )
    e_pmdec = (
        np.asarray(tb['pmdec_error'], dtype=float)
        if n and 'pmdec_error' in tb.colnames
        else np.full(n, np.nan)
    )
    gmag = (
        np.asarray(tb['phot_g_mean_mag'], dtype=float)
        if n and 'phot_g_mean_mag' in tb.colnames
        else np.full(n, np.nan)
    )
    bpmag = (
        np.asarray(tb['BPmag'], dtype=float)
        if n and 'BPmag' in tb.colnames
        else np.full(n, np.nan)
    )
    rpmag = (
        np.asarray(tb['RPmag'], dtype=float)
        if n and 'RPmag' in tb.colnames
        else np.full(n, np.nan)
    )
    if n and 'Epoch' in tb.colnames:
        epoch = np.asarray(tb['Epoch'], dtype=float)
    else:
        epoch = np.full(n, 2016.0 if dr_key == 'gaiadr3' else 2015.5)
    sid = (
        np.asarray(tb['source_id'])
        if n and 'source_id' in tb.colnames
        else np.arange(n)
    )

    racol, deccol = 'ra', 'dec'
    ra_out, dec_out = ra.copy(), dec.copy()
    if mjd is not None and n:
        time_gaia = Time(epoch, format='jyear')
        time_obs = Time(float(mjd), format='mjd')
        dt_yr = (time_obs - time_gaia).to(u.yr).value
        cosd = np.cos(np.deg2rad(dec))
        cosd = np.where(np.abs(cosd) < 1e-6, 1e-6, cosd)
        dRA = (dt_yr * pmra * u.mas / cosd).to(u.deg).value
        dDec = (dt_yr * pmdec * u.mas).to(u.deg).value
        if pm_median:
            ok = np.isfinite(dRA) & np.isfinite(dDec)
            dRA_m = float(np.median(dRA[ok])) if ok.any() else 0.0
            dDec_m = float(np.median(dDec[ok])) if ok.any() else 0.0
            ra_out = ra + dRA_m
            dec_out = dec + dDec_m
        else:
            ra_out = ra + dRA
            dec_out = dec + dDec
        racol, deccol = 'ra1', 'dec1'

    df = pd.DataFrame(
        {
            'source_id': sid,
            'ref_epoch': epoch,
            'ra': ra,
            'ra_error': np.full(n, np.nan),
            'dec': dec,
            'dec_error': np.full(n, np.nan),
            'pmra': pmra,
            'pmra_error': e_pmra,
            'pmdec': pmdec,
            'pmdec_error': e_pmdec,
            'g': gmag,
            'g_err': np.full(n, np.nan),
            'bp': bpmag,
            'bp_err': np.full(n, np.nan),
            'rp': rpmag,
            'rp_err': np.full(n, np.nan),
        }
    )
    df['bp_rp'] = df['bp'] - df['rp']
    df['bp_g'] = df['bp'] - df['g']
    df['g_rp'] = df['g'] - df['rp']
    for col in ('bp_rp_err', 'bp_g_err', 'g_rp_err'):
        df[col] = np.nan
    if racol == 'ra1':
        df['ra1'] = ra_out
        df['dec1'] = dec_out
        df['ref_epoch1'] = float(Time(float(mjd), format='mjd').decimalyear)

    if remove_null and len(df):
        df = df[np.isfinite(df[racol]) & np.isfinite(df[deccol])]

    if columns is not None:
        cols = list(columns)
        if racol == 'ra1':
            for extra in ('ra1', 'dec1', 'ref_epoch1'):
                if extra not in cols:
                    cols.append(extra)
        for col in cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[cols]

    return df, racol, deccol


# Stable name for JHAT monkeypatch identity checks.
jhat_get_gaia_sources.__name__ = _JHAT_GAIA_PATCH_MARKER


def install_jhat_gaia_vizier_patch() -> None:
    """
    Route JHAT Gaia lookups through :func:`jhat_get_gaia_sources` (idempotent).

    Replaces ``jhat.simple_jwst_phot.get_GAIA_sources`` so JHAT uses the same
    Gaia@AIP-first / ESA / Vizier-last path as the rest of st123.
    """
    try:
        import jhat.simple_jwst_phot as sjp
    except ImportError:
        return

    cur = getattr(sjp, 'get_GAIA_sources', None)
    if cur is not None and getattr(cur, '__name__', '') == _JHAT_GAIA_PATCH_MARKER:
        return

    sjp.get_GAIA_sources = jhat_get_gaia_sources
    logger.info('Installed JHAT Gaia patch (AIP cone first, Vizier last)')


def ensure_gaia_catalog(
    images: Sequence[PathLike],
    *,
    telescope: str = 'hst',
    cache_dir: PathLike | None = None,
    dr: str = 'gaiadr3',
    backend: GaiaBackend = 'auto',
    force: bool = False,
    min_radius_deg: float = GAIA_MIN_CONE_RADIUS_DEG,
) -> Path:
    """
    Ensure a field-level Gaia catalog exists for *images*.

    Downloads once into ``<cache_dir>/<dr>.ecsv`` (default ``<reduction>/gaia/``)
    via Vizier with ESA TAP fallback, and reuses the cache when the cone already
    covers all footprints. Cone radius is at least *min_radius_deg*
    (default 0.1 deg).
    """
    paths = [Path(p).expanduser() for p in images]
    if not paths:
        raise ValueError('ensure_gaia_catalog requires at least one image')
    if cache_dir is None:
        cache_dir = default_gaia_cache_dir(paths[0])
    if cache_dir is None:
        raise ValueError(
            'could not locate reduction/gaia cache dir; pass cache_dir= explicitly'
        )
    cache_dir = Path(cache_dir)
    coord, radius = _union_cone(
        paths, telescope=telescope, min_radius_deg=min_radius_deg
    )
    ecsv, meta_path, _ = gaia_cache_paths(cache_dir, dr=dr)
    table, meta = load_gaia_cache(cache_dir, dr=dr)
    if (
        not force
        and table is not None
        and meta is not None
        and _cache_covers(meta, coord, radius)
    ):
        logger.info(
            'Reusing field Gaia cache %s (%d sources)',
            ecsv,
            len(table),
        )
        return ecsv

    logger.info(
        'Fetching field Gaia catalog for %d image(s) -> %s',
        len(paths),
        cache_dir,
    )
    table = fetch_gaia_cone(coord, radius, dr=dr, backend=backend)
    used = str(
        (getattr(table, 'meta', None) or {}).get('st123_gaia_backend') or backend
    )
    return _save_gaia_cache(
        table,
        cache_dir=cache_dir,
        coord=coord,
        radius=radius,
        dr=dr,
        backend=used,
    )


def write_gaia_refcat(
    image: PathLike,
    output: PathLike,
    *,
    telescope: str = 'hst',
    cache_dir: PathLike | None = None,
    dr: str = 'gaiadr3',
    backend: GaiaBackend = 'auto',
) -> Path:
    """
    Write a whitespace phot/refcat table (``ra dec mag [x y]``) from the field
    Gaia cache, cut to *image*.

    Suitable as JHAT ``refcatname`` / science-frame alignment phot catalog.
    """
    image_s = str(Path(image).expanduser())
    gaia = query_gaia(
        image_s,
        dr=dr,
        telescope=telescope,
        backend=backend,
        cache_dir=cache_dir,
        use_cache=True,
    )
    out = Path(output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(gaia) == 0:
        out.write_text('ra dec mag x y\n')
        logger.warning('Wrote empty Gaia refcat %s', out)
        return out

    ra = np.asarray(gaia['ra'], dtype=float)
    dec = np.asarray(gaia['dec'], dtype=float)
    if 'phot_g_mean_mag' in gaia.colnames:
        mag = np.asarray(gaia['phot_g_mean_mag'], dtype=float)
    elif 'mag' in gaia.colnames:
        mag = np.asarray(gaia['mag'], dtype=float)
    else:
        mag = np.full(len(gaia), np.nan)
    w = as_datamodel(image_s).sci_wcs()
    x, y = w.all_world2pix(ra, dec, 0)
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write('ra dec mag x y\n')
        for i in range(len(ra)):
            if not (np.isfinite(ra[i]) and np.isfinite(dec[i])):
                continue
            m = mag[i] if np.isfinite(mag[i]) else 99.0
            fh.write(
                f'{ra[i]:.10f} {dec[i]:.10f} {m:.4f} {float(x[i]):.4f} {float(y[i]):.4f}\n'
            )
    logger.info('Wrote Gaia refcat %s (%d sources)', out, len(ra))
    return out


def query_gaia(
    image: str,
    dr: str = 'gaiadr3',
    telescope: str = 'jwst',
    save_file: str | bool = False,
    *,
    backend: GaiaBackend = 'auto',
    cache_dir: PathLike | None = None,
    use_cache: bool = True,
) -> Table:
    """
    Query Gaia for sources covering an image.

    Uses CDS VizieR (with ESA TAP fallback) and a field-level cache under
    ``<reduction>/gaia/`` so each pipeline stage reuses one download.

    Parameters
    ----------
    image : str
        Image file name.
    dr : str, optional
        Gaia data release (``gaiadr3`` / ``gaiadr2``).
    telescope : str, optional
        ``'jwst'`` uses the ImageModel GWCS; ``'hst'`` uses the SCI WCS.
    save_file : str or bool, optional
        Path to save the ``ra``/``dec`` list, or ``False`` to skip.
    backend : {'auto', 'vizier', 'tap'}, optional
        ``auto`` (default) tries Vizier then ESA TAP.
    cache_dir : path-like or None, optional
        Directory for the shared field catalog (default:
        ``<reduction>/gaia`` inferred from *image*).
    use_cache : bool, optional
        If True (default), read/write the shared field cache.

    Returns
    -------
    astropy.table.Table
        Gaia sources inside the image (columns include ``ra`` / ``dec``).
    """
    coord, radius = _image_cone(image, telescope=telescope)
    table: Table | None = None

    resolved_cache = Path(cache_dir).expanduser() if cache_dir else None
    if use_cache and resolved_cache is None:
        resolved_cache = default_gaia_cache_dir(image)

    if use_cache and resolved_cache is not None:
        cached, meta = load_gaia_cache(resolved_cache, dr=dr)
        if cached is not None and meta is not None and _cache_covers(meta, coord, radius):
            logger.info(
                'Gaia from field cache %s (%d field sources)',
                gaia_cache_paths(resolved_cache, dr=dr)[0].name,
                len(cached),
            )
            table = cached
        else:
            ensure_gaia_catalog(
                [image],
                telescope=telescope,
                cache_dir=resolved_cache,
                dr=dr,
                backend=backend,
            )
            table, _meta = load_gaia_cache(resolved_cache, dr=dr)

    if table is None:
        table = fetch_gaia_cone(coord, radius, dr=dr, backend=backend)

    if len(table) == 0:
        logger.info('Number of Gaia stars: 0')
        return table

    table = _add_pm_ratio(table)
    table = cut_gaia_sources(image, table)
    logger.info('Number of Gaia stars: %d', len(table))

    if save_file:
        logger.info('Saving Gaia query to %s', save_file)
        np.savetxt(save_file, np.array(table[['ra', 'dec']]), fmt='%s')

    return table
