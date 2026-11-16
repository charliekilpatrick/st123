"""
Multi-telescope footprint helpers for JWST mosaic box-splitting.

When deciding whether a mosaic box exceeds ``N_max``, count overlapping
NIRCam science frames **plus** weight footprints from MIRI and HST ACS/WFC3
chips so dual-mode fields split before DOLPHOT warmstart stacks become too
deep.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import shapely
from st123.datamodels import as_datamodel
from astropy.wcs import WCS

from st123.stages.download import parse_s_region
from st123.utils.helpers import get_instrument

logger = logging.getLogger(__name__)

PathLike = str | Path

# Instruments whose chips count toward multi-telescope N_max (not mosaicked
# as JWST science in the NIRCam mosaic step).
_HST_WEIGHT_INSTS = frozenset({'acs', 'wfc3'})


def _candidate_jhat_dirs(work: Path, *, telescope: str) -> list[Path]:
    """Prefer dedicated jhat_* trees, then legacy ``jhat/``."""
    tel = telescope.lower()
    preferred = work / ('jhat_hst' if tel == 'hst' else 'jhat_jwst')
    legacy = work / 'jhat'
    out: list[Path] = []
    for d in (preferred, legacy):
        if d.is_dir() and d not in out:
            out.append(d)
    return out


def sky_polygons_from_fits(path: PathLike) -> list[shapely.Polygon]:
    """
    Return sky (RA, Dec) polygons for every SCI extension with ``S_REGION``.

    Falls back to a single primary/SCI ``S_REGION`` when per-chip regions are
    absent. Returns an empty list when no usable region is found.
    """
    path = Path(path)
    pgons: list[shapely.Polygon] = []
    try:
        with as_datamodel(path).open(memmap=True) as hdul:
            for hdu in hdul:
                if getattr(hdu, 'name', '') != 'SCI':
                    continue
                region = hdu.header.get('S_REGION')
                if not region:
                    continue
                try:
                    pgons.append(parse_s_region(region))
                except Exception as exc:
                    logger.debug(
                        'Skip S_REGION on %s[%s]: %s', path.name, hdu.name, exc
                    )
            if not pgons:
                for hdu in hdul:
                    region = hdu.header.get('S_REGION')
                    if not region:
                        continue
                    try:
                        pgons.append(parse_s_region(region))
                        break
                    except Exception:
                        continue
    except Exception as exc:
        logger.warning('Could not read footprints from %s: %s', path, exc)
    return pgons


def project_sky_polygons_to_wcs(
    sky_pgons: Sequence[shapely.Polygon],
    wcs_opt: WCS,
) -> list[shapely.Polygon]:
    """Project sky polygons into the pixel frame of *wcs_opt*."""
    out: list[shapely.Polygon] = []
    for pgon in sky_pgons:
        try:
            sky = np.asarray(list(pgon.exterior.coords)[:-1], dtype=float)
            if sky.ndim != 2 or sky.shape[0] < 3:
                continue
            x, y = wcs_opt.all_world2pix(sky[:, 0], sky[:, 1], 0)
            xy = np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
            if not np.all(np.isfinite(xy)):
                continue
            pix = shapely.Polygon(xy)
            if pix.is_valid and pix.area > 0:
                out.append(pix)
        except Exception as exc:
            logger.debug('Skip sky->pix projection: %s', exc)
    return out


def _is_miri_name(name: str) -> bool:
    n = name.lower()
    return 'mir' in n or 'miri' in n


def _is_jwst_name(name: str) -> bool:
    return name.lower().startswith('jw')


def collect_footprint_weight_paths(
    reduction_dir: PathLike,
    *,
    exclude: Iterable[PathLike] | None = None,
    include_miri_raw: bool = True,
) -> list[Path]:
    """
    Gather MIRI + HST ACS/WFC3 FITS whose footprints should weight JWST N_max.

    Search order for MIRI: ``jhat_jwst`` / ``jhat`` MIRI JHAT, then
    ``raw/*mir*_cal.fits`` when JHAT is not yet available (NIRCam mosaic before
    MIRI align). HST: ``jhat_hst`` / ``jhat`` ACS and WFC3 only (no WFPC2).
    """
    work = Path(reduction_dir).expanduser().resolve()
    exclude_set = {Path(p).resolve() for p in (exclude or [])}
    found: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        rp = path.resolve()
        if rp in seen or rp in exclude_set:
            return
        if not path.is_file():
            return
        seen.add(rp)
        found.append(path)

    # MIRI JHAT (preferred) under JWST jhat tree / legacy jhat
    n_miri_jhat = 0
    for jhat in _candidate_jhat_dirs(work, telescope='jwst'):
        for p in sorted(jhat.glob('*_jhat.fits')):
            if _is_jwst_name(p.name) and _is_miri_name(p.name):
                before = len(found)
                _add(p)
                if len(found) > before:
                    n_miri_jhat += 1

    # MIRI raw cal only when JHAT is not ready yet (avoid double-counting)
    if include_miri_raw and n_miri_jhat == 0:
        raw = work / 'raw'
        if raw.is_dir():
            for p in sorted(raw.glob('jw*mir*_cal.fits')):
                _add(p)

    # HST ACS/WFC3 JHAT (per-MEF; chips expanded later via S_REGION)
    for jhat in _candidate_jhat_dirs(work, telescope='hst'):
        for p in sorted(jhat.glob('*_jhat.fits')):
            if _is_jwst_name(p.name) or p.name.lower().startswith('coadd_'):
                continue
            if 'l3_ref' in p.parts:
                continue
            try:
                inst = get_instrument(p).split('_')[0].lower()
            except Exception:
                continue
            if inst in _HST_WEIGHT_INSTS:
                _add(p)

    logger.info(
        'Footprint weights: %d file(s) (MIRI JHAT/raw + HST ACS/WFC3)',
        len(found),
    )
    return found


def build_weight_pgons(
    weight_files: Sequence[PathLike],
    wcs_opt: WCS,
) -> np.ndarray:
    """
    Build pixel-frame polygons for *weight_files* on *wcs_opt*.

    Each ACS/WFC SCI chip with ``S_REGION`` becomes one weight polygon.
    """
    pix_pgons: list[shapely.Polygon] = []
    for path in weight_files:
        sky = sky_polygons_from_fits(path)
        if not sky:
            continue
        pix_pgons.extend(project_sky_polygons_to_wcs(sky, wcs_opt))
    logger.info(
        'Footprint weights: %d chip/frame polygon(s) from %d file(s)',
        len(pix_pgons),
        len(weight_files),
    )
    return np.array(pix_pgons, dtype=object)
