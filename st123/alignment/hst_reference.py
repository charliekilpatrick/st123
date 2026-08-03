"""
Select the best HST level-3 coadd as the JHAT absolute-alignment reference.

Ranks ``*_drc.fits`` / ``*_drz.fits`` (and ``coadd_*.fits``) by how many Gaia
sources land on illuminated pixels that are also locally detectable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Level3GaiaScore:
    """Gaia / illumination score for one level-3 FITS product."""

    path: Path
    n_gaia_fov: int
    n_illuminated: int
    n_detectable: int
    instrument: str
    filt: str

    @property
    def score(self) -> int:
        """Primary ranking key: detectable Gaia on illuminated pixels."""
        return int(self.n_detectable)


def _sci_hdu_index(hdul: fits.HDUList) -> int:
    for i, hdu in enumerate(hdul):
        if getattr(hdu, 'name', '') == 'SCI' and hdu.data is not None:
            return i
    for i, hdu in enumerate(hdul):
        if hdu.data is not None and getattr(hdu.data, 'ndim', 0) == 2:
            return i
    raise ValueError('No 2-D science HDU found')


def _illuminated_mask(
    sci: np.ndarray,
    wht: Optional[np.ndarray] = None,
    dq: Optional[np.ndarray] = None,
) -> np.ndarray:
    mask = np.isfinite(sci)
    if wht is not None:
        mask &= np.asarray(wht) > 0
    if dq is not None:
        # Treat severe DQ (bit 512+ style) as unilluminated when present.
        mask &= np.asarray(dq) < 512
    return mask


def _local_detectable(
    sci: np.ndarray,
    illuminated: np.ndarray,
    x: float,
    y: float,
    *,
    half: int = 4,
    snr: float = 5.0,
) -> bool:
    """True when a Gaia position shows a local peak above *snr* × sky RMS."""
    ny, nx = sci.shape
    xi, yi = int(round(x)), int(round(y))
    if xi < half or yi < half or xi >= nx - half or yi >= ny - half:
        return False
    if not illuminated[yi, xi]:
        return False
    stamp = sci[yi - half : yi + half + 1, xi - half : xi + half + 1]
    ill = illuminated[yi - half : yi + half + 1, xi - half : xi + half + 1]
    if stamp.size == 0 or not np.any(ill):
        return False
    vals = stamp[ill]
    if vals.size < 8:
        return False
    # Robust sky from outer ring of the stamp.
    yy, xx = np.ogrid[-half : half + 1, -half : half + 1]
    ring = (xx * xx + yy * yy) >= (half - 1) ** 2
    sky_vals = stamp[ring & ill]
    if sky_vals.size < 4:
        sky_vals = vals
    sky = float(np.nanmedian(sky_vals))
    rms = float(np.nanmedian(np.abs(sky_vals - sky))) * 1.4826
    rms = max(rms, 1e-6)
    peak = float(np.nanmax(vals))
    return (peak - sky) >= snr * rms


def score_level3_gaia(
    image: PathLike,
    *,
    snr: float = 5.0,
    half: int = 4,
) -> Level3GaiaScore:
    """
    Count Gaia sources on illuminated, locally detectable pixels in *image*.
    """
    from st123.alignment.align import query_gaia
    from st123.utils.helpers import get_filter, get_instrument

    path = Path(image).expanduser().resolve()
    inst = get_instrument(path).split('_')[0].lower()
    filt = get_filter(path).lower()

    try:
        gaia = query_gaia(str(path), telescope='hst')
    except Exception as exc:
        logger.warning('Gaia query failed for %s: %s', path.name, exc)
        return Level3GaiaScore(path, 0, 0, 0, inst, filt)

    with fits.open(path, memmap=True) as hdul:
        idx = _sci_hdu_index(hdul)
        sci = np.asarray(hdul[idx].data, dtype=np.float64)
        hdr = hdul[idx].header
        wht = None
        dq = None
        if 'WHT' in hdul and hdul['WHT'].data is not None:
            wht = np.asarray(hdul['WHT'].data)
        if 'DQ' in hdul and hdul['DQ'].data is not None:
            dq = np.asarray(hdul['DQ'].data)
        w = WCS(hdr)

    illuminated = _illuminated_mask(sci, wht=wht, dq=dq)
    n_fov = len(gaia)
    n_illum = 0
    n_det = 0
    if n_fov == 0:
        return Level3GaiaScore(path, 0, 0, 0, inst, filt)

    xs, ys = w.all_world2pix(
        np.asarray(gaia['ra'], dtype=float),
        np.asarray(gaia['dec'], dtype=float),
        0,
    )
    ny, nx = sci.shape
    for x, y in zip(xs, ys):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or yi < 0 or xi >= nx or yi >= ny:
            continue
        if not illuminated[yi, xi]:
            continue
        n_illum += 1
        if _local_detectable(sci, illuminated, x, y, half=half, snr=snr):
            n_det += 1

    return Level3GaiaScore(path, n_fov, n_illum, n_det, inst, filt)


def list_level3_products(directory: PathLike) -> list[Path]:
    """Find coadd / drizzle products under *directory*."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return []
    out: list[Path] = []
    for pat in ('coadd_*.fits', '*_drc.fits', '*_drz.fits'):
        for path in sorted(root.glob(pat)):
            name = path.name.lower()
            if name.endswith('_wht.fits') or name.endswith('_ctx.fits'):
                continue
            out.append(path.resolve())
    # De-dupe preserving order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def pick_best_level3(
    candidates: Sequence[PathLike] | None = None,
    *,
    directory: PathLike | None = None,
    snr: float = 5.0,
) -> tuple[Optional[Path], list[Level3GaiaScore]]:
    """
    Pick the level-3 image with the most detectable Gaia stars on illuminated pixels.

    Prefers non-WFPC2 products on score ties (deeper modern detectors).
    """
    paths: list[Path] = []
    if candidates:
        paths.extend(Path(p).expanduser().resolve() for p in candidates)
    if directory is not None:
        paths.extend(list_level3_products(directory))
    # unique
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in paths:
        key = str(p)
        if key in seen or not p.is_file():
            continue
        seen.add(key)
        uniq.append(p)
    if not uniq:
        return None, []

    scores = [score_level3_gaia(p, snr=snr) for p in uniq]
    # Rank: detectable desc, illuminated desc, prefer non-WFPC2, then FOV count.
    scores_sorted = sorted(
        scores,
        key=lambda s: (
            s.n_detectable,
            s.n_illuminated,
            0 if s.instrument != 'wfpc2' else -1,
            s.n_gaia_fov,
        ),
        reverse=True,
    )
    best = scores_sorted[0]
    logger.info(
        'Best level-3 reference: %s (detectable Gaia=%d, illuminated=%d, FOV=%d)',
        best.path.name,
        best.n_detectable,
        best.n_illuminated,
        best.n_gaia_fov,
    )
    for s in scores_sorted:
        logger.info(
            '  L3 %-40s det=%3d illum=%3d fov=%3d (%s/%s)',
            s.path.name,
            s.n_detectable,
            s.n_illuminated,
            s.n_gaia_fov,
            s.instrument,
            s.filt,
        )
    return best.path, scores_sorted


def ensure_preliminary_level3s(
    raw_dir: PathLike,
    outdir: PathLike,
    *,
    num_cores: int = 4,
    force: bool = False,
) -> list[Path]:
    """
    Build per-filter AstroDrizzle coadds from *raw_dir* into *outdir*.

    Used before JHAT so a Gaia-rich level-3 exists as the absolute reference.
    Skips groups whose coadd already exists unless *force*.
    """
    from st123.mosaic.hst_drizzle import (
        drizzle_filter_group,
        group_hst_frames,
        _drizzle_suffix,
    )

    raw = Path(raw_dir).expanduser().resolve()
    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    frames: list[Path] = []
    for pat in ('*flc.fits', '*flt.fits', '*c0m.fits'):
        frames.extend(sorted(raw.glob(pat)))
    frames = [p for p in frames if not p.name.lower().endswith('c1m.fits')]
    if not frames:
        logger.warning('No raw science frames under %s for preliminary L3', raw)
        return []

    groups = group_hst_frames(frames)
    products: list[Path] = []
    for (inst, filt), imgs in sorted(groups.items()):
        suffix = _drizzle_suffix(inst)
        dest = out / f'coadd_{inst}_{filt}_{suffix}.fits'
        if dest.is_file() and not force:
            logger.info('Keeping existing preliminary L3 %s', dest.name)
            products.append(dest)
            continue
        logger.info(
            'Building preliminary L3 %s from %d raw frame(s)',
            dest.name,
            len(imgs),
        )
        product = drizzle_filter_group(
            imgs,
            dest,
            instrument=inst,
            num_cores=num_cores,
            clean=True,
            build=True,
        )
        products.append(Path(product))
    return products
