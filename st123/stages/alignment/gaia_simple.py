"""
Translation-only Gaia anchoring via CRPIX shifts (sparse-Gaia fallback).

Used when JHAT's multi-star ``general`` fit fails on deep coadds with only a
handful of Gaia calibrators. Measures flux centroids around Gaia predictions,
sigma-clips, and applies a flux-weighted mean (dx, dy) to CRPIX.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GaiaSimpleMatch:
    ra: float
    dec: float
    x_pred: float
    y_pred: float
    x_meas: float
    y_meas: float
    dx: float
    dy: float
    flux: float


def _sci_index(hdul: fits.HDUList) -> int:
    for i, h in enumerate(hdul):
        if getattr(h, 'name', '') == 'SCI' and h.data is not None:
            return i
    for i, h in enumerate(hdul):
        if h.data is not None and getattr(h.data, 'ndim', 0) >= 2:
            if 'CRPIX1' in h.header and 'CRVAL1' in h.header:
                return i
    raise ValueError('No science HDU with WCS found')


def _bad_mask(hdul: fits.HDUList, shape: tuple[int, int]) -> np.ndarray:
    ny, nx = shape
    for name in ('DQ', 'WHT', 'CTX'):
        if name not in hdul or hdul[name].data is None:
            continue
        arr = np.asarray(hdul[name].data)
        if arr.shape[-2:] != (ny, nx):
            continue
        if name == 'DQ':
            return np.asarray(arr != 0)
        if name == 'WHT':
            return np.asarray(~np.isfinite(arr) | (arr <= 0))
        if name == 'CTX':
            return np.asarray(arr == 0)
    return np.zeros((ny, nx), dtype=bool)


def _pixel_scale_arcsec(w: WCS) -> float:
    sc = proj_plane_pixel_scales(w)
    v = float(np.nanmedian(sc)) * 3600.0
    if not np.isfinite(v) or v <= 0:
        raise RuntimeError('gaia_simple: could not derive pixel scale')
    return v


def _centroid(
    data: np.ndarray,
    bad: np.ndarray,
    *,
    x0: float,
    y0: float,
    r_pix: float,
) -> tuple[float, float, float]:
    from photutils.centroids import centroid_com

    ny, nx = data.shape
    r = float(r_pix)
    x_min = max(0, int(math.floor(x0 - r - 1)))
    x_max = min(nx - 1, int(math.ceil(x0 + r + 1)))
    y_min = max(0, int(math.floor(y0 - r - 1)))
    y_max = min(ny - 1, int(math.ceil(y0 + r + 1)))
    sub = np.asarray(data[y_min : y_max + 1, x_min : x_max + 1], dtype=float)
    msub = np.asarray(bad[y_min : y_max + 1, x_min : x_max + 1], dtype=bool)
    yy, xx = np.ogrid[0 : sub.shape[0], 0 : sub.shape[1]]
    cx = float(x0) - float(x_min)
    cy = float(y0) - float(y_min)
    circ = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    good = circ & (~msub) & np.isfinite(sub)
    if int(np.count_nonzero(good)) < 5:
        raise RuntimeError('too few pixels')
    ann = ((xx - cx) ** 2 + (yy - cy) ** 2 <= (2 * r) ** 2) & (~circ)
    ann_good = ann & (~msub) & np.isfinite(sub)
    bkg = float(
        np.nanmedian(sub[ann_good])
        if int(np.count_nonzero(ann_good)) >= 10
        else np.nanmedian(sub[good])
    )
    sub2 = sub - bkg
    sub2[~good] = 0.0
    sub_pos = np.where(sub2 > 0, sub2, 0.0)
    if not np.any(sub_pos > 0):
        raise RuntimeError('no positive flux')
    cx2, cy2 = centroid_com(sub_pos)
    flux = float(np.nansum(sub2[good]))
    return float(x_min) + float(cx2), float(y_min) + float(cy2), flux


def _measure_offsets(
    w: WCS,
    data: np.ndarray,
    bad: np.ndarray,
    gaia: Table,
    *,
    search_radius_arcsec: float,
    max_sources: int = 200,
) -> list[GaiaSimpleMatch]:
    r_pix = float(search_radius_arcsec) / _pixel_scale_arcsec(w)
    rows = gaia
    magcol = 'phot_g_mean_mag' if 'phot_g_mean_mag' in rows.colnames else (
        'mag' if 'mag' in rows.colnames else None
    )
    if magcol is not None:
        rows = rows[np.argsort(np.asarray(rows[magcol], dtype=float))]
    if max_sources > 0:
        rows = rows[: int(max_sources)]
    ny, nx = data.shape
    out: list[GaiaSimpleMatch] = []
    for r in rows:
        ra = float(r['ra'])
        dec = float(r['dec'])
        try:
            x_pred, y_pred = w.world_to_pixel_values(ra, dec)
        except Exception:
            continue
        if not (np.isfinite(x_pred) and np.isfinite(y_pred)):
            continue
        if (
            x_pred < -r_pix
            or x_pred > (nx - 1) + r_pix
            or y_pred < -r_pix
            or y_pred > (ny - 1) + r_pix
        ):
            continue
        try:
            x_meas, y_meas, flux = _centroid(
                data, bad, x0=float(x_pred), y0=float(y_pred), r_pix=r_pix
            )
        except Exception:
            continue
        out.append(
            GaiaSimpleMatch(
                ra=ra,
                dec=dec,
                x_pred=float(x_pred),
                y_pred=float(y_pred),
                x_meas=float(x_meas),
                y_meas=float(y_meas),
                dx=float(x_meas - x_pred),
                dy=float(y_meas - y_pred),
                flux=float(flux),
            )
        )
    return out


def _clip_mean(
    matches: list[GaiaSimpleMatch], *, sigma: float = 3.0
) -> tuple[float, float, list[GaiaSimpleMatch]]:
    if not matches:
        return 0.0, 0.0, []
    dx = np.asarray([m.dx for m in matches], dtype=float)
    dy = np.asarray([m.dy for m in matches], dtype=float)
    r = np.hypot(dx, dy)
    rr = sigma_clip(r, sigma=float(sigma), maxiters=5, masked=True)
    keep = ~np.asarray(getattr(rr, 'mask', np.zeros_like(r, dtype=bool)))
    kept = [m for m, ok in zip(matches, keep) if bool(ok)]
    if not kept:
        return 0.0, 0.0, []
    w = np.asarray([max(0.0, float(m.flux)) for m in kept], dtype=float)
    if not np.any(w > 0):
        w = np.ones(len(kept), dtype=float)
    return (
        float(np.average([m.dx for m in kept], weights=w)),
        float(np.average([m.dy for m in kept], weights=w)),
        kept,
    )


def apply_crpix_shift(hdul: fits.HDUList, dx: float, dy: float) -> int:
    """Shift CRPIX1/2 on all WCS-carrying HDUs."""
    n = 0
    for h in hdul:
        if 'CRPIX1' in h.header and 'CRPIX2' in h.header:
            h.header['CRPIX1'] = (
                float(h.header['CRPIX1']) + float(dx),
                'gaia_simple dx (px)',
            )
            h.header['CRPIX2'] = (
                float(h.header['CRPIX2']) + float(dy),
                'gaia_simple dy (px)',
            )
            n += 1
    return n


def align_image_to_gaia_simple(
    image: PathLike,
    output: PathLike,
    *,
    coarse_radius_arcsec: float = 5.0,
    fine_radius_arcsec: float = 0.5,
    clip_sigma: float = 3.0,
    telescope: str = 'hst',
) -> dict:
    """
    Copy *image* to *output* and apply a two-pass Gaia CRPIX shift.

    Returns stats including ``dx_total``, ``dy_total``, ``n_match``.
    """
    from st123.alignment.gaia_catalog import query_gaia

    src = Path(image).expanduser().resolve()
    dst = Path(output).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    gaia = query_gaia(str(src), telescope=telescope, backend='vizier')
    if len(gaia) == 0:
        raise RuntimeError(f'gaia_simple: no Gaia sources for {src.name}')

    if dst.resolve() != src.resolve():
        import shutil

        shutil.copy2(src, dst)

    with fits.open(dst, mode='update', memmap=False) as hdul:
        idx = _sci_index(hdul)
        data = np.asarray(hdul[idx].data, dtype=float)
        bad = _bad_mask(hdul, data.shape[-2:])
        w0 = WCS(hdul[idx].header, hdul, naxis=2)

        coarse = _measure_offsets(
            w0, data, bad, gaia, search_radius_arcsec=coarse_radius_arcsec
        )
        dx_c, dy_c, coarse_kept = _clip_mean(coarse, sigma=clip_sigma)
        apply_crpix_shift(hdul, dx_c, dy_c)

        w1 = WCS(hdul[idx].header, hdul, naxis=2)
        fine = _measure_offsets(
            w1, data, bad, gaia, search_radius_arcsec=fine_radius_arcsec
        )
        dx_f, dy_f, fine_kept = _clip_mean(fine, sigma=clip_sigma)
        apply_crpix_shift(hdul, dx_f, dy_f)

        ph = hdul[0].header
        ph['GAIASIMP'] = (True, 'st123: simple Gaia CRPIX anchor')
        ph['GSCODX'] = (float(dx_c), 'coarse dx (px)')
        ph['GSCODY'] = (float(dy_c), 'coarse dy (px)')
        ph['GSFIDX'] = (float(dx_f), 'fine dx (px)')
        ph['GSFIDY'] = (float(dy_f), 'fine dy (px)')
        ph['NGAIAABS'] = (int(len(fine_kept)), 'gaia_simple fine matches')
        hdul.flush()

    stats = {
        'path': str(dst),
        'dx_total': float(dx_c + dx_f),
        'dy_total': float(dy_c + dy_f),
        'n_coarse': int(len(coarse_kept)),
        'n_match': int(len(fine_kept)),
        'n_gaia': int(len(gaia)),
    }
    logger.info(
        'gaia_simple %s: dx=%.3f dy=%.3f px (coarse=%d fine=%d / %d Gaia)',
        dst.name,
        stats['dx_total'],
        stats['dy_total'],
        stats['n_coarse'],
        stats['n_match'],
        stats['n_gaia'],
    )
    return stats


def rewrite_phot_radec(
    phot_path: PathLike,
    image: PathLike,
    *,
    output: Optional[PathLike] = None,
) -> Path:
    """
    Rewrite ``ra``/``dec`` columns in a JHAT phot table from *image* WCS.

    Keeps detection ``x``/``y``; sky coords follow the (Gaia-aligned) WCS.
    """
    import pandas as pd

    phot = Path(phot_path).expanduser().resolve()
    img = Path(image).expanduser().resolve()
    out = Path(output).expanduser().resolve() if output else phot

    df = pd.read_csv(phot, sep=r'\s+', engine='python')
    if 'x' not in df.columns or 'y' not in df.columns:
        raise ValueError(f'{phot.name} missing x/y columns')
    with fits.open(img, memmap=True) as hdul:
        idx = _sci_index(hdul)
        w = WCS(hdul[idx].header, hdul, naxis=2)
    ra, dec = w.pixel_to_world_values(
        np.asarray(df['x'], dtype=float),
        np.asarray(df['y'], dtype=float),
    )
    df['ra'] = ra
    df['dec'] = dec
    out.parent.mkdir(parents=True, exist_ok=True)
    # Match JHAT whitespace phot style.
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(' '.join(str(c) for c in df.columns) + '\n')
        for row in df.itertuples(index=False):
            fh.write(' '.join(f'{v}' for v in row) + '\n')
    logger.info('Rewrote ra/dec in %s from %s WCS (%d rows)', out.name, img.name, len(df))
    return out
