"""
Select the best HST level-3 coadd as the JHAT absolute-alignment reference.

Ranks ``*_drc.fits`` / ``*_drz.fits`` (and ``coadd_*.fits``) by how many Gaia
sources land on illuminated pixels that are also locally detectable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from astropy.io import fits
from st123.datamodels import as_datamodel
from astropy.table import Table
from astropy.wcs import WCS

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)

# Science-frame JHAT needs a dense L3 phot catalog (not sparse Gaia-only).
MIN_L3_REFCAT_SOURCES = 30


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
    """True when a Gaia position shows a local peak above *snr* x sky RMS."""
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
    from st123.stages.alignment.gaia_catalog import query_gaia
    from st123.utils.helpers import get_filter, get_instrument

    path = Path(image).expanduser().resolve()
    inst = get_instrument(path).split('_')[0].lower()
    filt = get_filter(path).lower()

    try:
        gaia = query_gaia(str(path), telescope='hst', backend='vizier')
    except Exception as exc:
        logger.warning('Gaia query failed for %s: %s', path.name, exc)
        return Level3GaiaScore(path, 0, 0, 0, inst, filt)

    with as_datamodel(path).open(memmap=True) as hdul:
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
    """
    Find coadd / drizzle products under *directory*.

    Searches the shared ``group_*/ref_*`` box layout and legacy flat
    ``coadd_*.fits`` products (e.g. ``reference_prelim/``).
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return []
    out: list[Path] = []
    patterns = (
        'group_*/ref_*/coadd_*.fits',
        'coadd_*.fits',
        '*_drc.fits',
        '*_drz.fits',
    )
    for pat in patterns:
        for path in sorted(root.glob(pat)):
            name = path.name.lower()
            if name.endswith('_wht.fits') or name.endswith('_ctx.fits'):
                continue
            if not path.is_file():
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


def count_phot_sources(phot_path: PathLike) -> int:
    """Count data rows in a whitespace JHAT ``*.phot.txt`` (0 if unreadable)."""
    path = Path(phot_path).expanduser()
    if not path.is_file():
        return 0
    try:
        text = path.read_text(encoding='utf-8', errors='replace').strip().splitlines()
    except OSError:
        return 0
    if len(text) <= 1:
        return 0
    return max(0, len(text) - 1)


def write_detection_refcat(
    image: PathLike,
    output: PathLike,
    *,
    nsigma: float = 5.0,
    fwhm: float = 2.5,
    max_sources: int = 5000,
) -> Path:
    """
    Build a dense JHAT phot catalog from DAOStarFinder on an L3 / coadd SCI.

    Writes ``ra dec mag x y`` (plus ``dmag``) using the image WCS so that after
    ``gaia_simple`` the sky frame is Gaia-tied while source density remains high
    enough for sparse-field WFPC2 matching.
    """
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    src = Path(image).expanduser().resolve()
    out = Path(output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with as_datamodel(src).open(memmap=True) as hdul:
        idx = _sci_hdu_index(hdul)
        data = np.asarray(hdul[idx].data, dtype=float)
        if data.ndim > 2:
            data = data[0]
        w = WCS(hdul[idx].header, hdul, naxis=2)
        wht = None
        if 'WHT' in hdul and hdul['WHT'].data is not None:
            wht = np.asarray(hdul['WHT'].data, dtype=float)
            if wht.ndim > 2:
                wht = wht[0]

    work = np.array(data, copy=True)
    if wht is not None:
        work = np.where(wht > 0, work, np.nan)
    finite = np.isfinite(work)
    if int(np.count_nonzero(finite)) < 1000:
        raise RuntimeError(f'write_detection_refcat: too few finite pixels in {src.name}')

    _, median, std = sigma_clipped_stats(work[finite], sigma=3.0, maxiters=5)
    std = float(max(std, 1e-6))
    finder = DAOStarFinder(fwhm=float(fwhm), threshold=float(nsigma) * std)
    # photutils rejects NaNs; fill with median for detection only.
    det = np.where(finite, work, median)
    sources = finder(det - median)
    if sources is None or len(sources) == 0:
        raise RuntimeError(f'write_detection_refcat: no sources in {src.name}')

    sources.sort('flux')
    sources.reverse()
    if len(sources) > int(max_sources):
        sources = sources[: int(max_sources)]

    x = np.asarray(sources['xcentroid'], dtype=float)
    y = np.asarray(sources['ycentroid'], dtype=float)
    ra, dec = w.all_pix2world(x, y, 0)
    flux = np.asarray(sources['flux'], dtype=float)
    flux = np.where(flux > 0, flux, np.nan)
    mag = -2.5 * np.log10(flux)
    catalog = Table(
        {
            'ra': ra,
            'dec': dec,
            'mag': mag,
            'x': x,
            'y': y,
            'dmag': np.full(len(mag), 0.05),
        }
    )
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write('ra dec mag x y dmag\n')
        for row in catalog:
            if not (
                np.isfinite(row['ra'])
                and np.isfinite(row['dec'])
                and np.isfinite(row['mag'])
            ):
                continue
            fh.write(
                f'{float(row["ra"]):.10f} {float(row["dec"]):.10f} '
                f'{float(row["mag"]):.4f} {float(row["x"]):.4f} '
                f'{float(row["y"]):.4f} {float(row["dmag"]):.4f}\n'
            )
    n = count_phot_sources(out)
    logger.info(
        'Wrote L3 detection refcat %s (%d sources, thr=%.1fsigma fwhm=%.1f)',
        out.name,
        n,
        nsigma,
        fwhm,
    )
    return out


def ensure_l3_science_refcat(
    l3_image: PathLike,
    output: PathLike,
    *,
    min_sources: int = MIN_L3_REFCAT_SOURCES,
    force: bool = False,
) -> Path:
    """
    Ensure a dense detection phot catalog exists for science-frame JHAT.

    Rebuilds when missing, *force*, or when the existing file has fewer than
    *min_sources* rows (guards against sparse Gaia-only leftovers).
    """
    out = Path(output).expanduser().resolve()
    n_exist = count_phot_sources(out)
    if not force and n_exist >= int(min_sources):
        logger.info(
            'Keeping existing L3 science refcat %s (%d sources)',
            out.name,
            n_exist,
        )
        return out
    if out.is_file() and n_exist < int(min_sources):
        logger.info(
            'Rebuilding sparse L3 science refcat %s (%d < %d sources)',
            out.name,
            n_exist,
            int(min_sources),
        )
    return write_detection_refcat(l3_image, out)


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

    logger.info(
        'Scoring %d L3 product(s) against Gaia '
        '(field cache under reduction/gaia/ when present)',
        len(uniq),
    )
    scores: list[Level3GaiaScore] = []
    for i, path in enumerate(uniq, start=1):
        logger.info('  [%d/%d] Gaia score %s ...', i, len(uniq), path.name)
        scores.append(score_level3_gaia(path, snr=snr))
        logger.info(
            '  [%d/%d] %s -> detectable=%d illuminated=%d fov=%d',
            i,
            len(uniq),
            path.name,
            scores[-1].n_detectable,
            scores[-1].n_illuminated,
            scores[-1].n_gaia_fov,
        )
    # Rank: detectable desc, illuminated desc, prefer non-WFPC2, then FOV
    # count, then redder broadband filters (F814W over F555W) - deeper
    # continuum usually yields tighter Gaia centroids on sparse fields.
    def _filter_rank(filt: str) -> int:
        key = str(filt or '').strip().lower()
        pref = {
            'f814w': 50,
            'f850lp': 48,
            'f775w': 45,
            'f606w': 40,
            'f625w': 38,
            'f555w': 30,
            'f475w': 25,
            'f438w': 20,
            'f336w': 10,
            'f275w': 8,
            'f225w': 5,
        }
        return pref.get(key, 0)

    scores_sorted = sorted(
        scores,
        key=lambda s: (
            s.n_detectable,
            s.n_illuminated,
            0 if s.instrument != 'wfpc2' else -1,
            s.n_gaia_fov,
            _filter_rank(s.filt),
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
    instruments: Sequence[str] | None = None,
) -> list[Path]:
    """
    Build per-filter AstroDrizzle coadds from *raw_dir* into *outdir*.

    Used before JHAT so a Gaia-rich level-3 exists as the absolute reference.
    Skips groups whose coadd already exists unless *force*.

    Parameters
    ----------
    instruments : sequence of str or None, optional
        If set, only drizzle these instruments (e.g. ``ACS``, ``WFC3``).
    """
    from st123.stages.mosaic.hst_drizzle import (
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
    if instruments:
        allow = {str(i).split('_')[0].strip().lower() for i in instruments if str(i).strip()}
        before = len(groups)
        groups = {key: vals for key, vals in groups.items() if key[0] in allow}
        logger.info(
            'Preliminary L3 instrument filter %s: %d -> %d group(s)',
            sorted(allow),
            before,
            len(groups),
        )
    products: list[Path] = []
    for (inst, filt), imgs in sorted(groups.items()):
        suffix = _drizzle_suffix(inst, imgs)
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
