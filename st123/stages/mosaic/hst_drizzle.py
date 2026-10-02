"""
HST AstroDrizzle mosaics from JHAT-aligned frames.

Groups ``*_jhat.fits`` (or calibrated MEFs) by instrument + filter, runs
:func:`drizzlepac.astrodrizzle.AstroDrizzle` per group, and writes coadds into
the shared JWST-style ``reduction/reference/group_*/ref_*`` box layout with a
``dolphot_frames.txt`` manifest per box.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Iterable, Mapping, MutableMapping, Optional, Sequence, Union

from astropy.io import fits
from st123.datamodels import as_datamodel
from st123.datamodels.hst.hst import HST_CALIBRATION_NAME_MARKERS, HSTDataModel
from st123.datamodels.hst.wfc3_ir import WFC3IRDataModel
from st123.datamodels.hst.wfpc2 import WFPC2DataModel

from st123.utils.helpers import get_filter, get_instrument
from st123.utils.logging import capture_output

PathLike = Union[str, os.PathLike]

logger = logging.getLogger(__name__)

_HST_INSTS = frozenset({'acs', 'wfc3', 'wfpc2'})


def mask_wfc3_ir_bad_pixels(
    flt_path: Path,
    *,
    sci_floor: float | None = None,
    bad_grow: int | None = None,
    dq_bit: int | None = None,
) -> dict:
    """
    Flag extreme negative WFC3/IR SCI pixels in DQ before AstroDrizzle.

    Native calwf3/MAST spikes (SCI << 0) with only soft DQ bits can still enter
    the ``driz_cr`` median/blot and poison neighboring frames. Mark pixels
    below *sci_floor* (plus a small morphological grow) with *dq_bit*, which
    must be excluded by ``final_bits=576``. Optionally zero SCI at those
    pixels so they cannot pull the sky/CR stats even if DQ is mishandled.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation

    if sci_floor is None:
        sci_floor = float(WFC3IRDataModel.SCI_FLOOR)
    if bad_grow is None:
        bad_grow = int(WFC3IRDataModel.BAD_GROW_PIX)
    if dq_bit is None:
        dq_bit = int(WFC3IRDataModel.BAD_DQ_BIT)
    if bad_grow < 0:
        raise ValueError('bad_grow must be >= 0')

    path = Path(flt_path)
    n_floor = 0
    n_grown = 0
    n_sci = 0
    with as_datamodel(path).open(mode='update') as hdul:
        for i, hdu in enumerate(hdul):
            if getattr(hdu, 'name', '') != 'SCI' or hdu.data is None:
                continue
            n_sci += 1
            data = np.asarray(hdu.data, dtype=np.float32)
            bad = np.isfinite(data) & (data < float(sci_floor))
            n_floor += int(np.count_nonzero(bad))
            if bad_grow > 0 and np.any(bad):
                grown = binary_dilation(bad, iterations=int(bad_grow))
                n_grown += int(np.count_nonzero(grown & ~bad))
                bad = grown
            if not np.any(bad):
                continue
            # Prefer EXTNAME=DQ at SCI+2 (standard FLT MEF); else search.
            dq_hdu = None
            if i + 2 < len(hdul) and getattr(hdul[i + 2], 'name', '') == 'DQ':
                dq_hdu = hdul[i + 2]
            else:
                for cand in hdul:
                    if getattr(cand, 'name', '') == 'DQ' and cand.data is not None:
                        if cand.data.shape == data.shape:
                            dq_hdu = cand
                            break
            if dq_hdu is None or dq_hdu.data is None:
                logger.warning('No DQ extension for SCI in %s; skip IR floor mask', path.name)
                continue
            dq = np.asarray(dq_hdu.data).astype(np.int64, copy=True)
            dq[bad] |= int(dq_bit)
            dq_hdu.data = dq.astype(dq_hdu.data.dtype, copy=False)
            # Blank extreme negatives so they cannot poison median/blot stats.
            data = data.copy()
            data[bad] = 0.0
            hdu.data = data.astype(hdu.data.dtype, copy=False)
        hdul.flush()

    summary = {
        'path': str(path),
        'n_sci': n_sci,
        'n_floor': n_floor,
        'n_grown': n_grown,
        'sci_floor': float(sci_floor),
        'dq_bit': int(dq_bit),
    }
    if n_floor or n_grown:
        logger.info(
            'WFC3/IR bad-pixel mask %s: floor=%d grown=+%d (SCI<%.1f, DQ|=%d)',
            path.name,
            n_floor,
            n_grown,
            float(sci_floor),
            int(dq_bit),
        )
    return summary


@contextmanager
def _suppress_drizzlepac_dgeo_prompt() -> Iterator[None]:
    """
    Auto-continue DrizzlePac's interactive DGEOFILE/NPOLFILE prompt.

    When ``updatewcs=False``, AstroDrizzle may call ``input()`` asking for
    ``q``/``c``. Always continue (``c``) for non-interactive pipeline runs.
    """
    import drizzlepac.processInput as process_input

    original = process_input.userStop

    def _auto_continue(_message: str) -> bool:
        # True => quit; False => continue without non-polynomial DGEO.
        return False

    process_input.userStop = _auto_continue
    try:
        yield
    finally:
        process_input.userStop = original


def _raw_sibling(path: Path) -> Path | None:
    """Return matching calibrated raw under ``../raw/`` when present."""
    stem = path.name
    for suffix in ('_jhat.fits', '.fits'):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    raw_dir = path.parent.parent / 'raw'
    if not raw_dir.is_dir():
        raw_dir = path.parent / 'raw'
    for cand in (
        raw_dir / f'{stem}_flc.fits',
        raw_dir / f'{stem}_flt.fits',
        raw_dir / f'{stem}_c0m.fits',
    ):
        if cand.is_file():
            return cand
    return None


_WCS_KEYS = (
    'WCSAXES', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
    'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
    'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
    'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
    'CDELT1', 'CDELT2', 'CROTA2',
    'LONPOLE', 'LATPOLE', 'RADESYS', 'EQUINOX',
    'WCSNAME', 'ORIENTAT',
)


def _synthesize_wfpc2_c1m(c0m_path: Path, c1m_path: Path) -> Path:
    """
    Build a minimal WFPC2 ``*_c1m.fits`` DQ MEF matching *c0m* SCI shapes.

    MAST downloads for this field often omit ``c1m``; AstroDrizzle still looks
    for it beside ``c0m``. Zero DQ means "all pixels good".
    """
    import numpy as np

    with as_datamodel(c0m_path).open(memmap=True) as hdul:
        primary = fits.PrimaryHDU(header=hdul[0].header.copy())
        primary.header['FILENAME'] = c1m_path.name
        primary.header['FILETYPE'] = 'SDP'
        hdus: list = [primary]
        for hdu in hdul:
            if getattr(hdu, 'name', '') != 'SCI' or hdu.data is None:
                continue
            # int16 avoids NumPy 2 uint16 bitwise_or casting failures in drizzlepac.
            dq = np.zeros(hdu.data.shape, dtype=np.int16)
            sci = fits.ImageHDU(data=dq, header=hdu.header.copy(), name='SCI')
            sci.header['BITPIX'] = 16
            for key in ('BSCALE', 'BZERO'):
                if key in sci.header:
                    del sci.header[key]
            hdus.append(sci)
        fits.HDUList(hdus).writeto(c1m_path, overwrite=True)
    return c1m_path


def _robust_mad(arr: np.ndarray) -> float:
    """Median absolute deviation -> sigma-equivalent (1.4826xMAD)."""
    import numpy as np

    flat = np.asarray(arr, dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size < 8:
        return float('nan')
    med = float(np.median(flat))
    return float(1.4826 * np.median(np.abs(flat - med)))


def _pixel_noise_mad(arr: np.ndarray) -> float:
    """
    Scene-resistant noise estimate from adjacent-pixel differences.

    Whole-column / whole-row MAD is dominated by stars and galaxy light on
    WFPC2 science chips; differencing suppresses that structure so only true
    high-noise overscan / pyramid-edge sectors are flagged.
    """
    import numpy as np

    flat = np.asarray(arr, dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size < 16:
        return float('nan')
    # Unit Gaussian: mad(diff) ? 1.4826 * sigma * sqrt2 -> sigma ? mad(diff) / sqrt2.
    return float(_robust_mad(np.diff(flat)) / np.sqrt(2.0))


def _high_variance_edge_mask(
    data: np.ndarray,
    *,
    var_edge: int,
    var_sigma: float,
    sci_floor: float,
    left_trim: int = 0,
) -> tuple[np.ndarray, int, int]:
    """
    Mask outer rows/columns whose *noise* exceeds *var_sigma*xinner noise.

    Uses adjacent-pixel MAD (not raw scatter) so real scene structure is not
    mistaken for bad sectors. Left overscan columns (*left_trim*) are excluded
    from row-noise estimates so bleed does not condemn an entire row.

    Returns ``(mask, n_bad_cols, n_bad_rows)``.
    """
    import numpy as np

    ny, nx = data.shape
    mask = np.zeros(data.shape, dtype=bool)
    if var_edge <= 0 or var_sigma <= 0:
        return mask, 0, 0
    band = int(min(var_edge, nx // 4, ny // 4))
    if band < 4:
        return mask, 0, 0

    inner = data[band : ny - band, band : nx - band]
    ok_inner = np.isfinite(inner) & (inner >= float(sci_floor))
    if int(ok_inner.sum()) < 500:
        return mask, 0, 0
    inner_noise = _pixel_noise_mad(inner[ok_inner])
    if not np.isfinite(inner_noise) or inner_noise <= 0:
        return mask, 0, 0
    thr = float(var_sigma) * inner_noise
    skip_left = int(max(0, min(left_trim, nx // 3)))

    n_cols = 0
    # Left edge is the overscan/bleed side - inspect the full band. Right edge
    # is usually clean; only flag the outermost half-band to avoid trimming
    # useful inter-chip coverage.
    left_js = list(range(band))
    right_js = list(range(nx - max(band // 2, 4), nx))
    for j in left_js + right_js:
        col = data[:, j]
        ok = np.isfinite(col) & (col >= float(sci_floor))
        n_ok = int(ok.sum())
        if n_ok < 20:
            # Sparse / mostly-floor columns (classic left overscan).
            mask[:, j] = True
            n_cols += 1
            continue
        noise = _pixel_noise_mad(col[ok])
        if np.isfinite(noise) and noise > thr:
            mask[:, j] = True
            n_cols += 1

    n_rows = 0
    # Same asymmetry for top/bottom: keep the outer half-band on the far side.
    row_is = list(range(band)) + list(range(ny - max(band // 2, 4), ny))
    for i in row_is:
        row = data[i, skip_left:]
        ok = np.isfinite(row) & (row >= float(sci_floor))
        n_ok = int(ok.sum())
        if n_ok < 20:
            mask[i, :] = True
            n_rows += 1
            continue
        noise = _pixel_noise_mad(row[ok])
        if np.isfinite(noise) and noise > thr:
            mask[i, :] = True
            n_rows += 1

    return mask, n_cols, n_rows


def mask_wfpc2_overscan(
    c0m_path: Path,
    c1m_path: Path,
    *,
    edge: int | None = None,
    left_extra: int | None = None,
    dq_bit: int | None = None,
    sci_floor: float | None = None,
    bad_grow: int | None = None,
    bad_col_frac: float | None = None,
    var_edge: int | None = None,
    var_sigma: float | None = None,
) -> dict:
    """
    Flag WFPC2 chip overscan / bad-edge / high-variance sectors in ``c1m``.

    Calibrated ``*_c0m`` chips retain left-edge overscan bleed and noisy
    pyramid-edge strips that project into AstroDrizzle coadds (especially near
    the four-chip intersection). Those pixels are marked with *dq_bit*
    (excluded by ``final_bits=1032``) and SCI below *sci_floor* is set to 0.

    Steps: uniform border (+ extra left width), floor cut, left-half bad
    columns, high-variance outer rows/columns within *var_edge*, then a small
    morphological grow.

    Defaults are read from :class:`~st123.datamodels.hst.wfpc2.WFPC2DataModel`
    class attributes.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation

    if edge is None:
        edge = int(WFPC2DataModel.OVERSCAN_EDGE_PIX)
    if left_extra is None:
        left_extra = int(WFPC2DataModel.OVERSCAN_LEFT_EXTRA)
    if dq_bit is None:
        dq_bit = int(WFPC2DataModel.OVERSCAN_DQ_BIT)
    if sci_floor is None:
        sci_floor = float(WFPC2DataModel.SCI_FLOOR)
    if bad_grow is None:
        bad_grow = int(WFPC2DataModel.BAD_GROW_PIX)
    if bad_col_frac is None:
        bad_col_frac = float(WFPC2DataModel.BAD_COL_FRAC)
    if var_edge is None:
        var_edge = int(WFPC2DataModel.VAR_EDGE_PIX)
    if var_sigma is None:
        var_sigma = float(WFPC2DataModel.VAR_SIGMA)

    if edge < 0 or left_extra < 0 or bad_grow < 0:
        raise ValueError('edge, left_extra, and bad_grow must be >= 0')
    n_edge = 0
    n_floor = 0
    n_cols = 0
    n_var_cols = 0
    n_var_rows = 0
    n_grown = 0
    sci_model = as_datamodel(c0m_path)
    with sci_model.open(mode='update') as sci_hdul, sci_model.open_dq(
        mode='update'
    ) as dq_hdul:
        sci_exts = [
            (i, h)
            for i, h in enumerate(sci_hdul)
            if getattr(h, 'name', '') == 'SCI' and h.data is not None
        ]
        for sci_i, sci_hdu in sci_exts:
            data = np.asarray(sci_hdu.data, dtype=np.float32)
            ny, nx = data.shape
            # Prefer matching SCI index in c1m; else match by order among SCI.
            if sci_i < len(dq_hdul) and dq_hdul[sci_i].data is not None:
                dq_hdu = dq_hdul[sci_i]
            else:
                dq_scis = [
                    h for h in dq_hdul
                    if getattr(h, 'name', '') == 'SCI' and h.data is not None
                ]
                # Pair by SCI order
                order = [i for i, _ in sci_exts].index(sci_i)
                if order >= len(dq_scis):
                    continue
                dq_hdu = dq_scis[order]

            dq = np.asarray(dq_hdu.data).copy()
            if dq.shape != data.shape:
                logger.warning(
                    'WFPC2 overscan mask skip %s ext %d: SCI %s vs DQ %s',
                    c0m_path.name,
                    sci_i,
                    data.shape,
                    dq.shape,
                )
                continue

            border = np.zeros(data.shape, dtype=bool)
            if edge > 0:
                e = min(edge, nx // 2, ny // 2)
                border[:, :e] = True
                border[:, nx - e :] = True
                border[:e, :] = True
                border[ny - e :, :] = True
            left_w = min(edge + left_extra, nx // 2)
            if left_w > 0:
                border[:, :left_w] = True

            floor = np.isfinite(data) & (data < float(sci_floor))
            # Kill left-half columns that are mostly below the SCI floor.
            col_mask = np.zeros(data.shape, dtype=bool)
            if bad_col_frac > 0:
                half = nx // 2
                frac = np.mean(floor[:, :half], axis=0)
                kill = np.where(frac >= float(bad_col_frac))[0]
                for j in kill:
                    col_mask[:, int(j)] = True
                n_cols += int(len(kill))

            var_mask, nv_c, nv_r = _high_variance_edge_mask(
                data,
                var_edge=int(var_edge),
                var_sigma=float(var_sigma),
                sci_floor=float(sci_floor),
                left_trim=int(left_w),
            )
            n_var_cols += int(nv_c)
            n_var_rows += int(nv_r)

            bad = border | floor | col_mask | var_mask
            if bad_grow > 0 and np.any(bad):
                grown = binary_dilation(bad, iterations=int(bad_grow))
                n_grown += int(np.count_nonzero(grown & ~bad))
                bad = grown

            n_edge += int(border.sum())
            n_floor += int(floor.sum())
            if np.any(bad):
                dq[bad] = np.bitwise_or(dq[bad].astype(np.int32), int(dq_bit))
                data = data.copy()
                data[bad] = 0.0
                sci_hdu.data = data.astype(sci_hdu.data.dtype, copy=False)
                dq_hdu.data = dq.astype(dq_hdu.data.dtype, copy=False)
        sci_hdul[0].header['ST123OVS'] = (
            True,
            (
                f'WFPC2 edge/var mask edge={edge} left+={left_extra} '
                f'var_edge={var_edge} floor={sci_floor} grow={bad_grow}'
            ),
        )
        sci_hdul.flush()
        dq_hdul.flush()
    logger.info(
        'WFPC2 edge/var mask %s: edge_pix=%d floor_pix=%d '
        'bad_cols=%d var_cols=%d var_rows=%d grown=%d '
        '(edge=%d left+=%d var_edge=%d bit=%d)',
        c0m_path.name,
        n_edge,
        n_floor,
        n_cols,
        n_var_cols,
        n_var_rows,
        n_grown,
        edge,
        left_extra,
        var_edge,
        dq_bit,
    )
    return {
        'path': str(c0m_path),
        'n_edge': n_edge,
        'n_floor': n_floor,
        'n_cols': n_cols,
        'n_var_cols': n_var_cols,
        'n_var_rows': n_var_rows,
        'n_grown': n_grown,
    }


def _ctx_single_bit_mask(ctx: np.ndarray) -> np.ndarray:
    """
    True where *ctx* has exactly one bit set (single contributing input).

    AstroDrizzle CTX packs one bit per input SCI plane; a power-of-two nonzero
    value means only one chip/frame covered that output pixel.
    """
    import numpy as np

    cu = np.asarray(ctx, dtype=np.uint32)
    return (cu != 0) & ((cu & (cu - np.uint32(1))) == 0)


# Isolated SCI specks (hot pixels drizzled in because final_bits includes
# CALACS bit 16) to inpaint after CTX fill. Stars are larger than this.
HOT_CLIP_MAX_PIX = 4
HOT_CLIP_NSIGMA = 10.0


def _isolated_sci_outliers(
    sci,
    good,
    *,
    max_size: int = HOT_CLIP_MAX_PIX,
    n_sigma: float = HOT_CLIP_NSIGMA,
):
    """
    Isolated 1-N pixel SCI spikes vs a 3x3 median (drizzled-in hot pixels).

    CTX fill only inpaints ``CTX==0`` holes. ACS/UVIS ``final_bits=112``
    lets CALACS hot pixels (bit 16) into SCI, so 4-frame F555W stacks still
    show speckles. Stars are larger than *max_size* ACS/UVIS pixels.
    """
    import numpy as np
    from scipy.ndimage import label, median_filter

    out = np.zeros(np.asarray(sci).shape, dtype=bool)
    g = np.asarray(good, dtype=bool)
    if int(np.count_nonzero(g)) < 50:
        return out
    arr = np.asarray(sci, dtype=np.float32)
    loc = median_filter(np.nan_to_num(arr, nan=0.0), size=3)
    res = arr - loc
    sample = res[g]
    med = float(np.median(sample))
    mad = 1.4826 * float(np.median(np.abs(sample - med)))
    if not np.isfinite(mad) or mad <= 0:
        return out
    hot = g & np.isfinite(res) & (np.abs(res) > float(n_sigma) * mad)
    if not np.any(hot):
        return out
    lab, n = label(hot)
    if n == 0:
        return out
    sizes = np.bincount(lab.ravel())
    keep = (sizes >= 1) & (sizes <= int(max_size))
    keep[0] = False
    return keep[lab]


def _inpaint_from_neighbors(sci: np.ndarray, hole: np.ndarray, valid: np.ndarray):
    """Nearest-neighbor fill of *hole* pixels from *valid* SCI values."""
    import numpy as np
    from scipy import ndimage

    out = np.asarray(sci, dtype=np.float32).copy()
    if not np.any(hole) or not np.any(valid):
        return out
    # Distance to the nearest valid pixel; indices are the donor location.
    inv = np.asarray(~valid, dtype=bool)
    _, (iy, ix) = ndimage.distance_transform_edt(inv, return_indices=True)
    sel = np.asarray(hole, dtype=bool)
    out[sel] = out[iy[sel], ix[sel]]
    return out


def _split_empty_components(
    empty: np.ndarray,
    *,
    max_pix: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split an empty-pixel mask into small interior holes vs exterior / large gaps.

    Small holes (not touching the array border, size ``<= max_pix``) can be
    interpolated for cosmetics. Border-touching or huge components are true
    uncovered footprint.
    """
    import numpy as np
    from scipy import ndimage

    empty_b = np.asarray(empty, dtype=bool)
    small = np.zeros(empty_b.shape, dtype=bool)
    exterior = np.zeros(empty_b.shape, dtype=bool)
    if not np.any(empty_b):
        return small, exterior
    lab, n = ndimage.label(empty_b)
    if n == 0:
        return small, exterior
    sizes = np.bincount(lab.ravel())
    border_labs = np.unique(
        np.concatenate(
            [lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]
        )
    )
    on_border = np.zeros(int(n) + 1, dtype=bool)
    on_border[border_labs[border_labs > 0]] = True
    is_ext = on_border.copy()
    is_ext |= sizes > int(max_pix)
    is_ext[0] = False
    is_small = (sizes > 0) & (sizes <= int(max_pix)) & ~on_border
    is_small[0] = False
    small = is_small[lab]
    exterior = is_ext[lab]
    return small, exterior


def _ensure_l3_dq_hdu(hdul: fits.HDUList, shape: tuple[int, ...]):
    """Return the coadd DQ ImageHDU, creating a zero array when missing."""
    import numpy as np

    for hdu in hdul:
        if str(getattr(hdu, 'name', '') or '').upper() == 'DQ' and hdu.data is not None:
            data = np.asarray(hdu.data)
            if tuple(data.shape) == tuple(shape):
                return hdu
    dq = fits.ImageHDU(
        np.zeros(shape, dtype=np.int16),
        name='DQ',
    )
    dq.header['EXTNAME'] = 'DQ'
    hdul.append(dq)
    return dq


def _product_is_wfpc2(path: Path, hdul: fits.HDUList) -> bool:
    """Detect WFPC2 drizzle products via INSTRUME or filename."""
    inst = str(hdul[0].header.get('INSTRUME') or '').upper()
    if 'WFPC2' in inst:
        return True
    return 'wfpc2' in path.name.lower()


def fill_drizzle_uncovered_with_sky(
    path: PathLike,
    *,
    drop_single_ctx_edge_pix: int | None = None,
) -> dict:
    """
    Cosmetically fill uncovered SCI pixels and flag them in DQ / WHT.

    AstroDrizzle leaves ``CTX==0`` (and often NaN SCI) outside the illuminated
    footprint. Align and mosaic coadds are sanitized so SCI has no NaNs:

    - Large / border-touching empty regions get the illuminated median sky
      (true non-coverage).
    - Small interior holes are nearest-neighbor interpolated from valid SCI
      so the image looks continuous.
    - WFPC2 single-bit CTX pixels near the footprint edge keep their drizzle
      SCI (real detector data) instead of being replaced with sky.

    In all three cases ``WHT=0`` so DOLPHOT ``*mask`` still maps them to DMIN.
    A ``DQ`` extension records why: bit 1 = no data (sky), bit 2 = inpainted
    hole, bit 4 = kept SCI but untrustworthy (single-coverage edge).

    For WFPC2, the single-bit edge cut uses
    :attr:`WFPC2DataModel.DROP_SINGLE_CTX_EDGE_PIX` but only when the mosaic
    already has overlapping coverage. Set the edge width to 0 to disable.

    Returns a summary dict. Idempotent.
    """
    import numpy as np
    from astropy.stats import sigma_clipped_stats
    from scipy import ndimage

    p = Path(path)
    summary: dict = {
        'path': str(p),
        'n_filled': 0,
        'n_inpaint': 0,
        'n_single_ctx': 0,
        'n_hot': 0,
        'sky': None,
        'skipped': False,
        'drop_single_ctx_edge_pix': 0,
    }
    no_data_bit = int(HSTDataModel.L3_DQ_NO_DATA)
    inpaint_bit = int(HSTDataModel.L3_DQ_INPAINT)
    unreliable_bit = int(HSTDataModel.L3_DQ_UNRELIABLE)
    max_inpaint = int(HSTDataModel.L3_INPAINT_MAX_PIX)
    with as_datamodel(p).open(mode='update', memmap=False) as hdul:
        try:
            sci_hdu = hdul['SCI']
        except KeyError:
            summary['skipped'] = True
            summary['reason'] = 'no SCI'
            return summary
        if drop_single_ctx_edge_pix is None:
            if _product_is_wfpc2(p, hdul):
                edge_pix = int(WFPC2DataModel.DROP_SINGLE_CTX_EDGE_PIX)
            else:
                edge_pix = 0
        else:
            edge_pix = int(drop_single_ctx_edge_pix)
        if edge_pix < 0:
            edge_pix = 0
        summary['drop_single_ctx_edge_pix'] = edge_pix

        sci = np.asarray(sci_hdu.data, dtype=np.float32)
        ctx_hdu = None
        wht_hdu = None
        for hdu in hdul:
            name = str(getattr(hdu, 'name', '') or '').upper()
            if name == 'CTX' and hdu.data is not None and ctx_hdu is None:
                ctx_hdu = hdu
            elif name == 'WHT' and hdu.data is not None and wht_hdu is None:
                wht_hdu = hdu
        single_edge = np.zeros(sci.shape, dtype=bool)
        small_holes = np.zeros(sci.shape, dtype=bool)
        exterior = np.zeros(sci.shape, dtype=bool)
        if ctx_hdu is None and wht_hdu is None:
            # Still scrub NaNs using finite SCI alone.
            bad = ~np.isfinite(sci)
            if not np.any(bad):
                summary['skipped'] = True
                summary['reason'] = 'no CTX/WHT and SCI finite'
                return summary
            good = np.isfinite(sci)
            exterior = bad
        else:
            uncovered = np.zeros(sci.shape, dtype=bool)
            ctx = None
            if ctx_hdu is not None:
                ctx = np.asarray(ctx_hdu.data)
                if ctx.shape == sci.shape:
                    uncovered |= ctx == 0
                    if edge_pix > 0:
                        single = _ctx_single_bit_mask(ctx)
                        # Distance to uncovered / empty-weight footprint edge.
                        illuminated = (ctx != 0)
                        n_illum = int(np.count_nonzero(illuminated))
                        n_multi = int(
                            np.count_nonzero(illuminated & ~single)
                        )
                        # No dither overlap: do not treat chip-gap edges as
                        # extra-bad (1-frame WFPC2 pyramids).
                        if (
                            n_illum < 50
                            or (n_multi / float(n_illum)) < CTX_OVERLAP_MIN_FRAC
                        ):
                            edge_pix = 0
                            summary['drop_single_ctx_edge_pix'] = 0
                            single = np.zeros(sci.shape, dtype=bool)
                        if wht_hdu is not None:
                            wht0 = np.asarray(wht_hdu.data, dtype=np.float32)
                            if wht0.shape == sci.shape:
                                illuminated = illuminated & np.isfinite(wht0) & (
                                    wht0 > 0
                                )
                        if np.any(~illuminated) and np.any(single):
                            dist = ndimage.distance_transform_edt(illuminated)
                            single_edge = single & (dist < float(edge_pix))
                else:
                    logger.debug(
                        'CTX shape %s != SCI %s in %s; ignoring CTX',
                        ctx.shape,
                        sci.shape,
                        p.name,
                    )
                    ctx = None
            if wht_hdu is not None:
                wht = np.asarray(wht_hdu.data, dtype=np.float32)
                if wht.shape == sci.shape:
                    uncovered |= ~np.isfinite(wht) | (wht <= 0)
            uncovered |= ~np.isfinite(sci)
            # Single-coverage edges keep SCI; do not treat them as empty holes.
            uncovered = uncovered & ~single_edge
            small_holes, exterior = _split_empty_components(
                uncovered, max_pix=max_inpaint
            )
            if ctx is not None:
                good = (ctx != 0) & ~single_edge & np.isfinite(sci)
                if wht_hdu is not None:
                    wht_g = np.asarray(wht_hdu.data, dtype=np.float32)
                    if wht_g.shape == sci.shape:
                        good = good & np.isfinite(wht_g) & (wht_g > 0)
            else:
                good = ~uncovered & ~single_edge & np.isfinite(sci)

        if int(np.count_nonzero(good)) < 50:
            summary['skipped'] = True
            summary['reason'] = 'too few good pixels'
            return summary
        _, sky, _ = sigma_clipped_stats(sci[good], sigma=3.0, maxiters=5)
        if not np.isfinite(sky):
            sky = float(np.nanmedian(sci[good]))
        if not np.isfinite(sky):
            summary['skipped'] = True
            summary['reason'] = 'sky not finite'
            return summary
        sky_f = float(sky)
        n_single = int(np.count_nonzero(single_edge))
        n_inpaint = int(np.count_nonzero(small_holes))
        n_ext = int(np.count_nonzero(exterior))
        sci = sci.copy()
        if n_ext:
            sci[exterior] = np.float32(sky_f)
        if n_inpaint:
            sci = _inpaint_from_neighbors(sci, small_holes, good)
        still = ~np.isfinite(sci)
        if np.any(still):
            sci[still] = np.float32(sky_f)
            exterior = exterior | still
            n_ext = int(np.count_nonzero(exterior))
        hot = _isolated_sci_outliers(sci, good)
        n_hot = int(np.count_nonzero(hot))
        if n_hot:
            remain = good & ~hot
            sci = _inpaint_from_neighbors(sci, hot, remain)
        sci_hdu.data = sci
        ignore = exterior | small_holes | single_edge | hot
        if wht_hdu is not None and np.asarray(wht_hdu.data).shape == sci.shape:
            wht = np.asarray(wht_hdu.data, dtype=np.float32).copy()
            wht[ignore] = 0.0
            wht_hdu.data = wht
        if ctx_hdu is not None and np.asarray(ctx_hdu.data).shape == sci.shape:
            ctx_out = np.asarray(ctx_hdu.data).copy()
            ctx_out[exterior | small_holes] = 0
            ctx_hdu.data = ctx_out
        dq_hdu = _ensure_l3_dq_hdu(hdul, sci.shape)
        dq = np.asarray(dq_hdu.data, dtype=np.int16).copy()
        if n_ext:
            dq[exterior] = np.int16(dq[exterior] | no_data_bit)
        if n_inpaint:
            dq[small_holes] = np.int16(dq[small_holes] | inpaint_bit)
        if n_hot:
            dq[hot] = np.int16(dq[hot] | inpaint_bit)
        if n_single:
            dq[single_edge] = np.int16(dq[single_edge] | unreliable_bit)
        dq_hdu.data = dq
        prim = hdul[0].header
        if 'ST123SGL' in prim:
            del prim['ST123SGL']
        prim['ST123CTX'] = (
            True,
            (
                f'st123: SCI fill ext={n_ext} inpaint={n_inpaint} '
                f'hot={n_hot} edge_single={n_single} sky={sky_f:.6g}'
            ),
        )
        prim['ST123CSY'] = (
            sky_f,
            'st123: median sky used for large uncovered SCI fill',
        )
        prim['ST123DQ'] = (
            True,
            'st123: L3 DQ bit1=no_data bit2=inpaint bit4=unreliable',
        )
        if n_single > 0:
            prim['ST123SGE'] = (
                int(edge_pix),
                'st123: WFPC2 edge single-bit CTX drop width (pix)',
            )
        if n_hot > 0:
            prim['ST123HOT'] = (
                int(n_hot),
                'st123: isolated SCI hot/cold pixels inpainted',
            )
        hdul.flush()
        summary['n_filled'] = n_ext
        summary['n_inpaint'] = n_inpaint
        summary['n_single_ctx'] = n_single
        summary['n_hot'] = n_hot
        summary['sky'] = sky_f
    logger.info(
        'Filled %s: sky=%d inpaint=%d hot=%d edge_single=%d (edge_pix=%d) sky=%.6g',
        p.name,
        summary['n_filled'],
        summary['n_inpaint'],
        summary['n_hot'],
        summary['n_single_ctx'],
        summary['drop_single_ctx_edge_pix'],
        summary['sky'] if summary['sky'] is not None else float('nan'),
    )
    return summary


def _finalize_drizzle_product(
    product: Path,
    desired: Path,
    *,
    output_stem: Path,
) -> Path:
    """
    Move AstroDrizzle output to *desired* and drop WFPC2 ``_drw`` duplicates.

    DrizzlePac forces ``_drw.fits`` for WFPC2 when given a bare stem; we always
    keep a single ``_drz.fits`` (or ``_drc.fits``) product. Uncovered pixels
    are then filled in SCI (sky or inpaint) and flagged in ``DQ`` / ``WHT=0``
    so photometry ignores them while the image stays visually continuous.
    """
    product = product.resolve()
    desired = desired.resolve()
    if product != desired:
        if desired.exists():
            desired.unlink()
        shutil.move(str(product), str(desired))
        product = desired

    # Remove sibling WFPC2 _drw leftovers and any accidental duplicate stems.
    parent = desired.parent
    stem = output_stem.name
    for leftover in parent.glob(f'{stem}_drw*.fits'):
        try:
            leftover.unlink()
            logger.info('Removed duplicate drizzle product %s', leftover.name)
        except OSError as exc:
            logger.warning('Could not remove %s: %s', leftover, exc)
    # If desired is *_drz.fits, also remove a bare *_drw.fits with same stem.
    if desired.name.endswith('_drz.fits'):
        twin = parent / desired.name.replace('_drz.fits', '_drw.fits')
        if twin.is_file() and twin.resolve() != desired:
            try:
                twin.unlink()
                logger.info('Removed duplicate drizzle product %s', twin.name)
            except OSError as exc:
                logger.warning('Could not remove %s: %s', twin, exc)
    try:
        fill_drizzle_uncovered_with_sky(product)
    except Exception as exc:
        logger.warning(
            'CTX/sky fill failed for %s: %s', product.name, exc
        )
    return product


def _copy_sci_wcs(src_hdul: fits.HDUList, dst_hdul: fits.HDUList) -> None:
    """Copy SCI WCS keywords from *src_hdul* onto matching SCI HDUs in *dst*."""
    src_scis = [h for h in src_hdul if getattr(h, 'name', '') == 'SCI']
    dst_scis = [h for h in dst_hdul if getattr(h, 'name', '') == 'SCI']
    for src_hdu, dst_hdu in zip(src_scis, dst_scis):
        for key in _WCS_KEYS:
            if key in src_hdu.header:
                dst_hdu.header[key] = src_hdu.header[key]
        # Copy residual distortion / SIP-like cards when present.
        for key, val in src_hdu.header.items():
            ku = str(key).upper()
            if ku.startswith(('A_', 'B_', 'AP_', 'BP_', 'IDC', 'D2IM', 'NPOL', 'WSLT')):
                dst_hdu.header[key] = val


def _stage_drizzle_input(
    src: Path,
    scratch_dir: Path,
    *,
    instrument: str,
    filt: str,
    run_cosmic_clean: bool = True,
) -> Path:
    """
    Build a scratch FITS suitable for AstroDrizzle.

    Prefer the calibrated ``raw/`` sibling (keeps ``NGOODPIX``, DQ, etc.) and
    overlay the aligned SCI WCS from the JHAT product when *src* is ``*_jhat``.
    Falls back to repairing headers on a JHAT copy when no raw sibling exists.

    When *run_cosmic_clean* is True, run astroscrappy on the scratch science
    copy and flag CR pixels in DQ / WFPC2 ``c1m`` before AstroDrizzle.
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    inst = instrument.split('_')[0].upper()
    filt_u = filt.upper()
    raw = _raw_sibling(src)
    jhat = src if '_jhat' in src.name.lower() else None
    # When *src* itself is a calibrated raw frame (prelim L3 path), use it.
    if raw is None and any(
        src.name.endswith(suf) for suf in ('_flc.fits', '_flt.fits', '_c0m.fits')
    ):
        raw = src

    if raw is not None:
        dst = scratch_dir / raw.name
        shutil.copy2(raw, dst)
        # WFPC2 AstroDrizzle expects a sibling *_c1m.fits DQ next to *_c0m.fits.
        if raw.name.endswith('_c0m.fits'):
            c1m = raw.with_name(raw.name.replace('_c0m.fits', '_c1m.fits'))
            c1m_dst = scratch_dir / c1m.name
            if c1m.is_file():
                shutil.copy2(c1m, c1m_dst)
            else:
                logger.warning(
                    'WFPC2 DQ missing for %s; synthesizing zero-DQ *_c1m.fits',
                    raw.name,
                )
                _synthesize_wfpc2_c1m(raw, c1m_dst)
        if jhat is not None:
            with as_datamodel(jhat).open(memmap=True) as jhat_hdul, as_datamodel(dst).open(mode='update') as dst_hdul:
                _copy_sci_wcs(jhat_hdul, dst_hdul)
                dst_hdul.flush()
    else:
        dst = scratch_dir / src.name
        shutil.copy2(src, dst)

    if run_cosmic_clean and any(
        dst.name.endswith(suf) for suf in ('_flc.fits', '_flt.fits', '_c0m.fits')
    ):
        from st123.stages.photometry.cosmic import run_cosmic, should_skip_cosmic

        if should_skip_cosmic(dst, instrument=instrument):
            logger.info(
                'Skipping astroscrappy for WFC3/IR %s (false-flags star cores)',
                dst.name,
            )
        else:
            already = False
            try:
                already = bool(as_datamodel(dst).header(0).get('ST123CR'))
            except Exception:
                already = False
            if already:
                logger.info(
                    'Skipping astroscrappy for %s (ST123CR already set)', dst.name
                )
            else:
                try:
                    run_cosmic(
                        dst,
                        instrument=instrument.split('_')[0].lower(),
                        add_crmask=True,
                        inplace=True,
                    )
                except Exception as exc:
                    logger.warning('astroscrappy failed for %s: %s', dst.name, exc)

    # WFC3/IR: mask extreme negative SCI (+ grow) so bad pixels cannot poison
    # median/blot even when driz_cr is disabled.
    try:
        from st123.stages.photometry.cosmic import is_wfc3_ir_frame

        if is_wfc3_ir_frame(dst) or (
            instrument.split('_')[0].upper() == 'WFC3'
            and filt_u in {'F110W', 'F125W', 'F140W', 'F160W'}
        ):
            try:
                mask_wfc3_ir_bad_pixels(dst)
            except Exception as exc:
                logger.warning(
                    'WFC3/IR bad-pixel mask failed for %s: %s', dst.name, exc
                )
    except Exception:
        pass

    # WFPC2: mask overscan / bad-edge strips in c1m and blank extreme negatives.
    if dst.name.endswith('_c0m.fits'):
        c1m_dst = dst.with_name(dst.name.replace('_c0m.fits', '_c1m.fits'))
        if c1m_dst.is_file():
            try:
                mask_wfpc2_overscan(dst, c1m_dst)
            except Exception as exc:
                logger.warning(
                    'WFPC2 overscan mask failed for %s: %s', dst.name, exc
                )

    with as_datamodel(dst).open(mode='update') as hdul:
        prim = hdul[0].header
        if 'INSTRUME' not in prim or not str(prim.get('INSTRUME') or '').strip():
            prim['INSTRUME'] = inst
        if 'FILTER' not in prim or not str(prim.get('FILTER') or '').strip():
            # Avoid writing numeric FILTER1 wheel positions as FILTER.
            existing = prim.get('FILTNAM1')
            if existing and str(existing).strip():
                try:
                    float(existing)
                except (TypeError, ValueError):
                    prim['FILTER'] = str(existing).strip()
                else:
                    prim['FILTER'] = filt_u
            else:
                prim['FILTER'] = filt_u
        if inst == 'WFPC2' and 'FILTNAM1' not in prim:
            prim['FILTNAM1'] = filt_u
        if 'DETECTOR' not in prim:
            # WFPC2 stores chip id on SCI; AstroDrizzle still wants a primary value.
            det = None
            for hdu in hdul:
                if hdu.header.get('DETECTOR') is not None:
                    det = hdu.header['DETECTOR']
                    break
            if det is None and inst == 'WFPC2':
                det = 'WFPC2'
            elif det is None and inst == 'WFC3':
                aper = str(prim.get('APERTURE') or '').upper()
                det = 'IR' if aper.startswith('IR') else 'UVIS'
            if det is not None:
                prim['DETECTOR'] = det
        # Avoid DGEO interactive stop: drop obsolete DGEOFILE without NPOLFILE.
        if 'DGEOFILE' in prim and 'NPOLFILE' not in prim:
            try:
                del prim['DGEOFILE']
            except KeyError:
                pass
        # AstroDrizzle drops inputs with EXPTIME==0. Some ACS FLCs ship with
        # EXPTIME=0 but valid EXPSTART/EXPEND (MJD) - repair before drizzle.
        repaired = _repair_zero_exptime(hdul)
        if repaired is not None:
            logger.warning(
                'Repaired EXPTIME=0 -> %.3fs from EXPSTART/EXPEND on %s',
                repaired,
                dst.name,
            )
        # Ensure SCI extensions have NGOODPIX when missing (rare jhat-only path).
        for hdu in hdul:
            if getattr(hdu, 'name', '') == 'SCI' and hdu.data is not None:
                if 'NGOODPIX' not in hdu.header:
                    import numpy as np

                    data = np.asarray(hdu.data)
                    hdu.header['NGOODPIX'] = int(np.isfinite(data).sum())
        hdul.flush()

    from st123.datamodels import sanitize_science_fits

    try:
        sanitize_science_fits(dst, materialize_headers=True)
    except Exception as exc:
        logger.debug('Sanitize staged drizzle input %s: %s', dst.name, exc)
    return dst


def _repair_zero_exptime(hdul) -> float | None:
    """
    If primary ``EXPTIME`` is missing/<=0, set it from EXPSTART/EXPEND (days->sec).

    Returns the repaired exposure time in seconds, or ``None`` if unchanged.
    """
    prim = hdul[0].header
    try:
        exp = float(prim.get('EXPTIME') or 0.0)
    except (TypeError, ValueError):
        exp = 0.0
    if exp > 0:
        return None
    start = prim.get('EXPSTART')
    end = prim.get('EXPEND')
    if start is None or end is None:
        # Some products only store times on SCI.
        for hdu in hdul:
            if start is None:
                start = hdu.header.get('EXPSTART')
            if end is None:
                end = hdu.header.get('EXPEND')
    try:
        derived = (float(end) - float(start)) * 86400.0
    except (TypeError, ValueError):
        return None
    if not (derived > 0 and derived < 1.0e6):
        return None
    prim['EXPTIME'] = (
        float(derived),
        'st123: repaired from EXPSTART/EXPEND (was 0)',
    )
    if 'TEXPTIME' in prim:
        try:
            if float(prim['TEXPTIME'] or 0) <= 0:
                prim['TEXPTIME'] = float(derived)
        except (TypeError, ValueError):
            prim['TEXPTIME'] = float(derived)
    return float(derived)


def _assert_drizzle_product_nonempty(product: Path) -> None:
    """Raise if AstroDrizzle wrote an empty (NAXIS=0) SCI product."""
    with as_datamodel(product).open(memmap=True) as hdul:
        sci = None
        for hdu in hdul:
            if getattr(hdu, 'name', '') == 'SCI':
                sci = hdu
                break
        if sci is None and len(hdul) > 1:
            sci = hdul[1]
        if sci is None or sci.data is None or getattr(sci.data, 'size', 0) == 0:
            raise RuntimeError(
                f'AstroDrizzle produced empty SCI in {product.name} '
                f'(often EXPTIME=0 on all inputs). Check '
                f'{product.with_name(product.stem + "_astrodrizzle.log").name}'
            )


def group_hst_frames(
    paths: Iterable[PathLike],
) -> dict[tuple[str, str], list[Path]]:
    """
    Group HST frames by ``(instrument, filter)``.

    Instrument and filter are read via :func:`~st123.utils.helpers.get_instrument`
    and :func:`~st123.utils.helpers.get_filter` (``FILTNAM1`` / ``FILTER`` /
    ``FILTER1`` / ``PHOTMODE`` fallbacks for JHAT products).

    Parameters
    ----------
    paths : iterable of path-like
        Science FITS paths (prefer ``*_jhat.fits``).

    Returns
    -------
    dict
        Mapping ``(instrument, filter)`` -> sorted list of :class:`~pathlib.Path`.
        Keys are lowercase (e.g. ``('wfc3', 'f336w')``).
    """
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in paths:
        p = Path(path)
        if not HSTDataModel.matches_science_filename(p):
            logger.warning(
                'Skipping non-science / calibration file %s', p.name
            )
            continue
        try:
            inst = get_instrument(p).split('_')[0].lower()
            filt = get_filter(p).lower()
        except Exception as exc:
            logger.warning('Skipping ungroupable frame %s (%s)', p, exc)
            continue
        if inst not in _HST_INSTS:
            logger.warning('Skipping non-HST instrument %s for %s', inst, p.name)
            continue
        groups[(inst, filt)].append(p.resolve())
    return {key: sorted(vals) for key, vals in sorted(groups.items())}


def _drizzle_suffix(
    instrument: str,
    images: Sequence[PathLike] | None = None,
) -> str:
    """
    Pipeline-style coadd suffix for the final product basename.

    Taken from the instrument datamodel :attr:`L3_EXTENSION`:

    - ACS/WFC and WFC3/UVIS -> ``drc``
    - WFC3/IR -> ``drz``
    - WFPC2 -> ``drz``
    """
    from st123.datamodels.instrument import InstrumentDataModel

    path = images[0] if images else None
    if path is not None:
        try:
            return type(as_datamodel(path)).L3_EXTENSION
        except Exception:
            pass
    det = None
    if path is not None:
        try:
            det = str(as_datamodel(path).detector or '')
        except Exception:
            det = None
    cls = InstrumentDataModel.class_for_l3(instrument, detector=det)
    return cls.L3_EXTENSION


def _default_final_scale(instrument: str) -> float | None:
    """Native-ish plate scale (arcsec/pix); ``None`` lets AstroDrizzle decide."""
    inst = instrument.lower()
    if inst == 'wfpc2':
        return 0.046
    if inst == 'acs':
        return 0.05
    if inst == 'wfc3':
        return 0.04
    return None


def _wcs_orientat_deg(celestial_wcs) -> float:
    """
    Position angle of +Y (degrees E of N) for DrizzlePac ``final_rot``.

    Matches STWCS / DrizzlePac: ``atan2(CD1_2, CD2_2)``.
    """
    import numpy as np

    cd = np.asarray(celestial_wcs.pixel_scale_matrix, dtype=float)
    return float(np.degrees(np.arctan2(cd[0, 1], cd[1, 1])))


def recenter_wcs_on_array(box_wcs):
    """
    Return an equivalent WCS with CRPIX at the array center.

    JWST coadd WCSes often keep a group-level CRPIX outside the stamp array.
    AstroDrizzle + negative CRPIX + ``final_rot`` then warps the footprint.
    Recentering preserves the sky mapping while giving HAP-safe CRPIX/CRVAL.
    """
    from astropy.wcs import WCS

    if getattr(box_wcs, 'pixel_shape', None) is not None:
        nx, ny = int(box_wcs.pixel_shape[0]), int(box_wcs.pixel_shape[1])
    else:
        nx, ny = int(box_wcs._naxis[0]), int(box_wcs._naxis[1])
    # World coords of the geometric array center (0-indexed pixel).
    ra, dec = box_wcs.pixel_to_world_values(
        0.5 * (nx - 1),
        0.5 * (ny - 1),
    )
    out = box_wcs.deepcopy() if hasattr(box_wcs, 'deepcopy') else WCS(box_wcs.to_header())
    # FITS 1-indexed CRPIX at array center (DrizzlePac / HAP convention).
    out.wcs.crpix = [(nx + 1) * 0.5, (ny + 1) * 0.5]
    out.wcs.crval = [float(ra), float(dec)]
    out.pixel_shape = (nx, ny)
    if getattr(out, '_naxis', None) is not None:
        out._naxis = [nx, ny]
    return out


def build_boxed_drizzle_wcs(
    box_wcs,
    pixel_scale_arcsec: float,
):
    """
    Rescale a shared mosaic box WCS to an HST drizzle pixel scale.

    Returns ``(header, wcs)`` covering the same sky footprint as *box_wcs*,
    with CRPIX recentered on the output array for AstroDrizzle.
    """
    from astropy.wcs import WCS

    from st123.stages.mosaic.mosaic import rescale_wcs_to_pixel_scale

    hdr = rescale_wcs_to_pixel_scale(box_wcs, float(pixel_scale_arcsec))
    out = WCS(hdr)
    out.pixel_shape = (int(hdr['NAXIS1']), int(hdr['NAXIS2']))
    out = recenter_wcs_on_array(out)
    from st123.stages.mosaic.mosaic import strip_header_obsgeo, wcs_without_observatory

    out = wcs_without_observatory(out)
    hdr = out.to_header(relax=True)
    strip_header_obsgeo(hdr)
    hdr['NAXIS1'] = int(out.pixel_shape[0])
    hdr['NAXIS2'] = int(out.pixel_shape[1])
    return hdr, out


def _pickle_safe_output_wcs(output_wcs):
    """
    Strip observatory location so ProcessPool unpickle does not obsfix.

    Stamp WCS objects inherit OBSGEO-XYZ from JWST i2d donors.
    ``WCS.to_header()`` then omits L/B/H, and worker reconstruction warns.
    """
    if output_wcs is None:
        return None
    from astropy.wcs import WCS as AstropyWCS

    from st123.stages.mosaic.mosaic import strip_header_obsgeo, wcs_without_observatory

    if isinstance(output_wcs, fits.Header):
        hdr = output_wcs.copy()
        strip_header_obsgeo(hdr)
        return hdr
    if isinstance(output_wcs, AstropyWCS):
        return wcs_without_observatory(output_wcs)
    return output_wcs


def astrodrizzle_wcs_kwargs_from_header(header) -> dict[str, object]:
    """
    Build AstroDrizzle ``final_*`` kwargs that pin the output grid to *header*.

    HAP-style geometry: CRVAL + CRPIX + outnx/outny + scale + rot. CRPIX is
    recentered onto the array so DrizzlePac cannot flip the stamp footprint.
    Do not pass a truncated ``final_refimage`` (DrizzlePac would inherit NAXIS).
    """
    import numpy as np
    from astropy.wcs import WCS

    w = WCS(header)
    if w.pixel_shape is not None:
        nx, ny = int(w.pixel_shape[0]), int(w.pixel_shape[1])
    else:
        nx = int(header['NAXIS1'])
        ny = int(header['NAXIS2'])
    w.pixel_shape = (nx, ny)
    w = recenter_wcs_on_array(w)
    scale = float(np.sqrt(np.abs(np.linalg.det(w.pixel_scale_matrix))) * 3600.0)
    return {
        'final_wcs': True,
        'final_ra': float(w.wcs.crval[0]),
        'final_dec': float(w.wcs.crval[1]),
        'final_crpix1': float(w.wcs.crpix[0]),
        'final_crpix2': float(w.wcs.crpix[1]),
        'final_outnx': nx,
        'final_outny': ny,
        'final_scale': scale,
        'final_rot': _wcs_orientat_deg(w),
    }


def _resolve_astrodrizzle_product(output_stem: Path) -> Path | None:
    """Locate the FITS product AstroDrizzle wrote for *output_stem*."""
    parent = output_stem.parent
    stem = output_stem.name
    candidates = [
        parent / f'{stem}_drc.fits',
        parent / f'{stem}_drz.fits',
        parent / f'{stem}_drw.fits',  # WFPC2 AstroDrizzle default; renamed later
        parent / f'{stem}.fits',
        # When output already includes _drc/_drz
        Path(f'{output_stem}.fits') if not str(output_stem).endswith('.fits') else output_stem,
    ]
    # Prefer explicit _drz/_drc over _drw when both exist.
    for cand in candidates:
        if cand.is_file() and not cand.name.endswith('_drw.fits'):
            return cand.resolve()
    # Also match sci/wht sidecar style when build=False
    for pattern in (f'{stem}_drc_sci.fits', f'{stem}_drz_sci.fits'):
        cand = parent / pattern
        if cand.is_file():
            return cand.resolve()
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    # Glob fallback
    matches = sorted(parent.glob(f'{stem}_dr*.fits'))
    for m in matches:
        if m.name.endswith(('_sci.fits', '_wht.fits', '_ctx.fits', '_drw.fits')):
            continue
        return m.resolve()
    for m in matches:
        if m.name.endswith(('_sci.fits', '_wht.fits', '_ctx.fits')):
            continue
        return m.resolve()
    return None


def _promote_drizzle_products(scratch_dir: Path, dest_dir: Path) -> None:
    """Move AstroDrizzle science products and logs out of *scratch_dir*.

    Calibration intermediates (``*staticMask*``, blot, median, CR masks) stay
    in scratch and are deleted with the directory. They are not science frames.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for item in scratch_dir.iterdir():
        if not item.is_file():
            continue
        name = item.name.lower()
        if any(marker in name for marker in HST_CALIBRATION_NAME_MARKERS):
            continue
        if item.suffix.lower() not in {'.fits', '.log'}:
            continue
        shutil.move(str(item), dest_dir / item.name)


def _write_group_frame_list(
    outdir: Path,
    *,
    refimage: Path,
    frames: Sequence[Path],
    instrument: str,
    filt: str,
    group: int | None = None,
    box: int | str | None = None,
    merge: bool = True,
) -> Path:
    """
    Write ``dolphot_frames.txt`` (and a filter-tagged copy) for one coadd.

    When *merge* is True and a primary manifest already exists (e.g. JWST wrote
    it first), keep existing frame paths and update the ``# ref`` line.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    existing_frames: list[str] = []
    primary = outdir / 'dolphot_frames.txt'
    if merge and primary.is_file():
        try:
            for line in primary.read_text(encoding='utf-8').splitlines():
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                existing_frames.append(str(Path(s).resolve()))
        except OSError:
            existing_frames = []

    frame_paths = sorted(
        {
            str(Path(p).resolve())
            for p in list(existing_frames) + [str(p) for p in frames]
        }
    )
    # Prefer a JWST i2d as the shared # ref when HST remosaics into a stamp that
    # already has NIRCam/MIRI coadds (nircammask cannot mask ACS/WFC3 drc/drz).
    ref_out = Path(refimage).resolve()
    if merge and primary.is_file():
        try:
            for line in primary.read_text(encoding='utf-8').splitlines():
                s = line.strip()
                if s.startswith('# ref '):
                    prev = Path(s[len('# ref ') :].strip())
                    if prev.is_file() and prev.name.lower().endswith('_i2d.fits'):
                        ref_out = prev.resolve()
                    break
        except OSError:
            pass
    if not str(ref_out).lower().endswith('_i2d.fits'):
        i2ds = sorted(outdir.glob('coadd_*_i2d.fits'))
        if i2ds:

            def _rank(p: Path) -> tuple[int, str]:
                name = p.name.lower()
                for i, key in enumerate(
                    ('f150w2', 'f200w', 'f150w', 'f115w', 'f090w', 'f070w')
                ):
                    if f'_{key}_' in name:
                        return (i, name)
                return (99, name)

            i2ds.sort(key=_rank)
            ref_out = i2ds[0].resolve()
    header: list[str] = []
    if group is not None and box is not None:
        header.append(f'# group={int(group)} box={box}\n')
    header.append(f'# instrument={instrument} filter={filt}\n')
    header.append(f'# ref {ref_out}\n')
    body_lines = header + [f'{p}\n' for p in frame_paths]
    tagged = outdir / f'dolphot_frames_{instrument}_{filt}.txt'
    text = ''.join(body_lines)
    tagged.write_text(text)
    primary.write_text(text)
    return primary


def _measure_extension_sky(
    data,
    dq=None,
    *,
    good_dq_bits: int = 0,
) -> float | None:
    """Robust sky (sigma-clipped median) for one SCI array; ``None`` if empty."""
    import numpy as np
    from astropy.stats import sigma_clipped_stats

    arr = np.asarray(data, dtype=float)
    if arr.ndim > 2:
        arr = arr[0]
    mask = np.isfinite(arr)
    if dq is not None:
        dqarr = np.asarray(dq)
        if dqarr.ndim > 2:
            dqarr = dqarr[0]
        # Keep only pixels with no bad DQ bits (and optionally allow good_dq_bits).
        bad = (dqarr.astype(np.int64) & ~int(good_dq_bits)) != 0
        mask = mask & ~bad
    if int(np.count_nonzero(mask)) < 500:
        return None
    _, med, _ = sigma_clipped_stats(arr[mask], sigma=3.0, maxiters=5)
    return float(med)


def subtract_per_chip_sky(
    path: PathLike,
    *,
    comment: str = 'st123: per-chip sky for UVIS/ACS gap fill',
) -> dict:
    """
    Subtract an independent sky from every SCI extension in *path*.

    AstroDrizzle ``localmin`` / ``globalmin`` apply one sky to all chips in an
    exposure, which leaves UVIS/ACS chip pedestals. With only a few dithers
    those pedestals appear as background jumps along the chip-gap fill.
    Subtracting each chip's own robust sky removes the jump; gap sectors then
    only show higher variance from single-frame coverage.
    """
    import numpy as np

    p = Path(path)
    report: dict = {'path': str(p), 'chips': [], 'n_subtracted': 0}
    with as_datamodel(p).open(mode='update', memmap=False) as hdul:
        sci_idxs = [
            i
            for i, h in enumerate(hdul)
            if getattr(h, 'name', '') == 'SCI' and h.data is not None
        ]
        if not sci_idxs:
            sci_idxs = [
                i
                for i, h in enumerate(hdul)
                if h.data is not None and getattr(h.data, 'ndim', 0) >= 2
            ]
        for i in sci_idxs:
            hdu = hdul[i]
            extver = hdu.header.get('EXTVER', i)
            dq = None
            for h2 in hdul:
                if (
                    getattr(h2, 'name', '') == 'DQ'
                    and h2.header.get('EXTVER', None) == extver
                    and h2.data is not None
                ):
                    dq = h2.data
                    break
            sky = _measure_extension_sky(hdu.data, dq)
            row = {'extver': extver, 'sky': sky}
            if sky is None or not np.isfinite(sky):
                report['chips'].append(row)
                continue
            hdu.data = np.asarray(hdu.data, dtype=float) - float(sky)
            # AstroDrizzle still reads MDRIZSKY when skysub runs; keep at 0.
            hdu.header['MDRIZSKY'] = (0.0, comment)
            hdu.header['ST123SKY'] = (float(sky), comment)
            row['subtracted'] = True
            report['chips'].append(row)
            report['n_subtracted'] += 1
        if report['n_subtracted'] and len(hdul):
            hdul[0].header['ST123PSK'] = (
                True,
                'st123: per-chip sky subtracted before drizzle',
            )
        hdul.flush()
    return report


# 1-2 frame stacks cannot fill DQ holes from a second dither.
THIN_STACK_NMAX = 2
# Mixed-epoch / deep stacks: minmed + driz_cr treats transients as CRs.
CROWDED_STACK_NMIN = 10
# Skip the WFPC2 single-CTX edge cut unless this fraction of illuminated
# pixels already has multi-bit (overlapping) coverage.
CTX_OVERLAP_MIN_FRAC = 0.05


def optical_drizzle_masking_policy(
    n_frames: int, *, is_ir: bool
) -> dict[str, object]:
    """
    Choose cosmic / driz_cr / static / combine knobs for an HST drizzle.

    Thin stacks (1-2 frames) cannot replace rejected pixels, so astroscrappy,
    AstroDrizzle CR rejection, and the static mask are disabled. Stacks of
    10+ frames use median combine without ``driz_cr`` so a transient (SN)
    is not flagged as a cosmic ray. Optical ``static=False`` always: the
    4-sigma-below-mode static mask flags galaxy structure as defects.
    """
    n = int(n_frames)
    if is_ir:
        return {
            'run_cosmic': False,
            'driz_cr': False,
            'static': None,
            'combine_type': 'median' if n < 4 else 'minmed',
        }
    thin = n <= THIN_STACK_NMAX
    crowded = n >= CROWDED_STACK_NMIN
    skip_reject = thin or crowded
    if n < 4 or crowded:
        combine = 'median'
    else:
        combine = 'minmed'
    return {
        'run_cosmic': not skip_reject,
        'driz_cr': False if skip_reject else True,
        'static': False,
        'combine_type': combine,
    }


def optical_drizzle_bits(
    n_frames: int, *, base_bits: int, is_ir: bool
) -> int:
    """
    AstroDrizzle ``final_bits`` for an optical stack.

    ACS/UVIS ``DRIZ_BITS=112`` includes hot pixels (bit 16). Those spikes
    survive into 1-2 frame SCI (SN 2014cx F555W). Drop bit 16 for every
    optical stack: 3+ frames fill from other dithers; thin stacks leave
    CTX==0 holes that ``fill_drizzle_uncovered_with_sky`` inpaints.
    IR bits are unchanged.
    """
    bits = int(base_bits)
    if is_ir:
        return bits
    bits &= ~int(HSTDataModel.HOT_DQ_BIT)
    return bits


def astrodrizzle_cr_flags(
    *,
    is_ir: bool,
    driz_cr: bool | None = None,
    static: bool | None = None,
) -> dict[str, object]:
    """
    AstroDrizzle CR / static-mask flags.

    UVIS/ACS default: ``driz_cr=True``, ``static=False``. IR default:
    :attr:`WFC3IRDataModel.DRIZ_CR` (False) and ``static`` unset. Pass
    *driz_cr* / *static* to override. ``static=None`` leaves DrizzlePac's
    default unset so callers can omit the keyword.
    """
    if driz_cr is None:
        do_cr = bool(WFC3IRDataModel.DRIZ_CR) if is_ir else True
    else:
        do_cr = bool(driz_cr)
    if static is None and not is_ir:
        static = False
    out: dict[str, object] = {
        'driz_cr': do_cr,
        'driz_cr_corr': do_cr,
    }
    if static is not None:
        out['static'] = bool(static)
    return out


def drizzle_filter_group(
    images: Sequence[PathLike],
    output_path: PathLike,
    *,
    final_pixfrac: float = 0.8,
    final_scale: Optional[float] = None,
    num_cores: int = 4,
    instrument: Optional[str] = None,
    clean: bool = True,
    build: bool = True,
    per_chip_sky: bool | None = None,
    output_wcs=None,
    driz_cr: bool | None = None,
    static: bool | None = None,
) -> Path:
    """
    Drizzle one instrument/filter group with AstroDrizzle.

    Parameters
    ----------
    images : sequence of path-like
        Input FITS paths (prefer aligned ``*_jhat.fits`` MEFs).
    output_path : path-like
        Desired coadd path, e.g. ``.../coadd_wfc3_f336w_drc.fits``. The
        ``_drc`` / ``_drz`` suffix is stripped before calling AstroDrizzle.
    final_pixfrac : float, optional
        AstroDrizzle ``final_pixfrac`` (default 0.8).
    final_scale : float or None, optional
        Output pixel scale in arcsec. ``None`` uses an instrument default.
        Ignored when *output_wcs* sets the grid (scale is taken from that WCS).
    num_cores : int, optional
        AstroDrizzle ``num_cores``.
    instrument : str or None, optional
        Instrument name for scale / product suffix; inferred from the first
        frame when omitted.
    clean : bool, optional
        AstroDrizzle ``clean`` flag.
    build : bool, optional
        AstroDrizzle ``build`` (single multi-extension product when True).
    per_chip_sky : bool or None, optional
        If True, subtract an independent sky from each SCI chip before
        drizzle (avoids UVIS/ACS chip-gap background jumps). ``None``
        enables this automatically for WFC3/UVIS and ACS.
    output_wcs : astropy.wcs.WCS or fits.Header or None, optional
        Shared mosaic-box sky grid. When set, AstroDrizzle is forced onto this
        footprint (same stamp as JWST boxed coadds) at the HST pixel scale
        encoded in the WCS/header.
    driz_cr : bool or None, optional
        Override AstroDrizzle ``driz_cr`` / ``driz_cr_corr``. ``None`` uses
        :func:`optical_drizzle_masking_policy` (off for 1-2 frames, off for
        10+ frames, on for UVIS/ACS/WFPC2 in between; always off for IR).
    static : bool or None, optional
        Override AstroDrizzle ``static``. ``None`` disables the static mask
        for optical data (it flags galaxy structure as defects).

    Returns
    -------
    pathlib.Path
        Path to the written drizzle product (renamed to *output_path* when needed).
    """
    from drizzlepac import astrodrizzle

    src_imgs = [Path(p).resolve() for p in images]
    if not src_imgs:
        raise ValueError('drizzle_filter_group requires at least one input image')

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    inst = (instrument or get_instrument(src_imgs[0])).split('_')[0].lower()
    try:
        filt = get_filter(src_imgs[0])
    except Exception:
        filt = 'unknown'
    suffix = _drizzle_suffix(inst, src_imgs)
    # AstroDrizzle appends _drc/_drz; pass a bare stem.
    name = out.name
    for end in ('_drc.fits', '_drz.fits', '.fits'):
        if name.endswith(end):
            name = name[: -len(end)]
            break
    output_stem = out.parent / name

    scale = final_scale
    if scale is None:
        scale = _default_final_scale(inst)

    forced_wcs_kwargs: dict[str, object] | None = None
    if output_wcs is not None:
        from astropy.wcs import WCS as AstropyWCS

        pixscale = float(scale or _default_final_scale(inst) or 0.05)
        if isinstance(output_wcs, fits.Header):
            from st123.stages.mosaic.mosaic import strip_header_obsgeo

            wcs_hdr = output_wcs.copy()
            strip_header_obsgeo(wcs_hdr)
        elif isinstance(output_wcs, AstropyWCS):
            from st123.stages.mosaic.mosaic import wcs_without_observatory

            clean_wcs = wcs_without_observatory(output_wcs)
            wcs_hdr, _ = build_boxed_drizzle_wcs(clean_wcs, pixscale)
        else:
            raise TypeError(
                f'output_wcs must be WCS or Header, got {type(output_wcs)!r}'
            )
        forced_wcs_kwargs = astrodrizzle_wcs_kwargs_from_header(wcs_hdr)
        scale = float(forced_wcs_kwargs['final_scale'])

    dd = dict(HSTDataModel.DRIZZLE_DEFAULTS)
    # Instrument-specific good-DQ bits (hst123 detector_defaults).
    bits_key = inst
    is_uvis = False
    if inst == 'wfc3':
        # Prefer IR bits when any input looks like WFC3/IR.
        bits_key = 'wfc3'
        is_uvis = True
        for p in src_imgs:
            try:
                aper = str(as_datamodel(p).header(0).get('APERTURE') or '').upper()
                if aper.startswith('IR'):
                    bits_key = 'wfc3_ir'
                    is_uvis = False
                    break
            except Exception:
                pass
    elif inst == 'acs':
        is_uvis = True  # two-chip WFC; same pedestal/gap issue
    driz_bits = int(HSTDataModel.drizzle_bits(bits_key) or HSTDataModel.drizzle_bits(inst))
    is_ir = bits_key == 'wfc3_ir'
    policy = optical_drizzle_masking_policy(len(src_imgs), is_ir=is_ir)
    driz_bits = optical_drizzle_bits(
        len(src_imgs), base_bits=driz_bits, is_ir=is_ir
    )
    if driz_cr is None:
        driz_cr = policy['driz_cr']
    if static is None:
        static = policy['static']

    scratch_dir = out.parent / f'.drizzle_scratch_{output_stem.name}'
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir, ignore_errors=True)
    staged = [
        str(
            _stage_drizzle_input(
                p,
                scratch_dir,
                instrument=inst,
                filt=filt,
                run_cosmic_clean=bool(policy['run_cosmic']),
            )
        )
        for p in src_imgs
    ]

    do_per_chip = bool(is_uvis) if per_chip_sky is None else bool(per_chip_sky)
    if do_per_chip:
        for sp in staged:
            rep = subtract_per_chip_sky(sp)
            logger.info(
                'Per-chip sky subtracted for %s (%d SCI): %s',
                Path(sp).name,
                rep.get('n_subtracted', 0),
                [
                    f"SCI{c.get('extver')}={c.get('sky'):.4f}"
                    if c.get('sky') is not None
                    else f"SCI{c.get('extver')}=None"
                    for c in rep.get('chips') or []
                ],
            )

    # Per-chip subtract zeros ACS/UVIS pedestals. Re-enable AstroDrizzle
    # skysub afterward with ``match`` so multi-epoch residual backgrounds
    # equalize; a single exposure-level delta preserves chip balance.
    # IR / single-chip paths keep the default globalmin+match skymethod.
    if do_per_chip:
        sky_method = 'match'
        do_skysub = True
    else:
        sky_method = 'globalmin+match'
        do_skysub = True

    # calwf3 already CR-flags IR (bit 512 in final_bits). AstroDrizzle driz_cr
    # false-rejects undersampled stars when one frame has a negative spike.
    cr_flags = astrodrizzle_cr_flags(
        is_ir=is_ir,
        driz_cr=driz_cr,
        static=static,
    )
    do_driz_cr = bool(cr_flags['driz_cr'])

    kwargs: MutableMapping[str, object] = {
        'output': str(output_stem),
        'runfile': str(output_stem) + '_astrodrizzle.log',
        'context': True,
        'build': build,
        'num_cores': int(num_cores or dd.get('num_cores', 4)),
        'preserve': False,
        'clean': clean,
        'skysub': do_skysub,
        'skymethod': sky_method,
        'skystat': 'mode',
        'updatewcs': False,
        'driz_sep_pixfrac': float(dd.get('driz_sep_pixfrac', final_pixfrac)),
        'combine_maskpt': float(dd.get('combine_maskpt', 0.2)),
        'combine_type': str(policy['combine_type']),
        'combine_nsigma': str(dd.get('combine_nsigma', '4 3')),
        'driz_cr': do_driz_cr,
        'driz_cr_corr': do_driz_cr,
        'driz_cr_snr': str(dd.get('driz_cr_snr', '3.5 3.0')),
        'driz_cr_grow': int(dd.get('driz_cr_grow', 1)),
        'driz_cr_scale': str(dd.get('driz_cr_scale', '1.2 0.7')),
        'final_pixfrac': float(final_pixfrac),
        'final_wcs': True,
        'final_units': 'cps',
        'driz_sep_bits': driz_bits,
        'final_bits': driz_bits,
        # Avoid NumPy 2 uint16 overflow in drizzlepac resetbits path.
        'resetbits': 0,
    }
    if 'static' in cr_flags:
        kwargs['static'] = cr_flags['static']
    if is_ir:
        logger.info(
            'WFC3/IR drizzle: bits=%d driz_cr=%s (sci_floor=%.1f)',
            driz_bits,
            do_driz_cr,
            float(WFC3IRDataModel.SCI_FLOOR),
        )
    if forced_wcs_kwargs is not None:
        kwargs.update(forced_wcs_kwargs)
        logger.info(
            'Forcing AstroDrizzle onto shared box WCS '
            '(outnx=%s outny=%s scale=%.4f" rot=%.3f deg)',
            kwargs.get('final_outnx'),
            kwargs.get('final_outny'),
            kwargs.get('final_scale'),
            kwargs.get('final_rot'),
        )
    elif scale is not None:
        kwargs['final_scale'] = float(scale)

    # DrizzlePac builds outroot via ``'_'.join(output.split('_')[:-1]).lower()``.
    # For ``coadd_0_0_wfpc2_f336w_drz.fits`` that collapses to ``coadd_0_0_wfpc2``,
    # so every WFPC2 (and likewise ACS/WFC3) filter in the same directory shares
    # ``*_staticMask.fits`` names. Parallel filter drizzle then races: one job
    # writes/cleans the mask, the next opens a truncated file as if it were
    # science. Isolate each job in its own scratch cwd so masks stay private.
    #
    # Pass an explicit ``*.fits`` output name so WFPC2 does not create a
    # parallel ``*_drw.fits`` product (DrizzlePac only forces ``_drw`` when the
    # output stem has no ``.fits`` suffix).
    output_base = output_stem.name
    desired = out.parent / f'{output_base}_{suffix}.fits'
    if out.suffix == '.fits' and (
        '_drc' in out.name or '_drz' in out.name or '_drw' in out.name
    ):
        desired = out if not out.name.endswith('_drw.fits') else (
            out.parent / out.name.replace('_drw.fits', '_drz.fits')
        )
    input_rel = [Path(p).name for p in staged]
    kwargs['output'] = desired.name
    kwargs['runfile'] = f'{output_base}_astrodrizzle.log'

    logger.info(
        'AstroDrizzle %d frame(s) -> %s (pixfrac=%.2f scale=%s cores=%d '
        'bits=%d driz_cr=%s static=%s cosmic=%s combine=%s)',
        len(staged),
        desired.name,
        final_pixfrac,
        scale,
        kwargs['num_cores'],
        driz_bits,
        do_driz_cr,
        cr_flags.get('static'),
        policy['run_cosmic'],
        policy['combine_type'],
    )
    cwd = Path.cwd()
    try:
        os.chdir(scratch_dir)
        with _suppress_drizzlepac_dgeo_prompt(), capture_output():
            astrodrizzle.AstroDrizzle(input_rel, **kwargs)
        _promote_drizzle_products(scratch_dir, out.parent)
    finally:
        os.chdir(cwd)
        shutil.rmtree(scratch_dir, ignore_errors=True)

    product = _resolve_astrodrizzle_product(output_stem)
    if product is None and desired.is_file():
        product = desired
    if product is None:
        raise FileNotFoundError(
            f'AstroDrizzle finished but no product found for stem {output_stem}'
        )
    _assert_drizzle_product_nonempty(product)

    return _finalize_drizzle_product(
        product, desired, output_stem=output_stem
    )


def _header_exptime(path: Path) -> float:
    """EXPTIME with EXPSTART/EXPEND fallback (seconds)."""
    try:
        with as_datamodel(path).open(memmap=True) as hdul:
            for hdu in hdul:
                try:
                    exp = float(hdu.header.get('EXPTIME') or 0.0)
                except (TypeError, ValueError):
                    exp = 0.0
                if exp > 0:
                    return exp
            prim = hdul[0].header
            start, end = prim.get('EXPSTART'), prim.get('EXPEND')
            if start is not None and end is not None:
                derived = (float(end) - float(start)) * 86400.0
                if derived > 0:
                    return derived
    except Exception:
        return 0.0
    return 0.0


def _total_exptime(paths: Sequence[Path]) -> float:
    return float(sum(_header_exptime(p) for p in paths))


def _pick_coadd_abs_ref(
    coadds: Sequence[Path],
    *,
    preferred: Sequence[PathLike] | None = None,
    n_frames: MutableMapping[Any, int] | None = None,
) -> Path | None:
    """
    Prefer *preferred* (e.g. in-box JWST i2d), else deepest usable coadd.

    Filter color still matters (F625/F814), but thin stacks (``n_frames < 3``)
    are demoted so a 2-frame F814W does not become the absolute reference over a
    deeper F555W/F606W coadd in the same box.
    """
    for cand in preferred or []:
        p = Path(cand)
        try:
            if p.is_file() and p.stat().st_size > 500_000:
                return p.resolve()
        except OSError:
            continue

    existing = [Path(p) for p in coadds if Path(p).is_file()]
    if not existing:
        return None

    n_map: dict[str, int] = {}
    if n_frames:
        for key, val in n_frames.items():
            try:
                n_map[str(Path(key).resolve())] = int(val)
            except (TypeError, ValueError, OSError):
                continue

    def score(p: Path) -> tuple[int, int, int]:
        name = p.name.lower()
        pref = 0
        if '_i2d' in name and ('f150' in name or 'f200' in name):
            pref = 450
        elif 'wfc3' in name and 'f625' in name:
            pref = 400
        elif 'wfc3' in name and 'f814' in name:
            pref = 350
        elif 'acs' in name and 'f814' in name:
            pref = 320
        elif 'wfpc2' in name and 'f814' in name:
            pref = 200
        elif 'wfc3' in name and 'f555' in name:
            pref = 180
        elif 'acs' in name and 'f606' in name:
            pref = 160
        elif 'wfc3' in name:
            pref = 100
        elif 'f814' in name:
            pref = 90
        try:
            size = int(p.stat().st_size)
        except OSError:
            size = 0
        # Prefer real image products over empty header-only shells (~80 kB).
        if size < 500_000:
            pref -= 500
        n = n_map.get(str(p.resolve()), 0)
        if n <= 0:
            n = 1
        # Depth beats filter color when one coadd is a thin stack.
        if n < 3:
            pref -= 250
        elif n >= 4:
            pref += 80
        depth = min(int(n), 10) * 25
        return (pref + depth, int(n), size)

    return max(existing, key=score)


def apply_sky_shift_to_fits(
    path: PathLike,
    dra_deg: float,
    ddec_deg: float,
    *,
    comment: str = 'st123: L3-unified abs shift',
) -> int:
    """Add (DeltaRA, DeltaDec) degrees to every SCI CRVAL in *path*. Returns n updated."""
    from st123.stages.alignment.hst_jhat import apply_sky_translation_to_sci, _sci_hdu_indices

    p = Path(path)
    with as_datamodel(p).open(mode='update', memmap=False) as hdul:
        idxs = _sci_hdu_indices(hdul)
        if not idxs:
            # Single-extension coadd / primary science.
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and getattr(hdu.data, 'ndim', 0) >= 2:
                    if 'CRVAL1' in hdu.header and 'CRVAL2' in hdu.header:
                        idxs = [i]
                        break
        n = apply_sky_translation_to_sci(
            hdul, float(dra_deg), float(ddec_deg), sci_indices=idxs, comment=comment
        )
        if n and len(hdul):
            import numpy as np

            dec0 = float(hdul[idxs[0]].header.get('CRVAL2', 0.0)) if idxs else 0.0
            hdul[0].header['ST123LUN'] = (
                True,
                'st123: L3-unified absolute shift applied',
            )
            hdul[0].header['ST123LRA'] = (
                float(dra_deg) * 3600.0 * float(np.cos(np.radians(dec0))),
                '[arcsec] L3-unify dRA cos(Dec)',
            )
            hdul[0].header['ST123LDE'] = (
                float(ddec_deg) * 3600.0,
                '[arcsec] L3-unify dDec',
            )
        hdul.flush()
    return int(n)


def unify_hst_astrometric_frame(
    group_records: Sequence[dict],
    *,
    outdir: PathLike,
    max_residual_arcsec: float | None = None,
    max_search_arcsec: float | None = None,
    max_iters: int = 2,
    num_cores: int = 4,
    final_pixfrac: float = 0.8,
    final_scale: Optional[float] = None,
    remosaic: bool = True,
    preferred_abs_ref: PathLike | Sequence[PathLike] | None = None,
    output_wcs=None,
) -> dict:
    """
    Mandatory post-drizzle pass: put all L3 coadds (and their L2 inputs) on one sky frame.

    For each coadd, measure the 2-D histogram offset vs the absolute-reference
    coadd (WFC3 F625 preferred, else WFPC2 F814). When |Delta| exceeds the L3 QA
    limit, apply -Delta to every L2 frame in that group **and** to the coadd WCS,
    then optionally remosaic the group so the drizzle product matches the L2
    frame used by DOLPHOT.

    Iterates up to *max_iters* times. Returns a report dict with ``ok``,
    ``abs_ref``, ``iterations``, and per-group corrections.
    """
    import json


    from st123.stages.alignment.hst_jhat import (
        FRAME_CATASTROPHIC_RECOVER_MAX_ARCSEC,
        FRAME_L3_IMAGE_OFFSET_TOL_ARCSEC,
        HST_ABS_OFFSET_MAX_ARCSEC,
        HST_L3_ALIGN_MAX_ARCSEC,
        measure_hst_sky_offset_2dhist,
        measure_hst_sky_offset_smooth_peak,
        validate_hst_coadds_alignment,
        validate_hst_group_internal_alignment,
        HST_INTERNAL_ALIGN_MAX_ARCSEC,
    )

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    limit = float(
        HST_L3_ALIGN_MAX_ARCSEC if max_residual_arcsec is None else max_residual_arcsec
    )
    search = float(
        HST_ABS_OFFSET_MAX_ARCSEC if max_search_arcsec is None else max_search_arcsec
    )

    # Mutate caller records in place so remosaic output paths propagate.
    records = [r for r in group_records if r.get('status') == 'ok' and r.get('output')]
    report: dict = {
        'ok': False,
        'abs_ref': None,
        'max_residual_arcsec': limit,
        'max_search_arcsec': search,
        'iterations': 0,
        'groups': [],
        'final_qa': None,
    }
    pref_list: list[PathLike] = []
    if preferred_abs_ref is not None:
        if isinstance(preferred_abs_ref, (str, Path, os.PathLike)):
            pref_list = [preferred_abs_ref]
        else:
            pref_list = list(preferred_abs_ref)

    if len(records) < 1:
        report['ok'] = True
        report['final_qa'] = {'ok': True, 'n_coadds': 0, 'max_abs_arcsec': 0.0}
        return report
    if len(records) < 2 and not pref_list:
        report['ok'] = True
        report['final_qa'] = {'ok': True, 'n_coadds': len(records), 'max_abs_arcsec': 0.0}
        return report

    coadd_n_frames = {
        Path(r['output']).resolve(): len(r.get('frames') or [])
        for r in records
        if r.get('output')
    }
    abs_ref = _pick_coadd_abs_ref(
        [Path(r['output']) for r in records],
        preferred=pref_list,
        n_frames=coadd_n_frames,
    )
    if abs_ref is None:
        report['error'] = 'no coadd abs_ref available'
        return report
    report['abs_ref'] = str(abs_ref)
    report['abs_ref_n_frames'] = coadd_n_frames.get(abs_ref.resolve())
    logger.info(
        'L3/L2 astrometric unify: abs_ref=%s (n_frames=%s; limit %.3f", search %.1f")',
        abs_ref.name,
        report['abs_ref_n_frames'],
        limit,
        search,
    )

    for it in range(int(max_iters)):
        report['iterations'] = it + 1
        iter_rows: list[dict] = []
        any_shift = False

        for rec in records:
            coadd = Path(rec['output'])
            frames = [Path(p) for p in rec.get('frames') or []]
            row: dict = {
                'instrument': rec.get('instrument'),
                'filter': rec.get('filter'),
                'coadd': str(coadd),
                'n_frames': len(frames),
                'is_abs_ref': coadd.resolve() == abs_ref.resolve(),
                'applied': False,
            }
            if row['is_abs_ref'] or not coadd.is_file():
                iter_rows.append(row)
                continue

            peak = measure_hst_sky_offset_smooth_peak(
                coadd,
                abs_ref,
                max_offset_arcsec=float(FRAME_CATASTROPHIC_RECOVER_MAX_ARCSEC),
            )
            off = {'ok': False}
            if (
                peak.get('ok')
                and float(peak.get('abs_arcsec') or 0.0)
                > float(FRAME_L3_IMAGE_OFFSET_TOL_ARCSEC)
            ):
                off = peak
            else:
                off = measure_hst_sky_offset_2dhist(
                    coadd, abs_ref, max_offset_arcsec=search
                )
            row['measure'] = {
                'ok': bool(off.get('ok')),
                'abs_arcsec': off.get('abs_arcsec'),
                'dra_arcsec': off.get('dra_arcsec'),
                'ddec_arcsec': off.get('ddec_arcsec'),
                'peak_count': off.get('peak_count'),
                'n_pairs': off.get('n_pairs'),
                'method': off.get('method'),
            }
            if not off.get('ok'):
                logger.warning(
                    'L3 unify: could not measure %s vs %s (pairs=%s peak=%s)',
                    coadd.name,
                    abs_ref.name,
                    off.get('n_pairs'),
                    off.get('peak_count'),
                )
                iter_rows.append(row)
                continue

            if float(off['abs_arcsec']) <= limit:
                logger.info(
                    'L3 unify: %s already on frame (|Delta|=%.3f")',
                    coadd.name,
                    off['abs_arcsec'],
                )
                iter_rows.append(row)
                continue

            # img-ref -> apply -Delta to L2 (+ remosaic) so they move onto abs_ref.
            # When a shared stamp *output_wcs* is locked, never mutate the L3
            # CRVAL (that walks coadds off the JWST stamp); remosaic instead.
            dra_deg = -float(off['dra_deg'])
            ddec_deg = -float(off['ddec_deg'])
            dra_as = -float(off['dra_arcsec'])
            ddec_as = -float(off['ddec_arcsec'])
            logger.info(
                'L3 unify iter%d: %s -> %s apply dRA=%+.3f" dDec=%+.3f" '
                '(was |Delta|=%.3f", peak=%s)%s',
                it + 1,
                coadd.name,
                abs_ref.name,
                dra_as,
                ddec_as,
                off['abs_arcsec'],
                off.get('peak_count'),
                '; stamp WCS locked' if output_wcs is not None else '',
            )

            n_l2 = 0
            for fp in frames:
                if fp.is_file():
                    n_l2 += apply_sky_shift_to_fits(
                        fp, dra_deg, ddec_deg, comment='st123: L3-unify to L2'
                    )
            if output_wcs is None:
                apply_sky_shift_to_fits(
                    coadd, dra_deg, ddec_deg, comment='st123: L3-unify to coadd'
                )
            elif not remosaic:
                # Shared stamp: refuse CRVAL walk; force remosaic path below.
                remosaic = True

            # Common CRVAL shift preserves relatives; if internal QA still
            # fails (pre-existing drift / measurement), re-harmonize in place.
            if len(frames) >= 2:
                from st123.stages.alignment.hst_jhat import harmonize_hst_group_wcs

                qa = validate_hst_group_internal_alignment(
                    frames, max_coherent_arcsec=HST_INTERNAL_ALIGN_MAX_ARCSEC
                )
                if not qa.get('ok'):
                    logger.warning(
                        'L3 unify: internal L2 QA soft-fail for %s/%s after '
                        'shift (max |Delta|=%.3f"); re-harmonizing',
                        rec.get('instrument'),
                        rec.get('filter'),
                        qa.get('max_abs_arcsec', 0.0),
                    )
                    harm = harmonize_hst_group_wcs(
                        frames,
                        max_internal_arcsec=HST_INTERNAL_ALIGN_MAX_ARCSEC,
                        abs_ref=None,
                        force=True,
                    )
                    row['harmonize_after'] = {
                        'ok': bool(harm.get('ok')),
                        'method': harm.get('method'),
                        'post_max_abs_arcsec': (harm.get('post') or {}).get(
                            'max_abs_arcsec'
                        ),
                    }
                    qa = validate_hst_group_internal_alignment(
                        frames, max_coherent_arcsec=HST_INTERNAL_ALIGN_MAX_ARCSEC
                    )
                row['internal_qa_after'] = {
                    'ok': bool(qa.get('ok')),
                    'max_abs_arcsec': qa.get('max_abs_arcsec'),
                }
                if not qa.get('ok'):
                    logger.error(
                        'L3 unify: internal L2 QA failed for %s/%s after shift '
                        '(max |Delta|=%.3f"); remosaic may ghost',
                        rec.get('instrument'),
                        rec.get('filter'),
                        qa.get('max_abs_arcsec', 0.0),
                    )
                    row['error'] = (
                        f'internal L2 QA failed after L3 unify '
                        f'(max |Delta|={qa.get("max_abs_arcsec")})'
                    )

            if remosaic and frames:
                try:
                    product = drizzle_filter_group(
                        frames,
                        coadd,
                        final_pixfrac=final_pixfrac,
                        final_scale=final_scale,
                        num_cores=num_cores,
                        instrument=rec.get('instrument'),
                        output_wcs=output_wcs,
                    )
                    rec['output'] = str(product)
                    row['remosaicked'] = str(product)
                    # Refresh abs_ref path object if we remosaicked the ref (we don't).
                except Exception as exc:
                    row['remosaic_error'] = str(exc)
                    logger.exception(
                        'L3 unify remosaic failed for %s/%s: %s',
                        rec.get('instrument'),
                        rec.get('filter'),
                        exc,
                    )

            row['applied'] = True
            row['dra_arcsec'] = dra_as
            row['ddec_arcsec'] = ddec_as
            row['n_l2_updated'] = n_l2
            any_shift = True
            iter_rows.append(row)

        report['groups'] = iter_rows
        internal_bad = [g for g in iter_rows if g.get('error')]

        # Re-pick abs_ref in case paths changed; verify all coadds.
        coadd_paths = [Path(r['output']) for r in records if Path(r['output']).is_file()]
        coadd_n_frames = {
            Path(r['output']).resolve(): len(r.get('frames') or [])
            for r in records
            if r.get('output')
        }
        abs_ref = (
            _pick_coadd_abs_ref(coadd_paths, n_frames=coadd_n_frames) or abs_ref
        )
        report['abs_ref'] = str(abs_ref)
        report['abs_ref_n_frames'] = coadd_n_frames.get(Path(abs_ref).resolve())
        final_qa = validate_hst_coadds_alignment(
            coadd_paths, max_coherent_arcsec=limit
        )
        report['final_qa'] = final_qa
        if final_qa.get('ok') and not internal_bad:
            report['ok'] = True
            logger.info(
                'L3/L2 astrometric unify OK after iter %d (max |Delta|=%.3f")',
                it + 1,
                final_qa.get('max_abs_arcsec', 0.0),
            )
            break
        if final_qa.get('ok') and internal_bad:
            logger.error(
                'L3 QA passed but internal L2 failed for %d group(s); not OK for DOLPHOT',
                len(internal_bad),
            )
            break
        if not any_shift:
            logger.error(
                'L3/L2 unify: residual remains (max |Delta|=%.3f") but no shift applied',
                final_qa.get('max_abs_arcsec', 0.0),
            )
            break
        logger.warning(
            'L3/L2 unify iter %d still failing (max |Delta|=%.3f"); continuing',
            it + 1,
            final_qa.get('max_abs_arcsec', 0.0),
        )

    qa_path = out / 'astrometric_frame_qa.json'
    try:
        qa_path.write_text(json.dumps(report, indent=2, default=str) + '\n')
        logger.info('Wrote %s', qa_path)
    except Exception as exc:
        logger.warning('Could not write %s: %s', qa_path, exc)
    report['qa_path'] = str(qa_path)
    return report


# Retained for callers/docs; visit sidecars are opt-in via --visit-coadds.
_IR_VISIT_COADD_FILTERS = frozenset({'f110w', 'f160w'})


def _drizzle_visit_coadds(
    frames: Sequence[PathLike],
    *,
    outdir: Path,
    instrument: str,
    filt: str,
    suffix: str,
    dropped_visits: set[str],
    final_pixfrac: float,
    final_scale: Optional[float],
    num_cores: int,
    group_id: int | None = None,
    box_id: int | str | None = None,
    output_wcs=None,
    all_visits: bool = False,
    assume_aligned: bool = True,
) -> list[dict]:
    """
    Optionally drizzle per-visit sidecars (not the primary instrument+filter coadd).

    When *all_visits* is False (default), only *dropped_visits* are written.
    When True, every visit is written. Visits that fail within-visit internal QA
    are skipped unless *assume_aligned* (``align`` already owns registration).
    """
    from collections import defaultdict

    from st123.stages.alignment.hst_jhat import (
        HST_INTERNAL_ALIGN_MAX_ARCSEC,
        _hst_visit_key,
        validate_hst_group_internal_alignment,
    )

    by_v: dict[str, list[Path]] = defaultdict(list)
    for p in frames:
        by_v[_hst_visit_key(Path(p))].append(Path(p).resolve())

    want_all = bool(all_visits)
    products: list[dict] = []
    for vid, vpaths in sorted(by_v.items()):
        if not (want_all or vid in dropped_visits):
            continue
        if (not assume_aligned) and len(vpaths) >= 2:
            vqa = validate_hst_group_internal_alignment(
                vpaths,
                max_coherent_arcsec=HST_INTERNAL_ALIGN_MAX_ARCSEC,
            )
            if not vqa.get('ok'):
                logger.warning(
                    'Skip visit coadd %s/%s visit=%s: within-visit QA failed '
                    '(max |Delta|=%s)',
                    instrument,
                    filt,
                    vid,
                    vqa.get('max_abs_arcsec'),
                )
                products.append(
                    {
                        'visit': vid,
                        'status': 'skipped_incoherent',
                        'n_frames': len(vpaths),
                        'max_abs_arcsec': vqa.get('max_abs_arcsec'),
                    }
                )
                continue
        if group_id is not None and box_id is not None:
            from st123.stages.mosaic.mosaic import mosaic_hst_coadd_basename

            visit_name = mosaic_hst_coadd_basename(
                group_id,
                box_id,
                instrument,
                filt,
                suffix=suffix,
                visit=vid,
            )
        else:
            visit_name = f'coadd_{instrument}_{filt}_{vid}_{suffix}.fits'
        visit_path = outdir / visit_name
        row: dict = {
            'visit': vid,
            'frames': [str(p) for p in vpaths],
            'output': str(visit_path),
            'status': 'pending',
            'dropped_from_filter_stack': vid in dropped_visits,
        }
        try:
            product = drizzle_filter_group(
                vpaths,
                visit_path,
                final_pixfrac=final_pixfrac,
                final_scale=final_scale,
                num_cores=num_cores,
                instrument=instrument,
                output_wcs=output_wcs,
            )
            row['output'] = str(product)
            row['status'] = 'ok'
            logger.info(
                'Wrote visit coadd %s (%d frames; dropped=%s)',
                product.name,
                len(vpaths),
                vid in dropped_visits,
            )
        except Exception as exc:
            row['status'] = 'failed'
            row['error'] = str(exc)
            logger.warning(
                'Visit coadd failed for %s/%s visit=%s: %s',
                instrument,
                filt,
                vid,
                exc,
            )
        products.append(row)
    return products


_HST_RAW_SCIENCE_SUFFIXES = ('_flc.fits', '_flt.fits', '_c0m.fits')


def hst_science_rootname(path: PathLike) -> str:
    """Rootname used to pair a JHAT product with its calibrated sibling."""
    name = Path(path).name.lower()
    for suffix in ('_jhat.fits', *_HST_RAW_SCIENCE_SUFFIXES, '_cal.fits'):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem.lower()


def collect_unaligned_hst_raw_frames(jhat_dir: PathLike) -> list[Path]:
    """Calibrated HST frames in ``reduction/raw`` that have no JHAT sibling.

    Align can leave sparse WFPC2 chips without ``*_jhat.fits`` when JHAT finds
    too few refcat matches. Mosaic still needs those frames so every requested
    filter can be dribbled onto the shared stamp.
    """
    jhat = Path(jhat_dir)
    raw_dir = jhat.parent / 'raw'
    if not raw_dir.is_dir():
        return []
    have = {
        hst_science_rootname(path)
        for path in jhat.glob('*_jhat.fits')
        if path.is_file()
    }
    found: list[Path] = []
    for path in sorted(raw_dir.iterdir()):
        name = path.name.lower()
        if not name.endswith(_HST_RAW_SCIENCE_SUFFIXES):
            continue
        if hst_science_rootname(path) in have:
            continue
        found.append(path.resolve())
    return found


def _collect_hst_jhat_frames(
    jhat_dir: PathLike,
    *,
    pattern: str = '*_jhat.fits',
) -> list[Path]:
    """List HST JHAT science frames under *jhat_dir* (skip JWST / coadds)."""
    from st123.datamodels.hst import filter_paths_for_stage

    jhat = Path(jhat_dir)
    frames = sorted(jhat.glob(pattern))
    if not frames:
        frames = sorted(
            p
            for p in jhat.glob('*.fits')
            if HSTDataModel.matches_science_filename(p)
        )
    frames = [
        p
        for p in frames
        if HSTDataModel.matches_science_filename(p)
        and not p.name.lower().startswith('coadd_')
        and not p.name.lower().startswith('jw')
        and 'l3_ref' not in p.parts
    ]
    extra = collect_unaligned_hst_raw_frames(jhat)
    if extra:
        logger.info(
            'Including %d calibrated HST frame(s) with no JHAT product',
            len(extra),
        )
        frames.extend(extra)
    seen: set[str] = set()
    uniq: list[Path] = []
    for path in frames:
        key = str(Path(path).resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(path)
    frames = filter_paths_for_stage(uniq, stage='mosaic-collect')
    kept, dropped = _drop_catastrophic_jhat_frames(
        frames, context=str(jhat.name)
    )
    del dropped
    return kept


def _drop_catastrophic_jhat_frames(
    paths: Sequence[Path],
    *,
    peer_n_calibrators: int | None = None,
    context: str = '',
) -> tuple[list[Path], list[dict]]:
    """Exclude JHAT frames with a catastrophic alignment residual."""
    from st123.stages.alignment.frame_qa import filter_catastrophic_jhat_frames

    kept, dropped = filter_catastrophic_jhat_frames(
        list(paths), peer_n_calibrators=peer_n_calibrators
    )
    rejected: list[dict] = []
    if dropped:
        names = ', '.join(p.name for p, _ in dropped)
        logger.error(
            'Dropping %d catastrophically misaligned JHAT frame(s)%s: %s',
            len(dropped),
            f' ({context})' if context else '',
            names,
        )
        for path, qual in dropped:
            disp = qual.get('residual_arcsec')
            rejected.append(
                {
                    'path': str(path),
                    'status': 'failed',
                    'error': (
                        'catastrophic alignment residual: '
                        f'{1000.0 * float(disp):.1f} mas'
                        if disp is not None
                        else 'catastrophic alignment residual'
                    ),
                    'n_calibrators': qual.get('n_calibrators'),
                    'residual_arcsec': disp,
                }
            )
    return kept, rejected


def _reject_catastrophic_l3_coadds(
    results: list[dict],
    qa_report: Mapping[str, Any] | None,
    *,
    abs_ref_name: str | None = None,
) -> list[dict]:
    """
    Fail and sideline L3 coadds that share no sources with the astrometric hub.

    Renames ``coadd_*.fits`` to ``*.rejected.fits`` so a later photometry
    stage cannot pick up a catastrophically shifted stack.
    """
    from st123.stages.alignment.frame_qa import isolated_unmatched_coadds

    if not qa_report:
        return []
    drop_names = set(
        isolated_unmatched_coadds(qa_report, abs_ref_name=abs_ref_name)
    )
    if not drop_names:
        return []
    rejected: list[dict] = []
    for rec in results:
        if rec.get('status') != 'ok':
            continue
        out = Path(rec.get('output') or '')
        if out.name not in drop_names:
            continue
        rec['status'] = 'failed'
        rec['error'] = (
            'catastrophic L3 alignment: no common sources vs aligned coadds'
        )
        logger.error(
            'Rejecting mosaic product %s (%s/%s): %s',
            out.name,
            rec.get('instrument'),
            rec.get('filter'),
            rec['error'],
        )
        if out.is_file():
            dest = out.with_name(out.name.replace('.fits', '.rejected.fits'))
            try:
                out.replace(dest)
                rec['rejected_output'] = str(dest)
                rec['output'] = str(dest)
            except Exception as exc:
                logger.warning('Could not sideline %s: %s', out.name, exc)
        rejected.append(rec)
    return rejected


def _parallel_drizzle_budget(n_jobs: int, num_cores: int) -> tuple[int, int]:
    """Return ``(n_workers, cores_per_worker)`` for parallel AstroDrizzle jobs."""
    n_jobs = max(1, int(n_jobs))
    num_cores = max(1, int(num_cores))
    n_workers = min(n_jobs, num_cores)
    cores_per = max(1, num_cores // n_workers)
    return n_workers, cores_per


def _frame_sets_disjoint(frame_groups: Sequence[Sequence[Path]]) -> bool:
    """True when no resolved path appears in more than one group."""
    seen: set[str] = set()
    for group in frame_groups:
        keys = {str(Path(p).resolve()) for p in group if Path(p).is_file()}
        if seen & keys:
            return False
        seen |= keys
    return True


def _prepare_filter_drizzle(
    inst: str,
    filt: str,
    imgs: Sequence[Path],
    *,
    outdir: Path,
    group_id: int,
    box_id: int | str,
    drop_outlier_visits: bool = False,
    assume_aligned: bool = True,
) -> dict:
    """
    Prep one filter group for AstroDrizzle (no AstroDrizzle yet).

    When *assume_aligned* is True (default), skip relative WCS harmonize and
    hard internal-alignment gates -- ``align`` is expected to have put JHAT
    frames on a common frame. Still heals incomplete multi-SCI chip refine
    leftovers and drops frames that remain partial after heal.

    Returns a record with ``status`` ``ready`` (drizzle inputs in
    ``drizzle_imgs``) or ``failed``.
    """
    from st123.stages.alignment.hst_jhat import (
        HST_INTERNAL_ALIGN_MAX_ARCSEC,
        HST_L3_ALIGN_MAX_ARCSEC,
        _frame_exptime,
        _hst_visit_key,
        harmonize_hst_group_wcs,
        heal_hst_partial_chip_refine,
        measure_hst_sky_offset_2dhist,
        validate_hst_group_internal_alignment,
        validate_hst_multi_sci_chip_refine,
        validate_hst_visits_internal_alignment,
    )
    from st123.stages.mosaic.mosaic import mosaic_hst_coadd_basename

    suffix = _drizzle_suffix(inst, imgs)
    coadd_name = mosaic_hst_coadd_basename(
        group_id, box_id, inst, filt, suffix=suffix
    )
    coadd_path = Path(outdir) / coadd_name
    record: dict = {
        'instrument': inst,
        'filter': filt,
        'group': group_id,
        'box': box_id,
        'frames': [str(p) for p in imgs],
        'output': str(coadd_path),
        'status': 'pending',
        'error': None,
        'internal_qa': None,
        'harmonize': None,
        'suffix': suffix,
        'drizzle_imgs': [],
        'assume_aligned': bool(assume_aligned),
    }
    try:
        from st123.datamodels.hst import filter_paths_for_stage

        # EXPFLAG only here. ST123INT is a group stamp (unsafe per-frame);
        # visit outliers are handled by --drop-outlier-visits.
        drizzle_imgs = filter_paths_for_stage(
            list(imgs),
            stage=f'mosaic-{inst}-{filt}',
        )
        if not drizzle_imgs:
            raise RuntimeError(
                f'No frames remain for {inst}/{filt} after EXPFLAG filtering'
            )
        if assume_aligned:
            record['harmonize'] = {
                'ok': True,
                'skipped': True,
                'reason': 'assume_aligned',
            }
            record['internal_qa'] = {
                'ok': True,
                'skipped': True,
                'reason': 'assume_aligned',
            }
        else:
            harm = harmonize_hst_group_wcs(
                drizzle_imgs,
                max_internal_arcsec=HST_INTERNAL_ALIGN_MAX_ARCSEC,
                abs_ref=None,
                force=False,
            )
            record['harmonize'] = {
                'ok': bool(harm.get('ok')),
                'anchor': harm.get('anchor'),
                'method': harm.get('method'),
                'pre_max_abs_arcsec': (harm.get('pre') or {}).get(
                    'max_abs_arcsec'
                ),
                'post_max_abs_arcsec': (harm.get('post') or {}).get(
                    'max_abs_arcsec'
                ),
            }
            if harm.get('method') == 'visit_split_harmonize':
                qa = validate_hst_visits_internal_alignment(
                    drizzle_imgs,
                    max_coherent_arcsec=HST_INTERNAL_ALIGN_MAX_ARCSEC,
                )
                bad_visits = {
                    str(v.get('visit'))
                    for v in (qa.get('visits') or [])
                    if not v.get('ok')
                }
                if bad_visits:
                    kept = [
                        p
                        for p in drizzle_imgs
                        if _hst_visit_key(p) not in bad_visits
                    ]
                    logger.warning(
                        'Dropping incoherent visit(s) %s from %s/%s '
                        '(%d -> %d frames)',
                        sorted(bad_visits),
                        inst,
                        filt,
                        len(drizzle_imgs),
                        len(kept),
                    )
                    drizzle_imgs = kept
                    record['dropped_visits'] = sorted(bad_visits)
                    if len(drizzle_imgs) < 1:
                        raise RuntimeError(
                            f'Internal L2 alignment QA failed for {inst}/{filt}: '
                            f'no visits remain after dropping {sorted(bad_visits)}'
                        )
                    qa = validate_hst_visits_internal_alignment(
                        drizzle_imgs,
                        max_coherent_arcsec=HST_INTERNAL_ALIGN_MAX_ARCSEC,
                    )
            else:
                qa = validate_hst_group_internal_alignment(
                    drizzle_imgs,
                    max_coherent_arcsec=HST_INTERNAL_ALIGN_MAX_ARCSEC,
                )
            record['internal_qa'] = {
                'ok': bool(qa.get('ok')),
                'max_abs_arcsec': qa.get('max_abs_arcsec'),
                'failed_pairs': qa.get('failed_pairs'),
                'scope': qa.get('scope', 'full_group'),
                'visits': qa.get('visits'),
            }
            if not qa.get('ok'):
                raise RuntimeError(
                    f'Internal L2 alignment QA failed for {inst}/{filt}: '
                    f'max |Delta|={qa.get("max_abs_arcsec"):.3f}" '
                    f'(limit {HST_INTERNAL_ALIGN_MAX_ARCSEC:.3f}; '
                    f'scope={qa.get("scope", "full_group")}); '
                    f'failed_pairs={qa.get("failed_pairs") or qa.get("visits")}'
                )
            if (
                harm.get('method') == 'visit_split_harmonize'
                and len(drizzle_imgs) >= 2
            ):
                by_v: dict[str, list] = defaultdict(list)
                for p in drizzle_imgs:
                    by_v[_hst_visit_key(p)].append(p)
                visit_anchors = {
                    vid: max(vpaths, key=_frame_exptime)
                    for vid, vpaths in by_v.items()
                    if vpaths
                }
                if len(visit_anchors) >= 2:
                    hub_vid = max(
                        visit_anchors,
                        key=lambda vid: _total_exptime(by_v[vid]),
                    )
                    hub = visit_anchors[hub_vid]
                    outlier_visits: list[str] = []
                    pair_rows: list[dict] = []
                    for vid, anchor in sorted(visit_anchors.items()):
                        if vid == hub_vid:
                            continue
                        off = measure_hst_sky_offset_2dhist(
                            anchor,
                            hub,
                            max_offset_arcsec=5.0,
                        )
                        abs_as = float(off.get('abs_arcsec') or 0.0)
                        pair_rows.append(
                            {
                                'visit': vid,
                                'hub': hub_vid,
                                'ok': bool(off.get('ok')),
                                'abs_arcsec': abs_as,
                                'peak_count': off.get('peak_count'),
                            }
                        )
                        if (
                            (not off.get('ok'))
                            or abs_as > HST_L3_ALIGN_MAX_ARCSEC
                        ):
                            outlier_visits.append(vid)
                    record['cross_visit_qa'] = {
                        'hub_visit': hub_vid,
                        'limit_arcsec': HST_L3_ALIGN_MAX_ARCSEC,
                        'pairs': pair_rows,
                        'outlier_visits': outlier_visits,
                    }
                    if outlier_visits and drop_outlier_visits:
                        kept = [
                            p
                            for p in drizzle_imgs
                            if _hst_visit_key(p) not in set(outlier_visits)
                        ]
                        logger.warning(
                            'Dropping cross-visit outlier(s) %s from %s/%s '
                            '(hub=%s, limit=%.3f"; %d -> %d frames)',
                            outlier_visits,
                            inst,
                            filt,
                            hub_vid,
                            HST_L3_ALIGN_MAX_ARCSEC,
                            len(drizzle_imgs),
                            len(kept),
                        )
                        drizzle_imgs = kept
                        record.setdefault('dropped_visits', [])
                        record['dropped_visits'] = sorted(
                            set(record['dropped_visits']) | set(outlier_visits)
                        )
                        if len(drizzle_imgs) < 1:
                            raise RuntimeError(
                                f'No frames remain for {inst}/{filt} after '
                                f'dropping cross-visit outliers {outlier_visits}'
                            )
                    elif outlier_visits:
                        logger.warning(
                            'Cross-visit outlier(s) %s in %s/%s remain in the '
                            'combined instrument+filter stack '
                            '(hub=%s, limit=%.3f"); pass --drop-outlier-visits '
                            'to exclude them, or --visit-coadds for sidecars',
                            outlier_visits,
                            inst,
                            filt,
                            hub_vid,
                            HST_L3_ALIGN_MAX_ARCSEC,
                        )
                        record['retained_outlier_visits'] = list(outlier_visits)

        heal = heal_hst_partial_chip_refine(drizzle_imgs)
        record['chip_refine_heal'] = {
            'n_healed': heal.get('n_healed'),
            'n_failed': heal.get('n_failed'),
            'frames': [
                {
                    'path': Path(r['path']).name,
                    'healed': r.get('healed'),
                    'error': r.get('error'),
                    'dx_pix': r.get('dx_pix'),
                    'dy_pix': r.get('dy_pix'),
                }
                for r in (heal.get('frames') or [])
                if r.get('healed') or r.get('error')
            ],
        }
        chip_qa = validate_hst_multi_sci_chip_refine(drizzle_imgs)
        record['chip_refine_qa'] = {
            'ok': bool(chip_qa.get('ok')),
            'n_failed': chip_qa.get('n_failed'),
            'frames': [
                {
                    'path': Path(r['path']).name,
                    'ok': r.get('ok'),
                    'n_sci': r.get('n_sci'),
                    'n_refined': r.get('n_refined'),
                    'error': r.get('error'),
                }
                for r in (chip_qa.get('frames') or [])
                if not r.get('ok') or (r.get('n_sci') or 0) >= 2
            ],
        }
        if not chip_qa.get('ok'):
            bad_names = {
                Path(r['path']).name
                for r in (chip_qa.get('frames') or [])
                if not r.get('ok')
            }
            kept = [p for p in drizzle_imgs if Path(p).name not in bad_names]
            logger.warning(
                'Dropping %d frame(s) with incomplete multi-SCI chip refine '
                'from %s/%s: %s',
                len(bad_names),
                inst,
                filt,
                sorted(bad_names),
            )
            record['dropped_partial_chip_refine'] = sorted(bad_names)
            drizzle_imgs = kept
            if len(drizzle_imgs) < 1:
                raise RuntimeError(
                    f'No frames remain for {inst}/{filt} after dropping '
                    f'partial multi-SCI chip refine: {sorted(bad_names)}'
                )
        record['drizzle_imgs'] = [str(Path(p).resolve()) for p in drizzle_imgs]
        record['frames'] = list(record['drizzle_imgs'])
        record['status'] = 'ready'
    except Exception as exc:
        record['status'] = 'failed'
        record['error'] = str(exc)
        logger.exception(
            'Drizzle prep failed for %s/%s (%d frames): %s',
            inst,
            filt,
            len(imgs),
            exc,
        )
    return record



def _filter_drizzle_worker(job: dict) -> dict:
    """
    AstroDrizzle one prepared filter group (module-level for ProcessPool).

    Expects a record from :func:`_prepare_filter_drizzle` plus drizzle knobs.
    """
    from st123.utils.logging import configure_worker_logging

    configure_worker_logging()
    record = dict(job.get('record') or {})
    if record.get('status') != 'ready':
        return record
    inst = str(record['instrument'])
    filt = str(record['filter'])
    drizzle_imgs = [Path(p) for p in record.get('drizzle_imgs') or []]
    coadd_path = Path(record['output'])
    out = Path(job['outdir'])
    group_id = job['group_id']
    box_id = job['box_id']
    try:
        product = drizzle_filter_group(
            drizzle_imgs,
            coadd_path,
            final_pixfrac=float(job['final_pixfrac']),
            final_scale=job.get('final_scale'),
            num_cores=int(job['num_cores']),
            instrument=inst,
            output_wcs=job.get('output_wcs'),
        )
        record['output'] = str(product)
        record['status'] = 'ok'
        record['frames'] = [str(p) for p in drizzle_imgs]
        # Tagged per-filter list only here - parallel workers must not race on
        # the shared dolphot_frames.txt (merged after all jobs finish).
        tagged = out / f'dolphot_frames_{inst}_{filt}.txt'
        tagged.write_text(
            f'# group={int(group_id)} box={box_id}\n'
            f'# instrument={inst} filter={filt}\n'
            f'# ref {Path(product).resolve()}\n'
            + ''.join(f'{Path(p).resolve()}\n' for p in drizzle_imgs)
        )
        logger.info('Wrote %s (%d frames)', product, len(drizzle_imgs))

        if bool(job.get('visit_coadds')):
            imgs = [Path(p) for p in job.get('input_imgs') or drizzle_imgs]
            record['visit_coadds'] = _drizzle_visit_coadds(
                imgs,
                outdir=out,
                instrument=inst,
                filt=filt,
                suffix=str(record.get('suffix') or _drizzle_suffix(inst, imgs)),
                dropped_visits=set(record.get('dropped_visits') or []),
                final_pixfrac=float(job['final_pixfrac']),
                final_scale=job.get('final_scale'),
                num_cores=int(job['num_cores']),
                group_id=group_id,
                box_id=box_id,
                output_wcs=job.get('output_wcs'),
                all_visits=True,
                assume_aligned=bool(record.get('assume_aligned', True)),
            )
    except Exception as exc:
        record['status'] = 'failed'
        record['error'] = str(exc)
        logger.exception(
            'Drizzle failed for %s/%s (%d frames): %s',
            inst,
            filt,
            len(drizzle_imgs),
            exc,
        )
    return record


def _run_filter_drizzle_jobs(
    jobs: Sequence[dict],
    *,
    num_cores: int,
    parallel: bool = True,
) -> list[dict]:
    """Run prepared filter-drizzle jobs serially or via a process pool."""
    if not jobs:
        return []
    n_workers, cores_per = _parallel_drizzle_budget(len(jobs), num_cores)
    use_pool = bool(parallel) and n_workers > 1 and len(jobs) > 1
    for job in jobs:
        job['num_cores'] = cores_per if use_pool else max(1, int(num_cores))
    if not use_pool:
        return [_filter_drizzle_worker(job) for job in jobs]

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    logger.info(
        'Parallel filter drizzle: %d job(s), workers=%d, cores/job=%d',
        len(jobs),
        n_workers,
        cores_per,
    )
    ctx = mp.get_context('spawn')
    ordered: list[dict | None] = [None] * len(jobs)
    from st123.utils.logging import configure_worker_logging

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=configure_worker_logging,
    ) as pool:
        future_map = {
            pool.submit(_filter_drizzle_worker, job): i for i, job in enumerate(jobs)
        }
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                ordered[idx] = fut.result()
            except Exception as exc:
                rec = dict(jobs[idx].get('record') or {})
                rec['status'] = 'failed'
                rec['error'] = f'{type(exc).__name__}: {exc}'
                ordered[idx] = rec
                logger.error(
                    'Filter drizzle worker crashed for %s/%s: %s',
                    rec.get('instrument'),
                    rec.get('filter'),
                    exc,
                )
    return [r for r in ordered if r is not None]


def _drizzle_box_groups(
    ranked: Sequence[tuple[tuple[str, str], Sequence[Path]]],
    *,
    outdir: Path,
    group_id: int,
    box_id: int | str,
    final_pixfrac: float,
    final_scale: Optional[float],
    num_cores: int,
    all_box_frames: Sequence[Path],
    preferred_abs_ref: Sequence[Path] | None = None,
    raise_on_unify_fail: bool = False,
    output_wcs=None,
    parallel_filters: bool = True,
    drop_outlier_visits: bool = False,
    visit_coadds: bool = False,
    assume_aligned: bool = True,
) -> list[dict]:
    """Drizzle ranked ``(inst, filt)`` groups into one mosaic box directory."""
    from st123.stages.alignment.hst_jhat import (
        HST_L3_ALIGN_MAX_ARCSEC,
        find_hst_abs_ref_image,
        harmonize_hst_visits_across_filters,
    )

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    # Optional cross-filter visit retie (legacy). Default assumes ``align``
    # already put JHAT products on a common absolute frame.
    box_frames = [Path(p).resolve() for p in all_box_frames if Path(p).is_file()]
    if (not assume_aligned) and len(box_frames) >= 2:
        try:
            visit_x = harmonize_hst_visits_across_filters(
                box_frames,
                abs_ref=find_hst_abs_ref_image(box_frames[0].parent),
            )
            if not visit_x.get('ok'):
                logger.warning(
                    'Visit cross-filter harmonize soft-failed in box %s/%s '
                    '(%d visit row(s)); continuing with per-filter QA',
                    group_id,
                    box_id,
                    len(visit_x.get('visits') or []),
                )
        except Exception as exc:
            logger.warning(
                'Visit cross-filter harmonize raised in box %s/%s: %s',
                group_id,
                box_id,
                exc,
            )

    from st123.stages.alignment.frame_qa import (
        median_healthy_n_calibrators,
        read_jhat_quality_headers,
    )
    from st123.stages.alignment.hst_jhat import validate_hst_coadds_alignment

    peer_n = median_healthy_n_calibrators(
        read_jhat_quality_headers(p).get('n_calibrators') for p in box_frames
    )
    filtered_ranked: list[tuple[tuple[str, str], list[Path]]] = []
    failed_prep: list[dict] = []
    for (inst, filt), imgs in ranked:
        kept, rejected = _drop_catastrophic_jhat_frames(
            [Path(p) for p in imgs],
            peer_n_calibrators=peer_n,
            context=f'{inst}/{filt} group {group_id} box {box_id}',
        )
        if not kept:
            suffix = _drizzle_suffix(inst, list(imgs))
            from st123.stages.mosaic.mosaic import mosaic_hst_coadd_basename

            coadd_name = mosaic_hst_coadd_basename(
                group_id, box_id, inst, filt, suffix=suffix
            )
            coadd_path = out / coadd_name
            if coadd_path.is_file():
                dest = coadd_path.with_name(
                    coadd_path.name.replace('.fits', '.rejected.fits')
                )
                try:
                    coadd_path.replace(dest)
                    logger.error(
                        'Sidelined stale mosaic product %s -> %s',
                        coadd_path.name,
                        dest.name,
                    )
                except Exception as exc:
                    logger.warning(
                        'Could not sideline stale %s: %s', coadd_path.name, exc
                    )
            failed_prep.append(
                {
                    'instrument': inst,
                    'filter': filt,
                    'group': group_id,
                    'box': box_id,
                    'frames': [str(p) for p in imgs],
                    'output': str(coadd_path),
                    'status': 'failed',
                    'error': (
                        rejected[0]['error']
                        if rejected
                        else 'catastrophic alignment residual'
                    ),
                    'dropped_catastrophic': [r['path'] for r in rejected],
                }
            )
            continue
        filtered_ranked.append(((inst, filt), kept))

    ready_jobs: list[dict] = []
    for (inst, filt), imgs in filtered_ranked:
        record = _prepare_filter_drizzle(
            inst,
            filt,
            imgs,
            outdir=out,
            group_id=group_id,
            box_id=box_id,
            drop_outlier_visits=bool(drop_outlier_visits),
            assume_aligned=bool(assume_aligned),
        )
        if record.get('status') != 'ready':
            failed_prep.append(record)
            continue
        ready_jobs.append(
            {
                'record': record,
                'outdir': str(out),
                'group_id': group_id,
                'box_id': box_id,
                'final_pixfrac': float(final_pixfrac),
                'final_scale': final_scale,
                'num_cores': int(num_cores),
                'output_wcs': _pickle_safe_output_wcs(output_wcs),
                'input_imgs': [str(Path(p).resolve()) for p in imgs],
                'visit_coadds': bool(visit_coadds),
            }
        )

    drizzle_results = _run_filter_drizzle_jobs(
        ready_jobs,
        num_cores=num_cores,
        parallel=parallel_filters,
    )
    results = failed_prep + drizzle_results

    ok = [r for r in results if r['status'] == 'ok']
    if ok:
        best = ok[0]
        # Rebuild merged dolphot_frames.txt after parallel filter writers finish.
        merged_frames: list[Path] = []
        for r in ok:
            merged_frames.extend(Path(p) for p in (r.get('frames') or []))
        if not merged_frames:
            merged_frames = [Path(p) for p in all_box_frames]
        _write_group_frame_list(
            out,
            refimage=Path(best['output']),
            frames=sorted({p.resolve() for p in merged_frames}),
            instrument=best['instrument'],
            filt=best['filter'],
            group=group_id,
            box=box_id,
            merge=True,
        )
        unify = None
        if assume_aligned:
            # Align owns registration. Still run L3 QA; if a coadd is off
            # the shared frame (classic WFPC2 ~1" pipeline WCS that visit
            # retie refused), unify rather than keeping the residual.
            from st123.stages.alignment.frame_qa import (
                build_frame_qa,
                read_frame_qa,
                warn_if_frame_qa_soft,
                write_frame_qa,
            )
            from st123.stages.alignment.hst_jhat import validate_hst_coadds_alignment

            coadd_paths = [Path(r['output']) for r in ok if Path(r['output']).is_file()]
            final_qa = validate_hst_coadds_alignment(
                coadd_paths, max_coherent_arcsec=HST_L3_ALIGN_MAX_ARCSEC
            )
            if final_qa.get('ok'):
                unify = {
                    'ok': True,
                    'abs_ref': None,
                    'iterations': 0,
                    'final_qa': final_qa,
                    'qa_path': None,
                    'skipped_shift': True,
                    'reason': 'assume_aligned',
                }
                try:
                    import json

                    qa_path = out / 'astrometric_frame_qa.json'
                    qa_path.write_text(
                        json.dumps(unify, indent=2, default=str) + '\n'
                    )
                    unify['qa_path'] = str(qa_path)
                    jhat_qa = None
                    for parent in (
                        out.parent.parent / 'jhat_hst',
                        out.parent / 'jhat_hst',
                    ):
                        jhat_qa = read_frame_qa(parent)
                        if jhat_qa:
                            break
                    max_mas = float(final_qa.get('max_abs_arcsec') or 0.0) * 1000.0
                    mosaic_qa = build_frame_qa(
                        mission='hst',
                        align_mode='MOSAIC_QA',
                        residual_mas=(
                            (jhat_qa.get('abs') or {}).get('residual_mas')
                            if jhat_qa
                            else None
                        ),
                        n_calibrators=(
                            (jhat_qa.get('abs') or {}).get('n_calibrators')
                            if jhat_qa
                            else None
                        ),
                        abs_method='align_frame_qa' if jhat_qa else None,
                        max_delta_mas=max_mas,
                        pairs=final_qa.get('pairs') or [],
                        frames=(jhat_qa.get('frames') if jhat_qa else None),
                        extra={
                            'mosaic_coadd_qa': final_qa,
                            'assume_aligned': True,
                        },
                    )
                    write_frame_qa(out, mosaic_qa)
                    warn_if_frame_qa_soft(
                        mosaic_qa,
                        log=logger,
                        context=f'group {group_id} box {box_id}',
                    )
                    logger.info(
                        'Wrote alignment QA (no mosaic shifts) %s '
                        '(max |Delta|=%.3f")',
                        qa_path,
                        float(final_qa.get('max_abs_arcsec') or 0.0),
                    )
                except Exception as exc:
                    logger.warning(
                        'Could not write astrometric_frame_qa.json: %s', exc
                    )
            else:
                logger.warning(
                    'L3 QA failed under assume_aligned (max |Delta|=%.3f"); '
                    'running astrometric unify',
                    float(final_qa.get('max_abs_arcsec') or 0.0),
                )
                unify = unify_hst_astrometric_frame(
                    ok,
                    outdir=out,
                    max_residual_arcsec=HST_L3_ALIGN_MAX_ARCSEC,
                    num_cores=num_cores,
                    final_pixfrac=final_pixfrac,
                    final_scale=final_scale,
                    remosaic=True,
                    preferred_abs_ref=preferred_abs_ref,
                    output_wcs=output_wcs,
                )
                unify['reason'] = 'assume_aligned_qa_failed'
        else:
            unify = unify_hst_astrometric_frame(
                ok,
                outdir=out,
                max_residual_arcsec=HST_L3_ALIGN_MAX_ARCSEC,
                num_cores=num_cores,
                final_pixfrac=final_pixfrac,
                final_scale=final_scale,
                remosaic=True,
                preferred_abs_ref=preferred_abs_ref,
                output_wcs=output_wcs,
            )
        for r in results:
            r['astrometric_unify'] = {
                'ok': bool(unify.get('ok')),
                'abs_ref': unify.get('abs_ref'),
                'iterations': unify.get('iterations'),
                'max_abs_arcsec': (unify.get('final_qa') or {}).get('max_abs_arcsec'),
                'qa_path': unify.get('qa_path'),
                'assume_aligned': bool(assume_aligned),
            }
            r['l3_qa'] = unify.get('final_qa')

        dropped_l3 = _reject_catastrophic_l3_coadds(
            results,
            unify.get('final_qa') if unify else None,
            abs_ref_name=(
                Path(unify['abs_ref']).name
                if unify and unify.get('abs_ref')
                else None
            ),
        )
        if dropped_l3 and unify is not None:
            remaining_ok = [
                r
                for r in results
                if r.get('status') == 'ok' and Path(r.get('output') or '').is_file()
            ]
            remaining_paths = [Path(r['output']) for r in remaining_ok]
            if len(remaining_paths) >= 2:
                re_qa = validate_hst_coadds_alignment(
                    remaining_paths, max_coherent_arcsec=HST_L3_ALIGN_MAX_ARCSEC
                )
                unify['final_qa'] = re_qa
                unify['ok'] = bool(re_qa.get('ok'))
            elif remaining_ok:
                unify['ok'] = True
            unify['dropped_catastrophic'] = [
                r.get('rejected_output') or r.get('output') for r in dropped_l3
            ]
            for r in results:
                r['l3_qa'] = unify.get('final_qa')
                ast = r.get('astrometric_unify') or {}
                ast['ok'] = bool(unify.get('ok'))
                r['astrometric_unify'] = ast
            ok = [r for r in results if r.get('status') == 'ok']

        if not unify.get('ok'):
            max_abs = (unify.get('final_qa') or {}).get('max_abs_arcsec')
            qa_path = unify.get('qa_path')
            # Default: coadds already written; align owns frame registration.
            log_fn = logger.error if raise_on_unify_fail else logger.warning
            log_fn(
                'Astrometric frame residual for group %s box %s '
                '(max |Delta|=%s); coadds kept -- see %s',
                group_id,
                box_id,
                max_abs,
                qa_path,
            )
            if raise_on_unify_fail:
                for r in ok:
                    r['status'] = 'failed'
                    r['error'] = (
                        'Astrometric unify failed: L3/L2 not on a common frame '
                        f'(max |Delta|={max_abs})'
                    )
                raise RuntimeError(
                    'Mandatory L3/L2 astrometric unify failed: coadds/L2 not on a '
                    f'common frame (max |Delta|={max_abs}"); refusing DOLPHOT inputs. '
                    f'See {qa_path}'
                )
        else:
            logger.info(
                'Astrometric frame OK group %s box %s '
                '(max |Delta|=%.3f"%s) -> %s',
                group_id,
                box_id,
                (unify.get('final_qa') or {}).get('max_abs_arcsec', 0.0),
                '; assume_aligned' if assume_aligned else '',
                unify.get('qa_path'),
            )
            best = next((r for r in results if r['status'] == 'ok'), ok[0])
            _write_group_frame_list(
                out,
                refimage=Path(best['output']),
                frames=sorted({Path(p) for p in all_box_frames}),
                instrument=best['instrument'],
                filt=best['filter'],
                group=group_id,
                box=box_id,
                merge=True,
            )
    return results


def _drizzle_box_worker(job: dict) -> dict:
    """Run one mosaic box (module-level for ProcessPoolExecutor)."""
    from st123.utils.logging import configure_worker_logging

    configure_worker_logging()
    ranked = [
        ((str(inst), str(filt)), [Path(p) for p in paths])
        for (inst, filt), paths in job['ranked']
    ]
    group_id = int(job['group_id'])
    box_id = job['box_id']
    try:
        results = _drizzle_box_groups(
            ranked,
            outdir=Path(job['outdir']),
            group_id=group_id,
            box_id=box_id,
            final_pixfrac=float(job['final_pixfrac']),
            final_scale=job.get('final_scale'),
            num_cores=int(job['num_cores']),
            all_box_frames=[Path(p) for p in job['all_box_frames']],
            preferred_abs_ref=[Path(p) for p in job.get('preferred_abs_ref') or []],
            raise_on_unify_fail=bool(job.get('raise_on_unify_fail', False)),
            output_wcs=job.get('output_wcs'),
            parallel_filters=bool(job.get('parallel_filters', True)),
            drop_outlier_visits=bool(job.get('drop_outlier_visits', False)),
            visit_coadds=bool(job.get('visit_coadds', False)),
            assume_aligned=bool(job.get('assume_aligned', True)),
        )
        return {
            'ok': True,
            'group_id': group_id,
            'box_id': box_id,
            'results': results,
            'error': None,
        }
    except Exception as exc:
        logger.exception(
            'HST drizzle failed for group %s box %s: %s',
            group_id,
            box_id,
            exc,
        )
        return {
            'ok': False,
            'group_id': group_id,
            'box_id': box_id,
            'results': [],
            'error': str(exc),
        }


def drizzle_project_boxed(
    plan: Any,
    *,
    instruments: Optional[Sequence[str]] = None,
    filters: Optional[Sequence[str]] = None,
    final_pixfrac: float = 0.8,
    final_scale: Optional[float] = None,
    num_cores: int = 4,
    raise_on_unify_fail: bool = False,
    parallel_boxes: bool = True,
    drop_outlier_visits: bool = False,
    visit_coadds: bool = False,
    assume_aligned: bool = True,
    require_coverage_ra: float | None = None,
    require_coverage_dec: float | None = None,
) -> list[dict]:
    """
    Drizzle HST frames into an existing :class:`~st123.stages.mosaic.mosaic.MosaicPlan`.

    Each plan box receives one ``coadd_{G}_{B}_{inst}_{filt}_{drc|drz}.fits``
    product (name from :meth:`InstrumentDataModel.l3_coadd_basename`).
    combining all visits in that instrument+filter pair (default). Optional
    *visit_coadds* writes one sidecar per HST visit/epoch even when
    *assume_aligned* (``--existing-box`` remosaics); *drop_outlier_visits* excludes
    cross-visit outliers from the primary stack. *require_coverage_ra/dec*
    keeps only frames whose footprint contains that sky point. *filters*
    restricts to named bands (e.g. ``F160W`` only).

    By default (*assume_aligned*=True) mosaic does not re-harmonize JHAT WCS
    before drizzle -- ``align`` owns registration. After coadds are written,
    L3 pairwise QA still runs; if a coadd is off the shared frame, mosaic
    runs :func:`unify_hst_astrometric_frame` (shift L2 + remosaic) instead of
    keeping the residual.

    When *parallel_boxes* is True and mosaic boxes have disjoint HST frame sets,
    boxes are drizzleed concurrently (``spawn`` process pool). Shared-frame boxes
    stay serial so in-place JHAT WCS edits cannot race. Filter coadds within a
    box are parallelized when boxes run serially (avoids nested process pools).
    """
    from st123.stages.alignment.hst_jhat import find_hst_l3_refcat

    boxes = list(getattr(plan, 'boxes', []) or [])
    if not boxes:
        logger.error('Mosaic plan has no boxes for HST drizzle')
        return []

    # Optional L3 refcat from first HST frame's jhat parent.
    sample = None
    for box in boxes:
        hst_frames = box.frames_for_mission('hst')
        if hst_frames:
            sample = Path(hst_frames[0]).parent
            break
    if sample is not None:
        l3_refcat = find_hst_l3_refcat(sample)
        if l3_refcat is not None:
            logger.info('L3 refcat available for secondary checks: %s', l3_refcat)

    allow = None
    if instruments:
        allow = {str(i).split('_')[0].lower() for i in instruments}

    box_jobs: list[dict] = []
    for box in boxes:
        hst_paths = [Path(p) for p in box.frames_for_mission('hst')]
        if not hst_paths:
            continue

        # Drop frames that do not actually overlap this box (stuck-split /
        # archival neighbors such as SN2006X ACS near SN2019ehk). Always
        # drizzle onto the shared stamp WCS (``stamp_wcs.fits`` / box WCS).
        box_wcs = None
        if getattr(box, 'wcs', None) is not None:
            from st123.stages.mosaic.mosaic import (
                ensure_box_stamp_wcs,
                filter_frames_covering_point,
                filter_frames_overlapping_box,
                stamp_sky_polygon,
            )

            box_wcs = ensure_box_stamp_wcs(box)
            before = len(hst_paths)
            kept = filter_frames_overlapping_box(
                hst_paths,
                stamp_sky_polygon(box_wcs),
                mosaic_wcs=None,
                min_overlap=0.01,
            )
            if require_coverage_ra is not None and require_coverage_dec is not None:
                kept = filter_frames_covering_point(
                    kept, float(require_coverage_ra), float(require_coverage_dec)
                )
            hst_paths = [Path(p) for p in kept]
            if before and not hst_paths:
                logger.warning(
                    'HST drizzle group %s box %s: all %d frame(s) outside '
                    'shared stamp / coverage point; skipping',
                    box.group_id,
                    box.box_id,
                    before,
                )
                continue
            if len(hst_paths) < before:
                logger.info(
                    'HST drizzle group %s box %s: kept %d/%d frame(s) '
                    'overlapping shared stamp',
                    box.group_id,
                    box.box_id,
                    len(hst_paths),
                    before,
                )

        groups = group_hst_frames(hst_paths)
        if allow is not None:
            groups = {
                key: vals for key, vals in groups.items() if key[0] in allow
            }
        if filters:
            allow_filt = {str(f).lower() for f in filters}
            groups = {
                key: vals
                for key, vals in groups.items()
                if str(key[1]).lower() in allow_filt
            }
        if not groups:
            continue
        ranked = sorted(
            groups.items(),
            key=lambda item: _total_exptime(item[1]),
            reverse=True,
        )
        preferred = sorted(box.outdir.glob('coadd_*_i2d.fits'))
        logger.info(
            'HST drizzle group %s box %s: %d frame(s), %d filter group(s) -> %s'
            '%s',
            box.group_id,
            box.box_id,
            len(hst_paths),
            len(ranked),
            box.outdir,
            ' [shared box WCS]' if box_wcs is not None else '',
        )
        box_jobs.append(
            {
                'ranked': [
                    ((inst, filt), [str(Path(p).resolve()) for p in paths])
                    for (inst, filt), paths in ranked
                ],
                'outdir': str(Path(box.outdir).resolve()),
                'group_id': int(box.group_id),
                'box_id': box.box_id,
                'final_pixfrac': float(final_pixfrac),
                'final_scale': final_scale,
                'num_cores': int(num_cores),
                'all_box_frames': [str(Path(p).resolve()) for p in hst_paths],
                'preferred_abs_ref': [str(Path(p).resolve()) for p in preferred],
                'output_wcs': _pickle_safe_output_wcs(box_wcs),
                'parallel_filters': True,
                'drop_outlier_visits': bool(drop_outlier_visits),
                'visit_coadds': bool(visit_coadds),
                'raise_on_unify_fail': bool(raise_on_unify_fail),
                'assume_aligned': bool(assume_aligned),
            }
        )

    if not box_jobs:
        return []

    frame_groups = [
        [Path(p) for p in job['all_box_frames']] for job in box_jobs
    ]
    can_parallel_boxes = (
        bool(parallel_boxes)
        and len(box_jobs) > 1
        and _frame_sets_disjoint(frame_groups)
    )
    all_results: list[dict] = []
    unify_errors: list[str] = []

    if can_parallel_boxes:
        n_workers, cores_per = _parallel_drizzle_budget(len(box_jobs), num_cores)
        # Nested process pools are not allowed - filters stay serial per box.
        for job in box_jobs:
            job['num_cores'] = cores_per
            job['parallel_filters'] = False
        logger.info(
            'Parallel mosaic boxes: %d box(es), workers=%d, cores/box=%d '
            '(filter coadds serial inside each box)',
            len(box_jobs),
            n_workers,
            cores_per,
        )
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = mp.get_context('spawn')
        ordered: list[dict | None] = [None] * len(box_jobs)
        from st123.utils.logging import configure_worker_logging

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=configure_worker_logging,
        ) as pool:
            future_map = {
                pool.submit(_drizzle_box_worker, job): i
                for i, job in enumerate(box_jobs)
            }
            for fut in as_completed(future_map):
                idx = future_map[fut]
                try:
                    ordered[idx] = fut.result()
                except Exception as exc:
                    job = box_jobs[idx]
                    ordered[idx] = {
                        'ok': False,
                        'group_id': job['group_id'],
                        'box_id': job['box_id'],
                        'results': [],
                        'error': f'{type(exc).__name__}: {exc}',
                    }
        box_outcomes = [o for o in ordered if o is not None]
    else:
        if len(box_jobs) > 1 and parallel_boxes and not _frame_sets_disjoint(
            frame_groups
        ):
            logger.info(
                'Mosaic boxes share HST frames - running boxes serially; '
                'parallelizing filter coadds inside each box'
            )
        for job in box_jobs:
            job['parallel_filters'] = True
            job['num_cores'] = int(num_cores)
        box_outcomes = [_drizzle_box_worker(job) for job in box_jobs]

    for outcome in box_outcomes:
        if not outcome.get('ok'):
            unify_errors.append(
                f"group {outcome.get('group_id')} box {outcome.get('box_id')}: "
                f"{outcome.get('error') or 'drizzle failed'}"
            )
            continue
        box_results = list(outcome.get('results') or [])
        all_results.extend(box_results)
        if any(
            r.get('status') == 'failed'
            and 'Astrometric unify failed' in str(r.get('error') or '')
            for r in box_results
        ):
            unify_errors.append(
                f"group {outcome.get('group_id')} box {outcome.get('box_id')}: "
                'astrometric unify failed'
            )

    if raise_on_unify_fail and unify_errors:
        raise RuntimeError(
            'Mandatory L3/L2 astrometric unify failed in one or more boxes: '
            + '; '.join(unify_errors)
        )
    return all_results


def drizzle_project(
    jhat_dir: PathLike,
    outdir: PathLike,
    *,
    final_pixfrac: float = 0.8,
    final_scale: Optional[float] = None,
    num_cores: int = 4,
    pattern: str = '*_jhat.fits',
    instruments: Optional[Sequence[str]] = None,
    filters: Optional[Sequence[str]] = None,
    nmax: int = 150,
    full_group: bool = False,
    footprint_weights: str = 'auto',
    drop_outlier_visits: bool = False,
    visit_coadds: bool = False,
    require_coverage_ra: float | None = None,
    require_coverage_dec: float | None = None,
) -> list[dict]:
    """
    Plan ``group_*/ref_*`` boxes from HST JHAT and drizzle into that layout.

    Parameters
    ----------
    jhat_dir : path-like
        Directory containing aligned ``*_jhat.fits`` (typically
        ``reduction/jhat_hst`` or ``jhat``).
    outdir : path-like
        ``reduction/reference`` or the reduction root containing ``reference/``.
    final_pixfrac, final_scale, num_cores
        Forwarded to :func:`drizzle_filter_group`.
    pattern : str, optional
        Glob for input frames (default ``*_jhat.fits``).
    instruments : sequence of str or None, optional
        If set, only drizzle these instruments (case-insensitive), e.g.
        ``('acs', 'wfc3')`` to exclude WFPC2 from dual-mode campaigns.
    filters : sequence of str or None, optional
        If set, only drizzle these filters (e.g. ``('F160W',)``).
    nmax, full_group, footprint_weights
        Forwarded to the shared mosaic box planner.

    Returns
    -------
    list of dict
        One record per ``(box, instrument, filter)`` with keys ``instrument``,
        ``filter``, ``frames``, ``output``, ``status``, and optional ``error``.
    """
    from st123.stages.mosaic.mosaic import plan_mosaic_boxes

    jhat = Path(jhat_dir)
    out = Path(outdir)
    if out.name == 'reference':
        base = out.parent
    else:
        base = out
        out = base / 'reference'
    out.mkdir(parents=True, exist_ok=True)

    frames = _collect_hst_jhat_frames(jhat, pattern=pattern)
    if not frames:
        logger.error('No input FITS under %s', jhat)
        return []

    if instruments:
        allow = {str(i).split('_')[0].lower() for i in instruments}
        frames = [
            p for p in frames if get_instrument(p).split('_')[0].lower() in allow
        ]
        logger.info(
            'HST instrument filter %s: %d frame(s) under %s',
            sorted(allow),
            len(frames),
            jhat,
        )
        if not frames:
            logger.error(
                'No HST frames left after --instruments %s under %s',
                sorted(allow),
                jhat,
            )
            return []

    plan = plan_mosaic_boxes(
        base,
        frames,
        nmax=nmax,
        full_group=full_group,
        footprint_weights=footprint_weights,
        verbose=True,
    )
    return drizzle_project_boxed(
        plan,
        instruments=instruments,
        filters=filters,
        final_pixfrac=final_pixfrac,
        final_scale=final_scale,
        num_cores=num_cores,
        raise_on_unify_fail=False,
        assume_aligned=True,
        drop_outlier_visits=drop_outlier_visits,
        visit_coadds=visit_coadds,
        require_coverage_ra=require_coverage_ra,
        require_coverage_dec=require_coverage_dec,
    )
