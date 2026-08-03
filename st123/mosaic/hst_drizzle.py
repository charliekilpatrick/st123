"""
HST AstroDrizzle mosaics from JHAT-aligned frames.

Groups ``*_jhat.fits`` (or calibrated MEFs) by instrument + filter, runs
:func:`drizzlepac.astrodrizzle.AstroDrizzle` per group, and writes coadds under
``reduction/reference/`` with a ``dolphot_frames.txt`` manifest per product.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Iterable, MutableMapping, Optional, Sequence, Union

from astropy.io import fits

from st123.utils.helpers import get_filter, get_instrument
from st123.utils.settings import (
    WFPC2_BAD_COL_FRAC,
    WFPC2_BAD_GROW_PIX,
    WFPC2_OVERSCAN_DQ_BIT,
    WFPC2_OVERSCAN_EDGE_PIX,
    WFPC2_OVERSCAN_LEFT_EXTRA,
    WFPC2_SCI_FLOOR,
    hst_driz_bits,
    hst_drizzle_defaults,
)

PathLike = Union[str, os.PathLike]

logger = logging.getLogger(__name__)

_HST_INSTS = frozenset({'acs', 'wfc3', 'wfpc2'})


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
    for it beside ``c0m``. Zero DQ means “all pixels good”.
    """
    import numpy as np

    with fits.open(c0m_path, memmap=True) as hdul:
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


def mask_wfpc2_overscan(
    c0m_path: Path,
    c1m_path: Path,
    *,
    edge: int = WFPC2_OVERSCAN_EDGE_PIX,
    left_extra: int = WFPC2_OVERSCAN_LEFT_EXTRA,
    dq_bit: int = WFPC2_OVERSCAN_DQ_BIT,
    sci_floor: float = WFPC2_SCI_FLOOR,
    bad_grow: int = WFPC2_BAD_GROW_PIX,
    bad_col_frac: float = WFPC2_BAD_COL_FRAC,
) -> dict:
    """
    Flag WFPC2 chip overscan / bad-edge pixels in ``c1m`` and blank extreme SCI.

    Calibrated ``*_c0m`` chips often retain a left-edge strip of large negative
    values that project into AstroDrizzle coadds. Those pixels are marked with
    *dq_bit* (excluded by ``final_bits=1032``) and SCI below *sci_floor* is
    set to 0 so they cannot bias sky / CR rejection.

    Extra left-edge width, whole-column kills (left half, high negative
    fraction), and a small morphological grow catch bleed that a uniform
    border miss.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation

    if edge < 0 or left_extra < 0 or bad_grow < 0:
        raise ValueError('edge, left_extra, and bad_grow must be >= 0')
    n_edge = 0
    n_floor = 0
    n_cols = 0
    n_grown = 0
    with fits.open(c0m_path, mode='update') as sci_hdul, fits.open(
        c1m_path, mode='update'
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

            bad = border | floor | col_mask
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
        sci_hdul[0].header['ST123OVSC'] = (
            True,
            (
                f'WFPC2 overscan/edge masked edge={edge} '
                f'left+={left_extra} floor={sci_floor} grow={bad_grow}'
            ),
        )
        sci_hdul.flush()
        dq_hdul.flush()
    logger.info(
        'WFPC2 overscan mask %s: edge_pix=%d floor_pix=%d '
        'bad_cols=%d grown=%d (edge=%d left+=%d bit=%d)',
        c0m_path.name,
        n_edge,
        n_floor,
        n_cols,
        n_grown,
        edge,
        left_extra,
        dq_bit,
    )
    return {
        'path': str(c0m_path),
        'n_edge': n_edge,
        'n_floor': n_floor,
        'n_cols': n_cols,
        'n_grown': n_grown,
    }


def _finalize_drizzle_product(
    product: Path,
    desired: Path,
    *,
    output_stem: Path,
) -> Path:
    """
    Move AstroDrizzle output to *desired* and drop WFPC2 ``_drw`` duplicates.

    DrizzlePac forces ``_drw.fits`` for WFPC2 when given a bare stem; we always
    keep a single ``_drz.fits`` (or ``_drc.fits``) product.
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
            with fits.open(jhat, memmap=True) as jhat_hdul, fits.open(
                dst, mode='update'
            ) as dst_hdul:
                _copy_sci_wcs(jhat_hdul, dst_hdul)
                dst_hdul.flush()
    else:
        dst = scratch_dir / src.name
        shutil.copy2(src, dst)

    if run_cosmic_clean and any(
        dst.name.endswith(suf) for suf in ('_flc.fits', '_flt.fits', '_c0m.fits')
    ):
        already = False
        try:
            already = bool(fits.getheader(dst, ext=0).get('ST123CR'))
        except Exception:
            already = False
        if already:
            logger.info('Skipping astroscrappy for %s (ST123CR already set)', dst.name)
        else:
            try:
                from st123.photometry.cosmic import run_cosmic

                run_cosmic(
                    dst,
                    instrument=instrument.split('_')[0].lower(),
                    add_crmask=True,
                    inplace=True,
                )
            except Exception as exc:
                logger.warning('astroscrappy failed for %s: %s', dst.name, exc)

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

    with fits.open(dst, mode='update') as hdul:
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
        # Ensure SCI extensions have NGOODPIX when missing (rare jhat-only path).
        for hdu in hdul:
            if getattr(hdu, 'name', '') == 'SCI' and hdu.data is not None:
                if 'NGOODPIX' not in hdu.header:
                    import numpy as np

                    data = np.asarray(hdu.data)
                    hdu.header['NGOODPIX'] = int(np.isfinite(data).sum())
        hdul.flush()
    return dst


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
        Mapping ``(instrument, filter)`` → sorted list of :class:`~pathlib.Path`.
        Keys are lowercase (e.g. ``('wfc3', 'f336w')``).
    """
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in paths:
        p = Path(path)
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


def _drizzle_suffix(instrument: str) -> str:
    """WFPC2 → ``drz``; ACS/WFC3 → ``drc``."""
    return 'drz' if instrument.lower() == 'wfpc2' else 'drc'


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


def _write_group_frame_list(
    outdir: Path,
    *,
    refimage: Path,
    frames: Sequence[Path],
    instrument: str,
    filt: str,
) -> Path:
    """Write ``dolphot_frames.txt`` (and a filter-tagged copy) for one coadd."""
    outdir.mkdir(parents=True, exist_ok=True)
    body_lines = [
        f'# instrument={instrument} filter={filt}\n',
        f'# ref {refimage.resolve()}\n',
    ]
    body_lines.extend(f'{Path(p).resolve()}\n' for p in frames)
    primary = outdir / 'dolphot_frames.txt'
    tagged = outdir / f'dolphot_frames_{instrument}_{filt}.txt'
    text = ''.join(body_lines)
    # Prefer deepest/longest coadd as the primary manifest when multiple exist:
    # callers overwrite primary each group; drizzle_project sets the preferred one last.
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
    with fits.open(p, mode='update', memmap=False) as hdul:
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
            hdul[0].header['ST123PSKY'] = (
                True,
                'st123: per-chip sky subtracted before drizzle',
            )
        hdul.flush()
    return report


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
) -> Path:
    """
    Drizzle one instrument/filter group with AstroDrizzle.

    Parameters
    ----------
    images : sequence of path-like
        Input FITS paths (prefer aligned ``*_jhat.fits`` MEFs).
    output_path : path-like
        Desired coadd path, e.g. ``…/coadd_wfc3_f336w_drc.fits``. The
        ``_drc`` / ``_drz`` suffix is stripped before calling AstroDrizzle.
    final_pixfrac : float, optional
        AstroDrizzle ``final_pixfrac`` (default 0.8).
    final_scale : float or None, optional
        Output pixel scale in arcsec. ``None`` uses an instrument default.
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
    suffix = _drizzle_suffix(inst)
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

    scratch_dir = out.parent / f'.drizzle_scratch_{output_stem.name}'
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir, ignore_errors=True)
    staged = [
        str(
            _stage_drizzle_input(
                p, scratch_dir, instrument=inst, filt=filt
            )
        )
        for p in src_imgs
    ]

    dd = dict(hst_drizzle_defaults)
    # Instrument-specific good-DQ bits (hst123 detector_defaults).
    bits_key = inst
    is_uvis = False
    if inst == 'wfc3':
        # Prefer IR bits when any input looks like WFC3/IR.
        bits_key = 'wfc3'
        is_uvis = True
        for p in src_imgs:
            try:
                aper = str(fits.getheader(p, ext=0).get('APERTURE') or '').upper()
                if aper.startswith('IR'):
                    bits_key = 'wfc3_ir'
                    is_uvis = False
                    break
            except Exception:
                pass
    elif inst == 'acs':
        is_uvis = True  # two-chip WFC; same pedestal/gap issue
    driz_bits = int(hst_driz_bits.get(bits_key, hst_driz_bits.get(inst, 0)))

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

    kwargs: MutableMapping[str, object] = {
        'output': str(output_stem),
        'runfile': str(output_stem) + '_astrodrizzle.log',
        'context': True,
        'build': build,
        'num_cores': int(num_cores or dd.get('num_cores', 4)),
        'preserve': False,
        'clean': clean,
        # When per-chip sky was already removed, disable AstroDrizzle skymatch
        # (it applies one sky to all chips and reintroduces gap jumps).
        'skysub': not do_per_chip,
        'skymethod': 'globalmin+match',
        'skystat': 'mode',
        'updatewcs': False,
        'driz_sep_pixfrac': float(dd.get('driz_sep_pixfrac', final_pixfrac)),
        'combine_maskpt': float(dd.get('combine_maskpt', 0.2)),
        'combine_type': 'minmed' if len(staged) >= 4 else 'median',
        'combine_nsigma': str(dd.get('combine_nsigma', '4 3')),
        'driz_cr': True,
        'driz_cr_corr': True,
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
    if scale is not None:
        kwargs['final_scale'] = float(scale)

    # DrizzlePac builds outroot via ``'_'.join(output.split('_')[:-1]).lower()``.
    # Absolute paths with underscores in the *basename* (e.g. coadd_wfc3_f336w)
    # then lowercase parent dirs (HST → hst) and break staticMask writes.
    # Always chdir to the output dir and pass basename + relative inputs.
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
    input_rel = [os.path.relpath(p, out.parent) for p in staged]
    kwargs['output'] = desired.name
    kwargs['runfile'] = f'{output_base}_astrodrizzle.log'

    logger.info(
        'AstroDrizzle %d frame(s) → %s (pixfrac=%.2f scale=%s cores=%d)',
        len(staged),
        desired.name,
        final_pixfrac,
        scale,
        kwargs['num_cores'],
    )
    cwd = Path.cwd()
    try:
        os.chdir(out.parent)
        with _suppress_drizzlepac_dgeo_prompt():
            astrodrizzle.AstroDrizzle(input_rel, **kwargs)
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

    return _finalize_drizzle_product(
        product, desired, output_stem=output_stem
    )


def _total_exptime(paths: Sequence[Path]) -> float:
    total = 0.0
    for p in paths:
        try:
            total += float(fits.getval(p, 'EXPTIME'))
        except Exception:
            try:
                total += float(fits.getval(p, 'EFFEXPTM'))
            except Exception:
                pass
    return total


def _pick_coadd_abs_ref(coadds: Sequence[Path]) -> Path | None:
    """Prefer WFC3 F625, then WFPC2 F814, then largest coadd."""
    existing = [Path(p) for p in coadds if Path(p).is_file()]
    if not existing:
        return None

    def score(p: Path) -> tuple[int, int]:
        name = p.name.lower()
        pref = 0
        if 'wfc3' in name and 'f625' in name:
            pref = 300
        elif 'wfpc2' in name and 'f814' in name:
            pref = 200
        elif 'wfc3' in name:
            pref = 100
        elif 'f814' in name:
            pref = 90
        try:
            size = int(p.stat().st_size)
        except OSError:
            size = 0
        return (pref, size)

    return max(existing, key=score)


def apply_sky_shift_to_fits(
    path: PathLike,
    dra_deg: float,
    ddec_deg: float,
    *,
    comment: str = 'st123: L3-unified abs shift',
) -> int:
    """Add (ΔRA, ΔDec) degrees to every SCI CRVAL in *path*. Returns n updated."""
    from st123.alignment.hst_jhat import apply_sky_translation_to_sci, _sci_hdu_indices

    p = Path(path)
    with fits.open(p, mode='update', memmap=False) as hdul:
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
            hdul[0].header['ST123L3UN'] = (
                True,
                'st123: L3-unified absolute shift applied',
            )
            hdul[0].header['ST123L3RA'] = (
                float(dra_deg) * 3600.0 * float(np.cos(np.radians(dec0))),
                '[arcsec] L3-unify dRA cos(Dec)',
            )
            hdul[0].header['ST123L3DE'] = (
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
) -> dict:
    """
    Mandatory post-drizzle pass: put all L3 coadds (and their L2 inputs) on one sky frame.

    For each coadd, measure the 2-D histogram offset vs the absolute-reference
    coadd (WFC3 F625 preferred, else WFPC2 F814). When |Δ| exceeds the L3 QA
    limit, apply −Δ to every L2 frame in that group **and** to the coadd WCS,
    then optionally remosaic the group so the drizzle product matches the L2
    frame used by DOLPHOT.

    Iterates up to *max_iters* times. Returns a report dict with ``ok``,
    ``abs_ref``, ``iterations``, and per-group corrections.
    """
    import json

    import numpy as np

    from st123.alignment.hst_jhat import (
        HST_ABS_OFFSET_MAX_ARCSEC,
        HST_L3_ALIGN_MAX_ARCSEC,
        measure_hst_sky_offset_2dhist,
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
    if len(records) < 2:
        report['ok'] = True
        report['final_qa'] = {'ok': True, 'n_coadds': len(records), 'max_abs_arcsec': 0.0}
        return report

    abs_ref = _pick_coadd_abs_ref([Path(r['output']) for r in records])
    if abs_ref is None:
        report['error'] = 'no coadd abs_ref available'
        return report
    report['abs_ref'] = str(abs_ref)
    logger.info(
        'L3/L2 astrometric unify: abs_ref=%s (limit %.3f", search %.1f")',
        abs_ref.name,
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
                    'L3 unify: %s already on frame (|Δ|=%.3f")',
                    coadd.name,
                    off['abs_arcsec'],
                )
                iter_rows.append(row)
                continue

            # img−ref → apply −Δ to L2 + coadd so they move onto abs_ref.
            dra_deg = -float(off['dra_deg'])
            ddec_deg = -float(off['ddec_deg'])
            dra_as = -float(off['dra_arcsec'])
            ddec_as = -float(off['ddec_arcsec'])
            logger.info(
                'L3 unify iter%d: %s → %s apply dRA=%+.3f" dDec=%+.3f" '
                '(was |Δ|=%.3f", peak=%s)',
                it + 1,
                coadd.name,
                abs_ref.name,
                dra_as,
                ddec_as,
                off['abs_arcsec'],
                off.get('peak_count'),
            )

            n_l2 = 0
            for fp in frames:
                if fp.is_file():
                    n_l2 += apply_sky_shift_to_fits(
                        fp, dra_deg, ddec_deg, comment='st123: L3-unify to L2'
                    )
            apply_sky_shift_to_fits(
                coadd, dra_deg, ddec_deg, comment='st123: L3-unify to coadd'
            )

            # Common CRVAL shift preserves relatives; if internal QA still
            # fails (pre-existing drift / measurement), re-harmonize in place.
            if len(frames) >= 2:
                from st123.alignment.hst_jhat import harmonize_hst_group_wcs

                qa = validate_hst_group_internal_alignment(
                    frames, max_coherent_arcsec=HST_INTERNAL_ALIGN_MAX_ARCSEC
                )
                if not qa.get('ok'):
                    logger.warning(
                        'L3 unify: internal L2 QA soft-fail for %s/%s after '
                        'shift (max |Δ|=%.3f"); re-harmonizing',
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
                        '(max |Δ|=%.3f"); remosaic may ghost',
                        rec.get('instrument'),
                        rec.get('filter'),
                        qa.get('max_abs_arcsec', 0.0),
                    )
                    row['error'] = (
                        f'internal L2 QA failed after L3 unify '
                        f'(max |Δ|={qa.get("max_abs_arcsec")})'
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
        abs_ref = _pick_coadd_abs_ref(coadd_paths) or abs_ref
        report['abs_ref'] = str(abs_ref)
        final_qa = validate_hst_coadds_alignment(
            coadd_paths, max_coherent_arcsec=limit
        )
        report['final_qa'] = final_qa
        if final_qa.get('ok') and not internal_bad:
            report['ok'] = True
            logger.info(
                'L3/L2 astrometric unify OK after iter %d (max |Δ|=%.3f")',
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
                'L3/L2 unify: residual remains (max |Δ|=%.3f") but no shift applied',
                final_qa.get('max_abs_arcsec', 0.0),
            )
            break
        logger.warning(
            'L3/L2 unify iter %d still failing (max |Δ|=%.3f"); continuing',
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


def drizzle_project(
    jhat_dir: PathLike,
    outdir: PathLike,
    *,
    final_pixfrac: float = 0.8,
    final_scale: Optional[float] = None,
    num_cores: int = 4,
    pattern: str = '*_jhat.fits',
) -> list[dict]:
    """
    Drizzle every ``(instrument, filter)`` group under *jhat_dir*.

    Parameters
    ----------
    jhat_dir : path-like
        Directory containing aligned ``*_jhat.fits`` (typically
        ``reduction/jhat``).
    outdir : path-like
        Output directory for coadds (typically ``reduction/reference``).
    final_pixfrac, final_scale, num_cores
        Forwarded to :func:`drizzle_filter_group`.
    pattern : str, optional
        Glob for input frames (default ``*_jhat.fits``).

    Returns
    -------
    list of dict
        One record per group with keys ``instrument``, ``filter``, ``frames``,
        ``output``, ``status``, and optional ``error``.
    """
    jhat = Path(jhat_dir)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    frames = sorted(jhat.glob(pattern))
    if not frames:
        # Fall back to any FITS if jhat naming differs.
        frames = sorted(
            p for p in jhat.glob('*.fits')
            if not p.name.endswith(('.sky.fits',))
        )
    # Never drizzle level-3 reference products parked beside science JHAT outputs.
    frames = [
        p for p in frames
        if not p.name.lower().startswith('coadd_')
        and 'l3_ref' not in p.parts
    ]
    if not frames:
        logger.error('No input FITS under %s', jhat)
        return []

    groups = group_hst_frames(frames)
    if not groups:
        logger.error('No HST instrument/filter groups under %s', jhat)
        return []

    # Prefer writing the deepest group's manifest last as primary dolphot_frames.txt.
    ranked = sorted(
        groups.items(),
        key=lambda item: _total_exptime(item[1]),
        reverse=True,
    )

    from st123.alignment.hst_jhat import (
        HST_INTERNAL_ALIGN_MAX_ARCSEC,
        HST_L3_ALIGN_MAX_ARCSEC,
        find_hst_l3_refcat,
        harmonize_hst_group_wcs,
        validate_hst_group_internal_alignment,
    )

    # Pre-drizzle: enforce internal (relative) L2 alignment only. Absolute
    # cross-filter ties are handled by the mandatory post-drizzle unify pass,
    # which can use finished coadds and propagates shifts back into L2.
    l3_refcat = find_hst_l3_refcat(jhat)
    if l3_refcat is not None:
        logger.info('L3 refcat available for secondary checks: %s', l3_refcat)

    results: list[dict] = []
    for (inst, filt), imgs in ranked:
        suffix = _drizzle_suffix(inst)
        coadd_name = f'coadd_{inst}_{filt}_{suffix}.fits'
        coadd_path = out / coadd_name
        record: dict = {
            'instrument': inst,
            'filter': filt,
            'frames': [str(p) for p in imgs],
            'output': str(coadd_path),
            'status': 'pending',
            'error': None,
            'internal_qa': None,
            'harmonize': None,
        }
        try:
            harm = harmonize_hst_group_wcs(
                imgs,
                max_internal_arcsec=HST_INTERNAL_ALIGN_MAX_ARCSEC,
                abs_ref=None,
                force=False,
            )
            record['harmonize'] = {
                'ok': bool(harm.get('ok')),
                'anchor': harm.get('anchor'),
                'method': harm.get('method'),
                'pre_max_abs_arcsec': (harm.get('pre') or {}).get('max_abs_arcsec'),
                'post_max_abs_arcsec': (harm.get('post') or {}).get('max_abs_arcsec'),
            }
            qa = validate_hst_group_internal_alignment(
                imgs,
                max_coherent_arcsec=HST_INTERNAL_ALIGN_MAX_ARCSEC,
            )
            record['internal_qa'] = {
                'ok': bool(qa.get('ok')),
                'max_abs_arcsec': qa.get('max_abs_arcsec'),
                'failed_pairs': qa.get('failed_pairs'),
            }
            if not qa.get('ok'):
                raise RuntimeError(
                    f'Internal L2 alignment QA failed for {inst}/{filt}: '
                    f'max |Δ|={qa.get("max_abs_arcsec"):.3f}" '
                    f'(limit {HST_INTERNAL_ALIGN_MAX_ARCSEC:.3f}"); '
                    f'failed_pairs={qa.get("failed_pairs")}'
                )

            product = drizzle_filter_group(
                imgs,
                coadd_path,
                final_pixfrac=final_pixfrac,
                final_scale=final_scale,
                num_cores=num_cores,
                instrument=inst,
            )
            record['output'] = str(product)
            record['status'] = 'ok'
            _write_group_frame_list(
                out,
                refimage=product,
                frames=imgs,
                instrument=inst,
                filt=filt,
            )
            logger.info('Wrote %s (%d frames)', product, len(imgs))
        except Exception as exc:
            record['status'] = 'failed'
            record['error'] = str(exc)
            logger.exception(
                'Drizzle failed for %s/%s (%d frames): %s',
                inst,
                filt,
                len(imgs),
                exc,
            )
        results.append(record)

    # Rewrite primary manifest with the deepest successful coadd as reference
    # and *all* jhat frames (for mixed-instrument DOLPHOT staging).
    ok = [r for r in results if r['status'] == 'ok']
    if ok:
        best = ok[0]
        all_frames = [Path(p) for imgs in groups.values() for p in imgs]
        _write_group_frame_list(
            out,
            refimage=Path(best['output']),
            frames=sorted(set(all_frames)),
            instrument=best['instrument'],
            filt=best['filter'],
        )

        # Mandatory: unify all L3 coadds onto one frame and propagate into L2
        # so DOLPHOT sees the same astrometry as the coadds.
        unify = unify_hst_astrometric_frame(
            ok,
            outdir=out,
            max_residual_arcsec=HST_L3_ALIGN_MAX_ARCSEC,
            num_cores=num_cores,
            final_pixfrac=final_pixfrac,
            final_scale=final_scale,
            remosaic=True,
        )
        for r in results:
            r['astrometric_unify'] = {
                'ok': bool(unify.get('ok')),
                'abs_ref': unify.get('abs_ref'),
                'iterations': unify.get('iterations'),
                'max_abs_arcsec': (unify.get('final_qa') or {}).get('max_abs_arcsec'),
                'qa_path': unify.get('qa_path'),
            }
            r['l3_qa'] = unify.get('final_qa')

        if not unify.get('ok'):
            max_abs = (unify.get('final_qa') or {}).get('max_abs_arcsec')
            qa_path = unify.get('qa_path')
            logger.error(
                'Astrometric unify FAILED (max |Δ|=%s); '
                'L2/L3 not safe for DOLPHOT - see %s',
                max_abs,
                qa_path,
            )
            for r in ok:
                r['status'] = 'failed'
                r['error'] = (
                    'Astrometric unify failed: L3/L2 not on a common frame '
                    f'(max |Δ|={max_abs})'
                )
            raise RuntimeError(
                'Mandatory L3/L2 astrometric unify failed: coadds/L2 not on a '
                f'common frame (max |Δ|={max_abs}"); refusing DOLPHOT inputs. '
                f'See {qa_path}'
            )
        else:
            logger.info(
                'Astrometric unify OK (max |Δ|=%.3f", abs_ref=%s) → %s',
                (unify.get('final_qa') or {}).get('max_abs_arcsec', 0.0),
                Path(str(unify.get('abs_ref') or '')).name,
                unify.get('qa_path'),
            )
            best = next((r for r in results if r['status'] == 'ok'), ok[0])
            _write_group_frame_list(
                out,
                refimage=Path(best['output']),
                frames=sorted(set(all_frames)),
                instrument=best['instrument'],
                filt=best['filter'],
            )
    return results
