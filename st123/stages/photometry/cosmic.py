"""
LAcosmic-style cosmic-ray cleaning for HST frames (astroscrappy).

Mirrors ``hst123.run_cosmic``: replace SCI with the cleaned array and optionally
flag CR pixels in the DQ extension (FLC/FLT) or WFPC2 ``*_c1m.fits`` sidecar
with :data:`st123.utils.settings.HST_CR_DQ_BIT`.

Writes are done with ``as_datamodel(...).open(mode='update')`` so trailing
``HeaderletHDU`` blocks on WFC3/ACS FLCs are preserved (a full ``writeto``
rewrite truncates them and breaks AstroDrizzle ``drizCR``).

WFC3/IR is **not** lacosmic-cleaned: undersampled IR PSFs are often false-flagged
as cosmics, and bit 4096 is outside ``final_bits=576``, which punches weight
holes at star cores in F110W/F160W coadds.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
from st123.datamodels import as_datamodel

from st123.utils.settings import HST_CR_DQ_BIT, hst_crpars

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


def is_wfc3_ir_frame(image) -> bool:
    """
    Return True when *image* is a WFC3/IR calibrated or JHAT product.

    Uses ``DETECTOR``, ``APERTURE``, and ``PHOTMODE`` (same cues as DOLPHOT
    ``classify_image_kind``).
    """
    from st123.datamodels import WFC3IRDataModel
    from st123.datamodels.instrument import as_datamodel

    model = as_datamodel(image)
    if model.instrument in ('ACS', 'WFPC2', 'NIRCAM', 'MIRI'):
        return False
    return isinstance(model, WFC3IRDataModel)


def should_skip_cosmic(image, *, instrument: Optional[str] = None) -> bool:
    """True when lacosmic/astroscrappy must not run (WFC3/IR)."""
    inst = (instrument or '').split('_')[0].lower()
    if inst in ('acs', 'wfpc2'):
        return False
    return is_wfc3_ir_frame(image)


def resolve_crpars(
    instrument: str,
    overrides: Optional[Mapping[str, float]] = None,
) -> dict[str, float]:
    """Return astroscrappy parameters for *instrument* (acs/wfc3/wfpc2)."""
    key = instrument.split('_')[0].lower()
    if key not in hst_crpars:
        raise ValueError(
            f'No hst_crpars for instrument={instrument!r}; '
            f'expected one of {sorted(hst_crpars)}'
        )
    params = dict(hst_crpars[key])
    if overrides:
        params.update(overrides)
    return params


def wfpc2_c1m_path(c0m) -> Path:
    """Sibling ``*_c1m.fits`` path for a WFPC2 ``*_c0m.fits`` science datamodel."""
    from st123.datamodels.hst.wfpc2 import WFPC2DataModel
    from st123.datamodels.instrument import InstrumentDataModel

    if isinstance(c0m, WFPC2DataModel):
        return c0m.dq_path
    if isinstance(c0m, InstrumentDataModel):
        return WFPC2DataModel.c1m_path_for(c0m.path)
    return WFPC2DataModel.c1m_path_for(c0m)


def clear_st123_cr_flags(
    image: PathLike,
    *,
    clear_sci_keyword: bool = True,
) -> dict:
    """
    Remove st123 lacosmic DQ flags (bit :data:`HST_CR_DQ_BIT`) and ``ST123CR``.

    Does **not** restore pre-lacosmic SCI values (those were overwritten in
    place). Clearing DQ is enough for AstroDrizzle to include those pixels
    again (IR ``final_bits`` excludes 4096).

    Parameters
    ----------
    image : path-like
        ``*_flc`` / ``*_flt`` / ``*_c0m`` (also clears sibling ``*_c1m``).
    clear_sci_keyword : bool, optional
        Delete primary ``ST123CR`` when present.

    Returns
    -------
    dict
        ``path``, ``n_dq_cleared``, ``st123cr_cleared``, ``c1m`` (if any).
    """
    from st123.datamodels.hst.wfpc2 import WFPC2DataModel
    from st123.datamodels.instrument import as_datamodel, path_of

    model = as_datamodel(image)
    src = path_of(model).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    n_cleared = 0
    st123cr_cleared = False
    c1m_path: Optional[Path] = None
    name_l = src.name.lower()
    bit = int(HST_CR_DQ_BIT)

    with as_datamodel(src).open(mode='update') as hdul:
        if clear_sci_keyword and 'ST123CR' in hdul[0].header:
            del hdul[0].header['ST123CR']
            st123cr_cleared = True
        for hdu in hdul:
            if getattr(hdu, 'name', '') != 'DQ' or hdu.data is None:
                continue
            dq = np.asarray(hdu.data)
            mask = (dq.astype(np.int64) & bit) != 0
            n = int(np.count_nonzero(mask))
            if n:
                hdu.data = (dq.astype(np.int64) & ~bit).astype(dq.dtype, copy=False)
                n_cleared += n
        if isinstance(model, WFPC2DataModel) or 'c0m' in name_l:
            c1m_cand = wfpc2_c1m_path(model)
            if c1m_cand.is_file():
                c1m_path = c1m_cand
                with model.open_dq(mode='update') as maskhdu:
                    for hdu in maskhdu:
                        if hdu.data is None:
                            continue
                        dq = np.asarray(hdu.data)
                        if dq.ndim != 2:
                            continue
                        mask = (dq.astype(np.int64) & bit) != 0
                        n = int(np.count_nonzero(mask))
                        if n:
                            hdu.data = (dq.astype(np.int64) & ~bit).astype(
                                dq.dtype, copy=False
                            )
                            n_cleared += n
                    maskhdu.flush()
        hdul.flush()

    summary = {
        'path': str(src),
        'n_dq_cleared': n_cleared,
        'st123cr_cleared': st123cr_cleared,
        'c1m': str(c1m_path) if c1m_path is not None else None,
    }
    logger.info(
        'Cleared ST123 CR flags on %s (%d DQ pix, ST123CR=%s)',
        src.name,
        n_cleared,
        st123cr_cleared,
    )
    return summary


def clear_st123_cr_directory(
    raw_dir: PathLike,
    *,
    patterns: Sequence[str] = ('*flc.fits', '*flt.fits', '*c0m.fits'),
    ir_only: bool = True,
) -> list[dict]:
    """
    One-shot cleanup of lacosmic DQ / ``ST123CR`` under *raw_dir*.

    When *ir_only* is True (default), only WFC3/IR frames are touched -- UVIS/ACS
    keep legitimate CR masks.
    """
    root = Path(raw_dir).expanduser().resolve()
    results: list[dict] = []
    seen: set[str] = set()
    for pat in patterns:
        for path in sorted(root.glob(pat)):
            if path.name.lower().endswith('c1m.fits'):
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            if ir_only and not is_wfc3_ir_frame(path):
                continue
            try:
                results.append(clear_st123_cr_flags(path))
            except Exception as exc:
                logger.error('CR-flag clear failed for %s: %s', path.name, exc)
                results.append(
                    {
                        'path': str(path),
                        'error': f'{type(exc).__name__}: {exc}',
                        'n_dq_cleared': 0,
                        'st123cr_cleared': False,
                    }
                )
    return results


def run_cosmic(
    image: PathLike,
    *,
    instrument: Optional[str] = None,
    add_crmask: bool = True,
    output: Optional[PathLike] = None,
    crpars: Optional[Mapping[str, float]] = None,
    inplace: bool = True,
) -> dict:
    """
    Clean cosmic rays in SCI extensions with astroscrappy.

    Parameters
    ----------
    image : path-like
        Input FITS (``*_flc`` / ``*_flt`` / ``*_c0m``).
    instrument : str or None, optional
        ``acs`` / ``wfc3`` / ``wfpc2``. Inferred from the header when omitted.
    add_crmask : bool, optional
        When True, set CR pixels to :data:`HST_CR_DQ_BIT` in DQ / ``c1m``.
    output : path-like or None, optional
        Output science path (default: overwrite *image* when *inplace*).
    crpars : mapping or None, optional
        Override keys from :data:`hst_crpars`.
    inplace : bool, optional
        If True and *output* is None, overwrite *image*.

    Returns
    -------
    dict
        Summary with ``path``, ``n_sci``, ``n_cr_pixels``, ``c1m`` (if any).
        WFC3/IR returns ``skipped=True`` without modifying the file.
    """
    from st123.datamodels.instrument import as_datamodel, path_of

    model = as_datamodel(image)
    src = path_of(model).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    inst = (instrument or model.instrument_name).split('_')[0].lower()
    if should_skip_cosmic(model, instrument=inst):
        logger.info(
            'Skipping astroscrappy for WFC3/IR %s (false-flags star cores)',
            src.name,
        )
        return {
            'path': str(src),
            'instrument': 'wfc3_ir',
            'n_sci': 0,
            'n_cr_pixels': 0,
            'c1m': None,
            'skipped': True,
            'reason': 'wfc3_ir',
        }

    try:
        from astroscrappy import detect_cosmics
    except ImportError as exc:
        raise ImportError(
            'run_cosmic requires astroscrappy (see requirements.txt)'
        ) from exc

    params = resolve_crpars(inst, crpars)
    out = Path(output).expanduser().resolve() if output is not None else src
    if out != src:
        shutil.copy2(src, out)
    elif not inplace and output is None:
        out = src.with_name(src.stem + '.crclean.fits')
        shutil.copy2(src, out)

    logger.info('Cleaning cosmic rays in %s (%s)', src.name, inst)
    n_cr = 0
    n_sci = 0
    c1m_updated: Optional[Path] = None
    name_l = out.name.lower()

    # In-place update preserves HDRLET / other non-standard extensions.
    with as_datamodel(out).open(mode='update') as hdul:
        for i, hdu in enumerate(hdul):
            if getattr(hdu, 'name', '') != 'SCI' or hdu.data is None:
                continue
            n_sci += 1
            data = np.asarray(hdu.data, dtype=np.float32)
            inmask = np.zeros(data.shape, dtype=bool)
            crmask, crclean = detect_cosmics(
                data.copy().astype('<f4'),
                inmask=inmask,
                readnoise=float(params['rdnoise']),
                gain=float(params['gain']),
                satlevel=float(params['saturate']),
                sigclip=float(params['sig_clip']),
                sigfrac=float(params['sig_frac']),
                objlim=float(params['obj_lim']),
            )
            # Assign into existing array to keep dtype/shape intact.
            hdu.data[:, :] = np.asarray(crclean, dtype=hdu.data.dtype)
            n_cr += int(np.count_nonzero(crmask))

            if not add_crmask:
                continue

            if 'flc' in name_l or 'flt' in name_l:
                dq_idx = i + 2
                if dq_idx < len(hdul) and getattr(hdul[dq_idx], 'name', '') == 'DQ':
                    dq = hdul[dq_idx].data
                    dq[np.asarray(crmask, dtype=bool)] = HST_CR_DQ_BIT
            elif 'c0m' in name_l:
                c1m_cand = wfpc2_c1m_path(out)
                if out != src:
                    src_c1m = wfpc2_c1m_path(src)
                    if src_c1m.is_file() and not c1m_cand.is_file():
                        shutil.copy2(src_c1m, c1m_cand)
                if not c1m_cand.is_file():
                    logger.warning(
                        'WFPC2 c1m missing for %s; cannot flag CR in DQ',
                        out.name,
                    )
                else:
                    with as_datamodel(out).open_dq(mode='update') as maskhdu:
                        if i < len(maskhdu) and maskhdu[i].data is not None:
                            maskhdu[i].data[np.asarray(crmask, dtype=bool)] = (
                                HST_CR_DQ_BIT
                            )
                            maskhdu.flush()
                            c1m_updated = c1m_cand

        hdul[0].header['ST123CR'] = (True, 'astroscrappy cleaned by st123')
        hdul.flush()

    summary = {
        'path': str(out),
        'instrument': inst,
        'n_sci': n_sci,
        'n_cr_pixels': n_cr,
        'c1m': str(c1m_updated) if c1m_updated is not None else None,
        'crpars': params,
    }
    logger.info(
        'astroscrappy %s: %d SCI ext, %d CR pixels flagged',
        out.name,
        n_sci,
        n_cr,
    )
    return summary


def clean_raw_directory(
    raw_dir: PathLike,
    *,
    patterns: Sequence[str] = ('*flc.fits', '*flt.fits', '*c0m.fits'),
    add_crmask: bool = True,
    skip_existing: bool = True,
) -> list[dict]:
    """
    Run :func:`run_cosmic` on science frames under *raw_dir*.

    Skips ``*c1m.fits`` and WFC3/IR (see :func:`should_skip_cosmic`). When
    *skip_existing* is True (default), skip files whose primary header already
    has ``ST123CR = True``.
    """
    root = Path(raw_dir).expanduser().resolve()
    results: list[dict] = []
    seen: set[str] = set()
    for pat in patterns:
        for path in sorted(root.glob(pat)):
            if path.name.lower().endswith('c1m.fits'):
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            if should_skip_cosmic(path):
                results.append(
                    {
                        'path': str(path),
                        'skipped': True,
                        'reason': 'wfc3_ir',
                        'n_cr_pixels': 0,
                    }
                )
                continue
            if skip_existing:
                try:
                    with as_datamodel(path).open(memmap=True) as hdul:
                        if hdul[0].header.get('ST123CR'):
                            results.append(
                                {
                                    'path': str(path),
                                    'skipped': True,
                                    'n_cr_pixels': 0,
                                }
                            )
                            continue
                except Exception:
                    pass
            try:
                summary = run_cosmic(path, add_crmask=add_crmask, inplace=True)
                results.append(summary)
            except Exception as exc:
                logger.error('cosmic clean failed for %s: %s', path.name, exc)
                results.append(
                    {
                        'path': str(path),
                        'error': f'{type(exc).__name__}: {exc}',
                        'n_cr_pixels': 0,
                    }
                )
    return results
