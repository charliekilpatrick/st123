"""
LAcosmic-style cosmic-ray cleaning for HST frames (astroscrappy).

Mirrors ``hst123.run_cosmic``: replace SCI with the cleaned array and optionally
flag CR pixels in the DQ extension (FLC/FLT) or WFPC2 ``*_c1m.fits`` sidecar
with :data:`st123.utils.settings.HST_CR_DQ_BIT`.

Writes are done with ``fits.open(..., mode='update')`` so trailing
``HeaderletHDU`` blocks on WFC3/ACS FLCs are preserved (a full ``writeto``
rewrite truncates them and breaks AstroDrizzle ``drizCR``).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
from astropy.io import fits

from st123.utils.helpers import get_instrument
from st123.utils.settings import HST_CR_DQ_BIT, hst_crpars

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


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


def wfpc2_c1m_path(c0m_path: PathLike) -> Path:
    """Sibling ``*_c1m.fits`` path for a WFPC2 ``*_c0m.fits`` science file."""
    path = Path(c0m_path)
    name = path.name
    if name.endswith('_c0m.fits'):
        return path.with_name(name.replace('_c0m.fits', '_c1m.fits'))
    if name.endswith('c0m.fits'):
        return path.with_name(name[:-8] + 'c1m.fits')
    return path.with_name(path.stem + '_c1m.fits')


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
    """
    try:
        from astroscrappy import detect_cosmics
    except ImportError as exc:
        raise ImportError(
            'run_cosmic requires astroscrappy (see requirements.txt)'
        ) from exc

    src = Path(image).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    inst = (instrument or get_instrument(src)).split('_')[0].lower()
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
    with fits.open(out, mode='update') as hdul:
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
                    with fits.open(c1m_cand, mode='update') as maskhdu:
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

    Skips ``*c1m.fits``. When *skip_existing* is True (default), skip files
    whose primary header already has ``ST123CR = True``.
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
            if skip_existing:
                try:
                    with fits.open(path, memmap=True) as hdul:
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
