"""
Prepare JWST frames for DOLPHOT (mask + sky + parameter file).

Independent of mosaicking: stage frames, write ``dolphot.param``, run
``nircammask`` / ``mirimask`` and ``calcsky``. MIRI defaults follow
``dolphotMIRI.pdf``.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Optional, Sequence, TypeVar, Union

from astropy.io import fits
from st123.datamodels import (
    ACSDataModel,
    HSTDataModel,
    JWSTDataModel,
    MIRIDataModel,
    NIRCamDataModel,
    WFC3IRDataModel,
    WFC3UVISDataModel,
    WFPC2DataModel,
    as_datamodel,
)

from st123.utils.logging import run_logged_subprocess

_T = TypeVar('_T')
_R = TypeVar('_R')

PathLike = Union[str, os.PathLike]

logger = logging.getLogger(__name__)

_FRAME_LIST_NAME = 'dolphot_frames.txt'


def _repair_zero_exptime_fits(path: PathLike) -> float | None:
    """
    Repair ``EXPTIME<=0`` from EXPSTART/EXPEND on a FITS file (in place).

    Delegates to :func:`st123.stages.mosaic.hst_drizzle._repair_zero_exptime`.
    """
    from st123.stages.mosaic.hst_drizzle import _repair_zero_exptime

    p = Path(path)
    if not p.is_file():
        return None
    try:
        with as_datamodel(p).open(mode='update', memmap=False) as hdul:
            repaired = _repair_zero_exptime(hdul)
            if repaired is not None:
                hdul.flush()
                logger.info(
                    'Repaired EXPTIME=0 -> %.3fs from EXPSTART/EXPEND on %s',
                    repaired,
                    p.name,
                )
            return repaired
    except Exception as exc:
        logger.warning('EXPTIME repair failed for %s: %s', p.name, exc)
        return None


def _parallel_map(
    func: Callable[[_T], _R],
    items: Sequence[_T],
    *,
    ncores: int = 1,
    label: str = 'task',
) -> list[_R]:
    """
    Map *func* over *items* with a thread pool (subprocess-bound DOLPHOT tools).

    Preserves input order in the returned list. Falls back to serial execution
    when ``ncores <= 1`` or there is at most one item.
    """
    seq = list(items)
    if not seq:
        return []
    workers = max(1, min(int(ncores or 1), len(seq)))
    if workers == 1:
        return [func(item) for item in seq]
    logger.info(
        'Parallel %s: %d item(s) on %d worker(s)',
        label,
        len(seq),
        workers,
    )
    by_index: dict[int, _R] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(func, item): i for i, item in enumerate(seq)}
        for fut in as_completed(futures):
            by_index[futures[fut]] = fut.result()
    return [by_index[i] for i in range(len(seq))]


# Orphaned HST Lookup / D2IM cards that survive splitgroups on single-ext
# chips. DOLPHOT ``headerWCS`` (UseWCS=1/2) only reads TAN (+ SIP A_*/B_*).
_DOLPHOT_WCS_STRIP_EXACT = frozenset(
    {
        'CPDIS1',
        'CPDIS2',
        'D2IMEXT',
        'D2IMERR1',
        'D2IMERR2',
        'D2IMDIS1',
        'D2IMDIS2',
    }
)
_DOLPHOT_MAX_SIP_ORDER = 5


def _is_dolphot_wcs_card(key: str) -> bool:
    ku = str(key).upper()
    if ku in _DOLPHOT_WCS_STRIP_EXACT:
        return True
    if ku.startswith(('DP1.', 'DP2.', 'D2IM1.', 'D2IM2.')):
        return True
    if ku.startswith(('A_', 'B_', 'AP_', 'BP_')):
        return True
    return ku in {
        'CTYPE1', 'CTYPE2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
        'CUNIT1', 'CUNIT2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
        'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'CDELT1', 'CDELT2',
        'LONPOLE', 'LATPOLE', 'RADESYS', 'EQUINOX',
    }


def sanitize_dolphot_wcs(path: PathLike, *, fit_inverse: bool = False) -> bool:
    """
    Rewrite image WCS into a form DOLPHOT ``headerWCS`` / ``UseWCS=2`` parse.

    * Accepts drizzle ``RA---TAN`` (L3 ``img0``) or JHAT ``RA---TAN-SIP`` (L2).
    * Strips orphaned HST Lookup / D2IM cards (no WCSDVARR after splitgroups).
    * Caps forward SIP at order 5 (DOLPHOT ``headerWCS`` limit).
    * Optionally refits a complete SIP including reverse ``AP_*`` / ``BP_*``
      (``fit_inverse=True``); off by default so JHAT forward SIP is preserved.
      ``UseWCS=2`` fits its own reverse internally either way.
    * Leaves pixel data untouched.

    Returns True when the header was modified.
    """
    p = Path(path)
    with as_datamodel(p).open(mode='readonly') as hdul:
        hdu_idx = None
        for i, hdu in enumerate(hdul):
            data = hdu.data
            if data is not None and getattr(data, 'ndim', 0) >= 2:
                hdu_idx = i
                break
        if hdu_idx is None:
            raise ValueError(f'No image data for WCS sanitize in {p}')
        old_hdr = hdul[hdu_idx].header.copy()
        shape = hdul[hdu_idx].data.shape

    ctype1 = str(old_hdr.get('CTYPE1', '')).strip().upper()
    ctype2 = str(old_hdr.get('CTYPE2', '')).strip().upper()
    if ctype1 not in ('RA---TAN', 'RA---TAN-SIP') or ctype2 not in (
        'DEC--TAN',
        'DEC--TAN-SIP',
    ):
        raise ValueError(
            f'{p.name}: DOLPHOT requires RA---TAN[/SIP] + DEC--TAN[/SIP], '
            f'got {ctype1!r} / {ctype2!r}'
        )

    work = old_hdr.copy()
    changed = False
    for key in list(work.keys()):
        ku = str(key).upper()
        if ku in _DOLPHOT_WCS_STRIP_EXACT or ku.startswith(
            ('DP1.', 'DP2.', 'D2IM1.', 'D2IM2.')
        ):
            del work[key]
            changed = True

    has_sip = ctype1.endswith('-SIP') or ctype2.endswith('-SIP')
    new_wcs_hdr = None

    if has_sip:
        if work.get('CTYPE1') != 'RA---TAN-SIP' or work.get('CTYPE2') != 'DEC--TAN-SIP':
            work['CTYPE1'] = 'RA---TAN-SIP'
            work['CTYPE2'] = 'DEC--TAN-SIP'
            changed = True
        if work.get('A_ORDER') is None or work.get('B_ORDER') is None:
            raise ValueError(
                f'{p.name}: CTYPE is TAN-SIP but A_ORDER/B_ORDER missing'
            )
        # Trim SIP polynomials above DOLPHOT's order-5 tables.
        for pref in ('A', 'B', 'AP', 'BP'):
            ord_key = f'{pref}_ORDER'
            if ord_key not in work:
                continue
            order = int(work[ord_key])
            if order <= _DOLPHOT_MAX_SIP_ORDER:
                continue
            for i in range(order + 1):
                for j in range(order + 1 - i):
                    if i + j < 2 or i + j <= _DOLPHOT_MAX_SIP_ORDER:
                        continue
                    k = f'{pref}_{i}_{j}'
                    if k in work:
                        del work[k]
                        changed = True
            work[ord_key] = _DOLPHOT_MAX_SIP_ORDER
            changed = True

        if fit_inverse:
            try:
                import numpy as np
                from astropy.coordinates import SkyCoord
                from astropy.wcs import WCS
                from astropy.wcs.utils import fit_wcs_from_points

                wcs = WCS(work, relax=True)
                needs_inv = wcs.sip is not None and (
                    wcs.sip.ap is None or wcs.sip.bp is None
                )
                if needs_inv:
                    ny, nx = int(shape[-2]), int(shape[-1])
                    n = 20
                    xs = np.linspace(0, max(nx - 1, 0), n)
                    ys = np.linspace(0, max(ny - 1, 0), n)
                    xx, yy = np.meshgrid(xs, ys)
                    sky = wcs.pixel_to_world(xx.ravel(), yy.ravel())
                    degree = min(int(work['A_ORDER']), _DOLPHOT_MAX_SIP_ORDER)
                    wcs_new = fit_wcs_from_points(
                        (xx.ravel(), yy.ravel()),
                        SkyCoord(sky.ra, sky.dec),
                        proj_point='center',
                        sip_degree=degree,
                    )
                    new_wcs_hdr = wcs_new.to_header(relax=True)
                    changed = True
            except Exception as exc:
                logger.warning(
                    'Could not fit reverse SIP for %s: %s', p.name, exc
                )
    else:
        if work.get('CTYPE1') != 'RA---TAN' or work.get('CTYPE2') != 'DEC--TAN':
            work['CTYPE1'] = 'RA---TAN'
            work['CTYPE2'] = 'DEC--TAN'
            changed = True
        for key in list(work.keys()):
            if str(key).upper().startswith(('A_', 'B_', 'AP_', 'BP_')):
                del work[key]
                changed = True

    if not changed:
        return False

    # Build the replacement WCS card set.
    if new_wcs_hdr is None:
        new_wcs_hdr = fits.Header()
        for key, val in work.items():
            ku = str(key).upper()
            if not _is_dolphot_wcs_card(key):
                continue
            if ku in _DOLPHOT_WCS_STRIP_EXACT or ku.startswith(
                ('DP1.', 'DP2.', 'D2IM1.', 'D2IM2.')
            ):
                continue
            new_wcs_hdr[key] = val

    with as_datamodel(p).open(mode='update') as hdul:
        src = hdul[hdu_idx].header
        for key in list(src.keys()):
            if _is_dolphot_wcs_card(key):
                try:
                    del src[key]
                except KeyError:
                    pass
        for key, val in new_wcs_hdr.items():
            if _is_dolphot_wcs_card(key):
                src[key] = val
        hdul.flush()

    logger.info('Sanitized DOLPHOT WCS on %s', p.name)
    return True


def flatten_dolphot_fits(path: PathLike) -> bool:
    """
    Rewrite a FITS file as a single-extension image (data in PRIMARY).

    DOLPHOT chip products after ``splitgroups`` / ``*mask`` are single-HDU.
    Drizzle coadds keep an empty PRIMARY + ``SCI`` (and often WHT/CTX); if
    those are used as ``img0``, DOLPHOT aborts with
    ``Number of extensions are not the same``. Returns True when a rewrite
    was performed.

    Writes via a temp file + :func:`os.replace` so hardlinked staged copies
    (warmstart) do not mutate the shared NIRCam/MIRI source inode.
    """
    import os
    import tempfile

    p = Path(path)
    with as_datamodel(p).open(mode='readonly') as hdul:
        if len(hdul) == 1 and hdul[0].data is not None and hdul[0].data.ndim >= 2:
            return False
        sci_hdu = None
        for hdu in hdul:
            data = hdu.data
            if data is not None and getattr(data, 'ndim', 0) >= 2:
                sci_hdu = hdu
                break
        if sci_hdu is None:
            raise ValueError(f'No image data to flatten in {p}')
        hdr = sci_hdu.header.copy()
        # Preserve useful primary cards (GAIN, INSTRUME, WCS fallbacks, ...).
        for key, val in hdul[0].header.items():
            if key in hdr or key in (
                'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
                'EXTEND', 'XTENSION', 'PCOUNT', 'GCOUNT', 'INHERIT',
            ):
                continue
            try:
                hdr[key] = val
            except Exception:
                continue
        data = sci_hdu.data.copy()
    fd, tmp_name = tempfile.mkstemp(
        prefix=f'.{p.name}.',
        suffix='.flatten_tmp',
        dir=str(p.parent),
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        fits.PrimaryHDU(data=data, header=hdr).writeto(tmp, overwrite=True)
        os.replace(tmp, p)
    except Exception:
        if tmp.is_file():
            tmp.unlink()
        raise
    logger.info('Flattened %s to single-extension DOLPHOT image', p.name)
    return True


def ensure_dolphot_cd_matrix(path: PathLike) -> bool:
    """
    Rewrite PC+CDELT WCS as a CD matrix (DOLPHOT prefers CD for TAN img0).

    JWST i2d products use PC+CDELT; DOLPHOT falls back to PC*CDELT with a
    warning. Materializing CD avoids that path and matches HST chip headers.
    Returns True when a rewrite was performed.
    """
    import os
    import tempfile

    p = Path(path)
    with as_datamodel(p).open(mode='readonly') as hdul:
        if hdul[0].data is None or getattr(hdul[0].data, 'ndim', 0) < 2:
            return False
        hdr = hdul[0].header.copy()
        data = hdul[0].data.copy()
    if all(k in hdr for k in ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2')):
        return False
    need = ('PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'CDELT1', 'CDELT2')
    if not all(k in hdr for k in need):
        return False
    cdelt1 = float(hdr['CDELT1'])
    cdelt2 = float(hdr['CDELT2'])
    hdr['CD1_1'] = float(hdr['PC1_1']) * cdelt1
    hdr['CD1_2'] = float(hdr['PC1_2']) * cdelt1
    hdr['CD2_1'] = float(hdr['PC2_1']) * cdelt2
    hdr['CD2_2'] = float(hdr['PC2_2']) * cdelt2
    for key in need:
        if key in hdr:
            del hdr[key]
    fd, tmp_name = tempfile.mkstemp(
        prefix=f'.{p.name}.',
        suffix='.cd_tmp',
        dir=str(p.parent),
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        fits.PrimaryHDU(data=data, header=hdr).writeto(tmp, overwrite=True)
        os.replace(tmp, p)
    except Exception:
        if tmp.is_file():
            tmp.unlink()
        raise
    logger.info('Rewrote PC+CDELT -> CD matrix on %s', p.name)
    return True


def _flatten_dolphot_ref_and_sky(staged_ref: Path) -> None:
    """Flatten ``img0`` and its ``.sky.fits`` so extension counts match chips."""
    flatten_dolphot_fits(staged_ref)
    ensure_dolphot_cd_matrix(staged_ref)
    sky = _companion_sky_path(staged_ref)
    if sky is None:
        sky_guess = staged_ref.parent / f'{staged_ref.stem}.sky.fits'
        sky = sky_guess if sky_guess.is_file() else None
    if sky is not None and sky.is_file():
        flatten_dolphot_fits(sky)
        ensure_dolphot_cd_matrix(sky)


def remap_xyt_extension(xyt_file: PathLike, extension: int) -> int:
    """
    Rewrite the extension column (col 1) of a DOLPHOT ``warmstart.xyt``.

    After :func:`flatten_dolphot_fits` puts ``img0`` data in PRIMARY, DOLPHOT
    processes ``EXTENSION 0``. NIRCam ``.phot`` seeds use extension ``1``
    (SCI); without remapping, ``readwarm`` keeps zero stars.
    """
    path = Path(xyt_file)
    lines_out: list[str] = []
    n = 0
    with path.open() as fin:
        for line in fin:
            parts = line.split()
            if len(parts) < 6:
                lines_out.append(line.rstrip('\n'))
                continue
            parts[0] = str(int(extension))
            lines_out.append(' '.join(parts))
            n += 1
    path.write_text('\n'.join(lines_out) + ('\n' if lines_out else ''))
    logger.info(
        'Remapped %d warmstart.xyt row(s) to extension=%d for flattened img0',
        n,
        int(extension),
    )
    return n


# Numeric overlap boxes (ref_0) or whole-group mosaics (ref_full).
_GROUP_BOX_RE = re.compile(r'group_(\d+)/ref_(\d+|full)')
_DOLPHOT_MISSING_MSG = (
    'DOLPHOT not found on PATH (no `dolphot` executable from shutil.which). '
    'Install DOLPHOT, add its bin/ directory to PATH, or pass '
    '--dolphot-bin /path/to/dolphot/bin. DOLPHOT is required for photometry '
    'prep (nircammask/mirimask/calcsky) and for running dolphot.'
)
_HST_INSTRUMENTS = frozenset({'acs', 'wfc3', 'wfpc2'})
# Mixed ACS+WFC3+WFPC2 DOLPHOT staging (option C / --instrument hst).
_HST_MIXED_INSTRUMENT = 'hst'
_HST_IMAGE_KINDS = frozenset({'acs', 'wfc3', 'wfc3_ir', 'wfpc2'})
_JWST_INSTRUMENTS = frozenset({'nircam', 'miri'})


def _hst_mask_instrument(kind: str) -> str:
    """Map :func:`classify_image_kind` result -> ``acsmask`` / ``wfc3mask`` / ``wfpc2mask``."""
    k = kind.lower()
    if k == 'acs':
        return 'acs'
    if k in ('wfc3', 'wfc3_ir'):
        return 'wfc3'
    if k == 'wfpc2':
        return 'wfpc2'
    raise ValueError(f'Not an HST image kind for masking: {kind}')


def _classify_hst_science(path: PathLike) -> str:
    """Classify an HST science frame; fall back to filename tokens if needed."""
    p = Path(path)
    try:
        kind = classify_image_kind(p)
        if kind in _HST_IMAGE_KINDS:
            return kind
    except ValueError:
        pass
    name = p.name.lower()
    if 'wfpc2' in name or name.startswith('u'):
        # WFPC2 roots are often uNNNN...; prefer header, but filename last-resort.
        if 'wfpc2' in name or '_c0m' in name:
            return 'wfpc2'
    if 'acs' in name:
        return 'acs'
    if 'wfc3' in name or name.startswith('i'):
        return 'wfc3'
    raise ValueError(f'Cannot classify HST frame for DOLPHOT prep: {path}')


def _is_jwst_dolphot_reference(path: PathLike) -> bool:
    """
    Return True when *path* is a JWST (NIRCam/MIRI) reference for HST prep.

    Used to skip HST ``*mask`` / ``calcsky`` on a staged NIRCam ``img0`` during
    NIRCam->HST warmstart. Detection prefers ``INSTRUME`` / ``TELESCOP``, then
    filename tokens (``_i2d``, ``nircam``, ``miri``).
    """
    p = Path(path)
    name = p.name.lower()
    if '_i2d' in name or 'nircam' in name or name.startswith('jw'):
        # HST drizzle products are _drc/_drz; JWST coadds are typically _i2d.
        if '_drc' not in name and '_drz' not in name:
            if '_i2d' in name or 'nircam' in name:
                return True
    if not p.is_file() or p.stat().st_size == 0:
        return '_i2d' in name or 'nircam' in name
    try:
        with as_datamodel(p).open(memmap=True) as hdul:
            for hdu in hdul:
                hdr = hdu.header
                inst = str(hdr.get('INSTRUME', '') or '').upper()
                tel = str(hdr.get('TELESCOP', '') or '').upper()
                if inst in ('NIRCAM', 'MIRI') or tel == 'JWST':
                    return True
    except Exception:
        pass
    return False


def _companion_sky_path(fits_path: Path) -> Optional[Path]:
    """Return an existing ``*.sky.fits`` sidecar next to *fits_path*, if any."""
    p = Path(fits_path)
    candidates = [
        p.with_name(p.name.replace('.fits', '') + '.sky.fits'),
        p.parent / f'{p.stem}.sky.fits',
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _parse_box_token(token: str) -> int | str:
    """Parse a mosaic box id from a manifest header or directory name."""
    value = str(token).strip()
    if not value:
        raise ValueError('empty box token')
    lower = value.lower()
    if lower == 'full':
        return 'full'
    try:
        return int(value)
    except ValueError:
        # Custom stamp labels (e.g. ``sn`` from ``ref_sn`` / ``box=sn``).
        return value


@dataclass(frozen=True)
class MosaicPhotJob:
    """One mosaic box ready for DOLPHOT staging.

    Attributes
    ----------
    group : int
        Mosaic group index (``group_<n>`` directory name).
    box : int or str
        Reference box id within the group (``ref_<n>`` or ``ref_full``).
    refimage : pathlib.Path
        Coadd or reference FITS for this box.
    frames : tuple of pathlib.Path
        Science frames listed for DOLPHOT (excluding the reference).
    phot_outdir : pathlib.Path
        Staging directory for mask/sky/param prep (typically ``phot_<group>_<box>``).
    frame_list : pathlib.Path
        ``dolphot_frames.txt`` manifest path (or placeholder when inferred).
    """

    group: int
    box: int | str
    refimage: Path
    frames: tuple[Path, ...]
    phot_outdir: Path
    frame_list: Path


def resolve_dolphot_bin(
    dolphot_bin: Optional[PathLike] = None,
    *,
    required: bool = False,
) -> Path | None:
    """
    Resolve the DOLPHOT ``bin`` directory.

    Order
    -----
    1. Explicit ``dolphot_bin`` if provided.
    2. Parent directory of ``shutil.which('dolphot')`` (PATH lookup).

    Parameters
    ----------
    dolphot_bin : str or os.PathLike or None, optional
        Override path to the directory containing ``dolphot``, ``nircammask``,
        ``mirimask``, and ``calcsky``.
    required : bool, optional
        If True and DOLPHOT cannot be resolved, raise ``FileNotFoundError``.
        If False, log a warning and return ``None``.

    Returns
    -------
    pathlib.Path or None
        Resolved DOLPHOT ``bin`` directory, or ``None`` when not found and
        ``required`` is False.
    """
    if dolphot_bin is not None:
        path = Path(dolphot_bin).expanduser().resolve()
        if path.is_dir():
            return path
        msg = f'DOLPHOT bin directory does not exist: {path}'
        if required:
            raise FileNotFoundError(msg)
        logger.warning('%s', msg)
        return None

    exe = shutil.which('dolphot')
    if exe:
        return Path(exe).resolve().parent

    if required:
        raise FileNotFoundError(_DOLPHOT_MISSING_MSG)
    logger.warning('%s', _DOLPHOT_MISSING_MSG)
    return None


def dolphot_bin_dir(
    dolphot_bin: Optional[PathLike] = None,
    *,
    required: bool = True,
) -> Path:
    """
    Return the DOLPHOT ``bin`` directory.

    Defaults to PATH discovery via :func:`resolve_dolphot_bin`. When
    ``required`` is True (default), missing DOLPHOT raises
    ``FileNotFoundError`` - use this before programmatically invoking
    ``nircammask`` / ``mirimask`` / ``calcsky`` / ``dolphot``.

    Parameters
    ----------
    dolphot_bin : str or os.PathLike or None, optional
        Override path to the DOLPHOT ``bin`` directory.
    required : bool, optional
        If True (default), raise when DOLPHOT cannot be resolved.

    Returns
    -------
    pathlib.Path
        Resolved DOLPHOT ``bin`` directory.
    """
    resolved = resolve_dolphot_bin(dolphot_bin, required=required)
    if resolved is None:
        raise FileNotFoundError(_DOLPHOT_MISSING_MSG)
    return resolved


def science_fits_paths(directory: PathLike) -> list[str]:
    """
    Return sorted science ``*.fits`` paths under a directory.

    ``*.sky.fits`` calcsky products are excluded because they must not be
    passed to ``nircammask`` / ``mirimask`` / ``calcsky``.

    Parameters
    ----------
    directory : str or os.PathLike
        Directory to scan for ``*.fits`` files.

    Returns
    -------
    list of str
        Sorted absolute or relative FITS paths (as strings) excluding
        ``*.sky.fits`` sidecars.
    """
    root = Path(directory)
    return sorted(
        str(path)
        for path in root.glob('*.fits')
        if not path.name.endswith('.sky.fits')
    )


def parse_dolphot_frame_list(
    path: PathLike,
) -> tuple[Path, list[Path], int, int | str]:
    """
    Parse a ``dolphot_frames.txt`` manifest written by ``mosaic``.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the ``dolphot_frames.txt`` manifest.

    Returns
    -------
    refimage : pathlib.Path
        Reference / coadd FITS path from the ``# ref`` header line.
    frames : list of pathlib.Path
        Science frame paths listed in the manifest body.
    group : int
        Mosaic group index from the ``# group=`` header or inferred from the path.
    box : int or str
        Reference box id from the ``# box=`` header or inferred from the path
        (``0``, ``1``, ... or ``'full'`` for whole-group mosaics).

    Raises
    ------
    ValueError
        If the manifest lacks a ``# ref`` line or contains no frame paths.
    """
    path = Path(path)
    text = path.read_text()
    refimage: Path | None = None
    frames: list[Path] = []
    group: int = 0
    box: int | str = 0
    saw_box_header = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('# group='):
            # ``# group=0 box=1`` or ``# group=0 box=full``
            parts = stripped[1:].split()
            for part in parts:
                if part.startswith('group='):
                    group = int(part.split('=', 1)[1])
                elif part.startswith('box='):
                    box = _parse_box_token(part.split('=', 1)[1])
                    saw_box_header = True
            continue
        if stripped.startswith('# ref '):
            refimage = Path(stripped[len('# ref ') :].strip())
            continue
        if stripped.startswith('#'):
            continue
        frames.append(Path(stripped))
    if refimage is None:
        raise ValueError(f'No ``# ref`` line in {path}')
    if not frames:
        raise ValueError(f'No frame paths in {path}')
    # Infer group/box from parent path when header omitted.
    if not saw_box_header and group == 0 and box == 0:
        match = _GROUP_BOX_RE.search(path.as_posix())
        if match:
            group = int(match.group(1))
            box = _parse_box_token(match.group(2))
    return refimage, frames, group, box


def filter_frames_for_instrument(
    frames: Sequence[PathLike],
    instrument: str,
) -> list[Path]:
    """
    Keep frames matching *instrument*.

    Classification uses :func:`classify_image_kind`. Frames that cannot be
    classified are dropped with a warning.

    Parameters
    ----------
    frames : sequence of str or os.PathLike
        Candidate science FITS paths.
    instrument : str
        ``'nircam'`` (short/long), ``'miri'``, ``'acs'``, ``'wfc3'``, or
        ``'wfpc2'``.

    Returns
    -------
    list of pathlib.Path
        Filtered frame paths in input order.
    """
    inst = instrument.lower()
    if inst == 'miri':
        keep_kinds = {'miri'}
    elif inst == 'nircam':
        keep_kinds = {'short', 'long'}
    elif inst == 'hst':
        keep_kinds = set(_HST_IMAGE_KINDS)
    elif inst == 'wfc3':
        keep_kinds = {'wfc3', 'wfc3_ir'}
    elif inst in _HST_INSTRUMENTS:
        keep_kinds = {inst}
    else:
        keep_kinds = {inst}
    out: list[Path] = []
    for frame in frames:
        path = Path(frame)
        try:
            kind = classify_image_kind(path)
        except ValueError:
            logger.warning('Skipping unclassifiable frame: %s', path)
            continue
        if kind in keep_kinds:
            out.append(path)
    return out


def resolve_coadd_ref(
    box_dir: PathLike,
    ref_filter: Optional[str] = None,
    *,
    fallback: Optional[PathLike] = None,
) -> Path:
    """
    Resolve a mosaic coadd reference image, optionally by filter name.

    Parameters
    ----------
    box_dir : str or os.PathLike
        Mosaic box directory (``reference/group_*/ref_*``).
    ref_filter : str or None, optional
        Filter name (e.g. ``'F560W'``). When set, prefer
        ``coadd_*_<filter>_i2d.fits`` (case-insensitive).
    fallback : str or os.PathLike or None, optional
        Path returned when *ref_filter* is ``None`` or no matching coadd exists.

    Returns
    -------
    pathlib.Path
        Resolved coadd path.

    Raises
    ------
    FileNotFoundError
        If *ref_filter* is set, no match is found, and *fallback* is ``None``.
    """
    box = Path(box_dir)
    if ref_filter:
        key = ref_filter.lower().replace(' ', '')
        matches = sorted(
            p
            for p in box.glob('coadd_*.fits')
            if f'_{key}_' in p.name.lower()
            and (
                p.name.endswith('_i2d.fits')
                or p.name.endswith('_drc.fits')
                or p.name.endswith('_drz.fits')
            )
        )
        if matches:
            # Prefer JWST i2d when both missions present for the same filter.
            matches.sort(
                key=lambda p: (
                    0 if p.name.endswith('_i2d.fits') else 1,
                    p.name.lower(),
                )
            )
            return matches[0]
        if fallback is None:
            raise FileNotFoundError(
                f'No coadd matching filter {ref_filter!r} under {box}'
            )
        logger.warning(
            'No coadd matching filter %s under %s; using %s',
            ref_filter,
            box,
            fallback,
        )
    if fallback is None:
        raise FileNotFoundError(f'No reference coadd under {box}')
    return Path(fallback)


# Preferred NIRCam free-photometry reference filters (SW first).
NIRCAM_SW_REF_FILTERS: tuple[str, ...] = (
    'F150W2',
    'F200W',
    'F150W',
    'F115W',
    'F090W',
    'F070W',
    'F182M',
    'F210M',
    'F187N',
    'F212N',
)


def resolve_nircam_coadd_ref(
    box_dir: PathLike,
    ref_filter: Optional[str] = None,
    *,
    fallback: Optional[PathLike] = None,
) -> Path:
    """
    Prefer a NIRCam SW ``*_i2d.fits`` coadd as the DOLPHOT reference.

    Never returns an HST ``*_drc`` / ``*_drz`` product (``nircammask`` cannot
    process those). Tries *ref_filter*, then :data:`NIRCAM_SW_REF_FILTERS`, then
    any non-MIRI ``*_i2d.fits`` in the box.
    """
    box = Path(box_dir)
    tried: list[str] = []
    if ref_filter:
        tried.append(str(ref_filter))
    for filt in NIRCAM_SW_REF_FILTERS:
        if filt not in tried:
            tried.append(filt)
    for filt in tried:
        try:
            path = resolve_coadd_ref(box, filt, fallback=None)
        except FileNotFoundError:
            continue
        if path.name.lower().endswith('_i2d.fits'):
            return path
    i2ds = sorted(
        p
        for p in box.glob('coadd_*_i2d.fits')
        if 'miri' not in p.name.lower()
        and '_f770w_' not in p.name.lower()
        and '_f1000w_' not in p.name.lower()
        and '_f1130w_' not in p.name.lower()
        and '_f1280w_' not in p.name.lower()
        and '_f1500w_' not in p.name.lower()
        and '_f1800w_' not in p.name.lower()
        and '_f2100w_' not in p.name.lower()
        and '_f2550w_' not in p.name.lower()
    )
    if i2ds:
        # Prefer shorter-wavelength / wider SW filters by name rank.
        def _rank(p: Path) -> tuple[int, str]:
            name = p.name.lower()
            for i, filt in enumerate(NIRCAM_SW_REF_FILTERS):
                if f'_{filt.lower()}_' in name:
                    return (i, name)
            return (len(NIRCAM_SW_REF_FILTERS), name)

        i2ds.sort(key=_rank)
        return i2ds[0]
    if fallback is not None and str(fallback).lower().endswith('_i2d.fits'):
        return Path(fallback)
    raise FileNotFoundError(
        f'No NIRCam *_i2d.fits coadd under {box} '
        f'(tried filters {tried}; nircammask requires a JWST reference)'
    )


def discover_mosaic_phot_jobs(
    reduction_dir: PathLike,
    *,
    instrument: Optional[str] = None,
    ref_filter: Optional[str] = None,
    phot_outdir_root: Optional[PathLike] = None,
    outdir_prefix: str = 'phot',
) -> list[MosaicPhotJob]:
    """
    Find mosaic boxes under ``<reduction>/reference/`` for DOLPHOT prep.

    Prefers ``dolphot_frames.txt`` manifests. If a coadd exists without a
    manifest, pairs it with JHAT frames under ``jhat_jwst/``, ``jhat_hst/``,
    or legacy ``jhat/``. Discovers JWST ``*_i2d.fits`` and HST ``*_drc.fits`` /
    ``*_drz.fits`` coadds under the shared ``group_*/ref_*`` layout.

    Parameters
    ----------
    reduction_dir : str or os.PathLike
        Mosaic reduction root (contains ``reference/`` and optionally JHAT dirs).
    instrument : str or None, optional
        When ``'miri'``, ``'nircam'``, ``'acs'``, ``'wfc3'``, ``'wfpc2'``, or
        ``'hst'``, keep only matching science frames.
    ref_filter : str or None, optional
        Prefer a coadd whose filename contains this filter (e.g. ``'F560W'``).
    phot_outdir_root : str or os.PathLike or None, optional
        Parent directory for staging runs. Default: *reduction_dir*.
        MIRI/HST runs typically use ``<project>/dolphot``.
    outdir_prefix : str, optional
        Staging directory name prefix (default ``'phot'`` -> ``phot_0_0``;
        use ``'miri'`` / ``'hst'`` -> ``miri_0_0`` / ``hst_0_0``).

    Returns
    -------
    list of MosaicPhotJob
        One job per mosaic box, sorted by manifest path. Empty when
        ``reference/`` is missing or no boxes are found. Boxes that yield no
        frames after instrument filtering are omitted.
    """
    root = Path(reduction_dir)
    out_root = Path(phot_outdir_root) if phot_outdir_root is not None else root
    jobs: list[MosaicPhotJob] = []
    ref_root = root / 'reference'
    if not ref_root.is_dir():
        return jobs

    inst = (instrument or '').lower() or None

    def _collect_jhat(inst_name: str | None) -> list[Path]:
        jhat_dirs = [
            root / 'jhat_jwst',
            root / 'jhat_hst',
            root / 'jhat',
        ]
        found: list[Path] = []
        for jd in jhat_dirs:
            if jd.is_dir():
                found.extend(
                    sorted(
                        p
                        for p in jd.glob('*jhat.fits')
                        if not p.name.lower().startswith('coadd_')
                    )
                )
        if inst_name is not None:
            found = filter_frames_for_instrument(found, inst_name)
        return found

    manifests = sorted(ref_root.glob('group_*/ref_*/' + _FRAME_LIST_NAME))
    if manifests:
        for manifest in manifests:
            refimage, frames, group, box = parse_dolphot_frame_list(manifest)
            box_dir = manifest.parent
            if inst == 'nircam':
                try:
                    refimage = resolve_nircam_coadd_ref(
                        box_dir, ref_filter, fallback=None
                    )
                except FileNotFoundError as exc:
                    logger.warning('Skipping box %s: %s', box_dir, exc)
                    continue
            else:
                refimage = resolve_coadd_ref(
                    box_dir,
                    ref_filter,
                    fallback=refimage,
                )
            if inst is not None:
                frames = filter_frames_for_instrument(frames, inst)
            if not frames and inst == 'nircam':
                # Shared dolphot_frames.txt may be HST-only after remosaic;
                # recover NIRCam paths from any tagged lists in this box.
                seen: set[Path] = set()
                recovered: list[Path] = []
                for tagged in sorted(box_dir.glob('dolphot_frames*.txt')):
                    try:
                        _r, tagged_frames, _g, _b = parse_dolphot_frame_list(
                            tagged
                        )
                    except Exception:
                        continue
                    for p in filter_frames_for_instrument(tagged_frames, 'nircam'):
                        key = p.resolve()
                        if key in seen:
                            continue
                        seen.add(key)
                        recovered.append(p)
                frames = recovered
            if not frames:
                logger.warning(
                    'No %s frames in %s; skipping box',
                    instrument or 'science',
                    manifest,
                )
                continue
            jobs.append(
                MosaicPhotJob(
                    group=group,
                    box=box,
                    refimage=refimage,
                    frames=tuple(frames),
                    phot_outdir=out_root / f'{outdir_prefix}_{group}_{box}',
                    frame_list=manifest,
                )
            )
        return jobs

    # Fallback: coadd present, no manifest (legacy mosaic run).
    jhat = _collect_jhat(inst)
    coadd_globs = (
        'group_*/ref_*/coadd_*_i2d.fits',
        'group_*/ref_*/coadd_*_drc.fits',
        'group_*/ref_*/coadd_*_drz.fits',
    )
    seen_boxes: set[Path] = set()
    for pat in coadd_globs:
        for coadd in sorted(ref_root.glob(pat)):
            box_dir = coadd.parent
            if box_dir in seen_boxes:
                continue
            seen_boxes.add(box_dir)
            match = _GROUP_BOX_RE.search(coadd.as_posix())
            group = int(match.group(1)) if match else 0
            box: int | str = _parse_box_token(match.group(2)) if match else 0
            if inst == 'nircam':
                try:
                    refimage = resolve_nircam_coadd_ref(
                        box_dir, ref_filter, fallback=coadd
                    )
                except FileNotFoundError as exc:
                    logger.warning('Skipping box %s: %s', box_dir, exc)
                    continue
            else:
                refimage = resolve_coadd_ref(box_dir, ref_filter, fallback=coadd)
            if not jhat:
                continue
            jobs.append(
                MosaicPhotJob(
                    group=group,
                    box=box,
                    refimage=refimage,
                    frames=tuple(jhat),
                    phot_outdir=out_root / f'{outdir_prefix}_{group}_{box}',
                    frame_list=box_dir / _FRAME_LIST_NAME,
                )
            )
    return jobs


def prepare_mosaic_phot_job(
    job: MosaicPhotJob,
    *,
    instrument: str = 'nircam',
    dolphot_bin: Optional[PathLike] = None,
    skip_mask: bool = False,
    skip_sky: bool = False,
    copy_files: bool = True,
    ncores: int = 1,
) -> Path:
    """
    Stage a mosaic box into ``phot_*`` / ``miri_*``, write ``dolphot.param``,
    mask + sky.

    Parameters
    ----------
    job : MosaicPhotJob
        Mosaic box descriptor from :func:`discover_mosaic_phot_jobs`.
    instrument : str, optional
        ``'nircam'`` or ``'miri'``; selects mask and calcsky defaults.
        MIRI uses recommended ``dolphotMIRI.pdf`` globals via
        :attr:`MIRIDataModel.DOLPHOT_BASE_PARAMS`.
    dolphot_bin : str or os.PathLike or None, optional
        Override path to the DOLPHOT ``bin`` directory.
    skip_mask : bool, optional
        If True, skip ``nircammask`` / ``mirimask``.
    skip_sky : bool, optional
        If True, skip ``calcsky``.
    copy_files : bool, optional
        If True (default), copy reference and science frames into
        ``job.phot_outdir`` before prep.
    ncores : int, optional
        Parallel workers for per-frame mask / calcsky.

    Returns
    -------
    pathlib.Path
        Path to the written ``dolphot.param`` file.
    """
    inst = instrument.lower()
    global_params = MIRIDataModel.DOLPHOT_BASE_PARAMS if inst == 'miri' else None
    phot_out = f'{job.phot_outdir.name}.phot'
    param = setup_paramfile(
        job.phot_outdir,
        job.refimage,
        list(job.frames),
        copy_files=copy_files,
        global_params=global_params,
        phot_out=phot_out,
    )
    work = [Path(p) for p in science_fits_paths(job.phot_outdir)]
    prepare_frames(
        work,
        instrument=inst,
        dolphot_bin=dolphot_bin,
        skip_mask=skip_mask,
        skip_sky=skip_sky,
        ncores=ncores,
    )
    return param


def _prepend_bin_env(env: Optional[MutableMapping[str, str]], bin_dir: Path) -> dict:
    out = dict(env or os.environ)
    bin_s = str(bin_dir)
    path = out.get('PATH', '')
    if bin_s not in path.split(os.pathsep):
        out['PATH'] = bin_s + (os.pathsep + path if path else '')
    return out


def classify_image_kind(path: PathLike) -> str:
    """
    Classify a FITS frame for per-image DOLPHOT parameters.

    Parameters
    ----------
    path : str or os.PathLike
        FITS path to classify.

    Returns
    -------
    str
        One of ``'short'`` / ``'long'`` (NIRCam), ``'miri'``, ``'acs'``,
        ``'wfc3'``, ``'wfc3_ir'``, or ``'wfpc2'``.

    Raises
    ------
    ValueError
        If the frame cannot be classified from the filename or FITS headers.
    """
    from st123.datamodels import classify_image_kind as classify_kind

    return classify_kind(path)


def per_image_params(kind: str) -> Mapping[str, str]:
    """
    Return the per-image DOLPHOT parameter dict for an image kind.

    Parameters
    ----------
    kind : str
        One of ``'short'``, ``'long'``, ``'miri'``, ``'acs'``, ``'wfc3'``,
        ``'wfc3_ir'``, or ``'wfpc2'``.

    Returns
    -------
    dict
        Mapping of DOLPHOT parameter names to string values for one image
        extension.

    Raises
    ------
    ValueError
        If *kind* is not recognized.
    """
    if kind == 'short':
        return NIRCamDataModel.DOLPHOT_SHORT_PARAMS
    if kind == 'long':
        return NIRCamDataModel.DOLPHOT_LONG_PARAMS
    if kind == 'miri':
        return MIRIDataModel.DOLPHOT_IMAGE_PARAMS
    if kind == 'acs':
        return ACSDataModel.DOLPHOT_IMAGE_PARAMS
    if kind == 'wfc3':
        return WFC3UVISDataModel.DOLPHOT_IMAGE_PARAMS
    if kind == 'wfc3_ir':
        return WFC3IRDataModel.DOLPHOT_IMAGE_PARAMS
    if kind == 'wfpc2':
        return WFPC2DataModel.DOLPHOT_IMAGE_PARAMS
    raise ValueError(f'Unknown image kind: {kind}')


def _run_cwd_for_files(files: Sequence[PathLike]) -> tuple[Optional[Path], list[str]]:
    """
    Prefer running DOLPHOT tools with short basenames from a shared parent dir.

    Absolute paths under deep JWST trees can overflow fixed C filename buffers
    in ``calcsky`` / mask utilities (observed as SIGABRT / buffer overflow).
    """
    paths = [Path(f).resolve() for f in files]
    if not paths:
        return None, []
    parents = {p.parent for p in paths}
    if len(parents) == 1:
        return next(iter(parents)), [p.name for p in paths]
    return None, [str(p) for p in paths]


def apply_nircammask(
    files: Sequence[PathLike],
    *,
    dolphot_bin: Optional[PathLike] = None,
    estnoise: bool = False,
    noetctime: bool = False,
    check: bool = True,
    ncores: int = 1,
) -> None:
    """
    Run ``nircammask`` on science frames (in-place).

    Matches the mosaic NIRCam prep. Current ``nircammask`` options (DOLPHOT
    NIRCam): ``-estnoise``, ``-noetctime``. ETC exposure time is the default;
    there is no ``-etctime`` flag.

    Parameters
    ----------
    files : sequence of str or os.PathLike
        FITS paths to mask in place.
    dolphot_bin : str or os.PathLike or None, optional
        Override path to the DOLPHOT ``bin`` directory.
    estnoise : bool, optional
        If True, pass ``-estnoise`` (readout noise from ``VAR_RNOISE``).
    noetctime : bool, optional
        If True, pass ``-noetctime`` to use ``EFFEXPTM`` instead of ETC time.
    check : bool, optional
        If True (default), raise when the subprocess exits non-zero.
    ncores : int, optional
        Parallel workers (one ``nircammask`` process per frame when ``>1``).

    Returns
    -------
    None
    """
    bin_dir = dolphot_bin_dir(dolphot_bin)
    cwd, names = _run_cwd_for_files(files)
    base = [str(bin_dir / 'nircammask')]
    if estnoise:
        base.append('-estnoise')
    if noetctime:
        base.append('-noetctime')
    env = _prepend_bin_env(None, bin_dir)

    def _one(name: str) -> str:
        run_logged_subprocess(
            [*base, name],
            check=check,
            env=env,
            cwd=cwd,
            logger=logger,
            label=f'nircammask {name}',
        )
        return name

    if max(1, int(ncores or 1)) <= 1 or len(names) <= 1:
        run_logged_subprocess(
            [*base, *names],
            check=check,
            env=env,
            cwd=cwd,
            logger=logger,
            label=f'nircammask ({len(names)} file(s))',
        )
        return
    _parallel_map(_one, names, ncores=ncores, label='nircammask')


def apply_mirimask(
    files: Sequence[PathLike],
    *,
    dolphot_bin: Optional[PathLike] = None,
    estnoise: bool = True,
    noetctime: bool = False,
    mask_lyot: bool = False,
    check: bool = True,
    ncores: int = 1,
) -> None:
    """
    Run ``mirimask`` on science frames (in-place).

    Default flags follow ``dolphotMIRI.pdf`` Sec.3.3 recommendations:
    ``-estnoise`` on, ETC exposure time on (do **not** pass ``-noetctime``).
    Back up originals before calling; ``mirimask`` rewrites the FITS files.

    Parameters
    ----------
    files : sequence of str or os.PathLike
        FITS paths to mask in place.
    dolphot_bin : str or os.PathLike or None, optional
        Override path to the DOLPHOT ``bin`` directory.
    estnoise : bool, optional
        If True (default), pass ``-estnoise``.
    noetctime : bool, optional
        If True, pass ``-noetctime`` to use ``EFFEXPTM`` instead of ETC time.
    mask_lyot : bool, optional
        If True, pass ``-mask_lyot`` to mask the Lyot stop region.
    check : bool, optional
        If True (default), raise when the subprocess exits non-zero.
    ncores : int, optional
        Parallel workers (one ``mirimask`` process per frame when ``>1``).

    Returns
    -------
    None
    """
    bin_dir = dolphot_bin_dir(dolphot_bin)
    cwd, names = _run_cwd_for_files(files)
    base = [str(bin_dir / 'mirimask')]
    if estnoise:
        base.append('-estnoise')
    if noetctime:
        base.append('-noetctime')
    if mask_lyot:
        base.append('-mask_lyot')
    env = _prepend_bin_env(None, bin_dir)

    def _one(name: str) -> str:
        run_logged_subprocess(
            [*base, name],
            check=check,
            env=env,
            cwd=cwd,
            logger=logger,
            label=f'mirimask {name}',
        )
        return name

    if max(1, int(ncores or 1)) <= 1 or len(names) <= 1:
        run_logged_subprocess(
            [*base, *names],
            check=check,
            env=env,
            cwd=cwd,
            logger=logger,
            label=f'mirimask ({len(names)} file(s))',
        )
        return
    _parallel_map(_one, names, ncores=ncores, label='mirimask')


def calc_sky(
    files: Sequence[PathLike],
    *,
    instrument: str = 'nircam',
    dolphot_bin: Optional[PathLike] = None,
    rin: Optional[float] = None,
    rout: Optional[float] = None,
    step: Optional[float] = None,
    sigma_low: Optional[float] = None,
    sigma_high: Optional[float] = None,
    check: bool = True,
    ncores: int = 1,
) -> None:
    """
    Run DOLPHOT ``calcsky`` on each science frame.

    Defaults follow the instrument manuals / hst123 detector defaults:

    - NIRCam: rin=15, rout=25, step=-64, sigma=2.25/2.00
    - MIRI: rin=10, rout=25, step=-64, sigma=2.25/2.00 (``dolphotMIRI.pdf`` Sec.3.4)
    - ACS / WFC3 UVIS: rin=15, rout=35, step=4
    - WFPC2: rin=10, rout=25, step=2

    When all frames share a directory, ``calcsky`` is invoked with basenames
    and ``cwd`` set to that directory to avoid C path-buffer overflows.
    Independent frames are processed in parallel when ``ncores > 1``.

    Parameters
    ----------
    files : sequence of str or os.PathLike
        FITS paths for which to compute sky maps.
    instrument : str, optional
        ``'nircam'``, ``'miri'``, ``'acs'``, ``'wfc3'``, or ``'wfpc2'``.
    dolphot_bin : str or os.PathLike or None, optional
        Override path to the DOLPHOT ``bin`` directory.
    rin : float or None, optional
        Inner sky annulus radius in pixels; instrument default when ``None``.
    rout : float or None, optional
        Outer sky annulus radius in pixels; instrument default when ``None``.
    step : float or None, optional
        Sky sampling step; instrument default when ``None``.
    sigma_low : float or None, optional
        Lower sigma-clipping threshold; instrument default when ``None``.
    sigma_high : float or None, optional
        Upper sigma-clipping threshold; instrument default when ``None``.
    check : bool, optional
        If True (default), raise when a subprocess exits non-zero.
    ncores : int, optional
        Parallel workers (one ``calcsky`` process per frame).

    Returns
    -------
    None
    """
    inst = instrument.lower()
    if inst == 'miri':
        defaults = MIRIDataModel.CALCSKY_PARAMS
    elif inst == 'acs':
        defaults = ACSDataModel.CALCSKY_PARAMS
    elif inst == 'wfc3':
        defaults = WFC3UVISDataModel.CALCSKY_PARAMS
    elif inst == 'wfpc2':
        defaults = WFPC2DataModel.CALCSKY_PARAMS
    else:
        defaults = NIRCamDataModel.CALCSKY_PARAMS
    rin = defaults['rin'] if rin is None else rin
    rout = defaults['rout'] if rout is None else rout
    step = defaults['step'] if step is None else step
    sigma_low = defaults['sigma_low'] if sigma_low is None else sigma_low
    sigma_high = defaults['sigma_high'] if sigma_high is None else sigma_high

    bin_dir = dolphot_bin_dir(dolphot_bin)
    calcsky = str(bin_dir / 'calcsky')
    env = _prepend_bin_env(None, bin_dir)
    cwd, names = _run_cwd_for_files(files)

    def _one(name: str) -> str:
        fits_base = name[:-5] if name.endswith('.fits') else name.replace('.fits', '')
        cmd = [
            calcsky,
            fits_base,
            str(rin),
            str(rout),
            str(step),
            str(sigma_low),
            str(sigma_high),
        ]
        run_logged_subprocess(
            cmd,
            check=check,
            env=env,
            cwd=cwd,
            logger=logger,
            label=f'calcsky {fits_base}',
        )
        return fits_base

    _parallel_map(_one, names, ncores=ncores, label='calcsky')


def _wfpc2_dq_companion(science) -> Optional[Path]:
    """Locate ``*_c1m.fits`` next to a WFPC2 ``*_c0m`` / ``*_jhat`` MEF."""
    from st123.datamodels.hst.wfpc2 import WFPC2DataModel
    from st123.datamodels.instrument import InstrumentDataModel, as_datamodel

    model = science if isinstance(science, WFPC2DataModel) else as_datamodel(science)
    if isinstance(model, WFPC2DataModel) and model.has_dq():
        return model.dq_path
    path = model.path if isinstance(model, InstrumentDataModel) else Path(science)
    return WFPC2DataModel.find_dq_beside(path)


def apply_hst_mask(
    files: Sequence[PathLike],
    instrument: str,
    *,
    dolphot_bin: Optional[PathLike] = None,
    check: bool = True,
    ncores: int = 1,
) -> None:
    """
    Run ``acsmask`` / ``wfc3mask`` / ``wfpc2mask`` on science frames (in-place).

    For WFPC2 MEFs, ``wfpc2mask`` is invoked as ``science c1m`` pairs when a
    sibling ``*_c1m.fits`` DQ file is present (required for native stacks).

    Parameters
    ----------
    files : sequence of path-like
        FITS paths to mask (MEFs or per-chip ``*.chipN.fits``).
    instrument : str
        ``'acs'``, ``'wfc3'``, or ``'wfpc2'``.
    dolphot_bin : path-like or None, optional
        DOLPHOT ``bin`` directory override.
    check : bool, optional
        Raise on non-zero exit when True.
    ncores : int, optional
        Parallel workers (one mask process per frame / MEF pair when ``>1``).
    """
    inst = instrument.lower()
    if inst not in _HST_INSTRUMENTS:
        raise ValueError(f'Unsupported HST mask instrument: {instrument}')
    bin_dir = dolphot_bin_dir(dolphot_bin)
    exe_name = f'{inst}mask'
    exe = bin_dir / exe_name
    if not exe.is_file():
        which = shutil.which(exe_name)
        if which:
            exe = Path(which)
        else:
            raise FileNotFoundError(
                f'{exe_name} not found under {bin_dir} or on PATH'
            )

    paths = [Path(p).resolve() for p in files]
    env = _prepend_bin_env(None, bin_dir)
    workers = max(1, int(ncores or 1))

    if inst == 'wfpc2':
        # One job per science file: [sci] or [sci, c1m].
        jobs: list[tuple[str, list[str]]] = []
        for sci in paths:
            args = [sci.name]
            name_l = sci.name.lower()
            if '.chip' not in name_l and not any(
                tok in name_l for tok in ('_drz', '_drc', '_drw', 'coadd_')
            ):
                dq = _wfpc2_dq_companion(sci)
                if dq is None:
                    logger.warning(
                        'WFPC2 c1m missing for %s; wfpc2mask may fail on native MEF',
                        sci.name,
                    )
                else:
                    dq_local = sci.parent / dq.name
                    if dq.resolve() != dq_local.resolve():
                        shutil.copy2(dq, dq_local)
                    args.append(dq_local.name)
            jobs.append((str(sci.parent), args))

        def _wfpc2_one(job: tuple[str, list[str]]) -> str:
            cwd, args = job
            run_logged_subprocess(
                [str(exe), *args],
                check=check,
                env=env,
                cwd=cwd,
                logger=logger,
                label=f'{exe_name} {" ".join(args)}',
            )
            return args[0]

        if workers <= 1 or len(jobs) <= 1:
            # Preserve historical single-process multi-file invocation.
            cmd: list[str] = [str(exe)]
            cwd = str(paths[0].parent) if paths else None
            for _cwd, args in jobs:
                cmd.extend(args)
            run_logged_subprocess(
                cmd,
                check=check,
                env=env,
                cwd=cwd,
                logger=logger,
                label=f'{exe_name} ({len(paths)} file(s), c1m-paired)',
            )
            return
        _parallel_map(_wfpc2_one, jobs, ncores=workers, label=exe_name)
        return

    cwd, names = _run_cwd_for_files(files)

    def _one(name: str) -> str:
        run_logged_subprocess(
            [str(exe), name],
            check=check,
            env=env,
            cwd=cwd,
            logger=logger,
            label=f'{exe_name} {name}',
        )
        return name

    if workers <= 1 or len(names) <= 1:
        run_logged_subprocess(
            [str(exe), *names],
            check=check,
            env=env,
            cwd=cwd,
            logger=logger,
            label=f'{exe_name} ({len(names)} file(s))',
        )
        return
    _parallel_map(_one, names, ncores=workers, label=exe_name)


def apply_splitgroups(
    files: Sequence[PathLike],
    *,
    dolphot_bin: Optional[PathLike] = None,
    check: bool = True,
    ncores: int = 1,
) -> list[Path]:
    """
    Run DOLPHOT ``splitgroups`` on multi-extension FITS frames.

    Parameters
    ----------
    files : sequence of path-like
        MEF science FITS paths.
    dolphot_bin : path-like or None, optional
        DOLPHOT ``bin`` directory override.
    check : bool, optional
        Raise on non-zero exit when True.
    ncores : int, optional
        Parallel workers (one ``splitgroups`` process per MEF).

    Returns
    -------
    list of pathlib.Path
        Per-chip ``*.chipN.fits`` products next to each input (SCI chips only
        when identifiable). If ``splitgroups`` is unavailable or produces no
        chips, returns the original paths.
    """
    bin_dir = resolve_dolphot_bin(dolphot_bin, required=False)
    exe = None
    if bin_dir is not None:
        candidate = bin_dir / 'splitgroups'
        if candidate.is_file():
            exe = candidate
    if exe is None:
        which = shutil.which('splitgroups')
        if which:
            exe = Path(which)
    if exe is None:
        logger.warning('splitgroups not found; keeping multi-extension FITS')
        return [Path(p) for p in files]

    env = _prepend_bin_env(None, exe.parent)
    paths = [Path(p).resolve() for p in files]

    def _one(src: Path) -> list[Path]:
        stem = src.name[:-5] if src.name.endswith('.fits') else src.name
        for old in src.parent.glob(f'{stem}.chip*.fits'):
            try:
                old.unlink()
            except OSError:
                pass
        run_logged_subprocess(
            [str(exe), str(src.name)],
            check=check,
            env=env,
            cwd=src.parent,
            logger=logger,
            label=f'splitgroups {src.name}',
        )
        chips = sorted(src.parent.glob(f'{stem}.chip*.fits'))
        if not chips:
            logger.warning('splitgroups produced no chips for %s', src.name)
            return [src]
        parent_inst = None
        parent_filt = None
        try:
            from st123.utils.helpers import get_filter, get_instrument

            parent_inst = get_instrument(src).split('_')[0].upper()
            parent_filt = get_filter(src).upper()
        except Exception:
            pass
        kept: list[Path] = []
        for chip in chips:
            keep = False
            try:
                with as_datamodel(chip).open(mode='update') as hdul:
                    hdr = hdul[0].header
                    extname = str(hdr.get('EXTNAME') or '').upper()
                    data = hdul[0].data
                    shape = () if data is None else tuple(data.shape)
                    is_sci = (not extname or extname == 'SCI') and len(shape) == 2
                    is_sci = is_sci and min(shape) >= 64 and max(shape) >= 256
                    if not is_sci:
                        logger.info(
                            'Removing non-SCI split %s (%s shape=%s)',
                            chip.name,
                            extname or 'no-EXTNAME',
                            shape,
                        )
                    else:
                        if parent_inst and not str(hdr.get('INSTRUME') or '').strip():
                            hdr['INSTRUME'] = parent_inst
                        if parent_filt and not str(hdr.get('FILTER') or '').strip():
                            hdr['FILTER'] = parent_filt
                        if parent_filt and parent_inst == 'WFPC2' and 'FILTNAM1' not in hdr:
                            hdr['FILTNAM1'] = parent_filt
                        hdul.flush()
                        keep = True
            except Exception as exc:
                logger.warning('Could not inspect chip %s: %s', chip.name, exc)
                keep = False
            if keep:
                kept.append(chip)
            else:
                try:
                    chip.unlink()
                except OSError:
                    pass
        return kept if kept else [src]

    per_mef = _parallel_map(_one, paths, ncores=ncores, label='splitgroups')
    chip_files: list[Path] = []
    for chips in per_mef:
        chip_files.extend(chips)
    return chip_files


def prepare_frames(
    files: Sequence[PathLike],
    *,
    instrument: str,
    dolphot_bin: Optional[PathLike] = None,
    skip_mask: bool = False,
    skip_sky: bool = False,
    ncores: int = 1,
) -> None:
    """
    Mask then compute sky for science frames.

    Order is always mask -> calcsky. Within each step, independent frames are
    processed in parallel when ``ncores > 1``.

    Parameters
    ----------
    files : sequence of str or os.PathLike
        FITS paths to prepare.
    instrument : str
        ``'nircam'``, ``'miri'``, ``'acs'``, ``'wfc3'``, or ``'wfpc2'``.
    dolphot_bin : str or os.PathLike or None, optional
        Override path to the DOLPHOT ``bin`` directory.
    skip_mask : bool, optional
        If True, skip the instrument mask utility.
    skip_sky : bool, optional
        If True, skip ``calcsky``.
    ncores : int, optional
        Parallel workers for per-frame mask / calcsky subprocesses.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If *instrument* is not supported.
    """
    inst = instrument.lower()
    workers = max(1, int(ncores or 1))
    if not skip_mask:
        if inst == 'miri':
            apply_mirimask(files, dolphot_bin=dolphot_bin, ncores=workers)
        elif inst == 'nircam':
            apply_nircammask(files, dolphot_bin=dolphot_bin, ncores=workers)
        elif inst in _HST_INSTRUMENTS:
            apply_hst_mask(files, inst, dolphot_bin=dolphot_bin, ncores=workers)
        else:
            raise ValueError(f'Unsupported instrument for prepare_frames: {instrument}')
    if not skip_sky:
        calc_sky(files, instrument=inst, dolphot_bin=dolphot_bin, ncores=workers)


def prepare_hst_frames(
    files: Sequence[PathLike],
    outdir: PathLike,
    instrument: str,
    *,
    refimage: Optional[PathLike] = None,
    dolphot_bin: Optional[PathLike] = None,
    skip_mask: bool = False,
    skip_sky: bool = False,
    skip_split: bool = False,
    copy_files: bool = True,
    ncores: int = 1,
    xytfile: Optional[PathLike] = None,
) -> Path:
    """
    Stage HST frames for DOLPHOT: splitgroups -> mask -> calcsky -> paramfile.

    Pipeline order is preserved per frame family:

    1. WFPC2 MEF ``wfpc2mask`` (needs ``c1m``) before split
    2. ``splitgroups`` on each MEF -> ``*.chipN.fits``
    3. ACS/WFC3 ``*mask`` on chip products (WFPC2 chips inherit BADPIX)
    4. ``calcsky`` on each chip / frame
    5. Coadd reference mask + calcsky (img0) - **skipped** for JWST/NIRCam
       references (warmstart); reuse an existing ``.sky.fits`` sidecar

    Steps 1-4 fan out across independent files with a thread pool of size
    ``ncores``; stages still run strictly in the order above.

    Parameters
    ----------
    files : sequence of path-like
        Science FITS (prefer ``*_jhat.fits``).
    outdir : path-like
        Staging directory (e.g. ``dolphot/wfc3_0_0`` or ``dolphot/hst_0_0``).
    instrument : str
        ``'acs'``, ``'wfc3'``, ``'wfpc2'``, or ``'hst'`` for a mixed run of
        all HST instruments against one reference coadd (per-frame mask/sky).
    refimage : path-like or None, optional
        Reference / coadd FITS. Defaults to the first science frame. May be a
        JWST/NIRCam coadd for warmstart (HST mask/sky on img0 are skipped).
    dolphot_bin : path-like or None, optional
        DOLPHOT ``bin`` override.
    skip_mask, skip_sky, skip_split : bool, optional
        Skip individual prep steps.
    copy_files : bool, optional
        Copy inputs into *outdir* before prep (default True).
    ncores : int, optional
        Parallel workers for per-file splitgroups / mask / calcsky.
    xytfile : path-like or None, optional
        Warm-start star list; basename written as ``xytfile`` in the paramfile.

    Returns
    -------
    pathlib.Path
        Path to the written ``dolphot.param``.
    """
    inst = instrument.lower()
    if inst not in _HST_INSTRUMENTS and inst != _HST_MIXED_INSTRUMENT:
        raise ValueError(
            f'prepare_hst_frames requires acs|wfc3|wfpc2|hst, got {instrument}'
        )
    mixed = inst == _HST_MIXED_INSTRUMENT
    workers = max(1, int(ncores or 1))

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    from st123.datamodels.hst import filter_paths_for_stage

    src_files = filter_paths_for_stage(
        [Path(p) for p in files],
        stage='dolphot-prep',
    )
    if not src_files:
        raise ValueError(
            'prepare_hst_frames: no science frames remain after EXPFLAG filtering'
        )
    ref_src = Path(refimage) if refimage is not None else src_files[0]
    ref_is_jwst = _is_jwst_dolphot_reference(ref_src)

    # Classify science inputs up front (mixed mode needs per-file instrument).
    src_kinds: dict[Path, str] = {}
    for src in src_files:
        if mixed:
            src_kinds[src.resolve()] = _classify_hst_science(src)
        else:
            src_kinds[src.resolve()] = (
                'wfc3_ir' if inst == 'wfc3' and 'ir' in src.name.lower() else inst
            )

    if copy_files:
        staged_ref = out / ref_src.name
        if not staged_ref.exists() or staged_ref.resolve() != ref_src.resolve():
            shutil.copy2(ref_src, staged_ref)
        # Preserve an existing JWST/NIRCam .sky next to the reference (warmstart).
        sky_src = _companion_sky_path(ref_src)
        if sky_src is not None:
            sky_dst = out / sky_src.name
            if not sky_dst.exists():
                shutil.copy2(sky_src, sky_dst)
        staged = []
        for src in src_files:
            dst = out / src.name
            if src.resolve() == ref_src.resolve():
                staged.append(staged_ref)
                continue
            if not dst.exists():
                shutil.copy2(src, dst)
            staged.append(dst)
            kind = src_kinds.get(src.resolve(), inst if not mixed else 'wfc3')
            # Stage WFPC2 c1m DQ beside science for wfpc2mask / drizzle parity.
            if _hst_mask_instrument(kind) == 'wfpc2' or (
                not mixed and inst == 'wfpc2'
            ):
                dq_src = _wfpc2_dq_companion(Path(src))
                if dq_src is not None:
                    dq_dst = out / dq_src.name
                    if not dq_dst.exists():
                        shutil.copy2(dq_src, dq_dst)
    else:
        staged_ref = ref_src
        staged = list(src_files)
        # Re-evaluate after staging in case only the staged copy is readable.
        ref_is_jwst = ref_is_jwst or _is_jwst_dolphot_reference(staged_ref)

    ref_is_coadd = any(
        tok in staged_ref.name.lower()
        for tok in ('_drc', '_drz', 'coadd_', '_i2d')
    )
    # Prefer header/name JWST detection over coadd heuristics.
    if not ref_is_jwst:
        ref_is_jwst = _is_jwst_dolphot_reference(staged_ref)
    # Split / mask / sky science MEFs only (not the drizzle reference).
    if ref_is_coadd:
        science_mefs = [
            p for p in staged if Path(p).resolve() != staged_ref.resolve()
        ]
    else:
        science_mefs = list(staged)
    if not science_mefs:
        science_mefs = list(staged)

    # ACS (and rare WFC3) archives can ship EXPTIME=0 with valid
    # EXPSTART/EXPEND; DOLPHOT rejects those frames. Repair before splitgroups.
    for mef in science_mefs:
        _repair_zero_exptime_fits(mef)

    # Map staged MEF -> mask instrument.
    mef_mask_inst: dict[Path, str] = {}
    for mef in science_mefs:
        mp = Path(mef)
        kind = None
        # Prefer classification from original src when names match.
        for src, k in src_kinds.items():
            if src.name == mp.name or src.stem in mp.name:
                kind = k
                break
        if kind is None:
            kind = _classify_hst_science(mp) if mixed else inst
        mef_mask_inst[mp.resolve()] = _hst_mask_instrument(kind)

    # WFPC2: mask native MEFs with c1m *before* splitgroups (C wfpc2mask needs DQ).
    if not skip_mask:
        wfpc2_mefs = [
            p
            for p in science_mefs
            if mef_mask_inst.get(Path(p).resolve()) == 'wfpc2'
            and '.chip' not in Path(p).name.lower()
        ]
        if wfpc2_mefs:
            apply_hst_mask(
                wfpc2_mefs,
                'wfpc2',
                dolphot_bin=dolphot_bin,
                ncores=workers,
            )

    if skip_split:
        work = list(science_mefs)
    else:
        work = apply_splitgroups(
            science_mefs, dolphot_bin=dolphot_bin, ncores=workers
        )
    # splitgroups copies primary EXPTIME; repair chips if MEF was already staged
    # with EXPTIME=0 before this fix (re-prep / skip_split paths).
    for chip in work:
        _repair_zero_exptime_fits(chip)

    # Group chip (or MEF) products by mask instrument for mask + calcsky.
    by_mask: dict[str, list[Path]] = {'acs': [], 'wfc3': [], 'wfpc2': []}
    for path in work:
        p = Path(path)
        mask_inst = None
        # Chip products: match parent MEF stem before ".chip".
        stem = p.name.split('.chip')[0] if '.chip' in p.name.lower() else p.stem
        for mef_res, mi in mef_mask_inst.items():
            if Path(mef_res).stem == stem:
                mask_inst = mi
                break
        if mask_inst is None:
            try:
                mask_inst = _hst_mask_instrument(_classify_hst_science(p))
            except ValueError:
                mask_inst = 'wfc3' if mixed else inst
        by_mask.setdefault(mask_inst, []).append(p)

    for mask_inst, group in by_mask.items():
        if not group:
            continue
        prepare_frames(
            group,
            instrument=mask_inst,
            dolphot_bin=dolphot_bin,
            # MEFs already masked for WFPC2; chips inherit BADPIX from splitgroups.
            skip_mask=skip_mask or mask_inst == 'wfpc2',
            skip_sky=skip_sky,
            ncores=workers,
        )

    # Coadd reference (img0): run *mask so CTX=0 / WHT=0 pixels become the
    # DOLPHOT DMIN ignore sentinel, then calcsky (needs a .sky for img0).
    # JWST/NIRCam refs are already nircammask'd + calcsky'd in the prior run.
    if ref_is_coadd and staged_ref.is_file() and not ref_is_jwst:
        ref_calc_inst = inst if not mixed else 'wfc3'
        if mixed:
            name = staged_ref.name.lower()
            if 'wfpc2' in name:
                ref_calc_inst = 'wfpc2'
            elif 'acs' in name:
                ref_calc_inst = 'acs'
            else:
                ref_calc_inst = 'wfc3'
        if not skip_mask:
            try:
                # Ensure uncovered SCI is sky-filled before mask rewrites DMIN.
                from st123.stages.mosaic.hst_drizzle import fill_drizzle_uncovered_with_sky

                fill_drizzle_uncovered_with_sky(staged_ref)
            except Exception as exc:
                logger.warning(
                    'CTX/sky fill on reference %s failed: %s',
                    staged_ref.name,
                    exc,
                )
            try:
                apply_hst_mask(
                    [staged_ref],
                    ref_calc_inst,
                    dolphot_bin=dolphot_bin,
                    ncores=1,
                )
            except Exception as exc:
                logger.warning(
                    '%smask on reference %s failed: %s',
                    ref_calc_inst,
                    staged_ref.name,
                    exc,
                )
        if not skip_sky:
            try:
                calc_sky(
                    [staged_ref],
                    instrument=ref_calc_inst,
                    dolphot_bin=dolphot_bin,
                    ncores=1,
                )
            except Exception as exc:
                logger.warning(
                    'calcsky on reference %s failed: %s',
                    staged_ref.name,
                    exc,
                )
        # Match chip-product layout (PRIMARY SCI) so DOLPHOT accepts img0.
        try:
            _flatten_dolphot_ref_and_sky(staged_ref)
        except Exception as exc:
            logger.warning(
                'Flatten reference %s for DOLPHOT failed: %s',
                staged_ref.name,
                exc,
            )
    elif ref_is_jwst:
        logger.info(
            'JWST reference %s: skipping HST mask/calcsky on img0 '
            '(reuse prior nircammask/calcsky products)',
            staged_ref.name,
        )
        sky = _companion_sky_path(staged_ref)
        if sky is None:
            logger.warning(
                'JWST reference %s has no .sky.fits sidecar; DOLPHOT may fail',
                staged_ref.name,
            )
        # NIRCam/MIRI i2d (+ sky) are still MEFs; HST chips are single-HDU.
        # Flatten so DOLPHOT does not abort with "Number of extensions...".
        # Flattened img0 is processed as EXTENSION 0; remap xyt col-1 from the
        # NIRCam SCI convention (1) so readwarm does not keep 0 stars.
        try:
            _flatten_dolphot_ref_and_sky(staged_ref)
            if xytfile is not None:
                xyt_path = Path(xytfile)
                if not xyt_path.is_file():
                    xyt_path = Path(outdir) / Path(xytfile).name
                if xyt_path.is_file():
                    remap_xyt_extension(xyt_path, 0)
        except Exception as exc:
            logger.warning(
                'Flatten JWST reference %s for DOLPHOT failed: %s',
                staged_ref.name,
                exc,
            )

    sci_for_param = [
        p for p in work
        if Path(p).resolve() != staged_ref.resolve()
    ]
    if not sci_for_param:
        sci_for_param = list(work)

    # Rewrite L2 chip + L3 ref WCS into DOLPHOT-parseable TAN / TAN-SIP
    # (strip orphaned Lookup/D2IM; ensure SIP <= order 5 + reverse coeffs).
    wcs_targets = [Path(p) for p in sci_for_param]
    if staged_ref.is_file():
        wcs_targets.append(Path(staged_ref))
    for path in wcs_targets:
        if not path.is_file() or path.stat().st_size == 0:
            continue
        if '.sky.' in path.name.lower():
            continue
        try:
            sanitize_dolphot_wcs(path)
        except Exception as exc:
            logger.warning('sanitize_dolphot_wcs(%s) failed: %s', path.name, exc)

    return setup_paramfile(
        out,
        staged_ref,
        sci_for_param,
        copy_files=False,
        global_params=HSTDataModel.DOLPHOT_BASE_PARAMS,
        phot_out=f'{out.name}.phot',
        xytfile=xytfile,
    )


def write_paramfile(
    outfile: PathLike,
    *,
    refimage: PathLike,
    images: Sequence[PathLike],
    global_params: Optional[Mapping[str, str]] = None,
    xytfile: Optional[PathLike] = None,
    image_kinds: Optional[Sequence[str]] = None,
) -> Path:
    """
    Write a DOLPHOT parameter file.

    Image bases are basenames without ``.fits``. Per-image params are chosen
    from ``short`` / ``long`` / ``miri`` via :func:`classify_image_kind` unless
    *image_kinds* is provided.

    Parameters
    ----------
    outfile : str or os.PathLike
        Output path for ``dolphot.param``.
    refimage : str or os.PathLike
        Reference image path (basename written as ``img0_file``).
    images : sequence of str or os.PathLike
        Science image paths (basenames written as ``img1_file``, ...).
    global_params : dict or None, optional
        Global DOLPHOT keywords merged into the parameter file. Defaults to
        :attr:`JWSTDataModel.DOLPHOT_BASE_PARAMS`, with MIRI keys added when
        any MIRI frame is present.
    xytfile : str or os.PathLike or None, optional
        Warm-start star list; basename written as ``xytfile``.
    image_kinds : sequence of str or None, optional
        Explicit per-image kinds (``'short'``, ``'long'``, ``'miri'``). Must
        match the length of *images* when provided.

    Returns
    -------
    pathlib.Path
        Path to the written parameter file.

    Raises
    ------
    ValueError
        If ``image_kinds`` length does not match ``images``.
    """
    out = Path(outfile)
    out.parent.mkdir(parents=True, exist_ok=True)

    ref_base = Path(refimage).name.replace('.fits', '')
    img_bases = [Path(p).name.replace('.fits', '') for p in images]
    kinds = list(image_kinds) if image_kinds is not None else [
        classify_image_kind(p) for p in images
    ]
    if len(kinds) != len(img_bases):
        raise ValueError('image_kinds length must match images')

    # Include MIRI / HST global knobs when those frames are present.
    gparams = (
        dict(global_params)
        if global_params is not None
        else dict(JWSTDataModel.DOLPHOT_BASE_PARAMS)
    )
    if any(k == 'miri' for k in kinds):
        merged = dict(MIRIDataModel.DOLPHOT_BASE_PARAMS)
        merged.update(gparams)
        # Keep caller overrides, but ensure MIRI-required keys exist.
        if 'MIRIvega' not in merged:
            merged['MIRIvega'] = MIRIDataModel.DOLPHOT_BASE_PARAMS['MIRIvega']
        if 'UseWCS' not in merged:
            merged['UseWCS'] = '2'
        gparams = merged
    if any(k in _HST_IMAGE_KINDS for k in kinds):
        merged = dict(HSTDataModel.DOLPHOT_BASE_PARAMS)
        merged.update(gparams)
        if 'UseWCS' not in merged:
            merged['UseWCS'] = '1'
        gparams = merged

    with out.open('w') as f:
        f.write(f'Nimg = {len(img_bases)}\n')
        f.write(f'img0_file = {ref_base}\n')
        for i, (img, kind) in enumerate(zip(img_bases, kinds), start=1):
            f.write(f'img{i}_file = {img}\n')
            for key, val in per_image_params(kind).items():
                f.write(f'img{i}_{key} = {val}\n')
        if xytfile is not None:
            f.write(f'xytfile = {Path(xytfile).name}\n')
        for key, val in gparams.items():
            f.write(f'{key} = {val}\n')
    return out


def setup_paramfile(
    phot_outdir: PathLike,
    refimage: PathLike,
    files: Sequence[PathLike],
    *,
    copy_files: bool = True,
    global_params: Optional[Mapping[str, str]] = None,
    xytfile: Optional[PathLike] = None,
    phot_out: Optional[str] = None,
    max_nimg: Optional[int] = None,
    return_plan: bool = False,
):
    """
    Stage images into a photometry directory and write ``dolphot.param``.

    Supports NIRCam and MIRI frames and an optional warm-start ``xytfile``.
    When the science-image count exceeds the soft limit
    (:data:`st123.stages.photometry.dolphot_split.DOLPHOT_MAX_NIMG`, default 400),
    the run is split into roughly equal parts that share the same reference
    (see :func:`st123.stages.photometry.dolphot_split.write_split_paramfiles`).

    Parameters
    ----------
    phot_outdir : str or os.PathLike
        Directory to stage frames and write ``dolphot.param``.
    refimage : str or os.PathLike
        Reference / coadd FITS path.
    files : sequence of str or os.PathLike
        Science FITS paths to include after the reference.
    copy_files : bool, optional
        If True (default), copy *refimage* and *files* into *phot_outdir*
        before writing the parameter file.
    global_params : dict or None, optional
        Global DOLPHOT keywords forwarded to :func:`write_paramfile`.
    xytfile : str or os.PathLike or None, optional
        Warm-start star list copied into *phot_outdir* when ``copy_files`` is
        True.
    phot_out : str or None, optional
        Final catalog basename (used for split part names / merge target).
    max_nimg : int or None, optional
        Soft per-run image cap. Defaults to ``DOLPHOT_MAX_NIMG`` (400).
    return_plan : bool, optional
        If True, return a :class:`~st123.stages.photometry.dolphot_split.DolphotRunPlan`
        instead of the primary parameter-file path.

    Returns
    -------
    pathlib.Path or DolphotRunPlan
        Primary ``dolphot.param`` path, or the full run plan when
        ``return_plan`` is True.
    """
    from st123.stages.photometry.dolphot_split import (
        DOLPHOT_MAX_NIMG,
        write_split_paramfiles,
    )

    outdir = Path(phot_outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if copy_files:
        from st123.datamodels import sanitize_science_fits

        shutil.copy(refimage, outdir)
        sanitize_science_fits(outdir / Path(refimage).name, materialize_headers=True)
        for src in files:
            shutil.copy(src, outdir)
            sanitize_science_fits(outdir / Path(src).name, materialize_headers=True)

    staged_ref = outdir / Path(refimage).name if copy_files else Path(refimage)
    staged = (
        [outdir / Path(p).name for p in files]
        if copy_files
        else [Path(p) for p in files]
    )
    xyt_name = None
    if xytfile is not None:
        xyt_src = Path(xytfile)
        xyt_dst = outdir / xyt_src.name
        if xyt_src.resolve() != xyt_dst.resolve():
            shutil.copy(xyt_src, xyt_dst)
        xyt_name = xyt_dst

    if phot_out is None:
        phot_out = f'{outdir.name}.phot'
    cap = DOLPHOT_MAX_NIMG if max_nimg is None else int(max_nimg)
    plan = write_split_paramfiles(
        outdir,
        refimage=staged_ref,
        images=staged,
        phot_out=phot_out,
        global_params=global_params,
        xytfile=xyt_name,
        max_nimg=cap,
    )
    return plan if return_plan else plan.param_file


def phot_to_xyt(
    photfile: PathLike,
    xyt_file: PathLike,
    *,
    types: Optional[Iterable[int]] = None,
    snr_min: Optional[float] = None,
    crowd_max: Optional[float] = None,
    sharp2_max: Optional[float] = None,
    min_sep_pix: Optional[float] = None,
    force_xy: Optional[Sequence[tuple[float, float]]] = None,
    force_tol_pix: float = 0.05,
    center_xy: Optional[tuple[float, float]] = None,
    max_radius_pix: Optional[float] = None,
) -> Path:
    """
    Build a warm-start star list from a DOLPHOT ``.phot`` catalog.

    Columns (1-based from the DOLPHOT manual): extension, Z, X, Y, type (col
    11), and SNR (col 6). Extension, Z, and type are written as integers.
    Optional quality cuts use sharpness (col 7) and crowding (col 10).

    When ``min_sep_pix`` is set, survivors are ranked by descending SNR and
    kept greedily so no two retained seeds are closer than that separation
    (MIRI-appropriate thinning). Coordinates in ``force_xy`` are always kept
    if present in the catalog (even when they fail quality cuts).

    Parameters
    ----------
    photfile : str or os.PathLike
        Input DOLPHOT ``.phot`` catalog path.
    xyt_file : str or os.PathLike
        Output warm-start list path (typically ``warmstart.xyt``).
    types : iterable of int or None, optional
        If provided, keep only objects whose DOLPHOT type appears in this set.
    snr_min : float or None, optional
        Minimum object SNR (DOLPHOT column 6).
    crowd_max : float or None, optional
        Maximum crowding (DOLPHOT column 10).
    sharp2_max : float or None, optional
        Maximum ``sharpness**2`` (DOLPHOT column 7); e.g. ``0.01``.
    min_sep_pix : float or None, optional
        Minimum separation in reference-image pixels between retained seeds.
    force_xy : sequence of (x, y) or None, optional
        Positions that must be retained when present in the catalog.
    force_tol_pix : float, optional
        Match tolerance for ``force_xy`` (pixels).
    center_xy : (x, y) or None, optional
        Reference-pixel center for the optional radius cut. Defaults to the
        first ``force_xy`` coordinate when that is provided.
    max_radius_pix : float or None, optional
        Keep only seeds within this radius of *center_xy* (plus any
        ``force_xy`` matches). Useful for single-target warmstarts.

    Returns
    -------
    pathlib.Path
        Path to the written ``xyt`` file.

    Raises
    ------
    ValueError
        If no stars pass the filters and are written to the output.
    """
    phot_path = Path(photfile)
    out = Path(xyt_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    type_set = set(int(t) for t in types) if types is not None else None
    force_list = list(force_xy) if force_xy else []
    radius_center = center_xy
    if radius_center is None and force_list:
        radius_center = (float(force_list[0][0]), float(force_list[0][1]))
    use_radius = (
        max_radius_pix is not None
        and max_radius_pix > 0
        and radius_center is not None
    )
    if max_radius_pix is not None and max_radius_pix > 0 and radius_center is None:
        raise ValueError(
            'max_radius_pix requires center_xy or force_xy to define the center'
        )

    # Collect rows that pass quality cuts (or are force-matched).
    # Each item: (snr, ext, z, x, y, obj_type, snr_str, forced)
    candidates: list[tuple[float, int, int, str, str, int, str, bool]] = []
    forced_rows: list[tuple[float, int, int, str, str, int, str, bool]] = []

    with phot_path.open() as fin:
        for line in fin:
            parts = line.split()
            if len(parts) < 11:
                continue
            ext = int(float(parts[0]))
            z = int(float(parts[1]))
            x_str, y_str = parts[2], parts[3]
            x = float(x_str)
            y = float(y_str)
            snr = float(parts[5])
            sharp = float(parts[6])
            crowd = float(parts[9])
            obj_type = int(float(parts[10]))
            snr_str = parts[5]

            is_forced = any(
                abs(x - fx) <= force_tol_pix and abs(y - fy) <= force_tol_pix
                for fx, fy in force_list
            )
            if is_forced:
                forced_rows.append(
                    (snr, ext, z, x_str, y_str, obj_type, snr_str, True)
                )
                continue

            if use_radius:
                cx, cy = radius_center
                if (x - cx) ** 2 + (y - cy) ** 2 > float(max_radius_pix) ** 2:
                    continue
            if type_set is not None and obj_type not in type_set:
                continue
            if snr_min is not None and snr < snr_min:
                continue
            if crowd_max is not None and crowd > crowd_max:
                continue
            if sharp2_max is not None and sharp * sharp > sharp2_max:
                continue
            candidates.append(
                (snr, ext, z, x_str, y_str, obj_type, snr_str, False)
            )

    # Greedy min-separation (when requested): forced seeds first, then
    # brightest remaining. Otherwise preserve catalog order (forced rows
    # that skipped quality cuts are emitted first, then quality survivors).
    kept: list[tuple[float, int, int, str, str, int, str, bool]] = []
    kept_xy: list[tuple[float, float]] = []
    use_min_sep = min_sep_pix is not None and min_sep_pix > 0

    def _accept(row: tuple[float, int, int, str, str, int, str, bool]) -> bool:
        x = float(row[3])
        y = float(row[4])
        if use_min_sep:
            for kx, ky in kept_xy:
                if (x - kx) ** 2 + (y - ky) ** 2 < float(min_sep_pix) ** 2:
                    return False
        kept.append(row)
        kept_xy.append((x, y))
        return True

    if use_min_sep:
        ordered = sorted(forced_rows, key=lambda r: -r[0]) + sorted(
            candidates, key=lambda r: -r[0]
        )
    else:
        ordered = list(forced_rows) + list(candidates)
    for row in ordered:
        _accept(row)

    with out.open('w') as fout:
        for _snr, ext, z, x_str, y_str, obj_type, snr_str, _forced in kept:
            fout.write(f'{ext} {z} {x_str} {y_str} {obj_type} {snr_str}\n')

    if not kept:
        raise ValueError(f'No stars written to warm-start list from {phot_path}')
    logger.info(
        'Wrote %s (%d stars; forced=%d, quality=%d%s)',
        out,
        len(kept),
        sum(1 for r in kept if r[7]),
        sum(1 for r in kept if not r[7]),
        (
            f', max_radius_pix={max_radius_pix:g}'
            if use_radius
            else ''
        ),
    )
    return out


# Recommended warm-start seed cuts for MIRI (match catalog.save_photfiles
# quality cuts, plus a MIRI-scale minimum separation).
MIRI_WARMSTART_XYT_TYPES = (1,)
MIRI_WARMSTART_SNR_MIN = 10.0
MIRI_WARMSTART_CROWD_MAX = 0.5
MIRI_WARMSTART_SHARP2_MAX = 0.01
# ~0.30" on a 0.031"/pix NIRCam reference -> ~10 pix.
MIRI_WARMSTART_MIN_SEP_ARCSEC = 0.30

# HST warmstart from NIRCam: denser fields -> milder SNR / separation cuts.
HST_WARMSTART_XYT_TYPES = (1,)
HST_WARMSTART_SNR_MIN = 5.0
HST_WARMSTART_CROWD_MAX = 0.5
HST_WARMSTART_SHARP2_MAX = 0.01
HST_WARMSTART_MIN_SEP_ARCSEC = 0.15


def nearest_phot_source(
    phot_file: PathLike,
    refimage: PathLike,
    *,
    ra: float,
    dec: float,
    n: int = 5,
) -> list[dict]:
    """
    Return the *n* nearest DOLPHOT catalog rows to (*ra*, *dec*).

    Uses the reference image WCS to convert sky -> pixel, then sorts by
    Euclidean distance in reference pixels. Each row dict includes global
    fit columns (x, y, chi, snr, sharp, crowd, type) plus ``dist_pix``.

    Parameters
    ----------
    phot_file : path-like
        DOLPHOT ``.phot`` catalog.
    refimage : path-like
        Reference FITS used as ``img0`` (WCS for the catalog x/y).
    ra, dec : float
        ICRS coordinates in degrees.
    n : int, optional
        Number of nearest neighbors to return (default 5).

    Returns
    -------
    list of dict
        Nearest sources, closest first.
    """
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS

    phot = Path(phot_file)
    data = np.loadtxt(phot)
    if data.ndim == 1:
        data = data[None, :]
    with as_datamodel(refimage).open() as hdul:
        hdu = hdul['SCI'] if 'SCI' in hdul else hdul[0]
        wcs = WCS(hdu.header)
    x0, y0 = wcs.world_to_pixel(SkyCoord(ra, dec, unit='deg'))
    x0, y0 = float(x0), float(y0)
    dist = np.hypot(data[:, 2] - x0, data[:, 3] - y0)
    order = np.argsort(dist)[: max(1, int(n))]
    out: list[dict] = []
    for j in order:
        row = data[j]
        out.append(
            {
                'index': int(j),
                'dist_pix': float(dist[j]),
                'x': float(row[2]),
                'y': float(row[3]),
                'chi': float(row[4]),
                'snr': float(row[5]),
                'sharp': float(row[6]),
                'crowd': float(row[9]),
                'type': int(row[10]),
                'ra': ra,
                'dec': dec,
                'target_x': x0,
                'target_y': y0,
            }
        )
    return out


def parse_param_image_list(param_file: PathLike) -> tuple[str, list[str]]:
    """
    Parse image basenames from a DOLPHOT parameter file.

    Parameters
    ----------
    param_file : str or os.PathLike
        Path to ``dolphot.param``.

    Returns
    -------
    ref_base : str
        ``img0_file`` basename (without ``.fits``).
    image_bases : list of str
        ``img1_file``, ``img2_file``, ... basenames in parameter-file order.

    Raises
    ------
    ValueError
        If ``img0_file`` is missing.
    """
    ref = None
    images: list[str] = []
    nimg = None
    with Path(param_file).open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, val = [x.strip() for x in line.split('=', 1)]
            if key == 'Nimg':
                nimg = int(val)
            elif key == 'img0_file':
                ref = val
            elif key.endswith('_file') and key.startswith('img'):
                images.append(val)
    if ref is None:
        raise ValueError(f'No img0_file in {param_file}')
    if nimg is not None and len(images) != nimg:
        # Prefer the explicit file list if Nimg is stale.
        pass
    return ref, images


def dolphot_command(
    outdir: PathLike,
    *,
    phot_out: str,
    param_file: str = 'dolphot.param',
    dolphot_bin: Optional[PathLike] = None,
    ncores: int = 1,
    nohup: bool = False,
) -> str:
    """
    Build a shell command to run DOLPHOT (does not execute it).

    Usage matches the binary::

        dolphot <output> -p<paramfile> MaxThreads=<ncores>

    ``ncores`` is the same value as the CLI ``--ncores`` / ``--workers`` flag.
    Resolves the DOLPHOT bin from ``dolphot_bin`` or ``PATH``; if missing,
    warns and emits a command that relies on the caller's ``PATH`` alone.

    Parameters
    ----------
    outdir : str or os.PathLike
        Working directory for the DOLPHOT run (``cd`` target in the command).
    phot_out : str
        Output catalog basename passed as the first ``dolphot`` argument.
    param_file : str, optional
        Parameter file basename (default ``'dolphot.param'``).
    dolphot_bin : str or os.PathLike or None, optional
        Override path to the DOLPHOT ``bin`` directory prepended to ``PATH``.
    ncores : int, optional
        ``MaxThreads`` value for DOLPHOT (minimum 1).
    nohup : bool, optional
        If True, wrap as a detachable ``nohup`` job writing ``dolphot.out`` /
        ``dolphot.err`` in *outdir*.

    Returns
    -------
    str
        Shell command string ready to copy and run.
    """
    out = Path(outdir).resolve()
    threads = max(1, int(ncores))
    bin_dir = resolve_dolphot_bin(dolphot_bin, required=False)
    # Use ``env`` so nohup does not treat ``PATH=...`` as the executable.
    path_prefix = (
        f'env PATH={bin_dir}:$PATH ' if bin_dir is not None else ''
    )
    run = f'{path_prefix}dolphot {phot_out} -p{param_file} MaxThreads={threads}'
    if nohup:
        return (
            f'cd {out} && '
            f'nohup {run} > dolphot.out 2> dolphot.err < /dev/null &'
        )
    return f'cd {out} && {run}'
