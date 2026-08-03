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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence, Union

from astropy.io import fits

from st123.utils.helpers import get_detector_chip
from st123.utils.logging import run_logged_subprocess
from st123.utils.settings import (
    acs_calcsky_params,
    acs_params,
    base_params,
    hst_base_params,
    long_params,
    miri_base_params,
    miri_calcsky_params,
    miri_params,
    nircam_calcsky_params,
    short_params,
    wfc3_calcsky_params,
    wfc3_ir_params,
    wfc3_params,
    wfpc2_calcsky_params,
    wfpc2_params,
)

PathLike = Union[str, os.PathLike]

logger = logging.getLogger(__name__)

_FRAME_LIST_NAME = 'dolphot_frames.txt'
_GROUP_BOX_RE = re.compile(r'group_(\d+)/ref_(\d+)')
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
    """Map :func:`classify_image_kind` result → ``acsmask`` / ``wfc3mask`` / ``wfpc2mask``."""
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
        # WFPC2 roots are often uNNNN…; prefer header, but filename last-resort.
        if 'wfpc2' in name or '_c0m' in name:
            return 'wfpc2'
    if 'acs' in name:
        return 'acs'
    if 'wfc3' in name or name.startswith('i'):
        return 'wfc3'
    raise ValueError(f'Cannot classify HST frame for DOLPHOT prep: {path}')


@dataclass(frozen=True)
class MosaicPhotJob:
    """One mosaic box ready for DOLPHOT staging.

    Attributes
    ----------
    group : int
        Mosaic group index (``group_<n>`` directory name).
    box : int
        Reference box index within the group (``ref_<n>`` directory name).
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
    box: int
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
    ``FileNotFoundError`` — use this before programmatically invoking
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


def parse_dolphot_frame_list(path: PathLike) -> tuple[Path, list[Path], int, int]:
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
    box : int
        Reference box index from the ``# box=`` header or inferred from the path.

    Raises
    ------
    ValueError
        If the manifest lacks a ``# ref`` line or contains no frame paths.
    """
    path = Path(path)
    text = path.read_text()
    refimage: Path | None = None
    frames: list[Path] = []
    group, box = 0, 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('# group='):
            # ``# group=0 box=1``
            parts = stripped[1:].split()
            for part in parts:
                if part.startswith('group='):
                    group = int(part.split('=', 1)[1])
                elif part.startswith('box='):
                    box = int(part.split('=', 1)[1])
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
    if group == 0 and box == 0:
        match = _GROUP_BOX_RE.search(path.as_posix())
        if match:
            group, box = int(match.group(1)), int(match.group(2))
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
            for p in box.glob('coadd_*_i2d.fits')
            if f'_{key}_' in p.name.lower()
        )
        if matches:
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
    manifest, pairs it with all ``jhat/*jhat.fits`` under *reduction_dir*.

    Parameters
    ----------
    reduction_dir : str or os.PathLike
        Mosaic reduction root (contains ``reference/`` and optionally ``jhat/``).
    instrument : str or None, optional
        When ``'miri'`` or ``'nircam'``, keep only matching science frames.
    ref_filter : str or None, optional
        Prefer a coadd whose filename contains this filter (e.g. ``'F560W'``).
    phot_outdir_root : str or os.PathLike or None, optional
        Parent directory for staging runs. Default: *reduction_dir*.
        MIRI-only runs typically use ``<project>/dolphot``.
    outdir_prefix : str, optional
        Staging directory name prefix (default ``'phot'`` → ``phot_0_0``;
        use ``'miri'`` → ``miri_0_0``).

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

    manifests = sorted(ref_root.glob('group_*/ref_*/' + _FRAME_LIST_NAME))
    if manifests:
        for manifest in manifests:
            refimage, frames, group, box = parse_dolphot_frame_list(manifest)
            refimage = resolve_coadd_ref(
                manifest.parent,
                ref_filter,
                fallback=refimage,
            )
            if instrument is not None:
                frames = filter_frames_for_instrument(frames, instrument)
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
    jhat = sorted((root / 'jhat').glob('*jhat.fits'))
    if instrument is not None:
        jhat = filter_frames_for_instrument(jhat, instrument)
    for coadd in sorted(ref_root.glob('group_*/ref_*/coadd_*_i2d.fits')):
        match = _GROUP_BOX_RE.search(coadd.as_posix())
        group = int(match.group(1)) if match else 0
        box = int(match.group(2)) if match else 0
        refimage = resolve_coadd_ref(coadd.parent, ref_filter, fallback=coadd)
        # When filtering by ref_filter, skip coadds that are not the chosen ref.
        if ref_filter and refimage.resolve() != coadd.resolve():
            continue
        if not jhat:
            continue
        jobs.append(
            MosaicPhotJob(
                group=group,
                box=box,
                refimage=refimage,
                frames=tuple(jhat),
                phot_outdir=out_root / f'{outdir_prefix}_{group}_{box}',
                frame_list=coadd.parent / _FRAME_LIST_NAME,
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
        :data:`st123.utils.settings.miri_base_params`.
    dolphot_bin : str or os.PathLike or None, optional
        Override path to the DOLPHOT ``bin`` directory.
    skip_mask : bool, optional
        If True, skip ``nircammask`` / ``mirimask``.
    skip_sky : bool, optional
        If True, skip ``calcsky``.
    copy_files : bool, optional
        If True (default), copy reference and science frames into
        ``job.phot_outdir`` before prep.

    Returns
    -------
    pathlib.Path
        Path to the written ``dolphot.param`` file.
    """
    inst = instrument.lower()
    global_params = miri_base_params if inst == 'miri' else None
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
    name = os.path.basename(str(path)).lower()
    chip = get_detector_chip(str(path))
    if chip:
        chip_l = chip.lower()
        if 'mir' in chip_l:
            return 'miri'
        if 'long' in chip_l:
            return 'long'
        if 'nrc' in chip_l:
            return 'short'

    if 'mirimage' in name or '/miri/' in str(path).lower():
        return 'miri'

    try:
        with fits.open(path, memmap=True) as hdul:
            for hdu in hdul:
                inst = str(hdu.header.get('INSTRUME', '')).upper()
                if inst == 'MIRI':
                    return 'miri'
                if inst == 'NIRCAM':
                    det = str(hdu.header.get('DETECTOR', '')).upper()
                    return 'long' if 'LONG' in det else 'short'
                if inst == 'ACS':
                    return 'acs'
                if inst == 'WFPC2':
                    return 'wfpc2'
                if inst == 'WFC3':
                    det = str(hdu.header.get('DETECTOR', '')).upper()
                    aper = str(hdul[0].header.get('APERTURE', '')).upper()
                    phot = str(hdu.header.get('PHOTMODE', '')).upper()
                    if 'IR' in det or aper.startswith('IR') or ' WFC3 IR' in f' {phot}':
                        return 'wfc3_ir'
                    return 'wfc3'
            # JHAT may strip INSTRUME; fall back to PHOTMODE / APERTURE.
            for hdu in hdul:
                phot = str(hdu.header.get('PHOTMODE', '')).upper()
                if phot.startswith('ACS') or ',ACS' in phot:
                    return 'acs'
                if phot.startswith('WFPC2') or 'WFPC2,' in phot:
                    return 'wfpc2'
                if phot.startswith('WFC3'):
                    if ' IR' in f' {phot}' or phot.split()[1:2] == ['IR']:
                        return 'wfc3_ir'
                    return 'wfc3'
            aper = str(hdul[0].header.get('APERTURE', '')).upper()
            if aper.startswith('UVIS'):
                return 'wfc3'
            if aper.startswith('IR'):
                return 'wfc3_ir'
            if aper.startswith('WFC') or aper.startswith('HRC'):
                return 'acs'
    except OSError:
        pass

    raise ValueError(f'Cannot classify DOLPHOT image kind for {path}')


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
        return short_params
    if kind == 'long':
        return long_params
    if kind == 'miri':
        return miri_params
    if kind == 'acs':
        return acs_params
    if kind == 'wfc3':
        return wfc3_params
    if kind == 'wfc3_ir':
        return wfc3_ir_params
    if kind == 'wfpc2':
        return wfpc2_params
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

    Returns
    -------
    None
    """
    bin_dir = dolphot_bin_dir(dolphot_bin)
    cwd, names = _run_cwd_for_files(files)
    cmd = [str(bin_dir / 'nircammask')]
    if estnoise:
        cmd.append('-estnoise')
    if noetctime:
        cmd.append('-noetctime')
    cmd.extend(names)
    run_logged_subprocess(
        cmd,
        check=check,
        env=_prepend_bin_env(None, bin_dir),
        cwd=cwd,
        logger=logger,
        label=f'nircammask ({len(names)} file(s))',
    )


def apply_mirimask(
    files: Sequence[PathLike],
    *,
    dolphot_bin: Optional[PathLike] = None,
    estnoise: bool = True,
    noetctime: bool = False,
    mask_lyot: bool = False,
    check: bool = True,
) -> None:
    """
    Run ``mirimask`` on science frames (in-place).

    Default flags follow ``dolphotMIRI.pdf`` §3.3 recommendations:
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

    Returns
    -------
    None
    """
    bin_dir = dolphot_bin_dir(dolphot_bin)
    cwd, names = _run_cwd_for_files(files)
    cmd = [str(bin_dir / 'mirimask')]
    if estnoise:
        cmd.append('-estnoise')
    if noetctime:
        cmd.append('-noetctime')
    if mask_lyot:
        cmd.append('-mask_lyot')
    cmd.extend(names)
    run_logged_subprocess(
        cmd,
        check=check,
        env=_prepend_bin_env(None, bin_dir),
        cwd=cwd,
        logger=logger,
        label=f'mirimask ({len(names)} file(s))',
    )


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
) -> None:
    """
    Run DOLPHOT ``calcsky`` on each science frame.

    Defaults follow the instrument manuals / hst123 detector defaults:

    - NIRCam: rin=15, rout=25, step=-64, σ=2.25/2.00
    - MIRI: rin=10, rout=25, step=-64, σ=2.25/2.00 (``dolphotMIRI.pdf`` §3.4)
    - ACS / WFC3 UVIS: rin=15, rout=35, step=4
    - WFPC2: rin=10, rout=25, step=2

    When all frames share a directory, ``calcsky`` is invoked with basenames
    and ``cwd`` set to that directory to avoid C path-buffer overflows.

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

    Returns
    -------
    None
    """
    inst = instrument.lower()
    if inst == 'miri':
        defaults = miri_calcsky_params
    elif inst == 'acs':
        defaults = acs_calcsky_params
    elif inst == 'wfc3':
        defaults = wfc3_calcsky_params
    elif inst == 'wfpc2':
        defaults = wfpc2_calcsky_params
    else:
        defaults = nircam_calcsky_params
    rin = defaults['rin'] if rin is None else rin
    rout = defaults['rout'] if rout is None else rout
    step = defaults['step'] if step is None else step
    sigma_low = defaults['sigma_low'] if sigma_low is None else sigma_low
    sigma_high = defaults['sigma_high'] if sigma_high is None else sigma_high

    bin_dir = dolphot_bin_dir(dolphot_bin)
    calcsky = str(bin_dir / 'calcsky')
    env = _prepend_bin_env(None, bin_dir)
    cwd, names = _run_cwd_for_files(files)

    for name in names:
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


def _wfpc2_dq_companion(science: Path) -> Optional[Path]:
    """Locate ``*_c1m.fits`` next to a WFPC2 ``*_c0m`` / ``*_jhat`` MEF."""
    name = science.name
    candidates: list[Path] = []
    if name.endswith('_c0m.fits'):
        candidates.append(science.with_name(name.replace('_c0m.fits', '_c1m.fits')))
    elif name.endswith('_jhat.fits'):
        stem = name[: -len('_jhat.fits')]
        candidates.append(science.with_name(f'{stem}_c1m.fits'))
        # Also look beside the original raw name if JHAT kept the root.
        candidates.append(science.with_name(f'{stem}_c0m'.replace('_c0m', '') + '_c1m.fits'))
        raw_sib = science.parent.parent / 'raw' / f'{stem}_c1m.fits'
        candidates.append(raw_sib)
        candidates.append(science.parent / f'{stem}_c1m.fits')
    else:
        stem = name[:-5] if name.endswith('.fits') else name
        candidates.append(science.with_name(f'{stem}_c1m.fits'))
        if '_c0m' in stem:
            candidates.append(
                science.with_name(stem.replace('_c0m', '_c1m') + '.fits')
            )
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def apply_hst_mask(
    files: Sequence[PathLike],
    instrument: str,
    *,
    dolphot_bin: Optional[PathLike] = None,
    check: bool = True,
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
    if inst == 'wfpc2':
        # Pair each MEF with its c1m; chip products have no separate DQ file.
        cmd: list[str] = [str(exe)]
        for sci in paths:
            cmd.append(sci.name)
            if '.chip' in sci.name.lower():
                continue
            dq = _wfpc2_dq_companion(sci)
            if dq is None:
                logger.warning(
                    'WFPC2 c1m missing for %s; wfpc2mask may fail on native MEF',
                    sci.name,
                )
                continue
            # Ensure DQ sits next to science for cwd-relative argv.
            dq_local = sci.parent / dq.name
            if dq.resolve() != dq_local.resolve():
                shutil.copy2(dq, dq_local)
            cmd.append(dq_local.name)
        cwd = str(paths[0].parent) if paths else None
        run_logged_subprocess(
            cmd,
            check=check,
            env=_prepend_bin_env(None, bin_dir),
            cwd=cwd,
            logger=logger,
            label=f'{exe_name} ({len(paths)} file(s), c1m-paired)',
        )
        return

    cwd, names = _run_cwd_for_files(files)
    cmd = [str(exe), *names]
    run_logged_subprocess(
        cmd,
        check=check,
        env=_prepend_bin_env(None, bin_dir),
        cwd=cwd,
        logger=logger,
        label=f'{exe_name} ({len(names)} file(s))',
    )


def apply_splitgroups(
    files: Sequence[PathLike],
    *,
    dolphot_bin: Optional[PathLike] = None,
    check: bool = True,
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
    chip_files: list[Path] = []
    for path in files:
        src = Path(path).resolve()
        # Remove stale chip products so we do not mix generations.
        for old in src.parent.glob(f'{src.name.replace(".fits", "")}.chip*.fits'):
            # stem.chipN.fits — also handle name.fits → name.chipN.fits
            pass
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
            chip_files.append(src)
            continue
        # Propagate INSTRUME/FILTER from the parent MEF (JHAT often strips them).
        parent_inst = None
        parent_filt = None
        try:
            from st123.utils.helpers import get_filter, get_instrument

            parent_inst = get_instrument(src).split('_')[0].upper()
            parent_filt = get_filter(src).upper()
        except Exception:
            pass
        for chip in chips:
            keep = False
            try:
                with fits.open(chip, mode='update') as hdul:
                    hdr = hdul[0].header
                    extname = str(hdr.get('EXTNAME') or '').upper()
                    data = hdul[0].data
                    shape = () if data is None else tuple(data.shape)
                    # Real science chips are 2-D with a substantial footprint.
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
                chip_files.append(chip)
            else:
                try:
                    chip.unlink()
                except OSError:
                    pass
    return chip_files


def prepare_frames(
    files: Sequence[PathLike],
    *,
    instrument: str,
    dolphot_bin: Optional[PathLike] = None,
    skip_mask: bool = False,
    skip_sky: bool = False,
) -> None:
    """
    Mask then compute sky for science frames.

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

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If *instrument* is not supported.
    """
    inst = instrument.lower()
    if not skip_mask:
        if inst == 'miri':
            apply_mirimask(files, dolphot_bin=dolphot_bin)
        elif inst == 'nircam':
            apply_nircammask(files, dolphot_bin=dolphot_bin)
        elif inst in _HST_INSTRUMENTS:
            apply_hst_mask(files, inst, dolphot_bin=dolphot_bin)
        else:
            raise ValueError(f'Unsupported instrument for prepare_frames: {instrument}')
    if not skip_sky:
        calc_sky(files, instrument=inst, dolphot_bin=dolphot_bin)


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
) -> Path:
    """
    Stage HST frames for DOLPHOT: splitgroups → mask → calcsky → paramfile.

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
        Reference / coadd FITS. Defaults to the first science frame.
    dolphot_bin : path-like or None, optional
        DOLPHOT ``bin`` override.
    skip_mask, skip_sky, skip_split : bool, optional
        Skip individual prep steps.
    copy_files : bool, optional
        Copy inputs into *outdir* before prep (default True).

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

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    src_files = [Path(p) for p in files]
    if not src_files:
        raise ValueError('prepare_hst_frames requires at least one science frame')
    ref_src = Path(refimage) if refimage is not None else src_files[0]

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

    ref_is_coadd = any(
        tok in staged_ref.name.lower()
        for tok in ('_drc', '_drz', 'coadd_')
    )
    # Split / mask / sky science MEFs only (not the drizzle reference).
    if ref_is_coadd:
        science_mefs = [
            p for p in staged if Path(p).resolve() != staged_ref.resolve()
        ]
    else:
        science_mefs = list(staged)
    if not science_mefs:
        science_mefs = list(staged)

    # Map staged MEF → mask instrument.
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
            apply_hst_mask(wfpc2_mefs, 'wfpc2', dolphot_bin=dolphot_bin)

    if skip_split:
        work = list(science_mefs)
    else:
        work = apply_splitgroups(science_mefs, dolphot_bin=dolphot_bin)

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
        )

    # Optional: calcsky on the coadd reference (DOLPHOT img0 often wants .sky).
    if not skip_sky and ref_is_coadd and staged_ref.is_file():
        ref_calc_inst = inst if not mixed else 'wfc3'
        if mixed:
            name = staged_ref.name.lower()
            if 'wfpc2' in name:
                ref_calc_inst = 'wfpc2'
            elif 'acs' in name:
                ref_calc_inst = 'acs'
            else:
                ref_calc_inst = 'wfc3'
        try:
            calc_sky([staged_ref], instrument=ref_calc_inst, dolphot_bin=dolphot_bin)
        except Exception as exc:
            logger.warning('calcsky on reference %s failed: %s', staged_ref.name, exc)

    sci_for_param = [
        p for p in work
        if Path(p).resolve() != staged_ref.resolve()
    ]
    if not sci_for_param:
        sci_for_param = list(work)

    return setup_paramfile(
        out,
        staged_ref,
        sci_for_param,
        copy_files=False,
        global_params=hst_base_params,
        phot_out=f'{out.name}.phot',
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
        Science image paths (basenames written as ``img1_file``, …).
    global_params : dict or None, optional
        Global DOLPHOT keywords merged into the parameter file. Defaults to
        :data:`st123.utils.settings.base_params`, with MIRI keys added when
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
    gparams = dict(global_params) if global_params is not None else dict(base_params)
    if any(k == 'miri' for k in kinds):
        merged = dict(miri_base_params)
        merged.update(gparams)
        # Keep caller overrides, but ensure MIRI-required keys exist.
        if 'MIRIvega' not in merged:
            merged['MIRIvega'] = miri_base_params['MIRIvega']
        if 'UseWCS' not in merged:
            merged['UseWCS'] = '2'
        gparams = merged
    if any(k in _HST_IMAGE_KINDS for k in kinds):
        merged = dict(hst_base_params)
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
    (:data:`st123.photometry.dolphot_split.DOLPHOT_MAX_NIMG`, default 400),
    the run is split into roughly equal parts that share the same reference
    (see :func:`st123.photometry.dolphot_split.write_split_paramfiles`).

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
        If True, return a :class:`~st123.photometry.dolphot_split.DolphotRunPlan`
        instead of the primary parameter-file path.

    Returns
    -------
    pathlib.Path or DolphotRunPlan
        Primary ``dolphot.param`` path, or the full run plan when
        ``return_plan`` is True.
    """
    from st123.photometry.dolphot_split import (
        DOLPHOT_MAX_NIMG,
        write_split_paramfiles,
    )

    outdir = Path(phot_outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if copy_files:
        shutil.copy(refimage, outdir)
        for src in files:
            shutil.copy(src, outdir)

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
# ~0.30" on a 0.031"/pix NIRCam reference → ~10 pix.
MIRI_WARMSTART_MIN_SEP_ARCSEC = 0.30


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
        ``img1_file``, ``img2_file``, … basenames in parameter-file order.

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
    path_prefix = f'PATH={bin_dir}:$PATH ' if bin_dir is not None else ''
    run = f'{path_prefix}dolphot {phot_out} -p{param_file} MaxThreads={threads}'
    if nohup:
        return (
            f'cd {out} && '
            f'nohup {run} > dolphot.out 2> dolphot.err < /dev/null &'
        )
    return f'cd {out} && {run}'
