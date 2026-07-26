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
    base_params,
    long_params,
    miri_base_params,
    miri_calcsky_params,
    miri_params,
    nircam_calcsky_params,
    short_params,
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


def discover_mosaic_phot_jobs(reduction_dir: PathLike) -> list[MosaicPhotJob]:
    """
    Find mosaic boxes under ``<reduction>/reference/`` for DOLPHOT prep.

    Prefers ``dolphot_frames.txt`` manifests. If a coadd exists without a
    manifest, pairs it with all ``jhat/*jhat.fits`` under *reduction_dir*.

    Parameters
    ----------
    reduction_dir : str or os.PathLike
        Mosaic reduction root (contains ``reference/`` and optionally ``jhat/``).

    Returns
    -------
    list of MosaicPhotJob
        One job per mosaic box, sorted by manifest path. Empty when
        ``reference/`` is missing or no boxes are found.
    """
    root = Path(reduction_dir)
    jobs: list[MosaicPhotJob] = []
    ref_root = root / 'reference'
    if not ref_root.is_dir():
        return jobs

    manifests = sorted(ref_root.glob('group_*/ref_*/' + _FRAME_LIST_NAME))
    if manifests:
        for manifest in manifests:
            refimage, frames, group, box = parse_dolphot_frame_list(manifest)
            jobs.append(
                MosaicPhotJob(
                    group=group,
                    box=box,
                    refimage=refimage,
                    frames=tuple(frames),
                    phot_outdir=root / f'phot_{group}_{box}',
                    frame_list=manifest,
                )
            )
        return jobs

    # Fallback: coadd present, no manifest (legacy mosaic run).
    jhat = sorted((root / 'jhat').glob('*jhat.fits'))
    for coadd in sorted(ref_root.glob('group_*/ref_*/coadd_*_i2d.fits')):
        match = _GROUP_BOX_RE.search(coadd.as_posix())
        group = int(match.group(1)) if match else 0
        box = int(match.group(2)) if match else 0
        if not jhat:
            continue
        jobs.append(
            MosaicPhotJob(
                group=group,
                box=box,
                refimage=coadd,
                frames=tuple(jhat),
                phot_outdir=root / f'phot_{group}_{box}',
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
    Stage a mosaic box into ``phot_*``, write ``dolphot.param``, mask + sky.

    Parameters
    ----------
    job : MosaicPhotJob
        Mosaic box descriptor from :func:`discover_mosaic_phot_jobs`.
    instrument : str, optional
        ``'nircam'`` or ``'miri'``; selects mask and calcsky defaults.
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
    param = setup_paramfile(
        job.phot_outdir,
        job.refimage,
        list(job.frames),
        copy_files=copy_files,
    )
    work = [Path(p) for p in science_fits_paths(job.phot_outdir)]
    prepare_frames(
        work,
        instrument=instrument,
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
        One of ``'short'`` (NIRCam short-wavelength), ``'long'`` (NIRCam
        long-wavelength), or ``'miri'`` (MIRI).

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
    except OSError:
        pass

    raise ValueError(f'Cannot classify DOLPHOT image kind for {path}')


def per_image_params(kind: str) -> Mapping[str, str]:
    """
    Return the per-image DOLPHOT parameter dict for an image kind.

    Parameters
    ----------
    kind : str
        One of ``'short'``, ``'long'``, or ``'miri'``.

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

    Defaults follow the instrument manuals:

    - NIRCam: rin=15, rout=25, step=-64, σ=2.25/2.00
    - MIRI: rin=10, rout=25, step=-64, σ=2.25/2.00 (``dolphotMIRI.pdf`` §3.4)

    When all frames share a directory, ``calcsky`` is invoked with basenames
    and ``cwd`` set to that directory to avoid C path-buffer overflows.

    Parameters
    ----------
    files : sequence of str or os.PathLike
        FITS paths for which to compute sky maps.
    instrument : str, optional
        ``'nircam'`` or ``'miri'``; selects default calcsky radii and sigmas.
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
    defaults = miri_calcsky_params if inst == 'miri' else nircam_calcsky_params
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
        ``'nircam'`` or ``'miri'``; selects the mask utility and calcsky defaults.
    dolphot_bin : str or os.PathLike or None, optional
        Override path to the DOLPHOT ``bin`` directory.
    skip_mask : bool, optional
        If True, skip ``nircammask`` / ``mirimask``.
    skip_sky : bool, optional
        If True, skip ``calcsky``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If *instrument* is not ``'nircam'`` or ``'miri'``.
    """
    inst = instrument.lower()
    if not skip_mask:
        if inst == 'miri':
            apply_mirimask(files, dolphot_bin=dolphot_bin)
        elif inst == 'nircam':
            apply_nircammask(files, dolphot_bin=dolphot_bin)
        else:
            raise ValueError(f'Unsupported instrument for prepare_frames: {instrument}')
    if not skip_sky:
        calc_sky(files, instrument=inst, dolphot_bin=dolphot_bin)


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

    # Include MIRI global knobs whenever any MIRI frame is present.
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
) -> Path:
    """
    Stage images into a photometry directory and write ``dolphot.param``.

    Supports NIRCam and MIRI frames and an optional warm-start ``xytfile``.

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

    Returns
    -------
    pathlib.Path
        Path to the written ``dolphot.param`` file.
    """
    outdir = Path(phot_outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if copy_files:
        shutil.copy(refimage, outdir)
        for src in files:
            shutil.copy(src, outdir)

    staged_ref = outdir / Path(refimage).name
    staged = [outdir / Path(p).name for p in files]
    xyt_name = None
    if xytfile is not None:
        xyt_src = Path(xytfile)
        xyt_dst = outdir / xyt_src.name
        if xyt_src.resolve() != xyt_dst.resolve():
            shutil.copy(xyt_src, xyt_dst)
        xyt_name = xyt_dst

    return write_paramfile(
        outdir / 'dolphot.param',
        refimage=staged_ref if copy_files else refimage,
        images=staged if copy_files else files,
        global_params=global_params,
        xytfile=xyt_name,
    )


def phot_to_xyt(
    photfile: PathLike,
    xyt_file: PathLike,
    *,
    types: Optional[Iterable[int]] = None,
) -> Path:
    """
    Build a warm-start star list from a DOLPHOT ``.phot`` catalog.

    Columns (1-based from the DOLPHOT manual): extension, Z, X, Y, type (col
    11), and SNR (col 6). Extension, Z, and type are written as integers.

    Parameters
    ----------
    photfile : str or os.PathLike
        Input DOLPHOT ``.phot`` catalog path.
    xyt_file : str or os.PathLike
        Output warm-start list path (typically ``warmstart.xyt``).
    types : iterable of int or None, optional
        If provided, keep only objects whose DOLPHOT type appears in this set.

    Returns
    -------
    pathlib.Path
        Path to the written ``xyt`` file.

    Raises
    ------
    ValueError
        If no stars pass the type filter and are written to the output.
    """
    phot_path = Path(photfile)
    out = Path(xyt_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    type_set = set(int(t) for t in types) if types is not None else None

    n_written = 0
    with phot_path.open() as fin, out.open('w') as fout:
        for line in fin:
            parts = line.split()
            if len(parts) < 11:
                continue
            ext = int(float(parts[0]))
            z = int(float(parts[1]))
            x = parts[2]
            y = parts[3]
            snr = parts[5]
            obj_type = int(float(parts[10]))
            if type_set is not None and obj_type not in type_set:
                continue
            fout.write(f'{ext} {z} {x} {y} {obj_type} {snr}\n')
            n_written += 1

    if n_written == 0:
        raise ValueError(f'No stars written to warm-start list from {phot_path}')
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

    Returns
    -------
    str
        Shell command string ready to copy and run.
    """
    out = Path(outdir).resolve()
    threads = max(1, int(ncores))
    bin_dir = resolve_dolphot_bin(dolphot_bin, required=False)
    if bin_dir is not None:
        return (
            f'cd {out} && '
            f'PATH={bin_dir}:$PATH '
            f'dolphot {phot_out} -p{param_file} MaxThreads={threads}'
        )
    return (
        f'cd {out} && '
        f'dolphot {phot_out} -p{param_file} MaxThreads={threads}'
    )
