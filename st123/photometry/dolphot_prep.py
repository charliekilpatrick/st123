"""
Prepare JWST frames for DOLPHOT (mask + sky + parameter file).

Mirrors the NIRCam workflow in :mod:`st123.mosaic.mosaic` and adds MIRI
support following ``dolphotMIRI.pdf`` (mirimask, calcsky, FitSky=2 params).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence, Union

from astropy.io import fits

from st123.utils.helpers import get_detector_chip
from st123.utils.settings import (
    DEFAULT_DOLPHOT_BIN,
    base_params,
    long_params,
    miri_base_params,
    miri_calcsky_params,
    miri_params,
    nircam_calcsky_params,
    short_params,
)

PathLike = Union[str, os.PathLike]


def dolphot_bin_dir(dolphot_bin: Optional[PathLike] = None) -> Path:
    """Return the DOLPHOT ``bin`` directory (default: local 3.1 install)."""
    return Path(dolphot_bin or DEFAULT_DOLPHOT_BIN)


def _prepend_bin_env(env: Optional[MutableMapping[str, str]], bin_dir: Path) -> dict:
    out = dict(env or os.environ)
    bin_s = str(bin_dir)
    path = out.get('PATH', '')
    if bin_s not in path.split(os.pathsep):
        out['PATH'] = bin_s + (os.pathsep + path if path else '')
    return out


def classify_image_kind(path: PathLike) -> str:
    """
    Return ``'short'``, ``'long'``, or ``'miri'`` for per-image DOLPHOT params.

    Classification uses the detector token in the filename (``nrc*`` /
    ``mirimage``) and, if needed, the ``INSTRUME`` FITS keyword.
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
    """Per-image parameter dict for ``short`` / ``long`` / ``miri``."""
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
    etctime: bool = True,
    check: bool = True,
) -> None:
    """
    Run ``nircammask`` on *files* (in-place), matching the mosaic NIRCam prep.

    Parameters
    ----------
    files
        FITS paths to mask.
    dolphot_bin
        Directory containing ``nircammask``.
    etctime
        If True (default), pass ``-etctime`` as in the existing mosaic pipeline.
    check
        Raise if the subprocess exits non-zero.
    """
    bin_dir = dolphot_bin_dir(dolphot_bin)
    cwd, names = _run_cwd_for_files(files)
    cmd = [str(bin_dir / 'nircammask')]
    if etctime:
        cmd.append('-etctime')
    cmd.extend(names)
    subprocess.run(
        cmd, check=check, env=_prepend_bin_env(None, bin_dir), cwd=cwd
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
    Run ``mirimask`` on *files* (in-place) per ``dolphotMIRI.pdf`` §3.3.

    Default flags follow the MIRI manual recommendations:
    ``-estnoise`` on, ETC exposure time on (do **not** pass ``-noetctime``).
    Back up originals before calling; ``mirimask`` rewrites the FITS files.
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
    subprocess.run(
        cmd, check=check, env=_prepend_bin_env(None, bin_dir), cwd=cwd
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
    Run DOLPHOT ``calcsky`` on each frame.

    Defaults follow the instrument manuals:
    - NIRCam: rin=15, rout=25, step=-64, σ=2.25/2.00
    - MIRI: rin=10, rout=25, step=-64, σ=2.25/2.00 (``dolphotMIRI.pdf`` §3.4)

    When all frames share a directory, ``calcsky`` is invoked with basenames
    and ``cwd`` set to that directory to avoid C path-buffer overflows.
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
        subprocess.run(cmd, check=check, env=env, cwd=cwd)


def prepare_frames(
    files: Sequence[PathLike],
    *,
    instrument: str,
    dolphot_bin: Optional[PathLike] = None,
    skip_mask: bool = False,
    skip_sky: bool = False,
) -> None:
    """Mask then compute sky for *files* for ``nircam`` or ``miri``."""
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
    Stage images into *phot_outdir* and write ``dolphot.param``.

    Drop-in generalization of :func:`st123.mosaic.mosaic.setup_paramfile` that
    also supports MIRI frames and an optional warm-start ``xytfile``.
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
    phot_file: PathLike,
    xyt_file: PathLike,
    *,
    types: Optional[Iterable[int]] = None,
) -> Path:
    """
    Build a warm-start star list from a DOLPHOT ``.phot`` catalog.

    Columns (1-based from the DOLPHOT manual): extension, Z, X, Y, type (col
    11), and SNR (col 6). Extension, Z, and type are written as integers.
    """
    phot_path = Path(phot_file)
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
    """Return ``(img0_file, [img1_file, ...])`` bases from a DOLPHOT param file."""
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
) -> str:
    """
    Shell command to run DOLPHOT (does not execute it).

    Usage matches the binary: ``dolphot <output> -p<paramfile>``.
    """
    bin_dir = dolphot_bin_dir(dolphot_bin)
    out = Path(outdir).resolve()
    return (
        f'cd {out} && '
        f'PATH={bin_dir}:$PATH '
        f'dolphot {phot_out} -p{param_file}'
    )
