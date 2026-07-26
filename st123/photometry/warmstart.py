"""
Warm-start DOLPHOT runs that append MIRI photometry to an existing NIRCam run.

Uses ``xytfile`` from a prior ``.phot`` catalog (DOLPHOT manual warm-start
section) and MIRI prep steps from ``dolphotMIRI.pdf``.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

from st123.photometry.dolphot_prep import (
    classify_image_kind,
    dolphot_command,
    parse_param_image_list,
    phot_to_xyt,
    prepare_frames,
    write_paramfile,
)
from st123.utils.settings import miri_base_params

PathLike = Union[str, os.PathLike]


@dataclass
class WarmStartResult:
    """Paths and launch command for a prepared warm-start DOLPHOT run.

    Attributes
    ----------
    outdir : pathlib.Path
        Staging directory containing staged frames, sky maps, and parameter file.
    param_file : pathlib.Path
        Combined NIRCam+MIRI ``dolphot.param`` path.
    xyt_file : pathlib.Path
        Warm-start star list (``warmstart.xyt``).
    phot_out : str
        DOLPHOT output catalog basename for the combined run.
    nircam_images : list of str
        Basenames of staged NIRCam science frames.
    miri_images : list of str
        Basenames of staged MIRI ``*_jhat.fits`` frames.
    command : str
        Shell command to launch DOLPHOT (not executed by setup).
    """

    outdir: Path
    param_file: Path
    xyt_file: Path
    phot_out: str
    nircam_images: list[str] = field(default_factory=list)
    miri_images: list[str] = field(default_factory=list)
    command: str = ''


def discover_miri_jhat(
    data_root: PathLike,
    *,
    alignment_summary: Optional[PathLike] = None,
    min_overlap: float = 0.0,
    require_success: bool = True,
) -> list[Path]:
    """
    Find overlapping MIRI ``*_jhat.fits`` frames under a data root.

    Prefer ``*_alignment_summary.txt`` SUCCESS rows with
    ``ref_overlap_frac >= min_overlap``. Fall back to a recursive glob under
    ``JWST/MIRI`` when no summary is available.

    Parameters
    ----------
    data_root : str or os.PathLike
        JWST dataset root to search (contains ``JWST/MIRI`` or alignment outputs).
    alignment_summary : str or os.PathLike or None, optional
        Explicit alignment summary table path. When ``None``, the first
        ``*_alignment_summary.txt`` under *data_root* is used if present.
    min_overlap : float, optional
        Minimum ``ref_overlap_frac`` for SUCCESS rows in the summary table.
    require_success : bool, optional
        If True (default), ignore summary rows whose status is not ``SUCCESS``.

    Returns
    -------
    list of pathlib.Path
        Sorted MIRI ``*_jhat.fits`` paths (stable order by full path then basename).
    """
    root = Path(data_root)
    summary = Path(alignment_summary) if alignment_summary else None
    if summary is None:
        candidates = sorted(root.glob('*_alignment_summary.txt'))
        summary = candidates[0] if candidates else None

    found: list[Path] = []
    if summary is not None and summary.is_file():
        for line in summary.read_text().splitlines()[1:]:
            if not line.strip() or line.startswith('-'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            # Columns: miri_path filter status ref_overlap_frac ... aligned_path
            status = parts[2]
            try:
                overlap = float(parts[3])
            except ValueError:
                continue
            aligned = parts[7]
            if require_success and status != 'SUCCESS':
                continue
            if overlap < min_overlap:
                continue
            if not aligned.endswith('_jhat.fits') or aligned == 'NA':
                continue
            path = Path(aligned)
            if path.is_file():
                found.append(path)

    if found:
        # Stable order: filter path then basename.
        return sorted(found, key=lambda p: (str(p).lower(), p.name))

    # Fallback: all MIRI JHAT products under the canonical tree.
    patterns = [
        'JWST/MIRI/**/alignment_output/*_jhat.fits',
        '**/MIRI/**/alignment_output/*_jhat.fits',
        '**/mirimage/*_jhat.fits',
    ]
    seen: set[Path] = set()
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file():
                seen.add(p.resolve())
    return sorted(seen, key=lambda p: (str(p).lower(), p.name))


def _link_or_copy(src: Path, dst: Path, *, use_hardlink: bool = True) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _stage_nircam_products(
    nircam_rundir: Path,
    outdir: Path,
    ref_base: str,
    image_bases: Sequence[str],
    *,
    use_hardlink: bool = True,
) -> tuple[Path, list[Path]]:
    """Hardlink/copy reference + science frames and existing ``.sky.fits``."""
    ref_src = nircam_rundir / f'{ref_base}.fits'
    if not ref_src.is_file():
        raise FileNotFoundError(f'Reference image missing: {ref_src}')
    ref_dst = outdir / ref_src.name
    _link_or_copy(ref_src, ref_dst, use_hardlink=use_hardlink)
    sky_src = nircam_rundir / f'{ref_base}.sky.fits'
    if sky_src.is_file():
        _link_or_copy(sky_src, outdir / sky_src.name, use_hardlink=use_hardlink)

    staged: list[Path] = []
    for base in image_bases:
        src = nircam_rundir / f'{base}.fits'
        if not src.is_file():
            raise FileNotFoundError(f'NIRCam frame missing: {src}')
        dst = outdir / src.name
        _link_or_copy(src, dst, use_hardlink=use_hardlink)
        sky = nircam_rundir / f'{base}.sky.fits'
        if sky.is_file():
            _link_or_copy(sky, outdir / sky.name, use_hardlink=use_hardlink)
        staged.append(dst)
    return ref_dst, staged


def _stage_miri_jhat(
    miri_sources: Sequence[Path],
    outdir: Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Copy MIRI JHAT frames into *outdir* (writable copies for mirimask)."""
    staged: list[Path] = []
    for src in miri_sources:
        dst = outdir / src.name
        if overwrite or not dst.exists():
            shutil.copy2(src, dst)
            # Stale sky maps from a prior partial prep must not block re-prep.
            if overwrite:
                sky = outdir / f"{dst.name.replace('.fits', '')}.sky.fits"
                if sky.exists():
                    sky.unlink()
        staged.append(dst)
    return staged


def find_photfile(nircam_rundir: PathLike) -> Path:
    """
    Locate the primary ``.phot`` catalog in a finished NIRCam DOLPHOT run.

    Parameters
    ----------
    nircam_rundir : str or os.PathLike
        Directory containing a completed NIRCam DOLPHOT run.

    Returns
    -------
    pathlib.Path
        Path to the main ``.phot`` catalog (largest candidate when multiple exist).

    Raises
    ------
    FileNotFoundError
        If no suitable ``.phot`` catalog is found under *nircam_rundir*.
    """
    rundir = Path(nircam_rundir)
    # Prefer a top-level *.phot that is not a sidecar (no extra dots before .phot)
    candidates = sorted(
        p
        for p in rundir.glob('*.phot')
        if p.is_file() and p.name.count('.') == 1
    )
    if not candidates:
        # Fallback: any *.phot without .psf/.res in the name
        candidates = sorted(
            p
            for p in rundir.glob('*.phot')
            if p.is_file() and '.psf.' not in p.name and '.res.' not in p.name
        )
    if not candidates:
        raise FileNotFoundError(f'No .phot catalog found under {rundir}')
    # Prefer the largest catalog (main photometry product).
    return max(candidates, key=lambda p: p.stat().st_size)


def setup_miri_warmstart(
    nircam_rundir: PathLike,
    outdir: PathLike,
    *,
    miri_jhat: Optional[Sequence[PathLike]] = None,
    data_root: Optional[PathLike] = None,
    alignment_summary: Optional[PathLike] = None,
    photfile: Optional[PathLike] = None,
    min_overlap: float = 0.0,
    dolphot_bin: Optional[PathLike] = None,
    phot_out: Optional[str] = None,
    prepare_miri: bool = True,
    use_hardlink: bool = True,
    xyt_types: Optional[Sequence[int]] = None,
    ncores: int = 1,
) -> WarmStartResult:
    """
    Create a warm-start DOLPHOT directory with NIRCam + overlapping MIRI frames.

    Steps
    -----
    1. Read ``dolphot.param`` / ``.phot`` from *nircam_rundir*.
    2. Stage NIRCam reference, frames, and sky maps into *outdir*.
    3. Copy overlapping MIRI ``*_jhat.fits``, run ``mirimask -estnoise`` and
       MIRI ``calcsky``, then stage their sky maps.
    4. Build ``warmstart.xyt`` from the NIRCam photometry.
    5. Write a combined ``dolphot.param`` with MIRI FitSky=2 recommendations
       and ``xytfile = warmstart.xyt``.

    Does **not** run ``dolphot``; use :attr:`WarmStartResult.command`.
    ``ncores`` sets ``MaxThreads`` on that launch command (same as ``--ncores``).

    Parameters
    ----------
    nircam_rundir : str or os.PathLike
        Completed NIRCam DOLPHOT run directory (contains ``dolphot.param``).
    outdir : str or os.PathLike
        Output directory for the combined warm-start staging tree.
    miri_jhat : sequence of str or os.PathLike or None, optional
        Explicit MIRI ``*_jhat.fits`` paths. When ``None``, frames are discovered
        via :func:`discover_miri_jhat`.
    data_root : str or os.PathLike or None, optional
        JWST data root for MIRI discovery when *miri_jhat* is ``None``. Defaults
        to a heuristic based on *nircam_rundir* layout.
    alignment_summary : str or os.PathLike or None, optional
        Alignment summary table forwarded to :func:`discover_miri_jhat`.
    photfile : str or os.PathLike or None, optional
        NIRCam ``.phot`` catalog for warm-start seeding. Defaults to
        :func:`find_photfile`.
    min_overlap : float, optional
        Minimum reference overlap fraction for MIRI frame discovery.
    dolphot_bin : str or os.PathLike or None, optional
        Override path to the DOLPHOT ``bin`` directory for MIRI prep.
    phot_out : str or None, optional
        Output catalog basename for the combined run. Defaults to
        ``<nircam_stem>_nircam_miri.phot``.
    prepare_miri : bool, optional
        If True (default), run ``mirimask`` and MIRI ``calcsky`` on frames
        missing ``.sky.fits`` sidecars.
    use_hardlink : bool, optional
        If True (default), hardlink NIRCam products into *outdir* when possible.
    xyt_types : sequence of int or None, optional
        DOLPHOT object types to retain in ``warmstart.xyt`` (forwarded to
        :func:`phot_to_xyt`).
    ncores : int, optional
        ``MaxThreads`` for the generated DOLPHOT launch command.

    Returns
    -------
    WarmStartResult
        Staged paths, frame lists, and shell command for the combined run.

    Raises
    ------
    FileNotFoundError
        If ``dolphot.param``, photometry seed, or MIRI frames are missing.
    """
    nircam_dir = Path(nircam_rundir).resolve()
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    param_src = nircam_dir / 'dolphot.param'
    if not param_src.is_file():
        raise FileNotFoundError(f'Missing {param_src}')

    ref_base, nircam_bases = parse_param_image_list(param_src)
    phot_src = Path(photfile) if photfile else find_photfile(nircam_dir)

    if miri_jhat is not None:
        miri_sources = [Path(p) for p in miri_jhat]
    else:
        root = Path(data_root) if data_root else nircam_dir.parent.parent
        # nircam_0_0 -> dolphot -> NGC3310
        if data_root is None:
            # Prefer the JWST data root two levels up from dolphot/<run>
            # e.g. .../NGC3310/dolphot/nircam_0_0 -> .../NGC3310
            guess = nircam_dir.parent.parent
            root = guess if (guess / 'JWST').is_dir() else nircam_dir.parent
        miri_sources = discover_miri_jhat(
            root,
            alignment_summary=alignment_summary,
            min_overlap=min_overlap,
        )
    if not miri_sources:
        raise FileNotFoundError(
            'No overlapping MIRI *_jhat.fits frames found; pass miri_jhat=...'
        )

    ref_dst, nircam_staged = _stage_nircam_products(
        nircam_dir,
        out,
        ref_base,
        nircam_bases,
        use_hardlink=use_hardlink,
    )

    # Re-copy MIRI JHAT before prep: mirimask is not idempotent (strips DQ).
    need_prep_names = {
        src.name
        for src in miri_sources
        if not (out / f"{src.name.replace('.fits', '')}.sky.fits").is_file()
    }
    miri_staged = _stage_miri_jhat(
        miri_sources,
        out,
        overwrite=bool(prepare_miri and need_prep_names),
    )

    if prepare_miri:
        need_prep = [p for p in miri_staged if p.name in need_prep_names]
        if need_prep:
            prepare_frames(need_prep, instrument='miri', dolphot_bin=dolphot_bin)

    xyt_path = out / 'warmstart.xyt'
    phot_to_xyt(phot_src, xyt_path, types=xyt_types)

    all_images = list(nircam_staged) + list(miri_staged)
    kinds = [classify_image_kind(p) for p in all_images]
    param_path = write_paramfile(
        out / 'dolphot.param',
        refimage=ref_dst,
        images=all_images,
        global_params=miri_base_params,
        xytfile=xyt_path,
        image_kinds=kinds,
    )

    if phot_out is None:
        stem = phot_src.name.replace('.phot', '')
        phot_out = f'{stem}_nircam_miri.phot'

    cmd = dolphot_command(
        out,
        phot_out=phot_out,
        param_file=param_path.name,
        dolphot_bin=dolphot_bin,
        ncores=ncores,
    )

    # Record a small README with the launch command.
    readme = out / 'WARMSTART_README.txt'
    readme.write_text(
        'DOLPHOT NIRCam+MIRI warm-start run\n'
        f'NIRCam source: {nircam_dir}\n'
        f'Photometry seed: {phot_src}\n'
        f'xytfile: {xyt_path.name} ({sum(1 for _ in xyt_path.open())} stars)\n'
        f'NIRCam frames: {len(nircam_staged)}\n'
        f'MIRI frames: {len(miri_staged)}\n'
        f'Parameter file: {param_path.name}\n'
        '\n'
        'Launch (not run by setup):\n'
        f'  {cmd}\n'
    )

    return WarmStartResult(
        outdir=out,
        param_file=param_path,
        xyt_file=xyt_path,
        phot_out=phot_out,
        nircam_images=[p.name for p in nircam_staged],
        miri_images=[p.name for p in miri_staged],
        command=cmd,
    )
