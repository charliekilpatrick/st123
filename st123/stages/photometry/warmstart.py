"""
Warm-start DOLPHOT runs seeded from an existing NIRCam catalog.

Supports:

* NIRCam -> MIRI (``setup_miri_warmstart``; ``dolphotMIRI.pdf``)
* NIRCam reference/catalog -> HST science (``setup_hst_warmstart``;
  ``HSTDataModel.DOLPHOT_BASE_PARAMS`` / JHAT-aligned ACS/WFC3/WFPC2)

Uses ``xytfile`` from a prior ``.phot`` catalog (DOLPHOT manual warm-start).
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

from st123.datamodels import as_datamodel

from st123.utils.helpers import is_full_frame_miri

# Backward-compatible alias for callers that import from warmstart.
is_mirimask_compatible = is_full_frame_miri

logger = logging.getLogger(__name__)

from st123.stages.photometry.dolphot import (
    HST_WARMSTART_CROWD_MAX,
    HST_WARMSTART_MIN_SEP_ARCSEC,
    HST_WARMSTART_SHARP2_MAX,
    HST_WARMSTART_SNR_MIN,
    HST_WARMSTART_XYT_TYPES,
    MIRI_WARMSTART_CROWD_MAX,
    MIRI_WARMSTART_MIN_SEP_ARCSEC,
    MIRI_WARMSTART_SHARP2_MAX,
    MIRI_WARMSTART_SNR_MIN,
    MIRI_WARMSTART_XYT_TYPES,
    classify_image_kind,
    dolphot_command,
    parse_param_image_list,
    phot_to_xyt,
    prepare_frames,
    prepare_hst_frames,
)
from st123.stages.photometry.dolphot_split import (
    DOLPHOT_MAX_NIMG,
    DolphotRunPlan,
    write_split_paramfiles,
)
from st123.datamodels import HSTDataModel, MIRIDataModel

PathLike = Union[str, os.PathLike]


@dataclass
class WarmStartResult:
    """Paths and launch command for a prepared warm-start DOLPHOT run.

    Attributes
    ----------
    outdir : pathlib.Path
        Staging directory containing staged frames, sky maps, and parameter file.
    param_file : pathlib.Path
        Primary ``dolphot.param`` (or part-0 param when split).
    xyt_file : pathlib.Path
        Warm-start star list (``warmstart.xyt``).
    phot_out : str
        Final DOLPHOT catalog basename (merged name when the run is split).
    nircam_images : list of str
        Basenames of staged NIRCam science frames.
    miri_images : list of str
        Basenames of staged MIRI ``*_jhat.fits`` frames.
    hst_images : list of str
        Basenames of staged HST science frames (NIRCam->HST warmstart).
    command : str
        Shell command for a single-part run (empty when split; see *commands*).
    plan : DolphotRunPlan or None
        Full run plan (includes split parts when ``Nimg`` exceeds the soft cap).
    commands : list of str
        One launch command per part (length 1 for unsplit runs).
    """

    outdir: Path
    param_file: Path
    xyt_file: Path
    phot_out: str
    nircam_images: list[str] = field(default_factory=list)
    miri_images: list[str] = field(default_factory=list)
    hst_images: list[str] = field(default_factory=list)
    command: str = ''
    plan: Optional[DolphotRunPlan] = None
    commands: list[str] = field(default_factory=list)


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
    unusable = outdir / 'unusable'
    for src in miri_sources:
        if not is_full_frame_miri(src):
            unusable.mkdir(parents=True, exist_ok=True)
            dst_bad = unusable / src.name
            if not dst_bad.exists():
                try:
                    shutil.copy2(src, dst_bad)
                except OSError:
                    pass
            logger.warning(
                'Skipping non-full-frame MIRI JHAT (unsupported subarray/cutout): %s',
                src,
            )
            continue
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

    def _is_usable_phot(path: Path) -> bool:
        # Skip empty stubs left by interrupted DOLPHOT runs.
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    # Prefer a top-level *.phot that is not a sidecar (no extra dots before .phot)
    candidates = sorted(
        p
        for p in rundir.glob('*.phot')
        if _is_usable_phot(p) and p.name.count('.') == 1
    )
    if not candidates:
        # Fallback: any *.phot without .psf/.res in the name
        candidates = sorted(
            p
            for p in rundir.glob('*.phot')
            if _is_usable_phot(p) and '.psf.' not in p.name and '.res.' not in p.name
        )
    if not candidates:
        raise FileNotFoundError(
            f'No non-empty .phot catalog found under {rundir}'
        )
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
    xyt_snr_min: Optional[float] = None,
    xyt_crowd_max: Optional[float] = None,
    xyt_sharp2_max: Optional[float] = None,
    xyt_min_sep_arcsec: Optional[float] = None,
    xyt_force_xy: Optional[Sequence[tuple[float, float]]] = None,
    xyt_max_radius_arcsec: Optional[float] = None,
    xyt_center_xy: Optional[tuple[float, float]] = None,
    prune_xyt_for_miri: bool = False,
    ncores: int = 1,
    max_nimg: int = DOLPHOT_MAX_NIMG,
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
    xyt_snr_min, xyt_crowd_max, xyt_sharp2_max : float or None, optional
        Quality cuts for ``warmstart.xyt`` (see :func:`phot_to_xyt`).
    xyt_min_sep_arcsec : float or None, optional
        Minimum seed separation on the NIRCam reference (arcsec).
    xyt_force_xy : sequence of (x, y) or None, optional
        Reference-pixel coordinates that must be retained (e.g. the SN).
    xyt_max_radius_arcsec : float or None, optional
        Keep only seeds within this radius of *xyt_center_xy* (or the first
        ``xyt_force_xy`` point). Use ~5" for single-target NGC3310 runs.
    xyt_center_xy : (x, y) or None, optional
        Center for the radius cut when not using ``xyt_force_xy``.
    prune_xyt_for_miri : bool, optional
        If True, apply the recommended MIRI seed cuts (type=1, SNR>=10,
        crowd<=0.5, sharp^2<=0.01, minsep=0.30") unless overridden above.
    ncores : int, optional
        ``MaxThreads`` for the generated DOLPHOT launch command.
    max_nimg : int, optional
        Soft science-image cap per DOLPHOT invocation (default 400). Larger
        lists are split into roughly equal parts that share the same reference
        and ``warmstart.xyt``; catalogs are merged after all parts finish.

    Returns
    -------
    WarmStartResult
        Staged paths, frame lists, and shell command(s) for the combined run.

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
    if not miri_staged:
        raise FileNotFoundError(
            f'No mirimask-compatible full-frame MIRI JHAT frames for {out}'
        )

    if prepare_miri:
        need_prep = [p for p in miri_staged if p.name in need_prep_names]
        if need_prep:
            prepare_frames(
                need_prep,
                instrument='miri',
                dolphot_bin=dolphot_bin,
                ncores=ncores,
            )

    if prune_xyt_for_miri:
        if xyt_types is None:
            xyt_types = MIRI_WARMSTART_XYT_TYPES
        if xyt_snr_min is None:
            xyt_snr_min = MIRI_WARMSTART_SNR_MIN
        if xyt_crowd_max is None:
            xyt_crowd_max = MIRI_WARMSTART_CROWD_MAX
        if xyt_sharp2_max is None:
            xyt_sharp2_max = MIRI_WARMSTART_SHARP2_MAX
        if xyt_min_sep_arcsec is None:
            xyt_min_sep_arcsec = MIRI_WARMSTART_MIN_SEP_ARCSEC

    min_sep_pix = None
    max_radius_pix = None
    pixscale = None
    need_pixscale = (
        (xyt_min_sep_arcsec is not None and xyt_min_sep_arcsec > 0)
        or (xyt_max_radius_arcsec is not None and xyt_max_radius_arcsec > 0)
    )
    if need_pixscale:
        from astropy.wcs import WCS
        import astropy.units as u

        with as_datamodel(ref_dst).open() as hdul:
            sci = hdul['SCI'] if 'SCI' in hdul else hdul[0]
            pixscale = float(
                abs(WCS(sci.header).proj_plane_pixel_scales()[0].to(u.arcsec).value)
            )
        if xyt_min_sep_arcsec is not None and xyt_min_sep_arcsec > 0:
            min_sep_pix = float(xyt_min_sep_arcsec) / pixscale
        if xyt_max_radius_arcsec is not None and xyt_max_radius_arcsec > 0:
            max_radius_pix = float(xyt_max_radius_arcsec) / pixscale

    xyt_path = out / 'warmstart.xyt'
    phot_to_xyt(
        phot_src,
        xyt_path,
        types=xyt_types,
        snr_min=xyt_snr_min,
        crowd_max=xyt_crowd_max,
        sharp2_max=xyt_sharp2_max,
        min_sep_pix=min_sep_pix,
        force_xy=xyt_force_xy,
        center_xy=xyt_center_xy,
        max_radius_pix=max_radius_pix,
    )

    all_images = list(nircam_staged) + list(miri_staged)
    kinds = [classify_image_kind(p) for p in all_images]

    if phot_out is None:
        stem = phot_src.name.replace('.phot', '')
        phot_out = f'{stem}_nircam_miri.phot'

    plan = write_split_paramfiles(
        out,
        refimage=ref_dst,
        images=all_images,
        phot_out=phot_out,
        global_params=MIRIDataModel.DOLPHOT_BASE_PARAMS,
        xytfile=xyt_path,
        image_kinds=kinds,
        max_nimg=max_nimg,
        read_filters=True,
    )
    param_path = plan.param_file

    commands = [
        dolphot_command(
            out,
            phot_out=part.phot_out,
            param_file=part.param_file,
            dolphot_bin=dolphot_bin,
            ncores=ncores,
        )
        for part in plan.parts
    ]
    cmd = commands[0] if len(commands) == 1 else ''

    # Record a small README with the launch command(s).
    n_xyt = sum(1 for _ in xyt_path.open())
    launch_lines = '\n'.join(f'  {c}' for c in commands)
    if plan.needs_merge:
        launch_lines += (
            '\n\nAfter all parts finish, merge with:\n'
            '  python -c "from st123.stages.photometry.dolphot_split import '
            f'finalize_split_outdir; finalize_split_outdir(r\'{out}\')"\n'
        )
    readme = out / 'WARMSTART_README.txt'
    readme.write_text(
        'DOLPHOT NIRCam+MIRI warm-start run\n'
        f'NIRCam source: {nircam_dir}\n'
        f'Photometry seed: {phot_src}\n'
        f'xytfile: {xyt_path.name} ({n_xyt} stars)\n'
        f'xyt prune_for_miri: {prune_xyt_for_miri}\n'
        f'xyt types: {list(xyt_types) if xyt_types is not None else None}\n'
        f'xyt snr_min: {xyt_snr_min}\n'
        f'xyt crowd_max: {xyt_crowd_max}\n'
        f'xyt sharp2_max: {xyt_sharp2_max}\n'
        f'xyt min_sep_arcsec: {xyt_min_sep_arcsec}\n'
        f'xyt max_radius_arcsec: {xyt_max_radius_arcsec}\n'
        f'xyt center_xy: {list(xyt_center_xy) if xyt_center_xy else None}\n'
        f'xyt force_xy: {list(xyt_force_xy) if xyt_force_xy else None}\n'
        f'NIRCam frames: {len(nircam_staged)}\n'
        f'MIRI frames: {len(miri_staged)}\n'
        f'Total science frames: {len(all_images)}\n'
        f'max_nimg (soft): {max_nimg}\n'
        f'split_parts: {len(plan.parts)}\n'
        f'Parameter file: {param_path.name}\n'
        f'Final phot_out: {phot_out}\n'
        '\n'
        'Launch (not run by setup):\n'
        f'{launch_lines}\n'
    )

    return WarmStartResult(
        outdir=out,
        param_file=param_path,
        xyt_file=xyt_path,
        phot_out=phot_out,
        nircam_images=[p.name for p in nircam_staged],
        miri_images=[p.name for p in miri_staged],
        command=cmd,
        plan=plan,
        commands=commands,
    )


def discover_hst_jhat(
    data_root: PathLike,
    *,
    patterns: Optional[Sequence[str]] = None,
    instruments: Optional[Sequence[str]] = None,
) -> list[Path]:
    """
    Find HST ``*_jhat.fits`` frames under a project or reduction root.

    Parameters
    ----------
    data_root : str or os.PathLike
        Project root (contains ``reduction/jhat`` / ``jhat_hst``) or a
        directory to search.
    patterns : sequence of str or None, optional
        Glob patterns relative to *data_root*. Defaults cover
        ``reduction/jhat_hst``, legacy ``reduction/jhat``, and nested trees.
    instruments : sequence of str or None, optional
        If set, keep only frames whose ``INSTRUME`` matches (e.g. ``ACS``,
        ``WFC3``) so WFPC2 can be excluded from dual-mode warmstarts.

    Returns
    -------
    list of pathlib.Path
        Sorted unique ``*_jhat.fits`` paths.
    """
    from st123.utils.helpers import get_instrument

    root = Path(data_root)
    pats = list(patterns) if patterns is not None else [
        'reduction/jhat_hst/*_jhat.fits',
        'reduction/jhat/*_jhat.fits',
        'jhat_hst/*_jhat.fits',
        'jhat/*_jhat.fits',
        '**/reduction/jhat_hst/*_jhat.fits',
        '**/reduction/jhat/*_jhat.fits',
    ]
    allow = None
    if instruments:
        allow = {
            str(i).split('_')[0].strip().lower()
            for i in instruments
            if str(i).strip() and str(i).strip().upper() != 'HST'
        }
        if not allow:
            allow = None
    seen: set[Path] = set()

    def _accept(p: Path) -> bool:
        if not p.is_file():
            return False
        name = p.name.lower()
        if 'mirimage' in name or name.startswith('jw'):
            return False
        parent = str(p.parent).lower()
        if '/jwst/' in parent or '/miri/' in parent or '/nircam/' in parent:
            return False
        if allow is not None:
            try:
                inst = get_instrument(p).split('_')[0].lower()
            except Exception:
                return False
            if inst not in allow:
                return False
        return True

    for pat in pats:
        for p in root.glob(pat):
            if _accept(p):
                seen.add(p.resolve())
        if seen:
            break
    from st123.datamodels.hst import filter_paths_for_stage

    return filter_paths_for_stage(
        sorted(seen, key=lambda p: (str(p).lower(), p.name)),
        stage='warmstart-discover',
    )


def setup_hst_warmstart(
    nircam_rundir: PathLike,
    outdir: PathLike,
    *,
    hst_jhat: Optional[Sequence[PathLike]] = None,
    data_root: Optional[PathLike] = None,
    instruments: Optional[Sequence[str]] = None,
    photfile: Optional[PathLike] = None,
    dolphot_bin: Optional[PathLike] = None,
    phot_out: Optional[str] = None,
    prepare_hst: bool = True,
    include_nircam_science: bool = False,
    use_hardlink: bool = True,
    xyt_types: Optional[Sequence[int]] = None,
    xyt_snr_min: Optional[float] = None,
    xyt_crowd_max: Optional[float] = None,
    xyt_sharp2_max: Optional[float] = None,
    xyt_min_sep_arcsec: Optional[float] = None,
    xyt_force_xy: Optional[Sequence[tuple[float, float]]] = None,
    xyt_max_radius_arcsec: Optional[float] = None,
    xyt_center_xy: Optional[tuple[float, float]] = None,
    prune_xyt_for_hst: bool = False,
    ncores: int = 1,
) -> WarmStartResult:
    """
    Create a warm-start DOLPHOT directory with a NIRCam ref + HST science.

    Stages the NIRCam ``img0`` (and optional NIRCam science), builds
    ``warmstart.xyt`` from the NIRCam ``.phot``, then runs
    :func:`prepare_hst_frames` on HST JHAT frames with ``HSTDataModel.DOLPHOT_BASE_PARAMS``
    and ``xytfile``. Does **not** HST-mask the NIRCam reference.

    Does **not** run ``dolphot``; use :attr:`WarmStartResult.command`.

    Parameters
    ----------
    nircam_rundir : path-like
        Completed NIRCam DOLPHOT run (``dolphot.param`` + ``.phot`` + img0).
    outdir : path-like
        Staging directory (e.g. ``dolphot/nircam_hst_0_0``).
    hst_jhat : sequence of path-like or None, optional
        Explicit HST ``*_jhat.fits`` paths. When ``None``, discover under
        *data_root* via :func:`discover_hst_jhat`.
    data_root : path-like or None, optional
        Project root for HST JHAT discovery.
    instruments : sequence of str or None, optional
        Restrict discovered HST JHAT to these instruments (e.g. ``ACS``,
        ``WFC3``). Ignored when *hst_jhat* is given explicitly.
    photfile : path-like or None, optional
        NIRCam ``.phot`` seed catalog.
    dolphot_bin : path-like or None, optional
        DOLPHOT ``bin`` override.
    phot_out : str or None, optional
        Output catalog basename (default ``<stem>_nircam_hst.phot``).
    prepare_hst : bool, optional
        If True (default), run HST mask/split/calcsky via
        :func:`prepare_hst_frames`.
    include_nircam_science : bool, optional
        If True, also stage NIRCam science frames into the paramfile.
        Default False (HST-only photometry on the NIRCam reference).
    use_hardlink : bool, optional
        Hardlink NIRCam products when possible.
    xyt_* / prune_xyt_for_hst :
        Seed catalog cuts (see :func:`phot_to_xyt`). When
        *prune_xyt_for_hst* is True, apply HST defaults (type=1, SNR>=5, ...)
        unless overridden.
    ncores : int, optional
        Parallelism for HST prep and ``MaxThreads`` on the launch command.

    Returns
    -------
    WarmStartResult
    """
    nircam_dir = Path(nircam_rundir).resolve()
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    param_src = nircam_dir / 'dolphot.param'
    if not param_src.is_file():
        raise FileNotFoundError(f'Missing {param_src}')

    ref_base, nircam_bases = parse_param_image_list(param_src)
    phot_src = Path(photfile) if photfile else find_photfile(nircam_dir)

    if hst_jhat is not None:
        hst_sources = [Path(p) for p in hst_jhat]
        if instruments:
            # Still honor ACS/WFC3 filters on an explicit path list.
            from st123.utils.helpers import get_instrument

            allow = {
                str(i).split('_')[0].strip().lower()
                for i in instruments
                if str(i).strip() and str(i).strip().upper() != 'HST'
            }
            if allow:
                kept: list[Path] = []
                for path in hst_sources:
                    try:
                        inst = get_instrument(path).split('_')[0].lower()
                    except Exception:
                        continue
                    if inst in allow:
                        kept.append(path)
                hst_sources = kept
    else:
        root = Path(data_root) if data_root else nircam_dir.parent.parent
        if data_root is None:
            guess = nircam_dir.parent.parent
            if (
                (guess / 'reduction' / 'jhat_hst').is_dir()
                or (guess / 'reduction' / 'jhat').is_dir()
                or (guess / 'HST').is_dir()
            ):
                root = guess
            elif (nircam_dir.parent / 'reduction' / 'jhat_hst').is_dir() or (
                nircam_dir.parent / 'reduction' / 'jhat'
            ).is_dir():
                root = nircam_dir.parent
            else:
                root = guess
        hst_sources = discover_hst_jhat(root, instruments=instruments)
    if not hst_sources:
        inst_msg = (
            f' for instruments {list(instruments)}' if instruments else ''
        )
        raise FileNotFoundError(
            f'No HST *_jhat.fits frames found{inst_msg}; '
            'pass hst_jhat=... or align HST first'
        )

    science_bases = list(nircam_bases) if include_nircam_science else []
    ref_dst, nircam_staged = _stage_nircam_products(
        nircam_dir,
        out,
        ref_base,
        science_bases,
        use_hardlink=use_hardlink,
    )

    if prune_xyt_for_hst:
        if xyt_types is None:
            xyt_types = HST_WARMSTART_XYT_TYPES
        if xyt_snr_min is None:
            xyt_snr_min = HST_WARMSTART_SNR_MIN
        if xyt_crowd_max is None:
            xyt_crowd_max = HST_WARMSTART_CROWD_MAX
        if xyt_sharp2_max is None:
            xyt_sharp2_max = HST_WARMSTART_SHARP2_MAX
        if xyt_min_sep_arcsec is None:
            xyt_min_sep_arcsec = HST_WARMSTART_MIN_SEP_ARCSEC

    min_sep_pix = None
    max_radius_pix = None
    need_pixscale = (
        (xyt_min_sep_arcsec is not None and xyt_min_sep_arcsec > 0)
        or (xyt_max_radius_arcsec is not None and xyt_max_radius_arcsec > 0)
    )
    if need_pixscale:
        from astropy.wcs import WCS
        import astropy.units as u

        with as_datamodel(ref_dst).open() as hdul:
            sci = hdul['SCI'] if 'SCI' in hdul else hdul[0]
            pixscale = float(
                abs(WCS(sci.header).proj_plane_pixel_scales()[0].to(u.arcsec).value)
            )
        if xyt_min_sep_arcsec is not None and xyt_min_sep_arcsec > 0:
            min_sep_pix = float(xyt_min_sep_arcsec) / pixscale
        if xyt_max_radius_arcsec is not None and xyt_max_radius_arcsec > 0:
            max_radius_pix = float(xyt_max_radius_arcsec) / pixscale

    xyt_path = out / 'warmstart.xyt'
    phot_to_xyt(
        phot_src,
        xyt_path,
        types=xyt_types,
        snr_min=xyt_snr_min,
        crowd_max=xyt_crowd_max,
        sharp2_max=xyt_sharp2_max,
        min_sep_pix=min_sep_pix,
        force_xy=xyt_force_xy,
        center_xy=xyt_center_xy,
        max_radius_pix=max_radius_pix,
    )

    if phot_out is None:
        stem = phot_src.name.replace('.phot', '')
        phot_out = f'{stem}_nircam_hst.phot'

    # Stage HST JHAT into outdir (writable copies for mask/split).
    hst_staged_src: list[Path] = []
    for src in hst_sources:
        dst = out / src.name
        if not dst.exists() or dst.resolve() != src.resolve():
            shutil.copy2(src, dst)
        hst_staged_src.append(dst)
        # WFPC2 c1m beside JHAT when available (prepare_hst_frames also searches).
        c1 = src.with_name(src.name.replace('_jhat.fits', '_c1m.fits'))
        if not c1.is_file():
            alt = src.parent / (src.name.replace('_jhat.fits', '') + '_c1m.fits')
            c1 = alt if alt.is_file() else c1
        # Also try raw stem without _jhat
        if not c1.is_file():
            stem = src.name.replace('_jhat.fits', '')
            for cand in (
                src.parent / f'{stem}_c1m.fits',
                src.parent.parent / 'raw' / f'{stem}_c1m.fits',
            ):
                if cand.is_file():
                    c1 = cand
                    break
        if c1.is_file():
            c1_dst = out / c1.name
            if not c1_dst.exists():
                shutil.copy2(c1, c1_dst)

    if prepare_hst:
        param_path = prepare_hst_frames(
            hst_staged_src,
            out,
            instrument='hst',
            refimage=ref_dst,
            dolphot_bin=dolphot_bin,
            copy_files=False,
            ncores=ncores,
            xytfile=xyt_path,
        )
        # prepare_hst_frames always sets phot_out to out.name.phot; rewrite
        # when a custom name / NIRCam science inclusion is requested.
        text = param_path.read_text()
        # Ensure HST DOLPHOT globals and xytfile survived.
        if 'xytfile' not in text:
            text = text.rstrip() + f'\nxytfile = {xyt_path.name}\n'
            param_path.write_text(text)
        # Optionally append NIRCam science frames to the param image list.
        if include_nircam_science and nircam_staged:
            from st123.stages.photometry.dolphot import write_paramfile

            # Re-parse science chips produced by prepare_hst_frames.
            _, hst_bases = parse_param_image_list(param_path)
            hst_paths = [out / f'{b}.fits' for b in hst_bases]
            all_images = list(nircam_staged) + [
                p for p in hst_paths if p.is_file()
            ]
            write_paramfile(
                param_path,
                refimage=ref_dst,
                images=all_images,
                global_params=HSTDataModel.DOLPHOT_BASE_PARAMS,
                xytfile=xyt_path,
            )
        # Prefer user phot_out name in the launch command even if param
        # still lists the default staging name.
    else:
        from st123.stages.photometry.dolphot import write_paramfile

        write_paramfile(
            out / 'dolphot.param',
            refimage=ref_dst,
            images=list(nircam_staged) + list(hst_staged_src),
            global_params=HSTDataModel.DOLPHOT_BASE_PARAMS,
            xytfile=xyt_path,
        )
        param_path = out / 'dolphot.param'

    # Collect staged HST chip / MEF products listed in the paramfile.
    _, sci_bases = parse_param_image_list(param_path)
    nircam_stems = {Path(p).stem for p in nircam_staged} | {ref_base}
    hst_names = [b for b in sci_bases if b not in nircam_stems]

    cmd = dolphot_command(
        out,
        phot_out=phot_out,
        param_file=param_path,
        dolphot_bin=dolphot_bin,
        ncores=ncores,
    )

    n_xyt = sum(1 for _ in xyt_path.open())
    readme = out / 'WARMSTART_README.txt'
    readme.write_text(
        'DOLPHOT NIRCam->HST warm-start run\n'
        f'NIRCam source: {nircam_dir}\n'
        f'Photometry seed: {phot_src}\n'
        f'Reference (img0): {ref_dst.name}\n'
        f'xytfile: {xyt_path.name} ({n_xyt} stars)\n'
        f'xyt prune_for_hst: {prune_xyt_for_hst}\n'
        f'xyt types: {list(xyt_types) if xyt_types is not None else None}\n'
        f'xyt snr_min: {xyt_snr_min}\n'
        f'xyt crowd_max: {xyt_crowd_max}\n'
        f'xyt sharp2_max: {xyt_sharp2_max}\n'
        f'xyt min_sep_arcsec: {xyt_min_sep_arcsec}\n'
        f'NIRCam science frames: {len(nircam_staged)}\n'
        f'HST JHAT inputs: {len(hst_sources)}\n'
        f'Parameter file: {param_path.name}\n'
        f'Phot output: {phot_out}\n'
        f'Globals: UseWCS=2 Align=0 Force1=1 PSFres=0 (HSTDataModel.DOLPHOT_BASE_PARAMS)\n'
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
        hst_images=hst_names,
        command=cmd,
        commands=[cmd],
    )
