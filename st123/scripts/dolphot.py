#!/usr/bin/env python3
"""Prepare JWST/HST frames for DOLPHOT (stage, mask, calcsky, dolphot.param).

Default with ``--base-dir``: discover mosaic coadds / ``dolphot_frames.txt``
under ``reference/group_*/ref_*/``. Pass ``--files`` and/or ``--refimage`` to
use an explicit frame list instead.

HST one-target mixed run (all cameras, best coadd as ``img0``)::

    dolphot-prep --instruments hst --base-dir /path/to/Target --ncores 8

See README "HST one-target end-to-end" for the full download->dolphot recipe.
"""

from __future__ import annotations

import glob
import logging
import shutil
from pathlib import Path

from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    add_instruments_arg,
    add_sky_coord_args,
    configure_logging_from_args,
    create_parser as build_parser,
    dataset_label,
    resolve_photometry_instrument,
    resolve_project_root,
    resolve_reduction_dir,
)
from st123.datamodels import HSTDataModel, MIRIDataModel
from st123.utils.logging import shutdown_logging

logger = logging.getLogger(__name__)

_HST_INSTRUMENTS = frozenset({'acs', 'wfc3', 'wfpc2'})
_HST_MIXED = 'hst'
_HST_CLI_INSTRUMENTS = frozenset({*_HST_INSTRUMENTS, _HST_MIXED})
_CLI_INSTRUMENTS = ('nircam', 'miri', 'acs', 'wfc3', 'wfpc2', 'hst')


def create_parser():
    parser = build_parser(
        description=(
            'Stage JHAT frames for DOLPHOT: write dolphot.param, run '
            'instrument mask + calcsky (and HST splitgroups). Default with '
            '--base-dir: discover mosaic coadds under '
            'reference/group_*/ref_*/ (and dolphot_frames.txt) and prep every '
            'mosaic box for the requested instrument. Override with '
            '--files and/or --refimage for an explicit frame list. '
            'HST: --instruments wfc3|wfpc2|acs --base-dir ... stages under '
            '<project>/dolphot/<instrument>_0_0/. '
            'Mixed: --instruments hst uses all JHAT frames and the '
            'best coadd reference under dolphot/hst_0_0/.'
        ),
    )
    add_instruments_arg(
        parser,
        default=['nircam'],
        help=(
            'Instrument / mission for DOLPHOT prep (case-insensitive). '
            'Aliases: hst (mixed ACS+WFC3+WFPC2), jwst (-> nircam), or '
            'explicit nircam|miri|acs|wfc3|wfpc2. Selects mask binary, '
            'calcsky defaults, and frame filtering. Alias: --instrument.'
        ),
    )
    add_sky_coord_args(parser, required=False)
    parser.add_argument(
        '--from-mosaic',
        action='store_true',
        help=(
            'Deprecated no-op: mosaic discovery is already the default when '
            '--base-dir is set without --files/--refimage. Kept for older '
            'scripts.'
        ),
    )
    parser.add_argument(
        '--files',
        nargs='+',
        default=None,
        help=(
            'Explicit FITS paths (JHAT frames). Disables default mosaic '
            'discovery.'
        ),
    )
    add_base_dir(
        parser,
        required=False,
        aliases=('--basedir', '--workdir', '--data-dir', '--indir'),
        help=(
            'Project root or reduction workdir. Default: discover mosaic '
            'products under reduction/reference/. With --files/--refimage, '
            'also used to glob *_jhat.fits when --files is omitted. '
            'Aliases: --indir, --workdir, --data-dir.'
        ),
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default=None,
        help=(
            'Staging directory. In mosaic mode, overrides the default '
            '(reduction/phot_* or <project>/dolphot/miri_*). HST default: '
            '<project>/dolphot/<instrument>_0_0.'
        ),
    )
    parser.add_argument(
        '--refimage',
        type=str,
        default=None,
        help=(
            'Explicit reference image for dolphot.param. Disables default '
            'mosaic discovery (use with --files or --base-dir JHAT glob).'
        ),
    )
    parser.add_argument(
        '--ref-filter',
        type=str,
        default=None,
        help=(
            'In mosaic mode, prefer coadd_*_<filter>_*.fits (JWST i2d or '
            'HST drc/drz) as the reference. Default: F560W for MIRI; for '
            'NIRCam, auto-prefer SW i2d (F150W2, F200W, F150W, ...).'
        ),
    )
    parser.add_argument(
        '--group',
        type=int,
        default=None,
        help='Only prep this mosaic group index (default: all discovered).',
    )
    parser.add_argument(
        '--box',
        type=str,
        default=None,
        help=(
            'Only prep this mosaic box id (int or label, e.g. 0 or sn). '
            'Default: all discovered boxes.'
        ),
    )
    parser.add_argument(
        '--dolphot-bin',
        type=str,
        default=None,
        help=(
            'DOLPHOT bin directory containing dolphot / *mask / calcsky / '
            'splitgroups. Default: directory of `dolphot` on PATH.'
        ),
    )
    parser.add_argument(
        '--skip-mask',
        action='store_true',
        help='Skip instrument mask step.',
    )
    parser.add_argument(
        '--skip-sky',
        action='store_true',
        help='Skip calcsky.',
    )
    parser.add_argument(
        '--skip-split',
        action='store_true',
        help='Skip HST splitgroups (multi-extension -> per-chip).',
    )
    add_common_runtime(parser, ncores=True, ncores_default=1, plot=False, verbose=True)
    return parser


def _log_run_commands(
    outdir: Path,
    *,
    phot_out: str,
    param_file: str,
    dolphot_bin: str | None,
    ncores: int,
) -> None:
    from st123.stages.photometry.dolphot import dolphot_command

    logger.info(
        'Run DOLPHOT with:\n  %s',
        dolphot_command(
            outdir,
            phot_out=phot_out,
            param_file=param_file,
            dolphot_bin=dolphot_bin,
            ncores=ncores,
        ),
    )
    logger.info(
        'Or detach with nohup:\n  %s',
        dolphot_command(
            outdir,
            phot_out=phot_out,
            param_file=param_file,
            dolphot_bin=dolphot_bin,
            ncores=ncores,
            nohup=True,
        ),
    )


def _pick_hst_reference(
    reduction: Path,
    frames: list[Path],
    *,
    instrument: str | None = None,
) -> Path:
    """
    Prefer deepest/longest-filter coadd under ``reference/group_*/ref_*``.

    Also accepts legacy flat ``reference/coadd_*.fits``. When *instrument* is a
    single camera (``wfc3`` / ``acs`` / ``wfpc2``), coadds whose filename
    contains that instrument are preferred. For mixed ``hst`` (or ``None``),
    prefer WFC3 -> ACS -> WFPC2, then
    :attr:`HSTDataModel.BEST_REFERENCE_FILTERS`.
    """
    ref_dir = reduction / 'reference'
    coadds: list[Path] = []
    if ref_dir.is_dir():
        boxed = sorted(ref_dir.glob('group_*/ref_*/coadd_*.fits'))
        flat = sorted(ref_dir.glob('coadd_*.fits'))
        for p in boxed + flat:
            name = p.name.lower()
            if not (
                name.endswith('_drc.fits') or name.endswith('_drz.fits')
            ):
                continue
            coadds.append(p)
    if coadds:
        inst = (instrument or '').lower()
        mixed = inst in ('', _HST_MIXED)

        def _rank(path: Path) -> tuple[int, int, int, str]:
            name = path.name.lower()
            # Prefer boxed layout over legacy flat products.
            layout_rank = 0 if 'group_' in path.as_posix() else 1
            if mixed:
                if 'wfc3' in name:
                    inst_rank = 0
                elif 'acs' in name:
                    inst_rank = 1
                elif 'wfpc2' in name:
                    inst_rank = 2
                else:
                    inst_rank = 3
            else:
                inst_rank = 0 if (inst and inst in name) else 1
            filt_rank = len(HSTDataModel.BEST_REFERENCE_FILTERS)
            for i, filt in enumerate(HSTDataModel.BEST_REFERENCE_FILTERS):
                if f'_{filt}_' in name:
                    filt_rank = i
                    break
            else:
                try:
                    from st123.utils.helpers import get_filter

                    filt = get_filter(path)
                    for i, pref in enumerate(HSTDataModel.BEST_REFERENCE_FILTERS):
                        if filt == pref:
                            filt_rank = i
                            break
                except Exception:
                    pass
            return (layout_rank, inst_rank, filt_rank, name)

        return sorted(coadds, key=_rank)[0]
    if frames:
        return frames[0]
    raise FileNotFoundError(f'No HST reference coadd or frames under {reduction}')


def _collect_hst_frames(reduction: Path, instrument: str) -> list[Path]:
    """Prefer jhat frames; fall back to reduction/raw calibrated products."""
    from st123.datamodels.hst import filter_paths_for_stage
    from st123.stages.photometry.dolphot import filter_frames_for_instrument

    jhat_dirs = (reduction / 'jhat_hst', reduction / 'jhat')
    raw = reduction / 'raw'
    files: list[Path] = []
    for jhat in jhat_dirs:
        if not jhat.is_dir():
            continue
        files.extend(
            sorted(
                p
                for p in jhat.glob('*_jhat.fits')
                if not p.name.lower().startswith('coadd_')
            )
        )
    if not files and raw.is_dir():
        patterns = ('*_flc.fits', '*_flt.fits', '*_c0m.fits')
        for pat in patterns:
            files.extend(sorted(raw.glob(pat)))
        files = sorted(set(files))
    else:
        files = sorted(set(files))
    files = filter_paths_for_stage(files, stage='dolphot-collect')
    if instrument.lower() == _HST_MIXED:
        return files
    return filter_frames_for_instrument(files, instrument)


def _run_hst(args) -> int:
    from st123.stages.photometry.dolphot import (
        filter_frames_for_instrument,
        prepare_hst_frames,
    )

    if args.base_dir is None and not args.files:
        logger.error('HST dolphot-prep requires --base-dir or --files')
        return 1

    instrument = args.instrument
    project = (
        Path(resolve_project_root(args.base_dir))
        if args.base_dir
        else Path.cwd()
    )
    reduction = (
        Path(resolve_reduction_dir(args.base_dir))
        if args.base_dir
        else project
    )
    if args.verbose:
        logger.info('Dataset: %s', dataset_label(args.base_dir or project))
        logger.info('HST DOLPHOT prep: instrument=%s', instrument)

    if args.files:
        files = [Path(f) for f in args.files]
        if instrument != _HST_MIXED:
            files = filter_frames_for_instrument(files, instrument)
    else:
        files = _collect_hst_frames(reduction, instrument)
    if not files:
        logger.error(
            'no %s frames under %s (jhat/ or raw/)',
            instrument,
            reduction,
        )
        return 1

    if args.refimage:
        refimage = Path(args.refimage)
    else:
        refimage = _pick_hst_reference(
            reduction, files, instrument=instrument
        )

    outdir = (
        Path(args.outdir)
        if args.outdir
        else project / 'dolphot' / f'{instrument}_0_0'
    )
    if args.verbose:
        logger.info(
            'Prep %s: ref=%s frames=%d outdir=%s',
            instrument,
            refimage.name,
            len(files),
            outdir,
        )

    param = prepare_hst_frames(
        files,
        outdir,
        instrument,
        refimage=refimage,
        dolphot_bin=args.dolphot_bin,
        skip_mask=args.skip_mask,
        skip_sky=args.skip_sky,
        skip_split=args.skip_split,
        ncores=args.ncores,
    )
    logger.info('Wrote %s (%d frames + ref)', param, len(files))
    _log_run_commands(
        outdir,
        phot_out=f'{outdir.name}.phot',
        param_file=param.name,
        dolphot_bin=args.dolphot_bin,
        ncores=args.ncores,
    )
    return 0


def _run_from_mosaic(args) -> int:
    from st123.stages.photometry.dolphot import (
        MosaicPhotJob,
        discover_mosaic_phot_jobs,
        prepare_hst_frames,
        prepare_mosaic_phot_job,
    )

    if args.base_dir is None:
        logger.error('mosaic discovery requires --base-dir')
        return 1

    reduction = Path(resolve_reduction_dir(args.base_dir))
    project = Path(resolve_project_root(args.base_dir))
    if args.verbose:
        logger.info('Dataset: %s', dataset_label(args.base_dir))
        logger.info('Reduction workdir: %s', reduction)

    instrument = args.instrument
    ref_filter = args.ref_filter
    if ref_filter is None and instrument == 'miri':
        ref_filter = 'F560W'

    hst_mode = instrument in _HST_CLI_INSTRUMENTS
    if instrument == 'miri':
        out_root = project / 'dolphot'
        outdir_prefix = 'miri'
    elif hst_mode:
        out_root = project / 'dolphot'
        outdir_prefix = instrument
    else:
        out_root = reduction
        outdir_prefix = 'phot'

    jobs = discover_mosaic_phot_jobs(
        reduction,
        instrument=instrument,
        ref_filter=ref_filter,
        phot_outdir_root=out_root,
        outdir_prefix=outdir_prefix,
    )
    want_group = getattr(args, 'group', None)
    want_box = getattr(args, 'box', None)
    if want_group is not None or want_box is not None:
        box_token = None if want_box is None else str(want_box).strip()
        filtered = []
        for job in jobs:
            if want_group is not None and int(job.group) != int(want_group):
                continue
            if box_token is not None and str(job.box) != box_token:
                continue
            filtered.append(job)
        jobs = filtered
    if not jobs:
        if hst_mode and want_group is None and want_box is None:
            # Legacy / no-box fallback: stage a single HST phot dir.
            return _run_hst(args)
        logger.error(
            'no mosaic coadds / dolphot_frames.txt under %s'
            '%s%s',
            reduction / 'reference',
            f' for group={want_group}' if want_group is not None else '',
            f' box={want_box}' if want_box is not None else '',
        )
        return 1

    # Optional single-outdir override (one box expected).
    if args.outdir is not None:
        if len(jobs) > 1:
            logger.error(
                '--outdir in mosaic mode requires a single mosaic box '
                '(found %d)',
                len(jobs),
            )
            return 1
        job0 = jobs[0]
        jobs = [
            MosaicPhotJob(
                group=job0.group,
                box=job0.box,
                refimage=job0.refimage,
                frames=job0.frames,
                phot_outdir=Path(args.outdir),
                frame_list=job0.frame_list,
            )
        ]

    for job in jobs:
        if args.verbose:
            logger.info(
                'Prep %s: ref=%s frames=%d outdir=%s',
                job.phot_outdir.name,
                job.refimage.name,
                len(job.frames),
                job.phot_outdir,
            )
        if hst_mode:
            param = prepare_hst_frames(
                list(job.frames),
                job.phot_outdir,
                instrument,
                refimage=job.refimage,
                dolphot_bin=args.dolphot_bin,
                skip_mask=args.skip_mask,
                skip_sky=args.skip_sky,
                skip_split=args.skip_split,
                ncores=args.ncores,
            )
        else:
            param = prepare_mosaic_phot_job(
                job,
                instrument=instrument,
                dolphot_bin=args.dolphot_bin,
                skip_mask=args.skip_mask,
                skip_sky=args.skip_sky,
                ncores=args.ncores,
            )
        logger.info('Wrote %s (%d frames + ref)', param, len(job.frames))
        phot_out = f'{job.phot_outdir.name}.phot'
        _log_run_commands(
            job.phot_outdir,
            phot_out=phot_out,
            param_file=param.name,
            dolphot_bin=args.dolphot_bin,
            ncores=args.ncores,
        )
    return 0


def _run_explicit(args) -> int:
    from st123.stages.photometry.dolphot import prepare_frames, setup_paramfile

    if args.instrument in _HST_CLI_INSTRUMENTS:
        return _run_hst(args)

    if args.files:
        files = [Path(f) for f in args.files]
    elif args.base_dir:
        search = Path(resolve_reduction_dir(args.base_dir))
        # Prefer jhat/ under a reduction workdir.
        jhat_dir = search / 'jhat'
        pattern = str(jhat_dir / '*_jhat.fits') if jhat_dir.is_dir() else str(
            search / '*_jhat.fits'
        )
        files = sorted(Path(p) for p in glob.glob(pattern))
    else:
        logger.error('provide --base-dir, or --files / --refimage')
        return 1

    missing = [p for p in files if not p.is_file()]
    if missing:
        logger.error('missing files: %s', missing[0])
        return 1
    if not files:
        logger.error('no input FITS files')
        return 1

    work = files
    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        staged = []
        for src in files:
            dst = outdir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            staged.append(dst)
        work = staged
        if args.refimage:
            ref_src = Path(args.refimage)
            ref_dst = outdir / ref_src.name
            if not ref_dst.exists():
                shutil.copy2(ref_src, ref_dst)
            if args.instrument == 'miri':
                global_params = MIRIDataModel.DOLPHOT_BASE_PARAMS
            elif args.instrument in _HST_INSTRUMENTS:
                global_params = HSTDataModel.DOLPHOT_BASE_PARAMS
            else:
                global_params = None
            plan = setup_paramfile(
                outdir,
                ref_dst,
                work,
                copy_files=False,
                global_params=global_params,
                phot_out=f'{outdir.name}.phot',
                return_plan=True,
            )
            logger.info(
                'Wrote %s (%d part(s))',
                plan.param_file,
                len(plan.parts),
            )
            for part in plan.parts:
                _log_run_commands(
                    outdir,
                    phot_out=part.phot_out,
                    param_file=part.param_file,
                    dolphot_bin=args.dolphot_bin,
                    ncores=args.ncores,
                )

    prepare_frames(
        work,
        instrument=args.instrument,
        dolphot_bin=args.dolphot_bin,
        skip_mask=args.skip_mask,
        skip_sky=args.skip_sky,
        ncores=args.ncores,
    )
    logger.info('Prepared %d %s frame(s)', len(work), args.instrument)
    return 0


def use_mosaic_discovery(args) -> bool:
    """
    True when dolphot-prep should discover mosaic coadds / frame lists.

    Default whenever ``--base-dir`` is set. Opt out with ``--files`` and/or
    ``--refimage``. ``--from-mosaic`` is accepted but redundant.
    """
    if getattr(args, 'files', None) or getattr(args, 'refimage', None):
        return False
    if getattr(args, 'base_dir', None) is None:
        return False
    return True


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)
    configure_logging_from_args(args, 'dolphot-prep')
    try:
        try:
            args.instrument = resolve_photometry_instrument(args.instruments)
        except ValueError as exc:
            logger.error('%s', exc)
            return 2
        try:
            if use_mosaic_discovery(args):
                return _run_from_mosaic(args)
            return _run_explicit(args)
        except FileNotFoundError as exc:
            logger.error('%s', exc)
            return 1
    finally:
        shutdown_logging()


if __name__ == '__main__':
    raise SystemExit(main())
