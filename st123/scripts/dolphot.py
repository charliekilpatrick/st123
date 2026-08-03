#!/usr/bin/env python3
"""Prepare JWST/HST frames for DOLPHOT (stage, mask, calcsky, dolphot.param).

HST one-target mixed run (all cameras, best coadd as ``img0``)::

    dolphot-prep --instrument hst --base-dir /path/to/Target --ncores 8

See README "HST one-target end-to-end" for the full download→dolphot recipe.
"""

from __future__ import annotations

import glob
import logging
import shutil
from pathlib import Path

from st123.photometry.dolphot import (
    MosaicPhotJob,
    discover_mosaic_phot_jobs,
    dolphot_command,
    filter_frames_for_instrument,
    prepare_frames,
    prepare_hst_frames,
    prepare_mosaic_phot_job,
    setup_paramfile,
)
from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    configure_logging_from_args,
    create_parser as build_parser,
    dataset_label,
    resolve_project_root,
    resolve_reduction_dir,
)
from st123.utils.helpers import get_filter
from st123.utils.settings import BEST_REFERENCE_FILTERS, hst_base_params, miri_base_params
from st123.utils.logging import shutdown_logging

logger = logging.getLogger(__name__)

_HST_INSTRUMENTS = frozenset({'acs', 'wfc3', 'wfpc2'})
_HST_MIXED = 'hst'
_HST_CLI_INSTRUMENTS = frozenset({*_HST_INSTRUMENTS, _HST_MIXED})


def create_parser():
    parser = build_parser(
        description=(
            'Stage JHAT frames for DOLPHOT: write dolphot.param, run '
            'instrument mask + calcsky (and HST splitgroups). Use '
            '--from-mosaic after mosaic, or pass --files / --refimage. '
            'HST: --instrument wfc3|wfpc2|acs --base-dir … stages under '
            '<project>/dolphot/<instrument>_0_0/. '
            'Mixed (option C): --instrument hst uses all JHAT frames and the '
            'best coadd reference under dolphot/hst_0_0/.'
        ),
    )
    parser.add_argument(
        '--instrument',
        choices=('nircam', 'miri', 'acs', 'wfc3', 'wfpc2', 'hst'),
        default='nircam',
        help=(
            'Instrument module (selects mask binary, calcsky defaults, and '
            'frame filtering). Use hst for a mixed ACS+WFC3+WFPC2 run against '
            'the best coadd reference.'
        ),
    )
    parser.add_argument(
        '--from-mosaic',
        action='store_true',
        help=(
            'Discover coadds under <reduction>/reference/group_*/ref_*/ '
            '(and dolphot_frames.txt when present) and prep phot/miri runs.'
        ),
    )
    parser.add_argument(
        '--files',
        nargs='+',
        default=None,
        help='Explicit FITS paths (JHAT frames). Ignored with --from-mosaic.',
    )
    add_base_dir(
        parser,
        required=False,
        aliases=('--basedir', '--workdir', '--data-dir', '--indir'),
        help=(
            'Project root or reduction workdir. With --from-mosaic, resolves '
            'to reduction/. With --files omitted (non-mosaic), glob for '
            '*_jhat.fits here. Aliases: --indir, --workdir, --data-dir.'
        ),
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default=None,
        help=(
            'Staging directory. With --from-mosaic, overrides the default '
            '(reduction/phot_* or <project>/dolphot/miri_*). HST default: '
            '<project>/dolphot/<instrument>_0_0.'
        ),
    )
    parser.add_argument(
        '--refimage',
        type=str,
        default=None,
        help='Reference image for dolphot.param (--files / HST mode).',
    )
    parser.add_argument(
        '--ref-filter',
        type=str,
        default=None,
        help=(
            'With --from-mosaic, prefer coadd_*_<filter>_i2d.fits as the '
            'reference. Default: F560W when --instrument miri, else the '
            'manifest # ref line.'
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
        help='Skip HST splitgroups (multi-extension → per-chip).',
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
    Prefer deepest/longest-filter coadd under reduction/reference, else first frame.

    When *instrument* is a single camera (``wfc3`` / ``acs`` / ``wfpc2``),
    coadds whose filename contains that instrument are preferred. For mixed
    ``hst`` (or ``None``), prefer WFC3 → ACS → WFPC2, then
    :data:`~st123.utils.settings.BEST_REFERENCE_FILTERS`.
    """
    ref_dir = reduction / 'reference'
    coadds: list[Path] = []
    if ref_dir.is_dir():
        coadds = sorted(
            p
            for p in ref_dir.iterdir()
            if p.is_file()
            and p.name.startswith('coadd_')
            and (
                p.name.endswith('_drc.fits')
                or p.name.endswith('_drz.fits')
            )
        )
    if coadds:
        inst = (instrument or '').lower()
        mixed = inst in ('', _HST_MIXED)

        def _rank(path: Path) -> tuple[int, int, str]:
            name = path.name.lower()
            if mixed:
                if 'coadd_wfc3_' in name:
                    inst_rank = 0
                elif 'coadd_acs_' in name:
                    inst_rank = 1
                elif 'coadd_wfpc2_' in name:
                    inst_rank = 2
                else:
                    inst_rank = 3
            else:
                inst_rank = 0 if (inst and f'coadd_{inst}_' in name) else 1
            filt_rank = len(BEST_REFERENCE_FILTERS)
            for i, filt in enumerate(BEST_REFERENCE_FILTERS):
                if f'_{filt}_' in name:
                    filt_rank = i
                    break
            else:
                try:
                    filt = get_filter(path)
                    for i, pref in enumerate(BEST_REFERENCE_FILTERS):
                        if filt == pref:
                            filt_rank = i
                            break
                except Exception:
                    pass
            return (inst_rank, filt_rank, name)

        return sorted(coadds, key=_rank)[0]
    if frames:
        return frames[0]
    raise FileNotFoundError(f'No HST reference coadd or frames under {reduction}')


def _collect_hst_frames(reduction: Path, instrument: str) -> list[Path]:
    """Prefer jhat frames; fall back to reduction/raw calibrated products."""
    jhat = reduction / 'jhat'
    raw = reduction / 'raw'
    files: list[Path] = []
    if jhat.is_dir():
        files = sorted(
            p
            for p in jhat.glob('*_jhat.fits')
            if not p.name.lower().startswith('coadd_')
        )
    if not files and raw.is_dir():
        patterns = ('*_flc.fits', '*_flt.fits', '*_c0m.fits')
        for pat in patterns:
            files.extend(sorted(raw.glob(pat)))
        files = sorted(set(files))
    if instrument.lower() == _HST_MIXED:
        return files
    return filter_frames_for_instrument(files, instrument)


def _run_hst(args) -> int:
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
    if args.base_dir is None:
        logger.error('--from-mosaic requires --base-dir')
        return 1
    if args.instrument in _HST_CLI_INSTRUMENTS:
        # HST coadds live flat under reference/; reuse HST staging path.
        return _run_hst(args)

    reduction = Path(resolve_reduction_dir(args.base_dir))
    project = Path(resolve_project_root(args.base_dir))
    if args.verbose:
        logger.info('Dataset: %s', dataset_label(args.base_dir))
        logger.info('Reduction workdir: %s', reduction)

    instrument = args.instrument
    ref_filter = args.ref_filter
    if ref_filter is None and instrument == 'miri':
        ref_filter = 'F560W'

    if instrument == 'miri':
        out_root = project / 'dolphot'
        outdir_prefix = 'miri'
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
    if not jobs:
        logger.error(
            'no mosaic coadds / dolphot_frames.txt under %s',
            reduction / 'reference',
        )
        return 1

    # Optional single-outdir override (one box expected).
    if args.outdir is not None:
        if len(jobs) > 1:
            logger.error(
                '--outdir with --from-mosaic requires a single mosaic box '
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
        param = prepare_mosaic_phot_job(
            job,
            instrument=instrument,
            dolphot_bin=args.dolphot_bin,
            skip_mask=args.skip_mask,
            skip_sky=args.skip_sky,
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
        logger.error('provide --files, --base-dir, or --from-mosaic')
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
                global_params = miri_base_params
            elif args.instrument in _HST_INSTRUMENTS:
                global_params = hst_base_params
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
    )
    logger.info('Prepared %d %s frame(s)', len(work), args.instrument)
    return 0


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)
    configure_logging_from_args(args, 'dolphot-prep')
    try:
        try:
            if args.from_mosaic:
                return _run_from_mosaic(args)
            return _run_explicit(args)
        except FileNotFoundError as exc:
            logger.error('%s', exc)
            return 1
    finally:
        shutdown_logging()


if __name__ == '__main__':
    raise SystemExit(main())
