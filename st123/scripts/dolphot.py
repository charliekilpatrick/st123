#!/usr/bin/env python3
"""Prepare JWST frames for DOLPHOT (stage, mask, calcsky, dolphot.param)."""

from __future__ import annotations

import glob
import logging
import shutil
from pathlib import Path

from st123.photometry.dolphot import (
    discover_mosaic_phot_jobs,
    dolphot_command,
    prepare_frames,
    prepare_mosaic_phot_job,
    setup_paramfile,
)
from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    configure_logging_from_args,
    create_parser as build_parser,
    dataset_label,
    resolve_reduction_dir,
)
from st123.utils.logging import shutdown_logging

logger = logging.getLogger(__name__)


def create_parser():
    parser = build_parser(
        description=(
            'Stage JHAT frames for DOLPHOT: write dolphot.param, run '
            'nircammask/mirimask and calcsky. Use --from-mosaic after mosaic, '
            'or pass --files / --refimage explicitly.'
        ),
    )
    parser.add_argument(
        '--instrument',
        choices=('nircam', 'miri'),
        default='nircam',
        help='Instrument module (selects mask binary and calcsky defaults).',
    )
    parser.add_argument(
        '--from-mosaic',
        action='store_true',
        help=(
            'Discover coadds under <reduction>/reference/group_*/ref_*/ '
            '(and dolphot_frames.txt when present) and prep phot_* runs.'
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
            'Staging directory for --files mode (writes dolphot.param when '
            '--refimage is set). Ignored with --from-mosaic.'
        ),
    )
    parser.add_argument(
        '--refimage',
        type=str,
        default=None,
        help='Reference image for dolphot.param (--files mode).',
    )
    parser.add_argument(
        '--dolphot-bin',
        type=str,
        default=None,
        help=(
            'DOLPHOT bin directory containing dolphot/nircammask/mirimask/calcsky. '
            'Default: directory of `dolphot` found on PATH (shutil.which).'
        ),
    )
    parser.add_argument(
        '--skip-mask',
        action='store_true',
        help='Skip nircammask/mirimask.',
    )
    parser.add_argument(
        '--skip-sky',
        action='store_true',
        help='Skip calcsky.',
    )
    add_common_runtime(parser, ncores=True, ncores_default=1, plot=False, verbose=True)
    return parser


def _run_from_mosaic(args) -> int:
    if args.base_dir is None:
        logger.error('--from-mosaic requires --base-dir')
        return 1
    reduction = Path(resolve_reduction_dir(args.base_dir))
    if args.verbose:
        logger.info('Dataset: %s', dataset_label(args.base_dir))
        logger.info('Reduction workdir: %s', reduction)
    jobs = discover_mosaic_phot_jobs(reduction)
    if not jobs:
        logger.error(
            'no mosaic coadds / dolphot_frames.txt under %s',
            reduction / 'reference',
        )
        return 1
    for job in jobs:
        if args.verbose:
            logger.info(
                'Prep phot_%s_%s: ref=%s frames=%d',
                job.group,
                job.box,
                job.refimage.name,
                len(job.frames),
            )
        param = prepare_mosaic_phot_job(
            job,
            instrument=args.instrument,
            dolphot_bin=args.dolphot_bin,
            skip_mask=args.skip_mask,
            skip_sky=args.skip_sky,
        )
        logger.info('Wrote %s (%d frames + ref)', param, len(job.frames))
        phot_out = f'phot_{job.group}_{job.box}.phot'
        logger.info(
            'Run DOLPHOT with: %s',
            dolphot_command(
                job.phot_outdir,
                phot_out=phot_out,
                param_file=param.name,
                dolphot_bin=args.dolphot_bin,
                ncores=args.ncores,
            ),
        )
    return 0


def _run_explicit(args) -> int:
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
            setup_paramfile(
                outdir,
                ref_dst,
                work,
                copy_files=False,
            )
            logger.info('Wrote %s', outdir / 'dolphot.param')
            logger.info(
                'Run DOLPHOT with: %s',
                dolphot_command(
                    outdir,
                    phot_out=f'{outdir.name}.phot',
                    param_file='dolphot.param',
                    dolphot_bin=args.dolphot_bin,
                    ncores=args.ncores,
                ),
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
