#!/usr/bin/env python3
"""Prepare JWST frames for DOLPHOT (stage, mask, calcsky, dolphot.param)."""

from __future__ import annotations

import glob
import shutil
import sys
from pathlib import Path

from st123.photometry.dolphot_prep import (
    discover_mosaic_phot_jobs,
    prepare_frames,
    prepare_mosaic_phot_job,
    setup_paramfile,
)
from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    create_parser as build_parser,
    dataset_label,
    resolve_reduction_dir,
)
from st123.utils.settings import DEFAULT_DOLPHOT_BIN


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
        default=DEFAULT_DOLPHOT_BIN,
        help=f'DOLPHOT bin directory (default: {DEFAULT_DOLPHOT_BIN}).',
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
    add_common_runtime(parser, ncores=False, plot=False, verbose=True)
    return parser


def _run_from_mosaic(args) -> int:
    if args.base_dir is None:
        print('ERROR: --from-mosaic requires --base-dir', file=sys.stderr)
        return 1
    reduction = Path(resolve_reduction_dir(args.base_dir))
    if args.verbose:
        print(f'Dataset: {dataset_label(args.base_dir)}')
        print(f'Reduction workdir: {reduction}')
    jobs = discover_mosaic_phot_jobs(reduction)
    if not jobs:
        print(
            f'ERROR: no mosaic coadds / dolphot_frames.txt under '
            f'{reduction / "reference"}',
            file=sys.stderr,
        )
        return 1
    for job in jobs:
        if args.verbose:
            print(
                f'Prep phot_{job.group}_{job.box}: '
                f'ref={job.refimage.name} frames={len(job.frames)}'
            )
        param = prepare_mosaic_phot_job(
            job,
            instrument=args.instrument,
            dolphot_bin=args.dolphot_bin,
            skip_mask=args.skip_mask,
            skip_sky=args.skip_sky,
        )
        print(f'Wrote {param} ({len(job.frames)} frames + ref)')
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
        print('ERROR: provide --files, --base-dir, or --from-mosaic', file=sys.stderr)
        return 1

    missing = [p for p in files if not p.is_file()]
    if missing:
        print(f'ERROR: missing files: {missing[0]}', file=sys.stderr)
        return 1
    if not files:
        print('ERROR: no input FITS files', file=sys.stderr)
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
            print(f'Wrote {outdir / "dolphot.param"}')

    prepare_frames(
        work,
        instrument=args.instrument,
        dolphot_bin=args.dolphot_bin,
        skip_mask=args.skip_mask,
        skip_sky=args.skip_sky,
    )
    print(f'Prepared {len(work)} {args.instrument} frame(s)')
    return 0


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)
    if args.from_mosaic:
        return _run_from_mosaic(args)
    return _run_explicit(args)


if __name__ == '__main__':
    raise SystemExit(main())
