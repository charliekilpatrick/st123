#!/usr/bin/env python3
"""Prepare JWST JHAT frames for DOLPHOT (nircammask/mirimask + calcsky)."""

from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

from st123.photometry.dolphot_prep import prepare_frames, setup_paramfile
from st123.utils.settings import DEFAULT_DOLPHOT_BIN


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Mask and compute sky maps for NIRCam or MIRI JHAT frames using '
            'the local DOLPHOT installation.'
        ),
    )
    parser.add_argument(
        '--instrument',
        choices=('nircam', 'miri'),
        required=True,
        help='Instrument module (selects mask binary and calcsky defaults).',
    )
    parser.add_argument(
        '--files',
        nargs='+',
        default=None,
        help='Explicit FITS paths. If omitted, use --indir glob.',
    )
    parser.add_argument(
        '--indir',
        type=str,
        default=None,
        help='Directory to search for *_jhat.fits when --files is omitted.',
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default=None,
        help=(
            'If set, copy frames here and optionally write dolphot.param '
            '(requires --refimage).'
        ),
    )
    parser.add_argument(
        '--refimage',
        type=str,
        default=None,
        help='Reference image for dolphot.param (with --outdir).',
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
    return parser


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)

    if args.files:
        files = [Path(f) for f in args.files]
    elif args.indir:
        files = sorted(Path(p) for p in glob.glob(str(Path(args.indir) / '*_jhat.fits')))
    else:
        print('ERROR: provide --files or --indir')
        return 1

    missing = [p for p in files if not p.is_file()]
    if missing:
        print(f'ERROR: missing files: {missing[0]}')
        return 1
    if not files:
        print('ERROR: no input FITS files')
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


if __name__ == '__main__':
    raise SystemExit(main())
