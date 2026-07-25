#!/usr/bin/env python3
"""Set up a NIRCam→MIRI DOLPHOT warm-start run (does not execute dolphot)."""

from __future__ import annotations

import argparse
from pathlib import Path

from st123.photometry.warmstart import setup_miri_warmstart
from st123.utils.settings import DEFAULT_DOLPHOT_BIN


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Prepare a warm-start DOLPHOT directory that appends overlapping '
            'MIRI JHAT frames to an existing NIRCam run (xytfile mode).'
        ),
    )
    parser.add_argument(
        '--nircam-dir',
        type=str,
        required=True,
        help='Existing NIRCam DOLPHOT run directory (dolphot.param + .phot).',
    )
    parser.add_argument(
        '--outdir',
        type=str,
        required=True,
        help='Output warm-start run directory to create.',
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help='JWST data root for MIRI discovery (default: guess from nircam-dir).',
    )
    parser.add_argument(
        '--alignment-summary',
        type=str,
        default=None,
        help='Alignment summary listing SUCCESS MIRI JHAT paths.',
    )
    parser.add_argument(
        '--miri-jhat',
        nargs='+',
        default=None,
        help='Explicit MIRI *_jhat.fits paths (overrides discovery).',
    )
    parser.add_argument(
        '--phot-file',
        type=str,
        default=None,
        help='NIRCam .phot catalog for warmstart.xyt (default: auto-detect).',
    )
    parser.add_argument(
        '--min-overlap',
        type=float,
        default=0.0,
        help='Minimum ref_overlap_frac when reading an alignment summary.',
    )
    parser.add_argument(
        '--phot-out',
        type=str,
        default=None,
        help='DOLPHOT output catalog name (default: <seed>_nircam_miri.phot).',
    )
    parser.add_argument(
        '--dolphot-bin',
        type=str,
        default=DEFAULT_DOLPHOT_BIN,
        help=f'DOLPHOT bin directory (default: {DEFAULT_DOLPHOT_BIN}).',
    )
    parser.add_argument(
        '--skip-miri-prep',
        action='store_true',
        help='Do not run mirimask/calcsky (frames already prepared).',
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy NIRCam products instead of hardlinking.',
    )
    return parser


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)
    result = setup_miri_warmstart(
        args.nircam_dir,
        args.outdir,
        miri_jhat=args.miri_jhat,
        data_root=args.data_root,
        alignment_summary=args.alignment_summary,
        phot_file=args.phot_file,
        min_overlap=args.min_overlap,
        dolphot_bin=args.dolphot_bin,
        phot_out=args.phot_out,
        prepare_miri=not args.skip_miri_prep,
        use_hardlink=not args.copy,
    )
    print(f'Warm-start directory: {result.outdir}')
    print(f'  NIRCam frames: {len(result.nircam_images)}')
    print(f'  MIRI frames:   {len(result.miri_images)}')
    print(f'  Param file:    {result.param_file}')
    print(f'  xyt file:      {result.xyt_file}')
    print(f'  Phot output:   {result.phot_out}')
    print()
    print('Run DOLPHOT with:')
    print(f'  {result.command}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
