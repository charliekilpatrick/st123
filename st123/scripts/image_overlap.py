#!/usr/bin/env python3
"""Find the reference image with maximum footprint overlap for science frames."""

from __future__ import annotations

from st123.mosaic.image_overlap import find_best_refs
from st123.scripts.utils.options import (
    add_common_runtime,
    add_image_arg,
    create_parser as build_parser,
)


def create_parser():
    parser = build_parser(
        description=(
            'Find the reference image with maximum footprint overlap '
            'for each science frame (e.g. MIRI cal products).'
        )
    )
    add_image_arg(
        parser,
        nargs='+',
        required=True,
        help='Science *_cal.fits image(s) to score for overlap.',
    )
    parser.add_argument(
        '--ref',
        nargs='+',
        required=True,
        help='Reference coadd *_i2d.fits image(s).',
    )
    parser.add_argument(
        '--outfile',
        type=str,
        default=None,
        help='Optional text file for summary lines (default: print only).',
    )
    add_common_runtime(parser, ncores=False, plot=False, verbose=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    science_images = list(args.image)
    refs = list(args.ref)

    print(f'Science images ({len(science_images)}):')
    for path in science_images:
        print(f'  {path}')
    print(f'Reference images ({len(refs)}):')
    for path in refs:
        print(f'  {path}')
    print()

    find_best_refs(science_images, refs, outfile=args.outfile)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
