#!/usr/bin/env python3
"""Find the reference image with maximum footprint overlap for science frames."""

from __future__ import annotations

import logging

from st123.scripts.utils.options import (
    add_common_runtime,
    add_image_arg,
    configure_logging_from_args,
    create_parser as build_parser,
)
from st123.utils.logging import shutdown_logging

logger = logging.getLogger(__name__)


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
        help='Optional text file for summary lines (default: log only).',
    )
    add_common_runtime(parser, plot=False, verbose=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    configure_logging_from_args(args, 'image-overlap')
    try:
        from st123.stages.mosaic.image_overlap import find_best_refs

        science_images = list(args.image)
        refs = list(args.ref)

        logger.info('Science images (%d):', len(science_images))
        for path in science_images:
            logger.info('  %s', path)
        logger.info('Reference images (%d):', len(refs))
        for path in refs:
            logger.info('  %s', path)

        find_best_refs(science_images, refs, outfile=args.outfile)
        return 0
    finally:
        shutdown_logging()


if __name__ == '__main__':
    raise SystemExit(main())
