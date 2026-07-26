#!/usr/bin/env python3
"""Derive an illuminated S_REGION polygon from a FITS science/DQ image."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from st123.mosaic.region import (
    SRegionPolygon,
    illuminated_s_region_from_fits,
    save_illuminated_region_plot,
)
from st123.scripts.utils.options import (
    add_common_runtime,
    configure_logging_from_args,
    create_parser as build_parser,
)
from st123.utils.logging import shutdown_logging

logger = logging.getLogger(__name__)


def create_parser():
    parser = build_parser(
        description=(
            'Construct an S_REGION polygon for the right-hand illuminated '
            'portion of a FITS image and plot a verification figure.'
        )
    )
    parser.add_argument('fits_file', type=Path, help='Input FITS file')
    parser.add_argument(
        '--hdu',
        type=int,
        default=None,
        help='Image HDU index (default: auto-detect)',
    )
    parser.add_argument(
        '--simplify',
        type=float,
        default=2.0,
        help='Polygon simplification tolerance in pixels (default: 2.0)',
    )
    parser.add_argument(
        '--adjacency',
        type=float,
        default=None,
        help=(
            'Include illuminated pixels within this many pixels of the primary '
            'region (default: max(80, 8%% of image width))'
        ),
    )
    parser.add_argument(
        '--bridge',
        type=float,
        default=None,
        help=(
            'Bridge disconnected ROI fragments by this many pixels when tracing '
            'the polygon (default: auto)'
        ),
    )
    parser.add_argument(
        '--dq-threshold',
        type=int,
        default=512,
        help='Maximum allowed DQ value for illuminated pixels (default: 512)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output plot path (default: <fits-stem>_illuminated_region.png)',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the plot interactively',
    )
    add_common_runtime(parser, ncores=False, plot=True, verbose=True)
    return parser


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)
    configure_logging_from_args(args, 'region')
    try:
        (
            s_region_polygon,
            region_mask,
            wcs,
            header,
            data,
            _,
        ) = illuminated_s_region_from_fits(
            args.fits_file,
            hdu_index=args.hdu,
            simplify_tolerance=args.simplify,
            adjacency_pixels=args.adjacency,
            bridge_pixels=args.bridge,
            dq_threshold=args.dq_threshold,
        )

        original_s_region = None
        if 'S_REGION' in header:
            original_s_region = SRegionPolygon.parse(header['S_REGION'])

        logger.info('%s', s_region_polygon.to_string())
        logger.info('Illuminated pixels: %d', int(region_mask.sum()))

        output_path = args.output
        if output_path is None:
            output_path = args.fits_file.with_name(
                f'{args.fits_file.stem}_illuminated_region.png'
            )

        save_illuminated_region_plot(
            data,
            s_region_polygon,
            wcs,
            output_path,
            original_s_region=original_s_region,
            title=args.fits_file.name,
        )
        logger.info('Wrote plot to %s', output_path)

        if args.show:
            plt.show()
        else:
            plt.close('all')
        return 0
    finally:
        shutdown_logging()


if __name__ == '__main__':
    raise SystemExit(main())
