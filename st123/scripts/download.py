#!/usr/bin/env python3
"""Download imaging products from MAST (HST and JWST)."""

from __future__ import annotations

import logging

from astropy import units as u

from st123.mast import normalize_filter_name, query_mast_hst, query_mast_jwst, resolve_outdir
from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    add_filters,
    add_mast_token,
    configure_logging_from_args,
    create_parser as build_parser,
    parse_filter_list,
    parse_instruments,
)
from st123.utils import parse_coord
from st123.utils.logging import shutdown_logging
from st123.utils.settings import DEFAULT_HST_INSTRUMENTS, DEFAULT_JWST_INSTRUMENTS

logger = logging.getLogger(__name__)


def create_parser():
    parser = build_parser(
        description=(
            'Download imaging from MAST (HST or JWST). Pass --token (or set '
            'MAST_API_TOKEN) to authenticate and include proprietary data. '
            'Canonical layout is telescope/instrument/filter/obsid '
            '(e.g. HST/WFC3/F814W/<obsid> or JWST/MIRI/F560W/<obsid>). '
            'Output directories are created automatically when missing. '
            'The object name is the basename of --base-dir.'
        ),
    )
    add_base_dir(
        parser,
        required=True,
        aliases=(
            '--basedir',
            '--workdir',
            '--data-dir',
            '--download-dir',
            '--outdir',
        ),
        help=(
            'Download root directory (object name = basename). '
            'Aliases: --basedir, --workdir, --data-dir, --download-dir, --outdir.'
        ),
    )
    parser.add_argument('--ra', type=str, required=True, help='RA of the target')
    parser.add_argument('--dec', type=str, required=True, help='DEC of the target')
    parser.add_argument(
        '--radius', type=float, default=3.0, help='Search radius in arcminutes'
    )
    parser.add_argument(
        '--telescope',
        choices=('jwst', 'hst'),
        default='jwst',
        help='Mission to download (default: jwst).',
    )
    parser.add_argument(
        '--stage', type=int, default=2, help='JWST calibration stage (2=CAL, 3=I2D)'
    )
    parser.add_argument(
        '--instruments',
        nargs='+',
        default=None,
        help=(
            'Instruments to include. Defaults: JWST→NIRCAM MIRI; '
            'HST→ACS WFC3 WFPC2. Space- or comma-separated.'
        ),
    )
    parser.add_argument(
        '--layout',
        choices=(
            'telescope/instrument/filter/obsid',
            'filter/obsid',
            'filter_obsid',
        ),
        default='telescope/instrument/filter/obsid',
        help='Per-observation directory layout under --base-dir.',
    )
    add_filters(parser)
    parser.add_argument(
        '--mirimage-only',
        action='store_true',
        help='Keep only MIRI imager (*mirimage*) products (JWST only).',
    )
    add_mast_token(parser)
    add_common_runtime(parser, ncores=False, plot=False, dry_run=True)
    return parser


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    configure_logging_from_args(args, 'download')
    try:
        coord = parse_coord(args.ra, args.dec)
        if coord is None:
            return 1

        try:
            outdir = resolve_outdir(args.base_dir)
        except (PermissionError, ValueError) as exc:
            logger.error('%s', exc)
            return 1

        telescope = str(args.telescope).lower()
        instruments = parse_instruments(args.instruments)
        if instruments is None:
            instruments = (
                list(DEFAULT_HST_INSTRUMENTS)
                if telescope == 'hst'
                else list(DEFAULT_JWST_INSTRUMENTS)
            )

        allowed = None
        if args.filters:
            allowed = [
                normalize_filter_name(f)
                for f in parse_filter_list(args.filters) or []
            ]

        try:
            if telescope == 'hst':
                # HST default radius 5' when user left JWST default of 3'
                radius_am = float(args.radius)
                if radius_am == 3.0:
                    radius_am = 5.0
                    logger.info(
                        'Using HST default search radius %.1f arcmin '
                        '(pass --radius to override)',
                        radius_am,
                    )
                n = query_mast_hst(
                    coord,
                    outdir=outdir,
                    radius=radius_am * u.arcmin,
                    token=args.token,
                    instruments=instruments,
                    layout=args.layout,
                    dry_run=args.dry_run,
                    allowed_filters=allowed,
                )
            else:
                mirimage_only = bool(args.mirimage_only)
                layout = args.layout
                if (
                    instruments is not None
                    and len(instruments) == 1
                    and instruments[0].upper() == 'MIRI'
                ):
                    if layout in (
                        'telescope/instrument/filter/obsid',
                        'filter/obsid',
                    ):
                        mirimage_only = True
                n = query_mast_jwst(
                    coord,
                    outdir=outdir,
                    radius=args.radius * u.arcmin,
                    stage=args.stage,
                    token=args.token,
                    instruments=instruments,
                    layout=layout,
                    mirimage_only=mirimage_only,
                    dry_run=args.dry_run,
                    allowed_filters=allowed,
                )
        except (RuntimeError, OSError) as exc:
            logger.error('%s', exc)
            return 1

        return 0 if n else 1
    finally:
        shutdown_logging()


if __name__ == '__main__':
    raise SystemExit(main())
