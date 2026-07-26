#!/usr/bin/env python3
"""Download imaging products from MAST (JWST today; HST / Roman next)."""

from __future__ import annotations

from astropy import units as u

from st123.mast import normalize_filter_name, query_mast_jwst, resolve_outdir
from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    add_filters,
    add_mast_token,
    create_parser as build_parser,
    parse_filter_list,
    parse_instruments,
)
from st123.utils import parse_coord


def create_parser():
    parser = build_parser(
        description=(
            'Download imaging from MAST. Pass --token (or set MAST_API_TOKEN) '
            'to authenticate and include proprietary data. '
            'Canonical layout is telescope/instrument/filter/obsid '
            '(e.g. JWST/MIRI/F560W/<obsid>). '
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
        '--stage', type=int, default=2, help='JWST calibration stage (2=CAL, 3=I2D)'
    )
    parser.add_argument(
        '--instruments',
        nargs='+',
        default=None,
        help=(
            'Instruments to include (default: NIRCAM MIRI). '
            'Space- or comma-separated, e.g. NIRCAM,MIRI.'
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
        help='Keep only MIRI imager (*mirimage*) products.',
    )
    add_mast_token(parser)
    add_common_runtime(parser, ncores=False, plot=False, dry_run=True)
    return parser


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    coord = parse_coord(args.ra, args.dec)
    if coord is None:
        return 1

    try:
        outdir = resolve_outdir(args.base_dir)
    except (PermissionError, ValueError) as exc:
        print(f'ERROR: {exc}')
        return 1

    instruments = parse_instruments(args.instruments)
    mirimage_only = bool(args.mirimage_only)
    layout = args.layout
    if instruments is not None and len(instruments) == 1 and instruments[0].upper() == 'MIRI':
        if layout in (
            'telescope/instrument/filter/obsid',
            'filter/obsid',
        ):
            mirimage_only = True

    allowed = None
    if args.filters:
        allowed = [
            normalize_filter_name(f) for f in parse_filter_list(args.filters) or []
        ]

    try:
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
        print(f'ERROR: {exc}')
        return 1

    return 0 if n else 1


if __name__ == '__main__':
    raise SystemExit(main())
