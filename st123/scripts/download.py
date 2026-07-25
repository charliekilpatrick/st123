#!/usr/bin/env python3
"""Download JWST imaging products from MAST."""

from __future__ import annotations

import argparse
import sys

from astropy import units as u

from st123.mast import normalize_filter_name, query_mast_jwst, resolve_outdir
from st123.utils import parse_coord


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Download JWST imaging from MAST. Pass --token (or set MAST_API_TOKEN) '
            'to authenticate and include proprietary data, as in hst123. '
            'For MIRI alignment_wrap layouts use '
            '``--instruments MIRI --layout filter/obsid`` '
            '(or the repo-root ``jwst_download.py`` shim).'
        ),
    )
    parser.add_argument('--ra', type=str, help='RA of the target', required=True)
    parser.add_argument('--dec', type=str, help='DEC of the target', required=True)
    parser.add_argument('--obj', type=str, help='Name of the object', required=True)
    parser.add_argument(
        '--outdir',
        default=None,
        type=str,
        help='Output directory for downloads (default: jwst_data/<obj>).',
    )
    parser.add_argument(
        '--download-dir',
        default=None,
        type=str,
        dest='download_dir',
        help='Alias for --outdir (matches jwst_RSGs jwst_download.py).',
    )
    parser.add_argument('--radius', type=float, default=3.0, help='Radius in arcminutes')
    parser.add_argument('--stage', type=int, default=2, help='Stage of the reduction')
    parser.add_argument(
        '--instruments',
        nargs='+',
        default=None,
        help='JWST instruments to include (default: NIRCAM MIRI).',
    )
    parser.add_argument(
        '--layout',
        choices=('filter_obsid', 'filter/obsid'),
        default='filter_obsid',
        help=(
            'Per-observation directory layout. '
            '``filter/obsid`` matches alignment_wrap discovery '
            '(default: filter_obsid legacy).'
        ),
    )
    parser.add_argument(
        '--filters',
        type=str,
        default=None,
        help='Optional comma-separated filter list (e.g. F560W,F770W).',
    )
    parser.add_argument(
        '--mirimage-only',
        action='store_true',
        help='Keep only MIRI imager (*mirimage*) products.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Query and list products without downloading.',
    )
    parser.add_argument(
        '--token',
        default=None,
        type=str,
        help=(
            'MAST authorization token for proprietary data '
            '(see https://auth.mast.stsci.edu/info). '
            'Also read from MAST_API_TOKEN or MAST_TOKEN if unset.'
        ),
    )
    return parser


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    coord = parse_coord(args.ra, args.dec)
    if coord is None:
        return 1

    outdir = resolve_outdir(args.obj, outdir=args.download_dir or args.outdir)
    instruments = args.instruments
    mirimage_only = bool(args.mirimage_only)
    layout = args.layout
    # Convenience: MIRI-only + filter/obsid implies imager products.
    if instruments is not None and len(instruments) == 1 and instruments[0].upper() == 'MIRI':
        if layout == 'filter/obsid':
            mirimage_only = True

    allowed = None
    if args.filters:
        allowed = [
            normalize_filter_name(f) for f in args.filters.split(',') if f.strip()
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
    except RuntimeError as exc:
        print(f'ERROR: {exc}')
        return 1

    return 0 if n else 1


def normalize_jwst_download_argv(argv: list[str] | None = None) -> list[str]:
    """Apply MIRI ``filter/obsid`` defaults used by ``scripts/jwst_download.py``."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if '--obj' not in args:
        args.extend(['--obj', 'target'])
    if '--instruments' not in args:
        args.extend(['--instruments', 'MIRI'])
    if '--layout' not in args:
        args.extend(['--layout', 'filter/obsid'])
    return args


def main_jwst_download(argv: list[str] | None = None) -> int:
    """Entry point for the legacy ``jwst_download`` / ``jwst-download`` CLI names."""
    return main(normalize_jwst_download_argv(argv))


if __name__ == '__main__':
    raise SystemExit(main())

