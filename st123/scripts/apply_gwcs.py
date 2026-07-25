#!/usr/bin/env python3
"""Attach a GWCS object to coadd ``*_i2d.fits`` datamodels."""

from __future__ import annotations

import argparse
import glob
import sys

from st123.mosaic.mosaic import apply_wcs_to_coadd


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Apply GWCS from mosaic helpers to coadd i2d products.',
    )
    parser.add_argument(
        'coadds',
        nargs='*',
        help='Coadd FITS files (default: group_*/ref_*/coadd*i2d.fits).',
    )
    parser.add_argument(
        '--glob',
        dest='pattern',
        default='group_*/ref_*/coadd*i2d.fits',
        help='Glob used when no coadd paths are given.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    coadds = list(args.coadds) if args.coadds else glob.glob(args.pattern)
    if not coadds:
        print(f'ERROR: no coadds found (pattern={args.pattern!r}).', file=sys.stderr)
        return 1
    for path in coadds:
        out = apply_wcs_to_coadd(path)
        print(f'{path} -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
