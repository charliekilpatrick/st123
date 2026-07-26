#!/usr/bin/env python3
"""Symlink FITS products into a reduction ``raw/`` directory."""

from __future__ import annotations

import glob
import os
from pathlib import Path

from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    create_parser as build_parser,
    instrument_raw_dir,
    resolve_reduction_dir,
)
from st123.utils.link import create_symlink, remove_proc_files


def create_parser():
    parser = build_parser(
        description=(
            'Symlink JWST FITS products into <base-dir>/reduction/raw. '
            'Prefer --base-dir + --instrument; legacy --datadir/--symlinkdir '
            'remain available when source and destination are distinct.'
        ),
    )
    add_base_dir(
        parser,
        required=False,
        help=(
            'Project root containing JWST/ (and reduction/). '
            'Symlinks are created under <base-dir>/reduction/raw unless '
            '--symlinkdir is given.'
        ),
    )
    parser.add_argument(
        '--instrument',
        type=str,
        default='NIRCAM',
        help=(
            'Instrument subdirectory under JWST/ to link '
            '(default: NIRCAM → JWST/NIRCam).'
        ),
    )
    # Distinct second directory when callers need an explicit source/dest pair.
    parser.add_argument(
        '--datadir',
        '--source-dir',
        dest='source_dir',
        type=str,
        default=None,
        help=(
            'Explicit source directory of FITS files (distinct from --base-dir). '
            'Aliases: --datadir, --source-dir.'
        ),
    )
    parser.add_argument(
        '--symlinkdir',
        '--reduction-dir',
        dest='symlink_dir',
        type=str,
        default=None,
        help=(
            'Explicit reduction directory that will contain raw/ '
            '(aliases: --symlinkdir, --reduction-dir).'
        ),
    )
    parser.add_argument(
        '--proc_dirs',
        nargs='*',
        type=str,
        default=[],
        help='Directories that already contain processed files to skip.',
    )
    add_common_runtime(parser, ncores=False, plot=False)
    return parser


def _resolve_link_paths(args) -> tuple[Path, Path]:
    """Return (source_dir, reduction_dir)."""
    if args.source_dir and args.symlink_dir:
        return Path(args.source_dir).expanduser(), Path(args.symlink_dir).expanduser()

    if args.base_dir is None and not args.source_dir:
        raise SystemExit(
            'ERROR: provide --base-dir (with optional --instrument) '
            'or both --datadir/--source-dir and --symlinkdir/--reduction-dir'
        )

    if args.base_dir is not None:
        base = Path(args.base_dir).expanduser()
        source = (
            Path(args.source_dir).expanduser()
            if args.source_dir
            else instrument_raw_dir(base, args.instrument)
        )
        dest = (
            Path(args.symlink_dir).expanduser()
            if args.symlink_dir
            else resolve_reduction_dir(base)
        )
        return source, dest

    # source_dir alone: symlink into cwd/reduction
    source = Path(args.source_dir).expanduser()
    dest = (
        Path(args.symlink_dir).expanduser()
        if args.symlink_dir
        else Path('reduction').resolve()
    )
    return source, dest


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        datadir, symlinkdir = _resolve_link_paths(args)
    except SystemExit as exc:
        print(exc)
        return 1

    raw_dir = os.path.join(str(symlinkdir), 'raw')
    os.makedirs(raw_dir, exist_ok=True)

    files = glob.glob(os.path.join(str(datadir), '**', '*.fits'), recursive=True)
    for procdir in args.proc_dirs:
        files = remove_proc_files(files, procdir)
    if args.verbose:
        print(f'Source: {datadir}')
        print(f'Reduction: {symlinkdir}')
    print(f'Creating symlinks for {len(files)} files under {raw_dir}')
    for file in files:
        link_path = os.path.join(raw_dir, os.path.basename(file))
        create_symlink(file, link_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
