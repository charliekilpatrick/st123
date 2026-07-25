#!/usr/bin/env python3
"""Symlink FITS products into a reduction ``raw/`` directory."""

from __future__ import annotations

import argparse
import glob
import os

from st123.utils.link import create_symlink, remove_proc_files


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Symlink data for a st123 run')
    parser.add_argument(
        '--datadir',
        type=str,
        help='Full path to directory with object data',
        required=True,
    )
    parser.add_argument(
        '--symlinkdir',
        type=str,
        help='Directory containing the raw/ folder where symlinks will be created',
        required=True,
    )
    parser.add_argument(
        '--proc_dirs',
        nargs='*',
        type=str,
        help='Directories that already contain processed files to skip',
        default=[],
        required=False,
    )
    return parser


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    files = glob.glob(os.path.join(args.datadir, '**', '*.fits'), recursive=True)
    for procdir in args.proc_dirs:
        files = remove_proc_files(files, procdir)
    print(f'Creating symlinks for {len(files)} files')
    for file in files:
        link_path = os.path.join(args.symlinkdir, 'raw', os.path.basename(file))
        create_symlink(file, link_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
