#!/usr/bin/env python3
"""Symlink FITS products into a reduction ``raw/`` directory."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    configure_logging_from_args,
    create_parser as build_parser,
    instrument_raw_dir,
    resolve_project_root,
    resolve_reduction_dir,
)
from st123.utils.link import create_symlink, remove_proc_files
from st123.utils.logging import shutdown_logging

logger = logging.getLogger(__name__)


def create_parser():
    parser = build_parser(
        description=(
            'Symlink HST/JWST FITS products into <base-dir>/reduction/raw. '
            'Prefer --base-dir + --telescope/--instrument; legacy '
            '--datadir/--symlinkdir remain available when source and destination '
            'are distinct.'
        ),
    )
    add_base_dir(
        parser,
        required=False,
        help=(
            'Project root containing JWST/ or HST/ (and reduction/). '
            'Symlinks are created under <base-dir>/reduction/raw unless '
            '--symlinkdir is given.'
        ),
    )
    parser.add_argument(
        '--telescope',
        choices=('JWST', 'HST', 'jwst', 'hst'),
        default=None,
        help='Telescope tree under --base-dir (JWST or HST). Inferred from --instrument when omitted.',
    )
    parser.add_argument(
        '--instrument',
        type=str,
        default='NIRCAM',
        help=(
            'Instrument subdirectory to link (default: NIRCAM → JWST/NIRCam). '
            'Use ALL with --telescope to link every instrument under that telescope.'
        ),
    )
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


def _source_dirs(args) -> list[Path]:
    """Resolve one or more source directories of FITS products."""
    if args.source_dir and args.symlink_dir and args.base_dir is None:
        return [Path(args.source_dir).expanduser()]

    if args.base_dir is None and not args.source_dir:
        raise ValueError(
            'provide --base-dir (with optional --telescope/--instrument) '
            'or both --datadir/--source-dir and --symlinkdir/--reduction-dir'
        )

    if args.source_dir:
        return [Path(args.source_dir).expanduser()]

    base = Path(args.base_dir).expanduser()
    tel = args.telescope.upper() if args.telescope else None
    inst = str(args.instrument).strip().upper()
    if inst == 'ALL':
        if tel is None:
            raise ValueError('--instrument ALL requires --telescope HST|JWST')
        root = resolve_project_root(base) / tel
        if not root.is_dir():
            raise ValueError(f'telescope tree missing: {root}')
        return sorted(p for p in root.iterdir() if p.is_dir())
    return [instrument_raw_dir(base, args.instrument, telescope=tel)]


def _resolve_reduction(args) -> Path:
    if args.symlink_dir:
        return Path(args.symlink_dir).expanduser()
    if args.base_dir is not None:
        return resolve_reduction_dir(Path(args.base_dir).expanduser())
    return Path('reduction').resolve()


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    configure_logging_from_args(args, 'link-raw')
    try:
        try:
            sources = _source_dirs(args)
            symlinkdir = _resolve_reduction(args)
        except ValueError as exc:
            logger.error('%s', exc)
            return 1

        raw_dir = os.path.join(str(symlinkdir), 'raw')
        os.makedirs(raw_dir, exist_ok=True)

        files: list[str] = []
        for datadir in sources:
            files.extend(
                glob.glob(os.path.join(str(datadir), '**', '*.fits'), recursive=True)
            )
        for procdir in args.proc_dirs:
            files = remove_proc_files(files, procdir)
        if args.verbose:
            for datadir in sources:
                logger.info('Source: %s', datadir)
            logger.info('Reduction: %s', symlinkdir)
        logger.info('Creating symlinks for %d files under %s', len(files), raw_dir)
        for file in files:
            link_path = os.path.join(raw_dir, os.path.basename(file))
            create_symlink(file, link_path)
        return 0
    finally:
        shutdown_logging()


if __name__ == '__main__':
    raise SystemExit(main())
