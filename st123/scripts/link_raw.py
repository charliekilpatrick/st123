#!/usr/bin/env python3
"""Symlink FITS products into a reduction ``raw/`` directory."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Optional, Sequence, Union

from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    configure_logging_from_args,
    create_parser as build_parser,
    ensure_writable_dir,
    instrument_raw_dir,
    resolve_project_root,
    resolve_reduction_dir,
)
from st123.utils.link import create_symlink, remove_proc_files
from st123.utils.logging import shutdown_logging
from st123.utils.settings import DOWNLOAD_DIR_NAME

PathLike = Union[str, os.PathLike]

logger = logging.getLogger(__name__)


def create_parser():
    parser = build_parser(
        description=(
            'Symlink HST/JWST FITS products into <base-dir>/reduction/raw. '
            'Prefer --base-dir + --telescope/--instrument; legacy '
            '--datadir/--symlinkdir remain available when source and destination '
            'are distinct. ``download`` runs this automatically after a '
            'successful MAST fetch.'
        ),
    )
    add_base_dir(
        parser,
        required=False,
        help=(
            'Project root containing download/, JWST/, or HST/ (and '
            'reduction/). Symlinks are created under '
            '<base-dir>/reduction/raw unless --symlinkdir is given.'
        ),
    )
    parser.add_argument(
        '--telescope',
        choices=('JWST', 'HST', 'jwst', 'hst'),
        default=None,
        help=(
            'Telescope tree under <base-dir>/download/ (or legacy '
            '<base-dir>/). Inferred from --instrument when omitted.'
        ),
    )
    parser.add_argument(
        '--instrument',
        type=str,
        default='NIRCAM',
        help=(
            'Instrument subdirectory to link (default: NIRCAM → '
            'download/JWST/NIRCam). Use ALL with --telescope to link every '
            'instrument under that telescope.'
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
    add_common_runtime(parser, plot=False)
    return parser


def resolve_link_source_dirs(
    *,
    base_dir: Optional[PathLike] = None,
    telescope: Optional[str] = None,
    instrument: str = 'ALL',
    source_dir: Optional[PathLike] = None,
) -> list[Path]:
    """Resolve one or more source directories of FITS products for linking."""
    if source_dir is not None and base_dir is None:
        return [Path(source_dir).expanduser()]

    if base_dir is None and source_dir is None:
        raise ValueError(
            'provide base_dir (with optional telescope/instrument) '
            'or an explicit source_dir'
        )

    if source_dir is not None:
        return [Path(source_dir).expanduser()]

    base = Path(base_dir).expanduser()
    tel = telescope.upper() if telescope else None
    inst = str(instrument).strip().upper()
    if inst == 'ALL':
        if tel is None:
            raise ValueError('instrument ALL requires telescope HST|JWST')
        project = resolve_project_root(base)
        preferred = project / DOWNLOAD_DIR_NAME / tel
        legacy = project / tel
        if preferred.is_dir():
            root = preferred
        elif legacy.is_dir():
            root = legacy
        else:
            raise ValueError(
                f'telescope tree missing: {preferred} (or legacy {legacy})'
            )
        return sorted(p for p in root.iterdir() if p.is_dir())
    return [instrument_raw_dir(base, instrument, telescope=tel)]


def link_raw_tree(
    *,
    base_dir: Optional[PathLike] = None,
    telescope: Optional[str] = None,
    instrument: str = 'ALL',
    source_dir: Optional[PathLike] = None,
    symlink_dir: Optional[PathLike] = None,
    proc_dirs: Optional[Sequence[PathLike]] = None,
    verbose: bool = False,
) -> int:
    """
    Symlink FITS under the telescope/instrument tree into ``reduction/raw``.

    Parameters
    ----------
    base_dir : path-like or None
        Project root (contains ``download/``, ``HST/``, or ``JWST/``).
    telescope : str or None
        ``HST`` or ``JWST`` (case-insensitive). Required when *instrument*
        is ``ALL``.
    instrument : str, optional
        Instrument name, or ``ALL`` to link every instrument under *telescope*.
    source_dir : path-like or None, optional
        Explicit FITS source directory (overrides base/telescope/instrument).
    symlink_dir : path-like or None, optional
        Reduction directory that will contain ``raw/``; default
        ``<base-dir>/reduction``.
    proc_dirs : sequence of path-like or None, optional
        Directories whose existing ``raw/`` products should be skipped.
    verbose : bool, optional
        Log source and reduction paths.

    Returns
    -------
    int
        Number of FITS paths considered for linking.
    """
    sources = resolve_link_source_dirs(
        base_dir=base_dir,
        telescope=telescope,
        instrument=instrument,
        source_dir=source_dir,
    )
    if symlink_dir is not None:
        reduction = Path(symlink_dir).expanduser()
    elif base_dir is not None:
        reduction = resolve_reduction_dir(Path(base_dir).expanduser())
    else:
        reduction = Path('reduction').resolve()

    raw_dir = str(ensure_writable_dir(Path(reduction) / 'raw', label='reduction/raw'))

    files: list[str] = []
    for datadir in sources:
        files.extend(
            glob.glob(os.path.join(str(datadir), '**', '*.fits'), recursive=True)
        )
    for procdir in proc_dirs or []:
        files = remove_proc_files(files, str(procdir))
    if verbose:
        for datadir in sources:
            logger.info('Source: %s', datadir)
        logger.info('Reduction: %s', reduction)
    logger.info('Creating symlinks for %d files under %s', len(files), raw_dir)
    for file in files:
        link_path = os.path.join(raw_dir, os.path.basename(file))
        create_symlink(file, link_path)
    return len(files)


def _source_dirs(args) -> list[Path]:
    """Resolve one or more source directories of FITS products (CLI helper)."""
    if args.source_dir and args.symlink_dir and args.base_dir is None:
        return [Path(args.source_dir).expanduser()]
    return resolve_link_source_dirs(
        base_dir=args.base_dir,
        telescope=args.telescope,
        instrument=args.instrument,
        source_dir=args.source_dir,
    )


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
            link_raw_tree(
                base_dir=args.base_dir,
                telescope=args.telescope,
                instrument=args.instrument,
                source_dir=args.source_dir,
                symlink_dir=args.symlink_dir,
                proc_dirs=args.proc_dirs,
                verbose=args.verbose,
            )
        except (ValueError, PermissionError) as exc:
            logger.error('%s', exc)
            return 1
        return 0
    finally:
        shutdown_logging()


if __name__ == '__main__':
    raise SystemExit(main())
