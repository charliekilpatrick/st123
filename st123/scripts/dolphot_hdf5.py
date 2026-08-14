#!/usr/bin/env python3
"""Parse finished DOLPHOT catalogs into compressed HDF5 files (one per reference)."""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    add_instruments_arg,
    configure_logging_from_args,
    create_parser as build_parser,
    dataset_label,
    resolve_photometry_instrument,
    resolve_project_root,
    resolve_reduction_dir,
)
from st123.utils.logging import shutdown_logging

logger = logging.getLogger(__name__)

_PHOT_DIR_RE = re.compile(
    r'^(?:phot|nircam|miri|acs|wfc3|wfpc2|hst|nircam_miri|nircam_hst)_(\d+)_(.+)$',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DolphotHdf5Job:
    """One finished DOLPHOT catalog eligible for HDF5 export."""

    outdir: Path
    phot_out: str

    @property
    def label(self) -> str:
        return self.outdir.name

    @property
    def phot_path(self) -> Path:
        return self.outdir / self.phot_out


def _instrument_matches(dirname: str, instrument: str | None) -> bool:
    if instrument is None or instrument.lower() in ('all', '*'):
        return True
    name = dirname.lower()
    inst = instrument.lower()
    if inst == 'nircam':
        return name.startswith('phot_') or (
            name.startswith('nircam_')
            and not name.startswith('nircam_miri')
            and not name.startswith('nircam_hst')
        )
    if inst == 'miri':
        return name.startswith('miri_') or name.startswith('nircam_miri')
    if inst == 'hst':
        return (
            name.startswith('nircam_hst')
            or name.startswith('hst_')
            or name.startswith('acs_')
            or name.startswith('wfc3_')
            or name.startswith('wfpc2_')
        )
    if inst in ('acs', 'wfc3', 'wfpc2'):
        return name.startswith(f'{inst}_') or name.startswith('nircam_hst')
    return name.startswith(f'{inst}_')


def discover_dolphot_hdf5_jobs(
    base_dir: str | Path,
    *,
    instrument: str | None = None,
    group: int | None = None,
    box: int | str | None = None,
) -> list[DolphotHdf5Job]:
    """
    Find finished DOLPHOT catalogs under a project (``*.phot`` + ``*.columns``).

    Searches ``reduction/`` and ``dolphot/``. Each matching directory is treated
    as one reference / mosaic box (one HDF5 sidecar).
    """
    project = Path(resolve_project_root(base_dir))
    reduction = Path(resolve_reduction_dir(base_dir))
    roots = [reduction, project / 'dolphot']
    jobs: list[DolphotHdf5Job] = []
    seen: set[Path] = set()

    for root in roots:
        if not root.is_dir():
            continue
        for outdir in sorted(p for p in root.iterdir() if p.is_dir()):
            if not _instrument_matches(outdir.name, instrument):
                continue
            match = _PHOT_DIR_RE.match(outdir.name)
            if match is not None:
                g = int(match.group(1))
                b = match.group(2)
                try:
                    b_tok: int | str = int(b)
                except ValueError:
                    b_tok = b
                if group is not None and g != int(group):
                    continue
                if box is not None and str(b_tok) != str(box):
                    continue
            phot_out = f'{outdir.name}.phot'
            phot = outdir / phot_out
            columns = Path(str(phot) + '.columns')
            if not phot.is_file() or not columns.is_file():
                continue
            key = outdir.resolve()
            if key in seen:
                continue
            seen.add(key)
            jobs.append(DolphotHdf5Job(outdir=outdir, phot_out=phot_out))
    return jobs


def create_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        description=(
            'Parse finished DOLPHOT photometry catalogs into compressed HDF5 '
            'files (one per reference / mosaic box), matching the hst123 '
            'catalog layout. Skips directories that already have a .h5 sidecar '
            'unless --force is set.'
        ),
    )
    add_base_dir(
        parser,
        required=True,
        help=(
            'Project root or reduction workdir. Discovers finished catalogs '
            'under reduction/ and dolphot/.'
        ),
    )
    add_instruments_arg(
        parser,
        default=['all'],
        help=(
            'Limit which finished runs to convert. Mission aliases: hst, jwst '
            '(→ nircam), all (default), or explicit nircam|miri|acs|wfc3|wfpc2. '
            'Alias: --instrument.'
        ),
    )
    parser.add_argument(
        '--group',
        type=int,
        default=None,
        help='Only catalogs for this mosaic group index.',
    )
    parser.add_argument(
        '--box',
        type=str,
        default=None,
        help='Only catalogs for this mosaic box index (int or label).',
    )
    parser.add_argument(
        '--dir',
        dest='dirs',
        nargs='+',
        default=None,
        help='Explicit DOLPHOT working directories (skip auto-discovery).',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing .h5 files.',
    )
    parser.add_argument(
        '--no-compression',
        action='store_true',
        help='Write HDF5 without gzip compression (faster, larger).',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List catalogs that would be converted without writing HDF5.',
    )
    add_common_runtime(parser, plot=False, verbose=True)
    return parser


def _jobs_from_args(args: argparse.Namespace) -> list[DolphotHdf5Job]:
    if args.dirs:
        jobs: list[DolphotHdf5Job] = []
        for raw in args.dirs:
            outdir = Path(raw).expanduser().resolve()
            phot_out = f'{outdir.name}.phot'
            phot = outdir / phot_out
            columns = Path(str(phot) + '.columns')
            if not phot.is_file() or not columns.is_file():
                raise FileNotFoundError(
                    f'missing {phot_out} / {phot_out}.columns under {outdir}'
                )
            jobs.append(DolphotHdf5Job(outdir=outdir, phot_out=phot_out))
        return jobs

    box = None if args.box is None else args.box
    raw = [str(t) for t in (args.instruments or ['all'])]
    keys = {t.strip().lower() for t in raw if str(t).strip()}

    # Default / explicit "all": every finished catalog under the project.
    if not keys or keys <= {'all', '*', 'both'}:
        return discover_dolphot_hdf5_jobs(
            args.base_dir,
            instrument=None,
            group=args.group,
            box=box,
        )

    modes: list[str] = []
    if keys == {'jwst'}:
        modes = ['nircam', 'miri']
    elif keys == {'hst'}:
        modes = ['hst']
    else:
        try:
            modes = [resolve_photometry_instrument(raw)]
        except ValueError:
            # Multiple concrete cameras: discover each mode separately.
            modes = []
            for token in raw:
                modes.append(resolve_photometry_instrument([token]))

    jobs: list[DolphotHdf5Job] = []
    seen: set[Path] = set()
    for mode in modes:
        for job in discover_dolphot_hdf5_jobs(
            args.base_dir,
            instrument=mode,
            group=args.group,
            box=box,
        ):
            key = job.outdir.resolve()
            if key in seen:
                continue
            seen.add(key)
            jobs.append(job)
    return jobs


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    configure_logging_from_args(args, 'dolphot-hdf5')
    try:
        from st123.photometry.dolphot_catalog_hdf5 import (
            ensure_dolphot_catalog_hdf5,
            hdf5_path_for_phot_base,
        )

        try:
            jobs = _jobs_from_args(args)
        except (FileNotFoundError, ValueError) as exc:
            logger.error('%s', exc)
            return 1

        if args.verbose:
            logger.info('Dataset: %s', dataset_label(args.base_dir))

        if not jobs:
            logger.error(
                'No finished DOLPHOT catalogs found under %s '
                '(need <run>/<run>.phot and .columns)',
                args.base_dir,
            )
            return 1

        logger.info('Found %d DOLPHOT catalog(s)', len(jobs))
        compression = not bool(args.no_compression)
        n_wrote = 0
        n_skip = 0
        n_fail = 0
        for job in jobs:
            h5 = hdf5_path_for_phot_base(job.phot_path)
            exists = h5.is_file()
            if args.dry_run:
                action = (
                    'overwrite'
                    if exists and args.force
                    else ('skip (exists)' if exists else 'write')
                )
                logger.info('  [%s] %s → %s', job.label, action, h5.name)
                continue
            if exists and not args.force:
                logger.info('  [%s] skip existing %s', job.label, h5.name)
                n_skip += 1
                continue
            try:
                out = ensure_dolphot_catalog_hdf5(
                    job.outdir,
                    phot_out=job.phot_out,
                    force=bool(args.force),
                    compression=compression,
                )
            except Exception as exc:
                logger.error('  [%s] HDF5 failed: %s', job.label, exc)
                n_fail += 1
                continue
            if out is None:
                logger.error(
                    '  [%s] incomplete catalog products under %s',
                    job.label,
                    job.outdir,
                )
                n_fail += 1
                continue
            logger.info('  [%s] wrote %s', job.label, out)
            n_wrote += 1

        if args.dry_run:
            logger.info('Dry run: %d catalog(s) listed', len(jobs))
            return 0

        logger.info(
            'DOLPHOT HDF5 done: %d wrote, %d skipped, %d failed',
            n_wrote,
            n_skip,
            n_fail,
        )
        return 0 if n_fail == 0 else 1
    finally:
        shutdown_logging()


if __name__ == '__main__':
    raise SystemExit(main())
