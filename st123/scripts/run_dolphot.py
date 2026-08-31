#!/usr/bin/env python3
"""Run prepared DOLPHOT photometry directories (MaxThreads per run)."""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    add_instruments_arg,
    add_sky_coord_args,
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
class DolphotRun:
    """One prepared DOLPHOT working directory."""

    outdir: Path
    phot_out: str
    param_file: str = 'dolphot.param'
    group: int | None = None
    box: int | str | None = None

    @property
    def label(self) -> str:
        return self.outdir.name


def _parse_box_token(token: str) -> int | str:
    try:
        return int(token)
    except ValueError:
        return token


def _group_box_from_dirname(name: str) -> tuple[int | None, int | str | None]:
    match = _PHOT_DIR_RE.match(name)
    if not match:
        return None, None
    return int(match.group(1)), _parse_box_token(match.group(2))


def discover_dolphot_runs(
    base_dir: str | Path,
    *,
    instrument: str = 'nircam',
    group: int | None = None,
    box: int | str | None = None,
) -> list[DolphotRun]:
    """
    Find prepared DOLPHOT directories under a project (require ``dolphot.param``).

    NIRCam looks in ``reduction/phot_*`` and ``dolphot/nircam_*`` (excluding
    warm-start ``nircam_miri_*`` / ``nircam_hst_*``). MIRI uses ``dolphot/miri_*``
    and ``dolphot/nircam_miri_*``. HST uses ``dolphot/{acs,wfc3,wfpc2,hst,nircam_hst}_*``.
    """
    project = Path(resolve_project_root(base_dir))
    reduction = Path(resolve_reduction_dir(base_dir))
    dolphot_root = project / 'dolphot'
    inst = instrument.lower()
    candidates: list[Path] = []

    if inst == 'nircam':
        candidates.extend(sorted(reduction.glob('phot_*')))
        if dolphot_root.is_dir():
            for path in sorted(dolphot_root.glob('nircam_*')):
                name = path.name.lower()
                if name.startswith('nircam_miri') or name.startswith('nircam_hst'):
                    continue
                candidates.append(path)
    elif inst == 'miri':
        if dolphot_root.is_dir():
            candidates.extend(sorted(dolphot_root.glob('miri_*')))
            candidates.extend(sorted(dolphot_root.glob('nircam_miri_*')))
    elif inst in ('hst', 'acs', 'wfc3', 'wfpc2'):
        # Warmstart-prep writes dolphot/nircam_hst_G_B; also pick up
        # camera-specific dolphot/{acs,wfc3,...}_* free runs when present.
        if dolphot_root.is_dir():
            candidates.extend(sorted(dolphot_root.glob('nircam_hst_*')))
            if inst == 'hst':
                for prefix in ('hst_', 'acs_', 'wfc3_', 'wfpc2_'):
                    candidates.extend(sorted(dolphot_root.glob(f'{prefix}*')))
            else:
                candidates.extend(sorted(dolphot_root.glob(f'{inst}_*')))
    else:
        if dolphot_root.is_dir():
            candidates.extend(sorted(dolphot_root.glob(f'{inst}_*')))

    runs: list[DolphotRun] = []
    seen: set[Path] = set()
    for outdir in candidates:
        if not outdir.is_dir():
            continue
        key = outdir.resolve()
        if key in seen:
            continue
        param = outdir / 'dolphot.param'
        if not param.is_file():
            continue
        g, b = _group_box_from_dirname(outdir.name)
        if group is not None and g is not None and int(g) != int(group):
            continue
        if box is not None and b is not None and str(b) != str(box):
            continue
        seen.add(key)
        runs.append(
            DolphotRun(
                outdir=outdir,
                phot_out=f'{outdir.name}.phot',
                param_file='dolphot.param',
                group=g,
                box=b,
            )
        )
    return runs


def resolve_dolphot_executable(dolphot_bin: str | Path | None = None) -> str:
    """Return path to the ``dolphot`` binary (or bare name on PATH)."""
    from st123.stages.photometry.dolphot import resolve_dolphot_bin

    bin_dir = resolve_dolphot_bin(dolphot_bin, required=False)
    if bin_dir is not None:
        exe = Path(bin_dir) / 'dolphot'
        if exe.is_file() and os.access(exe, os.X_OK):
            return str(exe)
    found = shutil.which('dolphot')
    return found or 'dolphot'


def build_dolphot_argv(
    run: DolphotRun,
    *,
    ncores: int,
    dolphot_bin: str | Path | None = None,
) -> list[str]:
    """Argv for one DOLPHOT invocation (cwd = ``run.outdir``)."""
    exe = resolve_dolphot_executable(dolphot_bin)
    return [
        exe,
        run.phot_out,
        f'-p{run.param_file}',
        f'MaxThreads={max(1, int(ncores))}',
    ]


def create_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        description=(
            'Run prepared DOLPHOT photometry directories. Discovers dirs with '
            'dolphot.param (e.g. reduction/phot_* for NIRCam) and executes '
            'dolphot <name>.phot -pdolphot.param MaxThreads=<ncores> in each. '
            'Use --parallel to run several at once; --wait (default) blocks '
            'until all finish; --background starts them and exits when all '
            'can be launched immediately (n_runs <= --parallel). After a '
            'successful wait-mode run, writes a compressed <run>.h5 catalog '
            'sidecar by default (same layout as hst123); use '
            '--no-write-dolphot-hdf5 to skip, or dolphot-hdf5 later.'
        ),
    )
    add_base_dir(
        parser,
        required=True,
        help=(
            'Project root or reduction workdir. Discovers prepared phot dirs '
            'under reduction/ and dolphot/.'
        ),
    )
    add_instruments_arg(
        parser,
        default=['nircam'],
        help=(
            'Which prepared runs to execute. Mission aliases: hst (mixed HST '
            'dirs), jwst (-> nircam), or explicit nircam|miri|acs|wfc3|wfpc2. '
            'Alias: --instrument.'
        ),
    )
    add_sky_coord_args(parser, required=False)
    parser.add_argument(
        '--group',
        type=int,
        default=None,
        help='Only runs for this mosaic group index.',
    )
    parser.add_argument(
        '--box',
        type=str,
        default=None,
        help='Only runs for this mosaic box index (int or label).',
    )
    parser.add_argument(
        '--dir',
        dest='dirs',
        nargs='+',
        default=None,
        help='Explicit DOLPHOT working directories (skip auto-discovery).',
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help=(
            'Maximum number of concurrent dolphot processes (each still uses '
            'MaxThreads=--ncores). Default: 1.'
        ),
    )
    parser.add_argument(
        '--wait',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Wait for all dolphot runs to finish (default). Use --no-wait / '
            '--background to start jobs and exit when n_runs <= --parallel.'
        ),
    )
    parser.add_argument(
        '--background',
        action='store_true',
        help='Alias for --no-wait (fire-and-forget when n_runs <= --parallel).',
    )
    parser.add_argument(
        '--dolphot-bin',
        type=str,
        default=None,
        help='DOLPHOT bin directory (default: directory of dolphot on PATH).',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List discovered runs and commands without executing dolphot.',
    )
    parser.set_defaults(write_dolphot_hdf5=True)
    parser.add_argument(
        '--no-write-dolphot-hdf5',
        dest='write_dolphot_hdf5',
        action='store_false',
        help=(
            'Skip writing compressed <run>.h5 catalog sidecars after '
            'successful wait-mode dolphot runs (default: write when h5py is '
            'available).'
        ),
    )
    parser.add_argument(
        '--write-dolphot-hdf5',
        dest='write_dolphot_hdf5',
        action='store_true',
        help='Write <run>.h5 after successful wait-mode runs (default).',
    )
    parser.add_argument(
        '--force-dolphot-hdf5',
        action='store_true',
        help='Overwrite existing <run>.h5 sidecars when writing HDF5.',
    )
    add_common_runtime(parser, ncores=True, ncores_default=1, plot=False, verbose=True)
    return parser


def _runs_from_args(args: argparse.Namespace) -> list[DolphotRun]:
    if args.dirs:
        runs: list[DolphotRun] = []
        for raw in args.dirs:
            outdir = Path(raw).expanduser().resolve()
            if not (outdir / 'dolphot.param').is_file():
                raise FileNotFoundError(
                    f'missing dolphot.param under explicit --dir {outdir}'
                )
            g, b = _group_box_from_dirname(outdir.name)
            runs.append(
                DolphotRun(
                    outdir=outdir,
                    phot_out=f'{outdir.name}.phot',
                    group=g,
                    box=b,
                )
            )
        return runs

    box = None if args.box is None else _parse_box_token(str(args.box))
    instrument = getattr(args, 'instrument', None) or resolve_photometry_instrument(
        args.instruments
    )
    return discover_dolphot_runs(
        args.base_dir,
        instrument=instrument,
        group=args.group,
        box=box,
    )


def _write_run_hdf5(
    run: DolphotRun,
    *,
    force: bool = False,
) -> Path | None:
    """Write compressed HDF5 for one finished run; return path or None."""
    from st123.stages.photometry.dolphot_catalog_hdf5 import ensure_dolphot_catalog_hdf5

    return ensure_dolphot_catalog_hdf5(
        run.outdir,
        phot_out=run.phot_out,
        force=force,
        compression=True,
    )


def _assert_run_hst_quality(run: DolphotRun) -> None:
    """
    Refuse to launch when staged HST science frames fail the EXPFLAG gate.

    JWST-only dirs (no HST science suffixes) pass through. Call after prep so
    remosaic / re-prep is the recovery path.
    """
    from st123.datamodels.hst import (
        filter_good_hst_frames,
        is_hst_science_path,
        log_rejected_hst_frames,
    )
    from st123.stages.photometry.dolphot import parse_param_image_list

    param = run.outdir / run.param_file
    if not param.is_file():
        return
    try:
        _ref, bases = parse_param_image_list(param)
    except Exception:
        return
    paths: list[Path] = []
    for base in bases:
        for cand in (
            run.outdir / f'{base}.fits',
            run.outdir / base,
        ):
            if cand.is_file() and is_hst_science_path(cand):
                paths.append(cand)
                break
    if not paths:
        return
    _kept, rejected = filter_good_hst_frames(paths)
    if rejected:
        log_rejected_hst_frames(rejected, stage=f'run-dolphot:{run.label}')
        raise RuntimeError(
            f'{run.label}: {len(rejected)} staged HST frame(s) fail EXPFLAG; '
            f're-run dolphot-prep after filtering raw/jhat'
        )


def _format_command(run: DolphotRun, *, ncores: int, dolphot_bin: str | None) -> str:
    argv = build_dolphot_argv(run, ncores=ncores, dolphot_bin=dolphot_bin)
    return f'cd {run.outdir} && {" ".join(argv)}'


def _run_one_wait(
    run: DolphotRun,
    *,
    ncores: int,
    dolphot_bin: str | None,
) -> tuple[DolphotRun, int]:
    argv = build_dolphot_argv(run, ncores=ncores, dolphot_bin=dolphot_bin)
    logger.info('Starting %s: %s', run.label, ' '.join(argv))
    t0 = time.perf_counter()
    proc = subprocess.run(
        argv,
        cwd=str(run.outdir),
        check=False,
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        'Finished %s rc=%d (%.1fs)',
        run.label,
        proc.returncode,
        elapsed,
    )
    return run, int(proc.returncode)


def _start_background(
    run: DolphotRun,
    *,
    ncores: int,
    dolphot_bin: str | None,
) -> subprocess.Popen:
    argv = build_dolphot_argv(run, ncores=ncores, dolphot_bin=dolphot_bin)
    out_log = run.outdir / 'dolphot.out'
    err_log = run.outdir / 'dolphot.err'
    logger.info(
        'Background %s: %s > %s 2> %s',
        run.label,
        ' '.join(argv),
        out_log.name,
        err_log.name,
    )
    # Detach so run-dolphot can exit while dolphot continues.
    stdout = open(out_log, 'w', encoding='utf-8')
    stderr = open(err_log, 'w', encoding='utf-8')
    return subprocess.Popen(
        argv,
        cwd=str(run.outdir),
        stdout=stdout,
        stderr=stderr,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    configure_logging_from_args(args, 'run-dolphot')
    try:
        try:
            args.instrument = resolve_photometry_instrument(args.instruments)
        except ValueError as exc:
            logger.error('%s', exc)
            return 2
        wait = bool(args.wait) and not bool(args.background)
        parallel = max(1, int(args.parallel))
        ncores = max(1, int(args.ncores))

        try:
            runs = _runs_from_args(args)
        except FileNotFoundError as exc:
            logger.error('%s', exc)
            return 1

        if args.verbose:
            logger.info('Dataset: %s', dataset_label(args.base_dir))
            logger.info(
                'Instrument=%s parallel=%d MaxThreads=%d mode=%s',
                args.instrument,
                parallel,
                ncores,
                'wait' if wait else 'background',
            )

        if not runs:
            logger.error(
                'No prepared DOLPHOT dirs found for instrument=%s under %s '
                '(need dolphot.param; run dolphot-prep first)',
                args.instrument,
                args.base_dir,
            )
            return 1

        logger.info('Found %d DOLPHOT run(s)', len(runs))
        for run in runs:
            cmd = _format_command(
                run, ncores=ncores, dolphot_bin=args.dolphot_bin
            )
            logger.info('  [%s] %s', run.label, cmd)
            try:
                _assert_run_hst_quality(run)
            except RuntimeError as exc:
                logger.error('%s', exc)
                return 1

        if args.dry_run:
            logger.info(
                'Dry run: %d run(s) would execute (parallel=%d, mode=%s)',
                len(runs),
                parallel,
                'wait' if wait else 'background',
            )
            return 0

        if not wait:
            if len(runs) > parallel:
                logger.error(
                    'Background mode requires n_runs (%d) <= --parallel (%d) '
                    'so every job can start immediately. Use --wait (default) '
                    'to queue, or raise --parallel.',
                    len(runs),
                    parallel,
                )
                return 2
            procs = [
                _start_background(
                    run, ncores=ncores, dolphot_bin=args.dolphot_bin
                )
                for run in runs
            ]
            for run, proc in zip(runs, procs):
                logger.info(
                    'Started %s pid=%s (logs: %s/dolphot.out)',
                    run.label,
                    proc.pid,
                    run.outdir,
                )
            logger.info(
                'Background launch done (%d process(es)); run-dolphot exiting. '
                'HDF5 sidecars are not written in background mode - run '
                'dolphot-hdf5 --base-dir %s after jobs finish.',
                len(procs),
                args.base_dir,
            )
            return 0

        # Wait mode: bounded concurrency, block until all finish.
        results: list[tuple[DolphotRun, int]] = []
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [
                pool.submit(
                    _run_one_wait,
                    run,
                    ncores=ncores,
                    dolphot_bin=args.dolphot_bin,
                )
                for run in runs
            ]
            for fut in as_completed(futures):
                results.append(fut.result())

        n_fail = sum(1 for _, rc in results if rc != 0)
        for run, rc in sorted(results, key=lambda item: item[0].label):
            level = logger.error if rc else logger.info
            level('  %s: rc=%d', run.label, rc)

        if bool(getattr(args, 'write_dolphot_hdf5', True)):
            force_h5 = bool(getattr(args, 'force_dolphot_hdf5', False))
            n_h5 = 0
            n_h5_skip = 0
            n_h5_fail = 0
            for run, rc in sorted(results, key=lambda item: item[0].label):
                if rc != 0:
                    continue
                try:
                    from st123.stages.photometry.dolphot_catalog_hdf5 import (
                        hdf5_path_for_phot_base,
                        phot_catalog_base_for_run,
                    )

                    dest = hdf5_path_for_phot_base(
                        phot_catalog_base_for_run(run.outdir, run.phot_out)
                    )
                    if dest.is_file() and not force_h5:
                        logger.info(
                            '  %s: HDF5 exists, skip %s', run.label, dest.name
                        )
                        n_h5_skip += 1
                        continue
                    out = _write_run_hdf5(run, force=force_h5)
                except ImportError as exc:
                    logger.warning(
                        'DOLPHOT HDF5 not written (install h5py): %s', exc
                    )
                    break
                except Exception as exc:
                    logger.error('  %s: HDF5 failed: %s', run.label, exc)
                    n_h5_fail += 1
                    continue
                if out is None:
                    logger.warning(
                        '  %s: catalog/columns missing; skip HDF5 '
                        '(run dolphot-hdf5 after products appear)',
                        run.label,
                    )
                    n_h5_fail += 1
                    continue
                logger.info('  %s: wrote HDF5 %s', run.label, out)
                n_h5 += 1
            logger.info(
                'DOLPHOT HDF5: %d wrote, %d skipped, %d failed/missing',
                n_h5,
                n_h5_skip,
                n_h5_fail,
            )
        else:
            logger.info(
                'Skipping DOLPHOT HDF5 (--no-write-dolphot-hdf5); '
                'use dolphot-hdf5 --base-dir %s later if needed',
                args.base_dir,
            )

        logger.info(
            'All DOLPHOT runs finished: %d ok, %d failed',
            len(results) - n_fail,
            n_fail,
        )
        return 0 if n_fail == 0 else 1
    finally:
        shutdown_logging()


if __name__ == '__main__':
    raise SystemExit(main())
