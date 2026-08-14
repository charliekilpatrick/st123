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
    configure_logging_from_args,
    create_parser as build_parser,
    dataset_label,
    resolve_project_root,
    resolve_reduction_dir,
)
from st123.utils.logging import shutdown_logging

logger = logging.getLogger(__name__)

_CLI_INSTRUMENTS = ('nircam', 'miri', 'acs', 'wfc3', 'wfpc2', 'hst')
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


def _parse_instrument(value: str) -> str:
    key = str(value).strip().lower()
    if key == 'nrc':
        key = 'nircam'
    if key not in _CLI_INSTRUMENTS:
        raise argparse.ArgumentTypeError(
            f'invalid instrument {value!r}; choose from '
            f'{", ".join(_CLI_INSTRUMENTS)}'
        )
    return key


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
    from st123.photometry.dolphot import resolve_dolphot_bin

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
            'can be launched immediately (n_runs <= --parallel).'
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
    parser.add_argument(
        '--instrument',
        type=_parse_instrument,
        default='nircam',
        metavar='INSTRUMENT',
        help=(
            'Which prepared runs to execute (case-insensitive). '
            'nircam → reduction/phot_* and dolphot/nircam_* (free runs; not '
            'warmstarts). miri → every dolphot/nircam_miri_* (+ miri_*). '
            'hst / acs / wfc3 → every dolphot/nircam_hst_G_B warmstart '
            '(dolphot nircam_hst_G_B.phot -pdolphot.param MaxThreads=…) '
            'plus matching dolphot/{hst,acs,wfc3,wfpc2}_* if present.'
        ),
    )
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
    return discover_dolphot_runs(
        args.base_dir,
        instrument=args.instrument,
        group=args.group,
        box=box,
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
                'Background launch done (%d process(es)); run-dolphot exiting',
                len(procs),
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
