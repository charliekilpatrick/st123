#!/usr/bin/env python3
"""
Print (or launch) DOLPHOT warm-start commands for eligible M51/M82 refs.

Uses the same ≥ ``MIN_MIRI_IMAGES`` gate as setup. Default is dry-run.
Never starts more than ``MAX_PARALLEL_DOLPHOT`` jobs when ``--go`` is set.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import (  # noqa: E402
    DATASETS,
    DEFAULT_NCORES,
    MAX_PARALLEL_DOLPHOT,
    MIN_MIRI_IMAGES,
    phot_out_name,
    warmstart_outdir,
)
from discover import discover_dataset  # noqa: E402

logger = logging.getLogger('m51_m82_launch')


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=sorted(DATASETS), action='append', default=None)
    p.add_argument('--min-miri', type=int, default=MIN_MIRI_IMAGES)
    p.add_argument('--ncores', type=int, default=DEFAULT_NCORES)
    p.add_argument(
        '--max-parallel',
        type=int,
        default=MAX_PARALLEL_DOLPHOT,
        help='Max concurrent dolphot jobs when using --go.',
    )
    p.add_argument(
        '--go',
        action='store_true',
        help='Actually start jobs (default: print commands only).',
    )
    return p


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    names = args.dataset or sorted(DATASETS)
    ncores = max(1, int(args.ncores))
    max_par = max(1, int(args.max_parallel))
    dolphot = Path(os.environ.get('DOLPHOT_BIN', '/data/software/dolphot/bin')) / 'dolphot'
    if not dolphot.is_file():
        # fall back to PATH
        import shutil

        which = shutil.which('dolphot')
        if not which:
            logger.error('dolphot not found')
            return 1
        dolphot = Path(which)

    commands: list[tuple[str, Path, list[str]]] = []
    for name in names:
        spec = DATASETS[name]
        for plan in discover_dataset(spec, min_miri=args.min_miri, write_lists=False):
            if not plan.eligible:
                continue
            out = warmstart_outdir(spec, plan.group, plan.box)
            phot = phot_out_name(spec, plan.group, plan.box)
            if not (out / 'dolphot.param').is_file():
                logger.warning('setup missing for %s_%s (%s)', name, plan.ref_key, out)
                continue
            if (out / phot).is_file() and (out / phot).stat().st_size > 0:
                logger.info('SKIP finished %s_%s', name, plan.ref_key)
                continue
            cmd = [
                str(dolphot),
                phot,
                f'-p{out / "dolphot.param"}',
                f'MaxThreads={ncores}',
            ]
            # dolphot expects -pPARAM relative to cwd usually as basename
            cmd = [
                str(dolphot),
                phot,
                '-pdolphot.param',
                f'MaxThreads={ncores}',
            ]
            commands.append((f'{name}_{plan.ref_key}', out, cmd))

    print(
        f'Mode: {"LIVE" if args.go else "DRY-RUN"}  '
        f'min_miri={args.min_miri}  ncores={ncores}  max_parallel={max_par}'
    )
    print(f'Jobs: {len(commands)}\n')
    for label, out, cmd in commands:
        print(f'# {label}  n_miri from plan (see discover)')
        print(f'cd {out} && {" ".join(cmd)}')
        print()

    if not args.go:
        print('Dry-run only. Re-run with --go to start (≤ max-parallel at a time).')
        return 0

    running: list[tuple[str, subprocess.Popen, Path]] = []
    queue = list(commands)
    while queue or running:
        running = [(lab, proc, out) for lab, proc, out in running if proc.poll() is None]
        while queue and len(running) < max_par:
            label, out, cmd = queue.pop(0)
            out_log = out / 'dolphot.out'
            err_log = out / 'dolphot.err'
            with out_log.open('w') as fo, err_log.open('w') as fe:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(out),
                    stdout=fo,
                    stderr=fe,
                    start_new_session=True,
                )
            (out / 'dolphot.pid').write_text(f'{proc.pid}\n')
            logger.info('started %s pid=%d', label, proc.pid)
            running.append((label, proc, out))
        time.sleep(5)

    logger.info('All launched jobs have exited.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
