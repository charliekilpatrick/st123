#!/usr/bin/env python3
"""
Print (or launch) DOLPHOT warm-start commands for eligible M51/M82 refs.

Uses the same ≥ ``MIN_MIRI_IMAGES`` gate as setup. Default is dry-run.
Never starts more than ``MAX_PARALLEL_DOLPHOT`` jobs when ``--go`` is set.

Supports split runs written by :func:`st123.photometry.dolphot_split.write_split_paramfiles`
(``dolphot_split.json``): each unfinished part is launched, then catalogs are
merged into the final ``*_nircam_miri.phot``.
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


def _dolphot_bin() -> Path:
    dolphot = Path(os.environ.get('DOLPHOT_BIN', '/data/software/dolphot/bin')) / 'dolphot'
    if dolphot.is_file():
        return dolphot
    import shutil

    which = shutil.which('dolphot')
    if not which:
        raise FileNotFoundError('dolphot not found')
    return Path(which)


def _collect_jobs(
    names: list[str],
    *,
    min_miri: int,
    ncores: int,
) -> list[tuple[str, Path, list[str]]]:
    """Return (label, outdir, cmd) for each unfinished part."""
    from st123.photometry.dolphot_split import (
        finalize_split_outdir,
        iter_launchable_parts,
        load_split_manifest,
    )

    dolphot = _dolphot_bin()
    commands: list[tuple[str, Path, list[str]]] = []
    for name in names:
        spec = DATASETS[name]
        for plan in discover_dataset(spec, min_miri=min_miri, write_lists=False):
            if not plan.eligible:
                continue
            out = warmstart_outdir(spec, plan.group, plan.box)
            phot = phot_out_name(spec, plan.group, plan.box)
            final_phot = out / phot
            if final_phot.is_file() and final_phot.stat().st_size > 0:
                logger.info('SKIP finished %s_%s', name, plan.ref_key)
                continue

            split = load_split_manifest(out)
            has_param = (out / 'dolphot.param').is_file() or (
                split is not None and (out / split.parts[0].param_file).is_file()
            )
            if not has_param:
                logger.warning('setup missing for %s_%s (%s)', name, plan.ref_key, out)
                continue

            # If all parts done but merge pending, merge now (live or dry-run note).
            pending = iter_launchable_parts(out, phot_out=phot)
            if not pending and split is not None and split.needs_merge:
                if final_phot.is_file() and final_phot.stat().st_size > 0:
                    logger.info('SKIP finished %s_%s', name, plan.ref_key)
                    continue
                logger.info('MERGE ready %s_%s', name, plan.ref_key)
                # Represent merge as a pseudo-command handled in --go.
                commands.append(
                    (f'{name}_{plan.ref_key}_MERGE', out, ['__MERGE__', phot])
                )
                continue

            for param_file, part_phot in pending:
                cmd = [
                    str(dolphot),
                    part_phot,
                    f'-p{param_file}',
                    f'MaxThreads={ncores}',
                ]
                label = f'{name}_{plan.ref_key}'
                if param_file != 'dolphot.param':
                    label = f'{label}:{param_file}'
                commands.append((label, out, cmd))
    return commands


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    names = args.dataset or sorted(DATASETS)
    ncores = max(1, int(args.ncores))
    max_par = max(1, int(args.max_parallel))

    try:
        commands = _collect_jobs(names, min_miri=args.min_miri, ncores=ncores)
    except FileNotFoundError as exc:
        logger.error('%s', exc)
        return 1

    print(
        f'Mode: {"LIVE" if args.go else "DRY-RUN"}  '
        f'min_miri={args.min_miri}  ncores={ncores}  max_parallel={max_par}'
    )
    print(f'Jobs: {len(commands)}\n')
    for label, out, cmd in commands:
        if cmd[0] == '__MERGE__':
            print(f'# {label}')
            print(
                f'cd {out} && python -c "from st123.photometry.dolphot_split '
                f'import finalize_split_outdir; '
                f'finalize_split_outdir(r\'{out}\')"'
            )
            print()
            continue
        print(f'# {label}')
        print(f'cd {out} && {" ".join(cmd)}')
        print()

    if not args.go:
        print('Dry-run only. Re-run with --go to start (≤ max-parallel at a time).')
        return 0

    from st123.photometry.dolphot_split import finalize_split_outdir

    running: list[tuple[str, subprocess.Popen, Path]] = []
    queue = list(commands)
    while queue or running:
        running = [(lab, proc, out) for lab, proc, out in running if proc.poll() is None]
        while queue and len(running) < max_par:
            label, out, cmd = queue.pop(0)
            if cmd[0] == '__MERGE__':
                merged = finalize_split_outdir(out)
                if merged is None:
                    logger.error('merge failed for %s', out)
                else:
                    logger.info('merged %s → %s', label, merged)
                continue
            # Per-part logs when split; otherwise dolphot.out / .err
            stem = Path(cmd[1]).stem
            out_log = out / f'{stem}.out' if 'part' in stem else out / 'dolphot.out'
            err_log = out / f'{stem}.err' if 'part' in stem else out / 'dolphot.err'
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

    # Final merge pass for any dirs whose parts finished during this session.
    for name in names:
        spec = DATASETS[name]
        for plan in discover_dataset(spec, min_miri=args.min_miri, write_lists=False):
            if not plan.eligible:
                continue
            out = warmstart_outdir(spec, plan.group, plan.box)
            phot = out / phot_out_name(spec, plan.group, plan.box)
            if phot.is_file() and phot.stat().st_size > 0:
                continue
            merged = finalize_split_outdir(out)
            if merged is not None:
                logger.info('merged %s', merged)

    logger.info('All launched jobs have exited.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
