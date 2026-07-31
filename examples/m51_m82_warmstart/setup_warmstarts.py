#!/usr/bin/env python3
"""
Stage pruned warm-start DOLPHOT directories for eligible M51/M82 references.

Eligibility (dataset-specific architecture)
------------------------------------------
A reference is staged only if it has ≥ ``MIN_MIRI_IMAGES`` (default 10) unique
usable MIRI JHAT frames in the alignment summary **and** a finished NIRCam
DOLPHOT run under ``dolphot/nircam_{g}_{b}``.

Does **not** launch ``dolphot``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as a script from this directory or via python path.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import (  # noqa: E402
    DATASETS,
    DEFAULT_NCORES,
    MIN_MIRI_IMAGES,
    jhat_list_path,
    nircam_dir,
    phot_out_name,
    warmstart_outdir,
)
from discover import discover_dataset  # noqa: E402

logger = logging.getLogger('m51_m82_setup')


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--dataset',
        choices=sorted(DATASETS),
        action='append',
        default=None,
        help='Dataset(s) to stage (default: both; M82 skipped if NIRCam missing).',
    )
    p.add_argument(
        '--min-miri',
        type=int,
        default=MIN_MIRI_IMAGES,
        help='Minimum MIRI JHAT frames required to stage a reference.',
    )
    p.add_argument('--ncores', type=int, default=DEFAULT_NCORES)
    p.add_argument(
        '--dry-run',
        action='store_true',
        help='Print eligible refs only; do not call dolphot-warmstart.',
    )
    p.add_argument('-v', '--verbose', action='store_true')
    return p


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s %(message)s',
    )
    from st123.photometry.warmstart import setup_miri_warmstart
    from st123.scripts.utils.options import default_alignment_summary

    names = args.dataset or sorted(DATASETS)
    ncores = max(1, int(args.ncores))
    staged = 0
    skipped = 0

    for name in names:
        spec = DATASETS[name]
        plans = discover_dataset(spec, min_miri=args.min_miri, write_lists=True)
        print(f'\n=== {name}: min_miri={args.min_miri} ===')
        for plan in plans:
            if not plan.eligible:
                logger.info('SKIP %s_%s: %s', name, plan.ref_key, plan.skip_reason)
                skipped += 1
                continue
            out = warmstart_outdir(spec, plan.group, plan.box)
            phot_out = phot_out_name(spec, plan.group, plan.box)
            existing = out / phot_out
            if existing.is_file() and existing.stat().st_size > 0:
                logger.info('SKIP %s (catalog exists): %s', plan.ref_key, existing)
                skipped += 1
                continue
            jhats = list(plan.jhat_paths)
            logger.info(
                'STAGE %s_%s  n_miri=%d  out=%s',
                name,
                plan.ref_key,
                len(jhats),
                out,
            )
            if args.dry_run:
                staged += 1
                continue
            summary = default_alignment_summary(spec.base_dir)
            setup_miri_warmstart(
                nircam_dir(spec, plan.group, plan.box),
                out,
                miri_jhat=jhats,
                data_root=spec.base_dir,
                alignment_summary=summary if Path(summary).is_file() else None,
                phot_out=phot_out,
                prune_xyt_for_miri=True,
                ncores=ncores,
            )
            # Ensure list file matches what was staged.
            jhat_list_path(spec, plan.group, plan.box).write_text(
                '\n'.join(str(p) for p in jhats) + '\n'
            )
            staged += 1

    print(f'\nDone. staged={staged} skipped={skipped} (min_miri={args.min_miri})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
