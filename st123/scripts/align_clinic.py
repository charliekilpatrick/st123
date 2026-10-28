#!/usr/bin/env python3
"""Diagnose and salvage HST alignment failures on local reduction trees."""

from __future__ import annotations

import logging
from pathlib import Path

from st123.scripts.utils.options import (
    add_align_clinic_args,
    add_base_dir,
    add_common_runtime,
    add_filters,
    add_instruments_arg,
    add_sky_coord_args,
    configure_logging_from_args,
    create_parser as build_parser,
)
from st123.utils.logging import shutdown_logging

logger = logging.getLogger(__name__)


def create_parser():
    parser = build_parser(
        description=(
            'Local HST alignment clinic: classify failed JHAT/mosaic '
            'reductions, flag unsalvageable frames, and optionally recover '
            'visits (Gaia visit shift, image lock). Heavier than the unit '
            'suite; intended for burbidge, not CI.'
        )
    )
    add_base_dir(parser, required=False)
    add_instruments_arg(parser)
    add_sky_coord_args(parser, required=False)
    add_filters(parser)
    add_align_clinic_args(parser)
    add_common_runtime(parser, plot=False, verbose=True, dry_run=True)
    return parser


def _collect_bases(args) -> list[Path]:
    from st123.stages.alignment.clinic import (
        load_targets_file,
        resolve_known_targets,
    )

    bases: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        key = path.resolve() if path.exists() else path
        if key in seen:
            return
        seen.add(key)
        bases.append(path)

    if getattr(args, 'base_dir', None):
        _add(Path(args.base_dir).expanduser())
    if getattr(args, 'scan_known', False):
        for spec in resolve_known_targets():
            if spec.get('available') and spec.get('base_dir'):
                _add(Path(str(spec['base_dir'])))
            else:
                logger.info(
                    'Known target %s not on disk (issue %s)',
                    spec.get('name'),
                    spec.get('issue'),
                )
    targets_file = getattr(args, 'targets_file', None)
    if targets_file:
        for path in load_targets_file(targets_file):
            _add(path)
    return bases


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    outfile = getattr(args, 'clinic_outfile', None)
    if outfile and not getattr(args, 'base_dir', None):
        from st123.utils.logging import setup_script_logging

        setup_script_logging(
            str(Path(outfile).expanduser().resolve().parent),
            'align-clinic',
            verbose=bool(getattr(args, 'verbose', False)),
        )
    else:
        configure_logging_from_args(args, 'align-clinic')
    try:
        from st123.stages.alignment.clinic import run_clinic

        bases = _collect_bases(args)
        if not bases:
            logger.error(
                'No reduction roots. Pass --base-dir and/or --scan-known.'
            )
            return 2
        logger.info('Clinic targets (%d):', len(bases))
        for path in bases:
            logger.info('  %s', path)
        apply = bool(getattr(args, 'apply', False))
        mosaic_after = bool(getattr(args, 'mosaic_after', False))
        if getattr(args, 'dry_run', False):
            apply = False
            mosaic_after = False
            logger.info('Dry run: diagnose only')
        combined = run_clinic(
            bases,
            apply=apply,
            mosaic_after=mosaic_after,
            outfile=getattr(args, 'clinic_outfile', None),
        )
        n_ok = 0
        n_bad = 0
        for report in combined.get('targets') or []:
            name = Path(str(report.get('base_dir') or '')).name
            check = report.get('after') or report
            flags = check.get('flags') or {}
            paths = check.get('success_paths') or {}
            logger.info(
                '%s: frames=%s salvageable=%s unsalvageable=%s '
                'flags=%s paths=%s mosaic_drop=%s rejected_only=%s',
                name,
                check.get('n_frames'),
                check.get('n_salvageable'),
                check.get('n_unsalvageable'),
                flags,
                paths,
                len(check.get('mosaic_drop') or []),
                len(check.get('rejected_only') or []),
            )
            if check.get('ok'):
                n_ok += 1
            else:
                n_bad += 1
        logger.info('Clinic done: %d ok, %d with remaining issues', n_ok, n_bad)
        return 0 if n_bad == 0 else 1
    finally:
        shutdown_logging()


if __name__ == '__main__':
    raise SystemExit(main())
