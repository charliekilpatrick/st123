#!/usr/bin/env python3
"""
End-to-end MIRI–reference overlap + alignment pipeline CLI.

Library logic lives in :mod:`st123.alignment.alignment_wrap`. This module
is the ``st123/scripts/`` entry point (``create_parser`` / ``main``).

Example::

    python -m st123.scripts.alignment_wrap \\
      --data-dir /data/rwisenbaker/jwst_data/NGC3310 \\
      --plot --continue-on-error --workers 8
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import traceback
from pathlib import Path

from st123.alignment.alignment_wrap import (
    AlignmentSummaryRow,
    _bootstrap_imports,
    _resolve_repo_root,
    align_from_frames,
    filter_name_from_miri_path,
    parse_filters_arg,
    resolve_data_dir,
    run_legacy_overlap_file_pipeline,
    run_overlaps,
    write_alignment_summary,
)


def create_parser(
    default_data_dir: Path | None = None,
) -> argparse.ArgumentParser:
    if default_data_dir is None:
        default_data_dir = Path('/data/rwisenbaker/jwst_data/M51')
    parser = argparse.ArgumentParser(
        description=(
            'Run MIRI/reference overlap matching then align each MIRI frame '
            'to its best-overlap reference. Dataset root is ``--data-dir`` '
            'with layout JWST/MIRI/<FILTER>/<obsid>/mastDownload/... '
            'plus reference/.'
        )
    )
    parser.add_argument(
        '--repo',
        type=Path,
        default=None,
        help='Path to the st123 repo (default: auto-detect).',
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=None,
        help=(
            'Dataset root containing '
            'JWST/MIRI/<FILTER>/<obsid>/mastDownload/... MIRI '
            'cals and reference/ coadds '
            f'(default: {default_data_dir}).'
        ),
    )
    parser.add_argument(
        '--galaxy',
        default=None,
        help=(
            'Optional dataset label used in summary filenames '
            '(default: basename of --data-dir).'
        ),
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default=None,
        help=(
            'Deprecated. Parent of a galaxy subdirectory; used only when '
            '--data-dir is omitted as <data-root>/<galaxy>.'
        ),
    )
    parser.add_argument(
        '--overlap-outdir',
        type=Path,
        default=None,
        help=(
            'Where to write overlap_summary.txt/json '
            '(default: <data-dir>/overlap).'
        ),
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Optional cap on number of MIRI frames (useful for a smoke test).',
    )
    parser.add_argument(
        '--filters',
        type=str,
        default=None,
        help=(
            'Comma-separated MIRI filters to process '
            '(e.g. F560W or F560W,F770W). Default: all filters.'
        ),
    )
    parser.add_argument(
        '--overlap-only',
        action='store_true',
        help='Only compute overlaps; skip alignment.',
    )
    parser.add_argument(
        '--align-only',
        action='store_true',
        help='Skip overlap recompute; load overlap_summary.json and align.',
    )
    parser.add_argument(
        '--overlap-json',
        type=Path,
        default=None,
        help='Existing overlap_summary.json to use with --align-only.',
    )
    parser.add_argument(
        '--nbright',
        type=int,
        default=800,
        help='Number of bright sources for JHAT (default: 800).',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Enable JHAT diagnostic plots during alignment.',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose JHAT / alignment output.',
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue aligning remaining frames if one fails.',
    )
    parser.add_argument(
        '--match-radius',
        type=float,
        default=0.1,
        help=(
            'Sky match radius in arcsec when merging overlapping reference '
            'catalogs (default: 0.1).'
        ),
    )
    parser.add_argument(
        '--no-clip-footprint',
        action='store_true',
        help=(
            'Do not clip the master catalog to the MIRI illuminated footprint '
            '(default: clip so Nbright prefers in-frame stars).'
        ),
    )
    parser.add_argument(
        '--no-refine',
        action='store_true',
        help='Disable iterative outlier-clipping refinement after the initial JHAT alignment.',
    )
    parser.add_argument(
        '--refine-sigma',
        type=float,
        default=2.0,
        help='Sigma threshold for iterative residual clipping (default: 2.0).',
    )
    parser.add_argument(
        '--refine-max-iter',
        type=int,
        default=5,
        help='Maximum iterative refinement iterations (default: 5).',
    )
    parser.add_argument(
        '--no-filter-calibrators',
        action='store_true',
        help=(
            'Disable filter-specific calibrator selection (F770W currently '
            'uses lower Nbright and tighter refine residual clipping). '
            'Default: enabled.'
        ),
    )
    parser.add_argument(
        '--no-fallback',
        action='store_true',
        help=(
            'Disable MIRI→MIRI relative fallback when NIRCam alignment fails '
            '(default: try fallback using the closest-wavelength overlapping '
            'successfully aligned MIRI frame).'
        ),
    )
    parser.add_argument(
        '--max-nircam-dispersion-mas',
        type=float,
        default=None,
        help=(
            'Uniform REFERENCE quality-hold threshold (mas): if REFERENCE '
            'dispersion exceeds this value, try MIRI_REL and keep it only when '
            'it improves the absolute dispersion. Default: per-filter map '
            '(F560W disabled; F770W 50; F1000W 35; F1130W 55; F1280W/F1500W/'
            'F1800W 50; F2100W 65). Set <= 0 to disable the quality hold.'
        ),
    )
    parser.add_argument(
        '--min-ref-overlap-frac',
        type=float,
        default=0.02,
        help=(
            'Minimum cumulative (union) fraction of the MIRI illuminated ROI '
            'that must be covered by overlapping reference footprints to '
            'attempt alignment (default: 0.02). Frames below this threshold '
            'are skipped and omitted from the alignment summary.'
        ),
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help=(
            'Number of parallel alignment workers within each filter wave '
            '(default: 1). Filters are still processed blue→red sequentially '
            'so fallback can use completed bluer alignments.'
        ),
    )
    parser.add_argument(
        '--legacy-overlap-file',
        type=Path,
        default=None,
        help=(
            'Run the original sequential MIRI/reference pair loop against an '
            'overlap text file (lines containing "Overlap maximized") instead '
            'of the full filter-wave pipeline. Writes successful_alignments.txt '
            'and failed_alignments.txt under the current working directory.'
        ),
    )
    parser.add_argument(
        '--legacy-outdir',
        type=Path,
        default=Path('alignment_output_auto'),
        help=(
            'Output root for --legacy-overlap-file pair products '
            '(default: alignment_output_auto).'
        ),
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    # Pre-parse --repo / --help so --help does not require heavy imports.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--repo', type=Path, default=None)
    pre.add_argument('-h', '--help', action='store_true')
    pre_args, _ = pre.parse_known_args(argv)

    try:
        repo = _resolve_repo_root(pre_args.repo)
    except FileNotFoundError:
        repo = Path('/data/rwisenbaker/st123')
    default_data_dir = Path('/data/rwisenbaker/jwst_data/M51')

    if pre_args.help:
        create_parser(default_data_dir).print_help()
        return 0

    repo = _resolve_repo_root(pre_args.repo)
    _bootstrap_imports(repo)

    # JHAT imports astroquery.gaia, which contacts ESA TAP on import.
    if socket.getdefaulttimeout() is None:
        socket.setdefaulttimeout(15)

    from st123.alignment.relative_align import run_alignment
    from st123.mosaic.image_overlap import BestOverlap, MirIFootprint, compute_overlap

    args = create_parser(default_data_dir).parse_args(argv)
    args.repo = repo

    if args.legacy_overlap_file is not None:
        return run_legacy_overlap_file_pipeline(
            args.legacy_overlap_file,
            outdir=args.legacy_outdir,
            plot=args.plot,
            verbose=args.verbose,
        )

    data_dir, dataset_label = resolve_data_dir(
        data_dir=args.data_dir,
        data_root=args.data_root,
        galaxy=args.galaxy,
        default_data_dir=default_data_dir,
    )
    args.data_dir = data_dir
    args.galaxy = dataset_label
    args.data_root = data_dir
    summary_rows: list[AlignmentSummaryRow] = []
    summary_path = data_dir / f'{dataset_label}_alignment_summary.txt'

    if args.overlap_only and args.align_only:
        print(
            'ERROR: choose at most one of --overlap-only / --align-only',
            file=sys.stderr,
        )
        return 2

    try:
        if args.align_only:
            json_path = args.overlap_json
            if json_path is None:
                json_path = data_dir / 'overlap' / 'overlap_summary.json'
            json_path = Path(json_path).expanduser().resolve()
            if not json_path.is_file():
                print(f'ERROR: overlap JSON not found: {json_path}', file=sys.stderr)
                return 1
            print(f'Loading overlaps from {json_path}')
            payload = json.loads(json_path.read_text())
            frames = payload['frames']
            filters = parse_filters_arg(args.filters)
            if filters:
                wanted = {f.upper() for f in filters}
                frames = [
                    f
                    for f in frames
                    if filter_name_from_miri_path(
                        f['miri_path'] if isinstance(f, dict) else f.miri_path
                    )
                    in wanted
                ]
                print(
                    f'Filter restriction {", ".join(filters)}: '
                    f'{len(frames)} frame(s) from overlap JSON'
                )
            if args.limit is not None:
                frames = frames[: args.limit]
        else:
            frames = run_overlaps(
                args,
                MirIFootprint=MirIFootprint,
                BestOverlap=BestOverlap,
                compute_overlap=compute_overlap,
            )
            if args.overlap_only:
                return 0

        failures, summary_rows = align_from_frames(
            frames,
            run_alignment=run_alignment,
            nbright=args.nbright,
            plot=args.plot,
            verbose=args.verbose,
            continue_on_error=args.continue_on_error,
            cache_dir=data_dir / 'overlap' / 'ref_phot_cache',
            match_radius_arcsec=args.match_radius,
            clip_to_align_footprint=not args.no_clip_footprint,
            refine=not args.no_refine,
            refine_sigma=args.refine_sigma,
            refine_max_iter=args.refine_max_iter,
            use_filter_calibrators=not args.no_filter_calibrators,
            fallback=not args.no_fallback,
            max_nircam_dispersion_mas=args.max_nircam_dispersion_mas,
            min_ref_overlap_frac=args.min_ref_overlap_frac,
            summary_outfile=summary_path,
            workers=args.workers,
            repo=args.repo,
        )
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        if getattr(args, 'verbose', False):
            traceback.print_exc()
        if summary_rows:
            summary_path = write_alignment_summary(summary_rows, summary_path)
            print(f'Alignment summary: {summary_path}')
        return 1

    summary_path = write_alignment_summary(summary_rows, summary_path)
    print(f'Alignment summary: {summary_path}')

    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
