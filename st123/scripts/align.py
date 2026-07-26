#!/usr/bin/env python3
"""
Unified JWST alignment CLI.

Modes
-----
``visit`` (default)
    Self-align one instrument's frames under ``reduction/`` (Gaia / visit
    mosaics). Typically ``--instrument NIRCAM``.

``reference``
    Overlap-match science frames to ``reference/`` coadds and align
    (REFERENCE → optional MIRI_REL); typically ``--instrument MIRI``.

``pair``
    Align one ``--image`` to a photometry catalog from ``--ref``
    (optional ``--photfile``). Selected automatically when ``--ref`` and
    ``--image`` are both set.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import sys
import traceback
from pathlib import Path

import numpy as np
import shapely

from st123.alignment.align import (
    align_to_mosaic,
    create_alignment_mosaic,
    create_dirs,
    fix_phot,
    get_input_images,
    get_visit_geoms,
    pick_visit,
    update_refcat,
    visit_filter_dict,
)
from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    add_filters,
    add_pair_align_args,
    as_path,
    create_parser as build_parser,
    dataset_label,
    resolve_reduction_dir,
    sync_legacy_ncores,
)
from st123.utils.helpers import create_filter_table, input_list

DEFAULT_REFERENCE_DATA_DIR = Path('/data/rwisenbaker/jwst_data/M51')
DEFAULT_PAIR_OUTDIR = 'alignment_output'


def create_parser(default_data_dir: Path | None = None):
    if default_data_dir is None:
        default_data_dir = DEFAULT_REFERENCE_DATA_DIR
    parser = build_parser(
        description=(
            'Align JWST imaging. Modes: visit (Gaia / visit mosaics under '
            'reduction/), reference (science → reference/ coadds), or pair '
            '(one --image to --ref; also selected when both are set).'
        )
    )
    add_base_dir(
        parser,
        required=False,
        default=None,
        aliases=(
            '--basedir',
            '--workdir',
            '--data-dir',
            '--download-dir',
            '--outdir',
        ),
        help=(
            'Project root (…/<object>), or output directory in pair mode. '
            'Visit mode uses <base-dir>/reduction; reference mode expects '
            f'JWST/<INSTRUMENT>/… and reference/ (default when omitted: '
            f'{default_data_dir}). Pair-mode default when omitted: '
            f'{DEFAULT_PAIR_OUTDIR}. '
            'Aliases: --workdir, --basedir, --data-dir, --download-dir, --outdir.'
        ),
    )
    parser.add_argument(
        '--mode',
        choices=('visit', 'reference', 'pair'),
        default='visit',
        help=(
            'visit: self-align one instrument under reduction/ (default). '
            'reference: align --instrument frames to reference/ coadds. '
            'pair: align --image to --ref (auto if both flags are set).'
        ),
    )
    parser.add_argument(
        '--instrument',
        type=str,
        default=None,
        help=(
            'Science instrument to align. Defaults: NIRCAM for --mode visit, '
            'MIRI for --mode reference (unused in pair mode).'
        ),
    )
    add_pair_align_args(parser)
    # Reference-mode options (ignored in visit/pair except shared runtime flags).
    parser.add_argument(
        '--repo',
        type=Path,
        default=None,
        help='Path to the st123 repo (reference mode; default: auto-detect).',
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default=None,
        help=(
            'Deprecated. Parent of a dataset subdirectory; used only when '
            '--base-dir is omitted in reference mode (legacy).'
        ),
    )
    parser.add_argument(
        '--overlap-outdir',
        type=Path,
        default=None,
        help='Where to write overlap_summary.txt/json (default: <base-dir>/overlap).',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Optional cap on number of science frames (reference mode smoke test).',
    )
    add_filters(
        parser,
        help=(
            'Comma-separated filters to process in reference mode '
            '(e.g. F560W,F770W). Default: all.'
        ),
    )
    parser.add_argument(
        '--overlap-only',
        action='store_true',
        help='Reference mode: only compute overlaps; skip alignment.',
    )
    parser.add_argument(
        '--align-only',
        action='store_true',
        help='Reference mode: skip overlap recompute; load overlap_summary.json.',
    )
    parser.add_argument(
        '--overlap-json',
        type=Path,
        default=None,
        help='Existing overlap_summary.json for --align-only.',
    )
    parser.add_argument(
        '--nbright',
        type=int,
        default=800,
        help='Number of bright sources for JHAT (pair/reference; default: 800).',
    )
    parser.add_argument(
        '--match-radius',
        type=float,
        default=0.1,
        help='Sky match radius (arcsec) when merging reference catalogs.',
    )
    parser.add_argument(
        '--no-clip-footprint',
        action='store_true',
        help='Do not clip the master catalog to the science illuminated footprint.',
    )
    parser.add_argument(
        '--no-refine',
        action='store_true',
        help='Disable iterative outlier-clipping refinement after JHAT.',
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
        help='Disable filter-specific calibrator selection (reference mode).',
    )
    parser.add_argument(
        '--no-fallback',
        action='store_true',
        help='Disable MIRI→MIRI relative fallback when REFERENCE alignment fails.',
    )
    parser.add_argument(
        '--max-nircam-dispersion-mas',
        type=float,
        default=None,
        help=(
            'Uniform REFERENCE quality-hold threshold (mas). Default: per-filter '
            'map. Set <= 0 to disable.'
        ),
    )
    parser.add_argument(
        '--min-ref-overlap-frac',
        type=float,
        default=0.02,
        help=(
            'Minimum cumulative reference overlap fraction to attempt alignment '
            '(default: 0.02).'
        ),
    )
    parser.add_argument(
        '--legacy-overlap-file',
        type=Path,
        default=None,
        help='Legacy sequential overlap-file pipeline (reference mode).',
    )
    parser.add_argument(
        '--legacy-outdir',
        type=Path,
        default=Path('alignment_output_auto'),
        help='Output root for --legacy-overlap-file (default: alignment_output_auto).',
    )
    add_common_runtime(parser, ncores_default=1, plot=True, verbose=True)
    # Stash for help text / resolve defaults in main.
    parser.set_defaults(_default_reference_data_dir=str(default_data_dir))
    return parser


def _resolve_instrument(mode: str, instrument: str | None) -> str:
    if instrument and str(instrument).strip():
        return str(instrument).strip().upper()
    return 'MIRI' if mode == 'reference' else 'NIRCAM'


def _resolve_mode(args: argparse.Namespace) -> str:
    """
    Resolve alignment mode.

    ``--ref`` / ``--image`` select pair mode automatically. Explicit
    ``--mode pair`` requires both paths.
    """
    has_ref = bool(getattr(args, 'ref', None))
    has_image = bool(getattr(args, 'image', None))
    if has_ref or has_image or args.mode == 'pair':
        if not (has_ref and has_image):
            raise ValueError(
                'pair mode requires both --ref and --image '
                '(optional: --photfile, --base-dir/--outdir)'
            )
        if args.mode == 'reference':
            raise ValueError(
                'do not combine --ref/--image with --mode reference; '
                'use --mode pair (or omit --mode)'
            )
        return 'pair'
    return args.mode


def run_pair_alignment(args: argparse.Namespace) -> int:
    """Align one ``--image`` to a catalog built from ``--ref``."""
    from st123.alignment.align import run_alignment

    ref = str(Path(args.ref).expanduser())
    align_image = str(Path(args.image).expanduser())
    for path, label in ((ref, '--ref'), (align_image, '--image')):
        if not os.path.exists(path):
            print(f'ERROR: {label} file not found: {path}', file=sys.stderr)
            return 1

    outdir = args.base_dir if args.base_dir is not None else DEFAULT_PAIR_OUTDIR
    try:
        guess_offset, out = run_alignment(
            ref_image=ref,
            align_image=align_image,
            outdir=outdir,
            photfile=args.photfile,
            nbright=args.nbright,
            plot=args.plot,
            verbose=args.verbose,
            clip_to_align_footprint=not args.no_clip_footprint,
            refine=not args.no_refine,
            refine_sigma=args.refine_sigma,
            refine_max_iter=args.refine_max_iter,
        )
    except Exception as exc:
        print(f'ERROR: alignment failed: {exc}', file=sys.stderr)
        if getattr(args, 'verbose', False):
            traceback.print_exc()
        return 1

    print(f'Guess offset (x, y): {guess_offset}')
    print(f'Done. Products in {out}')
    return 0


def _visit_patterns(instrument: str) -> list[str]:
    key = instrument.upper()
    if key in ('NIRCAM', 'NRC'):
        return ['*nrca*_cal.fits', '*nrcb*_cal.fits']
    raise ValueError(
        f'visit mode currently supports --instrument NIRCAM only (got {instrument!r})'
    )


def run_visit_alignment(
    *,
    base_dir: str | Path,
    instrument: str = 'NIRCAM',
    ncores: int = 1,
    verbose: bool = False,
) -> int:
    """Self-align instrument frames under ``reduction/`` (Gaia / visit mosaics)."""
    work_dir = str(resolve_reduction_dir(base_dir))
    if verbose:
        print(f'Dataset: {dataset_label(base_dir)}')
        print(f'Mode: visit')
        print(f'Instrument: {instrument}')
        print(f'Reduction workdir: {work_dir}')

    create_dirs(work_dir)
    patterns = _visit_patterns(instrument)
    input_images = get_input_images(pattern=patterns, workdir=work_dir)
    table = input_list(input_images)
    ngroups = np.unique(table['group'])
    visit_filter = visit_filter_dict(table)

    for group_id in ngroups:
        combined_photfile = os.path.join(
            work_dir, 'align', f'group_{group_id}', 'reference_catalog.txt'
        )
        group_table = table[table['group'] == group_id]
        group_visits = np.unique(group_table['visit'])
        visit_geoms = get_visit_geoms(group_table)
        align_polygon = None

        for visit_index in range(len(group_visits)):
            visit_id, overlap_frac = pick_visit(
                align_polygon, copy.copy(visit_geoms), visit_filter
            )
            visit_table = group_table[group_table['visit'] == visit_id]

            filters = np.unique(visit_table['filter']).value
            filter_table = create_filter_table(visit_table, filters)
            align_filter = visit_filter[visit_id]
            visit_outdir = os.path.join(
                work_dir, 'align', f'group_{group_id}', f'visit_{visit_id}'
            )

            if visit_index == 0:
                mosaic_name, guess_offset = create_alignment_mosaic(
                    filter_table,
                    visit_outdir,
                    align_filter=align_filter,
                    align_to='gaia',
                    ncores=ncores,
                )
            else:
                nbright = 50000 if overlap_frac < 0.3 else 800
                mosaic_name, guess_offset = create_alignment_mosaic(
                    filter_table,
                    visit_outdir,
                    align_filter=align_filter,
                    align_to=combined_photfile,
                    ncores=ncores,
                    Nbright=nbright,
                )

            mosaic_photfile = fix_phot(mosaic_name)
            _ = update_refcat(
                mosaic_name,
                mosaic_photfile,
                out_refcat=combined_photfile,
                align_pgon=align_polygon,
            )

            print(f'Mosaic photfile: {mosaic_photfile}')
            for _filt, filt_table in filter_table.items():
                align_to_mosaic(
                    mosaic_photfile,
                    [row['image'] for row in filt_table],
                    os.path.join(work_dir, 'jhat'),
                    guess_offset=guess_offset,
                    verbose=verbose,
                    ncores=ncores,
                )

            if align_polygon is None:
                align_polygon = visit_geoms[visit_id]
            else:
                align_polygon = shapely.unary_union(
                    [align_polygon, visit_geoms[visit_id]]
                )

            visit_geoms.pop(visit_id)
    return 0


def run_reference_alignment(args: argparse.Namespace) -> int:
    """Overlap + align science frames to ``reference/`` coadds."""
    from st123.alignment.align import (
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

    default_data_dir = Path(
        getattr(args, '_default_reference_data_dir', DEFAULT_REFERENCE_DATA_DIR)
    )

    pre_repo = getattr(args, 'repo', None)
    try:
        repo = _resolve_repo_root(pre_repo)
    except FileNotFoundError:
        repo = Path('/data/rwisenbaker/st123')
    _bootstrap_imports(repo)

    if socket.getdefaulttimeout() is None:
        socket.setdefaulttimeout(15)

    from st123.alignment.align import run_alignment
    from st123.mosaic.image_overlap import BestOverlap, MirIFootprint, compute_overlap

    args.repo = repo
    sync_legacy_ncores(args)
    args.data_dir = as_path(args.base_dir) if args.base_dir else None
    args.galaxy = None

    if args.legacy_overlap_file is not None:
        return run_legacy_overlap_file_pipeline(
            args.legacy_overlap_file,
            outdir=args.legacy_outdir,
            plot=args.plot,
            verbose=args.verbose,
        )

    data_dir, label = resolve_data_dir(
        data_dir=args.data_dir,
        data_root=args.data_root,
        galaxy=None,
        default_data_dir=default_data_dir,
    )
    args.data_dir = data_dir
    args.base_dir = str(data_dir)
    args.galaxy = label
    args.data_root = data_dir
    summary_rows: list[AlignmentSummaryRow] = []
    summary_path = data_dir / f'{label}_alignment_summary.txt'

    if args.verbose:
        print(f'Dataset: {label}')
        print(f'Mode: reference')
        print(f'Instrument: {args.instrument}')
        print(f'Data dir: {data_dir}')

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

        # Always continue on per-frame errors; FAILURE rows go to the summary.
        failures, summary_rows = align_from_frames(
            frames,
            run_alignment=run_alignment,
            nbright=args.nbright,
            plot=args.plot,
            verbose=args.verbose,
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
            workers=args.ncores,
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


def main(argv=None) -> int:
    # Pre-parse --help / --repo so reference-mode help does not require JHAT.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--repo', type=Path, default=None)
    pre.add_argument('-h', '--help', action='store_true')
    pre_args, _ = pre.parse_known_args(argv)

    default_data_dir = DEFAULT_REFERENCE_DATA_DIR
    if pre_args.help:
        create_parser(default_data_dir).print_help()
        return 0

    parser = create_parser(default_data_dir)
    args = parser.parse_args(argv)
    sync_legacy_ncores(args)
    try:
        args.mode = _resolve_mode(args)
    except ValueError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    args.instrument = _resolve_instrument(args.mode, args.instrument)

    if args.mode == 'pair':
        return run_pair_alignment(args)

    if args.mode == 'visit':
        base = args.base_dir if args.base_dir is not None else '.'
        try:
            return run_visit_alignment(
                base_dir=base,
                instrument=args.instrument,
                ncores=args.ncores,
                verbose=args.verbose,
            )
        except ValueError as exc:
            print(f'ERROR: {exc}', file=sys.stderr)
            return 1

    return run_reference_alignment(args)


if __name__ == '__main__':
    raise SystemExit(main())
