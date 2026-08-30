#!/usr/bin/env python3
"""
Unified JWST / HST alignment CLI.

Modes
-----
``visit`` (default)
    Self-align one instrument's frames under ``reduction/`` (Gaia / visit
    mosaics). Typically ``--instrument NIRCAM`` for JWST. With
    ``--telescope hst``, aligns ``reduction/raw`` frames (flc/flt/c0m) to
    Gaia via JHAT into ``reduction/jhat``.

``hst``
    Alias for ``--telescope hst --mode visit`` (HST Gaia JHAT batch).

``reference``
    Overlap-match science frames to ``reference/`` coadds and align
    (REFERENCE -> optional MIRI_REL); typically ``--instrument MIRI``.

``pair``
    Align one ``--image`` to a photometry catalog from ``--ref``
    (optional ``--photfile``). Selected automatically when ``--ref`` and
    ``--image`` are both set.

Multi-instrument orchestration
------------------------------
``--instruments NIRCAM MIRI ACS WFC3`` (or ``ALL``) runs:
NIRCam visit -> HST visit -> intermediate NIRCam mosaic -> MIRI reference.
``--instruments NIRCAM`` keeps the legacy visit-only path.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import socket
from pathlib import Path

import numpy as np

from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    add_filters,
    add_instruments_arg,
    add_pair_align_args,
    add_sky_coord_args,
    configure_logging_from_args,
    as_path,
    create_parser as build_parser,
    dataset_label,
    default_instruments_for_telescope,
    expand_mission_instruments,
    parse_filter_list,
    parse_instruments,
    resolve_instruments_with_telescope,
    resolve_reduction_dir,
    sync_legacy_ncores,
)
from st123.utils.helpers import create_filter_table, input_list
from st123.datamodels import as_datamodel
from st123.utils.logging import shutdown_logging
from st123.utils.settings import DEFAULT_PAIR_OUTDIR

logger = logging.getLogger(__name__)

# Machine-local fallback for ``align --mode reference`` when ``--base-dir`` is
# omitted (not a portable package default; override via CLI).
DEFAULT_REFERENCE_DATA_DIR = Path('/data/rwisenbaker/jwst_data/M51')

# Default multi-mission set for ``--instruments ALL``.
ALL_ALIGN_INSTRUMENTS: tuple[str, ...] = (
    'NIRCAM',
    'MIRI',
    'ACS',
    'WFC3',
    'WFPC2',
)
_JWST_ALIGN_INSTRUMENTS = frozenset({'nircam', 'nrc', 'miri'})
_HST_ALIGN_INSTRUMENTS = frozenset({'acs', 'wfc3', 'wfpc2', 'wfc'})


def create_parser(default_data_dir: Path | None = None):
    if default_data_dir is None:
        default_data_dir = DEFAULT_REFERENCE_DATA_DIR
    parser = build_parser(
        description=(
            'Align JWST or HST imaging. Modes: visit (Gaia / visit mosaics '
            'under reduction/; HST with --telescope hst), hst (HST Gaia JHAT '
            'batch), reference (science -> reference/ coadds), or pair '
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
            'Project root (.../<object>), or output directory in pair mode. '
            'Visit mode uses <base-dir>/reduction; HST visit writes to '
            'reduction/jhat from reduction/raw. Reference mode expects '
            f'JWST/<INSTRUMENT>/... and reference/ (default when omitted: '
            f'{default_data_dir}). Pair-mode default when omitted: '
            f'{DEFAULT_PAIR_OUTDIR}. '
            'Aliases: --workdir, --basedir, --data-dir, --download-dir, --outdir.'
        ),
    )
    parser.add_argument(
        '--telescope',
        choices=('jwst', 'hst'),
        default=None,
        help=(
            'Mission shorthand, equivalent to the default --instruments set: '
            'hst -> ACS WFC3 WFPC2; jwst -> NIRCAM MIRI. When omitted, visit '
            'mode defaults to a single NIRCam run (or use --instruments / '
            '--instrument). --mode hst implies --telescope hst.'
        ),
    )
    parser.add_argument(
        '--mode',
        choices=('visit', 'hst', 'reference', 'pair'),
        default='visit',
        help=(
            'visit: self-align one instrument under reduction/ (default; '
            'HST when --telescope hst). '
            'hst: same as --telescope hst --mode visit. '
            'reference: align --instrument frames to reference/ coadds. '
            'pair: align --image to --ref (auto if both flags are set).'
        ),
    )
    add_instruments_arg(
        parser,
        help=(
            'One or more instruments to align. Mission aliases: hst (= ACS '
            'WFC3 WFPC2), jwst (= NIRCAM MIRI), all (= NIRCAM MIRI ACS WFC3 '
            'WFPC2). Multi-instrument lists run an orchestrated pipeline: '
            'NIRCam visit -> HST visit -> intermediate NIRCam mosaic -> MIRI '
            'reference. NIRCAM alone keeps the legacy visit-only path. '
            'Overrides --telescope defaults when set. Alias: --instrument.'
        ),
    )
    add_sky_coord_args(parser, required=False)
    parser.add_argument(
        '--force-miri',
        action='store_true',
        help=(
            'Process MIRI even when the field has no NIRCam. By default, '
            'MIRI-only JWST fields skip JWST align stages when NIRCAM+MIRI '
            'are requested (use --instruments MIRI to force MIRI alone).'
        ),
    )
    parser.add_argument(
        '--skip-intermediate-mosaic',
        action='store_true',
        help=(
            'Orchestrator only: if any reference coadds already exist, skip '
            'the intermediate NIRCam mosaic before MIRI reference alignment. '
            'Implied when --existing-box points at a stamp that already has '
            'coadd*i2d.fits.'
        ),
    )
    parser.add_argument(
        '--nmax',
        type=int,
        default=150,
        help=(
            'Orchestrator only: --nmax forwarded to the intermediate NIRCam '
            'mosaic (default: 150).'
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
        help='Disable MIRI->MIRI relative fallback when REFERENCE alignment fails.',
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


def resolve_align_instruments(
    instruments_raw: list[str] | None,
    *,
    instrument: str | None = None,
) -> list[str] | None:
    """
    Normalize ``--instruments`` / ``--instrument`` into an upper-case list.

    Mission aliases ``hst`` / ``jwst`` / ``all`` are expanded. Returns ``None``
    when neither flag was set (caller uses mode defaults).
    """
    if instruments_raw:
        return expand_mission_instruments(parse_instruments(instruments_raw))
    if instrument and str(instrument).strip():
        return expand_mission_instruments([str(instrument).strip()])
    return None


def needs_orchestration(instruments: list[str] | None) -> bool:
    """True when *instruments* requires the multi-stage align pipeline."""
    if not instruments:
        return False
    upper = {str(i).upper() for i in instruments}
    upper.discard('NRC')
    if 'NRC' in {str(i).upper() for i in instruments}:
        upper.add('NIRCAM')
    if upper <= {'NIRCAM'}:
        return False
    return True


def partition_align_instruments(
    instruments: list[str],
) -> tuple[list[str], list[str]]:
    """Split into (jwst_instruments, hst_instruments), preserving order."""
    jwst: list[str] = []
    hst: list[str] = []
    unknown: list[str] = []
    for inst in instruments:
        key = str(inst).split('_')[0].strip().lower()
        name = str(inst).strip().upper()
        if key == 'nrc':
            name = 'NIRCAM'
            key = 'nircam'
        if key in _JWST_ALIGN_INSTRUMENTS:
            if name not in jwst:
                jwst.append(name)
        elif key in _HST_ALIGN_INSTRUMENTS:
            if name not in hst:
                hst.append(name)
        else:
            unknown.append(str(inst))
    if unknown:
        raise ValueError(
            'Unrecognized align instrument(s): '
            f'{unknown}; expected NIRCAM, MIRI, ACS, WFC3, WFPC2 (or ALL)'
        )
    return jwst, hst


def _resolve_mode(args: argparse.Namespace) -> str:
    """
    Resolve alignment mode.

    ``--ref`` / ``--image`` select pair mode automatically. Explicit
    ``--mode pair`` requires both paths. ``--mode hst`` forces HST visit.
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
        if args.mode == 'hst':
            raise ValueError(
                'do not combine --ref/--image with --mode hst; '
                'use HST visit without --ref/--image'
            )
        return 'pair'
    if args.mode == 'hst':
        args.telescope = 'hst'
        return 'visit'
    return args.mode


def run_hst_visit_alignment(
    *,
    base_dir: str | Path,
    verbose: bool = False,
    ncores: int = 4,
    skip_cosmic: bool = False,
    skip_prelim_l3: bool = False,
    instruments: list[str] | None = None,
) -> int:
    """Align HST ``reduction/raw`` via best level-3 Gaia reference into ``jhat``.

    Steps:
    1. Optional astroscrappy CR clean + DQ/c1m flagging on raw frames.
    2. Build preliminary per-filter level-3 coadds from raw.
    3. Pick the L3 with the most Gaia stars on illuminated, detectable pixels.
    4. Gaia-anchor that L3 with gaia_simple, then build a dense L3 detection
       phot catalog (DAOStarFinder) for science-frame JHAT.
    5. Align all science frames to that catalog (JHAT).

    Parameters
    ----------
    instruments : list of str or None, optional
        Restrict to these HST instruments (e.g. ``['ACS', 'WFC3']``).
    """
    from st123.stages.alignment.hst_jhat import (
        HST_JHAT_GAIA_L3_PARAMS,
        HST_JHAT_L3REF_PARAMS,
        _jhat_hst_output_path,
        align_hst_raw_dir,
        write_hst_alignment_summary,
    )
    from st123.stages.alignment.hst_reference import (
        ensure_preliminary_level3s,
        pick_best_level3,
    )

    if base_dir is None:
        raise ValueError('--base-dir is required for --telescope hst')

    from st123.scripts.utils.options import resolve_jhat_dir

    work_dir = Path(resolve_reduction_dir(base_dir))
    raw_dir = work_dir / 'raw'
    jhat_dir = resolve_jhat_dir(work_dir, 'hst', create=True)
    prelim_dir = work_dir / 'reference_prelim'
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f'HST raw directory not found: {raw_dir} '
            '(expected reduction/raw under --base-dir)'
        )

    if verbose:
        logger.info('Dataset: %s', dataset_label(base_dir))
        logger.info('Mode: visit (HST JHAT <- best level-3 Gaia ref)')
        logger.info('Raw dir: %s', raw_dir)
        logger.info('JHAT outdir: %s', jhat_dir)
        if instruments:
            logger.info('HST instruments: %s', ', '.join(instruments))

    jhat_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        'HST visit align plan (L2 pipeline, not L3-only): '
        '1) CR clean  2) prelim L3 coadds  3) score L3 vs Gaia  '
        '4) Gaia-anchor best L3  5) JHAT every raw frame  '
        '6) within-filter WCS harmonize. '
        'Cross-filter coadd<->coadd lives in mosaic, not align.'
    )

    if not skip_cosmic:
        try:
            from st123.stages.photometry.cosmic import clean_raw_directory

            logger.info('[1/6] astroscrappy CR clean on raw frames under %s', raw_dir)
            cr_results = clean_raw_directory(raw_dir, add_crmask=True)
            n_cr = sum(int(r.get('n_cr_pixels') or 0) for r in cr_results)
            logger.info(
                '[1/6] done: %d frame(s), %d CR pixels flagged',
                len(cr_results),
                n_cr,
            )
        except Exception as exc:
            logger.warning('astroscrappy step failed (continuing): %s', exc)
    else:
        logger.info('[1/6] skipped CR clean (--skip-cosmic / skip_cosmic=True)')

    ref_phot = None
    best_l3 = None
    if not skip_prelim_l3:
        try:
            logger.info(
                '[2/6] preliminary L3 coadds -> %s (reuse existing unless force)',
                prelim_dir,
            )
            ensure_preliminary_level3s(
                raw_dir,
                prelim_dir,
                num_cores=max(1, int(ncores)),
                force=False,
                instruments=instruments,
            )
            # Score all L3 candidates once (Gaia queries are slow).
            from st123.stages.alignment.hst_reference import list_level3_products
            from st123.utils.helpers import get_instrument

            l3_candidates = list_level3_products(prelim_dir)
            ref_dir = work_dir / 'reference'
            if ref_dir.is_dir():
                l3_candidates.extend(list_level3_products(ref_dir))
            if instruments:
                allow = {
                    str(i).split('_')[0].strip().lower()
                    for i in instruments
                    if str(i).strip()
                }

                def _l3_allowed(path: Path) -> bool:
                    try:
                        inst = get_instrument(path).split('_')[0].lower()
                    except Exception:
                        # Fallback: coadd_{inst}_{filt}_*.fits
                        name = path.name.lower()
                        return any(f'coadd_{a}_' in name for a in allow)
                    return inst in allow

                l3_candidates = [p for p in l3_candidates if _l3_allowed(p)]
            logger.info(
                '[3/6] scoring %d L3 candidate(s) vs Gaia '
                '(shared Vizier cache under reduction/gaia/)',
                len(l3_candidates),
            )
            if l3_candidates:
                from st123.stages.alignment.gaia_catalog import ensure_gaia_catalog

                gaia_cache = work_dir / 'gaia'
                # Include science frames so the cone covers requested footprints;
                # min radius floor handles sparse / small-FOV visits.
                gaia_images = list(l3_candidates)
                for pat in ('*flc.fits', '*flt.fits', '*c0m.fits'):
                    for path in sorted(raw_dir.glob(pat)):
                        if instruments:
                            try:
                                inst = get_instrument(path).split('_')[0].lower()
                            except Exception:
                                continue
                            if inst not in {
                                str(i).split('_')[0].strip().lower()
                                for i in instruments
                            }:
                                continue
                        gaia_images.append(path)
                try:
                    ensure_gaia_catalog(
                        gaia_images,
                        telescope='hst',
                        cache_dir=gaia_cache,
                        backend='vizier',
                    )
                except Exception as exc:
                    logger.warning(
                        'Field Gaia cache fetch failed (%s); '
                        'per-image queries will retry',
                        exc,
                    )
            best_l3, scores = pick_best_level3(candidates=l3_candidates)
            if best_l3 is not None:
                # Keep L3 JHAT products out of reduction/jhat science globs.
                l3_outdir = jhat_dir / 'l3_ref'
                l3_outdir.mkdir(parents=True, exist_ok=True)
                # Prefer a Gaia-aligned L3 *_jhat.fits; only reuse phot that
                # sits beside that product (sky coords on the Gaia frame).
                expected_l3_jhat = _jhat_hst_output_path(Path(best_l3), l3_outdir)
                l3_jhat_path = (
                    expected_l3_jhat if expected_l3_jhat.is_file() else None
                )
                from st123.stages.alignment.gaia_simple import (
                    align_image_to_gaia_simple,
                    rewrite_phot_radec,
                )

                # Fast L3 absolute anchor: gaia_simple CRPIX, then a *dense*
                # DAOStarFinder phot catalog on that Gaia-tied L3 (not Gaia-only;
                # sparse fields like NGC 3913 have too few Gaia for WFPC2 JHAT).
                import shutil

                from st123.stages.alignment.hst_reference import ensure_l3_science_refcat

                if l3_jhat_path is None:
                    l3_jhat_path = expected_l3_jhat
                    shutil.copy2(best_l3, l3_jhat_path)
                    logger.info(
                        '[4/6] Gaia-anchoring L3 with gaia_simple (cached Vizier) '
                        '-> %s',
                        l3_jhat_path.name,
                    )
                    align_image_to_gaia_simple(
                        l3_jhat_path,
                        l3_jhat_path,
                        telescope='hst',
                    )
                else:
                    logger.info(
                        '[4/6] reusing existing Gaia L3 product %s',
                        l3_jhat_path.name,
                    )
                    gaia_ok = False
                    try:
                        with as_datamodel(l3_jhat_path).open(memmap=True) as _h:
                            gaia_ok = bool(_h[0].header.get('GAIASIMP'))
                    except Exception:
                        gaia_ok = False
                    if not gaia_ok:
                        logger.info(
                            '[4/6] applying gaia_simple CRPIX shift to %s',
                            Path(l3_jhat_path).name,
                        )
                        align_image_to_gaia_simple(
                            l3_jhat_path,
                            l3_jhat_path,
                            telescope='hst',
                        )

                phot_name = f'{Path(best_l3).stem}.phot.txt'
                ref_phot = l3_outdir / phot_name
                try:
                    ensure_l3_science_refcat(l3_jhat_path, ref_phot)
                    # Keep ra/dec on the Gaia-aligned L3 WCS (x/y fixed).
                    rewrite_phot_radec(ref_phot, l3_jhat_path)
                except Exception as exc:
                    logger.warning(
                        'L3 detection refcat failed (%s); trying Gaia-only',
                        exc,
                    )
                    try:
                        from st123.stages.alignment.gaia_catalog import write_gaia_refcat

                        write_gaia_refcat(
                            l3_jhat_path,
                            ref_phot,
                            telescope='hst',
                            cache_dir=work_dir / 'gaia',
                        )
                    except Exception as exc2:
                        logger.warning('Gaia-only refcat also failed: %s', exc2)
                        ref_phot = None
                if ref_phot is not None and ref_phot.is_file() and ref_phot.stat().st_size > 0:
                    logger.info(
                        '[4/6] done: science frames will use L3 detection refcat %s',
                        ref_phot,
                    )
                else:
                    ref_phot = None
                    logger.warning(
                        'No L3 science refcat; falling back to per-frame Gaia'
                    )
            else:
                logger.warning(
                    'No level-3 products for Gaia reference; '
                    'falling back to per-frame Gaia'
                )
        except Exception as exc:
            logger.warning(
                'Level-3 reference selection failed (%s); '
                'falling back to per-frame Gaia',
                exc,
            )
    else:
        logger.info('[2-4/6] skipped prelim L3 / Gaia anchor (skip_prelim_l3=True)')

    sci_params = (
        dict(HST_JHAT_L3REF_PARAMS)
        if ref_phot is not None
        else dict(HST_JHAT_GAIA_L3_PARAMS)
    )
    if ref_phot is not None:
        logger.info(
            '[5/6] JHAT every raw science frame -> %s (refcat=%s); '
            'each frame can take minutes; JHAT stdout is captured',
            jhat_dir,
            Path(ref_phot).name,
        )
    else:
        logger.warning(
            '[5/6] JHAT every raw science frame with per-frame Gaia '
            '(relaxed cuts; prefer fixing L3 phot)'
        )
    results = align_hst_raw_dir(
        raw_dir,
        jhat_dir,
        soft_fail=True,
        gaia=ref_phot is None,
        photfilename=ref_phot,
        verbose=verbose,
        jhat_params=sci_params,
        instruments=instruments,
        workers=max(1, int(ncores)),
    )
    # Attach reference metadata to the summary.
    for row in results:
        row['reference_level3'] = str(best_l3) if best_l3 is not None else None
        row['reference_phot'] = str(ref_phot) if ref_phot is not None else None

    summary_path = write_hst_alignment_summary(
        results, jhat_dir / 'alignment_summary.json'
    )
    n_ok = sum(1 for r in results if r.get('status') == 'ok')
    n_fail = sum(1 for r in results if r.get('status') == 'failed')
    logger.info(
        'HST JHAT summary: %d ok, %d failed -> %s',
        n_ok,
        n_fail,
        summary_path,
    )
    for row in results:
        if row.get('status') == 'ok':
            logger.info('  OK  %s -> %s', Path(row['path']).name, row.get('outpath'))
        else:
            err = str(row.get('error') or '')
            # Sparse / empty WFPC2 chips fail JHAT with soft_fail=True -- expected.
            soft = (
                'initial cut' in err.lower()
                or err.startswith('KeyError')
                or 'Only ' in err and 'objects pass' in err
            )
            log_fn = logger.warning if soft else logger.error
            log_fn(
                '  FAIL %s: %s',
                Path(row['path']).name,
                row.get('error'),
            )
    # soft_fail=True on the batch: sparse / short WFPC2 frames may fail JHAT
    # without invalidating the visit. Only abort when nothing usable aligned.
    if n_ok == 0:
        logger.error('HST visit align produced no JHAT products')
        return 1
    if n_fail:
        logger.warning(
            'HST visit align continuing with %d/%d failed frame(s); '
            'usable JHAT products: %d',
            n_fail,
            n_ok + n_fail,
            n_ok,
        )
    return 0


def _resolve_existing_box_dir(base_dir: str | Path, existing_box: str | Path) -> Path:
    """Resolve ``--existing-box`` from a project root or reduction dir."""
    from st123.stages.mosaic.mosaic import resolve_existing_box_dir

    return resolve_existing_box_dir(base_dir, existing_box)


def _coadds_in_existing_box(base_dir: str | Path, existing_box: str | Path) -> list[str]:
    """``coadd*i2d.fits`` under ``--existing-box``, or raise if the box is missing."""
    from st123.stages.mosaic.mosaic import box_coadd_i2d_paths

    return box_coadd_i2d_paths(_resolve_existing_box_dir(base_dir, existing_box))


def _existing_reference_coadds(base_dir: str | Path) -> list[str]:
    """Return existing mosaic coadds under the project / reduction tree."""
    from st123.stages.alignment.align import discover_ref_images

    root = Path(base_dir).expanduser()
    found = list(discover_ref_images(root))
    if found:
        return found
    try:
        work = Path(resolve_reduction_dir(root))
    except Exception:
        return found
    if work.resolve() != root.resolve():
        found.extend(discover_ref_images(work))
        # Also search project root when base_dir was the reduction dir.
        parent = work.parent
        if parent.is_dir():
            found.extend(discover_ref_images(parent))
    # Deduplicate
    return sorted({str(Path(p).resolve()) for p in found})


def run_intermediate_nircam_mosaic(
    *,
    base_dir: str | Path,
    ncores: int = 4,
    nmax: int = 150,
    filters: list[str] | None = None,
    skip_if_exists: bool = False,
    verbose: bool = False,
) -> int:
    """Build intermediate NIRCam coadds for MIRI reference alignment."""
    existing = _existing_reference_coadds(base_dir)
    if skip_if_exists and existing:
        logger.info(
            'Skipping intermediate mosaic (%d coadd(s) already present)',
            len(existing),
        )
        return 0

    # Reload options before importing mosaic so a long-running align process
    # that started under an older checkout still sees new mosaic helpers
    # (avoids ImportError after mid-run package edits).
    import importlib

    import st123.scripts.utils.options as _options_mod

    importlib.reload(_options_mod)
    from st123.scripts import mosaic as mosaic_script

    importlib.reload(mosaic_script)

    argv = [
        '--base-dir',
        str(base_dir),
        '--ncores',
        str(max(1, int(ncores))),
        '--nmax',
        str(int(nmax)),
        '--footprint-weights',
        'auto',
    ]
    if filters:
        argv.extend(['--filters', ','.join(filters)])
    if verbose:
        argv.append('-v')
    logger.info('Intermediate NIRCam mosaic: mosaic %s', ' '.join(argv))
    return int(mosaic_script.main(argv))


def run_orchestrated_alignment(
    args: argparse.Namespace,
    instruments: list[str],
) -> int:
    """
    Multi-mission align: NIRCam visit -> HST visit -> intermediate mosaic -> MIRI ref.

    Stages whose instruments were not requested are skipped. Exit code is
    nonzero if any required stage fails.

    When NIRCAM+MIRI are requested but the field has MIRI and no NIRCam on
    disk, JWST stages are skipped (rc=0) unless ``--force-miri`` / explicit
    ``--instruments MIRI``.
    """
    from st123.utils.jwst_coverage import (
        count_jwst_frames_on_disk,
        should_skip_miri_only_jwst,
    )

    try:
        jwst_inst, hst_inst = partition_align_instruments(instruments)
    except ValueError as exc:
        logger.error('%s', exc)
        return 2

    if args.base_dir is None:
        logger.error('--base-dir is required for orchestrated multi-instrument align')
        return 2

    want_nircam = any(i.upper() in ('NIRCAM', 'NRC') for i in jwst_inst)
    want_miri = any(i.upper() == 'MIRI' for i in jwst_inst)
    ncores = int(getattr(args, 'ncores', 1) or 1)
    verbose = bool(getattr(args, 'verbose', False))
    stage_rcs: list[tuple[str, int]] = []
    force_miri = bool(getattr(args, 'force_miri', False))

    n_nrc, n_miri = count_jwst_frames_on_disk(args.base_dir)
    skip_jwst = should_skip_miri_only_jwst(
        n_nrc > 0,
        n_miri > 0,
        jwst_inst or instruments,
        force_miri=force_miri,
    )
    if skip_jwst:
        logger.warning(
            'MIRI-only JWST field (NIRCam frames=%d, MIRI frames=%d); '
            'skipping JWST align stages. Pass --force-miri (or '
            '--instruments MIRI) to process MIRI.',
            n_nrc,
            n_miri,
        )
        want_nircam = False
        want_miri = False

    logger.info(
        'Orchestrated align: instruments=%s '
        '(NIRCam visit -> HST visit -> intermediate mosaic -> MIRI reference)',
        ', '.join(instruments),
    )

    # 1) NIRCam visit
    if want_nircam:
        logger.info('=== Stage 1/4: NIRCam visit alignment ===')
        try:
            rc = run_visit_alignment(
                base_dir=args.base_dir,
                instrument='NIRCAM',
                ncores=ncores,
                verbose=verbose,
            )
        except (ValueError, FileNotFoundError) as exc:
            logger.error('NIRCam visit failed: %s', exc)
            rc = 1
        stage_rcs.append(('nircam_visit', rc))
        if rc != 0:
            logger.error('NIRCam visit stage failed (rc=%d)', rc)
    else:
        logger.info('=== Stage 1/4: skip NIRCam (not requested) ===')

    # 2) HST visit (ACS/WFC3/...)
    if hst_inst:
        logger.info(
            '=== Stage 2/4: HST visit alignment (%s) ===',
            ', '.join(hst_inst),
        )
        try:
            rc = run_hst_visit_alignment(
                base_dir=args.base_dir,
                verbose=verbose,
                ncores=ncores,
                instruments=hst_inst,
            )
        except (ValueError, FileNotFoundError) as exc:
            logger.error('HST visit failed: %s', exc)
            rc = 1
        stage_rcs.append(('hst_visit', rc))
        if rc != 0:
            logger.error('HST visit stage failed (rc=%d)', rc)
    else:
        logger.info('=== Stage 2/4: skip HST (not requested) ===')

    # 3) Intermediate NIRCam mosaic (needed for MIRI reference coadds)
    if want_miri:
        existing_box = getattr(args, 'existing_box', None)
        existing = _existing_reference_coadds(args.base_dir)
        skip_mosaic = bool(getattr(args, 'skip_intermediate_mosaic', False))
        box_skip_rc: int | None = None
        if existing_box:
            try:
                box_coadds = _coadds_in_existing_box(args.base_dir, existing_box)
            except FileNotFoundError as exc:
                logger.error('%s', exc)
                box_skip_rc = 1
            else:
                if box_coadds:
                    logger.info(
                        '=== Stage 3/4: skip intermediate mosaic '
                        '(--existing-box %s has %d coadd(s)) ===',
                        existing_box,
                        len(box_coadds),
                    )
                    box_skip_rc = 0
                else:
                    logger.error(
                        '--existing-box %s has no coadd*i2d.fits; run '
                        'mosaic --instruments NIRCAM --existing-box %s first',
                        existing_box,
                        existing_box,
                    )
                    box_skip_rc = 1
        if box_skip_rc is not None:
            stage_rcs.append(('intermediate_mosaic', box_skip_rc))
        elif not want_nircam and not existing:
            logger.error(
                'MIRI reference alignment requires NIRCam in --instruments '
                '(to build coadds) or existing reference/ coadd*i2d.fits'
            )
            stage_rcs.append(('intermediate_mosaic', 1))
        else:
            logger.info('=== Stage 3/4: intermediate NIRCam mosaic ===')
            mosaic_filters = parse_filter_list(getattr(args, 'filters', None))
            rc = run_intermediate_nircam_mosaic(
                base_dir=args.base_dir,
                ncores=ncores,
                nmax=int(getattr(args, 'nmax', 150) or 150),
                filters=mosaic_filters,
                skip_if_exists=skip_mosaic,
                verbose=verbose,
            )
            stage_rcs.append(('intermediate_mosaic', rc))
            if rc != 0:
                logger.error('Intermediate mosaic stage failed (rc=%d)', rc)
            elif not _existing_reference_coadds(args.base_dir):
                logger.error(
                    'Intermediate mosaic finished but no coadd*i2d.fits found'
                )
                stage_rcs[-1] = ('intermediate_mosaic', 1)
    else:
        logger.info('=== Stage 3/4: skip intermediate mosaic (MIRI not requested) ===')

    # 4) MIRI reference
    mosaic_failed = any(
        name == 'intermediate_mosaic' and rc != 0 for name, rc in stage_rcs
    )
    if want_miri:
        if mosaic_failed:
            logger.error('Skipping MIRI reference (intermediate mosaic stage failed)')
            stage_rcs.append(('miri_reference', 1))
        elif not _existing_reference_coadds(args.base_dir):
            logger.error(
                'Cannot run MIRI reference: no reference coadds under %s',
                args.base_dir,
            )
            stage_rcs.append(('miri_reference', 1))
        else:
            logger.info('=== Stage 4/4: MIRI reference alignment ===')
            miri_args = argparse.Namespace(**vars(args))
            miri_args.mode = 'reference'
            miri_args.instrument = 'MIRI'
            miri_args.telescope = 'jwst'
            # --filters on the orchestrator is forwarded to the intermediate
            # NIRCam mosaic only; MIRI reference processes all MIRI bands.
            miri_args.filters = None
            rc = run_reference_alignment(miri_args)
            stage_rcs.append(('miri_reference', rc))
            if rc != 0:
                logger.error('MIRI reference stage failed (rc=%d)', rc)
    else:
        logger.info('=== Stage 4/4: skip MIRI (not requested) ===')

    logger.info('Orchestrated align summary:')
    for name, rc in stage_rcs:
        logger.info('  %s: rc=%d', name, rc)
    if not stage_rcs:
        if skip_jwst and not hst_inst:
            logger.info('No alignment stages run (MIRI-only JWST skipped)')
            return 0
        logger.error('No alignment stages were requested')
        return 1
    return 0 if all(rc == 0 for _, rc in stage_rcs) else 1


def run_pair_alignment(args: argparse.Namespace) -> int:
    """Align one ``--image`` to a catalog built from ``--ref``."""
    from st123.stages.alignment.align import run_alignment

    ref = str(Path(args.ref).expanduser())
    align_image = str(Path(args.image).expanduser())
    for path, label in ((ref, '--ref'), (align_image, '--image')):
        if not os.path.exists(path):
            logger.error('%s file not found: %s', label, path)
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
        logger.error('alignment failed: %s', exc)
        if getattr(args, 'verbose', False):
            logger.debug('pair alignment traceback', exc_info=True)
        return 1

    logger.info('Guess offset (x, y): %s', guess_offset)
    logger.info('Done. Products in %s', out)
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
    """Self-align instrument frames under ``reduction/`` (Gaia / visit mosaics).

    Seeds the cascade from the best absolute hub visit (deep F200W/F150W
    preferred over largest footprint), gates that Gaia mosaic, re-aligns
    soft-fail / wrong-island JHAT products to the hub catalog, then applies a
    capped JWST-only CRVAL retie for small residuals.

    Returns
    -------
    int
        ``0`` on success; ``1`` if any pooled visit JHAT worker failed
        (same contract as reference mode).
    """
    # Deferred: pulls JWST Image3 / JHAT / CRDS (~tens of seconds).
    import shapely

    from st123.stages.alignment.align import (
        HUB_ABS_RETIE_MAX_APPLY_ARCSEC,
        HUB_ABS_RETIE_MIN_PEAK,
        HUB_ABS_RETIE_TOL_ARCSEC,
        HUB_ABS_SEARCH_ARCSEC,
        HUB_GAIA_MAX_MAS,
        align_to_mosaic,
        create_alignment_mosaic,
        create_dirs,
        fix_phot,
        get_input_images,
        get_visit_geoms,
        mosaic_abs_quality,
        pick_visit,
        rank_visits_as_abs_hubs,
        realign_jwst_jhats_to_catalog,
        retie_jwst_jhat_to_abs_ref,
        select_jwst_jhats_for_abs_redo,
        update_refcat,
        visit_filter_dict,
    )

    from st123.scripts.utils.options import resolve_jhat_dir

    work_dir_path = Path(resolve_reduction_dir(base_dir))
    work_dir = str(work_dir_path)
    jhat_dir = str(resolve_jhat_dir(work_dir_path, 'jwst', create=True))
    if verbose:
        logger.info('Dataset: %s', dataset_label(base_dir))
        logger.info('Mode: visit')
        logger.info('Instrument: %s', instrument)
        logger.info('Reduction workdir: %s', work_dir)
        logger.info('JHAT outdir: %s', jhat_dir)

    create_dirs(work_dir)
    patterns = _visit_patterns(instrument)
    logger.info(
        'Scanning raw/ for %s frames (patterns: %s)...',
        instrument,
        ', '.join(patterns),
    )
    input_images = get_input_images(pattern=patterns, workdir=work_dir)
    logger.info(
        'Found %d %s frame(s); reading headers / building visit table...',
        len(input_images),
        instrument,
    )
    table = input_list(input_images)
    ngroups = np.unique(table['group'])
    visit_filter = visit_filter_dict(table)
    n_failures = 0
    logger.info(
        'Visit plan: %d image(s), %d group(s), %d visit(s)',
        len(table),
        len(ngroups),
        len(np.unique(table['visit'])),
    )

    for group_id in ngroups:
        combined_photfile = os.path.join(
            work_dir, 'align', f'group_{group_id}', 'reference_catalog.txt'
        )
        group_table = table[table['group'] == group_id]
        group_visits = np.unique(group_table['visit'])
        visit_geoms = get_visit_geoms(group_table)
        align_polygon = None
        hub_order = rank_visits_as_abs_hubs(group_table, visit_filter, visit_geoms)
        hub_mosaic = None
        hub_locked = False
        logger.info(
            'Group %s: %d image(s), %d visit(s); abs-hub order=%s',
            group_id,
            len(group_table),
            len(group_visits),
            ', '.join(str(v) for v in hub_order),
        )

        def _process_visit(
            visit_id,
            *,
            align_to: str,
            overlap_frac: float,
            is_hub_seed: bool,
        ):
            nonlocal n_failures
            visit_table = group_table[group_table['visit'] == visit_id]
            filters = np.unique(visit_table['filter']).value
            filter_table = create_filter_table(visit_table, filters)
            align_filter = visit_filter[visit_id]
            visit_outdir = os.path.join(
                work_dir, 'align', f'group_{group_id}', f'visit_{visit_id}'
            )
            ref_label = (
                'Gaia' if align_to == 'gaia' else 'prior-visit / hub catalog'
            )
            logger.info(
                'Group %s visit %s: %d frame(s), align_filter=%s -> %s '
                '(overlap=%.2f%s); building visit mosaic under %s',
                group_id,
                visit_id,
                len(visit_table),
                align_filter,
                ref_label,
                float(overlap_frac) if overlap_frac is not None else -1.0,
                '; HUB SEED' if is_hub_seed else '',
                visit_outdir,
            )
            nbright = 800
            if (
                align_to != 'gaia'
                and overlap_frac is not None
                and float(overlap_frac) < 0.3
            ):
                nbright = 50000
            mosaic_name, guess_offset, mosaic_fail = create_alignment_mosaic(
                filter_table,
                visit_outdir,
                align_filter=align_filter,
                align_to=align_to,
                ncores=ncores,
                Nbright=nbright,
            )
            n_failures += int(mosaic_fail)
            logger.info(
                'Group %s visit %s mosaic done: %s (failures=%d, offset=%s)',
                group_id,
                visit_id,
                mosaic_name,
                int(mosaic_fail),
                guess_offset,
            )
            return mosaic_name, guess_offset

        def _finalize_visit(visit_id, mosaic_name, guess_offset) -> None:
            nonlocal n_failures, align_polygon
            mosaic_photfile = fix_phot(mosaic_name)
            _ = update_refcat(
                mosaic_name,
                mosaic_photfile,
                out_refcat=combined_photfile,
                align_pgon=align_polygon,
            )
            visit_table = group_table[group_table['visit'] == visit_id]
            filters = np.unique(visit_table['filter']).value
            filter_table = create_filter_table(visit_table, filters)
            logger.info('Mosaic photfile: %s', mosaic_photfile)
            # Batch every filter into one align_to_mosaic pool so L2 JHAT
            # workers stay saturated across filters (not filter-serial).
            all_images: list[str] = []
            for _filt, filt_table in filter_table.items():
                logger.info(
                    'Queueing %d %s frame(s) in visit %s for mosaic JHAT...',
                    len(filt_table),
                    _filt,
                    visit_id,
                )
                all_images.extend(str(row['image']) for row in filt_table)
            if all_images:
                logger.info(
                    'Aligning %d frame(s) across %d filter(s) in visit %s '
                    'to mosaic catalog (ncores=%d)...',
                    len(all_images),
                    len(filter_table),
                    visit_id,
                    ncores,
                )
                n_failures += int(
                    align_to_mosaic(
                        mosaic_photfile,
                        all_images,
                        jhat_dir,
                        guess_offset=guess_offset,
                        verbose=verbose,
                        ncores=ncores,
                    )
                )
            if align_polygon is None:
                align_polygon = visit_geoms[visit_id]
            else:
                align_polygon = shapely.unary_union(
                    [align_polygon, visit_geoms[visit_id]]
                )
            visit_geoms.pop(visit_id, None)

        # Absolute hub seed (Gaia): try candidates best-first.
        for hub_id in hub_order:
            if hub_id not in visit_geoms:
                continue
            mosaic_name, guess_offset = _process_visit(
                hub_id,
                align_to='gaia',
                overlap_frac=float(visit_geoms[hub_id].area),
                is_hub_seed=True,
            )
            quality = mosaic_abs_quality(mosaic_name, max_mas=HUB_GAIA_MAX_MAS)
            if quality['ok']:
                logger.info(
                    'Abs hub accepted: visit=%s mosaic=%s disp=%.1f mas ncal=%d '
                    '(limit=%.1f mas)',
                    hub_id,
                    Path(mosaic_name).name,
                    float(quality['dispersion_mas'] or -1.0),
                    int(quality['n_calibrators']),
                    HUB_GAIA_MAX_MAS,
                )
                hub_mosaic = mosaic_name
                hub_locked = True
                _finalize_visit(hub_id, mosaic_name, guess_offset)
                break
            logger.warning(
                'Abs hub candidate rejected: visit=%s disp=%s mas ncal=%d '
                '(need ncal>0 and disp<=%.1f mas); trying next hub',
                hub_id,
                quality.get('dispersion_mas'),
                int(quality.get('n_calibrators') or 0),
                HUB_GAIA_MAX_MAS,
            )
            # Leave visit in visit_geoms so it can align to a later hub catalog.

        if not hub_locked:
            if not visit_geoms:
                logger.error(
                    'Group %s: no visits left after hub failures', group_id
                )
                n_failures += 1
                continue
            fallback_id, overlap_frac = pick_visit(
                None, visit_geoms, visit_filter, hub_order=hub_order
            )
            logger.warning(
                'No hub passed Gaia gate; proceeding with best-effort seed '
                'visit=%s',
                fallback_id,
            )
            mosaic_name, guess_offset = _process_visit(
                fallback_id,
                align_to='gaia',
                overlap_frac=float(overlap_frac),
                is_hub_seed=True,
            )
            hub_mosaic = mosaic_name
            hub_locked = True
            _finalize_visit(fallback_id, mosaic_name, guess_offset)

        # Remaining visits relative to hub catalog.
        while visit_geoms:
            visit_id, overlap_frac = pick_visit(
                align_polygon, visit_geoms, visit_filter
            )
            mosaic_name, guess_offset = _process_visit(
                visit_id,
                align_to=combined_photfile,
                overlap_frac=float(overlap_frac),
                is_hub_seed=False,
            )
            _finalize_visit(visit_id, mosaic_name, guess_offset)

        # Post-pass: re-JHAT soft-fails / wrong-island JWST frames to hub, then
        # apply a capped CRVAL retie for small residuals only.
        if hub_mosaic and Path(hub_mosaic).is_file():
            jhat_frames = sorted(Path(jhat_dir).glob('*_jhat.fits'))
            hub_phot = None
            try:
                hub_phot = fix_phot(hub_mosaic)
            except Exception as exc:
                logger.warning(
                    'Could not build hub phot for abs QA redo: %s', exc
                )
            catalog_for_redo = (
                hub_phot
                if hub_phot and Path(hub_phot).is_file()
                else (
                    combined_photfile
                    if Path(combined_photfile).is_file()
                    else None
                )
            )
            if catalog_for_redo and jhat_frames:
                # Header soft-fails only -- do not 2dhist the full catalog here;
                # capped abs retie catches wrong-island WCSs via skip_large.
                redo = select_jwst_jhats_for_abs_redo(
                    jhat_frames,
                    hub_mosaic,
                    catalog_path=catalog_for_redo,
                    check_stale=False,
                    measure_abs=False,
                    ncores=ncores,
                )
                if redo:
                    logger.info(
                        'Abs QA redo: %d JWST JHAT soft-fail / stale frame(s) '
                        '-> hub catalog',
                        len(redo),
                    )
                    redo_report = realign_jwst_jhats_to_catalog(
                        redo,
                        catalog_path=catalog_for_redo,
                        raw_dir=work_dir_path / 'raw',
                        jhat_outdir=jhat_dir,
                        ncores=ncores,
                        verbose=verbose,
                    )
                    n_failures += int(redo_report.get('n_failures') or 0)
                    logger.info(
                        'Abs QA redo done: queued=%d failures=%d',
                        int(redo_report.get('n_queued') or 0),
                        int(redo_report.get('n_failures') or 0),
                    )
                    jhat_frames = sorted(Path(jhat_dir).glob('*_jhat.fits'))
            if jhat_frames:
                logger.info(
                    'Abs retie: %d JHAT frame(s) -> hub %s '
                    '(search=%.1f", tol=%.0f mas, max_apply=%.0f mas, '
                    'min_peak=%d, jwst_only=True, ncores=%d)',
                    len(jhat_frames),
                    Path(hub_mosaic).name,
                    HUB_ABS_SEARCH_ARCSEC,
                    HUB_ABS_RETIE_TOL_ARCSEC * 1000.0,
                    HUB_ABS_RETIE_MAX_APPLY_ARCSEC * 1000.0,
                    HUB_ABS_RETIE_MIN_PEAK,
                    ncores,
                )
                retie = retie_jwst_jhat_to_abs_ref(
                    jhat_frames,
                    hub_mosaic,
                    max_residual_arcsec=HUB_ABS_RETIE_TOL_ARCSEC,
                    max_search_arcsec=HUB_ABS_SEARCH_ARCSEC,
                    max_apply_arcsec=HUB_ABS_RETIE_MAX_APPLY_ARCSEC,
                    min_peak=HUB_ABS_RETIE_MIN_PEAK,
                    jwst_only=True,
                    ncores=ncores,
                )
                logger.info(
                    'Abs retie done: shifted=%d ok=%d measure-fail=%d '
                    'skip-large=%d weak-peak=%d skip-non-jwst=%d '
                    'max|Delta|=%.1f mas ok_flag=%s',
                    int(retie.get('n_shifted') or 0),
                    int(retie.get('n_ok') or 0),
                    int(retie.get('n_fail_measure') or 0),
                    int(retie.get('n_skip_large') or 0),
                    int(retie.get('n_weak_peak') or 0),
                    int(retie.get('n_skipped_non_jwst') or 0),
                    1000.0 * float(retie.get('max_abs_arcsec') or 0.0),
                    bool(retie.get('ok')),
                )
                if int(retie.get('n_fail_measure') or 0) > 0:
                    logger.info(
                        'Abs retie could not verify abs offset for %d '
                        'frame(s) vs hub (weak cross-match / overlap); '
                        'relative JHAT QA may still be fine',
                        int(retie['n_fail_measure']),
                    )
                skip_large = [
                    (Path(row['frame']), 'abs_retie_max_apply')
                    for row in (retie.get('frames') or [])
                    if row.get('skipped') == 'max_apply'
                    and Path(row.get('frame') or '').is_file()
                ]
                if skip_large and catalog_for_redo:
                    logger.warning(
                        'Abs retie skipped %d large shift(s) above %.0f mas; '
                        're-JHAT to hub catalog instead of CRVAL nudge',
                        len(skip_large),
                        HUB_ABS_RETIE_MAX_APPLY_ARCSEC * 1000.0,
                    )
                    skip_report = realign_jwst_jhats_to_catalog(
                        skip_large,
                        catalog_path=catalog_for_redo,
                        raw_dir=work_dir_path / 'raw',
                        jhat_outdir=jhat_dir,
                        ncores=ncores,
                        verbose=verbose,
                    )
                    n_failures += int(skip_report.get('n_failures') or 0)
                    logger.info(
                        'Abs retie skip-large redo done: queued=%d '
                        'failures=%d',
                        int(skip_report.get('n_queued') or 0),
                        int(skip_report.get('n_failures') or 0),
                    )
                    # Re-measure only the redone frames (capped again).
                    redone_paths = [fp for fp, _ in skip_large]
                    if redone_paths:
                        retie2 = retie_jwst_jhat_to_abs_ref(
                            redone_paths,
                            hub_mosaic,
                            max_residual_arcsec=HUB_ABS_RETIE_TOL_ARCSEC,
                            max_search_arcsec=HUB_ABS_SEARCH_ARCSEC,
                            max_apply_arcsec=HUB_ABS_RETIE_MAX_APPLY_ARCSEC,
                            min_peak=HUB_ABS_RETIE_MIN_PEAK,
                            jwst_only=True,
                            ncores=ncores,
                        )
                        logger.info(
                            'Abs retie (post skip-large redo): shifted=%d '
                            'ok=%d measure-fail=%d skip-large=%d '
                            'max|Delta|=%.1f mas',
                            int(retie2.get('n_shifted') or 0),
                            int(retie2.get('n_ok') or 0),
                            int(retie2.get('n_fail_measure') or 0),
                            int(retie2.get('n_skip_large') or 0),
                            1000.0
                            * float(retie2.get('max_abs_arcsec') or 0.0),
                        )
                elif int(retie.get('n_skip_large') or 0) > 0:
                    logger.warning(
                        'Abs retie skipped %d large shift(s) above %.0f mas; '
                        'no hub catalog available for JHAT redo',
                        int(retie['n_skip_large']),
                        HUB_ABS_RETIE_MAX_APPLY_ARCSEC * 1000.0,
                    )
        else:
            logger.warning(
                'Group %s: no hub mosaic available for abs retie', group_id
            )

    return 1 if n_failures else 0


def run_reference_alignment(args: argparse.Namespace) -> int:
    """Overlap + align science frames to ``reference/`` coadds."""
    from st123.stages.alignment.align import (
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

    from st123.stages.alignment.align import run_alignment
    from st123.stages.mosaic.image_overlap import BestOverlap, MirIFootprint, compute_overlap

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
        logger.info('Dataset: %s', label)
        logger.info('Mode: reference')
        logger.info('Instrument: %s', args.instrument)
        logger.info('Data dir: %s', data_dir)

    if args.overlap_only and args.align_only:
        logger.error('choose at most one of --overlap-only / --align-only')
        return 2

    try:
        if args.align_only:
            json_path = args.overlap_json
            if json_path is None:
                json_path = data_dir / 'overlap' / 'overlap_summary.json'
            json_path = Path(json_path).expanduser().resolve()
            if not json_path.is_file():
                logger.error('overlap JSON not found: %s', json_path)
                return 1
            logger.info('Loading overlaps from %s', json_path)
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
                logger.info(
                    'Filter restriction %s: %d frame(s) from overlap JSON',
                    ', '.join(filters),
                    len(frames),
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

        # Prefer a writable private phot cache when present (shared caches owned
        # by another user often reject overwrite/utime during i2d rebuilds).
        phot_cache = data_dir / 'overlap' / 'ref_phot_cache_ck'
        if not phot_cache.is_dir():
            phot_cache = data_dir / 'overlap' / 'ref_phot_cache'

        # Always continue on per-frame errors; FAILURE rows go to the summary.
        failures, summary_rows = align_from_frames(
            frames,
            run_alignment=run_alignment,
            nbright=args.nbright,
            plot=args.plot,
            verbose=args.verbose,
            cache_dir=phot_cache,
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
        logger.error('%s', exc)
        if getattr(args, 'verbose', False):
            logger.debug('reference alignment traceback', exc_info=True)
        if summary_rows:
            summary_path = write_alignment_summary(summary_rows, summary_path)
            logger.info('Alignment summary: %s', summary_path)
        return 1

    summary_path = write_alignment_summary(summary_rows, summary_path)
    logger.info('Alignment summary: %s', summary_path)

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
    configure_logging_from_args(args, 'align')
    try:
        try:
            args.mode = _resolve_mode(args)
        except ValueError as exc:
            logger.error('%s', exc)
            return 2

        # --telescope hst|jwst == default --instruments for that mission.
        # Explicit --instruments wins; omitted telescope keeps legacy
        # single-instrument visit defaults (NIRCam).
        multi = resolve_instruments_with_telescope(
            getattr(args, 'instruments', None),
            getattr(args, 'telescope', None),
            resolve_instruments_fn=resolve_align_instruments,
        )

        # Orchestrate when the resolved list is more than NIRCam-only.
        if multi is not None and needs_orchestration(multi):
            if args.mode == 'pair':
                logger.error('do not combine --instruments orchestration with pair mode')
                return 2
            # Keep --telescope consistent with the expanded mission set.
            if getattr(args, 'telescope', None) is None:
                jwst_i, hst_i = partition_align_instruments(multi)
                if hst_i and not jwst_i:
                    args.telescope = 'hst'
                elif jwst_i and not hst_i:
                    args.telescope = 'jwst'
            return run_orchestrated_alignment(args, multi)

        # Singular path: --instruments NIRCAM -> visit NIRCam.
        if multi is not None and len(multi) == 1:
            args.instrument = multi[0]
        else:
            args.instrument = None
        args.instrument = _resolve_instrument(args.mode, args.instrument)

        if args.mode == 'pair':
            return run_pair_alignment(args)

        if args.mode == 'visit':
            telescope = getattr(args, 'telescope', None)
            if telescope is not None and str(telescope).lower() == 'hst':
                # Defensive: HST lists normally orchestrate above.
                if args.base_dir is None:
                    logger.error('--base-dir is required for --telescope hst')
                    return 2
                try:
                    return run_hst_visit_alignment(
                        base_dir=args.base_dir,
                        verbose=args.verbose,
                        ncores=int(getattr(args, 'ncores', 4) or 4),
                        instruments=default_instruments_for_telescope('hst'),
                    )
                except (ValueError, FileNotFoundError) as exc:
                    logger.error('%s', exc)
                    return 1
            base = args.base_dir if args.base_dir is not None else '.'
            try:
                return run_visit_alignment(
                    base_dir=base,
                    instrument=args.instrument,
                    ncores=args.ncores,
                    verbose=args.verbose,
                )
            except ValueError as exc:
                logger.error('%s', exc)
                return 1

        return run_reference_alignment(args)
    finally:
        shutdown_logging()


if __name__ == '__main__':
    raise SystemExit(main())
