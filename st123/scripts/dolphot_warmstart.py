#!/usr/bin/env python3
"""Prepare NIRCam->MIRI/HST DOLPHOT warm-start runs (does not execute dolphot)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    configure_logging_from_args,
    create_parser as build_parser,
    default_alignment_summary,
    default_hst_warmstart_outdir,
    default_phot_dir,
    default_warmstart_outdir,
    parse_instruments,
    resolve_project_root,
)
from st123.utils.logging import shutdown_logging

logger = logging.getLogger(__name__)

# Seed / free-run dirs: phot_G_B, nircam_G_B (not nircam_miri_* / nircam_hst_*).
_SEED_DIR_RE = re.compile(
    r'^(?:phot|nircam)_(\d+)_(.+)$',
    re.IGNORECASE,
)


def group_box_from_seed_dirname(name: str) -> tuple[int, int | str]:
    """
    Parse ``group`` / ``box`` from a free DOLPHOT seed directory name.

    Examples: ``phot_0_sn`` -> ``(0, 'sn')``; ``nircam_1_2`` -> ``(1, 2)``.
    Falls back to ``(0, 0)`` when the name does not match.
    """
    match = _SEED_DIR_RE.match(Path(name).name)
    if not match:
        return 0, 0
    group = int(match.group(1))
    token = match.group(2)
    try:
        return group, int(token)
    except ValueError:
        return group, token

def resolve_warmstart_targets(
    instruments: list[str] | None,
    *,
    legacy_target: str | None = None,
) -> list[str]:
    """
    Map ``--instruments`` / legacy ``--target`` to warm-start target list.

    ``MIRI`` -> ``miri``; ``HST`` / ``ACS`` / ``WFC3`` / ``WFPC2`` -> ``hst``.
    NIRCam tokens are ignored (seed catalog, not a warm-start science target).
    """
    plan = resolve_warmstart_plan(instruments, legacy_target=legacy_target)
    return [target for target, _ in plan]


def resolve_warmstart_plan(
    instruments: list[str] | None,
    *,
    legacy_target: str | None = None,
) -> list[tuple[str, list[str] | None]]:
    """
    Return ``[(target, hst_instrument_filter), ...]``.

    For HST, *hst_instrument_filter* is ``None`` when the user asked for bare
    ``HST`` (all cameras), or e.g. ``['ACS', 'WFC3']`` when those were named
    explicitly so WFPC2 can be excluded.
    """
    if instruments:
        want_miri = False
        hst_named: list[str] = []
        want_all_hst = False
        for raw in instruments:
            key = str(raw).strip().upper().split('/')[0]
            if key == 'NRC':
                key = 'NIRCAM'
            if key == 'MIRI':
                want_miri = True
            elif key == 'HST':
                want_all_hst = True
            elif key in ('ACS', 'WFC3', 'WFPC2'):
                if key not in hst_named:
                    hst_named.append(key)
            elif key == 'NIRCAM':
                continue
            else:
                raise ValueError(
                    f'Unsupported warm-start instrument {raw!r}; '
                    'use MIRI and/or HST (ACS WFC3 WFPC2)'
                )
        plan: list[tuple[str, list[str] | None]] = []
        if want_miri:
            plan.append(('miri', None))
        if want_all_hst or hst_named:
            # Bare HST -> all cameras; ACS WFC3 -> those only (no WFPC2).
            hst_filter = hst_named if hst_named else None
            plan.append(('hst', hst_filter))
        if not plan:
            raise ValueError(
                '--instruments must include MIRI and/or HST '
                '(NIRCam alone is the seed catalog, not a warm-start target)'
            )
        return plan
    target = (legacy_target or 'miri').lower()
    if target not in ('miri', 'hst'):
        raise ValueError(f'Unsupported --target {legacy_target!r}')
    return [(target, None)]


def create_parser():
    parser = build_parser(
        description=(
            'Prepare a warm-start DOLPHOT directory from an existing free '
            'NIRCam (or phot_*) run (xytfile mode). Example:\n'
            '  dolphot-warmstart --instruments MIRI --base-dir "$PROJ" '
            '--ref-dir "$PHOTDIR" --prune-xyt --ncores "$NCORES" -v\n'
            'Stages overlapping MIRI or HST JHAT science against the seed '
            'reference. Does not execute dolphot (use run-dolphot afterward).'
        ),
    )
    add_base_dir(
        parser,
        required=False,
        help=(
            'Project root (.../<object> containing JWST/ or HST/, reduction/, '
            'dolphot/). Defaults: reduction/phot_0_0 seed; outdirs inherit '
            'group/box from --ref-dir (e.g. phot_0_sn -> nircam_miri_0_sn).'
        ),
    )
    parser.add_argument(
        '--instruments',
        nargs='+',
        default=None,
        help=(
            'Warm-start science instrument(s): MIRI and/or HST. '
            'ACS WFC3 (recommended) selects HST and stages only those '
            'cameras (excludes WFPC2). Bare HST includes all HST JHAT. '
            'Case-insensitive. Replaces --target.'
        ),
    )
    parser.add_argument(
        '--target',
        choices=('miri', 'hst'),
        default=None,
        help=(
            'Deprecated: use --instruments MIRI|HST. Warm-start science '
            'instrument when --instruments is omitted (default: miri).'
        ),
    )
    parser.add_argument(
        '--ref-dir',
        dest='ref_dir',
        type=str,
        default=None,
        help=(
            'Existing free DOLPHOT run directory to seed from '
            '(dolphot.param + .phot). Typically $PHOTDIR from dolphot-prep / '
            'run-dolphot (e.g. reduction/phot_0_sn). '
            'Default: <base-dir>/reduction/phot_0_0. When --outdir is omitted, '
            'group/box are taken from this directory name.'
        ),
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default=None,
        help=(
            'Warm-start run directory to create. '
            'Default: <base-dir>/dolphot/nircam_{miri,hst}_{group}_{box} '
            'using group/box from --ref-dir (or phot_0_0). '
            'With multiple --instruments, omit --outdir to use each default.'
        ),
    )
    parser.add_argument(
        '--alignment-summary',
        type=str,
        default=None,
        help='Alignment summary listing SUCCESS MIRI JHAT paths (MIRI target).',
    )
    parser.add_argument(
        '--miri-jhat',
        nargs='+',
        default=None,
        help='Explicit MIRI *_jhat.fits paths (overrides discovery).',
    )
    parser.add_argument(
        '--hst-jhat',
        nargs='+',
        default=None,
        help='Explicit HST *_jhat.fits paths (overrides discovery).',
    )
    parser.add_argument(
        '--photfile',
        '--phot-file',
        dest='photfile',
        type=str,
        default=None,
        help='Seed .phot catalog for warmstart.xyt (default: auto-detect).',
    )
    parser.add_argument(
        '--min-overlap',
        type=float,
        default=0.0,
        help='Minimum ref_overlap_frac when reading an alignment summary (MIRI).',
    )
    parser.add_argument(
        '--phot-out',
        type=str,
        default=None,
        help='DOLPHOT output catalog name (default depends on target).',
    )
    parser.add_argument(
        '--dolphot-bin',
        type=str,
        default=None,
        help=(
            'DOLPHOT bin directory containing dolphot/nircammask/mirimask/calcsky. '
            'Default: directory of `dolphot` found on PATH (shutil.which).'
        ),
    )
    parser.add_argument(
        '--skip-miri-prep',
        action='store_true',
        help='Do not run mirimask/calcsky (frames already prepared).',
    )
    parser.add_argument(
        '--skip-hst-prep',
        action='store_true',
        help='Do not run HST mask/split/calcsky.',
    )
    parser.add_argument(
        '--include-nircam-science',
        action='store_true',
        help='Also stage NIRCam science frames (HST target; default: HST only).',
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy seed products instead of hardlinking.',
    )
    parser.add_argument(
        '--prune-xyt',
        action='store_true',
        help=(
            'Thin warmstart.xyt with instrument defaults: MIRI type=1, '
            'SNR>=10, crowd<=0.5, sharp^2<=0.01, minsep=0.30"; HST type=1, '
            'SNR>=5, crowd<=0.5, sharp^2<=0.01, minsep=0.15" '
            '(overridable with --xyt-* flags).'
        ),
    )
    parser.add_argument(
        '--prune-xyt-for-miri',
        action='store_true',
        help='Deprecated alias for MIRI --prune-xyt defaults.',
    )
    parser.add_argument(
        '--prune-xyt-for-hst',
        action='store_true',
        help='Deprecated alias for HST --prune-xyt defaults.',
    )
    parser.add_argument(
        '--xyt-types',
        type=int,
        nargs='+',
        default=None,
        help='DOLPHOT object types to keep in warmstart.xyt (e.g. 1).',
    )
    parser.add_argument(
        '--xyt-snr-min',
        type=float,
        default=None,
        help='Minimum seed SNR for warmstart.xyt.',
    )
    parser.add_argument(
        '--xyt-crowd-max',
        type=float,
        default=None,
        help='Maximum crowding for warmstart.xyt seeds.',
    )
    parser.add_argument(
        '--xyt-sharp2-max',
        type=float,
        default=None,
        help='Maximum sharpness^2 for warmstart.xyt seeds.',
    )
    parser.add_argument(
        '--xyt-min-sep-arcsec',
        type=float,
        default=None,
        help='Minimum seed separation on the reference (arcsec).',
    )
    parser.add_argument(
        '--xyt-force-xy',
        type=str,
        default=None,
        help=(
            'Comma-separated reference X,Y to always keep in warmstart.xyt '
            '(e.g. 1584.25,2793.24).'
        ),
    )
    parser.add_argument(
        '--xyt-max-radius-arcsec',
        type=float,
        default=None,
        help=(
            'Keep only warmstart.xyt seeds within this radius of '
            '--xyt-force-xy / --xyt-center-xy.'
        ),
    )
    parser.add_argument(
        '--xyt-center-xy',
        type=str,
        default=None,
        help=(
            'Comma-separated reference X,Y center for --xyt-max-radius-arcsec '
            '(defaults to --xyt-force-xy when set).'
        ),
    )
    add_common_runtime(parser, ncores=True, ncores_default=1, plot=False, verbose=True)
    return parser


def _resolve_warmstart_paths(
    args,
    *,
    target: str | None = None,
) -> tuple[str, str, str | None, str | None]:
    """Return seed_dir, outdir, data_root, alignment_summary for one target."""
    target = (target or getattr(args, 'target', None) or 'miri').lower()
    seed = getattr(args, 'ref_dir', None)
    if seed and args.outdir:
        data_root = (
            str(resolve_project_root(args.base_dir))
            if args.base_dir
            else None
        )
        summary = args.alignment_summary
        if summary is None and args.base_dir and target == 'miri':
            summary = str(default_alignment_summary(args.base_dir))
        return seed, args.outdir, data_root, summary

    if args.base_dir is None and not seed:
        raise ValueError(
            'provide --base-dir or both --ref-dir and --outdir'
        )

    if args.base_dir is None:
        raise ValueError(
            'provide --base-dir when --outdir is omitted '
            '(needed for default warm-start output path)'
        )

    base = Path(args.base_dir)
    seed_dir = seed or str(default_phot_dir(base))
    group, box = group_box_from_seed_dirname(Path(seed_dir).name)
    if args.outdir:
        outdir = args.outdir
    elif target == 'hst':
        outdir = str(default_hst_warmstart_outdir(base, group=group, box=box))
    else:
        outdir = str(default_warmstart_outdir(base, group=group, box=box))
    data_root = str(resolve_project_root(base))
    summary = args.alignment_summary
    if summary is None and target == 'miri':
        summary = str(default_alignment_summary(base))
    return seed_dir, outdir, data_root, summary


def _parse_xy(flag: str, value: str | None) -> tuple[float, float] | None:
    if not value:
        return None
    parts = [float(x) for x in str(value).split(',')]
    if len(parts) != 2:
        raise ValueError(f'{flag} must be X,Y')
    return (parts[0], parts[1])


def _prune_flags_for_target(args, target: str) -> bool:
    if target == 'hst':
        return bool(args.prune_xyt or args.prune_xyt_for_hst)
    return bool(args.prune_xyt or args.prune_xyt_for_miri)


def _run_one_target(
    args,
    target: str,
    *,
    hst_instruments: list[str] | None = None,
) -> int:
    seed_dir, outdir, data_root, summary = _resolve_warmstart_paths(
        args, target=target
    )
    prune = _prune_flags_for_target(args, target)

    if args.verbose:
        logger.info('Target:     %s', target)
        logger.info('Seed dir:   %s', seed_dir)
        logger.info('Outdir:     %s', outdir)
        logger.info('Data root:  %s', data_root)
        if target == 'hst' and hst_instruments:
            logger.info('HST inst:   %s', ', '.join(hst_instruments))
        if target == 'miri':
            logger.info('Summary:    %s', summary)
        if prune:
            logger.info('Prune xyt:  yes (%s defaults)', target)

    force_xy_pt = _parse_xy('--xyt-force-xy', args.xyt_force_xy)
    center_xy = _parse_xy('--xyt-center-xy', args.xyt_center_xy)
    force_xy = [force_xy_pt] if force_xy_pt is not None else None

    if target == 'hst':
        from st123.stages.photometry.warmstart import setup_hst_warmstart

        result = setup_hst_warmstart(
            seed_dir,
            outdir,
            hst_jhat=args.hst_jhat,
            data_root=data_root,
            instruments=hst_instruments,
            photfile=args.photfile,
            dolphot_bin=args.dolphot_bin,
            phot_out=args.phot_out,
            prepare_hst=not args.skip_hst_prep,
            include_nircam_science=args.include_nircam_science,
            use_hardlink=not args.copy,
            xyt_types=args.xyt_types,
            xyt_snr_min=args.xyt_snr_min,
            xyt_crowd_max=args.xyt_crowd_max,
            xyt_sharp2_max=args.xyt_sharp2_max,
            xyt_min_sep_arcsec=args.xyt_min_sep_arcsec,
            xyt_force_xy=force_xy,
            xyt_max_radius_arcsec=args.xyt_max_radius_arcsec,
            xyt_center_xy=center_xy,
            prune_xyt_for_hst=prune,
            ncores=args.ncores,
        )
    else:
        from st123.stages.photometry.warmstart import setup_miri_warmstart

        result = setup_miri_warmstart(
            seed_dir,
            outdir,
            miri_jhat=args.miri_jhat,
            data_root=data_root,
            alignment_summary=summary,
            photfile=args.photfile,
            min_overlap=args.min_overlap,
            dolphot_bin=args.dolphot_bin,
            phot_out=args.phot_out,
            prepare_miri=not args.skip_miri_prep,
            use_hardlink=not args.copy,
            xyt_types=args.xyt_types,
            xyt_snr_min=args.xyt_snr_min,
            xyt_crowd_max=args.xyt_crowd_max,
            xyt_sharp2_max=args.xyt_sharp2_max,
            xyt_min_sep_arcsec=args.xyt_min_sep_arcsec,
            xyt_force_xy=force_xy,
            xyt_max_radius_arcsec=args.xyt_max_radius_arcsec,
            xyt_center_xy=center_xy,
            prune_xyt_for_miri=prune,
            ncores=args.ncores,
        )

    logger.info('Warm-start directory: %s', result.outdir)
    logger.info('  Seed frames:  %d', len(result.nircam_images))
    if target == 'hst':
        logger.info('  HST frames:   %d', len(result.hst_images))
    else:
        logger.info('  MIRI frames:  %d', len(result.miri_images))
    logger.info('  Param file:   %s', result.param_file)
    logger.info('  xyt file:     %s', result.xyt_file)
    logger.info('  Phot output:  %s', result.phot_out)
    if result.plan is not None:
        logger.info('  Split parts:  %d', len(result.plan.parts))
    logger.info('Run DOLPHOT with:')
    for cmd in result.commands or ([result.command] if result.command else []):
        logger.info('  %s', cmd)
    return 0


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)
    configure_logging_from_args(args, 'dolphot-warmstart-prep')
    try:
        instruments = parse_instruments(getattr(args, 'instruments', None))
        try:
            plan = resolve_warmstart_plan(
                instruments, legacy_target=args.target
            )
        except ValueError as exc:
            logger.error('%s', exc)
            return 2

        if args.outdir is not None and len(plan) > 1:
            logger.error(
                '--outdir cannot be combined with multiple --instruments; '
                'omit --outdir to use each target default, or prep one at a time'
            )
            return 2

        rc = 0
        for target, hst_instruments in plan:
            # Stash for helpers that still read args.target
            args.target = target
            try:
                target_rc = _run_one_target(
                    args, target, hst_instruments=hst_instruments
                )
            except ValueError as exc:
                logger.error('%s', exc)
                return 1
            except FileNotFoundError as exc:
                logger.error('%s', exc)
                return 1
            if target_rc != 0:
                rc = target_rc
        return rc
    finally:
        shutdown_logging()


if __name__ == '__main__':
    raise SystemExit(main())
