#!/usr/bin/env python3
"""Set up a NIRCam→MIRI DOLPHOT warm-start run (does not execute dolphot)."""

from __future__ import annotations

import logging
from pathlib import Path

from st123.photometry.warmstart import setup_miri_warmstart
from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    configure_logging_from_args,
    create_parser as build_parser,
    default_alignment_summary,
    default_phot_dir,
    default_warmstart_outdir,
    resolve_project_root,
)
from st123.utils.logging import shutdown_logging

logger = logging.getLogger(__name__)


def create_parser():
    parser = build_parser(
        description=(
            'Prepare a warm-start DOLPHOT directory that appends overlapping '
            'MIRI JHAT frames to an existing NIRCam run (xytfile mode). '
            'Prefer --base-dir; use --nircam-dir/--outdir only when those '
            'paths must be set independently.'
        ),
    )
    add_base_dir(
        parser,
        required=False,
        help=(
            'Project root (…/<object> containing JWST/, reduction/, dolphot/). '
            'Defaults: reduction/phot_0_0 and dolphot/nircam_miri_0_0.'
        ),
    )
    parser.add_argument(
        '--nircam-dir',
        type=str,
        default=None,
        help=(
            'Existing NIRCam DOLPHOT run directory (dolphot.param + .phot). '
            'Default: <base-dir>/reduction/phot_0_0.'
        ),
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default=None,
        help=(
            'Warm-start run directory to create. '
            'Default: <base-dir>/dolphot/nircam_miri_0_0.'
        ),
    )
    parser.add_argument(
        '--alignment-summary',
        type=str,
        default=None,
        help='Alignment summary listing SUCCESS MIRI JHAT paths.',
    )
    parser.add_argument(
        '--miri-jhat',
        nargs='+',
        default=None,
        help='Explicit MIRI *_jhat.fits paths (overrides discovery).',
    )
    parser.add_argument(
        '--photfile',
        '--phot-file',
        dest='photfile',
        type=str,
        default=None,
        help='NIRCam .phot catalog for warmstart.xyt (default: auto-detect).',
    )
    parser.add_argument(
        '--min-overlap',
        type=float,
        default=0.0,
        help='Minimum ref_overlap_frac when reading an alignment summary.',
    )
    parser.add_argument(
        '--phot-out',
        type=str,
        default=None,
        help='DOLPHOT output catalog name (default: <seed>_nircam_miri.phot).',
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
        '--copy',
        action='store_true',
        help='Copy NIRCam products instead of hardlinking.',
    )
    parser.add_argument(
        '--prune-xyt-for-miri',
        action='store_true',
        help=(
            'Thin warmstart.xyt for MIRI: type=1, SNR>=10, crowd<=0.5, '
            'sharp^2<=0.01, minsep=0.30 arcsec (overridable below).'
        ),
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
        help='Minimum NIRCam SNR for warmstart.xyt seeds.',
    )
    parser.add_argument(
        '--xyt-crowd-max',
        type=float,
        default=None,
        help='Maximum NIRCam crowding for warmstart.xyt seeds.',
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
        help='Minimum seed separation on the NIRCam reference (arcsec).',
    )
    parser.add_argument(
        '--xyt-force-xy',
        type=str,
        default=None,
        help=(
            'Comma-separated reference X,Y to always keep in warmstart.xyt '
            '(e.g. 1584.25,2793.24 for SN 2026sqf).'
        ),
    )
    parser.add_argument(
        '--xyt-max-radius-arcsec',
        type=float,
        default=None,
        help=(
            'Keep only warmstart.xyt seeds within this radius of '
            '--xyt-force-xy / --xyt-center-xy (e.g. 5 for SN 2026sqf).'
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


def _resolve_warmstart_paths(args) -> tuple[str, str, str | None, str | None]:
    """Return nircam_dir, outdir, data_root, alignment_summary."""
    if args.nircam_dir and args.outdir:
        data_root = (
            str(resolve_project_root(args.base_dir))
            if args.base_dir
            else None
        )
        summary = args.alignment_summary
        if summary is None and args.base_dir:
            summary = str(default_alignment_summary(args.base_dir))
        return args.nircam_dir, args.outdir, data_root, summary

    if args.base_dir is None:
        raise ValueError(
            'provide --base-dir or both --nircam-dir and --outdir'
        )

    base = Path(args.base_dir)
    nircam = args.nircam_dir or str(default_phot_dir(base))
    outdir = args.outdir or str(default_warmstart_outdir(base))
    data_root = str(resolve_project_root(base))
    summary = args.alignment_summary
    if summary is None:
        summary = str(default_alignment_summary(base))
    return nircam, outdir, data_root, summary


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)
    configure_logging_from_args(args, 'dolphot-warmstart')
    try:
        try:
            nircam_dir, outdir, data_root, summary = _resolve_warmstart_paths(args)
        except ValueError as exc:
            logger.error('%s', exc)
            return 1

        if args.verbose:
            logger.info('NIRCam dir: %s', nircam_dir)
            logger.info('Outdir:     %s', outdir)
            logger.info('Data root:  %s', data_root)
            logger.info('Summary:    %s', summary)

        force_xy = None
        if args.xyt_force_xy:
            parts = [float(x) for x in str(args.xyt_force_xy).split(',')]
            if len(parts) != 2:
                logger.error('--xyt-force-xy must be X,Y')
                return 1
            force_xy = [(parts[0], parts[1])]
        center_xy = None
        if args.xyt_center_xy:
            parts = [float(x) for x in str(args.xyt_center_xy).split(',')]
            if len(parts) != 2:
                logger.error('--xyt-center-xy must be X,Y')
                return 1
            center_xy = (parts[0], parts[1])

        try:
            result = setup_miri_warmstart(
                nircam_dir,
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
                prune_xyt_for_miri=args.prune_xyt_for_miri,
                ncores=args.ncores,
            )
        except FileNotFoundError as exc:
            logger.error('%s', exc)
            return 1
        logger.info('Warm-start directory: %s', result.outdir)
        logger.info('  NIRCam frames: %d', len(result.nircam_images))
        logger.info('  MIRI frames:   %d', len(result.miri_images))
        logger.info('  Param file:    %s', result.param_file)
        logger.info('  xyt file:      %s', result.xyt_file)
        logger.info('  Phot output:   %s', result.phot_out)
        if result.plan is not None:
            logger.info('  Split parts:   %d', len(result.plan.parts))
        logger.info('Run DOLPHOT with:')
        for cmd in result.commands or ([result.command] if result.command else []):
            logger.info('  %s', cmd)
        return 0
    finally:
        shutdown_logging()


if __name__ == '__main__':
    raise SystemExit(main())
