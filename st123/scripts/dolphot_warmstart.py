#!/usr/bin/env python3
"""Set up a NIRCam→MIRI DOLPHOT warm-start run (does not execute dolphot)."""

from __future__ import annotations

from pathlib import Path

from st123.photometry.warmstart import setup_miri_warmstart
from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    create_parser as build_parser,
    default_alignment_summary,
    default_phot_dir,
    default_warmstart_outdir,
    resolve_project_root,
)
from st123.utils.settings import DEFAULT_DOLPHOT_BIN


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
        '--phot-file',
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
        default=DEFAULT_DOLPHOT_BIN,
        help=f'DOLPHOT bin directory (default: {DEFAULT_DOLPHOT_BIN}).',
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
    add_common_runtime(parser, ncores=False, plot=False, verbose=True)
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
        raise SystemExit(
            'ERROR: provide --base-dir or both --nircam-dir and --outdir'
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
    try:
        nircam_dir, outdir, data_root, summary = _resolve_warmstart_paths(args)
    except SystemExit as exc:
        print(exc)
        return 1

    if args.verbose:
        print(f'NIRCam dir: {nircam_dir}')
        print(f'Outdir:     {outdir}')
        print(f'Data root:  {data_root}')
        print(f'Summary:    {summary}')

    result = setup_miri_warmstart(
        nircam_dir,
        outdir,
        miri_jhat=args.miri_jhat,
        data_root=data_root,
        alignment_summary=summary,
        phot_file=args.phot_file,
        min_overlap=args.min_overlap,
        dolphot_bin=args.dolphot_bin,
        phot_out=args.phot_out,
        prepare_miri=not args.skip_miri_prep,
        use_hardlink=not args.copy,
    )
    print(f'Warm-start directory: {result.outdir}')
    print(f'  NIRCam frames: {len(result.nircam_images)}')
    print(f'  MIRI frames:   {len(result.miri_images)}')
    print(f'  Param file:    {result.param_file}')
    print(f'  xyt file:      {result.xyt_file}')
    print(f'  Phot output:   {result.phot_out}')
    print()
    print('Run DOLPHOT with:')
    print(f'  {result.command}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
