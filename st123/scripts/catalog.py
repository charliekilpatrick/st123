#!/usr/bin/env python3
"""Parse DOLPHOT catalogs: combined CSV or visit-binned AB light curves."""

from __future__ import annotations

import glob
import logging
from pathlib import Path

from st123.scripts.utils.options import (
    add_base_dir,
    add_catalog_args,
    add_common_runtime,
    add_sky_coord_args,
    configure_logging_from_args,
    create_parser as build_parser,
)
from st123.utils.logging import shutdown_logging

logger = logging.getLogger(__name__)


def create_parser():
    parser = build_parser(
        description=(
            'Parse DOLPHOT photometry. With --photfile and --index (or '
            '--ra/--dec), write a visit-binned AB light curve. Otherwise '
            'combine per-filter CSV catalogs under --base-dir.'
        ),
    )
    add_base_dir(
        parser,
        required=False,
        default='ngc628_phot',
        aliases=('--basedir', '--workdir', '--data-dir', '--photdir'),
        help=(
            'Directory containing rsg*.csv and *columns files (legacy CSV '
            'combine). Aliases: --photdir, --data-dir.'
        ),
    )
    add_catalog_args(parser)
    add_sky_coord_args(parser)
    add_common_runtime(parser, plot=False, verbose=True)
    return parser


def _write_table(table, path: Path) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == '.csv':
        table.write(path, format='ascii.csv', overwrite=True)
    elif suffix in {'.fits', '.fit'}:
        table.write(path, overwrite=True)
    else:
        if suffix != '.ecsv':
            path = path.with_suffix('.ecsv')
        table.write(path, format='ascii.ecsv', overwrite=True)
    logger.info('Wrote %s', path)


def _resolve_source_index(args, phot: Path) -> int:
    if getattr(args, 'source_index', None) is not None:
        return int(args.source_index)
    ra = getattr(args, 'ra', None)
    dec = getattr(args, 'dec', None)
    if ra is None or dec is None:
        raise ValueError('light-curve mode needs --index or --ra and --dec')
    from st123.stages.photometry.dolphot import (
        nearest_phot_source,
        parse_param_image_list,
    )
    from st123.utils.helpers import parse_coord

    coord = parse_coord(ra, dec)
    if coord is None:
        raise ValueError(f'cannot parse --ra/--dec: {ra} {dec}')
    param = phot.parent / 'dolphot.param'
    if not param.is_file():
        raise FileNotFoundError(f'Need dolphot.param next to {phot} for --ra/--dec')
    ref_base, _ = parse_param_image_list(param)
    ref = phot.parent / f'{ref_base}.fits'
    if not ref.is_file():
        raise FileNotFoundError(f'Reference image {ref} not found for sky lookup')
    nearest = nearest_phot_source(
        phot, ref, ra=float(coord.ra.degree), dec=float(coord.dec.degree), n=1
    )
    if not nearest:
        raise ValueError(f'no DOLPHOT source near {ra} {dec}')
    idx = int(nearest[0]['index'])
    logger.info(
        'Nearest source index %d (%.2f pix from target)',
        idx,
        nearest[0]['dist_pix'],
    )
    return idx


def _run_lightcurve(args) -> int:
    from st123.stages.photometry.catalog import source_lightcurve

    phot = Path(args.photfile).expanduser()
    if not phot.is_file():
        logger.error('photfile not found: %s', phot)
        return 1
    outfile = args.outfile
    if not outfile:
        logger.error('light-curve mode requires --outfile')
        return 1
    try:
        index = _resolve_source_index(args, phot)
    except (ValueError, FileNotFoundError) as exc:
        logger.error('%s', exc)
        return 1
    table = source_lightcurve(
        phot,
        index,
        max_gap_days=float(getattr(args, 'max_gap_days', 1.0) or 1.0),
        max_flag=int(getattr(args, 'max_flag', 7)),
    )
    _write_table(table, Path(outfile).expanduser())
    return 0


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)
    configure_logging_from_args(args, 'catalog')
    try:
        if getattr(args, 'photfile', None):
            return _run_lightcurve(args)

        import numpy as np
        import pandas as pd

        from st123.stages.photometry.catalog import create_common_catalog

        photdir = args.base_dir

        csv_files = sorted(glob.glob(f'{photdir}/rsg*.csv'))
        if not csv_files:
            logger.error('no rsg*.csv files under %s', photdir)
            return 1

        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            df.set_index('idx', inplace=True)
            dfs.append(df)

        source_ids = []
        for df in dfs:
            source_ids.extend(df.index.values)
        common_ids = np.unique(source_ids)

        columns = sorted(glob.glob(f'{photdir}/*columns'))
        outfile = args.outfile or 'ngc628_combined_phot.csv'
        create_common_catalog(common_ids, dfs, columns, outfile)
        logger.info('Wrote %s', outfile)
        return 0
    finally:
        shutdown_logging()


if __name__ == '__main__':
    raise SystemExit(main())
