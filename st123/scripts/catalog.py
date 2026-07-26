#!/usr/bin/env python3
"""Build a combined photometry catalog from DOLPHOT column exports."""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from st123.photometry.catalog import create_common_catalog
from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    create_parser as build_parser,
)


def create_parser():
    parser = build_parser(
        description='Combine per-filter CSV catalogs into one table.',
    )
    add_base_dir(
        parser,
        required=False,
        default='ngc628_phot',
        aliases=('--basedir', '--workdir', '--data-dir', '--photdir'),
        help=(
            'Directory containing rsg*.csv and *columns files. '
            'Aliases: --photdir, --data-dir.'
        ),
    )
    parser.add_argument(
        '--outfile',
        type=str,
        default='ngc628_combined_phot.csv',
        help='Output combined catalog path.',
    )
    add_common_runtime(parser, ncores=False, plot=False, verbose=True)
    return parser


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)
    photdir = args.base_dir

    csv_files = sorted(glob.glob(f'{photdir}/rsg*.csv'))
    if not csv_files:
        print(f'ERROR: no rsg*.csv files under {photdir}')
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
    create_common_catalog(common_ids, dfs, columns, args.outfile)
    print(f'Wrote {args.outfile}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
