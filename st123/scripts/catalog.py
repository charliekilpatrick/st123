#!/usr/bin/env python3
"""Build a combined photometry catalog from DOLPHOT column exports."""

from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

from st123.photometry.catalog import create_common_catalog


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Combine per-filter CSV catalogs into one table.',
    )
    parser.add_argument(
        '--photdir',
        type=str,
        default='ngc628_phot',
        help='Directory containing rsg*.csv and *columns files.',
    )
    parser.add_argument(
        '--outfile',
        type=str,
        default='ngc628_combined_phot.csv',
        help='Output combined catalog path.',
    )
    return parser


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)
    photdir = args.photdir

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
