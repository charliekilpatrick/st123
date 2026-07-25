#!/usr/bin/env python3
"""Align JWST visits with JHAT / Gaia (group-level pipeline)."""

from __future__ import annotations

import argparse
import copy
import os

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
from st123.utils.helpers import create_filter_table, input_list


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Align JWST visits with JHAT / Gaia')
    parser.add_argument('--workdir', type=str, default='.', help='Root directory to search for data')
    parser.add_argument('--object', type=str, default='m92', help='Object to reduce')
    parser.add_argument('--ncores', type=int, default=1, help='Number of CPU cores')
    return parser


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    work_dir = args.workdir
    obj = args.object
    ncores = args.ncores

    create_dirs(work_dir, obj)

    input_images = get_input_images(workdir=work_dir)
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
                    verbose=False,
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


if __name__ == '__main__':
    raise SystemExit(main())
