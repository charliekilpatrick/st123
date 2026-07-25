#!/usr/bin/env python3
"""Build mosaics / coadds from JHAT-aligned frames and prepare DOLPHOT."""

from __future__ import annotations

import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np

from st123.mosaic.mosaic import (
    apply_nircammask,
    calc_sky,
    coadd,
    convolve_images,
    copy_files,
    create_coadd_mosaic,
    create_dirs,
    create_gwcs,
    edit_spec_groups,
    mp_init,
    setup_paramfile,
    split_observations,
    update_path,
)
from st123.utils.helpers import input_list


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Mosaic / coadd JWST JHAT frames')
    parser.add_argument('--basedir', type=str, default='.', help='Root directory to search for data')
    parser.add_argument('--object', type=str, default='dolphot', help='Object to reduce')
    parser.add_argument('--nmax', type=int, default=150, help='Maximum number of images in a dolphot run')
    parser.add_argument('--spec_groups', type=str, default=None, help='List of images to be grouped')
    parser.add_argument('--ncores', type=int, default=1, help='Number of CPU cores')
    return parser


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    base_dir = args.basedir
    obj = args.object
    nmax = args.nmax
    spec_group_file = args.spec_groups
    ncores = args.ncores

    phot_root = os.path.join(base_dir, obj)
    inputfiles = glob.glob(os.path.join(base_dir, 'jhat', '*jhat.fits'))
    table = input_list(inputfiles)
    if spec_group_file:
        table = edit_spec_groups(table, spec_group_file)
    ngroups = np.unique(table['group'])

    out_dict = create_dirs(base_dir, len(ngroups))

    Pool(initializer=mp_init, processes=ncores, initargs=(0, 0, []))

    for group_id in ngroups:
        group_table = table[table['group'] == group_id]
        outdir = out_dict[group_id]
        split_obs = split_observations(table=group_table, N_max=nmax)
        split_obs.boxsplit()
        filter_tables = split_obs.get_sw_filter_tables(tol=0.05)
        mosaic_wcs = split_obs.wcs

        for box_index in range(len(split_obs.split_boxes)):
            subimages, filter_table, bbox = (
                split_obs.subimages[box_index],
                filter_tables[box_index],
                split_obs.split_boxes[box_index],
            )
            box_outdir = os.path.join(outdir, f'ref_{box_index}')
            phot_outdir = os.path.join(phot_root, f'phot_{group_id}_{box_index}')
            if not os.path.exists(box_outdir):
                os.makedirs(box_outdir)

            if not os.path.exists(phot_outdir):
                os.makedirs(phot_outdir)

            minx = int(np.abs(np.floor(min(bbox.exterior.xy[0]))))
            maxx = int(np.abs(np.ceil(max(bbox.exterior.xy[0]))))
            miny = int(np.abs(np.floor(min(bbox.exterior.xy[1]))))
            maxy = int(np.abs(np.ceil(max(bbox.exterior.xy[1]))))
            wcs_slice = (slice(miny, maxy), slice(minx, maxx))
            box_wcs = mosaic_wcs.slice(wcs_slice)
            wcs_hdr = box_wcs.to_header()
            wcs_hdr['NAXIS1'], wcs_hdr['NAXIS2'] = box_wcs._naxis[0], box_wcs._naxis[1]
            gwcs_path = create_gwcs(outdir=box_outdir, sci_header=wcs_hdr)

            filter_keys = sorted(filter_table.keys())
            target_filter = filter_keys[-1]

            copy_files(filter_table, box_outdir)
            filter_table = update_path(filter_table, box_outdir)

            convolve_images(filter_table, target_filter)

            mosaics = []
            for filter_name in filter_keys:
                driz_image = create_coadd_mosaic(
                    filter_table[filter_name],
                    outdir=box_outdir,
                    filt=filter_name,
                    centroid=None,
                    output_shape=None,
                    gwcs_file=gwcs_path,
                )
                mosaics.append(driz_image)

            coadd_filename = os.path.join(
                box_outdir, f'coadd_{group_id}_{box_index}_{target_filter}_i2d.fits'
            )
            coadd(mosaics, target_filter, coadd_filename)

            setup_paramfile(phot_outdir, coadd_filename, subimages)

            phot_images = glob.glob(os.path.join(phot_outdir, '*fits'))
            apply_nircammask(phot_images)
            calc_sky(phot_images)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
