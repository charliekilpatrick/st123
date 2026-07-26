#!/usr/bin/env python3
"""Build mosaics / coadds from JHAT-aligned frames (no DOLPHOT prep)."""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

from st123.mosaic.mosaic import (
    apply_wcs_to_coadd,
    coadd,
    convolve_images,
    copy_files,
    create_coadd_mosaic,
    create_dirs,
    create_gwcs,
    edit_spec_groups,
    split_observations,
    update_path,
    write_dolphot_frame_list,
)
from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    create_parser as build_parser,
    dataset_label,
    resolve_reduction_dir,
)
from st123.utils.helpers import input_list


def create_parser():
    parser = build_parser(
        description=(
            'Mosaic / coadd JWST JHAT frames. DOLPHOT staging (nircammask, '
            'calcsky, dolphot.param) is a separate step: dolphot-prep.'
        ),
    )
    add_base_dir(
        parser,
        required=False,
        default='.',
        help=(
            'Project root (…/<object>) or reduction workdir. Coadds are '
            'written under <reduction>/reference/. Aliases: --basedir, '
            '--workdir, --data-dir.'
        ),
    )
    parser.add_argument(
        '--nmax',
        type=int,
        default=150,
        help='Maximum number of images per mosaic box',
    )
    parser.add_argument(
        '--spec_groups',
        type=str,
        default=None,
        help='List of images to be grouped',
    )
    add_common_runtime(parser, plot=False, verbose=True)
    return parser


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    base_dir = str(resolve_reduction_dir(args.base_dir))
    nmax = args.nmax
    spec_group_file = args.spec_groups
    if args.verbose:
        print(f'Dataset: {dataset_label(args.base_dir)}')
        print(f'Reduction workdir: {base_dir}')

    inputfiles = sorted(glob.glob(os.path.join(base_dir, 'jhat', '*jhat.fits')))
    if not inputfiles:
        print(
            f'ERROR: no jhat/*jhat.fits under {base_dir}. '
            'Pass --base-dir to the dataset root (…/<object>) or the '
            'reduction workdir that contains jhat/.',
            file=sys.stderr,
        )
        return 1
    table = input_list(inputfiles)
    if spec_group_file:
        table = edit_spec_groups(table, spec_group_file)
    ngroups = np.unique(table['group'])

    out_dict = create_dirs(base_dir, len(ngroups))

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
            if not os.path.exists(box_outdir):
                os.makedirs(box_outdir)

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
                    gwcs_file=gwcs_path,
                )
                mosaics.append(driz_image)

            coadd_filename = os.path.join(
                box_outdir, f'coadd_{group_id}_{box_index}_{target_filter}_i2d.fits'
            )
            coadd(mosaics, target_filter, coadd_filename)
            apply_wcs_to_coadd(coadd_filename)

            # Manifest for dolphot-prep (no DOLPHOT binaries invoked here).
            write_dolphot_frame_list(
                box_outdir,
                refimage=coadd_filename,
                frames=subimages,
                group=int(group_id),
                box=int(box_index),
            )
            if args.verbose:
                print(
                    f'Wrote coadd {coadd_filename}; '
                    f'run dolphot-prep --from-mosaic --base-dir {args.base_dir}'
                )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
