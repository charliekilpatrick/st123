#!/usr/bin/env python3
"""Build mosaics / coadds from JHAT-aligned frames (no DOLPHOT prep)."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import numpy as np

from st123.mosaic.hst_drizzle import drizzle_project
from st123.mosaic.mosaic import (
    apply_wcs_to_coadd,
    coadd,
    convolve_images,
    copy_files,
    create_coadd_mosaic,
    create_dirs,
    create_gwcs,
    edit_spec_groups,
    mosaic_pixel_scale_arcsec,
    rescale_wcs_to_pixel_scale,
    split_observations,
    update_path,
    write_dolphot_frame_list,
)
from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    add_filters,
    configure_logging_from_args,
    create_parser as build_parser,
    dataset_label,
    parse_filter_list,
    resolve_reduction_dir,
)
from st123.utils.helpers import create_filter_table, input_list
from st123.utils.logging import shutdown_logging

logger = logging.getLogger(__name__)


def create_parser():
    parser = build_parser(
        description=(
            'Mosaic / coadd JHAT frames (JWST resample or HST AstroDrizzle). '
            'DOLPHOT staging is a separate step: dolphot-prep.'
        ),
    )
    parser.add_argument(
        '--telescope',
        choices=('jwst', 'hst'),
        default='jwst',
        help=(
            'Telescope mosaic backend. jwst (default): overlap boxes + '
            'JWST resample coadds. hst: AstroDrizzle per instrument/filter '
            'from reduction/jhat → reduction/reference.'
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
        help='Maximum number of images per mosaic box (JWST only)',
    )
    parser.add_argument(
        '--spec_groups',
        type=str,
        default=None,
        help='List of images to be grouped (JWST only)',
    )
    add_filters(
        parser,
        help=(
            'Comma-separated filter list to mosaic (e.g. F150W,F444W,F560W). '
            'When omitted, short-wave NIRCam filters are selected automatically '
            'and PSF-matched into one coadd. When set, each filter is mosaicked '
            'separately onto the same sky footprint with the JWST-recommended '
            'pixel scale for that channel (NIRCam SW/LW or MIRI). JWST only.'
        ),
    )
    add_common_runtime(parser, ncores=True, ncores_default=4, plot=False, verbose=True)
    return parser


def _run_hst_mosaic(args) -> int:
    """AstroDrizzle each (instrument, filter) group from reduction/jhat."""
    base_dir = Path(resolve_reduction_dir(args.base_dir))
    jhat_dir = base_dir / 'jhat'
    outdir = base_dir / 'reference'
    if args.verbose:
        logger.info('Dataset: %s', dataset_label(args.base_dir))
        logger.info('HST drizzle: %s → %s', jhat_dir, outdir)
    if not jhat_dir.is_dir():
        logger.error('missing jhat/ under %s', base_dir)
        return 1
    results = drizzle_project(
        jhat_dir,
        outdir,
        num_cores=int(args.ncores),
    )
    if not results:
        return 1
    n_ok = sum(1 for r in results if r['status'] == 'ok')
    n_fail = sum(1 for r in results if r['status'] != 'ok')
    for r in results:
        if r['status'] == 'ok':
            logger.info(
                'OK %s/%s → %s (%d frames)',
                r['instrument'],
                r['filter'],
                r['output'],
                len(r['frames']),
            )
        else:
            logger.error(
                'FAIL %s/%s: %s',
                r['instrument'],
                r['filter'],
                r.get('error'),
            )
    logger.info('HST drizzle done: %d ok, %d failed', n_ok, n_fail)
    return 0 if n_ok and n_fail == 0 else (0 if n_ok else 1)


def _box_wcs_header(mosaic_wcs, bbox):
    """Slice the group WCS to the box pixel bounds (shared sky footprint)."""
    minx = int(np.abs(np.floor(min(bbox.exterior.xy[0]))))
    maxx = int(np.abs(np.ceil(max(bbox.exterior.xy[0]))))
    miny = int(np.abs(np.floor(min(bbox.exterior.xy[1]))))
    maxy = int(np.abs(np.ceil(max(bbox.exterior.xy[1]))))
    wcs_slice = (slice(miny, maxy), slice(minx, maxx))
    box_wcs = mosaic_wcs.slice(wcs_slice)
    wcs_hdr = box_wcs.to_header()
    wcs_hdr['NAXIS1'], wcs_hdr['NAXIS2'] = (
        box_wcs._naxis[0],
        box_wcs._naxis[1],
    )
    return box_wcs, wcs_hdr


def _forced_filter_tables(reftable, forced_filters: list[str]) -> dict:
    """Build a non-empty filter→table map for requested filters present in box."""
    available = {str(f).lower() for f in reftable['filter']}
    selected = [f for f in forced_filters if f in available]
    missing = [f for f in forced_filters if f not in available]
    if missing:
        logger.warning(
            'Requested filters not present in this box (skipping): %s',
            ', '.join(missing),
        )
    if not selected:
        return {}
    filter_table = create_filter_table(reftable, selected)
    return {k: v for k, v in filter_table.items() if len(v) > 0}


def _run_default_sw_coadd(
    *,
    filter_table: dict,
    box_outdir: str,
    wcs_hdr,
    group_id: int,
    box_index: int,
    subimages,
    base_dir_arg,
    verbose: bool,
) -> str:
    """Optimal SW-filter selection path: PSF-match and write one coadd."""
    filter_keys = sorted(filter_table.keys())
    target_filter = filter_keys[-1]

    gwcs_path = create_gwcs(outdir=box_outdir, sci_header=wcs_hdr)
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
        box_outdir,
        f'coadd_{group_id}_{box_index}_{target_filter}_i2d.fits',
    )
    coadd(mosaics, target_filter, coadd_filename)
    apply_wcs_to_coadd(coadd_filename)

    write_dolphot_frame_list(
        box_outdir,
        refimage=coadd_filename,
        frames=subimages,
        group=int(group_id),
        box=int(box_index),
    )
    if verbose:
        logger.info(
            'Wrote coadd %s; run dolphot-prep --from-mosaic --base-dir %s',
            coadd_filename,
            base_dir_arg,
        )
    return coadd_filename


def _run_forced_filter_coadds(
    *,
    filter_table: dict,
    box_wcs,
    box_outdir: str,
    group_id: int,
    box_index: int,
    subimages,
    base_dir_arg,
    verbose: bool,
) -> list[str]:
    """
    Mosaic each requested filter onto the shared sky footprint.

    Pixel scale follows JWST channel recommendations; RA/Dec coverage matches
    ``box_wcs``.
    """
    written: list[str] = []
    for filter_name, ftable in filter_table.items():
        single = {filter_name: ftable}
        copy_files(single, box_outdir)
        single = update_path(single, box_outdir)

        inst = str(single[filter_name]['instrument'][0])
        pixscale = mosaic_pixel_scale_arcsec(filter_name, inst)
        filt_hdr = rescale_wcs_to_pixel_scale(box_wcs, pixscale)
        gwcs_path = create_gwcs(
            outdir=box_outdir,
            sci_header=filt_hdr,
            filename=f'mosaic_gwcs_{filter_name}.asdf',
        )
        if verbose:
            logger.info(
                'Mosaicking %s (%s) at %.3f"/pix onto shared footprint',
                filter_name,
                inst,
                pixscale,
            )

        driz_image = create_coadd_mosaic(
            single[filter_name],
            outdir=box_outdir,
            filt=filter_name,
            gwcs_file=gwcs_path,
            pixel_scale=pixscale,
        )
        coadd_filename = os.path.join(
            box_outdir,
            f'coadd_{group_id}_{box_index}_{filter_name}_i2d.fits',
        )
        coadd([driz_image], filter_name, coadd_filename)
        apply_wcs_to_coadd(coadd_filename)
        written.append(coadd_filename)
        if verbose:
            logger.info(
                'Wrote coadd %s; run dolphot-prep --from-mosaic --base-dir %s',
                coadd_filename,
                base_dir_arg,
            )

    # One manifest per box: all JHAT frames, first coadd as DOLPHOT reference.
    if written:
        write_dolphot_frame_list(
            box_outdir,
            refimage=written[0],
            frames=subimages,
            group=int(group_id),
            box=int(box_index),
        )
    return written


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    configure_logging_from_args(args, 'mosaic')
    try:
        if str(getattr(args, 'telescope', 'jwst')).lower() == 'hst':
            return _run_hst_mosaic(args)

        base_dir = str(resolve_reduction_dir(args.base_dir))
        nmax = args.nmax
        spec_group_file = args.spec_groups
        forced_filters = parse_filter_list(args.filters)
        if forced_filters:
            forced_filters = [f.lower() for f in forced_filters]
        if args.verbose:
            logger.info('Dataset: %s', dataset_label(args.base_dir))
            logger.info('Reduction workdir: %s', base_dir)
            if forced_filters:
                logger.info('Forced filters: %s', ', '.join(forced_filters))
            else:
                logger.info('Filter mode: automatic short-wave NIRCam selection')

        inputfiles = sorted(glob.glob(os.path.join(base_dir, 'jhat', '*jhat.fits')))
        if not inputfiles:
            logger.error(
                'no jhat/*jhat.fits under %s. '
                'Pass --base-dir to the dataset root (…/<object>) or the '
                'reduction workdir that contains jhat/.',
                base_dir,
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
            mosaic_wcs = split_obs.wcs

            if forced_filters is None:
                filter_tables = split_obs.get_sw_filter_tables(tol=0.05)
            else:
                filter_tables = [
                    _forced_filter_tables(reftable, forced_filters)
                    for reftable in split_obs.reftables
                ]

            for box_index in range(len(split_obs.split_boxes)):
                subimages = split_obs.subimages[box_index]
                filter_table = filter_tables[box_index]
                bbox = split_obs.split_boxes[box_index]
                box_outdir = os.path.join(outdir, f'ref_{box_index}')
                if not os.path.exists(box_outdir):
                    os.makedirs(box_outdir)

                if not filter_table:
                    logger.warning(
                        'No mosaickable filters for group %s box %s; skipping',
                        group_id,
                        box_index,
                    )
                    continue

                box_wcs, wcs_hdr = _box_wcs_header(mosaic_wcs, bbox)

                if forced_filters is None:
                    _run_default_sw_coadd(
                        filter_table=filter_table,
                        box_outdir=box_outdir,
                        wcs_hdr=wcs_hdr,
                        group_id=int(group_id),
                        box_index=box_index,
                        subimages=subimages,
                        base_dir_arg=args.base_dir,
                        verbose=args.verbose,
                    )
                else:
                    _run_forced_filter_coadds(
                        filter_table=filter_table,
                        box_wcs=box_wcs,
                        box_outdir=box_outdir,
                        group_id=int(group_id),
                        box_index=box_index,
                        subimages=subimages,
                        base_dir_arg=args.base_dir,
                        verbose=args.verbose,
                    )
        return 0
    finally:
        shutdown_logging()


if __name__ == '__main__':
    raise SystemExit(main())
