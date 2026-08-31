"""
Image footprints, overlap scoring, and Level-3 mosaic / coadd helpers.

Submodules
----------
- :mod:`st123.stages.mosaic.region` - illuminated footprints and ``S_REGION`` polygons
- :mod:`st123.stages.mosaic.image_overlap` - science vs reference footprint overlap
- :mod:`st123.stages.mosaic.mosaic` - overlap splitting, PSF matching, coadds, GWCS
- :mod:`st123.stages.mosaic.hst_drizzle` - HST AstroDrizzle helpers

All public symbols are lazy so console scripts can import a single submodule
(e.g. ``hst_drizzle``) without loading DrizzlePac / JWST resample stacks.
"""

from __future__ import annotations

from typing import Any

_HST_EXPORTS = frozenset(
    {
        'drizzle_filter_group',
        'drizzle_project',
        'drizzle_project_boxed',
        'subtract_per_chip_sky',
        'group_hst_frames',
        'unify_hst_astrometric_frame',
    }
)
_OVERLAP_EXPORTS = frozenset(
    {
        'AreaMetrics',
        'BestOverlap',
        'MirIFootprint',
        'OverlapResult',
        'ScienceFootprint',
        'compute_cumulative_overlap_fraction',
        'compute_overlap',
        'find_best_refs',
        'load_header_s_region',
        'overlap_area_pixels',
        'polygon_area',
    }
)
_REGION_EXPORTS = frozenset(
    {
        'SRegionPolygon',
        'auto_bridge_pixels',
        'default_adjacency_pixels',
        'expand_illuminated_region',
        'find_dq_hdu',
        'find_image_hdu',
        'illuminated_mask',
        'illuminated_mask_from_dq',
        'illuminated_s_region_from_fits',
        'illuminated_s_region_string',
        'infer_coordinate_frame',
        'mask_to_pixel_polygon',
        'pixel_polygon_to_s_region',
        'save_illuminated_region_plot',
        'select_right_illuminated_component',
    }
)
_MOSAIC_EXPORTS = frozenset(
    {
        'FULL_GROUP_LABEL',
        'MosaicBox',
        'MosaicPlan',
        'apply_wcs_to_coadd',
        'assign_gwcs',
        'coadd',
        'convolve_images',
        'copy_files',
        'create_ccddata',
        'create_coadd_mosaic',
        'create_default_mosaic',
        'create_dirs',
        'create_gwcs',
        'create_psf_kernel',
        'edit_spec_groups',
        'find_optimal_wcs',
        'get_pgons',
        'is_hst_mosaic_instrument',
        'is_jwst_mosaic_instrument',
        'mosaic_box_dirname',
        'mosaic_coadd_basename',
        'mosaic_hst_coadd_basename',
        'mosaic_pixel_scale_arcsec',
        'mp_init',
        'plan_mosaic_boxes',
        'plan_existing_box',
        'plan_centered_box',
        'resolve_existing_box_dir',
        'box_coadd_i2d_paths',
        'build_centered_stamp_wcs',
        'rescale_wcs_to_pixel_scale',
        'run_jwst_filter_image3_jobs',
        'jwst_filter_image3_worker',
        'slice_box_wcs',
        'filter_frames_overlapping_box',
        'filter_frames_covering_point',
        'STAMP_WCS_BASENAME',
        'assign_stable_box_ids',
        'ensure_box_stamp_wcs',
        'load_stamp_wcs',
        'local_bbox_for_wcs',
        'stamp_sky_center',
        'stamp_sky_polygon',
        'write_stamp_wcs',
        'split_observations',
        'update_path',
        'update_photmjsr',
        'write_dolphot_frame_list',
        'unify_jwst_astrometric_frame',
        'harmonize_jwst_frames_to_ref',
    }
)

__all__ = sorted(
    _HST_EXPORTS | _OVERLAP_EXPORTS | _REGION_EXPORTS | _MOSAIC_EXPORTS
)


def __getattr__(name: str) -> Any:
    if name in _HST_EXPORTS:
        from st123.stages.mosaic import hst_drizzle as _mod

        return getattr(_mod, name)
    if name in _OVERLAP_EXPORTS:
        from st123.stages.mosaic import image_overlap as _mod

        return getattr(_mod, name)
    if name in _REGION_EXPORTS:
        from st123.stages.mosaic import region as _mod

        return getattr(_mod, name)
    if name in _MOSAIC_EXPORTS:
        from st123.stages.mosaic import mosaic as _mod

        return getattr(_mod, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
    return list(__all__)
