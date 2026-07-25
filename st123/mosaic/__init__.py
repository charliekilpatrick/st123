"""
Image footprints, overlap scoring, and Level-3 mosaic / coadd helpers.

Submodules
----------
- :mod:`st123.mosaic.region` — illuminated footprints and ``S_REGION`` polygons
- :mod:`st123.mosaic.image_overlap` — science vs reference footprint overlap
- :mod:`st123.mosaic.mosaic` — overlap splitting, PSF matching, coadds, GWCS,
  DOLPHOT prep

Heavy symbols from :mod:`st123.mosaic.mosaic` are resolved lazily so importing
footprint helpers does not require optional stack packages (``ccdproc``, etc.).
"""

from __future__ import annotations

from typing import Any

from st123.mosaic.image_overlap import (
    AreaMetrics,
    BestOverlap,
    MirIFootprint,
    OverlapResult,
    ScienceFootprint,
    compute_cumulative_overlap_fraction,
    compute_overlap,
    find_best_refs,
    load_header_s_region,
    overlap_area_pixels,
    polygon_area,
)
from st123.mosaic.region import (
    SRegionPolygon,
    auto_bridge_pixels,
    default_adjacency_pixels,
    expand_illuminated_region,
    find_dq_hdu,
    find_image_hdu,
    illuminated_mask,
    illuminated_mask_from_dq,
    illuminated_s_region_from_fits,
    illuminated_s_region_string,
    infer_coordinate_frame,
    mask_to_pixel_polygon,
    pixel_polygon_to_s_region,
    save_illuminated_region_plot,
    select_right_illuminated_component,
)

_MOSAIC_LAZY = frozenset(
    {
        'apply_nircammask',
        'apply_wcs_to_coadd',
        'assign_gwcs',
        'calc_sky',
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
        'mp_init',
        'setup_paramfile',
        'split_observations',
        'update_path',
        'update_photmjsr',
    }
)

__all__ = [
    'AreaMetrics',
    'BestOverlap',
    'MirIFootprint',
    'OverlapResult',
    'SRegionPolygon',
    'ScienceFootprint',
    'apply_nircammask',
    'apply_wcs_to_coadd',
    'assign_gwcs',
    'auto_bridge_pixels',
    'calc_sky',
    'coadd',
    'compute_cumulative_overlap_fraction',
    'compute_overlap',
    'convolve_images',
    'copy_files',
    'create_ccddata',
    'create_coadd_mosaic',
    'create_default_mosaic',
    'create_dirs',
    'create_gwcs',
    'create_psf_kernel',
    'default_adjacency_pixels',
    'edit_spec_groups',
    'expand_illuminated_region',
    'find_best_refs',
    'find_dq_hdu',
    'find_image_hdu',
    'find_optimal_wcs',
    'get_pgons',
    'illuminated_mask',
    'illuminated_mask_from_dq',
    'illuminated_s_region_from_fits',
    'illuminated_s_region_string',
    'infer_coordinate_frame',
    'load_header_s_region',
    'mask_to_pixel_polygon',
    'mp_init',
    'overlap_area_pixels',
    'pixel_polygon_to_s_region',
    'polygon_area',
    'save_illuminated_region_plot',
    'select_right_illuminated_component',
    'setup_paramfile',
    'split_observations',
    'update_path',
    'update_photmjsr',
]


def __getattr__(name: str) -> Any:
    if name in _MOSAIC_LAZY:
        from st123.mosaic import mosaic as mosaic_mod

        value = getattr(mosaic_mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
    return sorted(__all__)
