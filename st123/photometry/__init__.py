"""
Photometry helpers for individual images and combined catalogs.

Submodules
----------
- :mod:`st123.photometry.aperture` — forced EE aperture photometry on coadds
- :mod:`st123.photometry.catalog` — DOLPHOT column mapping and combined catalogs
- :mod:`st123.photometry.dolphot` — mask / calcsky / paramfile prep
- :mod:`st123.photometry.warmstart` — NIRCam→MIRI warm-start run setup
"""

from __future__ import annotations

from st123.photometry.aperture import (
    AB_ZEROPOINT_UJY,
    ApertureParams,
    default_ee_fraction,
    forced_aperture_photometry,
    get_aperture_params,
    mjy_sr_to_ujy_arcsec2,
    pixel_scale_arcsec,
    read_coords_table,
    resolve_positions,
    surface_brightness_to_ujy,
    ujy_to_abmag,
)
from st123.photometry.catalog import (
    create_common_catalog,
    get_filters,
    map_columns,
    save_photfiles,
)
from st123.photometry.dolphot import (
    MosaicPhotJob,
    apply_mirimask,
    apply_nircammask,
    calc_sky,
    discover_mosaic_phot_jobs,
    dolphot_bin_dir,
    dolphot_command,
    phot_to_xyt,
    prepare_frames,
    prepare_mosaic_phot_job,
    resolve_dolphot_bin,
    science_fits_paths,
    setup_paramfile,
    write_paramfile,
)
from st123.photometry.warmstart import (
    WarmStartResult,
    discover_miri_jhat,
    setup_miri_warmstart,
)

__all__ = [
    'AB_ZEROPOINT_UJY',
    'ApertureParams',
    'MosaicPhotJob',
    'WarmStartResult',
    'apply_mirimask',
    'apply_nircammask',
    'calc_sky',
    'create_common_catalog',
    'default_ee_fraction',
    'discover_miri_jhat',
    'discover_mosaic_phot_jobs',
    'dolphot_bin_dir',
    'dolphot_command',
    'forced_aperture_photometry',
    'get_aperture_params',
    'get_filters',
    'map_columns',
    'mjy_sr_to_ujy_arcsec2',
    'phot_to_xyt',
    'pixel_scale_arcsec',
    'prepare_frames',
    'prepare_mosaic_phot_job',
    'read_coords_table',
    'resolve_dolphot_bin',
    'resolve_positions',
    'save_photfiles',
    'science_fits_paths',
    'setup_miri_warmstart',
    'setup_paramfile',
    'surface_brightness_to_ujy',
    'ujy_to_abmag',
    'write_paramfile',
]
