"""
Photometry helpers for individual images and combined catalogs.

Submodules
----------
- :mod:`st123.photometry.aperture` — forced EE aperture photometry on coadds
- :mod:`st123.photometry.catalog` — DOLPHOT column mapping and combined catalogs
- :mod:`st123.photometry.dolphot` — mask / calcsky / paramfile prep
- :mod:`st123.photometry.dolphot_split` — split / merge large DOLPHOT runs
- :mod:`st123.photometry.warmstart` — NIRCam→MIRI warm-start run setup

Exports are lazy so ``dolphot-prep`` need not load aperture / catalog code at
import time.
"""

from __future__ import annotations

from typing import Any

_APERTURE_EXPORTS = frozenset(
    {
        'AB_ZEROPOINT_UJY',
        'ApertureParams',
        'default_ee_fraction',
        'forced_aperture_photometry',
        'get_aperture_params',
        'mjy_sr_to_ujy_arcsec2',
        'pixel_scale_arcsec',
        'read_coords_table',
        'resolve_positions',
        'surface_brightness_to_ujy',
        'ujy_to_abmag',
    }
)
_CATALOG_EXPORTS = frozenset(
    {
        'create_common_catalog',
        'get_filters',
        'map_columns',
        'save_photfiles',
    }
)
_DOLPHOT_EXPORTS = frozenset(
    {
        'MosaicPhotJob',
        'apply_hst_mask',
        'apply_mirimask',
        'apply_nircammask',
        'apply_splitgroups',
        'calc_sky',
        'discover_mosaic_phot_jobs',
        'dolphot_bin_dir',
        'dolphot_command',
        'nearest_phot_source',
        'phot_to_xyt',
        'prepare_frames',
        'prepare_hst_frames',
        'prepare_mosaic_phot_job',
        'resolve_dolphot_bin',
        'sanitize_dolphot_wcs',
        'science_fits_paths',
        'setup_paramfile',
        'write_paramfile',
    }
)
_SPLIT_EXPORTS = frozenset(
    {
        'DOLPHOT_COMPILE_MAX_NIMG',
        'DOLPHOT_MAX_NIMG',
        'DolphotRunPlan',
        'chunk_images',
        'finalize_split_outdir',
        'merge_dolphot_phot_catalogs',
        'write_split_paramfiles',
    }
)
_WARMSTART_EXPORTS = frozenset(
    {
        'WarmStartResult',
        'discover_hst_jhat',
        'discover_miri_jhat',
        'setup_hst_warmstart',
        'setup_miri_warmstart',
    }
)

__all__ = sorted(
    _APERTURE_EXPORTS
    | _CATALOG_EXPORTS
    | _DOLPHOT_EXPORTS
    | _SPLIT_EXPORTS
    | _WARMSTART_EXPORTS
)


def __getattr__(name: str) -> Any:
    if name in _APERTURE_EXPORTS:
        from st123.photometry import aperture as _mod

        return getattr(_mod, name)
    if name in _CATALOG_EXPORTS:
        from st123.photometry import catalog as _mod

        return getattr(_mod, name)
    if name in _DOLPHOT_EXPORTS:
        from st123.photometry import dolphot as _mod

        return getattr(_mod, name)
    if name in _SPLIT_EXPORTS:
        from st123.photometry import dolphot_split as _mod

        return getattr(_mod, name)
    if name in _WARMSTART_EXPORTS:
        from st123.photometry import warmstart as _mod

        return getattr(_mod, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
