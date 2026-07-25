"""
Photometry helpers for individual images and combined catalogs.

Submodules
----------
- :mod:`st123.photometry.catalog` — DOLPHOT column mapping and combined catalogs
- :mod:`st123.photometry.dolphot_prep` — mask / calcsky / paramfile prep
- :mod:`st123.photometry.warmstart` — NIRCam→MIRI warm-start run setup
"""

from __future__ import annotations

from st123.photometry.catalog import (
    create_common_catalog,
    get_filters,
    map_columns,
    save_photfiles,
)
from st123.photometry.dolphot_prep import (
    apply_mirimask,
    apply_nircammask,
    calc_sky,
    dolphot_command,
    phot_to_xyt,
    prepare_frames,
    setup_paramfile,
    write_paramfile,
)
from st123.photometry.warmstart import (
    WarmStartResult,
    discover_miri_jhat,
    setup_miri_warmstart,
)

__all__ = [
    'WarmStartResult',
    'apply_mirimask',
    'apply_nircammask',
    'calc_sky',
    'create_common_catalog',
    'discover_miri_jhat',
    'dolphot_command',
    'get_filters',
    'map_columns',
    'phot_to_xyt',
    'prepare_frames',
    'save_photfiles',
    'setup_miri_warmstart',
    'setup_paramfile',
    'write_paramfile',
]
