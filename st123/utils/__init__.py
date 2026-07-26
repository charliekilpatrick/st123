"""
Shared utilities, CLI helpers, link helpers, and JHAT/DOLPHOT parameter sets.

Submodules
----------
- :mod:`st123.utils.helpers` — coordinates, FITS bookkeeping, visits, xmatch
  (merged former ``st123.util`` + ``st123.utils``)
- :mod:`st123.utils.link` — symlink helpers for reduction ``raw/`` trees
- :mod:`st123.utils.settings` — filters, MAST/alignment defaults, JHAT/DOLPHOT params
- :mod:`st123.utils.compatibility` — cross-package compatibility adapters
- :mod:`st123.utils.logging` — POTPyRI-style console/file logging; captures
  external stdout/stderr (JHAT, Image3, DOLPHOT, MAST) into the log file
"""

from __future__ import annotations

from st123.utils.helpers import (
    create_filter_table,
    edit_visits_groups,
    get_chip,
    get_detector_chip,
    get_filter,
    get_instrument,
    get_module,
    get_sky_pgons,
    get_zpt,
    input_list,
    is_number,
    organize_reduction_tables,
    organize_visit_tables,
    parse_coord,
    pick_deepest_images,
    xmatch_common,
)
from st123.utils.link import create_symlink, remove_proc_files
from st123.utils.settings import (
    BEST_FILTER_TYPES,
    BEST_REFERENCE_FILTERS,
    DEFAULT_DOWNLOAD_LAYOUT,
    DEFAULT_HST_FILTERS,
    DEFAULT_HST_INSTRUMENTS,
    DEFAULT_JWST_INSTRUMENTS,
    DEFAULT_MAX_REFERENCE_DISPERSION_MAS,
    DEFAULT_PAIR_OUTDIR,
    FILTER_MAX_REFERENCE_DISPERSION_MAS,
    FILTERS_BY_INSTRUMENT,
    HST_PRODUCT_RULES,
    acceptable_filters,
    base_params,
    long_params,
    relaxed_gaia_params,
    relaxed_jwst_params,
    short_params,
    strict_gaia_params,
    strict_jwst_params,
)

__all__ = [
    'BEST_FILTER_TYPES',
    'BEST_REFERENCE_FILTERS',
    'DEFAULT_DOWNLOAD_LAYOUT',
    'DEFAULT_HST_FILTERS',
    'DEFAULT_HST_INSTRUMENTS',
    'DEFAULT_JWST_INSTRUMENTS',
    'DEFAULT_MAX_REFERENCE_DISPERSION_MAS',
    'DEFAULT_PAIR_OUTDIR',
    'FILTER_MAX_REFERENCE_DISPERSION_MAS',
    'FILTERS_BY_INSTRUMENT',
    'HST_PRODUCT_RULES',
    'acceptable_filters',
    'base_params',
    'create_filter_table',
    'create_symlink',
    'edit_visits_groups',
    'get_chip',
    'get_detector_chip',
    'get_filter',
    'get_instrument',
    'get_module',
    'get_sky_pgons',
    'get_zpt',
    'input_list',
    'is_number',
    'long_params',
    'organize_reduction_tables',
    'organize_visit_tables',
    'parse_coord',
    'pick_deepest_images',
    'relaxed_gaia_params',
    'relaxed_jwst_params',
    'remove_proc_files',
    'short_params',
    'strict_gaia_params',
    'strict_jwst_params',
    'xmatch_common',
]
