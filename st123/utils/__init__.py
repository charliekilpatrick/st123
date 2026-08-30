"""
Shared utilities, CLI helpers, link helpers, and JHAT/DOLPHOT parameter sets.

Submodules
----------
- :mod:`st123.utils.helpers` - coordinates, FITS bookkeeping, visits, xmatch
- :mod:`st123.utils.link` - symlink helpers for reduction ``raw/`` trees
- :mod:`st123.utils.settings` - filters, MAST/alignment defaults, JHAT/DOLPHOT params
- :mod:`st123.utils.compatibility` - cross-package compatibility adapters
- :mod:`st123.utils.logging` - console/file logging; captures external stdout

Exports are resolved lazily so ``import st123.utils.logging`` does not pull in
``helpers`` (and its MAST/shapely dependency chain).
"""

from __future__ import annotations

from typing import Any

_HELPER_EXPORTS = frozenset(
    {
        'create_filter_table',
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
        'organize_reduction_tables',
        'organize_visit_tables',
        'parse_coord',
        'pick_deepest_images',
        'xmatch_common',
    }
)
_LINK_EXPORTS = frozenset({'create_symlink', 'remove_proc_files'})
_SETTINGS_EXPORTS = frozenset(
    {
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
        'long_params',
        'CROWDED_JHAT_NBRIGHT',
        'crowded_jwst_params',
        'relaxed_gaia_params',
        'relaxed_jwst_params',
        'short_params',
        'strict_gaia_params',
        'strict_jwst_params',
    }
)

__all__ = sorted(_HELPER_EXPORTS | _LINK_EXPORTS | _SETTINGS_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _HELPER_EXPORTS:
        from st123.utils import helpers as _helpers

        return getattr(_helpers, name)
    if name in _LINK_EXPORTS:
        from st123.utils import link as _link

        return getattr(_link, name)
    if name in _SETTINGS_EXPORTS:
        from st123.utils import settings as _settings

        return getattr(_settings, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
