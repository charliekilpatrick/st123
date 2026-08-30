"""
MAST query, filter, and download helpers for HST and JWST imaging.

Submodules
----------
- :mod:`st123.mast.mast` — MAST auth, region queries, product filters, S_REGION
- :mod:`st123.mast.download` — high-level download orchestration

Exports are lazy so importing :mod:`st123.mast.mast` (e.g. ``parse_s_region``)
does not pull the download orchestration module.
"""

from __future__ import annotations

from typing import Any

_DOWNLOAD_EXPORTS = frozenset(
    {
        'MastDownloadResult',
        'query_and_download_miri',
        'query_mast_hst',
        'query_mast_jwst',
        'resolve_outdir',
        'suppress_stdout',
    }
)
_MAST_EXPORTS = frozenset(
    {
        'DEFAULT_DOWNLOAD_LAYOUT',
        'DEFAULT_HST_FILTERS',
        'DEFAULT_HST_INSTRUMENTS',
        'DEFAULT_JWST_INSTRUMENTS',
        'HST_PRODUCT_RULES',
        'collect_hst_products',
        'coverage_fraction',
        'download_hst_observations',
        'download_jwst_observations',
        'filter_hst_observations',
        'filter_hst_products',
        'filter_jwst_observations',
        'filter_jwst_observations_by_stage',
        'filter_jwst_products',
        'observation_matches_calib_stage',
        'galaxy_query_radius',
        'is_hst_science_product',
        'mast_login',
        'prepare_mast_auth',
        'reset_mast_login_state',
        'normalize_filter_name',
        'normalize_instrument_dirname',
        'normalize_telescope_dirname',
        'observation_download_subdir',
        'parse_s_region',
        'polygons_from_obs_table',
        'prune_non_full_frame_miri',
        'query_hst',
        'query_jwst',
        'query_region',
        'resolve_mast_token',
    }
)

__all__ = sorted(_DOWNLOAD_EXPORTS | _MAST_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _DOWNLOAD_EXPORTS:
        from st123.mast import download as _mod

        return getattr(_mod, name)
    if name in _MAST_EXPORTS:
        from st123.mast import mast as _mod

        return getattr(_mod, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
