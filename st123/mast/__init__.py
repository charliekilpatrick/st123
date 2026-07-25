"""
MAST query, filter, and download helpers for HST and JWST imaging.

Submodules
----------
- :mod:`st123.mast.mast` — MAST auth, region queries, product filters, S_REGION
- :mod:`st123.mast.download` — high-level JWST download orchestration
"""

from __future__ import annotations

from st123.mast.download import (
    query_and_download_miri,
    query_mast_jwst,
    resolve_outdir,
    suppress_stdout,
)
from st123.mast.mast import (
    DEFAULT_HST_FILTERS,
    DEFAULT_HST_INSTRUMENTS,
    DEFAULT_JWST_INSTRUMENTS,
    HST_PRODUCT_RULES,
    collect_hst_products,
    coverage_fraction,
    download_jwst_observations,
    filter_hst_observations,
    filter_jwst_observations,
    filter_jwst_products,
    galaxy_query_radius,
    is_hst_science_product,
    mast_login,
    normalize_filter_name,
    observation_download_subdir,
    parse_s_region,
    polygons_from_obs_table,
    query_hst,
    query_jwst,
    query_region,
    resolve_mast_token,
)

__all__ = [
    'DEFAULT_HST_FILTERS',
    'DEFAULT_HST_INSTRUMENTS',
    'DEFAULT_JWST_INSTRUMENTS',
    'HST_PRODUCT_RULES',
    'collect_hst_products',
    'coverage_fraction',
    'download_jwst_observations',
    'filter_hst_observations',
    'filter_jwst_observations',
    'filter_jwst_products',
    'galaxy_query_radius',
    'is_hst_science_product',
    'mast_login',
    'normalize_filter_name',
    'observation_download_subdir',
    'parse_s_region',
    'polygons_from_obs_table',
    'query_and_download_miri',
    'query_hst',
    'query_jwst',
    'query_mast_jwst',
    'query_region',
    'resolve_mast_token',
    'resolve_outdir',
    'suppress_stdout',
]
