"""
JWST / HST alignment drivers: JHAT relative align, MIRI pipeline, visit
alignment, and HST Gaia alignment.

Imports are lazy so ``import st123.stages.alignment.hst_jhat`` (HST CLI path) does not
pull in the JWST Image3 / CRDS stack from :mod:`st123.stages.alignment.align`.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    'FILTER_MAX_REFERENCE_DISPERSION_MAS',
    'AlignmentSummaryRow',
    'CalibratorSettings',
    'FrameOverlaps',
    'SuccessfulAlignment',
    'add_bin_dq',
    'align_from_frames',
    'HST_ABS_OFFSET_MAX_ARCSEC',
    'HST_INTERNAL_ALIGN_MAX_ARCSEC',
    'HST_L3_ALIGN_MAX_ARCSEC',
    'FRAME_ABS_TOL_ARCSEC',
    'FRAME_INTERNAL_TOL_ARCSEC',
    'FRAME_SPARSE_TOL_ARCSEC',
    'align_hst_image',
    'align_hst_raw_dir',
    'align_jwst_image',
    'align_to_mosaic',
    'build_frame_qa',
    'ensure_wfpc2_jhat_patch',
    'find_hst_abs_ref_image',
    'find_hst_l3_refcat',
    'harmonize_hst_group_wcs',
    'harmonize_hst_jhat_dir',
    'measure_hst_sky_offset_2dhist',
    'validate_hst_coadds_alignment',
    'validate_hst_group_internal_alignment',
    'build_master_ref_catalog',
    'build_ref_catalog',
    'calc_dispersion',
    'calibrator_settings_for_filter',
    'combine_dispersion_mas',
    'count_alignment_calibrators',
    'create_alignment_mosaic',
    'discover_miri_images',
    'discover_ref_images',
    'expand_mask',
    'filter_wavelength_um',
    'find_frame_overlaps',
    'generate_level3_mosaic',
    'guess_shift',
    'harvest_alignment_metrics',
    'is_level3_i2d',
    'jwncal_is_plausible',
    'jwst_dispersion',
    'jwst_phot',
    'max_reference_dispersion_mas',
    'parse_filters_arg',
    'peer_sky_dispersion_mas',
    'query_gaia',
    'rank_fallback_parents',
    'read_dispersion_mas',
    'read_frame_qa',
    'refine_alignment_iteratively',
    'run_alignment',
    'run_jhat',
    'run_nircam_align_job',
    'run_overlaps',
    'run_reference_align_job',
    'select_fallback_parent',
    'stamp_quality_headers',
    'wfpc2_filter_key_and_name',
    'write_alignment_provenance',
    'write_alignment_summary',
    'write_frame_qa',
    'write_hst_alignment_summary',
]

_HST_EXPORTS = frozenset(
    {
        'HST_ABS_OFFSET_MAX_ARCSEC',
        'HST_INTERNAL_ALIGN_MAX_ARCSEC',
        'HST_L3_ALIGN_MAX_ARCSEC',
        'align_hst_image',
        'align_hst_raw_dir',
        'ensure_wfpc2_jhat_patch',
        'find_hst_abs_ref_image',
        'find_hst_l3_refcat',
        'harmonize_hst_group_wcs',
        'harmonize_hst_jhat_dir',
        'measure_hst_sky_offset_2dhist',
        'validate_hst_coadds_alignment',
        'validate_hst_group_internal_alignment',
        'wfpc2_filter_key_and_name',
        'write_hst_alignment_summary',
    }
)

_FRAME_QA_EXPORTS = frozenset(
    {
        'FRAME_ABS_TOL_ARCSEC',
        'FRAME_INTERNAL_TOL_ARCSEC',
        'FRAME_SPARSE_TOL_ARCSEC',
        'build_frame_qa',
        'read_frame_qa',
        'stamp_quality_headers',
        'write_frame_qa',
    }
)

_GAIA_EXPORTS = frozenset({'query_gaia'})


def __getattr__(name: str) -> Any:
    if name in _HST_EXPORTS:
        from st123.stages.alignment import hst_jhat as _hst

        return getattr(_hst, name)
    if name in _FRAME_QA_EXPORTS:
        from st123.stages.alignment import frame_qa as _fq

        return getattr(_fq, name)
    if name in _GAIA_EXPORTS:
        from st123.stages.alignment import gaia_catalog as _gaia

        return getattr(_gaia, name)
    if name in __all__:
        from st123.stages.alignment import align as _align

        return getattr(_align, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
