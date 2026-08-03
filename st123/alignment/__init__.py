"""
JWST / HST alignment drivers: JHAT relative align, MIRI pipeline, visit
alignment, and HST Gaia alignment.

Primary entry points
--------------------
- :func:`align_jwst_image` — single-image JHAT alignment (JWST)
- :func:`align_hst_image` / :func:`align_hst_raw_dir` — HST JHAT (Gaia)
- :func:`run_alignment` — relative align + optional iterative refine
- :func:`align_from_frames` — filter-wave REFERENCE / MIRI_REL orchestration
- :func:`run_overlaps` — MIRI↔reference footprint overlap discovery
- :func:`rank_fallback_parents` — MIRI→MIRI parent ranking
- :func:`calibrator_settings_for_filter` — per-filter JHAT / refine knobs

Visit and reference share the pool/logging/JHAT/metrics contract in
:mod:`st123.alignment.align`; they diverge on topology (mosaic cascade vs
overlap + MIRI_REL). HST batch alignment lives in :mod:`st123.alignment.hst_jhat`.
CLI wrappers live under :mod:`st123.scripts` (``align``, …).
"""

from __future__ import annotations

from st123.alignment.hst_jhat import (
    HST_ABS_OFFSET_MAX_ARCSEC,
    HST_INTERNAL_ALIGN_MAX_ARCSEC,
    HST_L3_ALIGN_MAX_ARCSEC,
    align_hst_image,
    align_hst_raw_dir,
    ensure_wfpc2_jhat_patch,
    find_hst_abs_ref_image,
    find_hst_l3_refcat,
    harmonize_hst_group_wcs,
    harmonize_hst_jhat_dir,
    measure_hst_sky_offset_2dhist,
    validate_hst_coadds_alignment,
    validate_hst_group_internal_alignment,
    wfpc2_filter_key_and_name,
    write_hst_alignment_summary,
)
from st123.alignment.align import (
    FILTER_MAX_REFERENCE_DISPERSION_MAS,
    AlignmentSummaryRow,
    CalibratorSettings,
    FrameOverlaps,
    SuccessfulAlignment,
    add_bin_dq,
    align_from_frames,
    align_jwst_image,
    align_to_mosaic,
    build_master_ref_catalog,
    build_ref_catalog,
    calc_dispersion,
    calibrator_settings_for_filter,
    combine_dispersion_mas,
    count_alignment_calibrators,
    create_alignment_mosaic,
    discover_miri_images,
    discover_ref_images,
    expand_mask,
    filter_wavelength_um,
    find_frame_overlaps,
    generate_level3_mosaic,
    guess_shift,
    harvest_alignment_metrics,
    is_level3_i2d,
    jwncal_is_plausible,
    jwst_dispersion,
    jwst_phot,
    max_reference_dispersion_mas,
    parse_filters_arg,
    peer_sky_dispersion_mas,
    query_gaia,
    rank_fallback_parents,
    read_dispersion_mas,
    refine_alignment_iteratively,
    run_alignment,
    run_jhat,
    run_nircam_align_job,
    run_overlaps,
    run_reference_align_job,
    select_fallback_parent,
    write_alignment_provenance,
    write_alignment_summary,
)

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
    'align_hst_image',
    'align_hst_raw_dir',
    'align_jwst_image',
    'align_to_mosaic',
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
    'refine_alignment_iteratively',
    'run_alignment',
    'run_jhat',
    'run_nircam_align_job',
    'run_overlaps',
    'run_reference_align_job',
    'select_fallback_parent',
    'wfpc2_filter_key_and_name',
    'write_alignment_provenance',
    'write_alignment_summary',
    'write_hst_alignment_summary',
]
