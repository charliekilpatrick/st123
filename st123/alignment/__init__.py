"""
JWST alignment drivers: JHAT relative align, MIRI pipeline, and visit alignment.

Primary entry points
--------------------
- :func:`align_jwst_image` — single-image JHAT alignment
- :func:`run_alignment` — relative align + optional iterative refine
- :func:`align_from_frames` — filter-wave REFERENCE / MIRI_REL orchestration
- :func:`run_overlaps` — MIRI↔reference footprint overlap discovery
- :func:`rank_fallback_parents` — MIRI→MIRI parent ranking
- :func:`calibrator_settings_for_filter` — per-filter JHAT / refine knobs

Visit and reference share the pool/logging/JHAT/metrics contract in
:mod:`st123.alignment.align`; they diverge on topology (mosaic cascade vs
overlap + MIRI_REL). Everything lives in that module; this package re-exports
the public surface. CLI wrappers live under :mod:`st123.scripts` (``align``, …).
"""

from __future__ import annotations

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
    'align_jwst_image',
    'align_to_mosaic',
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
    'write_alignment_provenance',
    'write_alignment_summary',
]
