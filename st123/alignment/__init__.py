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

CLI wrappers live under :mod:`st123.scripts` (``relative_align``,
``alignment_wrap``, ``align``).
"""

from __future__ import annotations

from st123.alignment.align import (
    add_bin_dq,
    align_jwst_image,
    align_to_mosaic,
    calc_dispersion,
    create_alignment_mosaic,
    expand_mask,
    generate_level3_mosaic,
    guess_shift,
    jwst_dispersion,
    jwst_phot,
    query_gaia,
    run_jhat,
)
from st123.alignment.alignment_fallback import (
    SuccessfulAlignment,
    combine_dispersion_mas,
    filter_wavelength_um,
    rank_fallback_parents,
    select_fallback_parent,
    write_alignment_provenance,
)
from st123.alignment.alignment_wrap import (
    AlignmentSummaryRow,
    FrameOverlaps,
    align_from_frames,
    discover_miri_images,
    discover_ref_images,
    find_frame_overlaps,
    harvest_alignment_metrics,
    parse_filters_arg,
    run_overlaps,
    write_alignment_summary,
)
from st123.alignment.calibrators import (
    FILTER_MAX_REFERENCE_DISPERSION_MAS,
    CalibratorSettings,
    calibrator_settings_for_filter,
    max_reference_dispersion_mas,
)
from st123.alignment.relative_align import (
    build_master_ref_catalog,
    build_ref_catalog,
    read_dispersion_mas,
    refine_alignment_iteratively,
    run_alignment,
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
    'create_alignment_mosaic',
    'discover_miri_images',
    'discover_ref_images',
    'expand_mask',
    'filter_wavelength_um',
    'find_frame_overlaps',
    'generate_level3_mosaic',
    'guess_shift',
    'harvest_alignment_metrics',
    'jwst_dispersion',
    'jwst_phot',
    'max_reference_dispersion_mas',
    'parse_filters_arg',
    'query_gaia',
    'rank_fallback_parents',
    'read_dispersion_mas',
    'refine_alignment_iteratively',
    'run_alignment',
    'run_jhat',
    'run_overlaps',
    'select_fallback_parent',
    'write_alignment_provenance',
    'write_alignment_summary',
]
