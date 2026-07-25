"""
st123: space-telescope download, JHAT alignment, mosaicking, and DOLPHOT helpers.
"""

from st123.alignment import (
    add_bin_dq,
    align_from_frames,
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
    run_alignment,
    run_jhat,
    run_overlaps,
)

try:
    from st123._version import version as __version__
except ImportError:  # pragma: no cover - editable/source tree without build
    try:
        from importlib.metadata import PackageNotFoundError, version as _pkg_version

        __version__ = _pkg_version('st123')
    except PackageNotFoundError:
        __version__ = '0.0.0+unknown'

__all__ = [
    'add_bin_dq',
    'align_from_frames',
    'align_jwst_image',
    'align_to_mosaic',
    'calc_dispersion',
    'create_alignment_mosaic',
    'expand_mask',
    'generate_level3_mosaic',
    'guess_shift',
    'jwst_dispersion',
    'jwst_phot',
    'query_gaia',
    'run_alignment',
    'run_jhat',
    'run_overlaps',
    '__version__',
]
