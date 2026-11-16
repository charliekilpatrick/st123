"""
st123: space-telescope download, JHAT alignment, mosaicking, and DOLPHOT helpers.

Heavy alignment / JWST / JHAT imports are deferred so console scripts can start
and log before loading the science stack (~tens of seconds otherwise).
"""

from __future__ import annotations

from typing import Any

try:
    from st123._version import version as __version__
except ImportError:  # pragma: no cover - editable/source tree without build
    try:
        from importlib.metadata import PackageNotFoundError, version as _pkg_version

        __version__ = _pkg_version('st123')
    except PackageNotFoundError:
        __version__ = '0.0.0+unknown'

# Public names historically re-exported from :mod:`st123.stages.alignment`.
_ALIGNMENT_EXPORTS = frozenset(
    {
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
    }
)

__all__ = [
    *_ALIGNMENT_EXPORTS,
    '__version__',
]


def __getattr__(name: str) -> Any:
    if name in _ALIGNMENT_EXPORTS:
        from st123.stages import alignment as _alignment

        return getattr(_alignment, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
