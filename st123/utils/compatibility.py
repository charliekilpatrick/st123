"""
Cross-package compatibility helpers.

Home for small adapters that keep st123 working when pinned third-party
stacks disagree (API renames, shape contracts, etc.). Add new package
bridges here rather than scattering one-off shims.

Current contents
----------------
``ensure_local_crds_context`` pins ``CRDS_CONTEXT`` to a mapping that exists
under ``CRDS_PATH`` when the default/server context is missing (common on
partially synced offline caches).

``jwst`` 1.20.x was written against photutils <3. With ``photutils>=3`` two
failures show up in Image3 ``source_catalog``:

1. ``tweakreg_catalog._sourcefinder_wrapper`` still defaults to ``npixels=``;
   ``SourceFinder`` requires ``n_pixels``, and the old key is dropped when
   kwargs are filtered by ``inspect.signature``.

2. ``JWSTSourceCatalog.xypos`` does ``np.transpose((xcentroid, ycentroid))``.
   For a single detection those centroids are scalars, so ``xypos`` becomes
   shape ``(2,)``. ``CircularAnnulus`` then treats the aperture as scalar and
   ``to_mask()`` returns one ``ApertureMask``, which is not iterable — breaking
   ``_aper_local_background`` (``TypeError: 'ApertureMask' object is not
   iterable``).

:func:`patch_jwst_for_photutils3` patches both sites once so Image3 /
SourceCatalogStep works with the pinned st123 stack (``jwst==1.20.2``,
``photutils==3.0.0``).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

_PATCHED = False
logger = logging.getLogger(__name__)


def ensure_local_crds_context(
    observatory: str = 'jwst',
    preferred: str | None = None,
) -> str | None:
    """
    Ensure ``CRDS_CONTEXT`` names a mapping file present under ``CRDS_PATH``.

    When the CRDS default / server operational context is not cached (and the
    network cannot sync it), Image3 fails with ``CrdsDownloadError``. This
    helper pins ``CRDS_CONTEXT`` to ``preferred`` when that file exists,
    otherwise the current default if cached, otherwise the newest local
    ``{observatory}_NNNN.pmap``.

    Parameters
    ----------
    observatory : str, optional
        CRDS observatory name (``jwst``, ``hst``, …).
    preferred : str, optional
        Preferred context file name (e.g. ``jwst_1464.pmap`` from input
        ``CRDS_CTX`` headers). Used when present on disk.

    Returns
    -------
    str or None
        Context file name in effect, or ``None`` if no local mapping is found.
    """
    crds_path = Path(os.environ.get('CRDS_PATH', '') or '')
    mappings = crds_path / 'mappings' / observatory
    if not mappings.is_dir():
        return os.environ.get('CRDS_CONTEXT')

    def _exists(name: str | None) -> bool:
        return bool(name) and (mappings / str(name)).is_file()

    current = os.environ.get('CRDS_CONTEXT')
    if _exists(current):
        return str(current)

    if preferred and _exists(preferred):
        os.environ['CRDS_CONTEXT'] = str(preferred)
        logger.info('Pinned CRDS_CONTEXT=%s (preferred local mapping)', preferred)
        return str(preferred)

    default = None
    try:
        import crds

        default = crds.get_default_context(observatory)
    except Exception:
        default = None
    if _exists(default):
        if current and current != default:
            os.environ['CRDS_CONTEXT'] = str(default)
            logger.info(
                'Pinned CRDS_CONTEXT=%s (default; %s missing locally)',
                default,
                current,
            )
        return str(default)

    local = list(mappings.glob(f'{observatory}_*.pmap'))
    if not local:
        return current

    def _version(path: Path) -> int:
        match = re.search(r'_(\d+)\.pmap$', path.name)
        return int(match.group(1)) if match else -1

    best = max(local, key=_version).name
    os.environ['CRDS_CONTEXT'] = best
    logger.warning(
        'Pinned CRDS_CONTEXT=%s; configured context %s is not cached under %s',
        best,
        current or default or '(none)',
        mappings,
    )
    return best

# photutils <3 → >=3 names used by SourceFinder / SourceCatalog.
_PHOTUTILS3_ALIASES = {
    'npixels': 'n_pixels',
    'nlevels': 'n_levels',
    'nproc': 'n_processes',
    'localbkg_width': 'local_bkg_width',
    'apermask_method': 'aperture_mask_method',
}


def _translate_photutils3_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of kwargs with photutils>=3 names preferred.

    Parameters
    ----------
    kwargs : dict
        Source-finder keyword arguments possibly using photutils<3 names.

    Returns
    -------
    dict
        Keyword arguments with legacy aliases translated or dropped.
    """
    out = dict(kwargs)
    for old, new in _PHOTUTILS3_ALIASES.items():
        if new not in out and old in out:
            out[new] = out.pop(old)
        else:
            out.pop(old, None)
    return out


def _patch_sourcefinder_wrapper() -> None:
    import jwst.tweakreg.tweakreg_catalog as tc

    # Newer jwst already normalizes SourceFinder kwargs.
    if getattr(tc, 'PHOTUTILS_GE_3', None) is not None:
        return

    original = tc._sourcefinder_wrapper

    def _sourcefinder_wrapper_photutils3(
        data, threshold_img, kernel_fwhm, mask=None, **kwargs
    ):
        kwargs = _translate_photutils3_kwargs(kwargs)
        if 'n_pixels' not in kwargs:
            kwargs['n_pixels'] = 10
        return original(data, threshold_img, kernel_fwhm, mask=mask, **kwargs)

    tc._sourcefinder_wrapper = _sourcefinder_wrapper_photutils3


def _patch_source_catalog_xypos() -> None:
    """Ensure ``xypos`` is always ``(N, 2)`` so aperture masks are a list."""
    import numpy as np
    from astropy.utils import lazyproperty
    from jwst.source_catalog.source_catalog import JWSTSourceCatalog

    @lazyproperty
    def xypos(self):
        """
        Return the (x, y) source centroids as an ``(N, 2)`` array.

        ``np.atleast_2d`` keeps the single-source case from collapsing to
        shape ``(2,)``, which makes photutils>=3 treat the aperture as scalar.
        """
        return np.atleast_2d(np.transpose((self.xcentroid, self.ycentroid)))

    JWSTSourceCatalog.xypos = xypos


def patch_jwst_for_photutils3() -> bool:
    """
    Apply photutils>=3 compatibility patches to installed ``jwst``.

    Returns
    -------
    bool
        ``True`` if patches were applied (or already applied), ``False`` if
        the installed stack does not need them.
    """
    global _PATCHED
    if _PATCHED:
        return True

    try:
        import photutils
        from packaging.version import Version
    except ImportError:
        return False

    if Version(photutils.__version__.split('+')[0]) < Version('3'):
        return False

    try:
        _patch_sourcefinder_wrapper()
        _patch_source_catalog_xypos()
    except ImportError:
        return False

    _PATCHED = True
    return True
