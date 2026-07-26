"""
Cross-package compatibility helpers.

Home for small adapters that keep st123 working when pinned third-party
stacks disagree (API renames, shape contracts, etc.). Add new package
bridges here rather than scattering one-off shims.

Current contents
----------------
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

from typing import Any

_PATCHED = False

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
