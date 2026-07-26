"""Tests for cross-package compatibility helpers."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import photutils
import pytest
from astropy.table import Table
from packaging.version import Version
from photutils.aperture import CircularAnnulus
from photutils.segmentation import SourceFinder

from st123.utils import compatibility


@pytest.mark.skipif(
    Version(photutils.__version__.split('+')[0]) < Version('3'),
    reason='photutils>=3 required for this regression',
)
def test_translate_photutils3_kwargs():
    out = compatibility._translate_photutils3_kwargs(
        {'npixels': 25, 'nlevels': 32, 'progress_bar': False}
    )
    assert out['n_pixels'] == 25
    assert out['n_levels'] == 32
    assert 'npixels' not in out
    assert out['progress_bar'] is False


@pytest.mark.skipif(
    Version(photutils.__version__.split('+')[0]) < Version('3'),
    reason='photutils>=3 required for this regression',
)
def test_patch_prevents_missing_n_pixels_error():
    """Simulate jwst 1.20.x defaults + photutils 3 SourceFinder signature."""
    import jwst.tweakreg.tweakreg_catalog as tc
    from jwst.source_catalog.source_catalog import JWSTSourceCatalog

    calls = {}

    def fake_original(data, threshold_img, kernel_fwhm, mask=None, **kwargs):
        # Mimic jwst 1.20.2: default npixels, then keep only signature kwargs.
        default_kwargs = {'npixels': 10, 'progress_bar': False}
        kwargs = {**default_kwargs, **kwargs}
        finder_args = list(inspect.signature(SourceFinder).parameters)
        finder_dict = {k: kwargs[k] for k in dict(kwargs) if k in finder_args}
        SourceFinder(**finder_dict)  # raises without n_pixels
        calls['finder_dict'] = finder_dict
        return MagicMock(), MagicMock()

    real_wrapper = tc._sourcefinder_wrapper
    real_xypos = JWSTSourceCatalog.xypos
    saved_flag = getattr(tc, 'PHOTUTILS_GE_3', 'MISSING')
    compatibility._PATCHED = False
    try:
        if hasattr(tc, 'PHOTUTILS_GE_3'):
            delattr(tc, 'PHOTUTILS_GE_3')
        with patch.object(tc, '_sourcefinder_wrapper', fake_original):
            assert compatibility.patch_jwst_for_photutils3() is True
            tc._sourcefinder_wrapper(
                MagicMock(), MagicMock(), 2.0, npixels=25, nlevels=32
            )
    finally:
        tc._sourcefinder_wrapper = real_wrapper
        JWSTSourceCatalog.xypos = real_xypos
        if saved_flag == 'MISSING':
            if hasattr(tc, 'PHOTUTILS_GE_3'):
                delattr(tc, 'PHOTUTILS_GE_3')
        else:
            tc.PHOTUTILS_GE_3 = saved_flag
        compatibility._PATCHED = False

    assert calls['finder_dict']['n_pixels'] == 25


@pytest.mark.skipif(
    Version(photutils.__version__.split('+')[0]) < Version('3'),
    reason='photutils>=3 required for this regression',
)
def test_xypos_patch_keeps_single_source_masks_iterable():
    """Single-source centroids must yield a list of ApertureMask objects."""
    from jwst.source_catalog.source_catalog import JWSTSourceCatalog

    real_xypos = JWSTSourceCatalog.xypos
    compatibility._PATCHED = False
    try:
        assert compatibility.patch_jwst_for_photutils3() is True

        class _FakeCat:
            xcentroid = 10.0
            ycentroid = 20.0

        xy = JWSTSourceCatalog.xypos.fget(_FakeCat())
        assert xy.shape == (1, 2)
        masks = CircularAnnulus(xy, r_in=3.0, r_out=5.0).to_mask(method='center')
        assert isinstance(masks, list)
        assert len(masks) == 1
    finally:
        JWSTSourceCatalog.xypos = real_xypos
        compatibility._PATCHED = False


def test_atleast_2d_xypos_shape_contract():
    """Document the N=1 collapse that photutils>=3 treats as a scalar aperture."""
    collapsed = np.transpose((10.0, 20.0))
    assert collapsed.shape == (2,)
    fixed = np.atleast_2d(collapsed)
    assert fixed.shape == (1, 2)


@pytest.mark.skipif(
    Version(photutils.__version__.split('+')[0]) < Version('3'),
    reason='photutils>=3 required for this regression',
)
def test_unpatched_single_source_aperture_mask_not_iterable():
    """
    Upstream failure mode from Image3 source_catalog under photutils 3.

    ``JWSTSourceCatalog.xypos`` used ``np.transpose((x, y))``; with one source
    that is shape ``(2,)``, ``CircularAnnulus.to_mask`` returns a scalar
    ``ApertureMask``, and ``for mask in bkg_aper_masks`` raises TypeError.
    """
    xy = np.transpose((32.0, 32.0))
    assert xy.shape == (2,)
    masks = CircularAnnulus(xy, r_in=3.0, r_out=5.0).to_mask(method='center')
    with pytest.raises(TypeError, match='not iterable'):
        for _mask in masks:
            pass


@pytest.mark.skipif(
    Version(photutils.__version__.split('+')[0]) < Version('3'),
    reason='photutils>=3 required for this regression',
)
def test_aper_local_background_single_source_after_compat_patch():
    """
    Regression for align visit mosaics: patched xypos must let
    ``_aper_local_background`` finish for a single detection.
    """
    from astropy import units as u
    from jwst.source_catalog.source_catalog import JWSTSourceCatalog

    real_xypos = JWSTSourceCatalog.xypos
    compatibility._PATCHED = False
    try:
        assert compatibility.patch_jwst_for_photutils3() is True

        data = np.random.default_rng(0).normal(loc=1.0, scale=0.05, size=(64, 64))
        model = MagicMock()
        model.data.value = data
        model.data.unit = u.MJy / u.sr

        class _SingleSourceStub:
            # Scalars reproduce the photutils>=3 collapse without the patch.
            xcentroid = 32.0
            ycentroid = 32.0
            aperture_params = {
                'bkg_aperture_inner_radius': 4.0,
                'bkg_aperture_outer_radius': 7.0,
            }

            def __init__(self, image_model):
                self.model = image_model

            @property
            def xypos(self):
                return JWSTSourceCatalog.xypos.fget(self)

            @property
            def _xypos_finite(self):
                return JWSTSourceCatalog._xypos_finite.fget(self)

        stub = _SingleSourceStub(model)
        assert stub.xypos.shape == (1, 2)

        bkg_median, bkg_median_err = JWSTSourceCatalog._aper_local_background.fget(
            stub
        )
        assert len(bkg_median) == 1
        assert len(bkg_median_err) == 1
        assert np.isfinite(bkg_median.value[0])
    finally:
        JWSTSourceCatalog.xypos = real_xypos
        compatibility._PATCHED = False


def test_generate_level3_mosaic_applies_photutils3_compat(tmp_path: Path):
    """Visit-align mosaics must install the compat patches before Image3 runs."""
    from st123.alignment import align as align_lib

    table = Table(
        {
            'filter': ['f200w'],
            'image': [str(tmp_path / 'a.fits')],
        }
    )
    asn = MagicMock()
    asn.dump.return_value = ('name', '{}')
    pipe = MagicMock()

    with (
        patch.object(align_lib, 'patch_jwst_for_photutils3') as mock_patch,
        patch.object(align_lib, 'input_list', return_value=table),
        patch.object(align_lib.asn_from_list, 'asn_from_list', return_value=asn),
        patch.object(align_lib.calwebb_image3, 'Image3Pipeline', return_value=pipe),
    ):
        out = align_lib.generate_level3_mosaic(
            [str(tmp_path / 'a.fits')], str(tmp_path / 'mosaic')
        )

    mock_patch.assert_called_once()
    pipe.run.assert_called_once()
    assert out.endswith('f200w_i2d.fits')
