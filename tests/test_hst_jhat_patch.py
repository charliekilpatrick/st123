"""Tests for :mod:`st123.alignment.hst_jhat` WFPC2 / filter helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from st123.alignment.hst_jhat import (
    ensure_wfpc2_jhat_patch,
    propagate_jhat_wcs_to_all_sci,
    wfpc2_filter_key_and_name,
)


def test_wfpc2_filter_prefers_filtnam_over_numeric_filter1():
    h = fits.Header()
    h['INSTRUME'] = 'WFPC2'
    h['FILTER1'] = 0  # unpatched JHAT: TypeError ('CLEAR' not in int)
    h['FILTNAM1'] = 'F300W'
    k, name = wfpc2_filter_key_and_name(h)
    assert k == 'FILTNAM1'
    assert name == 'F300W'


def test_wfpc2_filter_filtnam2_when_filtnam1_clear():
    h = fits.Header()
    h['FILTNAM1'] = 'CLEAR'
    h['FILTNAM2'] = 'F814W'
    k, name = wfpc2_filter_key_and_name(h)
    assert k == 'FILTNAM2'
    assert name == 'F814W'


def test_wfpc2_filter_fallback_clear():
    h = fits.Header()
    k, name = wfpc2_filter_key_and_name(h)
    assert k == 'FILTNAM1'
    assert name == 'CLEAR'


def test_ensure_wfpc2_jhat_patch_idempotent():
    pytest.importorskip('jhat')
    import jhat.simple_jwst_phot as sjp

    ensure_wfpc2_jhat_patch()
    first = sjp.hst_photclass.load_image
    assert getattr(first, '__name__', '') == 'load_image_st123_wfpc2'
    ensure_wfpc2_jhat_patch()
    second = sjp.hst_photclass.load_image
    assert second is first


def _wcs_header(*, crval1: float, crval2: float, crpix1: float = 50.0, crpix2: float = 50.0):
    h = fits.Header()
    h['NAXIS'] = 2
    h['NAXIS1'] = 100
    h['NAXIS2'] = 100
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    h['CRPIX1'] = crpix1
    h['CRPIX2'] = crpix2
    h['CRVAL1'] = crval1
    h['CRVAL2'] = crval2
    h['CD1_1'] = -1.0e-5
    h['CD1_2'] = 0.0
    h['CD2_1'] = 0.0
    h['CD2_2'] = 1.0e-5
    h['CUNIT1'] = 'deg'
    h['CUNIT2'] = 'deg'
    return h


def test_propagate_jhat_wcs_to_all_sci(tmp_path: Path):
    data = np.ones((100, 100), dtype=np.float32)
    # Source: 4 chips, identical CRVAL baseline with small chip-to-chip offsets.
    src = tmp_path / 'u_test_c0m.fits'
    aln = tmp_path / 'u_test_jhat.fits'
    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = 'WFPC2'
    src_hdus = [primary]
    aln_hdus = [fits.PrimaryHDU(header=primary.header.copy())]
    base_ra, base_dec = 177.66, 55.35
    for i, (dra, ddec) in enumerate(
        [(0.0, 0.0), (0.01, 0.0), (0.0, 0.01), (-0.01, 0.0)]
    ):
        hs = _wcs_header(crval1=base_ra + dra, crval2=base_dec + ddec)
        ha = _wcs_header(crval1=base_ra + dra, crval2=base_dec + ddec)
        if i == 0:
            # JHAT only tweaked SCI1: +1 arcsec in Dec.
            ha['CRVAL2'] = base_dec + ddec + (1.0 / 3600.0)
        src_hdus.append(fits.ImageHDU(data.copy(), header=hs, name='SCI'))
        aln_hdus.append(fits.ImageHDU(data.copy(), header=ha, name='SCI'))
    fits.HDUList(src_hdus).writeto(src)
    fits.HDUList(aln_hdus).writeto(aln)

    stats = propagate_jhat_wcs_to_all_sci(aln, src)
    assert stats['n_sci'] == 4
    assert stats['n_updated'] == 3

    with fits.open(aln) as hdul, fits.open(src) as src_hdul:
        assert hdul[0].header.get('ST123WPROP') is True
        for k in range(4):
            # All aligned SCI should now be ~1" north of source SCI.
            w0 = WCS(src_hdul[k + 1].header, naxis=2)
            w1 = WCS(hdul[k + 1].header, naxis=2)
            ra0, dec0 = w0.pixel_to_world_values(50, 50)
            ra1, dec1 = w1.pixel_to_world_values(50, 50)
            ddec_as = (dec1 - dec0) * 3600.0
            assert abs(ddec_as - 1.0) < 0.05
