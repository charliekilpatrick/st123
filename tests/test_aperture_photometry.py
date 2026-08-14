"""Tests for forced coadd aperture photometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from st123.photometry import aperture as ap
from st123.scripts import coadd_phot as coadd_phot_script


def _wcs_header(nx: int = 64, ny: int = 64, pixscale_arcsec: float = 0.1) -> fits.Header:
    hdr = fits.Header()
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = nx
    hdr['NAXIS2'] = ny
    hdr['CTYPE1'] = 'RA---TAN'
    hdr['CTYPE2'] = 'DEC--TAN'
    hdr['CRPIX1'] = nx / 2.0
    hdr['CRPIX2'] = ny / 2.0
    hdr['CRVAL1'] = 180.0
    hdr['CRVAL2'] = 0.0
    hdr['CDELT1'] = -pixscale_arcsec / 3600.0
    hdr['CDELT2'] = pixscale_arcsec / 3600.0
    hdr['CUNIT1'] = 'deg'
    hdr['CUNIT2'] = 'deg'
    hdr['BUNIT'] = 'MJy/sr'
    return hdr


def _write_synthetic_coadd(
    path: Path,
    *,
    nx: int = 64,
    ny: int = 64,
    pixscale_arcsec: float = 0.1,
    peak_mjy_sr: float = 10.0,
    x0: float = 32.0,
    y0: float = 32.0,
    fwhm_px: float = 3.0,
) -> Path:
    yy, xx = np.mgrid[0:ny, 0:nx]
    sigma = fwhm_px / 2.355
    data = peak_mjy_sr * np.exp(
        -0.5 * (((xx - x0) / sigma) ** 2 + ((yy - y0) / sigma) ** 2)
    )
    err = np.full_like(data, 0.05)
    hdr = _wcs_header(nx=nx, ny=ny, pixscale_arcsec=pixscale_arcsec)
    primary = fits.PrimaryHDU(
        header=fits.Header(
            {
                'INSTRUME': 'NIRCAM',
                'FILTER': 'F200W',
                'TELESCOP': 'JWST',
            }
        )
    )
    sci = fits.ImageHDU(data=data.astype(np.float32), header=hdr, name='SCI')
    err_hdu = fits.ImageHDU(data=err.astype(np.float32), header=hdr, name='ERR')
    fits.HDUList([primary, sci, err_hdu]).writeto(path, overwrite=True)
    return path


def test_default_ee_fraction():
    assert ap.default_ee_fraction('NIRCAM') == pytest.approx(0.90)
    assert ap.default_ee_fraction('MIRI') == pytest.approx(0.80)
    assert ap.default_ee_fraction('nircam_full') == pytest.approx(0.90)


def test_mjy_sr_to_ujy_arcsec2_roundtrip_scalar():
    # 1 MJy/sr = 1e12 μJy / sr; 1 sr = (206265 arcsec)^2
    value = 1.0
    sb = ap.mjy_sr_to_ujy_arcsec2(value)
    expected = 1e12 / (206264.80624709636**2)
    assert sb == pytest.approx(expected, rel=1e-6)


def test_ujy_to_abmag_and_error():
    # ZP 23.9 → 1 μJy is AB = 23.9
    mag, mag_err = ap.ujy_to_abmag(1.0, 0.1)
    assert mag == pytest.approx(23.9)
    assert mag_err == pytest.approx((2.5 / np.log(10.0)) * 0.1)
    mag_bad, mag_err_bad = ap.ujy_to_abmag(-1.0, 0.1)
    assert np.isnan(mag_bad) and np.isnan(mag_err_bad)


def test_resolve_positions_sky_and_pixel(tmp_path: Path):
    path = _write_synthetic_coadd(tmp_path / 'coadd_i2d.fits')
    _, _, wcs, _ = ap.load_sci_extensions(path)
    x, y, ra, dec = ap.resolve_positions(x=32.0, y=32.0, wcs=wcs)
    assert x[0] == pytest.approx(32.0)
    assert y[0] == pytest.approx(32.0)
    x2, y2, ra2, dec2 = ap.resolve_positions(ra=ra[0], dec=dec[0], wcs=wcs)
    assert x2[0] == pytest.approx(32.0, abs=1e-6)
    assert y2[0] == pytest.approx(32.0, abs=1e-6)
    assert ra2[0] == pytest.approx(ra[0])
    assert dec2[0] == pytest.approx(dec[0])


def test_forced_aperture_photometry_with_mocked_apcorr(tmp_path: Path, monkeypatch):
    path = _write_synthetic_coadd(tmp_path / 'coadd_0_0_f200w_i2d.fits')
    params = ap.ApertureParams(
        radius_px=3.0,
        apcorr=1.1,
        sky_in_px=6.0,
        sky_out_px=9.0,
        ee_fraction=0.90,
        filter='F200W',
        instrument='NIRCAM',
    )

    def _boom(*_a, **_k):
        raise AssertionError('CRDS lookup should be mocked away')

    monkeypatch.setattr(ap, 'get_aperture_params', _boom)
    table = ap.forced_aperture_photometry(
        path, x=32.0, y=32.0, aperture_params=params
    )
    assert len(table) == 1
    assert table['ee_fraction'][0] == pytest.approx(0.90)
    assert table['apcorr'][0] == pytest.approx(1.1)
    assert np.isfinite(table['flux_ujy'][0])
    assert table['flux_ujy'][0] > 0
    assert np.isfinite(table['abmag'][0])
    assert np.isfinite(table['abmag_err'][0])
    assert table['abmag_err'][0] > 0


def test_read_coords_table_xy_and_sky(tmp_path: Path):
    xy_path = tmp_path / 'xy.txt'
    Table({'x': [10.0, 11.0], 'y': [20.0, 21.0]}).write(
        xy_path, format='ascii.ecsv', overwrite=True
    )
    kwargs, mode = ap.read_coords_table(xy_path)
    assert mode == 'xy'
    assert list(kwargs['x']) == [10.0, 11.0]

    sky_path = tmp_path / 'sky.txt'
    Table({'ra': [180.0], 'dec': [0.0], 'x': [1.0], 'y': [2.0]}).write(
        sky_path, format='ascii.ecsv', overwrite=True
    )
    kwargs, mode = ap.read_coords_table(sky_path)
    assert mode == 'sky'
    assert 'ra' in kwargs and 'x' not in kwargs


def test_coadd_phot_cli_writes_ecsv(tmp_path: Path, monkeypatch):
    path = _write_synthetic_coadd(tmp_path / 'coadd_i2d.fits')
    out = tmp_path / 'out.ecsv'
    params = ap.ApertureParams(
        radius_px=3.0,
        apcorr=1.0,
        sky_in_px=6.0,
        sky_out_px=9.0,
        ee_fraction=0.90,
        filter='F200W',
        instrument='NIRCAM',
    )
    monkeypatch.setattr(
        ap,
        'get_aperture_params',
        lambda *a, **k: params,
    )
    rc = coadd_phot_script.main(
        ['--image', str(path), '--xy', '32,32', '--outfile', str(out), '-v']
    )
    assert rc == 0
    assert out.is_file()
    table = Table.read(out)
    assert 'abmag' in table.colnames
    assert len(table) == 1


def test_pixel_scale_arcsec():
    wcs = WCS(_wcs_header(pixscale_arcsec=0.11))
    assert ap.pixel_scale_arcsec(wcs) == pytest.approx(0.11, rel=1e-6)
