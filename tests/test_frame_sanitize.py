"""Tests for st123 frame sanitize + combiner weight normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
import warnings


def _jwst_l2_stub(path: Path, *, with_nans: bool = True) -> Path:
    sci = np.ones((32, 32), dtype=np.float32)
    err = np.full((32, 32), 0.1, dtype=np.float32)
    dq = np.zeros((32, 32), dtype=np.uint32)
    if with_nans:
        sci[0, 0] = np.nan
        err[1, 1] = np.inf
    pri = fits.PrimaryHDU(
        header=fits.Header(
            {
                'TELESCOP': 'JWST',
                'INSTRUME': 'NIRCAM',
                'FILTER': 'F200W',
            }
        )
    )
    sci_hdr = fits.Header(
        {
            'MJD-BEG': 60000.0,
            'MJD-AVG': 60000.1,
            'MJD-END': 60000.2,
            'OBSGEO-X': -2.0e8,
            'OBSGEO-Y': 1.6e9,
            'OBSGEO-Z': 2.7e8,
            'CRPIX1': 16.0,
            'CRPIX2': 16.0,
            'CRVAL1': 180.0,
            'CRVAL2': 0.0,
            'CD1_1': -8.6e-6,
            'CD1_2': 0.0,
            'CD2_1': 0.0,
            'CD2_2': 8.6e-6,
            'CTYPE1': 'RA---TAN',
            'CTYPE2': 'DEC--TAN',
            'CUNIT1': 'deg',
            'CUNIT2': 'deg',
        }
    )
    hdul = fits.HDUList(
        [
            pri,
            fits.ImageHDU(data=sci, header=sci_hdr, name='SCI'),
            fits.ImageHDU(data=err, name='ERR'),
            fits.ImageHDU(data=dq, name='DQ'),
        ]
    )
    hdul.writeto(path, overwrite=True)
    return path


def test_sanitize_jwst_l2_fills_nonfinite_and_materializes_headers(tmp_path: Path):
    from st123.datamodels import sanitize_jwst_l2

    path = _jwst_l2_stub(tmp_path / 'frame_jhat.fits', with_nans=True)
    report = sanitize_jwst_l2(path, materialize_headers=True)
    assert report['ok'] is True
    assert report['n_sci_filled'] >= 1
    assert report['n_err_fixed'] >= 1

    with fits.open(path) as hdul:
        sci = hdul['SCI'].data
        err = hdul['ERR'].data
        dq = hdul['DQ'].data
        assert np.isfinite(sci).all()
        assert np.isfinite(err).all()
        assert int(dq[0, 0]) & 1
        hdr = hdul['SCI'].header
        assert 'DATE-BEG' in hdr
        assert 'OBSGEO-L' in hdr
        assert 'OBSGEO-B' in hdr
        assert 'OBSGEO-H' in hdr

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with fits.open(path) as hdul:
            WCS(hdul['SCI'].header, relax=True)
    msgs = [str(w.message) for w in caught if issubclass(w.category, FITSFixedWarning)]
    assert not any('datfix' in m or 'obsfix' in m for m in msgs)


def test_sanitize_jwst_l2_idempotent(tmp_path: Path):
    from st123.datamodels import sanitize_jwst_l2

    path = _jwst_l2_stub(tmp_path / 'frame_jhat.fits', with_nans=True)
    first = sanitize_jwst_l2(path)
    second = sanitize_jwst_l2(path)
    assert first['n_sci_filled'] >= 1
    assert second['n_sci_filled'] == 0


def test_science_frame_from_path(tmp_path: Path):
    from st123.datamodels import NIRCamDataModel, open_datamodel

    path = _jwst_l2_stub(tmp_path / 'nrc_jhat.fits', with_nans=False)
    frame = open_datamodel(path)
    assert isinstance(frame, NIRCamDataModel)
    assert frame.is_jwst
    assert frame.instrument == 'NIRCAM'
    report = frame.sanitize()
    assert report['ok'] is True


def test_coadd_combiner_weights_no_invalid_divide(tmp_path: Path, monkeypatch):
    """Empty WHT regions must not emit divide warnings during coadd normalize."""
    from st123.stages.mosaic import mosaic as mosaic_mod

    # Minimal i2d-like products with zero-weight corners.
    def _i2d(name: str) -> Path:
        sci = np.ones((8, 8), dtype=np.float32)
        err = np.ones((8, 8), dtype=np.float32) * 0.1
        wht = np.ones((8, 8), dtype=np.float32)
        wht[:2, :2] = 0.0
        pri = fits.PrimaryHDU(
            header=fits.Header(
                {
                    'FILTER': 'F200W',
                    'EFFEXPTM': 100.0,
                    'TMEASURE': 100.0,
                    'DURATION': 100.0,
                    'FILENAME': name,
                }
            )
        )
        sci_h = fits.Header({'PHOTMJSR': 1.0})
        path = tmp_path / name
        fits.HDUList(
            [
                pri,
                fits.ImageHDU(sci, header=sci_h, name='SCI'),
                fits.ImageHDU(err, name='ERR'),
                fits.ImageHDU(wht, name='WHT'),
            ]
        ).writeto(path)
        return path

    f1 = _i2d('a_i2d.fits')
    f2 = _i2d('b_i2d.fits')
    out = tmp_path / 'coadd_i2d.fits'

    # Avoid heavy CCDData/WCS path failures: stub create_ccddata.
    from astropy.nddata import CCDData, StdDevUncertainty
    from astropy import units as u

    def _fake_ccd(path):
        with fits.open(path) as hdul:
            return CCDData(
                hdul['SCI'].data,
                uncertainty=StdDevUncertainty(hdul['ERR'].data),
                unit=u.MJy / u.sr,
            )

    monkeypatch.setattr(mosaic_mod, 'create_ccddata', _fake_ccd)
    monkeypatch.setattr(mosaic_mod, 'update_photmjsr', lambda *a, **k: 1.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        mosaic_mod.coadd([str(f1), str(f2)], 'f200w', filename=str(out))

    div_warns = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning)
        and 'invalid value encountered in divide' in str(w.message)
    ]
    assert not div_warns
    assert out.is_file()
