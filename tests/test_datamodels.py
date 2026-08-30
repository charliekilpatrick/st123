"""Datamodel class hierarchy and factory."""

from __future__ import annotations

from pathlib import Path

import pytest
from astropy.io import fits

import numpy as np

from st123.datamodels import (
    ACSHRCDataModel,
    ACSWFCDataModel,
    HSTDataModel,
    InstrumentDataModel,
    JWSTDataModel,
    MIRIDataModel,
    NIRCamDataModel,
    WFC3IRDataModel,
    WFC3UVISDataModel,
    WFPC2DataModel,
    classify_image_kind,
    open_datamodel,
)


def _write(path: Path, **hdr) -> Path:
    fits.PrimaryHDU(header=fits.Header(hdr)).writeto(path, overwrite=True)
    return path


def test_open_datamodel_nircam(tmp_path: Path):
    path = _write(
        tmp_path / 'jw_nrc_cal.fits',
        TELESCOP='JWST',
        INSTRUME='NIRCAM',
        DETECTOR='NRCA1',
    )
    model = open_datamodel(path)
    assert isinstance(model, NIRCamDataModel)
    assert model.is_jwst
    assert model.image_kind == 'short'


def test_open_datamodel_nircam_long(tmp_path: Path):
    path = _write(
        tmp_path / 'jw_nrclong_cal.fits',
        TELESCOP='JWST',
        INSTRUME='NIRCAM',
        DETECTOR='NRCBLONG',
    )
    assert open_datamodel(path).image_kind == 'long'


def test_open_datamodel_miri(tmp_path: Path):
    path = _write(tmp_path / 'jw_miri_cal.fits', TELESCOP='JWST', INSTRUME='MIRI')
    model = open_datamodel(path)
    assert isinstance(model, MIRIDataModel)
    assert model.image_kind == 'miri'


def test_open_datamodel_hst_detectors(tmp_path: Path):
    wfc = open_datamodel(
        _write(
            tmp_path / 'a_flc.fits',
            TELESCOP='HST',
            INSTRUME='ACS',
            DETECTOR='WFC',
        )
    )
    hrc = open_datamodel(
        _write(
            tmp_path / 'b_flt.fits',
            TELESCOP='HST',
            INSTRUME='ACS',
            DETECTOR='HRC',
        )
    )
    uvis = open_datamodel(
        _write(
            tmp_path / 'c_flc.fits',
            TELESCOP='HST',
            INSTRUME='WFC3',
            DETECTOR='UVIS',
        )
    )
    ir = open_datamodel(
        _write(
            tmp_path / 'd_flt.fits',
            TELESCOP='HST',
            INSTRUME='WFC3',
            DETECTOR='IR',
        )
    )
    wfpc2 = open_datamodel(
        _write(tmp_path / 'e_c0m.fits', TELESCOP='HST', INSTRUME='WFPC2')
    )
    assert isinstance(wfc, ACSWFCDataModel)
    assert isinstance(hrc, ACSHRCDataModel)
    assert isinstance(uvis, WFC3UVISDataModel)
    assert isinstance(ir, WFC3IRDataModel)
    assert isinstance(wfpc2, WFPC2DataModel)
    assert wfc.image_kind == 'acs'
    assert hrc.image_kind == 'acs'
    assert uvis.image_kind == 'wfc3'
    assert ir.image_kind == 'wfc3_ir'
    assert wfpc2.image_kind == 'wfpc2'


def test_classify_image_kind_filename_tokens():
    assert classify_image_kind('x_nrcb1_jhat.fits') == 'short'
    assert classify_image_kind('x_nrcblong_jhat.fits') == 'long'
    assert classify_image_kind('x_mirimage_jhat.fits') == 'miri'


def test_instrument_from_path_dispatches(tmp_path: Path):
    path = _write(
        tmp_path / 'jw_nrc_cal.fits',
        TELESCOP='JWST',
        INSTRUME='NIRCAM',
    )
    model = InstrumentDataModel.from_path(path)
    assert isinstance(model, NIRCamDataModel)
    assert isinstance(model, JWSTDataModel)
    assert isinstance(model, InstrumentDataModel)


def test_hst_generic_for_unknown_detector(tmp_path: Path):
    path = _write(
        tmp_path / 'sbc_flt.fits',
        TELESCOP='HST',
        INSTRUME='ACS',
        DETECTOR='SBC',
    )
    model = open_datamodel(path)
    assert isinstance(model, HSTDataModel)
    assert not isinstance(model, ACSWFCDataModel)
    assert not isinstance(model, ACSHRCDataModel)
    assert model.image_kind == 'acs'


def test_as_datamodel_identity(tmp_path: Path):
    from st123.datamodels import as_datamodel

    path = _write(
        tmp_path / 'jw_nrc_cal.fits',
        TELESCOP='JWST',
        INSTRUME='NIRCAM',
        FILTER='F150W',
        DETECTOR='NRCA1',
    )
    model = open_datamodel(path)
    assert as_datamodel(model) is model
    again = as_datamodel(path)
    assert isinstance(again, NIRCamDataModel)
    assert again.filter_name == 'f150w'
    assert again.instrument_name == 'nircam'
    assert again.keyword('FILTER') == 'F150W'
    assert again.sci_wcs().naxis == 2


def test_wfpc2_c0m_c1m_one_datamodel(tmp_path: Path):
    sci = tmp_path / 'u9ob0101m_c0m.fits'
    dq = tmp_path / 'u9ob0101m_c1m.fits'
    pri = fits.PrimaryHDU(
        header=fits.Header(
            {'TELESCOP': 'HST', 'INSTRUME': 'WFPC2', 'FILTNAM1': 'F814W'}
        )
    )
    sci_h = fits.Header(
        {
            'CRPIX1': 8.0,
            'CRPIX2': 8.0,
            'CRVAL1': 180.0,
            'CRVAL2': 0.0,
            'CD1_1': -1e-5,
            'CD1_2': 0.0,
            'CD2_1': 0.0,
            'CD2_2': 1e-5,
            'CTYPE1': 'RA---TAN',
            'CTYPE2': 'DEC--TAN',
        }
    )
    fits.HDUList(
        [pri, fits.ImageHDU(np.ones((16, 16), dtype=np.float32), header=sci_h, name='SCI')]
    ).writeto(sci)
    fits.HDUList(
        [
            fits.PrimaryHDU(header=fits.Header({'TELESCOP': 'HST', 'INSTRUME': 'WFPC2'})),
            fits.ImageHDU(np.zeros((16, 16), dtype=np.int16), name='SCI'),
        ]
    ).writeto(dq)

    from_sci = open_datamodel(sci)
    from_dq = open_datamodel(dq)
    assert isinstance(from_sci, WFPC2DataModel)
    assert isinstance(from_dq, WFPC2DataModel)
    assert from_sci.path.resolve() == sci.resolve()
    assert from_dq.path.resolve() == sci.resolve()
    assert from_sci.dq_path.resolve() == dq.resolve()
    assert from_dq.dq_path.resolve() == dq.resolve()
    assert from_sci.has_dq()
    assert from_sci.filter_name == 'f814w'
    with from_sci.open() as hdul:
        sci_hdu = hdul['SCI']
        dq_arr = from_sci.matching_dq_array(hdul, sci_hdu, 1)
    assert dq_arr.shape == (16, 16)
    with from_sci.open() as hdul:
        assert hdul[0].header.get('ST123SAN') is True
    assert dq_arr.shape == (16, 16)
    assert from_sci.exptime is None
    assert from_sci.distortion_keywords()['CTYPE1'] == 'RA---TAN'
    poly = from_sci.sky_polygon()
    assert len(poly.exterior.coords) >= 4


def test_datamodel_metadata_accessors(tmp_path: Path):
    from st123.datamodels import as_datamodel, as_datamodels, path_of
    from st123.utils import helpers

    path = tmp_path / 'jw_nrc_cal.fits'
    data = np.ones((8, 8), dtype=np.float32)
    primary = fits.PrimaryHDU(
        header=fits.Header(
            {
                'TELESCOP': 'JWST',
                'INSTRUME': 'NIRCAM',
                'FILTER': 'F150W',
                'DETECTOR': 'NRCA1',
                'MODULE': 'A',
                'EFFEXPTM': 42.0,
                'VISIT_ID': 'V009',
                'PUPIL': 'CLEAR',
                'DATE-OBS': '2024-01-02',
                'TIME-OBS': '03:04:05',
                'IDCTAB': 'n/a',
            }
        )
    )
    sci_h = fits.Header(
        {
            'CRPIX1': 4.0,
            'CRPIX2': 4.0,
            'CRVAL1': 10.0,
            'CRVAL2': 20.0,
            'CD1_1': -1e-5,
            'CD1_2': 0.0,
            'CD2_1': 0.0,
            'CD2_2': 1e-5,
            'CTYPE1': 'RA---TAN-SIP',
            'CTYPE2': 'DEC--TAN-SIP',
            'A_ORDER': 2,
            'B_ORDER': 2,
            'S_REGION': (
                'POLYGON ICRS 10.0 20.0 10.01 20.0 10.01 20.01 10.0 20.01'
            ),
            'PHOTPLAM': 15000.0,
        }
    )
    fits.HDUList(
        [primary, fits.ImageHDU(data, header=sci_h, name='SCI')]
    ).writeto(path)

    model = as_datamodel(path)
    assert model.exptime == 42.0
    assert model.wavelength_um == pytest.approx(1.5)
    assert model.visit_id == 'V009'
    assert model.obs_datetime == '2024-01-02T03:04:05'
    assert model.pupil == 'CLEAR'
    assert 'POLYGON ICRS' in model.s_region
    assert model.distortion_keywords()['A_ORDER'] == 2
    with model.open() as hdul:
        assert hdul[0].header.get('ST123SAN') is True
    assert helpers.get_filter(model) == 'f150w'
    assert helpers.get_instrument(model) == 'nircam'
    assert path_of(model) == path
    models = as_datamodels([model, path])
    assert models[0] is model
    assert isinstance(models[1], NIRCamDataModel)


def test_wavelength_um_from_photometric_headers(tmp_path: Path):
    missing = _write(tmp_path / 'no_wave.fits', TELESCOP='JWST', INSTRUME='MIRI')
    assert open_datamodel(missing).wavelength_um == float('inf')

    microns = _write(
        tmp_path / 'wavelen.fits',
        TELESCOP='JWST',
        INSTRUME='MIRI',
        WAVELEN=7.7,
    )
    assert open_datamodel(microns).wavelength_um == pytest.approx(7.7)

    angstroms = _write(
        tmp_path / 'hst_photplam.fits',
        TELESCOP='HST',
        INSTRUME='WFC3',
        DETECTOR='UVIS',
        PHOTPLAM=8140.0,
    )
    assert open_datamodel(angstroms).wavelength_um == pytest.approx(0.814)
