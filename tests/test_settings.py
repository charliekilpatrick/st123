"""Unit tests for datamodel filter catalogs and leftover layout settings."""

from __future__ import annotations

from st123.datamodels import (
    ACSDataModel,
    EuclidDataModel,
    EuclidNIRDataModel,
    EuclidVISDataModel,
    HSTDataModel,
    InstrumentDataModel,
    JWSTDataModel,
    MIRIDataModel,
    NIRCamDataModel,
    RomanDataModel,
    RomanWFIDataModel,
    WFC3IRDataModel,
    WFC3UVISDataModel,
    WFPC2DataModel,
)
from st123.utils import settings


def test_datamodel_filters_cover_supported_bandpasses():
    import st123.datamodels  # noqa: F401  # register subclasses

    names = InstrumentDataModel.all_filters()
    for name in (
        'F606W',
        'F275W',
        'F160W',
        'F200W',
        'F770W',
        'F062',
        'VIS',
        'YE',
        'NIR_J',
    ):
        assert name in names


def test_instrument_filter_lists_and_telescopes():
    assert HSTDataModel.telescope == 'HST'
    assert JWSTDataModel.telescope == 'JWST'
    assert EuclidDataModel.telescope == 'EUCLID'
    assert RomanDataModel.telescope == 'ROMAN'
    assert WFPC2DataModel.FILTERS[0] == 'F122M'
    assert 'F435W' in ACSDataModel.FILTERS
    assert 'F275W' in WFC3UVISDataModel.FILTERS
    assert 'F160W' in WFC3IRDataModel.FILTERS
    assert 'F200W' in NIRCamDataModel.FILTERS
    assert 'F770W' in MIRIDataModel.FILTERS
    assert 'VIS' in EuclidVISDataModel.FILTERS
    assert 'YE' in EuclidNIRDataModel.FILTERS
    assert 'F158' in RomanWFIDataModel.FILTERS


def test_hst_jwst_mission_instrument_lists():
    assert HSTDataModel.DEFAULT_MAST_FILTERS is None
    assert 'WFC3' in HSTDataModel.INSTRUMENTS
    assert JWSTDataModel.INSTRUMENTS == ('NIRCAM', 'MIRI')
    assert EuclidDataModel.INSTRUMENTS == ('VIS', 'NISP')
    assert RomanDataModel.INSTRUMENTS == ('WFI',)
    assert MIRIDataModel.MAX_REFERENCE_DISPERSION_MAS['F560W'] is None
    assert MIRIDataModel.MAX_REFERENCE_DISPERSION_MAS['F1000W'] == 35.0
    assert MIRIDataModel.DEFAULT_MAX_REFERENCE_DISPERSION_MAS == 70.0


def test_euclid_roman_instrument_geometry():
    assert EuclidVISDataModel.PIXEL_SCALE_ARCSEC == 0.101
    assert EuclidVISDataModel.DETECTOR_COUNT == 36
    assert len(EuclidVISDataModel.detector_ids()) == 36
    assert EuclidNIRDataModel.PIXEL_SCALE_ARCSEC == 0.30
    assert EuclidNIRDataModel.canonical_filter('NIR_J') == 'JE'
    assert EuclidNIRDataModel.canonical_filter('Y') == 'YE'
    assert RomanWFIDataModel.PIXEL_SCALE_ARCSEC == 0.11
    assert RomanWFIDataModel.DETECTOR_COUNT == 18
    assert RomanWFIDataModel.detector_ids()[-1] == 'WFI18'
    assert RomanWFIDataModel.filter_from_filename(
        'r0012301008002013005_0005_wfi06_f184_cal.asdf'
    ) == 'F184'
    assert not RomanWFIDataModel.matches('WFPC2')
    assert RomanWFIDataModel.matches('WFI')
    assert not EuclidVISDataModel.matches('NISP')


def test_dolphot_and_jhat_params_on_classes():
    assert JWSTDataModel.DOLPHOT_BASE_PARAMS['FitSky'] == '2'
    assert MIRIDataModel.DOLPHOT_IMAGE_PARAMS['raper'] == '3'
    assert NIRCamDataModel.CALCSKY_PARAMS['rin'] == 15
    assert MIRIDataModel.CALCSKY_PARAMS['rin'] == 10
    assert JWSTDataModel.JHAT_STRICT['refcat_racol'] == 'ra'


def test_layout_settings_remain():
    assert settings.DEFAULT_DOWNLOAD_LAYOUT == 'telescope/instrument/filter/obsid'
    assert settings.DEFAULT_PAIR_OUTDIR == 'alignment_output'
    assert settings.DOWNLOAD_DIR_NAME == 'download'


def test_best_reference_filters_are_lowercase_hst_bands():
    assert HSTDataModel.BEST_REFERENCE_FILTERS[0] == 'f625w'
    known = InstrumentDataModel.all_filters()
    for filt in HSTDataModel.BEST_REFERENCE_FILTERS:
        assert filt == filt.lower()
        assert filt.upper() in known
