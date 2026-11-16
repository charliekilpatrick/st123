"""JWST datamodel identity and on-disk NIRCam / MIRI coverage counts."""

from __future__ import annotations

from pathlib import Path

from astropy.table import Table

from st123.datamodels import JWSTDataModel, MIRIDataModel, NIRCamDataModel


def test_nircam_and_miri_matches_mast_names_and_paths():
    assert NIRCamDataModel.matches('NIRCAM/IMAGE')
    assert NIRCamDataModel.matches('NRC')
    assert NIRCamDataModel.matches('jw01234567001_02101_00001_nrca1_cal.fits')
    assert not NIRCamDataModel.matches('MIRI/IMAGE')
    assert not NIRCamDataModel.matches(None)

    assert MIRIDataModel.matches('MIRI/IMAGE')
    assert MIRIDataModel.matches('MIRI')
    assert MIRIDataModel.matches('jw02666006001_02101_00001_mirimage_cal.fits')
    assert MIRIDataModel.matches('/data/download/JWST/MIRI/F560W/x_cal.fits')
    assert not MIRIDataModel.matches('NIRCAM/IMAGE')
    assert not MIRIDataModel.matches(None)


def test_mast_instrument_name_column_via_matches():
    table = Table(
        {
            'instrument_name': ['MIRI/IMAGE', 'MIRI/IMAGE', 'NIRCAM/IMAGE'],
        }
    )
    has_nircam = any(NIRCamDataModel.matches(n) for n in table['instrument_name'])
    has_miri = any(MIRIDataModel.matches(n) for n in table['instrument_name'])
    assert has_nircam and has_miri

    miri_only = Table({'instrument_name': ['MIRI/IMAGE', 'MIRI']})
    assert not any(NIRCamDataModel.matches(n) for n in miri_only['instrument_name'])
    assert any(MIRIDataModel.matches(n) for n in miri_only['instrument_name'])
    empty = Table({'instrument_name': []})
    assert not any(NIRCamDataModel.matches(n) for n in empty['instrument_name'])
    assert not any(MIRIDataModel.matches(n) for n in empty['instrument_name'])


def test_coverage_under_counts_nircam_and_miri(tmp_path: Path):
    raw = tmp_path / 'reduction' / 'raw'
    raw.mkdir(parents=True)
    (raw / 'jw02666006001_02101_00001_mirimage_cal.fits').write_bytes(b'x')
    (raw / 'jw01234567001_02101_00001_nrca1_cal.fits').write_bytes(b'x')
    n_nrc, n_miri = JWSTDataModel.coverage_under(
        tmp_path / 'download' / 'JWST',
        raw,
        tmp_path / 'raw',
    )
    assert n_nrc == 1
    assert n_miri == 1

    miri_raw = tmp_path / 'only' / 'reduction' / 'raw'
    miri_raw.mkdir(parents=True)
    (miri_raw / 'jw02666006001_02101_00001_mirimage_cal.fits').write_bytes(b'x')
    n_nrc, n_miri = JWSTDataModel.coverage_under(miri_raw)
    assert n_nrc == 0
    assert n_miri == 1
