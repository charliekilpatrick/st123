"""Tests for MAST helpers used by the download entry point."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from astropy.table import Table

import numpy as np
from astropy.io import fits

from st123.mast import (
    filter_jwst_observations_by_stage,
    filter_jwst_products,
    is_hst_science_product,
    mast_login,
    observation_matches_calib_stage,
    parse_s_region,
    prune_non_full_frame_miri,
    resolve_mast_token,
)
from st123.mast.mast import download_jwst_observations


def test_prune_non_full_frame_miri(tmp_path: Path):
    good = tmp_path / 'jw_full_mirimage_cal.fits'
    bad = tmp_path / 'jw_cut_mirimage_cal.fits'
    for path, shape, sub in (
        (good, (1024, 1032), 'FULL'),
        (bad, (128, 136), 'SUB64'),
    ):
        primary = fits.PrimaryHDU()
        primary.header['INSTRUME'] = 'MIRI'
        primary.header['DETECTOR'] = 'MIRIMAGE'
        primary.header['SUBARRAY'] = sub
        fits.HDUList(
            [
                primary,
                fits.ImageHDU(data=np.ones(shape, dtype=np.float32), name='SCI'),
            ]
        ).writeto(path)

    rejected = prune_non_full_frame_miri(tmp_path, remove=True)
    assert any(Path(p).name == bad.name for p in rejected)
    assert good.is_file()
    assert not bad.is_file()


def test_filter_jwst_products_stage2_and_3():
    products = Table(
        {
            'productType': ['SCIENCE', 'SCIENCE', 'INFO'],
            'productSubGroupDescription': ['CAL', 'I2D', 'CAL'],
            'calib_level': [2, 3, 2],
            'productFilename': ['a_cal.fits', 'b_i2d.fits', 'c.txt'],
        }
    )
    stage2 = filter_jwst_products(products, stage=2)
    assert len(stage2) == 1
    assert stage2[0]['productFilename'] == 'a_cal.fits'
    stage3 = filter_jwst_products(products, stage=3)
    assert len(stage3) == 1
    assert stage3[0]['productFilename'] == 'b_i2d.fits'
    with pytest.raises(ValueError):
        filter_jwst_products(products, stage=9)


def test_observation_matches_calib_stage():
    assert observation_matches_calib_stage(3, 2) is True
    assert observation_matches_calib_stage(2, 2) is True
    assert observation_matches_calib_stage(1, 2) is False
    assert observation_matches_calib_stage(-1, 2) is False
    assert observation_matches_calib_stage(2, 3) is False
    # Unknown levels: keep and attempt download
    assert observation_matches_calib_stage(None, 2) is True
    assert observation_matches_calib_stage('x', 2) is True


def test_filter_jwst_observations_by_stage():
    obs = Table(
        {
            'obsid': [1, 2, 3, 4],
            'calib_level': [-1, 1, 2, 3],
            'filters': ['F770W', 'F770W', 'F770W', 'F770W'],
        }
    )
    stage2 = filter_jwst_observations_by_stage(obs, stage=2)
    assert list(stage2['obsid']) == [3, 4]
    stage3 = filter_jwst_observations_by_stage(obs, stage=3)
    assert list(stage3['obsid']) == [4]


def test_download_skips_unavailable_calib_level_silently(caplog):
    obs = Table(
        {
            'obsid': [253894283, 213233923],
            'filters': ['F2100W', 'F2100W'],
            'calib_level': [-1, 3],
            'obs_collection': ['JWST', 'JWST'],
            'instrument_name': ['MIRI/IMAGE', 'MIRI/IMAGE'],
        }
    )
    products = Table(
        {
            'productType': ['SCIENCE'],
            'productSubGroupDescription': ['CAL'],
            'calib_level': [2],
            'productFilename': ['jw_ok_cal.fits'],
        }
    )
    with (
        caplog.at_level(logging.INFO, logger='st123.mast'),
        patch('st123.mast.mast.Observations.get_product_list', return_value=products),
        patch('st123.mast.mast.Observations.download_products') as mock_dl,
        patch('st123.mast.mast.resolve_mast_token', return_value=None),
    ):
        n = download_jwst_observations(
            obs, outdir='/tmp/st123_dl_test', stage=2, dry_run=True
        )
    assert n == 1
    mock_dl.assert_not_called()
    text = caplog.text
    assert '253894283' not in text
    assert 'no stage-2 science products' not in text
    assert 'Skipping 1 observation' in text
    assert '213233923' in text or 'F2100W' in text


def test_is_hst_science_product():
    assert is_hst_science_product('j123_flc.fits', 'ACS/WFC')
    assert not is_hst_science_product('readme.txt', 'ACS/WFC')


def test_parse_s_region():
    region = 'POLYGON ICRS 10.0 20.0 10.1 20.0 10.1 20.1 10.0 20.1'
    poly = parse_s_region(region)
    assert poly.area > 0


def test_mast_login_required_without_token(monkeypatch):
    monkeypatch.delenv('MAST_API_TOKEN', raising=False)
    monkeypatch.delenv('MAST_TOKEN', raising=False)
    with pytest.raises(RuntimeError):
        mast_login(None, required=True)
    assert mast_login(None, required=False) is False


def test_mast_login_success(monkeypatch):
    monkeypatch.delenv('MAST_API_TOKEN', raising=False)
    with patch('st123.mast.mast.Observations.login') as mock_login:
        assert mast_login('tok123', required=True) is True
        mock_login.assert_called_once_with(token='tok123')


def test_resolve_mast_token_explicit():
    assert resolve_mast_token('x') == 'x'


def test_normalize_filter_and_path_helpers():
    from st123.mast.mast import (
        normalize_filter_name,
        normalize_instrument_dirname,
        normalize_telescope_dirname,
        observation_download_subdir,
    )

    assert normalize_filter_name('F560W;CLEAR') == 'F560W'
    assert normalize_filter_name('f200w') == 'F200W'
    assert normalize_filter_name('') == 'UNKNOWN'
    assert normalize_filter_name('???') == '_'

    assert normalize_telescope_dirname('JWST') == 'JWST'
    assert normalize_telescope_dirname('HST') == 'HST'
    assert normalize_telescope_dirname('Roman') == 'Roman'
    assert normalize_telescope_dirname('Euclid') == 'Euclid'

    assert normalize_instrument_dirname('NIRCAM/IMAGE') == 'NIRCam'
    assert normalize_instrument_dirname('MIRI') == 'MIRI'
    assert normalize_instrument_dirname('WFC3/UVIS') == 'WFC3'

    sub = observation_download_subdir('F200W', '12345', telescope='JWST')
    assert 'JWST' in Path(sub).parts or 'JWST' in str(sub)
    assert 'F200W' in str(sub)
    assert '12345' in str(sub)
