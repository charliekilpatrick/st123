"""Tests for MAST helpers used by the download entry point."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from astropy.table import Table

from st123.mast import (
    filter_jwst_observations_by_stage,
    filter_jwst_products,
    is_hst_science_product,
    mast_login,
    observation_matches_calib_stage,
    parse_s_region,
    resolve_mast_token,
)
from st123.mast.mast import download_jwst_observations


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


def test_download_skips_unavailable_calib_level_silently(capsys):
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
    with patch('st123.mast.mast.Observations.get_product_list', return_value=products), patch(
        'st123.mast.mast.Observations.download_products'
    ) as mock_dl, patch('st123.mast.mast.resolve_mast_token', return_value=None):
        n = download_jwst_observations(obs, outdir='/tmp/st123_dl_test', stage=2, dry_run=True)
    assert n == 1
    mock_dl.assert_not_called()
    out = capsys.readouterr().out
    assert '253894283' not in out
    assert 'no stage-2 science products' not in out
    assert 'Skipping 1 observation' in out
    assert '213233923' in out or 'F2100W' in out


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
