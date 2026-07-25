"""Tests for MAST helpers used by the download entry point."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from astropy.table import Table

from st123.mast import (
    filter_jwst_products,
    is_hst_science_product,
    mast_login,
    parse_s_region,
    resolve_mast_token,
)


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
