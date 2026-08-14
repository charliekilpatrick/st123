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
    filter_hst_observations,
    filter_jwst_observations_by_stage,
    filter_jwst_products,
    is_hst_science_product,
    mast_login,
    observation_matches_calib_stage,
    parse_s_region,
    prune_non_full_frame_miri,
    resolve_mast_token,
)
from st123.mast.mast import (
    download_hst_observations,
    download_jwst_observations,
    prepare_mast_auth,
    reset_mast_login_state,
)


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
    assert is_hst_science_product('j123_flt.fits', 'ACS/HRC')
    assert is_hst_science_product('j123_flt.fits', 'ACS/SBC')
    assert is_hst_science_product('i123_flc.fits', 'WFC3/UVIS')
    assert is_hst_science_product('i123_flt.fits', 'WFC3/IR')
    assert is_hst_science_product('u123_c0m.fits', 'WFPC2')
    assert is_hst_science_product('u123_c1m.fits', 'WFPC2')
    # ACS/WFC science is flc (CTE-corrected); do not take the older flt.
    assert not is_hst_science_product('j123_flt.fits', 'ACS/WFC')
    assert not is_hst_science_product('j123_flc.fits', 'ACS/HRC')
    assert not is_hst_science_product('readme.txt', 'ACS/WFC')
    assert not is_hst_science_product('j123_flc.fits', 'WFC3/IR')


def test_filter_hst_observations_pipeline_only_excludes_hap():
    obs = Table(
        {
            'obs_collection': ['HST', 'HST', 'HST', 'JWST'],
            'dataproduct_type': ['IMAGE', 'IMAGE', 'IMAGE', 'IMAGE'],
            'instrument_name': ['ACS/WFC', 'ACS/WFC', 'ACS/WFC', 'NIRCAM'],
            'filters': ['F814W', 'F814W', 'F814W', 'F200W'],
            'intentType': ['science', 'science', 'science', 'science'],
            'dataRights': ['PUBLIC', 'PUBLIC', 'PUBLIC', 'PUBLIC'],
            'project': ['HST', 'HAP', 'HAP', 'JWST'],
            'obsid': [1, 2, 3, 4],
            'obs_id': [
                'jey335010',
                'hst_17070_35_acs_wfc_f814w_jey335',
                'hst_skycell-p2575x04y07_acs_wfc_f814w_all',
                'jwst_x',
            ],
            't_min': [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = filter_hst_observations(obs, instruments=['ACS', 'WFC3', 'WFPC2'])
    assert list(out['obsid']) == [1]
    assert list(out['obs_id']) == ['jey335010']

    # Opt out of pipeline-only when callers want HAP rows.
    both = filter_hst_observations(
        obs, instruments=['ACS'], pipeline_only=False
    )
    assert list(both['obsid']) == [1, 2, 3]


def test_filter_hst_observations_keeps_acs_wfc_and_hrc():
    """Bare ACS downloads ACS/WFC (flc) and ACS/HRC (flt)."""
    from st123.mast.mast import has_supported_hst_science_products

    assert has_supported_hst_science_products('ACS/WFC')
    assert has_supported_hst_science_products('ACS/HRC')
    assert has_supported_hst_science_products('ACS/SBC')

    obs = Table(
        {
            'obs_collection': ['HST', 'HST', 'HST'],
            'dataproduct_type': ['IMAGE', 'IMAGE', 'IMAGE'],
            'instrument_name': ['ACS/WFC', 'ACS/HRC', 'WFC3/UVIS'],
            'filters': ['F814W', 'F814W', 'F814W'],
            'intentType': ['science', 'science', 'science'],
            'dataRights': ['PUBLIC', 'PUBLIC', 'PUBLIC'],
            'project': ['HST', 'HST', 'HST'],
            'obsid': [1, 2, 3],
            'obs_id': ['wfc', 'hrc', 'uvis'],
            't_min': [1.0, 2.0, 3.0],
        }
    )
    out = filter_hst_observations(obs, instruments=['ACS', 'WFC3'])
    assert list(out['obsid']) == [1, 2, 3]
    assert list(out['instrument_name']) == ['ACS/WFC', 'ACS/HRC', 'WFC3/UVIS']


def test_flatten_mast_download_dir(tmp_path: Path):
    from st123.mast.mast import flatten_mast_download_dir

    obsid = tmp_path / 'HST' / 'ACS' / 'F814W' / '102617486'
    nested = obsid / 'mastDownload' / 'HST' / 'jey335ehq'
    nested.mkdir(parents=True)
    src = nested / 'jey335ehq_flc.fits'
    src.write_bytes(b'fits')
    (nested / 'readme.txt').write_text('x')
    moved = flatten_mast_download_dir(obsid)
    dest = obsid / 'jey335ehq_flc.fits'
    assert dest.is_file()
    assert not src.exists()
    assert moved == [str(dest)]
    assert not (obsid / 'mastDownload').exists()


def test_download_hst_dedupes_product_filenames(tmp_path: Path, caplog):
    """Same FLC listed under two obsids is downloaded only once."""
    obs = Table(
        {
            'obsid': [101, 102],
            'obs_id': ['jey335010', 'jey335010b'],
            'filters': ['F814W', 'F814W'],
            'obs_collection': ['HST', 'HST'],
            'instrument_name': ['ACS/WFC', 'ACS/WFC'],
            'project': ['HST', 'HST'],
        }
    )
    products_a = Table(
        {
            'type': ['S', 'S', 'D'],
            'productFilename': [
                'jey335elq_flc.fits',
                'jey335ehq_flc.fits',
                'something_drz.fits',
            ],
            'productType': ['SCIENCE', 'SCIENCE', 'SCIENCE'],
        }
    )
    products_b = Table(
        {
            'type': ['S', 'S'],
            'productFilename': [
                'jey335elq_flc.fits',  # duplicate of obs 101
                'jey335ehq_flc.fits',
            ],
            'productType': ['SCIENCE', 'SCIENCE'],
        }
    )

    def _plist(obs_row):
        return products_a if int(obs_row['obsid']) == 101 else products_b

    with (
        caplog.at_level(logging.INFO, logger='st123.mast.mast'),
        patch('st123.mast.mast.Observations.get_product_list', side_effect=_plist),
        patch('st123.mast.mast.Observations.download_products') as mock_dl,
        patch('st123.mast.mast.resolve_mast_token', return_value=None),
    ):
        n = download_hst_observations(obs, outdir=str(tmp_path), dry_run=False)

    # First obsid downloaded; second counted ready (duplicate listings).
    assert n == 2
    assert mock_dl.call_count == 1
    downloaded = mock_dl.call_args[0][0]
    assert sorted(downloaded['productFilename']) == [
        'jey335ehq_flc.fits',
        'jey335elq_flc.fits',
    ]
    assert 'duplicate' in caplog.text.lower() or 'already' in caplog.text.lower()


def test_parse_s_region():
    region = 'POLYGON ICRS 10.0 20.0 10.1 20.0 10.1 20.1 10.0 20.1'
    poly = parse_s_region(region)
    assert poly.area > 0


def test_mast_login_required_without_token(monkeypatch):
    monkeypatch.delenv('MAST_API_TOKEN', raising=False)
    monkeypatch.delenv('MAST_TOKEN', raising=False)
    reset_mast_login_state()
    with pytest.raises(RuntimeError):
        mast_login(None, required=True)
    assert mast_login(None, required=False) is False


def test_mast_login_success_and_cached(monkeypatch):
    monkeypatch.delenv('MAST_API_TOKEN', raising=False)
    monkeypatch.delenv('MAST_TOKEN', raising=False)
    reset_mast_login_state()
    with patch('st123.mast.mast.Observations.login') as mock_login:
        assert mast_login('tok123', required=True) is True
        assert mast_login('tok123', required=True) is True
        mock_login.assert_called_once_with(token='tok123')


def test_prepare_mast_auth_public_without_token(monkeypatch, caplog):
    monkeypatch.delenv('MAST_API_TOKEN', raising=False)
    monkeypatch.delenv('MAST_TOKEN', raising=False)
    reset_mast_login_state()
    with caplog.at_level(logging.INFO, logger='st123.mast.mast'):
        assert prepare_mast_auth(None) is False
    assert 'public data only' in caplog.text.lower()


def test_download_hst_skips_product_list_when_local(tmp_path: Path, caplog):
    """Re-runs must not call MAST get_product_list when science FITS exist."""
    obs = Table(
        {
            'obsid': [185903893],
            'obs_id': ['jey312010'],
            'filters': ['F555W'],
            'obs_collection': ['HST'],
            'instrument_name': ['ACS/WFC'],
            'project': ['HST'],
        }
    )
    sub = tmp_path / 'HST' / 'ACS' / 'F555W' / '185903893'
    sub.mkdir(parents=True)
    (sub / 'jey312k1q_flc.fits').write_bytes(b'fits')
    (sub / 'jey312k5q_flc.fits').write_bytes(b'fits')

    with (
        caplog.at_level(logging.INFO, logger='st123.mast.mast'),
        patch('st123.mast.mast.Observations.get_product_list') as mock_plist,
        patch('st123.mast.mast.Observations.download_products') as mock_dl,
        patch('st123.mast.mast.resolve_mast_token', return_value=None),
    ):
        n = download_hst_observations(obs, outdir=str(tmp_path), dry_run=False)

    assert n == 1
    mock_plist.assert_not_called()
    mock_dl.assert_not_called()
    assert 'skipping mast product list' in caplog.text.lower()


def test_is_transient_mast_error_detects_timeouts():
    from st123.mast.mast import _is_transient_mast_error

    assert _is_transient_mast_error(TimeoutError('Timeout limit of 600 exceeded'))
    assert _is_transient_mast_error(ConnectionError('reset'))
    assert _is_transient_mast_error(RuntimeError('Timeout limit of 600 exceeded'))
    assert not _is_transient_mast_error(ValueError('bad product row'))


def test_download_hst_retries_transient_product_list(tmp_path: Path, monkeypatch):
    """Transient get_product_list failures retry then succeed (issue #4)."""
    from st123.mast.download import MastDownloadResult
    from st123.mast import mast as mast_mod

    monkeypatch.setattr(mast_mod, '_HST_MAST_RETRY_DELAY_SEC', 0.0)
    obs = Table(
        {
            'obsid': [101],
            'obs_id': ['jey335010'],
            'filters': ['F814W'],
            'obs_collection': ['HST'],
            'instrument_name': ['ACS/WFC'],
            'project': ['HST'],
        }
    )
    products = Table(
        {
            'type': ['S'],
            'productFilename': ['jey335elq_flc.fits'],
            'productType': ['SCIENCE'],
        }
    )
    calls = {'n': 0}

    def _plist(_obs_row):
        calls['n'] += 1
        if calls['n'] < 3:
            raise TimeoutError('Timeout limit of 600 exceeded')
        return products

    with (
        patch('st123.mast.mast.Observations.get_product_list', side_effect=_plist),
        patch('st123.mast.mast.Observations.download_products') as mock_dl,
        patch('st123.mast.mast.resolve_mast_token', return_value=None),
        patch('st123.mast.mast.time.sleep'),
    ):
        result = download_hst_observations(obs, outdir=str(tmp_path), dry_run=False)

    assert isinstance(result, MastDownloadResult)
    assert result.n_observations == 1
    assert result.n_failed == 0
    assert calls['n'] == 3
    mock_dl.assert_called_once()


def test_download_hst_marks_failed_after_product_list_retries(
    tmp_path: Path, monkeypatch, caplog
):
    """Exhausted product-list retries count as incomplete (issue #4)."""
    from st123.mast.download import MastDownloadResult
    from st123.mast import mast as mast_mod

    monkeypatch.setattr(mast_mod, '_HST_MAST_RETRY_DELAY_SEC', 0.0)
    obs = Table(
        {
            'obsid': [101, 102],
            'obs_id': ['a', 'b'],
            'filters': ['F555W', 'F814W'],
            'obs_collection': ['HST', 'HST'],
            'instrument_name': ['ACS/WFC', 'ACS/WFC'],
            'project': ['HST', 'HST'],
        }
    )
    products = Table(
        {
            'type': ['S'],
            'productFilename': ['ok_flc.fits'],
            'productType': ['SCIENCE'],
        }
    )

    def _plist(obs_row):
        if int(obs_row['obsid']) == 101:
            raise TimeoutError('Timeout limit of 600 exceeded')
        return products

    with (
        caplog.at_level(logging.ERROR, logger='st123.mast.mast'),
        patch('st123.mast.mast.Observations.get_product_list', side_effect=_plist),
        patch('st123.mast.mast.Observations.download_products') as mock_dl,
        patch('st123.mast.mast.resolve_mast_token', return_value=None),
        patch('st123.mast.mast.time.sleep'),
    ):
        result = download_hst_observations(obs, outdir=str(tmp_path), dry_run=False)

    assert isinstance(result, MastDownloadResult)
    assert result.n_observations == 1
    assert result.n_failed == 1
    assert result.incomplete
    assert mock_dl.call_count == 1
    assert 'incomplete' in caplog.text.lower()


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
