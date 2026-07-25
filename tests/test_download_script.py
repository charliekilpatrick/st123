"""Tests for download script and the library helpers it calls."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from st123.mast import (
    normalize_filter_name,
    observation_download_subdir,
    query_mast_jwst,
    resolve_mast_token,
    resolve_outdir,
)
from st123.scripts import download as download_script
from st123.utils import is_number, parse_coord


def test_resolve_outdir_default_and_explicit(tmp_path):
    assert resolve_outdir('NGC3310') == 'jwst_data/NGC3310'
    custom = str(tmp_path / 'out')
    assert resolve_outdir('NGC3310', outdir=custom) == custom


def test_parse_coord_decimal_and_sexagesimal():
    c1 = parse_coord(150.0, 2.0)
    assert isinstance(c1, SkyCoord)
    assert c1.ra.degree == pytest.approx(150.0)
    c2 = parse_coord('10:00:00', '+02:00:00')
    assert isinstance(c2, SkyCoord)
    assert parse_coord('not-a-coord', 'also-bad') is None
    assert is_number('12.5')
    assert not is_number('abc')


def test_resolve_mast_token_precedence(monkeypatch):
    monkeypatch.delenv('MAST_API_TOKEN', raising=False)
    monkeypatch.delenv('MAST_TOKEN', raising=False)
    assert resolve_mast_token(None) is None
    assert resolve_mast_token('  abc  ') == 'abc'
    monkeypatch.setenv('MAST_API_TOKEN', 'from_api')
    assert resolve_mast_token(None) == 'from_api'
    monkeypatch.setenv('MAST_TOKEN', 'from_token')
    assert resolve_mast_token(None) == 'from_api'
    monkeypatch.delenv('MAST_API_TOKEN')
    assert resolve_mast_token(None) == 'from_token'


def test_query_mast_jwst_empty_table(tmp_path):
    from astropy.table import Table

    coord = SkyCoord(150.0, 2.0, unit='deg')
    outdir = tmp_path / 'dl'
    with (
        patch('st123.mast.download.resolve_mast_token', return_value=None),
        patch('st123.mast.download.query_jwst', return_value=Table()),
        patch('st123.mast.download.download_jwst_observations') as mock_dl,
    ):
        n = query_mast_jwst(coord, str(outdir), radius=3 * u.arcmin)
    assert n == 0
    mock_dl.assert_not_called()
    assert outdir.is_dir()


def test_download_parser_requires_core_args():
    parser = download_script.create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        ['--ra', '150.0', '--dec', '2.0', '--obj', 'TEST', '--radius', '1.5']
    )
    assert args.obj == 'TEST'
    assert args.radius == 1.5


def test_miri_download_layout_and_filter_normalization():
    assert normalize_filter_name('F560W;CLEAR') == 'F560W'
    assert observation_download_subdir('F560W;CLEAR', 123, layout='filter_obsid') == (
        'F560W_123'
    )
    assert observation_download_subdir('F560W', 123, layout='filter/obsid') == (
        'F560W/123'
    )


def test_download_parser_accepts_download_dir_and_layout():
    parser = download_script.create_parser()
    args = parser.parse_args(
        [
            '--ra',
            '150.0',
            '--dec',
            '2.0',
            '--obj',
            'TEST',
            '--download-dir',
            '/tmp/out',
            '--layout',
            'filter/obsid',
            '--instruments',
            'MIRI',
            '--dry-run',
        ]
    )
    assert args.download_dir == '/tmp/out'
    assert args.layout == 'filter/obsid'
    assert args.dry_run is True


def test_download_main_success(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with patch('st123.scripts.download.query_mast_jwst', return_value=3) as mock_q:
        rc = download_script.main(
            ['--ra', '150.0', '--dec', '2.0', '--obj', 'OBJ', '--outdir', str(tmp_path / 'o')]
        )
    assert rc == 0
    mock_q.assert_called_once()


def test_download_main_bad_coords():
    rc = download_script.main(['--ra', 'bad', '--dec', 'coords', '--obj', 'OBJ'])
    assert rc == 1


def test_download_main_runtime_error(tmp_path):
    with patch(
        'st123.scripts.download.query_mast_jwst',
        side_effect=RuntimeError('login failed'),
    ):
        rc = download_script.main(
            ['--ra', '150.0', '--dec', '2.0', '--obj', 'OBJ', '--outdir', str(tmp_path)]
        )
    assert rc == 1
