"""Tests for download script and the library helpers it calls."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from st123.stages.download import (
    normalize_filter_name,
    observation_download_subdir,
    query_mast_jwst,
    resolve_mast_token,
    resolve_outdir,
)
from st123.scripts import download as download_script
from st123.utils import is_number, parse_coord


def test_resolve_outdir_requires_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    custom = tmp_path / 'nested' / 'out'
    assert resolve_outdir(str(custom)) == str(custom)
    assert custom.is_dir()
    skipped = tmp_path / 'skip_me'
    assert resolve_outdir(str(skipped), create=False) == str(skipped)
    assert not skipped.exists()
    with pytest.raises(ValueError):
        resolve_outdir(None)


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
    coord = SkyCoord(150.0, 2.0, unit='deg')
    outdir = tmp_path / 'dl'
    with (
        patch('st123.stages.download.download.query_jwst') as mock_q,
        patch('st123.stages.download.download.download_jwst_observations') as mock_dl,
    ):
        from astropy.table import Table

        mock_q.return_value = Table()
        n = query_mast_jwst(coord, outdir=str(outdir), radius=1 * u.arcmin)
    assert n == 0
    assert n.skipped_miri_only is False
    mock_dl.assert_not_called()
    assert outdir.is_dir()


def test_query_mast_jwst_skips_miri_only_by_default(tmp_path):
    from astropy.table import Table

    from st123.stages.download.download import MastDownloadResult

    coord = SkyCoord(150.0, 2.0, unit='deg')
    outdir = tmp_path / 'dl'
    obs = Table(
        {
            'instrument_name': ['MIRI/IMAGE', 'MIRI/IMAGE'],
            'filters': ['F560W', 'F1000W'],
            'calib_level': [2, 2],
        }
    )
    with (
        patch('st123.stages.download.download.query_jwst', return_value=obs),
        patch(
            'st123.stages.download.download.filter_jwst_observations_by_stage',
            side_effect=lambda table, stage: table,
        ),
        patch('st123.stages.download.download.download_jwst_observations') as mock_dl,
    ):
        result = query_mast_jwst(
            coord,
            outdir=str(outdir),
            radius=1 * u.arcmin,
            instruments=['NIRCAM', 'MIRI'],
        )
    assert isinstance(result, MastDownloadResult)
    assert result.n_observations == 0
    assert result.skipped_miri_only is True
    mock_dl.assert_not_called()

    with (
        patch('st123.stages.download.download.query_jwst', return_value=obs),
        patch(
            'st123.stages.download.download.filter_jwst_observations_by_stage',
            side_effect=lambda table, stage: table,
        ),
        patch(
            'st123.stages.download.download.download_jwst_observations', return_value=2
        ) as mock_dl,
    ):
        forced = query_mast_jwst(
            coord,
            outdir=str(outdir),
            radius=1 * u.arcmin,
            instruments=['NIRCAM', 'MIRI'],
            force_miri=True,
        )
    assert forced.n_observations == 2
    assert forced.skipped_miri_only is False
    mock_dl.assert_called_once()


def test_download_parser_accepts_force_miri():
    parser = download_script.create_parser()
    args = parser.parse_args(
        [
            '--ra',
            '150.0',
            '--dec',
            '2.0',
            '--base-dir',
            '/tmp/out',
            '--force-miri',
        ]
    )
    assert args.force_miri is True


def test_download_parser_requires_core_args():
    parser = download_script.create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        # --base-dir is required
        parser.parse_args(['--ra', '150.0', '--dec', '2.0'])
    args = parser.parse_args(
        [
            '--ra',
            '150.0',
            '--dec',
            '2.0',
            '--base-dir',
            '/tmp/out',
            '--radius',
            '1.5',
        ]
    )
    assert args.base_dir == '/tmp/out'
    assert args.radius == 1.5


def test_miri_download_layout_and_filter_normalization():
    from st123.stages.download import normalize_instrument_dirname, normalize_telescope_dirname

    assert normalize_filter_name('F560W;CLEAR') == 'F560W'
    assert normalize_telescope_dirname('JWST') == 'JWST'
    assert normalize_instrument_dirname('MIRI/IMAGE') == 'MIRI'
    assert normalize_instrument_dirname('NIRCAM') == 'NIRCam'
    assert observation_download_subdir(
        'F560W;CLEAR',
        123,
        layout='telescope/instrument/filter/obsid',
        telescope='JWST',
        instrument='MIRI',
    ) == 'JWST/MIRI/F560W/123'
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
            '--download-dir',
            '/tmp/out',
            '--layout',
            'telescope/instrument/filter/obsid',
            '--instruments',
            'MIRI',
            '--dry-run',
        ]
    )
    assert args.base_dir == '/tmp/out'
    assert args.layout == 'telescope/instrument/filter/obsid'
    assert args.dry_run is True


def test_parse_instruments_comma_and_space():
    from st123.scripts.utils.options import parse_instruments, resolve_instruments

    assert parse_instruments(None) is None
    assert parse_instruments(['NIRCAM,MIRI']) == ['NIRCAM', 'MIRI']
    assert parse_instruments(['NIRCAM', 'MIRI']) == ['NIRCAM', 'MIRI']
    assert parse_instruments(['NIRCAM, MIRI', 'NIRISS']) == [
        'NIRCAM',
        'MIRI',
        'NIRISS',
    ]
    assert resolve_instruments(['hst']) == ['ACS', 'WFC3', 'WFPC2']
    assert resolve_instruments(['jwst']) == ['NIRCAM', 'MIRI']
    assert resolve_instruments(['jwst', 'hst']) == [
        'NIRCAM',
        'MIRI',
        'ACS',
        'WFC3',
        'WFPC2',
    ]
    assert resolve_instruments(['jwst', 'hst']) == resolve_instruments(
        ['NIRCAM', 'MIRI', 'ACS', 'WFC3', 'WFPC2']
    )
    assert resolve_instruments(['all']) == resolve_instruments(['jwst', 'hst'])


def test_parse_telescopes_multi_and_comma():
    assert download_script.parse_telescopes(None) == ['jwst']
    assert download_script.parse_telescopes(['jwst']) == ['jwst']
    assert download_script.parse_telescopes(['hst', 'jwst']) == ['hst', 'jwst']
    assert download_script.parse_telescopes(['hst,jwst']) == ['hst', 'jwst']
    assert download_script.parse_telescopes(['JWST', 'hst', 'jwst']) == [
        'jwst',
        'hst',
    ]
    with pytest.raises(ValueError, match='Unsupported telescope'):
        download_script.parse_telescopes(['roman'])


def test_infer_and_resolve_telescopes_from_instruments():
    assert download_script.infer_telescopes_from_instruments(
        ['NIRCAM', 'MIRI', 'ACS', 'WFC3']
    ) == ['jwst', 'hst']
    assert download_script.infer_telescopes_from_instruments(['ACS']) == ['hst']
    assert download_script.infer_telescopes_from_instruments(['MIRI']) == ['jwst']
    assert download_script.resolve_telescopes(
        None, ['NIRCAM', 'ACS', 'WFC3']
    ) == ['jwst', 'hst']
    assert download_script.resolve_telescopes(None, None) == ['jwst']
    assert download_script.resolve_telescopes(['hst'], ['NIRCAM', 'ACS']) == [
        'hst'
    ]
    with pytest.raises(ValueError, match='Cannot infer'):
        download_script.infer_telescopes_from_instruments(['NOTAREAL'])


def test_download_jwst_hst_aliases_partition_like_explicit():
    """``--instruments jwst hst`` == NIRCAM MIRI ACS WFC3 WFPC2."""
    from st123.scripts.utils.options import resolve_instruments

    aliases = resolve_instruments(['jwst', 'hst'])
    explicit = resolve_instruments(['NIRCAM', 'MIRI', 'ACS', 'WFC3', 'WFPC2'])
    assert aliases == explicit
    tels = download_script.resolve_telescopes(None, aliases)
    assert tels == ['jwst', 'hst']
    by_tel = download_script.partition_instruments_by_telescope(aliases, tels)
    assert by_tel == {
        'jwst': ['NIRCAM', 'MIRI'],
        'hst': ['ACS', 'WFC3', 'WFPC2'],
    }


def test_partition_instruments_by_telescope_mixed():
    by_tel = download_script.partition_instruments_by_telescope(
        ['NIRCAM', 'MIRI', 'ACS', 'WFC3'],
        ['hst', 'jwst'],
    )
    assert by_tel['jwst'] == ['NIRCAM', 'MIRI']
    assert by_tel['hst'] == ['ACS', 'WFC3']


def test_partition_instruments_defaults_per_telescope():
    by_tel = download_script.partition_instruments_by_telescope(
        None, ['hst', 'jwst']
    )
    assert by_tel['jwst'] == ['NIRCAM', 'MIRI']
    assert by_tel['hst'] == ['ACS', 'WFC3', 'WFPC2']


def test_download_parser_accepts_multi_telescope():
    parser = download_script.create_parser()
    args = parser.parse_args(
        [
            '--telescope',
            'hst',
            'jwst',
            '--ra',
            '150.0',
            '--dec',
            '2.0',
            '--base-dir',
            '/tmp/out',
            '--instruments',
            'NIRCAM',
            'MIRI',
            'ACS',
            'WFC3',
        ]
    )
    assert args.telescope == ['hst', 'jwst']
    assert args.instruments == ['NIRCAM', 'MIRI', 'ACS', 'WFC3']


def test_download_main_success(monkeypatch, tmp_path):
    from st123.stages.download.download import MastDownloadResult

    monkeypatch.chdir(tmp_path)
    project = tmp_path / 'o'
    with (
        patch(
            'st123.stages.download.download.query_mast_jwst',
            return_value=MastDownloadResult(3),
        ) as mock_q,
        patch('st123.scripts.link_raw.link_raw_tree', return_value=2) as mock_link,
    ):
        rc = download_script.main(
            [
                '--ra',
                '150.0',
                '--dec',
                '2.0',
                '--outdir',
                str(project),
            ]
        )
    assert rc == 0
    mock_q.assert_called_once()
    assert mock_q.call_args.kwargs['outdir'] == str(project / 'download')
    assert mock_q.call_args.kwargs['force_miri'] is False
    # Default JWST instruments are linked individually (never instrument=ALL).
    assert mock_link.call_count == 2
    linked = {(c.kwargs['telescope'], c.kwargs['instrument']) for c in mock_link.call_args_list}
    assert linked == {('JWST', 'NIRCAM'), ('JWST', 'MIRI')}
    assert all(c.kwargs['base_dir'] == str(project) for c in mock_link.call_args_list)


def test_download_main_miri_only_skip_is_success(monkeypatch, tmp_path):
    from st123.stages.download.download import MastDownloadResult

    monkeypatch.chdir(tmp_path)
    project = tmp_path / 'o'
    with (
        patch(
            'st123.stages.download.download.query_mast_jwst',
            return_value=MastDownloadResult(0, skipped_miri_only=True),
        ),
        patch('st123.scripts.link_raw.link_raw_tree') as mock_link,
    ):
        rc = download_script.main(
            [
                '--ra',
                '150.0',
                '--dec',
                '2.0',
                '--outdir',
                str(project),
            ]
        )
    assert rc == 0
    mock_link.assert_not_called()


def test_download_main_links_single_instrument(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with (
        patch('st123.stages.download.download.query_mast_hst', return_value=1),
        patch('st123.scripts.link_raw.link_raw_tree', return_value=4) as mock_link,
    ):
        rc = download_script.main(
            [
                '--telescope',
                'hst',
                '--ra',
                '150.0',
                '--dec',
                '2.0',
                '--base-dir',
                str(tmp_path / 'o'),
                '--instruments',
                'ACS',
                '--radius',
                '5',
            ]
        )
    assert rc == 0
    assert mock_link.call_args.kwargs['telescope'] == 'HST'
    assert mock_link.call_args.kwargs['instrument'] == 'ACS'


def test_download_main_skip_link_raw(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with (
        patch('st123.stages.download.download.query_mast_jwst', return_value=1),
        patch('st123.scripts.link_raw.link_raw_tree') as mock_link,
    ):
        rc = download_script.main(
            [
                '--ra',
                '150.0',
                '--dec',
                '2.0',
                '--outdir',
                str(tmp_path / 'o'),
                '--skip-link-raw',
            ]
        )
    assert rc == 0
    mock_link.assert_not_called()


def test_download_main_bad_coords():
    rc = download_script.main(
        ['--ra', 'bad', '--dec', 'coords', '--base-dir', '/tmp/x']
    )
    assert rc == 1


def test_download_main_zero_products(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with (
        patch('st123.stages.download.download.query_mast_jwst', return_value=0),
        patch('st123.scripts.link_raw.link_raw_tree') as mock_link,
    ):
        rc = download_script.main(
            ['--ra', '150.0', '--dec', '2.0', '--outdir', str(tmp_path)]
        )
    assert rc == 1
    mock_link.assert_not_called()


def test_download_main_hst_already_on_disk_is_success(monkeypatch, tmp_path):
    """Re-running download when products exist should exit 0 and still link-raw."""
    monkeypatch.chdir(tmp_path)
    project = tmp_path / 'o'
    with (
        patch('st123.stages.download.download.query_mast_hst', return_value=2) as mock_q,
        patch('st123.scripts.link_raw.link_raw_tree', return_value=4) as mock_link,
    ):
        rc = download_script.main(
            [
                '--telescope',
                'hst',
                '--ra',
                '150.0',
                '--dec',
                '2.0',
                '--base-dir',
                str(project),
                '--instruments',
                'ACS',
                '--radius',
                '5',
            ]
        )
    assert rc == 0
    mock_q.assert_called_once()
    mock_link.assert_called_once()


def test_download_main_hst_incomplete_exits_nonzero(monkeypatch, tmp_path):
    """Partial MAST inventory must exit 1 after linking what landed (issue #4)."""
    from st123.stages.download.download import MastDownloadResult

    monkeypatch.chdir(tmp_path)
    project = tmp_path / 'o'
    with (
        patch(
            'st123.stages.download.download.query_mast_hst',
            return_value=MastDownloadResult(1, n_failed=1),
        ),
        patch('st123.scripts.link_raw.link_raw_tree', return_value=2) as mock_link,
    ):
        rc = download_script.main(
            [
                '--telescope',
                'hst',
                '--ra',
                '150.0',
                '--dec',
                '2.0',
                '--base-dir',
                str(project),
                '--instruments',
                'ACS',
            ]
        )
    assert rc == 1
    mock_link.assert_called_once()


def test_download_main_multi_telescope_combined(monkeypatch, tmp_path):
    """Instruments alone imply HST+JWST; --telescope is not required."""
    monkeypatch.chdir(tmp_path)
    project = tmp_path / 'o'
    with (
        patch('st123.stages.download.download.query_mast_hst', return_value=2) as mock_hst,
        patch('st123.stages.download.download.query_mast_jwst', return_value=5) as mock_jwst,
        patch('st123.scripts.link_raw.link_raw_tree', return_value=3) as mock_link,
    ):
        rc = download_script.main(
            [
                '--ra',
                '150.0',
                '--dec',
                '2.0',
                '--base-dir',
                str(project),
                '--instruments',
                'NIRCAM',
                'MIRI',
                'ACS',
                'WFC3',
                '--radius',
                '3',
            ]
        )
    assert rc == 0
    mock_hst.assert_called_once()
    mock_jwst.assert_called_once()
    assert mock_hst.call_args.kwargs['instruments'] == ['ACS', 'WFC3']
    assert mock_jwst.call_args.kwargs['instruments'] == ['NIRCAM', 'MIRI']
    # One link-raw call per requested instrument (not ALL under each telescope).
    assert mock_link.call_count == 4
    linked = {
        (c.kwargs['telescope'], c.kwargs['instrument'])
        for c in mock_link.call_args_list
    }
    assert linked == {
        ('JWST', 'NIRCAM'),
        ('JWST', 'MIRI'),
        ('HST', 'ACS'),
        ('HST', 'WFC3'),
    }
