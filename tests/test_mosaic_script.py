"""Tests for mosaic / apply-gwcs script helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.table import Table

from st123.mosaic.mosaic import create_dirs, edit_spec_groups, mp_init, update_path
from st123.scripts import apply_gwcs as apply_gwcs_script
from st123.scripts import mosaic as mosaic_script


def test_mosaic_create_dirs(tmp_path: Path):
    out = create_dirs(str(tmp_path), n=2)
    assert set(out.keys()) == {0, 1}
    assert (tmp_path / 'reference' / 'group_0').is_dir()
    assert (tmp_path / 'reference' / 'group_1').is_dir()


def test_update_path():
    table = Table({'image': ['/old/a.fits', '/old/b.fits']})
    filter_table = {'F200W': table}
    updated = update_path(filter_table, '/new')
    assert list(updated['F200W']['image']) == ['/new/a.fits', '/new/b.fits']


def test_edit_spec_groups(tmp_path: Path):
    table = Table(
        {
            'image': ['/data/a.fits', '/data/b.fits', '/data/c.fits'],
            'group': [0, 0, 1],
        }
    )
    spec = tmp_path / 'spec.txt'
    # np.loadtxt with dtype=str on a single token can return a 0-d array;
    # write two names so iteration is stable.
    spec.write_text('a.fits\nb.fits\n')
    edited = edit_spec_groups(table, str(spec))
    assert edited['group'][0] == 2
    assert edited['group'][1] == 2
    assert edited['group'][2] == 1


def test_mp_init_sets_globals():
    mp_init(1, 2, ['a.fits'])
    import st123.mosaic.mosaic as mosaic_mod

    assert mosaic_mod.success == 1
    assert mosaic_mod.failed == 2
    assert mosaic_mod.success_files == ['a.fits']


def test_mosaic_parser():
    parser = mosaic_script.create_parser()
    args = parser.parse_args(
        ['--basedir', '/b', '--object', 'obj', '--nmax', '10', '--ncores', '2']
    )
    assert args.nmax == 10
    assert args.object == 'obj'


def test_mosaic_main_no_jhat_files(tmp_path: Path):
    empty = Table({'group': np.array([], dtype=int)})
    with (
        patch('st123.scripts.mosaic.glob.glob', return_value=[]),
        patch('st123.scripts.mosaic.input_list', return_value=empty),
        patch('st123.scripts.mosaic.Pool'),
    ):
        rc = mosaic_script.main(
            ['--basedir', str(tmp_path), '--object', 'dolphot', '--ncores', '1']
        )
    assert rc == 0


def test_apply_gwcs_parser_and_main_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parser = apply_gwcs_script.create_parser()
    args = parser.parse_args([])
    assert 'coadd' in args.pattern
    rc = apply_gwcs_script.main([])
    assert rc == 1


def test_apply_gwcs_main_with_paths(tmp_path: Path):
    coadd = tmp_path / 'coadd_0_i2d.fits'
    coadd.write_bytes(b'x')
    with patch(
        'st123.scripts.apply_gwcs.apply_wcs_to_coadd',
        return_value=str(tmp_path / 'coadd_corrected_0_i2d.fits'),
    ) as mock_apply:
        rc = apply_gwcs_script.main([str(coadd)])
    assert rc == 0
    mock_apply.assert_called_once_with(str(coadd))
