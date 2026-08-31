"""Tests for link-raw script and st123.link helpers."""

from __future__ import annotations

from pathlib import Path

from st123.utils.link import create_symlink, remove_proc_files
from st123.scripts import link_raw


def test_create_symlink_and_skip_existing(tmp_path: Path):
    src = tmp_path / 'src.fits'
    src.write_bytes(b'fits')
    # Parent dirs (including raw/) are created automatically.
    dst = tmp_path / 'nested' / 'raw' / 'src.fits'
    create_symlink(str(src), str(dst))
    assert dst.is_symlink()
    assert dst.resolve() == src.resolve()
    # second call should leave existing link alone
    create_symlink(str(src), str(dst))
    assert dst.is_symlink()


def test_remove_proc_files(tmp_path: Path):
    datadir = tmp_path / 'data'
    datadir.mkdir()
    a = datadir / 'a.fits'
    b = datadir / 'b.fits'
    a.write_bytes(b'a')
    b.write_bytes(b'b')

    proc = tmp_path / 'proc'
    (proc / 'raw').mkdir(parents=True)
    # symlink a into proc/raw so it is considered already processed
    (proc / 'raw' / 'a.fits').symlink_to(a)

    remaining = remove_proc_files([str(a), str(b)], str(proc))
    assert str(b) in remaining
    assert str(a) not in remaining


def test_link_raw_main(tmp_path: Path):
    datadir = tmp_path / 'data'
    nested = datadir / 'visit1'
    nested.mkdir(parents=True)
    fits_path = nested / 'img.fits'
    fits_path.write_bytes(b'data')
    symlinkdir = tmp_path / 'reduction'
    # Do not pre-create raw/; link-raw must create it.
    rc = link_raw.main(
        ['--datadir', str(datadir), '--symlinkdir', str(symlinkdir)]
    )
    assert rc == 0
    linked = symlinkdir / 'raw' / 'img.fits'
    assert linked.is_symlink()
    assert linked.resolve() == fits_path.resolve()


def test_link_raw_main_multi_instrument(tmp_path: Path):
    """--instruments ACS WFC3 links each camera tree into reduction/raw."""
    project = tmp_path / 'proj'
    for inst, fname in (('ACS', 'a_flc.fits'), ('WFC3', 'w_flc.fits')):
        src = project / 'download' / 'HST' / inst / 'obs1'
        src.mkdir(parents=True)
        (src / fname).write_bytes(b'fits')

    rc = link_raw.main(
        [
            '--base-dir',
            str(project),
            '--instruments',
            'ACS',
            'WFC3',
            '-v',
        ]
    )
    assert rc == 0
    raw = project / 'reduction' / 'raw'
    assert (raw / 'a_flc.fits').is_symlink()
    assert (raw / 'w_flc.fits').is_symlink()


def test_link_raw_skips_bad_expflag(tmp_path: Path):
    """HST science products with EXPFLAG!=NORMAL are not symlinked."""
    from astropy.io import fits

    project = tmp_path / 'proj'
    src = project / 'download' / 'HST' / 'WFC3' / 'obs1'
    src.mkdir(parents=True)
    good = src / 'good_flt.fits'
    bad = src / 'bad_flt.fits'
    fits.PrimaryHDU(
        header=fits.Header(
            {'TELESCOP': 'HST', 'INSTRUME': 'WFC3', 'EXPFLAG': 'NORMAL'}
        )
    ).writeto(good)
    fits.PrimaryHDU(
        header=fits.Header(
            {
                'TELESCOP': 'HST',
                'INSTRUME': 'WFC3',
                'EXPFLAG': 'INDETERMINATE',
            }
        )
    ).writeto(bad)

    rc = link_raw.main(
        ['--base-dir', str(project), '--instruments', 'WFC3', '-v']
    )
    assert rc == 0
    raw = project / 'reduction' / 'raw'
    assert (raw / 'good_flt.fits').is_symlink()
    assert not (raw / 'bad_flt.fits').exists()


def test_link_raw_parser():
    parser = link_raw.create_parser()
    args = parser.parse_args(['--datadir', '/d', '--symlinkdir', '/s', '--proc_dirs', '/p'])
    assert args.source_dir == '/d'
    assert args.symlink_dir == '/s'
    assert args.proc_dirs == ['/p']
    multi = parser.parse_args(
        ['--base-dir', '/p', '--instruments', 'ACS', 'WFC3', 'WFPC2']
    )
    assert multi.instruments == ['ACS', 'WFC3', 'WFPC2']
