"""Tests for link-raw script and st123.link helpers."""

from __future__ import annotations

from pathlib import Path

from st123.utils.link import create_symlink, remove_proc_files
from st123.scripts import link_raw


def test_create_symlink_and_skip_existing(tmp_path: Path):
    src = tmp_path / 'src.fits'
    src.write_bytes(b'fits')
    dst_dir = tmp_path / 'raw'
    dst_dir.mkdir()
    dst = dst_dir / 'src.fits'
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
    (symlinkdir / 'raw').mkdir(parents=True)

    rc = link_raw.main(
        ['--datadir', str(datadir), '--symlinkdir', str(symlinkdir)]
    )
    assert rc == 0
    linked = symlinkdir / 'raw' / 'img.fits'
    assert linked.is_symlink()
    assert linked.resolve() == fits_path.resolve()


def test_link_raw_parser():
    parser = link_raw.create_parser()
    args = parser.parse_args(['--datadir', '/d', '--symlinkdir', '/s', '--proc_dirs', '/p'])
    assert args.datadir == '/d'
    assert args.proc_dirs == ['/p']
