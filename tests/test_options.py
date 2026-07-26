"""Tests for the unified CLI options module."""

from __future__ import annotations

from pathlib import Path

from st123.scripts import align as align_script
from st123.scripts import download as download_script
from st123.scripts import link_raw
from st123.scripts import mosaic as mosaic_script
from st123.scripts.utils.options import (
    create_parser,
    dataset_label,
    default_phot_dir,
    resolve_project_root,
    resolve_reduction_dir,
    st123_version_string,
)


def test_version_flag():
    parser = create_parser('demo')
    try:
        parser.parse_args(['--version'])
    except SystemExit as exc:
        assert exc.code == 0


def test_version_string_contains_st123():
    assert st123_version_string().startswith('st123 ')


def test_resolve_reduction_dir_project_root(tmp_path: Path):
    (tmp_path / 'JWST').mkdir()
    assert resolve_reduction_dir(tmp_path) == (tmp_path / 'reduction').resolve()


def test_resolve_reduction_dir_legacy_workdir(tmp_path: Path):
    assert resolve_reduction_dir(tmp_path) == tmp_path.resolve()


def test_resolve_reduction_dir_existing_markers(tmp_path: Path):
    (tmp_path / 'raw').mkdir()
    assert resolve_reduction_dir(tmp_path) == tmp_path.resolve()


def test_resolve_reduction_dir_prefers_reduction_over_reference_symlink(
    tmp_path: Path,
):
    """Project-root reference/ symlink must not steal reduction resolution."""
    reduction = tmp_path / 'reduction'
    (tmp_path / 'JWST').mkdir()
    (reduction / 'jhat').mkdir(parents=True)
    (reduction / 'reference').mkdir(parents=True)
    (tmp_path / 'reference').symlink_to(reduction / 'reference')
    assert resolve_reduction_dir(tmp_path) == reduction.resolve()


def test_resolve_project_root_from_reduction(tmp_path: Path):
    red = tmp_path / 'reduction'
    red.mkdir()
    (red / 'raw').mkdir()
    assert resolve_project_root(red) == tmp_path.resolve()


def test_dataset_label_from_base_dir(tmp_path: Path):
    root = tmp_path / 'NGC3310'
    (root / 'JWST').mkdir(parents=True)
    assert dataset_label(root) == 'NGC3310'
    assert dataset_label(root / 'reduction') == 'NGC3310'


def test_default_phot_dir_has_no_extra_obj_level(tmp_path: Path):
    root = tmp_path / 'NGC3310'
    (root / 'JWST').mkdir(parents=True)
    phot = default_phot_dir(root)
    assert phot == (root / 'reduction' / 'phot_0_0').resolve()
    # Only one NGC3310 in the full path (the project root)
    assert str(phot).count('NGC3310') == 1
    assert phot.parts[-2:] == ('reduction', 'phot_0_0')


def test_align_accepts_base_dir_without_obj():
    parser = align_script.create_parser()
    args = parser.parse_args(['--base-dir', '/tmp/w', '--ncores', '2'])
    assert args.base_dir == '/tmp/w'
    assert args.ncores == 2
    assert not hasattr(args, 'obj') or getattr(args, 'obj', None) is None


def test_align_accepts_legacy_workdir():
    parser = align_script.create_parser()
    args = parser.parse_args(['--workdir', '/tmp/w', '--workers', '4'])
    assert args.base_dir == '/tmp/w'
    assert args.ncores == 4


def test_mosaic_accepts_base_dir():
    parser = mosaic_script.create_parser()
    args = parser.parse_args(['--base-dir', '/b', '--ncores', '2'])
    assert args.base_dir == '/b'


def test_download_requires_base_dir():
    parser = download_script.create_parser()
    args = parser.parse_args(
        ['--ra', '10', '--dec', '20', '--base-dir', '/data/out']
    )
    assert args.base_dir == '/data/out'


def test_link_raw_base_dir_instrument():
    parser = link_raw.create_parser()
    args = parser.parse_args(['--base-dir', '/data/proj', '--instrument', 'NIRCAM'])
    assert args.base_dir == '/data/proj'
    assert args.instrument == 'NIRCAM'


def test_link_raw_legacy_datadir_symlinkdir():
    parser = link_raw.create_parser()
    args = parser.parse_args(['--datadir', '/d', '--symlinkdir', '/s'])
    assert args.source_dir == '/d'
    assert args.symlink_dir == '/s'
