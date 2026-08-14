"""Tests for telescope-specific JHAT directory resolution."""

from __future__ import annotations

from pathlib import Path

from st123.scripts.utils.options import resolve_jhat_dir


def test_resolve_jhat_dir_create_new_prefers_dedicated(tmp_path: Path):
    work = tmp_path / 'reduction'
    work.mkdir()
    hst = resolve_jhat_dir(work, 'hst', create=True)
    assert hst == work / 'jhat_hst'
    assert hst.is_dir()
    legacy = work / 'jhat'
    assert legacy.is_symlink()
    assert legacy.resolve() == hst.resolve()


def test_resolve_jhat_dir_keeps_legacy_mid_campaign(tmp_path: Path):
    work = tmp_path / 'reduction'
    legacy = work / 'jhat'
    legacy.mkdir(parents=True)
    (legacy / 'ieec01_jhat.fits').write_bytes(b'')
    (legacy / 'jw012_jhat.fits').write_bytes(b'')
    assert resolve_jhat_dir(work, 'hst', create=True) == legacy
    assert resolve_jhat_dir(work, 'jwst', create=True) == legacy


def test_resolve_jhat_dir_prefers_populated_dedicated(tmp_path: Path):
    work = tmp_path / 'reduction'
    dedicated = work / 'jhat_hst'
    dedicated.mkdir(parents=True)
    (dedicated / 'ieec01_jhat.fits').write_bytes(b'')
    legacy = work / 'jhat'
    legacy.mkdir()
    (legacy / 'old_jhat.fits').write_bytes(b'')
    assert resolve_jhat_dir(work, 'hst', create=False) == dedicated
