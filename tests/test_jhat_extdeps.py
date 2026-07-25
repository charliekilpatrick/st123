"""Ensure the custom extdeps/jhat build is what the alignment stack imports."""

from __future__ import annotations

from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
expected_jhat_root = (repo_root / 'extdeps' / 'jhat' / 'jhat').resolve()


def test_custom_jhat_import_and_location():
    jhat = pytest.importorskip('jhat')
    assert getattr(jhat, '__version__', '').endswith('+st123')
    jhat_file = Path(jhat.__file__).resolve()
    assert expected_jhat_root in jhat_file.parents or jhat_file.parent == expected_jhat_root
    # Core API used by st123.alignment.align / relative_align.
    assert hasattr(jhat, 'st_wcs_align')
    assert hasattr(jhat, 'jwst_photclass')


def test_extdeps_jhat_tree_present():
    assert (repo_root / 'extdeps' / 'jhat' / 'setup.py').is_file()
    assert (repo_root / 'extdeps' / 'jhat' / 'jhat' / '__init__.py').is_file()
    assert (repo_root / 'extdeps' / 'README.md').is_file()
    readme = (repo_root / 'extdeps' / 'jhat' / 'README.md').read_text()
    assert 'custom' in readme.lower()
    assert 'not' in readme.lower() and 'pypi' in readme.lower()
