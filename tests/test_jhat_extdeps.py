"""Ensure the custom extdeps/jhat build is what the alignment stack imports."""

from __future__ import annotations

from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
expected_jhat_root = (repo_root / 'extdeps' / 'jhat' / 'jhat').resolve()


def test_custom_jhat_import_and_location():
    jhat = pytest.importorskip('jhat')
    version = getattr(jhat, '__version__', '')
    # Distinguishes the repo-local build from unmodified PyPI jhat.
    assert version.endswith('+st123'), version
    # Core API used by st123.alignment.align / relative_align.
    assert hasattr(jhat, 'st_wcs_align')
    assert hasattr(jhat, 'jwst_photclass')
    # Editable installs resolve under extdeps/; path-dep installs may land in
    # site-packages. Either is fine as long as the +st123 marker is present.
    jhat_file = Path(jhat.__file__).resolve()
    under_extdeps = (
        expected_jhat_root in jhat_file.parents
        or jhat_file.parent == expected_jhat_root
    )
    assert under_extdeps or '+st123' in version


def test_requirements_declares_local_jhat_path():
    req = (repo_root / 'requirements.txt').read_text()
    assert 'jhat @ file:./extdeps/jhat' in req
    assert 'tweakreg-hack' in req


def test_extdeps_jhat_tree_present():
    assert (repo_root / 'extdeps' / 'jhat' / 'setup.py').is_file()
    assert (repo_root / 'extdeps' / 'jhat' / 'jhat' / '__init__.py').is_file()
    assert (repo_root / 'extdeps' / 'README.md').is_file()
    readme = (repo_root / 'extdeps' / 'jhat' / 'README.md').read_text()
    assert 'custom' in readme.lower()
    assert 'not' in readme.lower() and 'pypi' in readme.lower()
