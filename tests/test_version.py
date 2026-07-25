"""Dynamic package version from setuptools-scm / install metadata."""

from __future__ import annotations

import re
from pathlib import Path

import st123


def test_st123_version_is_set():
    assert isinstance(st123.__version__, str)
    assert st123.__version__
    assert st123.__version__ != '0.0.0+unknown'


def test_st123_version_pep440ish():
    # Accept release, pre/dev, and local (+gHASH) segments from setuptools-scm.
    assert re.match(
        r'^\d+(\.\d+)*([a-zA-Z]+\d+)?(\.dev\d+)?(\+[\w.]+)?$',
        st123.__version__,
    )


def test_pyproject_declares_dynamic_version():
    text = Path(__file__).resolve().parents[1].joinpath('pyproject.toml').read_text()
    assert 'setuptools-scm' in text
    assert '"version"' in text or "'version'" in text
    assert 'version = "' not in text.split('[project]')[1].split('dynamic')[0]
    assert 'version_file' in text
