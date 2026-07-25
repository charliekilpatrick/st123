"""Shared fixtures for st123 script / library entry-point tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import write_illuminated_fits, write_ref_with_s_region


@pytest.fixture
def illuminated_fits(tmp_path: Path) -> Path:
    return write_illuminated_fits(tmp_path / 'science_cal.fits', include_s_region=False)


@pytest.fixture
def ref_i2d_fits(tmp_path: Path) -> Path:
    return write_ref_with_s_region(tmp_path / 'coadd_i2d.fits')
