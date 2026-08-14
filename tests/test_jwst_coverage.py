"""Tests for MIRI-only JWST skip policy helpers."""

from __future__ import annotations

from pathlib import Path

from astropy.table import Table

from st123.utils.jwst_coverage import (
    count_jwst_frames_on_disk,
    explicit_miri_only,
    force_miri_effective,
    mast_table_jwst_coverage,
    should_skip_miri_only_jwst,
    should_skip_miri_only_jwst_on_disk,
)


def test_should_skip_miri_only_policy():
    assert should_skip_miri_only_jwst(
        False, True, ['NIRCAM', 'MIRI'], force_miri=False
    )
    assert should_skip_miri_only_jwst(False, True, None, force_miri=False)
    assert not should_skip_miri_only_jwst(
        True, True, ['NIRCAM', 'MIRI'], force_miri=False
    )
    assert not should_skip_miri_only_jwst(
        False, True, ['NIRCAM', 'MIRI'], force_miri=True
    )
    assert not should_skip_miri_only_jwst(
        False, True, ['MIRI'], force_miri=False
    )
    assert not should_skip_miri_only_jwst(
        False, False, ['NIRCAM', 'MIRI'], force_miri=False
    )


def test_explicit_miri_only_and_force():
    assert explicit_miri_only(['MIRI'])
    assert not explicit_miri_only(['NIRCAM', 'MIRI'])
    assert not explicit_miri_only(None)
    assert force_miri_effective(False, ['MIRI'])
    assert force_miri_effective(True, ['NIRCAM', 'MIRI'])
    assert not force_miri_effective(False, ['NIRCAM', 'MIRI'])


def test_mast_table_jwst_coverage():
    table = Table(
        {
            'instrument_name': ['MIRI/IMAGE', 'MIRI/IMAGE', 'NIRCAM/IMAGE'],
        }
    )
    assert mast_table_jwst_coverage(table) == (True, True)
    miri_only = Table({'instrument_name': ['MIRI/IMAGE', 'MIRI']})
    assert mast_table_jwst_coverage(miri_only) == (False, True)
    assert mast_table_jwst_coverage(Table()) == (False, False)


def test_count_jwst_frames_on_disk(tmp_path: Path):
    raw = tmp_path / 'reduction' / 'raw'
    raw.mkdir(parents=True)
    (raw / 'jw02666006001_02101_00001_mirimage_cal.fits').write_bytes(b'x')
    (raw / 'jw01234567001_02101_00001_nrca1_cal.fits').write_bytes(b'x')
    n_nrc, n_miri = count_jwst_frames_on_disk(tmp_path)
    assert n_nrc == 1
    assert n_miri == 1

    miri_dir = tmp_path / 'only'
    miri_raw = miri_dir / 'reduction' / 'raw'
    miri_raw.mkdir(parents=True)
    (miri_raw / 'jw02666006001_02101_00001_mirimage_cal.fits').write_bytes(b'x')
    assert should_skip_miri_only_jwst_on_disk(miri_dir, ['NIRCAM', 'MIRI'])
    assert not should_skip_miri_only_jwst_on_disk(
        miri_dir, ['NIRCAM', 'MIRI'], force_miri=True
    )
