"""Tests for HST EXPFLAG / residual quality gates."""

from __future__ import annotations

from pathlib import Path

from astropy.io import fits

from st123.datamodels.hst import (
    filter_good_hst_frames,
    is_good_hst_alignment,
    is_good_hst_expflag,
    prune_bad_hst_expflag,
    read_expflag,
)


def _hst_flc(path: Path, *, expflag: str | None = 'NORMAL', **extra) -> Path:
    hdr = fits.Header({'TELESCOP': 'HST', 'INSTRUME': 'WFC3'})
    if expflag is not None:
        hdr['EXPFLAG'] = expflag
    for k, v in extra.items():
        hdr[k] = v
    fits.PrimaryHDU(header=hdr).writeto(path, overwrite=True)
    return path


def test_expflag_normal_and_missing_ok(tmp_path: Path):
    good = _hst_flc(tmp_path / 'a_flt.fits', expflag='NORMAL')
    missing = _hst_flc(tmp_path / 'b_flt.fits', expflag=None)
    # Explicitly omit EXPFLAG
    fits.PrimaryHDU(
        header=fits.Header({'TELESCOP': 'HST', 'INSTRUME': 'ACS'})
    ).writeto(missing, overwrite=True)
    assert is_good_hst_expflag(good)
    assert is_good_hst_expflag(missing)
    assert read_expflag(good) == 'NORMAL'
    from st123.datamodels import as_datamodel

    assert read_expflag(as_datamodel(good)) == 'NORMAL'


def test_expflag_indeterminate_rejected(tmp_path: Path):
    bad = _hst_flc(tmp_path / 'c_flt.fits', expflag='INDETERMINATE')
    assert not is_good_hst_expflag(bad)
    kept, rejected = filter_good_hst_frames([bad])
    assert kept == []
    assert rejected == [bad]


def test_jwst_paths_always_kept(tmp_path: Path):
    jw = tmp_path / 'jw01234_nrc_cal.fits'
    fits.PrimaryHDU(
        header=fits.Header({'TELESCOP': 'JWST', 'INSTRUME': 'NIRCAM'})
    ).writeto(jw)
    assert is_good_hst_expflag(jw)
    kept, rejected = filter_good_hst_frames([jw])
    assert kept == [jw]
    assert rejected == []


def test_alignment_ceilings(tmp_path: Path):
    ok = _hst_flc(
        tmp_path / 'd_jhat.fits',
        expflag='NORMAL',
        ST123INT=0.08,
        JWDISPM=0.12,
    )
    # Group-level ST123INT pollution must not reject when internal gate is off.
    polluted = _hst_flc(
        tmp_path / 'e_jhat.fits',
        expflag='NORMAL',
        ST123INT=1.08,
        JWDISPM=0.10,
    )
    bad_abs = _hst_flc(
        tmp_path / 'f_jhat.fits',
        expflag='NORMAL',
        ST123INT=0.08,
        JWDISPM=0.80,
    )
    assert is_good_hst_alignment(ok)  # defaults: no internal gate
    assert is_good_hst_alignment(polluted)
    assert not is_good_hst_alignment(bad_abs)
    assert not is_good_hst_alignment(polluted, max_internal_arcsec=0.50)
    kept, rejected = filter_good_hst_frames(
        [ok, polluted, bad_abs],
        max_abs_arcsec=0.50,
    )
    assert kept == [ok, polluted]
    assert rejected == [bad_abs]


def test_prune_bad_hst_expflag(tmp_path: Path):
    good = _hst_flc(tmp_path / 'g_flc.fits', expflag='NORMAL')
    bad = _hst_flc(tmp_path / 'b_flc.fits', expflag='TDF-DOWN')
    pruned = prune_bad_hst_expflag(tmp_path, remove=True)
    assert str(bad) in pruned
    assert good.is_file()
    assert not bad.exists()
