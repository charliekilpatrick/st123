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


def test_expflag_interrupted_rejected_after_sanitize(tmp_path: Path):
    """Header sanitize must not keep EXPFLAG=INTERRUPTED frames."""
    from st123.datamodels import sanitize_science_fits

    bad = _hst_flc(tmp_path / 'jey378lkq_flc.fits', expflag='INTERRUPTED')
    sanitize_science_fits(bad, materialize_headers=True)
    assert read_expflag(bad) == 'INTERRUPTED'
    assert not is_good_hst_expflag(bad)
    kept, rejected = filter_good_hst_frames([bad])
    assert kept == []
    assert rejected == [bad]


def test_prune_bad_hst_expflag(tmp_path: Path):
    good = _hst_flc(tmp_path / 'g_flc.fits', expflag='NORMAL')
    bad = _hst_flc(tmp_path / 'b_flc.fits', expflag='TDF-DOWN')
    bad2 = _hst_flc(tmp_path / 'b2_flc.fits', expflag='EXCESSIVE DOWNTIME')
    pruned = prune_bad_hst_expflag(tmp_path, remove=True)
    assert str(bad) in pruned
    assert str(bad2) in pruned
    assert len(pruned) == 2
    assert good.is_file()
    assert not bad.exists()
    assert not bad2.exists()


def test_hst_flc_under_visit_named_dir_not_euclid(tmp_path: Path):
    """Regression: path substring VISIT must not select Euclid VIS."""
    from st123.datamodels import as_datamodel
    from st123.datamodels.hst.wfc3_uvis import WFC3UVISDataModel

    visit_dir = tmp_path / 'test_visit_relative_recovery'
    visit_dir.mkdir()
    bad = _hst_flc(visit_dir / 'b_flc.fits', expflag='TDF-DOWN')
    model = as_datamodel(bad)
    assert isinstance(model, WFC3UVISDataModel)
    assert not is_good_hst_expflag(bad)


def test_visit_relative_recovery_skips_bad_expflag(tmp_path: Path):
    """align visit_relative must not write JHAT for TDF-DOWN frames."""
    from st123.stages.alignment.hst_jhat import recover_failed_hst_visit_relative

    out = tmp_path / 'jhat_hst'
    out.mkdir()
    bad = _hst_flc(tmp_path / 'b_flc.fits', expflag='TDF-DOWN')
    assert not is_good_hst_expflag(bad)
    results = [
        {
            'path': str(bad),
            'status': 'failed',
            'outpath': None,
            'error': 'Only 0 objects pass the initial cut',
        }
    ]
    recover_failed_hst_visit_relative(results, out, pipeline_fallback=True)
    assert results[0]['status'] == 'failed'
    assert results[0]['align_mode'] == 'REJECTED_EXPFLAG'
    assert results[0]['outpath'] is None
    assert not list(out.glob('*_jhat.fits'))


def _wfc3_ir_flt(
    path: Path,
    *,
    nextend: int = 5,
    exptime: float = 599.0,
    with_samp_time: bool = True,
) -> Path:
    """Minimal WFC3/IR flt primary (+ optional SAMP/TIME/SCI stubs)."""
    from astropy.io import fits
    import numpy as np

    hdr = fits.Header()
    hdr['INSTRUME'] = 'WFC3'
    hdr['DETECTOR'] = 'IR'
    hdr['TELESCOP'] = 'HST'
    hdr['FILTER'] = 'F160W'
    hdr['EXPTIME'] = float(exptime)
    hdr['NEXTEND'] = int(nextend)
    hdr['EXPFLAG'] = 'NORMAL'
    hdus = [fits.PrimaryHDU(header=hdr)]
    if with_samp_time:
        sci = fits.ImageHDU(
            np.zeros((16, 16), dtype=np.float32), name='SCI'
        )
        samp = fits.ImageHDU(
            np.ones((16, 16), dtype=np.float32), name='SAMP'
        )
        time = fits.ImageHDU(
            np.ones((16, 16), dtype=np.float32), name='TIME'
        )
        hdus.extend([sci, samp, time])
        # Pad with empty extensions so NEXTEND can match a real IR MEF.
        while len(hdus) - 1 < int(nextend):
            hdus.append(fits.ImageHDU(name=f'PAD{len(hdus)}'))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def test_wfc3_ir_nextend_and_exptime_gate(tmp_path: Path):
    from st123.datamodels.hst import (
        filter_unusable_wfc3_ir_frames,
        is_usable_wfc3_ir_flt,
        wfc3_ir_unusable_reason,
    )

    good = _wfc3_ir_flt(tmp_path / 'good_flt.fits', nextend=5, exptime=599.0)
    tiny = _wfc3_ir_flt(tmp_path / 'tiny_flt.fits', nextend=5, exptime=1.7)
    short_mef = _wfc3_ir_flt(
        tmp_path / 'short_flt.fits',
        nextend=1,
        exptime=599.0,
        with_samp_time=False,
    )
    uvis = _hst_flc(tmp_path / 'uvis_flc.fits', expflag='NORMAL')

    assert is_usable_wfc3_ir_flt(good)
    assert wfc3_ir_unusable_reason(tiny) is not None
    assert 'EXPTIME' in (wfc3_ir_unusable_reason(tiny) or '')
    assert wfc3_ir_unusable_reason(short_mef) is not None
    assert is_usable_wfc3_ir_flt(uvis)  # non-IR always passes

    kept, dropped = filter_unusable_wfc3_ir_frames(
        [good, tiny, short_mef, uvis]
    )
    assert good.resolve() in [p.resolve() for p in kept]
    assert uvis.resolve() in [p.resolve() for p in kept]
    assert len(dropped) == 2
