"""Tests for NIRCam absolute-hub visit seeding and quality gates."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from astropy.table import Table

from st123.stages.alignment.align import (
    HUB_ABS_RETIE_MAX_APPLY_ARCSEC,
    is_jwst_science_jhat,
    jhat_product_needs_realign,
    mosaic_abs_quality,
    pick_visit,
    rank_visits_as_abs_hubs,
    retie_jwst_jhat_to_abs_ref,
    score_visit_abs_hub,
    select_jwst_jhats_for_abs_redo,
)


def _table() -> Table:
    # Visit 0: large F150W footprint but shallower.
    # Visit 1: deep F200W (preferred hub).
    rows = []
    for i in range(4):
        rows.append(
            {
                'visit': 0,
                'filter': 'f150w',
                'image': f'v0_{i}.fits',
                'exptime': 100.0,
                'pupil': 'CLEAR',
            }
        )
    for i in range(16):
        rows.append(
            {
                'visit': 1,
                'filter': 'f200w',
                'image': f'v1_{i}.fits',
                'exptime': 500.0,
                'pupil': 'CLEAR',
            }
        )
    return Table(rows)


def test_score_prefers_deep_f200w_over_large_f150w():
    table = _table()
    s0 = score_visit_abs_hub(table, 0, 'f150w', footprint_area=10.0)
    s1 = score_visit_abs_hub(table, 1, 'f200w', footprint_area=1.0)
    assert s1 > s0


def test_rank_visits_as_abs_hubs_orders_f200w_first():
    table = _table()
    geoms = {
        0: type('G', (), {'area': 10.0})(),
        1: type('G', (), {'area': 1.0})(),
    }
    visit_filter = {0: 'f150w', 1: 'f200w'}
    order = rank_visits_as_abs_hubs(table, visit_filter, geoms)
    assert order[0] == 1
    assert order[1] == 0


def test_pick_visit_uses_hub_order_when_seeding():
    geoms = {
        0: type('G', (), {'area': 10.0})(),
        1: type('G', (), {'area': 1.0})(),
    }
    visit_filter = {0: 'f150w', 1: 'f200w'}
    vid, _ = pick_visit(None, dict(geoms), visit_filter, hub_order=[1, 0])
    assert vid == 1


def test_mosaic_abs_quality_accepts_good_gaia(tmp_path: Path):
    path = tmp_path / 'hub_jhat_i2d.fits'
    hdr = fits.Header(
        {
            'GADISPM': 0.015,  # 15 mas
            'GANCAL': 200,
        }
    )
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32), header=hdr).writeto(path)
    qa = mosaic_abs_quality(path, max_mas=40.0)
    assert qa['ok'] is True
    assert qa['n_calibrators'] == 200
    assert qa['dispersion_mas'] == 15.0


def test_mosaic_abs_quality_rejects_soft_fail(tmp_path: Path):
    path = tmp_path / 'soft_jhat_i2d.fits'
    hdr = fits.Header(
        {
            'GADISPM': 0.080,
            'GANCAL': 0,
        }
    )
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32), header=hdr).writeto(path)
    qa = mosaic_abs_quality(path, max_mas=40.0)
    assert qa['ok'] is False


def _jhat(tmp_path: Path, name: str, *, ncal: int, telescop: str = 'JWST') -> Path:
    path = tmp_path / name
    hdr = fits.Header(
        {
            'TELESCOP': telescop,
            'INSTRUME': 'NIRCAM' if telescop == 'JWST' else 'WFC3',
            'JWDISPM': 0.020,
            'JWNCAL': int(ncal),
        }
    )
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32), header=hdr).writeto(path)
    return path


def test_jhat_product_needs_realign_soft_fail(tmp_path: Path):
    soft = _jhat(tmp_path, 'jw_soft_jhat.fits', ncal=0)
    need, reason = jhat_product_needs_realign(soft)
    assert need is True
    assert reason == 'soft_fail'

    good = _jhat(tmp_path, 'jw_good_jhat.fits', ncal=100)
    need, reason = jhat_product_needs_realign(good)
    assert need is False
    assert reason == 'ok'


def test_jhat_product_needs_realign_stale_vs_catalog(tmp_path: Path):
    jhat = _jhat(tmp_path, 'jw_old_jhat.fits', ncal=50)
    cat = tmp_path / 'ref.phot.txt'
    cat.write_text('ra dec\n')
    # Make catalog newer than jhat.
    older = jhat.stat().st_mtime - 10.0
    import os

    os.utime(jhat, (older, older))
    need, reason = jhat_product_needs_realign(jhat, catalog_path=cat)
    assert need is True
    assert reason == 'stale_vs_catalog'


def test_is_jwst_science_jhat_filters_hst(tmp_path: Path):
    jw = _jhat(tmp_path, 'jw01234001001_jhat.fits', ncal=10)
    hst = _jhat(tmp_path, 'ie9801xbq_jhat.fits', ncal=10, telescop='HST')
    assert is_jwst_science_jhat(jw) is True
    assert is_jwst_science_jhat(hst) is False


def test_select_jwst_jhats_for_abs_redo_soft_fail(tmp_path: Path):
    hub = _jhat(tmp_path, 'hub_jhat_i2d.fits', ncal=40)
    soft = _jhat(tmp_path, 'jw_soft_jhat.fits', ncal=0)
    good = _jhat(tmp_path, 'jw_good_jhat.fits', ncal=80)
    hst = _jhat(tmp_path, 'ib2q01xaq_jhat.fits', ncal=0, telescop='HST')
    selected = select_jwst_jhats_for_abs_redo(
        [soft, good, hst],
        hub,
        check_stale=False,
        max_abs_mas=200.0,
        min_peak=5,
    )
    names = {p.name for p, _ in selected}
    assert 'jw_soft_jhat.fits' in names
    assert 'jw_good_jhat.fits' not in names
    assert 'ib2q01xaq_jhat.fits' not in names


def test_select_jwst_jhats_skips_abs_measure_by_default(tmp_path: Path):
    """Default measure_abs=False must not 2dhist good products."""
    hub = _jhat(tmp_path, 'hub_jhat_i2d.fits', ncal=40)
    good = _jhat(tmp_path, 'jw_good_jhat.fits', ncal=80)
    with patch(
        'st123.stages.alignment.align.measure_frame_abs_offset_mas'
    ) as meas:
        selected = select_jwst_jhats_for_abs_redo(
            [good],
            hub,
            check_stale=False,
            measure_abs=False,
        )
    assert selected == []
    meas.assert_not_called()


def test_retie_jwst_only_and_max_apply(tmp_path: Path):
    hub = tmp_path / 'hub.fits'
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32)).writeto(hub)
    jw = _jhat(tmp_path, 'jw_frame_jhat.fits', ncal=20)
    hst = _jhat(tmp_path, 'u2460109t_jhat.fits', ncal=20, telescop='HST')

    def _fake_harmonize(paths, abs_ref, **kwargs):
        assert all(Path(p).name.startswith('jw') for p in paths)
        assert kwargs.get('max_apply_arcsec') == HUB_ABS_RETIE_MAX_APPLY_ARCSEC
        assert kwargs.get('min_peak') == 8
        assert kwargs.get('ncores') == 4
        return {
            'ok': True,
            'n_shifted': 0,
            'n_ok': len(list(paths)),
            'n_fail_measure': 0,
            'n_skip_large': 0,
            'n_weak_peak': 0,
            'max_abs_arcsec': 0.0,
        }

    with patch(
        'st123.stages.mosaic.mosaic.harmonize_jwst_frames_to_ref',
        side_effect=_fake_harmonize,
    ):
        report = retie_jwst_jhat_to_abs_ref(
            [jw, hst],
            hub,
            jwst_only=True,
            ncores=4,
        )
    assert report['n_skipped_non_jwst'] == 1
    assert report['n_ok'] == 1