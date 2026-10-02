"""Unified HST/JWST frame_qa schema, header stamps, and gate thresholds."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits

from st123.stages.alignment.frame_qa import (
    FRAME_ABS_TOL_ARCSEC,
    FRAME_CATASTROPHIC_TOL_ARCSEC,
    FRAME_INTERNAL_TOL_ARCSEC,
    FRAME_L3_NO_MATCH_MAX,
    FRAME_MIN_MATCH_HEALTHY,
    FRAME_SPARSE_TOL_ARCSEC,
    abs_ok,
    build_frame_qa,
    coherent_tol_arcsec,
    filter_catastrophic_jhat_frames,
    frame_qa_is_catastrophic,
    internal_ok,
    isolated_unmatched_coadds,
    is_catastrophic_alignment,
    median_healthy_n_calibrators,
    read_frame_qa,
    stamp_quality_headers,
    warn_if_frame_qa_soft,
    write_alignment_summary_table,
    write_frame_qa,
)
from st123.stages.alignment.hst_jhat import (
    HST_INTERNAL_ALIGN_MAX_ARCSEC,
    HST_L3_ALIGN_MAX_ARCSEC,
    find_hst_abs_ref_image,
    stamp_hst_jhat_quality,
)


def test_shared_tolerances_match_jwst_class():
    assert FRAME_ABS_TOL_ARCSEC == pytest.approx(0.05)
    assert FRAME_INTERNAL_TOL_ARCSEC == pytest.approx(0.05)
    assert FRAME_SPARSE_TOL_ARCSEC == pytest.approx(0.08)
    assert HST_INTERNAL_ALIGN_MAX_ARCSEC == FRAME_INTERNAL_TOL_ARCSEC
    assert HST_L3_ALIGN_MAX_ARCSEC == FRAME_ABS_TOL_ARCSEC


def test_coherent_tol_healthy_vs_sparse():
    assert coherent_tol_arcsec(FRAME_MIN_MATCH_HEALTHY) == FRAME_INTERNAL_TOL_ARCSEC
    assert coherent_tol_arcsec(FRAME_MIN_MATCH_HEALTHY - 1) == FRAME_SPARSE_TOL_ARCSEC
    assert abs_ok(0.040, n_calibrators=20)
    assert not abs_ok(0.070, n_calibrators=20)
    assert abs_ok(0.070, n_calibrators=5)  # sparse soft gate
    assert not abs_ok(0.090, n_calibrators=5)
    assert internal_ok(0.040, n_match=20)
    assert not internal_ok(0.070, n_match=20)


def test_frame_qa_schema_roundtrip(tmp_path: Path):
    report = build_frame_qa(
        mission='hst',
        hub_id='coadd_wfc3_f814w_drc.fits',
        align_mode='JHAT',
        abs_ref='/tmp/hub.fits',
        residual_mas=40.0,
        n_calibrators=18,
        abs_method='catalog_residual',
        max_delta_mas=35.0,
        frames=[{'path': 'a_jhat.fits', 'status': 'ok'}],
    )
    assert report['mission'] == 'hst'
    assert report['abs']['residual_mas'] == pytest.approx(40.0)
    assert report['abs']['ok'] is True
    assert report['internal']['ok'] is True
    assert report['ok'] is True
    path = write_frame_qa(tmp_path, report)
    assert path.name == 'frame_qa.json'
    loaded = read_frame_qa(tmp_path)
    assert loaded is not None
    assert loaded['abs']['n_calibrators'] == 18
    assert loaded['tol']['abs_arcsec'] == FRAME_ABS_TOL_ARCSEC


def test_frame_qa_soft_fail_at_70_mas():
    report = build_frame_qa(
        mission='jwst',
        residual_mas=70.0,
        n_calibrators=20,
        max_delta_mas=40.0,
    )
    assert report['abs']['ok'] is False
    assert report['ok'] is False
    assert report['catastrophic']['ok'] is True
    assert report['mosaic_ok'] is False
    assert frame_qa_is_catastrophic(report) is False
    assert warn_if_frame_qa_soft(report) is False


def test_frame_qa_catastrophic_at_250_mas():
    report = build_frame_qa(
        mission='hst',
        residual_mas=316.0,
        n_calibrators=16,
        max_delta_mas=40.0,
        frames=[
            {'path': 'j9np06y0q_jhat.fits', 'n_calibrators': 16, 'dispersion_mas': 316.0},
            {'path': 'jey342bkq_jhat.fits', 'n_calibrators': 470, 'dispersion_mas': 12.0},
        ],
    )
    assert FRAME_CATASTROPHIC_TOL_ARCSEC == pytest.approx(0.2)
    assert is_catastrophic_alignment(0.316, 16, peer_n_calibrators=470)
    assert not is_catastrophic_alignment(0.040, 20, peer_n_calibrators=470)
    assert not is_catastrophic_alignment(0.070, 20, peer_n_calibrators=470)
    assert report['catastrophic']['ok'] is False
    assert report['mosaic_ok'] is False
    assert frame_qa_is_catastrophic(report) is True
    assert warn_if_frame_qa_soft(report) is False


def test_peer_sparse_false_lock_is_catastrophic():
    # Residual below 200 mas but n_cal is a tiny fraction of healthy peers.
    assert is_catastrophic_alignment(0.090, 16, peer_n_calibrators=470)
    assert not is_catastrophic_alignment(0.054, 16, peer_n_calibrators=470)
    assert is_catastrophic_alignment(
        0.040, 400, gaia_residual_arcsec=2.48, n_gaia=80
    )
    assert not is_catastrophic_alignment(
        0.040, 400, gaia_residual_arcsec=2.48, n_gaia=3
    )


def test_unanchored_l3_filters_flags_zero_detectable():
    from st123.stages.alignment.hst_reference import (
        Level3GaiaScore,
        unanchored_l3_filters,
    )

    scores = [
        Level3GaiaScore(Path('a'), 31, 30, 26, 'acs', 'f555w'),
        Level3GaiaScore(Path('b'), 30, 30, 0, 'acs', 'f606w'),
        Level3GaiaScore(Path('c'), 31, 29, 26, 'acs', 'f814w'),
    ]
    assert unanchored_l3_filters(scores) == {'f606w'}


def test_frame_exptime_skips_excessive_downtime(tmp_path: Path):
    from st123.stages.alignment.hst_jhat import _frame_exptime

    dead = tmp_path / 'xxq_jhat.fits'
    live = tmp_path / 'xyq_jhat.fits'
    hdu = fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32))
    hdu.header['EXPTIME'] = 0.0
    hdu.header['EXPFLAG'] = 'EXCESSIVE DOWNTIME'
    hdu.header['EXPSTART'] = 53952.0
    hdu.header['EXPEND'] = 53952.01
    hdu.writeto(dead)
    hdu2 = fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32))
    hdu2.header['EXPTIME'] = 746.0
    hdu2.header['EXPFLAG'] = 'NORMAL'
    hdu2.writeto(live)
    assert _frame_exptime(dead) == 0.0
    assert _frame_exptime(live) == pytest.approx(746.0)


def test_isolated_unmatched_coadds_drops_bad_filter():
    report = {
        'ok': False,
        'pairs': [
            {
                'a': 'coadd_0_0_acs_f606w_drc.fits',
                'b': 'coadd_0_0_acs_f814w_drc.fits',
                'ok': False,
                'n_match': 0,
            },
            {
                'a': 'coadd_0_0_acs_f606w_drc.fits',
                'b': 'coadd_0_0_wfc3_f555w_drc.fits',
                'ok': False,
                'n_match': 1,
            },
            {
                'a': 'coadd_0_0_acs_f814w_drc.fits',
                'b': 'coadd_0_0_wfc3_f555w_drc.fits',
                'ok': True,
                'n_match': 80,
            },
        ],
    }
    isolated = isolated_unmatched_coadds(report)
    assert isolated == ['coadd_0_0_acs_f606w_drc.fits']
    two = {
        'pairs': [
            {
                'a': 'coadd_f606w.fits',
                'b': 'coadd_f814w.fits',
                'ok': False,
                'n_match': 0,
            }
        ]
    }
    assert isolated_unmatched_coadds(two, abs_ref_name='coadd_f814w.fits') == [
        'coadd_f606w.fits'
    ]
    assert FRAME_L3_NO_MATCH_MAX == 1


def test_filter_catastrophic_jhat_frames(tmp_path: Path):
    good = tmp_path / 'good_jhat.fits'
    bad = tmp_path / 'bad_jhat.fits'
    fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32)).writeto(good)
    fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32)).writeto(bad)
    with fits.open(good, mode='update') as hdul:
        hdul[0].header['JWDISPD'] = 0.012
        hdul[0].header['JWNCAL'] = 460
        hdul.flush()
    with fits.open(bad, mode='update') as hdul:
        hdul[0].header['JWDISPD'] = 0.316
        hdul[0].header['JWNCAL'] = 16
        hdul.flush()
    kept, dropped = filter_catastrophic_jhat_frames([good, bad])
    assert kept == [good.resolve()]
    assert len(dropped) == 1
    assert dropped[0][0] == bad.resolve()
    assert median_healthy_n_calibrators([16, 460, 470]) == 460


def test_filter_keeps_st123cat_false_despite_crowded_phot(tmp_path: Path):
    """Galaxy-knot phot residual can be >200 mas after a verified image lock."""
    recovered = tmp_path / 'j9np06xyq_jhat.fits'
    fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32)).writeto(recovered)
    with fits.open(recovered, mode='update') as hdul:
        hdul[0].header['JWDISPD'] = 0.315
        hdul[0].header['JWNCAL'] = 24
        hdul[0].header['ST123CAT'] = False
        hdul.flush()
    kept, dropped = filter_catastrophic_jhat_frames([recovered])
    assert kept == [recovered.resolve()]
    assert dropped == []


def test_world_to_pixel_quiet_returns_nan_not_warnings():
    import warnings

    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    from st123.stages.alignment.hst_jhat import world_to_pixel_quiet

    h = fits.Header()
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    h['CRPIX1'] = 50.0
    h['CRPIX2'] = 50.0
    h['CRVAL1'] = 150.0
    h['CRVAL2'] = 2.0
    h['CD1_1'] = -1.0e-4
    h['CD1_2'] = 0.0
    h['CD2_1'] = 0.0
    h['CD2_2'] = 1.0e-4
    w = WCS(h)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        x, y = world_to_pixel_quiet(
            w,
            np.array([150.0, 999.0]),
            np.array([2.0, 2.0]),
        )
    assert np.isfinite(x[0]) and np.isfinite(y[0])
    assert not np.isfinite(x[1])
    assert not any('all_world2pix' in str(w.message) for w in caught)


def test_stamp_quality_headers(tmp_path: Path):
    path = tmp_path / 'demo_jhat.fits'
    fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32)).writeto(path)
    stamp_quality_headers(
        path,
        align_mode='JHAT',
        original_ref='/hub.fits',
        aligned_to='/hub.fits',
        abs_mean_arcsec=0.041,
        abs_median_arcsec=0.038,
        abs_std_arcsec=0.012,
        n_calibrators=22,
        internal_max_arcsec=0.033,
        catalog_basename='hub.phot.txt',
    )
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        assert hdr['ALGNMODE'] == 'JHAT'
        assert hdr['ALGNREF'] == '/hub.fits'
        assert hdr['JWDISPM'] == pytest.approx(0.041)
        assert hdr['JWDISPD'] == pytest.approx(0.038)
        assert hdr['JWDISPS'] == pytest.approx(0.012)
        assert hdr['JWNCAL'] == 22
        assert hdr['ST123INT'] == pytest.approx(0.033)
        assert 'JWCAT' in hdr


def test_write_alignment_summary_table(tmp_path: Path):
    out = write_alignment_summary_table(
        [
            {
                'path': 'a_jhat.fits',
                'filter': 'f814w',
                'status': 'ok',
                'n_calibrators': 20,
                'dispersion_mas': 41.2,
                'internal_max_delta_mas': 30.0,
                'align_mode': 'JHAT',
                'algnref': 'hub.fits',
                'aligned_to': 'hub.fits',
            }
        ],
        tmp_path / 'alignment_summary.txt',
    )
    text = out.read_text()
    assert 'dispersion_mas' in text
    assert '41.200' in text
    assert 'internal_max_delta_mas' in text


def _tiny_jhat(path: Path) -> Path:
    hdu = fits.PrimaryHDU(np.zeros((16, 16), dtype=np.float32))
    hdu.header['INSTRUME'] = 'WFC3'
    hdu.header['FILTER'] = 'F814W'
    hdu.writeto(path)
    return path


def test_stamp_hst_jhat_quality_writes_headers(tmp_path: Path):
    jhat = _tiny_jhat(tmp_path / 'iey902_jhat.fits')
    refcat = tmp_path / 'hub.phot.txt'
    refcat.write_text('ra dec mag\n150.0 2.0 18.0\n')
    residual = {
        'ok': True,
        'n_match': 25,
        'residual_arcsec': 0.042,
        'residual_mean_arcsec': 0.045,
        'residual_std_arcsec': 0.010,
        'abs_arcsec': 0.042,
    }
    with patch(
        'st123.stages.alignment.hst_jhat.measure_hst_narrowband_residual_vs_refcat',
        return_value=residual,
    ):
        report = stamp_hst_jhat_quality(
            jhat,
            refcat=refcat,
            abs_ref='/tmp/hub.fits',
            align_mode='JHAT',
            internal_max_arcsec=0.031,
        )
    assert report['dispersion_mas'] == pytest.approx(42.0)
    assert report['n_calibrators'] == 25
    assert report['ok'] is True
    with fits.open(jhat) as hdul:
        assert hdul[0].header['JWDISPM'] == pytest.approx(0.045)
        assert hdul[0].header['JWNCAL'] == 25
        assert hdul[0].header['ST123INT'] == pytest.approx(0.031)
        assert hdul[0].header['ALGNMODE'] == 'JHAT'


def test_find_hst_abs_ref_prefers_quality_stamped_f814(tmp_path: Path):
    jhat = tmp_path / 'jhat_hst'
    jhat.mkdir()
    ref = tmp_path / 'reference' / 'group_0' / 'ref_sn'
    ref.mkdir(parents=True)
    wfpc2 = ref / 'coadd_0_0_wfpc2_f814w_drz.fits'
    f814 = ref / 'coadd_0_0_wfc3_f814w_drc.fits'
    for p, size in ((wfpc2, 700_000), (f814, 650_000)):
        p.write_bytes(b'\0' * size)
    # Stamp the WFC3 hub with healthy JWDISPM / JWNCAL.
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32)).writeto(f814, overwrite=True)
    with fits.open(f814, mode='update') as hdul:
        hdul[0].header['JWDISPM'] = 0.03
        hdul[0].header['JWNCAL'] = 40
        # Keep file large enough for find_hst_abs_ref_image size gate.
        hdul[0].data = np.zeros((400, 400), dtype=np.float32)
        hdul.flush()
    # Enlarge WFPC2 stub past size gate without quality stamp.
    wfpc2.write_bytes(b'\0' * 700_000)

    chosen = find_hst_abs_ref_image(jhat)
    assert chosen is not None
    assert 'wfc3' in chosen.name.lower()
    assert 'f814' in chosen.name.lower()


def test_finalize_writes_frame_qa(tmp_path: Path):
    from st123.stages.alignment.hst_jhat import finalize_hst_jhat_dir_quality

    jhat_dir = tmp_path / 'jhat_hst'
    jhat_dir.mkdir()
    a = _tiny_jhat(jhat_dir / 'aaa_jhat.fits')
    b = _tiny_jhat(jhat_dir / 'bbb_jhat.fits')
    refcat = jhat_dir / 'hub.phot.txt'
    refcat.write_text('ra dec mag\n150.0 2.0 18.0\n')
    results = [
        {'path': str(a), 'status': 'ok', 'outpath': str(a), 'align_mode': 'JHAT'},
        {'path': str(b), 'status': 'ok', 'outpath': str(b), 'align_mode': 'JHAT'},
    ]
    residual = {
        'ok': True,
        'n_match': 20,
        'residual_arcsec': 0.04,
        'residual_mean_arcsec': 0.041,
        'residual_std_arcsec': 0.01,
        'abs_arcsec': 0.04,
        'dra_deg': 0.0,
        'ddec_deg': 0.0,
    }
    with (
        patch(
            'st123.stages.alignment.hst_jhat.validate_hst_group_internal_alignment',
            return_value={'ok': True, 'max_abs_arcsec': 0.03},
        ),
        patch(
            'st123.stages.alignment.hst_jhat.measure_hst_narrowband_residual_vs_refcat',
            return_value=residual,
        ),
        patch(
            'st123.stages.alignment.hst_jhat.polish_hst_jhat_abs_residual',
            return_value={'applied': False, 'ok': True},
        ),
        patch(
            'st123.stages.alignment.hst_jhat.measure_hst_sky_offset_vs_gaia',
            return_value={'ok': False},
        ),
        patch('st123.stages.alignment.hst_jhat.find_hst_l3_refcat', return_value=refcat),
        patch('st123.stages.alignment.hst_jhat.find_hst_abs_ref_image', return_value=None),
        patch('st123.utils.helpers.get_filter', return_value='F814W'),
        patch('st123.utils.helpers.get_instrument', return_value='wfc3_uvis'),
    ):
        qa = finalize_hst_jhat_dir_quality(jhat_dir, results)

    assert (jhat_dir / 'frame_qa.json').is_file()
    assert (jhat_dir / 'alignment_summary.txt').is_file()
    payload = json.loads((jhat_dir / 'frame_qa.json').read_text())
    assert payload['mission'] == 'hst'
    assert payload['internal']['max_delta_mas'] == pytest.approx(30.0)
    assert payload['abs']['residual_mas'] == pytest.approx(40.0)
    assert qa['ok'] is True
    assert qa['mosaic_ok'] is True


def test_finalize_rejects_catastrophic_jhat(tmp_path: Path):
    from st123.stages.alignment.hst_jhat import finalize_hst_jhat_dir_quality

    jhat_dir = tmp_path / 'jhat_hst'
    jhat_dir.mkdir()
    bad = _tiny_jhat(jhat_dir / 'j9np06y0q_jhat.fits')
    good = _tiny_jhat(jhat_dir / 'jey342bkq_jhat.fits')
    refcat = jhat_dir / 'hub.phot.txt'
    refcat.write_text('ra dec mag\n150.0 2.0 18.0\n')
    results = [
        {'path': str(bad), 'status': 'ok', 'outpath': str(bad), 'align_mode': 'JHAT'},
        {'path': str(good), 'status': 'ok', 'outpath': str(good), 'align_mode': 'JHAT'},
    ]

    def _residual(jhat, *args, **kwargs):
        name = Path(jhat).name
        if 'j9np06' in name:
            return {
                'ok': True,
                'n_match': 16,
                'residual_arcsec': 0.316,
                'residual_mean_arcsec': 0.316,
                'residual_std_arcsec': 0.05,
                'abs_arcsec': 0.316,
                'dra_deg': 0.0,
                'ddec_deg': 0.0,
            }
        return {
            'ok': True,
            'n_match': 460,
            'residual_arcsec': 0.012,
            'residual_mean_arcsec': 0.012,
            'residual_std_arcsec': 0.004,
            'abs_arcsec': 0.012,
            'dra_deg': 0.0,
            'ddec_deg': 0.0,
        }

    with (
        patch(
            'st123.stages.alignment.hst_jhat.validate_hst_group_internal_alignment',
            return_value={'ok': True, 'max_abs_arcsec': 0.03},
        ),
        patch(
            'st123.stages.alignment.hst_jhat.measure_hst_narrowband_residual_vs_refcat',
            side_effect=_residual,
        ),
        patch(
            'st123.stages.alignment.hst_jhat.polish_hst_jhat_abs_residual',
            return_value={'applied': False, 'ok': True},
        ),
        patch(
            'st123.stages.alignment.hst_jhat.measure_hst_sky_offset_vs_gaia',
            return_value={'ok': False},
        ),
        patch('st123.stages.alignment.hst_jhat.find_hst_l3_refcat', return_value=refcat),
        patch('st123.stages.alignment.hst_jhat.find_hst_abs_ref_image', return_value=None),
        patch(
            'st123.utils.helpers.get_filter',
            side_effect=lambda p: 'F606W' if 'j9np06' in Path(p).name else 'F555W',
        ),
        patch('st123.utils.helpers.get_instrument', return_value='acs'),
    ):
        qa = finalize_hst_jhat_dir_quality(jhat_dir, results)

    by_path = {Path(r['outpath']).name: r for r in results}
    assert by_path['j9np06y0q_jhat.fits']['status'] == 'failed'
    assert 'catastrophic' in str(by_path['j9np06y0q_jhat.fits']['error'])
    assert by_path['jey342bkq_jhat.fits']['status'] == 'ok'
    assert qa['catastrophic']['ok'] is False
    with fits.open(bad) as hdul:
        assert hdul[0].header['ST123CAT'] is True


def test_histogram_sky_offset_finds_outer_peak():
    from st123.stages.alignment.hst_jhat import histogram_sky_offset

    rng = np.random.default_rng(0)
    n_true = 80
    dra = np.concatenate(
        [np.full(n_true, 2.48) + rng.normal(0, 0.04, n_true), rng.normal(0, 1.5, 40)]
    )
    ddec = np.concatenate(
        [np.full(n_true, -1.40) + rng.normal(0, 0.04, n_true), rng.normal(0, 1.5, 40)]
    )
    st = histogram_sky_offset(
        dra, ddec, max_offset_arcsec=15.0, bin_arcsec=0.1, min_peak=8
    )
    assert st['ok'] is True
    assert st['abs_arcsec'] == pytest.approx(2.85, abs=0.3)
    assert st['peak_count'] >= 20
    assert st['peak_mode'] == 'outer'


def test_sky_offset_all_pairs_recovers_shift_nn_misses():
    """Chance close Gaia neighbours must not hide a coherent 2.5" peak."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    from st123.stages.alignment.hst_jhat import (
        _sky_offset_pairs_arcsec,
        histogram_sky_offset,
    )

    rng = np.random.default_rng(4)
    n = 80
    ra0 = 15.0
    dec0 = 2.0
    dra_true_as = 2.48
    ddec_true_as = -1.40
    ra_img = ra0 + rng.normal(0.0, 8.0 / 3600.0, n)
    dec_img = dec0 + rng.normal(0.0, 8.0 / 3600.0, n)
    cosdec = float(np.cos(np.radians(dec0)))
    ra_true = ra_img - (dra_true_as / 3600.0) / cosdec
    dec_true = dec_img - ddec_true_as / 3600.0
    # Random 0.4\" decoys so NN snaps inward instead of to the true match.
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    ra_decoy = ra_img - (0.40 * np.sin(phi) / 3600.0) / cosdec
    dec_decoy = dec_img - (0.40 * np.cos(phi) / 3600.0)
    c_img = SkyCoord(ra_img * u.deg, dec_img * u.deg)
    c_ref = SkyCoord(
        np.concatenate([ra_true, ra_decoy]) * u.deg,
        np.concatenate([dec_true, dec_decoy]) * u.deg,
    )
    dra_nn, ddec_nn, _ = _sky_offset_pairs_arcsec(
        c_img, c_ref, 15.0, all_pairs=False
    )
    nn = histogram_sky_offset(
        dra_nn, ddec_nn, max_offset_arcsec=15.0, bin_arcsec=0.1, min_peak=8
    )
    dra_all, ddec_all, _ = _sky_offset_pairs_arcsec(
        c_img, c_ref, 15.0, all_pairs=True
    )
    allp = histogram_sky_offset(
        dra_all, ddec_all, max_offset_arcsec=15.0, bin_arcsec=0.1, min_peak=8
    )
    assert allp['ok'] is True
    assert allp['abs_arcsec'] == pytest.approx(2.85, abs=0.3)
    assert allp['peak_mode'] == 'outer'
    # NN snaps to the 0.4" decoys and never forms the 2.5" peak.
    assert (not nn['ok']) or nn['abs_arcsec'] < 1.0


def test_fft_xcorr_recovers_pixel_shift():
    from st123.stages.alignment.hst_jhat import _fft_xcorr_shift_pixels

    yy, xx = np.mgrid[0:64, 0:64]
    ref = np.exp(-((xx - 32.0) ** 2 + (yy - 32.0) ** 2) / (2 * 4.0**2))
    img = np.exp(-((xx - 28.0) ** 2 + (yy - 35.0) ** 2) / (2 * 4.0**2))
    dy, dx, peak = _fft_xcorr_shift_pixels(ref, img)
    assert peak > 0
    assert dy == -3
    assert dx == 4


def test_quality_is_catastrophic_honors_gaia_residual():
    from st123.stages.alignment.hst_jhat import _quality_is_catastrophic

    q = {
        'dispersion_mas': 40.0,
        'n_calibrators': 400,
        'gaia_residual_arcsec': 2.48,
        'n_gaia': 80,
        'catastrophic': True,
    }
    assert _quality_is_catastrophic(q, peer_n_calibrators=400)
    q['catastrophic'] = False
    assert not _quality_is_catastrophic(q, peer_n_calibrators=400)
    del q['catastrophic']
    assert _quality_is_catastrophic(q, peer_n_calibrators=400)
    q['gaia_residual_arcsec'] = 0.03
    assert not _quality_is_catastrophic(q, peer_n_calibrators=400)


def test_quality_is_catastrophic_image_peak_overrides_gaia():
    """Crowded-galaxy Gaia NN must not reject a verified nucleus lock."""
    from st123.stages.alignment.hst_jhat import _quality_is_catastrophic

    q = {
        'dispersion_mas': 316.5,
        'n_calibrators': 14,
        'gaia_residual_arcsec': 8.32,
        'n_gaia': 200,
        'image_peak_arcsec': 0.182,
        'image_residual_arcsec': 0.158,
    }
    assert not _quality_is_catastrophic(q, peer_n_calibrators=400)


def test_recover_catastrophic_hst_jhat_applies_gaia_shift(tmp_path: Path):
    from st123.stages.alignment.hst_jhat import recover_catastrophic_hst_jhat

    jhat = _tiny_jhat(tmp_path / 'j9np06xyq_jhat.fits')
    with fits.open(jhat, mode='update') as hdul:
        hdul[0].header['EXPTIME'] = 746.0
        hdul.flush()
    gaia = {
        'ok': True,
        'method': 'gaia_2dhist',
        'abs_arcsec': 2.48,
        'dra_arcsec': 2.04,
        'ddec_arcsec': -1.40,
        'dra_deg': 2.04 / 3600.0,
        'ddec_deg': -1.40 / 3600.0,
        'n_match': 122,
        'peak_count': 80,
        'peak_mode': 'outer',
    }
    with (
        patch(
            'st123.stages.alignment.hst_jhat._frame_exptime', return_value=746.0
        ),
        patch(
            'st123.stages.alignment.hst_jhat.restore_pipeline_relative_wcs_with_common_shift',
            return_value=[{'applied': True}],
        ) as restore,
        patch(
            'st123.stages.alignment.hst_jhat.measure_hst_sky_offset_2dhist',
            return_value={'ok': False},
        ),
        patch(
            'st123.stages.alignment.hst_jhat.measure_hst_sky_offset_vs_gaia',
            return_value=gaia,
        ),
        patch(
            'st123.stages.alignment.hst_jhat.refine_hst_narrowband_to_refcat',
            return_value={'ok': True},
        ),
    ):
        report = recover_catastrophic_hst_jhat(
            [jhat], abs_ref=tmp_path / 'missing.fits', refcat=None
        )
    assert report['applied'] is True
    assert report['ok'] is True
    assert report['method'] == 'gaia_2dhist'
    assert report['abs_arcsec'] == pytest.approx(2.48)
    assert restore.call_count == 2
    apply_shift = restore.call_args_list[1]
    assert apply_shift.kwargs['dra_deg'] == pytest.approx(-gaia['dra_deg'])
    assert apply_shift.kwargs['ddec_deg'] == pytest.approx(-gaia['ddec_deg'])


def test_finalize_recovers_catastrophic_before_reject(tmp_path: Path):
    from st123.stages.alignment.hst_jhat import finalize_hst_jhat_dir_quality

    jhat_dir = tmp_path / 'jhat_hst'
    jhat_dir.mkdir()
    bad = _tiny_jhat(jhat_dir / 'j9np06y0q_jhat.fits')
    good = _tiny_jhat(jhat_dir / 'jey342bkq_jhat.fits')
    refcat = jhat_dir / 'hub.phot.txt'
    refcat.write_text('ra dec mag\n150.0 2.0 18.0\n')
    results = [
        {'path': str(bad), 'status': 'ok', 'outpath': str(bad), 'align_mode': 'JHAT'},
        {'path': str(good), 'status': 'ok', 'outpath': str(good), 'align_mode': 'JHAT'},
    ]
    calls = {'n': 0}

    def _residual(jhat, *args, **kwargs):
        calls['n'] += 1
        name = Path(jhat).name
        # After recovery restamp, F606W is healthy.
        recovered = calls['n'] > 2
        if 'j9np06' in name and not recovered:
            return {
                'ok': True,
                'n_match': 16,
                'residual_arcsec': 0.316,
                'residual_mean_arcsec': 0.316,
                'residual_std_arcsec': 0.05,
                'abs_arcsec': 0.316,
                'dra_deg': 0.0,
                'ddec_deg': 0.0,
            }
        n_cal = 16 if 'j9np06' in name else 460
        res = 0.020 if 'j9np06' in name else 0.012
        return {
            'ok': True,
            'n_match': n_cal,
            'residual_arcsec': res,
            'residual_mean_arcsec': res,
            'residual_std_arcsec': 0.004,
            'abs_arcsec': res,
            'dra_deg': 0.0,
            'ddec_deg': 0.0,
        }

    with (
        patch(
            'st123.stages.alignment.hst_jhat.validate_hst_group_internal_alignment',
            return_value={'ok': True, 'max_abs_arcsec': 0.03},
        ),
        patch(
            'st123.stages.alignment.hst_jhat.measure_hst_narrowband_residual_vs_refcat',
            side_effect=_residual,
        ),
        patch(
            'st123.stages.alignment.hst_jhat.polish_hst_jhat_abs_residual',
            return_value={'applied': False, 'ok': True},
        ),
        patch(
            'st123.stages.alignment.hst_jhat.measure_hst_sky_offset_vs_gaia',
            return_value={'ok': False},
        ),
        patch(
            'st123.stages.alignment.hst_jhat.recover_catastrophic_hst_jhat',
            return_value={
                'ok': True,
                'applied': True,
                'method': 'gaia_2dhist',
                'abs_arcsec': 2.48,
            },
        ),
        patch('st123.stages.alignment.hst_jhat.find_hst_l3_refcat', return_value=refcat),
        patch('st123.stages.alignment.hst_jhat.find_hst_abs_ref_image', return_value=None),
        patch(
            'st123.utils.helpers.get_filter',
            side_effect=lambda p: 'F606W' if 'j9np06' in Path(p).name else 'F555W',
        ),
        patch('st123.utils.helpers.get_instrument', return_value='acs'),
    ):
        qa = finalize_hst_jhat_dir_quality(jhat_dir, results)

    by_path = {Path(r['outpath']).name: r for r in results}
    assert by_path['j9np06y0q_jhat.fits']['status'] == 'ok'
    assert by_path['j9np06y0q_jhat.fits']['recover']['applied'] is True
    assert by_path['jey342bkq_jhat.fits']['status'] == 'ok'
    assert qa['mosaic_ok'] is True


def test_polish_reverts_when_residual_worsens(tmp_path: Path):
    from st123.stages.alignment.hst_jhat import polish_hst_jhat_abs_residual

    jhat = _tiny_jhat(tmp_path / 'y0q_jhat.fits')
    with fits.open(jhat, mode='update') as hdul:
        hdul[0].header['CRVAL1'] = 10.0
        hdul[0].header['CRVAL2'] = 20.0
        hdul.flush()
    refcat = tmp_path / 'hub.phot.txt'
    refcat.write_text('ra dec mag\n10.0 20.0 18.0\n')
    calls = {'n': 0}

    def _measure(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] == 1:
            return {
                'ok': True,
                'n_match': 16,
                'abs_arcsec': 0.055,
                'residual_arcsec': 0.055,
                'dra_deg': 0.0001,
                'ddec_deg': 0.0001,
            }
        return {
            'ok': True,
            'n_match': 16,
            'abs_arcsec': 0.316,
            'residual_arcsec': 0.316,
            'dra_deg': 0.0,
            'ddec_deg': 0.0,
        }

    with (
        patch(
            'st123.stages.alignment.hst_jhat.measure_hst_narrowband_residual_vs_refcat',
            side_effect=_measure,
        ),
        patch(
            'st123.stages.alignment.hst_jhat.apply_sky_translation_to_sci',
            return_value=1,
        ) as apply_shift,
    ):
        report = polish_hst_jhat_abs_residual(jhat, refcat, min_matches=12)

    assert report['applied'] is False
    assert report['reverted'] is True
    assert apply_shift.call_count == 2
    d_apply = apply_shift.call_args_list[0].args[1:3]
    d_revert = apply_shift.call_args_list[1].args[1:3]
    assert d_revert == (-d_apply[0], -d_apply[1])


def test_l3_qa_fails_zero_match_pair(tmp_path: Path):
    from st123.stages.alignment.hst_jhat import validate_hst_coadds_alignment

    a = tmp_path / 'coadd_f606w.fits'
    b = tmp_path / 'coadd_f814w.fits'
    a.write_bytes(b'')
    b.write_bytes(b'')
    miss = {
        'ok': False,
        'n_match': 0,
        'abs_arcsec': 0.0,
        'dra_arcsec': 0.0,
        'ddec_arcsec': 0.0,
        'method': 'none',
    }
    with (
        patch(
            'st123.stages.alignment.hst_jhat.measure_hst_sky_offset_2dhist',
            return_value=miss,
        ),
        patch(
            'st123.stages.alignment.hst_jhat.measure_hst_sky_offset_smooth_peak',
            return_value=miss,
        ),
        patch(
            'st123.stages.alignment.hst_jhat.measure_hst_frame_relative_offset_pixel',
            return_value=miss,
        ),
        patch(
            'st123.stages.alignment.hst_jhat._measure_offset_one_sci',
            return_value=miss,
        ),
    ):
        report = validate_hst_coadds_alignment([a, b])
    assert report['ok'] is False
    assert report['failed_pairs']
    assert report['failed_pairs'][0]['reason'] == 'no_common_sources'
    assert report['failed_pairs'][0]['n_match'] == 0


def _peak_stamp(path: Path, x: int, y: int, *, nx: int = 320, ny: int = 320) -> Path:
    """North-up stamp with a Gaussian blob at pixel (x, y)."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    sci = np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / (4.0 ** 2)).astype(
        np.float32
    )
    hdr = fits.Header(
        {
            'CRVAL1': 15.0,
            'CRVAL2': -7.5,
            'CRPIX1': nx / 2.0 + 0.5,
            'CRPIX2': ny / 2.0 + 0.5,
            'CD1_1': -0.1 / 3600.0,
            'CD1_2': 0.0,
            'CD2_1': 0.0,
            'CD2_2': 0.1 / 3600.0,
            'CTYPE1': 'RA---TAN',
            'CTYPE2': 'DEC--TAN',
            'CUNIT1': 'deg',
            'CUNIT2': 'deg',
        }
    )
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(sci, header=hdr, name='SCI')]
    ).writeto(path)
    return path


def test_smooth_peak_offset_west_north(tmp_path: Path):
    """Shared-stamp galaxy peak recovers 12\" west / 4\" north (F606W class)."""
    from st123.stages.alignment.hst_jhat import measure_hst_sky_offset_smooth_peak

    ref = _peak_stamp(tmp_path / 'f555w.fits', 80, 80)
    # +x is west (CD1_1 < 0); +y is north. 0.1\"/pix.
    img = _peak_stamp(tmp_path / 'f606w.fits', 80 + 120, 80 + 40)
    st = measure_hst_sky_offset_smooth_peak(img, ref, max_offset_arcsec=90.0)
    assert st['ok'] is True
    assert st['dra_arcsec'] == pytest.approx(-12.0, abs=0.3)
    assert st['ddec_arcsec'] == pytest.approx(4.0, abs=0.3)
    assert st['abs_arcsec'] == pytest.approx(12.649, abs=0.4)


def test_l3_qa_fails_25arcsec_galaxy_peak(tmp_path: Path):
    from st123.stages.alignment.hst_jhat import validate_hst_coadds_alignment

    a = tmp_path / 'coadd_f555w.fits'
    b = tmp_path / 'coadd_f606w.fits'
    a.write_bytes(b'')
    b.write_bytes(b'')
    peak = {
        'ok': True,
        'method': 'smooth_peak',
        'n_match': 1,
        'abs_arcsec': 25.784,
        'dra_arcsec': -25.084,
        'ddec_arcsec': 5.965,
    }
    with patch(
        'st123.stages.alignment.hst_jhat.measure_hst_sky_offset_smooth_peak',
        return_value=peak,
    ):
        report = validate_hst_coadds_alignment([a, b])
    assert report['ok'] is False
    assert report['failed_pairs'][0]['reason'] == 'image_offset'
    assert report['failed_pairs'][0]['abs_arcsec'] == pytest.approx(25.784)
    assert report['max_abs_arcsec'] == pytest.approx(25.784)
