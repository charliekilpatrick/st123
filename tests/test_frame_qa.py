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
    FRAME_INTERNAL_TOL_ARCSEC,
    FRAME_MIN_MATCH_HEALTHY,
    FRAME_SPARSE_TOL_ARCSEC,
    abs_ok,
    build_frame_qa,
    coherent_tol_arcsec,
    internal_ok,
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
    assert warn_if_frame_qa_soft(report) is False


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
