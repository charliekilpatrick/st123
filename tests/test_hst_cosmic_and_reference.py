"""Tests for WFPC2 c1m helpers, astroscrappy CR clean, and L3 Gaia scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from st123.photometry.cosmic import run_cosmic, wfpc2_c1m_path
from st123.photometry.dolphot import _wfpc2_dq_companion
from st123.alignment.hst_reference import (
    Level3GaiaScore,
    _illuminated_mask,
    _local_detectable,
    count_phot_sources,
    ensure_l3_science_refcat,
    pick_best_level3,
    write_detection_refcat,
)
from st123.utils.settings import HST_CR_DQ_BIT, hst_crpars, hst_driz_bits


def _wcs_header(nx: int = 64, ny: int = 64) -> fits.Header:
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.cdelt = np.array([-0.05 / 3600.0, 0.05 / 3600.0])
    w.wcs.crval = [177.65595, 55.35359]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    hdr = w.to_header()
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = nx
    hdr['NAXIS2'] = ny
    return hdr


def test_hst_settings_cr_and_bits():
    assert hst_driz_bits['wfpc2'] == 1032
    assert hst_driz_bits['wfc3'] == 96
    assert 'wfpc2' in hst_crpars
    assert HST_CR_DQ_BIT == 4096


def test_wfpc2_c1m_path_and_companion(tmp_path: Path):
    c0m = tmp_path / 'u65w4801r_c0m.fits'
    c1m = tmp_path / 'u65w4801r_c1m.fits'
    fits.HDUList([fits.PrimaryHDU()]).writeto(c0m)
    fits.HDUList([fits.PrimaryHDU()]).writeto(c1m)
    assert wfpc2_c1m_path(c0m) == c1m
    assert _wfpc2_dq_companion(c0m) == c1m.resolve()

    jhat = tmp_path / 'u65w4801r_jhat.fits'
    fits.HDUList([fits.PrimaryHDU()]).writeto(jhat)
    assert _wfpc2_dq_companion(jhat) == c1m.resolve()


def test_run_cosmic_flags_wfpc2_c1m(tmp_path: Path):
    """Synthetic hot pixels should be cleaned and flagged in c1m."""
    rng = np.random.default_rng(0)
    ny = nx = 128
    sci = rng.normal(100.0, 2.0, size=(ny, nx)).astype(np.float32)
    # Plant obvious cosmic-ray spikes.
    cr_coords = [(20, 20), (40, 60), (80, 90), (100, 30), (50, 110)]
    for y, x in cr_coords:
        sci[y, x] = 5000.0
        sci[y + 1, x] = 3000.0

    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = 'WFPC2'
    primary.header['FILTNAM1'] = 'F814W'
    primary.header['ATODGAIN'] = 7.0
    primary.header['EXPTIME'] = 500.0
    sci_hdu = fits.ImageHDU(sci, name='SCI')
    c0m = tmp_path / 'plant_c0m.fits'
    fits.HDUList([primary, sci_hdu]).writeto(c0m)

    dq = np.zeros((ny, nx), dtype=np.int16)
    c1m = tmp_path / 'plant_c1m.fits'
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(dq, name='SCI')]
    ).writeto(c1m)

    summary = run_cosmic(c0m, instrument='wfpc2', add_crmask=True, inplace=True)
    assert summary['n_sci'] == 1
    assert summary['n_cr_pixels'] >= 3

    with fits.open(c0m) as hdul:
        cleaned = hdul['SCI'].data
        assert hdul[0].header.get('ST123CR') is True
    with fits.open(c1m) as hdul:
        dq2 = hdul[1].data

    # At least half of planted CR cores should be flagged.
    flagged = sum(1 for y, x in cr_coords if dq2[y, x] == HST_CR_DQ_BIT)
    assert flagged >= 3
    # Cleaned values at CR cores should drop well below the planted spike.
    assert float(cleaned[20, 20]) < 1000.0


def test_run_cosmic_preserves_extra_hdus(tmp_path: Path):
    """In-place update must not drop non-SCI/ERR/DQ extensions (HDRLET stand-in)."""
    ny = nx = 32
    rng = np.random.default_rng(1)
    sci = rng.normal(10.0, 1.0, size=(ny, nx)).astype(np.float32)
    sci[5, 5] = 8000.0
    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = 'WFC3'
    primary.header['FILTER'] = 'F625W'
    err = fits.ImageHDU(np.ones_like(sci), name='ERR')
    dq = fits.ImageHDU(np.zeros((ny, nx), dtype=np.int16), name='DQ')
    # Trailing extension that writeto historically truncated/corrupted.
    extra = fits.ImageHDU(np.arange(112320, dtype=np.float32).reshape(-1), name='HDRLET')
    path = tmp_path / 'plant_flc.fits'
    fits.HDUList(
        [primary, fits.ImageHDU(sci, name='SCI'), err, dq, extra]
    ).writeto(path)
    size_before = path.stat().st_size
    n_before = len(fits.open(path))

    run_cosmic(path, instrument='wfc3', add_crmask=True, inplace=True)

    with fits.open(path) as hdul:
        assert len(hdul) == n_before
        assert 'HDRLET' in hdul
        assert hdul['HDRLET'].data is not None
        assert hdul['HDRLET'].data.size == 112320
        assert hdul['DQ'].data[5, 5] == HST_CR_DQ_BIT
    # Size should stay in the same ballpark (update, not truncated rewrite).
    assert path.stat().st_size >= size_before * 0.98


def test_illuminated_and_detectable_helpers():
    sci = np.full((32, 32), 10.0, dtype=float)
    sci[10, 10] = 50.0
    wht = np.ones_like(sci)
    wht[:, :2] = 0
    mask = _illuminated_mask(sci, wht=wht)
    assert not mask[0, 0]
    assert mask[10, 10]
    assert _local_detectable(sci, mask, 10.0, 10.0, half=3, snr=3.0)
    assert not _local_detectable(sci, mask, 1.0, 1.0, half=3, snr=3.0)


def test_pick_best_level3_ranks_detectable(tmp_path: Path, monkeypatch):
    """Synthetic scores: higher detectable Gaia wins."""
    a = tmp_path / 'coadd_a_drc.fits'
    b = tmp_path / 'coadd_b_drc.fits'
    for path in (a, b):
        hdr = _wcs_header()
        primary = fits.PrimaryHDU()
        primary.header['INSTRUME'] = 'WFC3'
        primary.header['FILTER'] = 'F625W'
        sci = fits.ImageHDU(np.ones((64, 64), dtype=np.float32), header=hdr, name='SCI')
        wht = fits.ImageHDU(np.ones((64, 64), dtype=np.float32), name='WHT')
        fits.HDUList([primary, sci, wht]).writeto(path)

    def fake_score(image, **kwargs):
        path = Path(image)
        if path.name.startswith('coadd_a'):
            return Level3GaiaScore(path, 10, 8, 2, 'wfc3', 'f625w')
        return Level3GaiaScore(path, 10, 9, 7, 'wfc3', 'f625w')

    monkeypatch.setattr(
        'st123.alignment.hst_reference.score_level3_gaia', fake_score
    )
    best, scores = pick_best_level3(candidates=[a, b])
    assert best is not None
    assert best.name.startswith('coadd_b')
    assert scores[0].n_detectable == 7


def _toy_l3_with_stars(path: Path, *, n_stars: int = 40) -> Path:
    rng = np.random.default_rng(0)
    ny = nx = 128
    data = rng.normal(10.0, 1.0, size=(ny, nx)).astype(np.float32)
    for _ in range(n_stars):
        x = int(rng.integers(8, nx - 8))
        y = int(rng.integers(8, ny - 8))
        yy, xx = np.mgrid[-3:4, -3:4]
        data[y - 3 : y + 4, x - 3 : x + 4] += 80.0 * np.exp(
            -(xx * xx + yy * yy) / 2.0
        )
    hdr = _wcs_header(nx, ny)
    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = 'WFC3'
    primary.header['FILTER'] = 'F625W'
    sci = fits.ImageHDU(data, header=hdr, name='SCI')
    wht = fits.ImageHDU(np.ones((ny, nx), dtype=np.float32), name='WHT')
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, sci, wht]).writeto(path, overwrite=True)
    return path


def test_write_detection_refcat_dense(tmp_path: Path):
    img = _toy_l3_with_stars(tmp_path / 'coadd_wfc3_f625w_drc.fits')
    out = tmp_path / 'ref.phot.txt'
    write_detection_refcat(img, out, nsigma=4.0, fwhm=2.0)
    n = count_phot_sources(out)
    assert n >= 20
    header = out.read_text().splitlines()[0]
    assert header.startswith('ra dec mag x y')


def test_list_level3_and_find_hst_abs_ref_boxed(tmp_path: Path):
    from st123.alignment.hst_jhat import find_hst_abs_ref_image
    from st123.alignment.hst_reference import list_level3_products

    ref = tmp_path / 'reference'
    boxed = ref / 'group_0' / 'ref_5'
    boxed.mkdir(parents=True)
    boxed_coadd = _toy_l3_with_stars(
        boxed / 'coadd_0_5_wfc3_f625w_drc.fits', n_stars=5
    )
    flat = _toy_l3_with_stars(ref / 'coadd_acs_f814w_drc.fits', n_stars=5)
    # find_hst_abs_ref_image ignores tiny files (<500 KB).
    for path in (boxed_coadd, flat):
        with path.open('ab') as fh:
            fh.write(b'\0' * 600_000)
    found = list_level3_products(ref)
    assert boxed_coadd.resolve() in found
    assert flat.resolve() in found
    jhat = tmp_path / 'jhat_hst'
    jhat.mkdir()
    assert find_hst_abs_ref_image(jhat) == boxed_coadd.resolve()


def test_ensure_l3_science_refcat_rebuilds_sparse(tmp_path: Path):
    img = _toy_l3_with_stars(tmp_path / 'coadd.fits', n_stars=50)
    out = tmp_path / 'coadd.phot.txt'
    # Stale Gaia-only leftover (too few rows for WFPC2 JHAT).
    out.write_text(
        'ra dec mag x y\n'
        '177.0 55.0 18.0 10.0 10.0\n'
        '177.1 55.1 19.0 20.0 20.0\n'
    )
    assert count_phot_sources(out) == 2
    ensure_l3_science_refcat(img, out, min_sources=30)
    assert count_phot_sources(out) >= 30
