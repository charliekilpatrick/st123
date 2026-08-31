"""Unit tests for HST AstroDrizzle grouping and mosaic/dolphot CLI flags."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from st123.stages.mosaic.hst_drizzle import (
    _assert_drizzle_product_nonempty,
    _finalize_drizzle_product,
    _header_exptime,
    _repair_zero_exptime,
    fill_drizzle_uncovered_with_sky,
    group_hst_frames,
    mask_wfc3_ir_bad_pixels,
    mask_wfpc2_overscan,
)
from st123.scripts import dolphot as dolphot_script
from st123.scripts import mosaic as mosaic_script
from st123.utils import helpers
from st123.datamodels.hst.wfc3_ir import WFC3IRDataModel
from st123.datamodels.hst.wfpc2 import WFPC2DataModel


def _write_hst_frame(
    path: Path,
    *,
    instrument: str,
    filt: str | None = None,
    filtnam1: str | None = None,
    filter1: object | None = None,
    photmode: str | None = None,
    aperture: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU()
    if instrument:
        primary.header['INSTRUME'] = instrument
    if filt:
        primary.header['FILTER'] = filt
    if filtnam1:
        primary.header['FILTNAM1'] = filtnam1
    if filter1 is not None:
        primary.header['FILTER1'] = filter1
    if aperture:
        primary.header['APERTURE'] = aperture
    primary.header['EXPTIME'] = 100.0
    sci = fits.ImageHDU(np.ones((8, 8), dtype=np.float32), name='SCI')
    if photmode:
        sci.header['PHOTMODE'] = photmode
    fits.HDUList([primary, sci]).writeto(path, overwrite=True)
    return path


def test_group_hst_frames_by_instrument_filter(tmp_path: Path):
    wfc3_a = _write_hst_frame(
        tmp_path / 'a_jhat.fits', instrument='WFC3', filt='F336W'
    )
    wfc3_b = _write_hst_frame(
        tmp_path / 'b_jhat.fits', instrument='WFC3', filt='F336W'
    )
    wfc3_c = _write_hst_frame(
        tmp_path / 'c_jhat.fits', instrument='WFC3', filt='F625W'
    )
    wfpc2 = _write_hst_frame(
        tmp_path / 'd_jhat.fits',
        instrument='WFPC2',
        filtnam1='F814W',
        filter1=38,
    )
    groups = group_hst_frames([wfc3_a, wfc3_b, wfc3_c, wfpc2])
    assert set(groups) == {('wfc3', 'f336w'), ('wfc3', 'f625w'), ('wfpc2', 'f814w')}
    assert groups[('wfc3', 'f336w')] == sorted([wfc3_a.resolve(), wfc3_b.resolve()])
    assert groups[('wfpc2', 'f814w')] == [wfpc2.resolve()]


def test_group_hst_frames_photmode_fallback(tmp_path: Path):
    """JHAT products may strip INSTRUME/FILTER; PHOTMODE still encodes both."""
    path = tmp_path / 'stripped_jhat.fits'
    primary = fits.PrimaryHDU()
    primary.header['TELESCOP'] = 'HST'
    primary.header['APERTURE'] = 'UVIS'
    primary.header['EXPTIME'] = 180.0
    sci = fits.ImageHDU(np.ones((8, 8), dtype=np.float32), name='SCI')
    sci.header['PHOTMODE'] = 'WFC3 UVIS1 F336W MJD#60334.1378'
    fits.HDUList([primary, sci]).writeto(path)

    assert helpers.get_instrument(path) == 'wfc3'
    assert helpers.get_filter(path) == 'f336w'
    groups = group_hst_frames([path])
    assert list(groups) == [('wfc3', 'f336w')]


def test_get_filter_prefers_filtnam1_over_numeric_filter1(tmp_path: Path):
    path = _write_hst_frame(
        tmp_path / 'wfpc2.fits',
        instrument='WFPC2',
        filtnam1='F450W',
        filter1=38,
    )
    assert helpers.get_filter(path) == 'f450w'


def test_mosaic_parser_accepts_telescope_hst():
    parser = mosaic_script.create_parser()
    args = parser.parse_args(
        ['--telescope', 'hst', '--base-dir', '/data/proj', '--ncores', '4', '-v']
    )
    assert args.telescope == 'hst'
    assert args.base_dir == '/data/proj'
    assert args.ncores == 4


def test_mosaic_parser_telescope_default_none():
    """Omitted --telescope keeps legacy JWST SW mosaic (not NIRCAM+MIRI)."""
    parser = mosaic_script.create_parser()
    args = parser.parse_args(['--base-dir', '/b'])
    assert args.telescope is None


def test_dolphot_prep_parser_accepts_hst_instruments():
    from st123.scripts.utils.options import resolve_photometry_instrument

    parser = dolphot_script.create_parser()
    for inst in ('acs', 'wfc3', 'wfpc2', 'hst'):
        args = parser.parse_args(
            ['--instruments', inst, '--base-dir', '/data/HST/SN', '-v']
        )
        assert args.instruments == [inst]
        assert resolve_photometry_instrument(args.instruments) == inst
        assert args.base_dir == '/data/HST/SN'


def test_mask_wfpc2_overscan_flags_edges_and_floor(tmp_path: Path):
    ny = nx = 64
    sci = np.full((ny, nx), 10.0, dtype=np.float32)
    sci[:, :8] = -200.0  # left overscan strip
    sci[30, 30] = -80.0  # interior floor pixel
    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = 'WFPC2'
    c0m = tmp_path / 'u_test_c0m.fits'
    c1m = tmp_path / 'u_test_c1m.fits'
    fits.HDUList(
        [primary, fits.ImageHDU(sci.copy(), name='SCI')]
    ).writeto(c0m)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.zeros((ny, nx), dtype=np.int16), name='SCI'),
        ]
    ).writeto(c1m)

    summary = mask_wfpc2_overscan(
        c0m, c1m, edge=8, left_extra=0, sci_floor=-50.0, bad_grow=0
    )
    assert summary['n_edge'] > 0
    assert summary['n_floor'] > 0

    with fits.open(c0m) as hdul:
        data = hdul['SCI'].data
        assert float(data[30, 30]) == 0.0
        assert float(data[10, 2]) == 0.0
        assert hdul[0].header.get('ST123OVS') is True
    with fits.open(c1m) as hdul:
        dq = hdul[1].data
        assert int(dq[10, 2]) & WFPC2DataModel.OVERSCAN_DQ_BIT
        assert int(dq[30, 30]) & WFPC2DataModel.OVERSCAN_DQ_BIT
        # Interior good pixel stays unflagged.
        assert int(dq[40, 40]) == 0


def test_wfc3_ir_driz_cr_disabled_and_floor_bit_excluded():
    assert WFC3IRDataModel.DRIZ_CR is False
    assert (WFC3IRDataModel.BAD_DQ_BIT & WFC3IRDataModel.DRIZ_BITS) == 0


def test_mask_wfc3_ir_bad_pixels_floor_and_grow(tmp_path: Path):
    ny = nx = 32
    sci = np.full((ny, nx), 10.0, dtype=np.float32)
    sci[15, 15] = -236.0  # native MAST-like spike
    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = 'WFC3'
    primary.header['DETECTOR'] = 'IR'
    primary.header['FILTER'] = 'F160W'
    path = tmp_path / 'iejn62apq_flt.fits'
    fits.HDUList(
        [
            primary,
            fits.ImageHDU(sci.copy(), name='SCI'),
            fits.ImageHDU(np.ones((ny, nx), dtype=np.float32), name='ERR'),
            fits.ImageHDU(np.zeros((ny, nx), dtype=np.int16), name='DQ'),
        ]
    ).writeto(path)

    summary = mask_wfc3_ir_bad_pixels(
        path, sci_floor=-50.0, bad_grow=1, dq_bit=WFC3IRDataModel.BAD_DQ_BIT
    )
    assert summary['n_floor'] == 1
    assert summary['n_grown'] >= 1

    with fits.open(path) as hdul:
        assert float(hdul['SCI'].data[15, 15]) == 0.0
        assert int(hdul['DQ'].data[15, 15]) & WFC3IRDataModel.BAD_DQ_BIT
        # Grown neighbor also flagged / zeroed.
        assert int(hdul['DQ'].data[15, 16]) & WFC3IRDataModel.BAD_DQ_BIT
        assert float(hdul['SCI'].data[20, 20]) == 10.0
        assert int(hdul['DQ'].data[20, 20]) == 0


def test_mask_wfpc2_overscan_left_extra_and_grow(tmp_path: Path):
    ny = nx = 64
    sci = np.full((ny, nx), 10.0, dtype=np.float32)
    sci[:, :12] = -100.0
    c0m = tmp_path / 'u_left_c0m.fits'
    c1m = tmp_path / 'u_left_c1m.fits'
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(sci.copy(), name='SCI')]
    ).writeto(c0m)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.zeros((ny, nx), dtype=np.int16), name='SCI'),
        ]
    ).writeto(c1m)

    summary = mask_wfpc2_overscan(
        c0m,
        c1m,
        edge=4,
        left_extra=8,
        sci_floor=-50.0,
        bad_grow=2,
        bad_col_frac=0.5,
    )
    assert summary['n_edge'] > 0
    assert summary['n_cols'] > 0
    with fits.open(c1m) as hdul:
        dq = hdul[1].data
        # Left-extra region flagged.
        assert int(dq[32, 10]) & WFPC2DataModel.OVERSCAN_DQ_BIT
        # Grow reaches a few pixels past the left strip.
        assert int(dq[32, 13]) & WFPC2DataModel.OVERSCAN_DQ_BIT
        assert int(dq[32, 40]) == 0


def test_mask_wfpc2_high_variance_edge_columns(tmp_path: Path):
    """Noisy outer columns beyond the uniform edge are flagged via noise cut."""
    ny = nx = 128
    rng = np.random.default_rng(1)
    sci = rng.normal(10.0, 0.5, size=(ny, nx)).astype(np.float32)
    # High-noise sector just outside an 8-px uniform edge (cols 10-18).
    # Add a bright "star" column deeper in so raw scatter alone would false-flag.
    sci[:, 10:18] = rng.normal(10.0, 20.0, size=(ny, 8)).astype(np.float32)
    sci[40:48, 50] = 500.0
    c0m = tmp_path / 'u_var_c0m.fits'
    c1m = tmp_path / 'u_var_c1m.fits'
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(sci.copy(), name='SCI')]
    ).writeto(c0m)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.zeros((ny, nx), dtype=np.int16), name='SCI'),
        ]
    ).writeto(c1m)

    summary = mask_wfpc2_overscan(
        c0m,
        c1m,
        edge=8,
        left_extra=0,
        sci_floor=-50.0,
        bad_grow=0,
        bad_col_frac=0.99,
        var_edge=24,
        var_sigma=4.0,
    )
    assert summary['n_var_cols'] > 0
    with fits.open(c1m) as hdul:
        dq = hdul[1].data
        assert int(dq[64, 12]) & WFPC2DataModel.OVERSCAN_DQ_BIT
        # Deep interior (incl. bright column) stays clean - scene != noise.
        assert int(dq[64, 64]) == 0
        assert int(dq[44, 50]) == 0


def test_finalize_drizzle_product_removes_drw(tmp_path: Path):
    stem = tmp_path / 'coadd_wfpc2_f450w'
    drw = tmp_path / 'coadd_wfpc2_f450w_drw.fits'
    drz = tmp_path / 'coadd_wfpc2_f450w_drz.fits'
    fits.HDUList([fits.PrimaryHDU(np.ones((4, 4), dtype=np.float32))]).writeto(drw)
    out = _finalize_drizzle_product(drw, drz, output_stem=stem)
    assert out == drz.resolve()
    assert drz.is_file()
    assert not drw.exists()


def test_fill_drizzle_uncovered_with_sky_ctx_and_nan(tmp_path: Path):
    """CTX=0 and NaN SCI pixels become the illuminated median; WHT stays 0."""
    ny = nx = 32
    sci = np.full((ny, nx), 2.5, dtype=np.float32)
    sci[0:8, :] = np.nan  # uncovered NaNs
    sci[20, 20] = 100.0  # bright outlier (should not dominate sky)
    wht = np.ones((ny, nx), dtype=np.float32)
    wht[0:8, :] = 0.0
    # Multi-bit CTX in the core so single-bit drop (WFPC2 default) is a no-op here.
    ctx = np.full((ny, nx), 3, dtype=np.int32)  # bits 0+1
    ctx[0:8, :] = 0
    path = tmp_path / 'coadd_wfpc2_f606w_drz.fits'
    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = 'WFPC2'
    fits.HDUList(
        [
            primary,
            fits.ImageHDU(sci, name='SCI'),
            fits.ImageHDU(wht, name='WHT'),
            fits.ImageHDU(ctx, name='CTX'),
        ]
    ).writeto(path)

    summary = fill_drizzle_uncovered_with_sky(path)
    assert summary['n_filled'] > 0
    assert summary['sky'] is not None
    with fits.open(path) as hdul:
        data = np.asarray(hdul['SCI'].data)
        assert np.isfinite(data).all()
        assert np.allclose(data[0:8, :], summary['sky'], rtol=0, atol=1e-5)
        assert float(hdul['WHT'].data[2, 2]) == 0.0
        assert int(hdul['CTX'].data[2, 2]) == 0
        # Illuminated core unchanged (aside from the bright pixel).
        assert float(data[16, 16]) == 2.5
        assert hdul[0].header.get('ST123CTX') is True


def test_fill_drizzle_drops_wfpc2_edge_single_bit_ctx(tmp_path: Path):
    """WFPC2 drops only near-edge single-bit CTX; interior singles are kept."""
    ny = nx = 48
    sci = np.full((ny, nx), 2.5, dtype=np.float32)
    wht = np.full((ny, nx), 100.0, dtype=np.float32)
    ctx = np.full((ny, nx), 3, dtype=np.int32)  # dual coverage core
    # Uncovered rim on the left.
    ctx[:, 0:2] = 0
    wht[:, 0:2] = 0.0
    sci[:, 0:2] = np.nan
    # Noisy single-coverage overhang next to the rim (should be dropped).
    sci[:, 2:8] = 9.0
    ctx[:, 2:8] = 1
    # Interior single-bit island far from the rim (should be kept).
    sci[20:28, 30:38] = 4.0
    ctx[20:28, 30:38] = 1
    path = tmp_path / 'coadd_wfpc2_f814w_drz.fits'
    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = 'WFPC2'
    fits.HDUList(
        [
            primary,
            fits.ImageHDU(sci, name='SCI'),
            fits.ImageHDU(wht, name='WHT'),
            fits.ImageHDU(ctx, name='CTX'),
        ]
    ).writeto(path)

    summary = fill_drizzle_uncovered_with_sky(
        path, drop_single_ctx_edge_pix=16
    )
    assert summary['drop_single_ctx_edge_pix'] == 16
    assert summary['n_single_ctx'] > 0
    with fits.open(path) as hdul:
        # Edge single-bit overhang cleared.
        assert np.allclose(hdul['SCI'].data[:, 2:8], summary['sky'])
        assert np.all(hdul['WHT'].data[:, 2:8] == 0)
        assert np.all(hdul['CTX'].data[:, 2:8] == 0)
        # Interior single-bit island retained.
        assert float(hdul['SCI'].data[24, 34]) == 4.0
        assert int(hdul['CTX'].data[24, 34]) == 1
        assert float(hdul['SCI'].data[16, 16]) == 2.5
        assert hdul[0].header.get('ST123SGE') == 16
        assert hdul[0].header.get('ST123SGL') is None

    # WFC3 products do not apply the edge-single cut by default.
    path2 = tmp_path / 'coadd_wfc3_f625w_drc.fits'
    primary2 = fits.PrimaryHDU()
    primary2.header['INSTRUME'] = 'WFC3'
    sci2 = np.full((ny, nx), 2.5, dtype=np.float32)
    sci2[:, 2:8] = 9.0
    wht2 = np.full((ny, nx), 100.0, dtype=np.float32)
    ctx2 = np.full((ny, nx), 3, dtype=np.int32)
    ctx2[:, 0:2] = 0
    wht2[:, 0:2] = 0.0
    ctx2[:, 2:8] = 1
    fits.HDUList(
        [
            primary2,
            fits.ImageHDU(sci2, name='SCI'),
            fits.ImageHDU(wht2, name='WHT'),
            fits.ImageHDU(ctx2, name='CTX'),
        ]
    ).writeto(path2)
    summary2 = fill_drizzle_uncovered_with_sky(path2)
    assert summary2['drop_single_ctx_edge_pix'] == 0
    with fits.open(path2) as hdul:
        assert float(hdul['SCI'].data[16, 4]) == 9.0
        assert int(hdul['CTX'].data[16, 4]) == 1


def test_repair_zero_exptime_from_expstart_expend(tmp_path: Path):
    """ACS FLCs can ship EXPTIME=0 with valid EXPSTART/EXPEND (MJD)."""
    path = tmp_path / 'acs_zero_exptime.fits'
    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = 'ACS'
    primary.header['EXPTIME'] = 0.0
    # ~381 s
    primary.header['EXPSTART'] = 60256.96379241
    primary.header['EXPEND'] = 60256.96820280
    sci = fits.ImageHDU(np.ones((8, 8), dtype=np.float32), name='SCI')
    fits.HDUList([primary, sci]).writeto(path)

    with fits.open(path, mode='update') as hdul:
        repaired = _repair_zero_exptime(hdul)
        hdul.flush()
    assert repaired is not None
    assert 380.0 < repaired < 382.0
    assert 380.0 < float(fits.getval(path, 'EXPTIME')) < 382.0
    assert 380.0 < _header_exptime(path) < 382.0


def test_assert_drizzle_product_nonempty_rejects_naxis0(tmp_path: Path):
    path = tmp_path / 'empty_drc.fits'
    primary = fits.PrimaryHDU()
    primary.header['EXPTIME'] = 0.0
    sci = fits.ImageHDU(name='SCI')  # NAXIS=0 / no data
    fits.HDUList([primary, sci]).writeto(path)
    try:
        _assert_drizzle_product_nonempty(path)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_astrodrizzle_wcs_kwargs_from_shared_box():
    """Boxed HST drizzle must pin final_* geometry to the shared stamp."""
    from astropy.wcs import WCS

    from st123.stages.mosaic.hst_drizzle import (
        _wcs_orientat_deg,
        astrodrizzle_wcs_kwargs_from_header,
        build_boxed_drizzle_wcs,
        recenter_wcs_on_array,
    )
    from st123.stages.mosaic.mosaic import slice_box_wcs
    import shapely

    group = WCS(naxis=2)
    group.wcs.crpix = [500.0, 500.0]
    group.wcs.cdelt = [-0.031 / 3600.0, 0.031 / 3600.0]
    group.wcs.crval = [185.742, 15.830]
    group.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    group.pixel_shape = (1000, 1000)
    group._naxis = [1000, 1000]
    bbox = shapely.box(100, 200, 400, 500)
    box_wcs = slice_box_wcs(group, bbox)
    hdr, out_wcs = build_boxed_drizzle_wcs(box_wcs, 0.05)
    kwargs = astrodrizzle_wcs_kwargs_from_header(hdr)
    nx, ny = int(hdr['NAXIS1']), int(hdr['NAXIS2'])
    assert kwargs['final_wcs'] is True
    assert kwargs['final_outnx'] == nx
    assert kwargs['final_outny'] == ny
    assert kwargs['final_scale'] == pytest.approx(0.05, rel=0.05)
    # CRPIX must sit on the array (not a group-level exterior CRPIX).
    assert 1.0 <= float(kwargs['final_crpix1']) <= nx
    assert 1.0 <= float(kwargs['final_crpix2']) <= ny
    assert kwargs['final_crpix1'] == pytest.approx((nx + 1) * 0.5)
    assert kwargs['final_crpix2'] == pytest.approx((ny + 1) * 0.5)
    assert kwargs['final_ra'] == pytest.approx(float(out_wcs.wcs.crval[0]))
    assert kwargs['final_dec'] == pytest.approx(float(out_wcs.wcs.crval[1]))
    assert kwargs['final_rot'] == pytest.approx(_wcs_orientat_deg(out_wcs))
    recentered = recenter_wcs_on_array(out_wcs)
    fp0 = np.asarray(out_wcs.calc_footprint(center=False), dtype=float)
    fp1 = np.asarray(recentered.calc_footprint(center=False), dtype=float)
    assert np.allclose(fp0, fp1, atol=1e-8)


def test_filter_frames_overlapping_box_drops_nonoverlap(tmp_path: Path):
    from astropy.wcs import WCS

    from st123.stages.mosaic.mosaic import filter_frames_overlapping_box
    import shapely

    def _write(path: Path, ra: float, dec: float) -> Path:
        primary = fits.PrimaryHDU()
        primary.header['INSTRUME'] = 'ACS'
        sci = fits.ImageHDU(np.ones((32, 32), dtype=np.float32), name='SCI')
        sci.header['CRPIX1'] = 16.0
        sci.header['CRPIX2'] = 16.0
        sci.header['CRVAL1'] = ra
        sci.header['CRVAL2'] = dec
        sci.header['CDELT1'] = -0.05 / 3600.0
        sci.header['CDELT2'] = 0.05 / 3600.0
        sci.header['CTYPE1'] = 'RA---TAN'
        sci.header['CTYPE2'] = 'DEC--TAN'
        fits.HDUList([primary, sci]).writeto(path)
        return path

    inside = _write(tmp_path / 'in_jhat.fits', 185.742, 15.830)
    outside = _write(tmp_path / 'out_jhat.fits', 185.80, 15.90)
    group = WCS(naxis=2)
    group.wcs.crpix = [100.0, 100.0]
    group.wcs.cdelt = [-0.05 / 3600.0, 0.05 / 3600.0]
    group.wcs.crval = [185.742, 15.830]
    group.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    group.pixel_shape = (200, 200)
    group._naxis = [200, 200]
    # Pixel box around the CRVAL pointing.
    bbox = shapely.box(50, 50, 150, 150)
    kept = filter_frames_overlapping_box(
        [inside, outside], bbox, mosaic_wcs=group, min_overlap=0.01
    )
    assert str(inside) in kept
    assert str(outside) not in kept
