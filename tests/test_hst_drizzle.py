"""Unit tests for HST AstroDrizzle grouping and mosaic/dolphot CLI flags."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from st123.mosaic.hst_drizzle import (
    _finalize_drizzle_product,
    group_hst_frames,
    mask_wfpc2_overscan,
)
from st123.scripts import dolphot as dolphot_script
from st123.scripts import mosaic as mosaic_script
from st123.utils import helpers
from st123.utils.settings import WFPC2_OVERSCAN_DQ_BIT


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


def test_mosaic_parser_defaults_jwst():
    parser = mosaic_script.create_parser()
    args = parser.parse_args(['--base-dir', '/b'])
    assert args.telescope == 'jwst'


def test_dolphot_prep_parser_accepts_hst_instruments():
    parser = dolphot_script.create_parser()
    for inst in ('acs', 'wfc3', 'wfpc2'):
        args = parser.parse_args(
            ['--instrument', inst, '--base-dir', '/data/HST/SN', '-v']
        )
        assert args.instrument == inst
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
        assert hdul[0].header.get('ST123OVSC') is True
    with fits.open(c1m) as hdul:
        dq = hdul[1].data
        assert int(dq[10, 2]) & WFPC2_OVERSCAN_DQ_BIT
        assert int(dq[30, 30]) & WFPC2_OVERSCAN_DQ_BIT
        # Interior good pixel stays unflagged.
        assert int(dq[40, 40]) == 0


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
        assert int(dq[32, 10]) & WFPC2_OVERSCAN_DQ_BIT
        # Grow reaches a few pixels past the left strip.
        assert int(dq[32, 13]) & WFPC2_OVERSCAN_DQ_BIT
        assert int(dq[32, 40]) == 0


def test_finalize_drizzle_product_removes_drw(tmp_path: Path):
    stem = tmp_path / 'coadd_wfpc2_f450w'
    drw = tmp_path / 'coadd_wfpc2_f450w_drw.fits'
    drz = tmp_path / 'coadd_wfpc2_f450w_drz.fits'
    fits.HDUList([fits.PrimaryHDU(np.ones((4, 4), dtype=np.float32))]).writeto(drw)
    out = _finalize_drizzle_product(drw, drz, output_stem=stem)
    assert out == drz.resolve()
    assert drz.is_file()
    assert not drw.exists()
