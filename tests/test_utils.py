"""Unit tests for ``st123.utils`` (helpers, settings, link)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from helpers import make_wcs_header
from st123 import utils as utils_pkg
from st123.utils import helpers, link, settings


def _write_jwst_cal(
    path: Path,
    *,
    filt: str = 'F150W',
    instrument: str = 'NIRCAM',
    module: str = 'A',
    detector: str = 'NRCA1',
    pupil: str = 'CLEAR',
    visit_id: str = 'V001',
    exptime: float = 100.0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ones((20, 20), dtype=np.float32)
    primary = fits.PrimaryHDU()
    primary.header['FILTER'] = filt
    primary.header['INSTRUME'] = instrument
    primary.header['MODULE'] = module
    primary.header['DETECTOR'] = detector
    primary.header['PUPIL'] = pupil
    primary.header['VISIT_ID'] = visit_id
    primary.header['EFFEXPTM'] = exptime
    primary.header['DATE-OBS'] = '2024-01-01'
    primary.header['TIME-OBS'] = '12:00:00'
    sci_hdr = make_wcs_header(shape=data.shape)
    sci_hdr['PHOTPLAM'] = 15000.0
    sci_hdr['PHOTFLAM'] = 1.0e-20
    fits.HDUList(
        [
            primary,
            fits.ImageHDU(data=data, header=sci_hdr, name='SCI'),
        ]
    ).writeto(path, overwrite=True)
    return path


def test_is_full_frame_miri(tmp_path: Path):
    good = tmp_path / 'full_mirimage_cal.fits'
    data = np.ones((1024, 1032), dtype=np.float32)
    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = 'MIRI'
    primary.header['DETECTOR'] = 'MIRIMAGE'
    primary.header['SUBARRAY'] = 'FULL'
    primary.header['FILTER'] = 'F770W'
    fits.HDUList(
        [primary, fits.ImageHDU(data=data, name='SCI')]
    ).writeto(good)

    bad = tmp_path / 'cutout_mirimage_cal.fits'
    primary2 = fits.PrimaryHDU()
    primary2.header['INSTRUME'] = 'MIRI'
    primary2.header['DETECTOR'] = 'MIRIMAGE'
    primary2.header['SUBARRAY'] = 'SUB64'
    primary2.header['FILTER'] = 'F770W'
    fits.HDUList(
        [
            primary2,
            fits.ImageHDU(data=np.ones((128, 136), dtype=np.float32), name='SCI'),
        ]
    ).writeto(bad)

    assert helpers.is_full_frame_miri(good)
    assert helpers.is_mirimask_compatible(good)
    assert not helpers.is_full_frame_miri(bad)
    assert not helpers.is_full_frame_miri(tmp_path / 'missing.fits')


# --- settings -----------------------------------------------------------------


def test_settings_jhat_param_dicts():
    for name in (
        'strict_gaia_params',
        'relaxed_gaia_params',
        'strict_jwst_params',
        'relaxed_jwst_params',
    ):
        params = getattr(settings, name)
        assert isinstance(params, dict)
        assert params['telescope'] == 'jwst'
        assert 'd2d_max' in params
    assert settings.strict_jwst_params['refcat_racol'] == 'ra'
    assert 'FitSky' in settings.base_params
    assert 'raper' in settings.short_params
    assert 'raper' in settings.long_params


# --- link ---------------------------------------------------------------------


def test_link_create_symlink_and_remove_proc(tmp_path: Path):
    src = tmp_path / 'src.fits'
    src.write_bytes(b'x')
    dst = tmp_path / 'raw' / 'src.fits'
    dst.parent.mkdir()
    link.create_symlink(str(src), str(dst))
    assert dst.is_symlink()

    other = tmp_path / 'other.fits'
    other.write_bytes(b'y')
    remaining = link.remove_proc_files([str(src), str(other)], str(tmp_path))
    assert str(other) in remaining
    assert str(src) not in remaining


# --- helpers: coordinates -----------------------------------------------------


def test_is_number_and_parse_coord():
    assert helpers.is_number('12.5')
    assert helpers.is_number(3)
    assert not helpers.is_number('abc')
    assert not helpers.is_number(None)

    c = helpers.parse_coord(150.0, 2.0)
    assert isinstance(c, SkyCoord)
    assert c.ra.degree == pytest.approx(150.0)
    assert c.dec.degree == pytest.approx(2.0)

    c2 = helpers.parse_coord('10:00:00', '+02:00:00')
    assert isinstance(c2, SkyCoord)

    assert helpers.parse_coord('not-a-coord', 'also-bad') is None


def test_parse_coord_strips_curly_and_ascii_quotes():
    """Shell exports with curly quotes must still parse (issue #3)."""
    plain = helpers.parse_coord('09:53:42.00', '+01:34:06.00')
    assert plain is not None

    curly = helpers.parse_coord('\u201c09:53:42.00\u201d', '\u201c+01:34:06.00\u201d')
    assert curly is not None
    assert curly.ra.degree == pytest.approx(plain.ra.degree)
    assert curly.dec.degree == pytest.approx(plain.dec.degree)

    ascii_q = helpers.parse_coord('"09:53:42.00"', '"+01:34:06.00"')
    assert ascii_q is not None
    assert ascii_q.ra.degree == pytest.approx(plain.ra.degree)

    mixed = helpers.parse_coord("\u201809:53:42.00\u2019", "'+01:34:06.00'")
    assert mixed is not None
    assert mixed.dec.degree == pytest.approx(plain.dec.degree)


# --- helpers: FITS metadata ---------------------------------------------------


def test_get_filter_module_instrument_chip(tmp_path: Path):
    path = _write_jwst_cal(tmp_path / 'img_cal.fits')
    assert helpers.get_filter(str(path)) == 'f150w'
    assert helpers.get_module(str(path)) == 'a'
    assert helpers.get_instrument(str(path)) == 'nircam'
    # JWST files expose DETECTOR rather than CCDCHIP.
    assert helpers.get_chip(str(path)) == 'NRCA1'


def test_get_module_miri_and_nircam_without_module(tmp_path: Path):
    """MIRI has no MODULE keyword; NIRCam can fall back to DETECTOR letter."""
    miri = tmp_path / 'jw03295006001_02101_00001_mirimage_jhat.fits'
    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = 'MIRI'
    primary.header['DETECTOR'] = 'MIRIMAGE'
    primary.header['FILTER'] = 'F560W'
    fits.HDUList(
        [primary, fits.ImageHDU(np.ones((5, 5), dtype=np.float32), name='SCI')]
    ).writeto(miri)
    assert helpers.get_module(str(miri)) == 'miri'

    nrc = tmp_path / 'jw09246001001_02101_00001_nrcb1_jhat.fits'
    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = 'NIRCAM'
    primary.header['DETECTOR'] = 'NRCB1'
    primary.header['FILTER'] = 'F150W'
    # Intentionally omit MODULE.
    fits.HDUList(
        [primary, fits.ImageHDU(np.ones((5, 5), dtype=np.float32), name='SCI')]
    ).writeto(nrc)
    assert helpers.get_module(str(nrc)) == 'b'


def test_get_filter_falls_back_to_filter2(tmp_path: Path):
    path = tmp_path / 'filt2.fits'
    primary = fits.PrimaryHDU()
    primary.header['FILTER1'] = 'CLEAR1'
    primary.header['FILTER2'] = 'F200W'
    primary.header['INSTRUME'] = 'NIRCAM'
    primary.header['DETECTOR'] = 'NRCA1'
    fits.HDUList([primary, fits.ImageHDU(np.ones((5, 5)), name='SCI')]).writeto(
        path
    )
    assert helpers.get_filter(str(path)) == 'f200w'


def test_get_detector_chip_from_filename():
    name = 'jw01234001001_02101_00001_nrca1_cal.fits'
    assert helpers.get_detector_chip(name) == 'nrca1'
    assert helpers.get_detector_chip('no_detector_here.fits') is None


def test_get_zpt_abmag(tmp_path: Path):
    path = _write_jwst_cal(tmp_path / 'zpt.fits')
    zpt = helpers.get_zpt(str(path), zptype='abmag')
    assert zpt is not None
    assert np.isfinite(zpt)


# --- helpers: tables / xmatch -------------------------------------------------


def test_create_filter_table():
    table = Table(
        {
            'filter': ['f150w', 'f200w', 'f150w'],
            'image': ['a.fits', 'b.fits', 'c.fits'],
        }
    )
    by_filt = helpers.create_filter_table(table, ['f150w', 'f200w'])
    assert set(by_filt) == {'f150w', 'f200w'}
    assert len(by_filt['f150w']) == 2
    assert len(by_filt['f200w']) == 1


def test_xmatch_common_close_sources():
    c1 = SkyCoord([150.0, 150.01], [2.0, 2.0], unit='deg')
    c2 = SkyCoord([150.0, 151.0], [2.0, 2.0], unit='deg')
    matched = helpers.xmatch_common(c1, c2, dist_limit=5.0)
    assert len(matched) >= 1
    assert matched['d2d'].max() < 5.0


def test_input_list_builds_obstable(tmp_path: Path):
    a = _write_jwst_cal(tmp_path / 'a_nrca1_cal.fits', visit_id='100', exptime=50.0)
    b = _write_jwst_cal(
        tmp_path / 'b_nrca1_cal.fits',
        visit_id='100',
        exptime=75.0,
        filt='F200W',
    )
    # Need S_REGION for edit_visits_groups → get_sky_pgons
    for path in (a, b):
        with fits.open(path, mode='update') as hdul:
            wcs_hdr = hdul['SCI'].header
            from astropy.wcs import WCS

            w = WCS(wcs_hdr)
            ny, nx = hdul['SCI'].data.shape
            corners = np.array([[0, 0], [nx - 1, 0], [nx - 1, ny - 1], [0, ny - 1]], float)
            ra, dec = w.pixel_to_world_values(corners[:, 0], corners[:, 1])
            verts = ' '.join(f'{r:.9f} {d:.9f}' for r, d in zip(ra, dec))
            hdul['SCI'].header['S_REGION'] = f'POLYGON ICRS {verts}'

    obstable = helpers.input_list([str(a), str(b)])
    assert len(obstable) == 2
    assert 'visit' in obstable.colnames
    assert 'group' in obstable.colnames
    assert set(obstable['filter']) == {'f150w', 'f200w'}


def test_package_reexports():
    assert callable(utils_pkg.parse_coord)
    assert callable(utils_pkg.input_list)
    assert callable(utils_pkg.create_symlink)
    assert isinstance(utils_pkg.strict_jwst_params, dict)
    assert 'F606W' in utils_pkg.acceptable_filters
    # Expanded multi-mission filter catalog lives in settings
    assert 'F200W' in utils_pkg.acceptable_filters  # JWST/NIRCam
    assert 'F770W' in utils_pkg.acceptable_filters  # JWST/MIRI
    assert 'F062' in utils_pkg.acceptable_filters  # Roman/WFI
    assert 'VIS' in utils_pkg.acceptable_filters  # Euclid
    assert utils_pkg.FILTERS_BY_INSTRUMENT['NIRCAM']['telescope'] == 'JWST'
    assert utils_pkg.FILTERS_BY_INSTRUMENT['WFI']['telescope'] == 'Roman'
    assert 'f606w' in utils_pkg.BEST_REFERENCE_FILTERS


def test_pick_deepest_images_prefers_best_filter(tmp_path: Path):
    """Prefer a ``BEST_REFERENCE_FILTERS`` band over a non-preferred band."""
    preferred = _write_jwst_cal(
        tmp_path / 'a_nrca1_cal.fits',
        filt='F606W',
        exptime=100.0,
    )
    other = _write_jwst_cal(
        tmp_path / 'b_nrca1_cal.fits',
        filt='F200W',  # not in BEST_REFERENCE_FILTERS
        exptime=1000.0,
    )
    chosen = helpers.pick_deepest_images([str(preferred), str(other)])
    assert chosen == [str(preferred)]


def test_pick_deepest_images_reffilter_override(tmp_path: Path):
    a = _write_jwst_cal(tmp_path / 'a_nrca1_cal.fits', filt='F606W', exptime=50.0)
    b = _write_jwst_cal(tmp_path / 'b_nrca1_cal.fits', filt='F200W', exptime=500.0)
    chosen = helpers.pick_deepest_images(
        [str(a), str(b)],
        reffilter='F200W',
    )
    assert chosen == [str(b)]


def test_organize_visit_tables_byvisit():
    table = Table(
        {
            'visit': [1, 1, 2],
            'image': ['a.fits', 'b.fits', 'c.fits'],
        }
    )
    single = helpers.organize_visit_tables(table, byvisit=False)
    assert len(single) == 1
    assert len(single[0]) == 3

    split = helpers.organize_visit_tables(table, byvisit=True)
    assert len(split) == 2
    assert sorted(len(t) for t in split) == [1, 2]
