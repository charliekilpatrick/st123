"""Tests for abs-ref depth scoring, chip-refine completion, and visit retie."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from st123.stages.alignment.hst_jhat import (
    refine_hst_wcs_per_chip_from_refcat,
    validate_hst_multi_sci_chip_refine,
)
from st123.stages.mosaic.hst_drizzle import _pick_coadd_abs_ref


def _touch_coadd(path: Path, size_bytes: int = 600_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'\0' * int(size_bytes))
    return path


def test_pick_coadd_abs_ref_demotes_thin_f814w(tmp_path: Path):
    f814 = _touch_coadd(tmp_path / 'coadd_1_0_wfc3_f814w_drc.fits')
    f555 = _touch_coadd(tmp_path / 'coadd_1_0_wfc3_f555w_drc.fits', size_bytes=700_000)
    chosen = _pick_coadd_abs_ref(
        [f814, f555],
        n_frames={f814: 2, f555: 4},
    )
    assert chosen == f555.resolve()


def test_pick_coadd_abs_ref_keeps_deep_f814w(tmp_path: Path):
    f814 = _touch_coadd(tmp_path / 'coadd_0_0_wfc3_f814w_drc.fits')
    f555 = _touch_coadd(tmp_path / 'coadd_0_0_wfc3_f555w_drc.fits')
    chosen = _pick_coadd_abs_ref(
        [f814, f555],
        n_frames={f814: 4, f555: 4},
    )
    assert chosen == f814.resolve()


def _dual_sci_frame(
    path: Path,
    *,
    star_on_chip2_only: bool = True,
    crpix_chip2: tuple[float, float] = (50.0, 50.0),
) -> Path:
    """Two-SCI MEF with an optional bright star only on SCI chip 2."""
    rng = np.random.default_rng(0)
    ny = nx = 100

    def _wcs_hdr(crval1: float, crval2: float, crpix1: float, crpix2: float):
        h = fits.Header()
        h['CCDCHIP'] = 0
        h['CTYPE1'] = 'RA---TAN'
        h['CTYPE2'] = 'DEC--TAN'
        h['CRPIX1'] = crpix1
        h['CRPIX2'] = crpix2
        h['CRVAL1'] = crval1
        h['CRVAL2'] = crval2
        h['CD1_1'] = -1.0e-5
        h['CD1_2'] = 0.0
        h['CD2_1'] = 0.0
        h['CD2_2'] = 1.0e-5
        h['CUNIT1'] = 'deg'
        h['CUNIT2'] = 'deg'
        return h

    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = 'WFC3'
    primary.header['FILTER'] = 'F814W'
    primary.header['ROOTNAME'] = 'ieec43abq'

    # Chip1 (SCI index 1): noise only when star_on_chip2_only.
    d1 = rng.normal(0.0, 0.2, size=(ny, nx)).astype(np.float32)
    h1 = _wcs_hdr(196.29, -49.52, 50.0, 50.0)
    h1['CCDCHIP'] = 1

    # Chip2: noise + bright Gaussian so centroid refine succeeds.
    d2 = rng.normal(0.0, 0.2, size=(ny, nx)).astype(np.float32)
    yy, xx = np.mgrid[0:ny, 0:nx]
    d2 += (500.0 * np.exp(-((xx - 55) ** 2 + (yy - 52) ** 2) / (2.0 * 1.5**2))).astype(
        np.float32
    )
    h2 = _wcs_hdr(196.30, -49.54, crpix_chip2[0], crpix_chip2[1])
    h2['CCDCHIP'] = 2

    if not star_on_chip2_only:
        d1 += (
            500.0 * np.exp(-((xx - 55) ** 2 + (yy - 52) ** 2) / (2.0 * 1.5**2))
        ).astype(np.float32)

    fits.HDUList(
        [
            primary,
            fits.ImageHDU(d2, header=h2, name='SCI'),
            fits.ImageHDU(d1, header=h1, name='SCI'),
        ]
    ).writeto(path, overwrite=True)
    return path


def test_validate_hst_multi_sci_chip_refine_fails_partial(tmp_path: Path):
    path = _dual_sci_frame(tmp_path / 'partial_jhat.fits')
    with fits.open(path, mode='update') as hdul:
        hdul[0].header['ST123CHP'] = True
        hdul[0].header['ST123CNU'] = 1
        hdul.flush()
    qa = validate_hst_multi_sci_chip_refine([path])
    assert qa['ok'] is False
    assert qa['n_failed'] == 1


def test_heal_hst_partial_chip_refine_copies_sibling(tmp_path: Path):
    """Recover 1/2 UVIS refine by copying SCI-ERR CRPIX delta onto the sibling."""
    from st123.stages.alignment.hst_jhat import heal_hst_partial_chip_refine

    path = tmp_path / 'iejn01gsq_jhat.fits'
    d = np.zeros((100, 100), dtype=np.float32)

    def _hdr(crpix1, crpix2, *, chip: int, comment: str | None = None):
        h = fits.Header()
        h['CCDCHIP'] = chip
        h['CTYPE1'] = 'RA---TAN'
        h['CTYPE2'] = 'DEC--TAN'
        h['CRPIX1'] = crpix1
        h['CRPIX2'] = crpix2
        if comment:
            h.comments['CRPIX1'] = comment
            h.comments['CRPIX2'] = comment.replace('dx', 'dy')
        h['CRVAL1'] = 185.7
        h['CRVAL2'] = 15.8
        h['CD1_1'] = -1.0e-5
        h['CD1_2'] = 0.0
        h['CD2_1'] = 0.0
        h['CD2_2'] = 1.0e-5
        return h

    pri = fits.PrimaryHDU()
    pri.header['INSTRUME'] = 'WFC3'
    pri.header['ST123CHP'] = True
    pri.header['ST123CNU'] = 1
    fits.HDUList(
        [
            pri,
            fits.ImageHDU(
                d,
                header=_hdr(
                    2047.33, 1026.56, chip=2, comment='st123: per-SCI refine dx'
                ),
                name='SCI',
                ver=1,
            ),
            fits.ImageHDU(
                d, header=_hdr(2048.0, 1026.0, chip=2), name='ERR', ver=1
            ),
            fits.ImageHDU(
                d, header=_hdr(2048.0, 1026.0, chip=1), name='SCI', ver=2
            ),
            fits.ImageHDU(
                d, header=_hdr(2048.0, 1026.0, chip=1), name='ERR', ver=2
            ),
        ]
    ).writeto(path, overwrite=True)

    assert validate_hst_multi_sci_chip_refine([path])['ok'] is False
    heal = heal_hst_partial_chip_refine([path])
    assert heal['n_healed'] == 1
    assert validate_hst_multi_sci_chip_refine([path])['ok'] is True
    with fits.open(path) as hdul:
        assert int(hdul[0].header['ST123CNU']) == 2
        assert hdul[0].header.get('ST123CAF') is True
        assert abs(float(hdul[3].header['CRPIX1']) - (2048.0 - 0.67)) < 0.02
        assert 'sibling heal' in str(hdul[3].header.comments['CRPIX1'])


def test_validate_hst_multi_sci_chip_refine_ok_complete(tmp_path: Path):
    path = _dual_sci_frame(tmp_path / 'complete_jhat.fits')
    with fits.open(path, mode='update') as hdul:
        hdul[0].header['ST123CHP'] = True
        hdul[0].header['ST123CNU'] = 2
        hdul[0].header['ST123CAF'] = True
        hdul.flush()
    qa = validate_hst_multi_sci_chip_refine([path])
    assert qa['ok'] is True


def test_per_chip_refine_copies_sibling_shift(tmp_path: Path):
    """When only one SCI matches the refcat, copy CRPIX Delta onto the other."""
    path = _dual_sci_frame(tmp_path / 'ieec43abq_jhat.fits')
    # Sky position of the chip2 star at pixel (55, 52) with CRPIX=(50,50):
    # Deltapix = (+5, +2) -> CRVAL at that pixel ~ CRVAL + CD*(pix-CRPIX).
    refcat = tmp_path / 'ref.phot.txt'
    # Place several catalog stars near the chip2 detection so matches >= min.
    lines = ['ra dec mag\n']
    # Chip2 CRVAL=(196.30, -49.54), CD1_1=-1e-5, CD2_2=1e-5, star at (55,52)
    # world ~ (196.30 + (-1e-5)*(55-50), -49.54 + (1e-5)*(52-50))
    ra0 = 196.30 + (-1.0e-5) * 5.0
    dec0 = -49.54 + (1.0e-5) * 2.0
    for i in range(12):
        lines.append(f'{ra0 + i * 1e-7:.8f} {dec0 + i * 1e-7:.8f} {18.0 + 0.01 * i}\n')
    refcat.write_text(''.join(lines))

    with fits.open(path) as hdul:
        crpix1_before = (
            float(hdul[1].header['CRPIX1']),
            float(hdul[1].header['CRPIX2']),
        )
        crpix2_before = (
            float(hdul[2].header['CRPIX1']),
            float(hdul[2].header['CRPIX2']),
        )

    stats = refine_hst_wcs_per_chip_from_refcat(
        path, refcat, min_matches=5, search_radius_arcsec=1.0
    )
    assert stats['n_sci'] == 2
    assert stats['n_updated'] == 2
    assert stats.get('n_copied', 0) >= 1
    assert stats.get('complete') is True

    with fits.open(path) as hdul:
        assert hdul[0].header.get('ST123CAF') is True
        assert int(hdul[0].header.get('ST123CNU')) == 2
        # Chip that was refined / copied should differ from the empty-chip default
        # on at least one SCI (the star chip moves CRPIX toward the detection).
        after = [
            (float(hdul[1].header['CRPIX1']), float(hdul[1].header['CRPIX2'])),
            (float(hdul[2].header['CRPIX1']), float(hdul[2].header['CRPIX2'])),
        ]
    assert after[0] != crpix1_before or after[1] != crpix2_before


def test_visit_cross_filter_applies_to_all_frames(tmp_path: Path):
    from st123.stages.alignment.hst_jhat import harmonize_hst_visits_across_filters

    f814 = _dual_sci_frame(tmp_path / 'ieec43abq_jhat.fits')
    f555 = _dual_sci_frame(
        tmp_path / 'ieec43acq_jhat.fits', star_on_chip2_only=False
    )
    with fits.open(f555, mode='update') as hdul:
        hdul[0].header['FILTER'] = 'F555W'
        hdul[0].header['EXPTIME'] = 500.0
        # Offset F555W so relative step has something to apply when mocked.
        for i in (1, 2):
            hdul[i].header['CRVAL1'] = float(hdul[i].header['CRVAL1']) + 0.02 / 3600.0
        hdul.flush()
    with fits.open(f814, mode='update') as hdul:
        hdul[0].header['EXPTIME'] = 100.0
        hdul.flush()

    fake_off = {
        'ok': True,
        'dra_deg': 0.02 / 3600.0,
        'ddec_deg': 0.0,
        'dra_arcsec': 0.02,
        'ddec_arcsec': 0.0,
        'abs_arcsec': 0.02,
    }
    with (
        patch(
            'st123.stages.alignment.hst_jhat.measure_hst_sky_offset_2dhist',
            return_value=fake_off,
        ),
        patch(
            'st123.stages.alignment.hst_jhat.apply_common_abs_shift_vs_ref',
            return_value={
                'ok': True,
                'applied': True,
                'method': 'gaia_match',
                'n_frames': 2,
            },
        ) as mock_abs,
    ):
        report = harmonize_hst_visits_across_filters([f814, f555], abs_ref=None)

    assert report['n_visits'] == 1
    assert report['visits'][0]['n_frames'] == 2
    mock_abs.assert_called_once()
    abs_args = mock_abs.call_args[0][0]
    assert {Path(p).name for p in abs_args} == {
        'ieec43abq_jhat.fits',
        'ieec43acq_jhat.fits',
    }
