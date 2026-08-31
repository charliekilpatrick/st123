"""Tests for HST combine-all-visits default and coverage filtering."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from st123.stages.mosaic.mosaic import (
    filter_frames_covering_point,
    plan_existing_box,
)
from st123.stages.mosaic.hst_drizzle import _prepare_filter_drizzle


def _tan_wcs(ra: float, dec: float, n: int = 32) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [n / 2, n / 2]
    w.wcs.cdelt = [-0.01, 0.01]
    w.pixel_shape = (n, n)
    return w


def _write_jhat(path: Path, ra: float, dec: float, n: int = 32) -> Path:
    w = _tan_wcs(ra, dec, n=n)
    sci = np.ones((n, n), dtype=np.float32)
    hdr = w.to_header()
    hdr['TELESCOP'] = 'HST'
    hdr['INSTRUME'] = 'WFC3'
    hdr['FILTER'] = 'F814W'
    hdr['ROOTNAME'] = path.stem[:9]
    hdr['DATE-OBS'] = '2020-01-01'
    hdr['TIME-OBS'] = '00:00:00'
    hdr['EXPTIME'] = 100.0
    fits.HDUList(
        [
            fits.PrimaryHDU(
                header=fits.Header(
                    {
                        'TELESCOP': 'HST',
                        'INSTRUME': 'WFC3',
                        'FILTER': 'F814W',
                        'DATE-OBS': '2020-01-01',
                        'TIME-OBS': '00:00:00',
                        'EXPTIME': 100.0,
                    }
                )
            ),
            fits.ImageHDU(sci, header=hdr, name='SCI'),
        ]
    ).writeto(path, overwrite=True)
    return path


def test_filter_frames_covering_point(tmp_path: Path):
    on = _write_jhat(tmp_path / 'on_jhat.fits', 180.0, 0.0)
    off = _write_jhat(tmp_path / 'off_jhat.fits', 190.0, 5.0)
    kept = filter_frames_covering_point([on, off], 180.0, 0.0)
    assert kept == [str(on)]


def test_plan_existing_box_require_coverage(tmp_path: Path):
    base = tmp_path / 'reduction'
    box = base / 'reference' / 'group_0' / 'ref_sn'
    box.mkdir(parents=True)
    w = _tan_wcs(185.73, 15.82, n=64)
    fits.PrimaryHDU(header=w.to_header()).writeto(box / 'stamp_wcs.fits')
    on = _write_jhat(tmp_path / 'on_jhat.fits', 185.73, 15.82, n=64)
    off = _write_jhat(tmp_path / 'off_jhat.fits', 186.5, 16.5, n=64)
    # Default: stamp FoV + stamp-center coverage (no explicit ra/dec needed).
    plan = plan_existing_box(
        base,
        box,
        [on, off],
        min_overlap=0.0,
    )
    assert len(plan.boxes) == 1
    frames = [Path(p).name for p in plan.boxes[0].frames]
    assert 'on_jhat.fits' in frames
    assert 'off_jhat.fits' not in frames


def test_plan_existing_box_explicit_coverage_override(tmp_path: Path):
    base = tmp_path / 'reduction'
    box = base / 'reference' / 'group_0' / 'ref_sn'
    box.mkdir(parents=True)
    w = _tan_wcs(185.73, 15.82, n=64)
    fits.PrimaryHDU(header=w.to_header()).writeto(box / 'stamp_wcs.fits')
    on = _write_jhat(tmp_path / 'on_jhat.fits', 185.73, 15.82, n=64)
    off = _write_jhat(tmp_path / 'off_jhat.fits', 186.5, 16.5, n=64)
    plan = plan_existing_box(
        base,
        box,
        [on, off],
        require_coverage_ra=185.73,
        require_coverage_dec=15.82,
        min_overlap=0.0,
    )
    frames = [Path(p).name for p in plan.boxes[0].frames]
    assert 'on_jhat.fits' in frames
    assert 'off_jhat.fits' not in frames


def test_prepare_filter_drizzle_keeps_outliers_by_default(tmp_path: Path):
    """drop_outlier_visits=False retains cross-visit outliers in the stack."""
    frames = [
        _write_jhat(tmp_path / f'iejn01a_jhat.fits', 180.0, 0.0),
        _write_jhat(tmp_path / f'ie9801a_jhat.fits', 180.0, 0.0),
    ]

    fake_harm = {
        'ok': True,
        'method': 'visit_split_harmonize',
        'anchor': str(frames[0]),
        'pre': {'max_abs_arcsec': 0.5},
        'post': {'max_abs_arcsec': 0.5},
    }
    fake_qa = {
        'ok': True,
        'max_abs_arcsec': 0.01,
        'scope': 'per_visit',
        'visits': [
            {'visit': 'iejn01', 'ok': True},
            {'visit': 'ie9801', 'ok': True},
        ],
    }

    def _offset(a, b, **kwargs):
        del a, b, kwargs
        return {'ok': True, 'abs_arcsec': 1.0, 'peak_count': 10}

    with (
        patch(
            'st123.stages.alignment.hst_jhat.harmonize_hst_group_wcs',
            return_value=fake_harm,
        ),
        patch(
            'st123.stages.alignment.hst_jhat.validate_hst_visits_internal_alignment',
            return_value=fake_qa,
        ),
        patch(
            'st123.stages.alignment.hst_jhat.measure_hst_sky_offset_2dhist',
            side_effect=_offset,
        ),
        patch(
            'st123.stages.alignment.hst_jhat.validate_hst_multi_sci_chip_refine',
            return_value={'ok': True, 'n_failed': 0, 'frames': []},
        ),
        patch(
            'st123.stages.alignment.hst_jhat._frame_exptime', return_value=100.0
        ),
        patch(
            'st123.stages.mosaic.hst_drizzle._total_exptime', return_value=100.0
        ),
        patch(
            'st123.stages.alignment.hst_jhat._hst_visit_key',
            side_effect=lambda p: Path(p).name[:6],
        ),
    ):
        rec = _prepare_filter_drizzle(
            'wfc3',
            'f814w',
            frames,
            outdir=tmp_path,
            group_id=0,
            box_id='sn',
            drop_outlier_visits=False,
            assume_aligned=False,
        )
    assert rec['status'] == 'ready'
    assert len(rec['drizzle_imgs']) == 2
    assert rec.get('retained_outlier_visits')


def test_prepare_filter_drizzle_assume_aligned_skips_harmonize(tmp_path: Path):
    """Default mosaic path does not re-harmonize JHAT WCS."""
    frames = [
        _write_jhat(tmp_path / 'iejn01a_jhat.fits', 180.0, 0.0),
        _write_jhat(tmp_path / 'ie9801a_jhat.fits', 180.0, 0.0),
    ]
    with (
        patch(
            'st123.stages.alignment.hst_jhat.harmonize_hst_group_wcs'
        ) as mock_harm,
        patch(
            'st123.stages.alignment.hst_jhat.heal_hst_partial_chip_refine',
            return_value={'n_healed': 0, 'n_failed': 0, 'frames': []},
        ),
        patch(
            'st123.stages.alignment.hst_jhat.validate_hst_multi_sci_chip_refine',
            return_value={'ok': True, 'n_failed': 0, 'frames': []},
        ),
    ):
        rec = _prepare_filter_drizzle(
            'wfc3',
            'f814w',
            frames,
            outdir=tmp_path,
            group_id=0,
            box_id='sn',
        )
    assert rec['status'] == 'ready'
    assert rec['harmonize'].get('skipped') is True
    assert rec['harmonize'].get('reason') == 'assume_aligned'
    mock_harm.assert_not_called()
