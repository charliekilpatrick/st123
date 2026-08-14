"""Unit tests for mosaic pixel-scale helpers and --filters CLI wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from st123.mosaic.mosaic import (
    MIRI_PIXEL_SCALE,
    NIRCAM_LW_PIXEL_SCALE,
    NIRCAM_SW_PIXEL_SCALE,
    is_nircam_sw_broadband,
    mosaic_pixel_scale_arcsec,
    nircam_sw_filter_code,
    rescale_wcs_to_pixel_scale,
)
from st123.scripts import mosaic as mosaic_script
from st123.scripts.utils.options import parse_filter_list


def test_mosaic_pixel_scale_by_filter_and_instrument():
    assert mosaic_pixel_scale_arcsec('F150W') == NIRCAM_SW_PIXEL_SCALE
    assert mosaic_pixel_scale_arcsec('f200w') == NIRCAM_SW_PIXEL_SCALE
    assert mosaic_pixel_scale_arcsec('F335M') == NIRCAM_LW_PIXEL_SCALE
    assert mosaic_pixel_scale_arcsec('F444W', 'NIRCAM') == NIRCAM_LW_PIXEL_SCALE
    assert mosaic_pixel_scale_arcsec('F560W') == MIRI_PIXEL_SCALE
    assert mosaic_pixel_scale_arcsec('F1000W', 'MIRI') == MIRI_PIXEL_SCALE


def test_nircam_sw_broadband_excludes_miri_and_narrowbands():
    """Regression: F1130W must use code 1130, not truncated 113."""
    assert nircam_sw_filter_code('f1130w') == 1130
    assert nircam_sw_filter_code('F150W2') == 150
    assert is_nircam_sw_broadband('f150w2')
    assert is_nircam_sw_broadband('f200w')
    assert not is_nircam_sw_broadband('f1130w')
    assert not is_nircam_sw_broadband('f560w')
    assert not is_nircam_sw_broadband('f444w')
    assert not is_nircam_sw_broadband('f187n')


def test_parse_filter_list_comma_separated():
    assert parse_filter_list('F150W, F444W,F560W') == ['F150W', 'F444W', 'F560W']
    assert parse_filter_list(None) is None
    assert parse_filter_list('') is None


def _tan_wcs(naxis: int = 200, scale_arcsec: float = 0.031) -> WCS:
    scale_deg = scale_arcsec / 3600.0
    hdr = fits.Header(
        {
            'CRPIX1': naxis / 2.0,
            'CRPIX2': naxis / 2.0,
            'CRVAL1': 159.7,
            'CRVAL2': 53.5,
            'CDELT1': -scale_deg,
            'CDELT2': scale_deg,
            'PC1_1': 1.0,
            'PC1_2': 0.0,
            'PC2_1': 0.0,
            'PC2_2': 1.0,
            'CTYPE1': 'RA---TAN',
            'CTYPE2': 'DEC--TAN',
            'CUNIT1': 'deg',
            'CUNIT2': 'deg',
            'NAXIS': 2,
            'NAXIS1': naxis,
            'NAXIS2': naxis,
        }
    )
    w = WCS(hdr)
    w.pixel_shape = (naxis, naxis)
    return w


def test_rescale_wcs_preserves_crval_and_changes_scale():
    w = _tan_wcs(naxis=200, scale_arcsec=0.031)
    hdr = rescale_wcs_to_pixel_scale(w, MIRI_PIXEL_SCALE)
    assert hdr['CRVAL1'] == pytest.approx(159.7)
    assert hdr['CRVAL2'] == pytest.approx(53.5)
    assert 'PC1_1' in hdr and 'CDELT1' in hdr
    new = WCS(hdr)
    new.pixel_shape = (int(hdr['NAXIS1']), int(hdr['NAXIS2']))
    new_scale = float(new.proj_plane_pixel_scales()[0].to(u.arcsec).value)
    assert new_scale == pytest.approx(MIRI_PIXEL_SCALE, rel=0.02)
    # Coarser pixels → fewer pixels for the same sky footprint.
    assert int(hdr['NAXIS1']) < 200
    assert int(hdr['NAXIS2']) < 200


def test_mosaic_parser_accepts_filters():
    parser = mosaic_script.create_parser()
    args = parser.parse_args(
        ['--base-dir', '.', '--filters', 'F150W,F560W', '--ncores', '2']
    )
    assert args.filters == 'F150W,F560W'
    assert args.ncores == 2


def test_forced_filter_tables_skips_missing(caplog):
    import logging

    reftable = Table(
        {
            'image': ['a.fits', 'b.fits'],
            'filter': ['f150w', 'f150w'],
            'instrument': ['NIRCAM', 'NIRCAM'],
        }
    )
    with caplog.at_level(logging.WARNING):
        out = mosaic_script._forced_filter_tables(reftable, ['f150w', 'f560w'])
    assert list(out.keys()) == ['f150w']
    assert len(out['f150w']) == 2
    assert 'f560w' in caplog.text.lower() or 'F560W' in caplog.text or 'f560w' in caplog.text


def _jwst_plan_fixture(tmp_path: Path):
    """Minimal project + plan for JWST mosaic dispatch tests."""
    from st123.mosaic.mosaic import MosaicBox, MosaicPlan

    project = tmp_path / 'NGC3310'
    reduction = project / 'reduction'
    jhat = reduction / 'jhat'
    box_outdir = reduction / 'reference' / 'group_0' / 'ref_0'
    jhat.mkdir(parents=True)
    box_outdir.mkdir(parents=True)
    (project / 'JWST').mkdir()
    fake = jhat / 'jw_test_nrcb1_jhat.fits'
    fake.write_text('x')

    table = Table(
        {
            'image': [str(fake)],
            'exptime': [100.0],
            'datetime': ['2026-01-01T00:00:00'],
            'filter': ['f150w'],
            'instrument': ['NIRCAM'],
            'module': ['b'],
            'zeropoint': [25.0],
            'detector': ['nrcb1'],
            'chip': [1],
            'visit': [1],
            'group': [0],
        }
    )
    plan = MosaicPlan(
        base_dir=reduction,
        reference_dir=reduction / 'reference',
        table=table,
        boxes=[
            MosaicBox(
                group_id=0,
                box_id=0,
                outdir=box_outdir,
                bbox=MagicMock(),
                frames=[str(fake)],
                wcs=_tan_wcs(),
            )
        ],
    )
    plan.boxes[0].bbox.exterior.xy = (
        np.array([0.0, 10.0, 10.0, 0.0]),
        np.array([0.0, 0.0, 10.0, 10.0]),
    )
    return project, fake, table, plan


def test_mosaic_main_forced_filters_calls_per_filter_path(tmp_path: Path):
    """With --filters, mosaic uses forced-filter coadds (not SW PSF-match)."""
    from st123.mosaic.mosaic import MosaicBox

    project, fake, table, plan = _jwst_plan_fixture(tmp_path)

    with (
        patch(
            'st123.mosaic.mosaic.plan_mosaic_boxes', return_value=plan
        ),
        patch(
            'st123.scripts.mosaic._collect_mission_jhat',
            return_value=[str(fake)],
        ),
        patch(
            'st123.utils.helpers.input_list', return_value=table
        ),
        patch(
            'st123.scripts.mosaic._run_default_sw_coadd'
        ) as default_path,
        patch(
            'st123.scripts.mosaic._run_forced_filter_coadds'
        ) as forced_path,
        patch.object(
            MosaicBox,
            'frames_for_mission',
            return_value=[str(fake)],
        ),
    ):
        forced_path.return_value = ['coadd.fits']

        rc = mosaic_script.main(
            [
                '--base-dir',
                str(project),
                '--filters',
                'F150W',
                '--ncores',
                '1',
            ]
        )

    assert rc == 0
    forced_path.assert_called_once()
    default_path.assert_not_called()


def test_mosaic_main_default_uses_shared_stamp_per_filter_path(tmp_path: Path):
    """Default mosaic (no --filters) uses shared-stamp per-filter coadds."""
    from st123.mosaic.mosaic import MosaicBox

    project, fake, table, plan = _jwst_plan_fixture(tmp_path)

    with (
        patch(
            'st123.mosaic.mosaic.plan_mosaic_boxes', return_value=plan
        ),
        patch(
            'st123.scripts.mosaic._collect_mission_jhat',
            return_value=[str(fake)],
        ),
        patch(
            'st123.utils.helpers.input_list', return_value=table
        ),
        patch(
            'st123.scripts.mosaic._run_default_sw_coadd'
        ) as default_path,
        patch(
            'st123.scripts.mosaic._run_forced_filter_coadds'
        ) as forced_path,
        patch.object(
            MosaicBox,
            'frames_for_mission',
            return_value=[str(fake)],
        ),
    ):
        forced_path.return_value = ['coadd.fits']

        rc = mosaic_script.main(
            ['--base-dir', str(project), '--ncores', '1']
        )

    assert rc == 0
    forced_path.assert_called_once()
    default_path.assert_not_called()
