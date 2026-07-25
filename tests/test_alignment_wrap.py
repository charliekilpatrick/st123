"""Tests for the MIRI alignment_wrap pipeline helpers and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from astropy.io import fits

from helpers import write_illuminated_fits, write_ref_with_s_region
from st123.alignment import alignment_wrap as wrap
from st123.alignment.alignment_fallback import (
    SuccessfulAlignment,
    combine_dispersion_mas,
    rank_fallback_parents,
)
from st123.alignment.calibrators import (
    FILTER_MAX_REFERENCE_DISPERSION_MAS,
    F770W_CALIBRATOR_SETTINGS,
    calibrator_settings_for_filter,
    max_reference_dispersion_mas,
)
from st123.mosaic.image_overlap import (
    BestOverlap,
    MirIFootprint,
    ScienceFootprint,
    compute_cumulative_overlap_fraction,
    compute_overlap,
)
from st123.scripts import alignment_wrap as wrap_script


def _write_miri_cal(path: Path, *, filter_name: str = 'F560W', crval=(150.0, 2.0)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_illuminated_fits(path, crval=crval, include_s_region=True)
    with fits.open(path, mode='update') as hdul:
        hdul[0].header['FILTER'] = filter_name
        hdul[0].header['INSTRUME'] = 'MIRI'
    return path


def test_calibrator_settings_and_quality_hold_thresholds():
    assert calibrator_settings_for_filter('F560W').max_residual_arcsec is None
    f770 = calibrator_settings_for_filter('F770W')
    assert f770.nbright == F770W_CALIBRATOR_SETTINGS.nbright
    assert f770.max_residual_arcsec == pytest.approx(0.08)
    assert max_reference_dispersion_mas('F560W') is None
    assert max_reference_dispersion_mas('F770W') == FILTER_MAX_REFERENCE_DISPERSION_MAS['F770W']
    assert max_reference_dispersion_mas('F9999W') == 70.0


def test_combine_dispersion_and_rank_parents():
    assert combine_dispersion_mas(30.0, 40.0) == pytest.approx(50.0)
    parents = [
        SuccessfulAlignment(
            miri_path='/a_f560.fits',
            jhat_path='/a_jhat.fits',
            filter='F560W',
            wavelength_um=5.6,
            dispersion_mas=10.0,
            relative_dispersion_mas=10.0,
            align_mode='REFERENCE',
            original_ref='/ref.fits',
            aligned_to='/ref.fits',
        ),
        SuccessfulAlignment(
            miri_path='/b_f1000.fits',
            jhat_path='/b_jhat.fits',
            filter='F1000W',
            wavelength_um=10.0,
            dispersion_mas=12.0,
            relative_dispersion_mas=12.0,
            align_mode='REFERENCE',
            original_ref='/ref.fits',
            aligned_to='/ref.fits',
        ),
    ]
    with patch(
        'st123.alignment.alignment_fallback.sky_overlap_fraction',
        side_effect=lambda child, parent: 0.8 if 'f560' in parent else 0.9,
    ):
        ranked = rank_fallback_parents(
            '/child_f770.fits',
            'F770W',
            parents,
            max_parents=5,
        )
    assert ranked
    assert ranked[0][0].filter == 'F560W'


def test_filter_name_from_miri_path_layouts():
    p1 = (
        '/data/x/F560W/144084448/mastDownload/JWST/'
        'jw01783007001_02101_00001_mirimage/jw01783007001_02101_00001_mirimage_cal.fits'
    )
    assert wrap.filter_name_from_miri_path(p1) == 'F560W'
    p2 = (
        '/data/x/F770W_999/mastDownload/JWST/'
        'jw_x_mirimage/jw_x_mirimage_cal.fits'
    )
    assert wrap.filter_name_from_miri_path(p2) == 'F770W'
    assert wrap.parse_filters_arg('F560W, F770W') == ['F560W', 'F770W']
    assert wrap.parse_filters_arg(None) is None


def test_discover_and_filter_miri_images(tmp_path: Path):
    data_dir = tmp_path / 'NGC3310'
    cal = (
        data_dir
        / 'F560W'
        / '123'
        / 'mastDownload'
        / 'JWST'
        / 'jw_x_mirimage'
        / 'jw_x_mirimage_cal.fits'
    )
    _write_miri_cal(cal, filter_name='F560W')
    other = cal.parent / 'jw_x_mirimage_rate.fits'
    other.write_bytes(b'')

    found = wrap.discover_miri_images(data_dir)
    assert [Path(p).name for p in found] == ['jw_x_mirimage_cal.fits']
    assert wrap.filter_miri_images(found, ['F560W']) == found
    assert wrap.filter_miri_images(found, ['F770W']) == []


def test_discover_ref_images(tmp_path: Path):
    data_dir = tmp_path / 'GAL'
    ref = data_dir / 'reference' / 'group_0' / 'ref_0' / 'coadd_0_0_f150w2_i2d.fits'
    ref.parent.mkdir(parents=True)
    write_ref_with_s_region(ref)
    refs = wrap.discover_ref_images(data_dir)
    assert refs == [str(ref.resolve())]


def test_alignment_summary_roundtrip(tmp_path: Path):
    row = wrap.AlignmentSummaryRow(
        miri_path='/data/a_cal.fits',
        filter='F560W',
        status='SUCCESS',
        ref_overlap_frac=0.95,
        n_calibrators=120,
        dispersion_mas=11.2,
        align_mode='REFERENCE',
        aligned_path='/data/alignment_output/a_jhat.fits',
        original_ref='/data/ref.fits',
        aligned_to='/data/ref.fits',
    )
    out = tmp_path / 'sum.txt'
    wrap.write_alignment_summary([row], out)
    text = out.read_text()
    assert 'F560W' in text
    assert 'REFERENCE' in text
    assert '11.200' in text


def test_compute_cumulative_overlap_fraction(tmp_path: Path):
    sci = write_illuminated_fits(tmp_path / 'sci.fits', include_s_region=True)
    ref = write_ref_with_s_region(tmp_path / 'ref.fits')
    footprint = ScienceFootprint.from_fits(str(sci))
    frac = compute_cumulative_overlap_fraction(footprint, [str(ref)])
    assert frac == pytest.approx(1.0, abs=1e-9)
    assert compute_cumulative_overlap_fraction(footprint, []) == 0.0


def test_find_frame_overlaps_builds_union_fraction(tmp_path: Path):
    sci = write_illuminated_fits(tmp_path / 'miri.fits', include_s_region=True)
    with fits.open(sci, mode='update') as hdul:
        hdul[0].header['FILTER'] = 'F560W'
    ref = write_ref_with_s_region(tmp_path / 'ref.fits')
    frames = wrap.find_frame_overlaps(
        [str(sci)],
        [str(ref)],
        MirIFootprint=MirIFootprint,
        BestOverlap=BestOverlap,
        compute_overlap=compute_overlap,
    )
    assert len(frames) == 1
    assert frames[0].best.ref_path == str(ref)
    assert frames[0].union_overlap_fraction > 0.0


def test_alignment_wrap_parser_and_help():
    parser = wrap_script.create_parser()
    args = parser.parse_args(
        [
            '--data-dir',
            '/tmp/data',
            '--align-only',
            '--workers',
            '2',
            '--filters',
            'F560W',
        ]
    )
    assert args.align_only is True
    assert args.workers == 2
    assert args.filters == 'F560W'
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(['--help'])
    assert exc.value.code == 0


def test_alignment_wrap_main_missing_overlap_json(tmp_path: Path):
    data_dir = tmp_path / 'empty'
    data_dir.mkdir()
    rc = wrap_script.main(
        ['--data-dir', str(data_dir), '--align-only', '--workers', '1']
    )
    assert rc == 1


def test_alignment_wrap_main_overlap_only(tmp_path: Path):
    data_dir = tmp_path / 'GAL'
    cal = (
        data_dir
        / 'F560W'
        / '1'
        / 'mastDownload'
        / 'JWST'
        / 'jw_x_mirimage'
        / 'jw_x_mirimage_cal.fits'
    )
    _write_miri_cal(cal)
    ref = data_dir / 'reference' / 'group_0' / 'ref_0' / 'coadd_0_0_f150w2_i2d.fits'
    ref.parent.mkdir(parents=True)
    write_ref_with_s_region(ref)

    rc = wrap_script.main(
        ['--data-dir', str(data_dir), '--overlap-only', '--workers', '1']
    )
    assert rc == 0
    overlap_json = data_dir / 'overlap_output' / 'overlap_summary.json'
    assert overlap_json.is_file()
    payload = json.loads(overlap_json.read_text())
    assert payload['frames']


def test_seed_offsets_only_for_tight_refine():
    """F560W leaves max_residual unset; F770W tight refine enables offset seeding."""
    assert calibrator_settings_for_filter('F560W').max_residual_arcsec is None
    assert calibrator_settings_for_filter('F770W').max_residual_arcsec is not None


def test_best_overlap_miri_path_alias():
    metrics = MagicMock()
    best = BestOverlap(science_path='/a.fits', ref_path='/r.fits', overlap_area=metrics)
    assert best.miri_path == '/a.fits'


def test_alignment_package_exports_drivers():
    from st123 import alignment

    assert callable(alignment.align_jwst_image)
    assert callable(alignment.run_alignment)
    assert callable(alignment.align_from_frames)
    assert callable(alignment.run_overlaps)
