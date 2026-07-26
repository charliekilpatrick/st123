"""Tests for the MIRI reference-alignment pipeline helpers and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from astropy.io import fits

from helpers import write_illuminated_fits, write_ref_with_s_region
from st123.alignment import align as align_lib
from st123.alignment.align import (
    FILTER_MAX_REFERENCE_DISPERSION_MAS,
    F770W_CALIBRATOR_SETTINGS,
    AlignWorkerResult,
    SuccessfulAlignment,
    calibrator_settings_for_filter,
    combine_dispersion_mas,
    max_reference_dispersion_mas,
    rank_fallback_parents,
)
from st123.mosaic.image_overlap import (
    BestOverlap,
    MirIFootprint,
    ScienceFootprint,
    compute_cumulative_overlap_fraction,
    compute_overlap,
)
from st123.scripts import align as align_script


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


def test_filter_wavelength_um_and_blue_to_red_sort():
    from st123.alignment.align import filter_wavelength_um, sort_frames_blue_to_red

    assert filter_wavelength_um('F560W') == pytest.approx(5.6)
    assert filter_wavelength_um('F1000W') == pytest.approx(10.0)
    assert filter_wavelength_um('F200W') == pytest.approx(2.0)
    assert filter_wavelength_um('not-a-filter') == float('inf')

    filt_map = {
        '/c_f1000.fits': 'F1000W',
        '/a_f560.fits': 'F560W',
        '/b_f770.fits': 'F770W',
    }
    frames = [
        {'miri_path': '/c_f1000.fits'},
        {'miri_path': '/a_f560.fits'},
        {'miri_path': '/b_f770.fits'},
    ]
    ordered = sort_frames_blue_to_red(frames, filter_from_path=filt_map.get)
    assert [f['miri_path'] for f in ordered] == [
        '/a_f560.fits',
        '/b_f770.fits',
        '/c_f1000.fits',
    ]


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
        'st123.alignment.align.sky_overlap_fraction',
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
    p0 = (
        '/data/x/JWST/MIRI/F560W/144084448/mastDownload/JWST/'
        'jw01783007001_02101_00001_mirimage/jw01783007001_02101_00001_mirimage_cal.fits'
    )
    assert align_lib.filter_name_from_miri_path(p0) == 'F560W'
    p1 = (
        '/data/x/F560W/144084448/mastDownload/JWST/'
        'jw01783007001_02101_00001_mirimage/jw01783007001_02101_00001_mirimage_cal.fits'
    )
    assert align_lib.filter_name_from_miri_path(p1) == 'F560W'
    p2 = (
        '/data/x/F770W_999/mastDownload/JWST/'
        'jw_x_mirimage/jw_x_mirimage_cal.fits'
    )
    assert align_lib.filter_name_from_miri_path(p2) == 'F770W'
    assert align_lib.parse_filters_arg('F560W, F770W') == ['F560W', 'F770W']
    assert align_lib.parse_filters_arg(None) is None


def test_discover_and_filter_miri_images(tmp_path: Path):
    data_dir = tmp_path / 'NGC3310'
    cal = (
        data_dir
        / 'JWST'
        / 'MIRI'
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

    found = align_lib.discover_miri_images(data_dir)
    assert [Path(p).name for p in found] == ['jw_x_mirimage_cal.fits']
    assert align_lib.filter_miri_images(found, ['F560W']) == found
    assert align_lib.filter_miri_images(found, ['F770W']) == []


def test_discover_ref_images(tmp_path: Path):
    data_dir = tmp_path / 'GAL'
    ref = data_dir / 'reference' / 'group_0' / 'ref_0' / 'coadd_0_0_f150w2_i2d.fits'
    ref.parent.mkdir(parents=True)
    write_ref_with_s_region(ref)
    refs = align_lib.discover_ref_images(data_dir)
    assert refs == [str(ref.resolve())]


def test_discover_ref_images_under_reduction(tmp_path: Path):
    data_dir = tmp_path / 'GAL'
    ref = (
        data_dir
        / 'reduction'
        / 'reference'
        / 'group_0'
        / 'ref_0'
        / 'coadd_0_0_f150w2_i2d.fits'
    )
    ref.parent.mkdir(parents=True)
    write_ref_with_s_region(ref)
    refs = align_lib.discover_ref_images(data_dir)
    assert refs == [str(ref.resolve())]


def test_alignment_summary_roundtrip(tmp_path: Path):
    row = align_lib.AlignmentSummaryRow(
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
    align_lib.write_alignment_summary([row], out)
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
    frames = align_lib.find_frame_overlaps(
        [str(sci)],
        [str(ref)],
        MirIFootprint=MirIFootprint,
        BestOverlap=BestOverlap,
        compute_overlap=compute_overlap,
    )
    assert len(frames) == 1
    assert frames[0].best.ref_path == str(ref)
    assert frames[0].union_overlap_fraction > 0.0


def test_align_reference_parser_and_help():
    parser = align_script.create_parser()
    args = parser.parse_args(
        [
            '--data-dir',
            '/tmp/data',
            '--mode',
            'reference',
            '--align-only',
            '--workers',
            '2',
            '--filters',
            'F560W',
        ]
    )
    assert args.mode == 'reference'
    assert args.align_only is True
    assert args.base_dir == '/tmp/data'
    assert args.ncores == 2
    assert args.filters == 'F560W'
    # --continue-on-error was removed; always continue on per-frame failures.
    with pytest.raises(SystemExit):
        parser.parse_args(['--mode', 'reference', '--continue-on-error'])
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(['--help'])
    assert exc.value.code == 0


def test_align_reference_main_missing_overlap_json(tmp_path: Path):
    data_dir = tmp_path / 'empty'
    data_dir.mkdir()
    rc = align_script.main(
        [
            '--data-dir',
            str(data_dir),
            '--mode',
            'reference',
            '--align-only',
            '--workers',
            '1',
        ]
    )
    assert rc == 1


def test_align_reference_main_overlap_only(tmp_path: Path):
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

    rc = align_script.main(
        [
            '--data-dir',
            str(data_dir),
            '--mode',
            'reference',
            '--overlap-only',
            '--workers',
            '1',
        ]
    )
    assert rc == 0
    overlap_json = data_dir / 'overlap' / 'overlap_summary.json'
    assert overlap_json.is_file()
    payload = json.loads(overlap_json.read_text())
    assert payload['frames']


def test_align_from_frames_continues_on_failure(tmp_path: Path):
    """Per-frame FAILURE rows are recorded; processing does not abort."""
    miri = (
        tmp_path
        / 'JWST'
        / 'MIRI'
        / 'F560W'
        / '1'
        / 'mastDownload'
        / 'JWST'
        / 'jw_x_mirimage'
        / 'jw_x_mirimage_cal.fits'
    )
    miri.parent.mkdir(parents=True)
    miri.write_bytes(b'')
    ref = '/fake/ref.fits'
    frames = [
        {
            'miri_path': str(miri),
            'best': {'ref_path': ref, 'overlap_area': {}},
            'overlapping': [{'ref_path': ref, 'overlap_area': {}, 'ref_area': {}}],
            'union_overlap_fraction': 0.5,
        }
    ]
    summary = tmp_path / 'GAL_alignment_summary.txt'

    def fake_run(jobs, _fn, *, workers, label, on_result):
        del workers, label
        for job in jobs:
            on_result(
                AlignWorkerResult(
                    miri_path=job['miri_path'],
                    filter=job['filter'],
                    mode=job.get('mode', 'reference'),
                    ok=False,
                    row={
                        'miri_path': job['miri_path'],
                        'filter': job['filter'],
                        'status': 'FAILURE',
                        'n_calibrators': 'NA',
                        'dispersion_mas': 'NA',
                        'aligned_path': 'NA',
                        'align_mode': 'NA',
                        'original_ref': job.get('best_ref', 'NA'),
                        'aligned_to': 'NA',
                        'ref_overlap_frac': job.get('ref_overlap_frac', 'NA'),
                    },
                    error='synthetic failure',
                )
            )

    with (
        patch.object(align_lib, '_run_jobs_parallel', side_effect=fake_run),
        patch.object(align_lib, '_resolve_repo_root', return_value=tmp_path),
    ):
        n_fail, rows = align_lib.align_from_frames(
            frames,
            run_alignment=MagicMock(),
            nbright=100,
            plot=False,
            verbose=False,
            fallback=False,
            summary_outfile=summary,
            workers=1,
            repo=tmp_path,
        )

    assert n_fail == 1
    assert len(rows) == 1
    assert rows[0].status == 'FAILURE'
    assert summary.is_file()
    assert 'FAILURE' in summary.read_text()


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
    assert alignment.run_nircam_align_job is alignment.run_reference_align_job


def _write_jhat_product(
    outdir: Path,
    cal_path: Path,
    *,
    dispersion_arcsec: float | None,
) -> Path:
    """Write a JHAT product whose stem matches ``find_jhat_product``."""
    jhat = outdir / cal_path.name.replace('_cal.fits', '_jhat.fits')
    hdr = fits.Header()
    hdr['FILTER'] = 'F560W'
    if dispersion_arcsec is not None:
        hdr['JWDISPM'] = dispersion_arcsec
        hdr['JWDISPS'] = 0.01
        hdr['JWNCAL'] = 12
    fits.PrimaryHDU(header=hdr).writeto(jhat, overwrite=True)
    return jhat


def test_visit_and_reference_workers_share_harvest_contract(tmp_path: Path):
    """Both workers SUCCESS only when JHAT headers carry finite dispersion."""
    outdir = tmp_path / 'out'
    outdir.mkdir()
    cal = _write_miri_cal(tmp_path / 'frame_cal.fits')
    repo = str(tmp_path)

    visit_job = {
        'miri_path': str(cal),
        'align_image': str(cal),
        'outdir': str(outdir),
        'gaia': False,
        'photfilename': str(tmp_path / 'ref.phot.txt'),
        'xshift': 0.0,
        'yshift': 0.0,
        'Nbright': 100,
        'sig': 2,
        'filter': 'F560W',
        'mode': 'VISIT',
        'repo': repo,
        'verbose': False,
    }
    ref_job = {
        'miri_path': str(cal),
        'filter': 'F560W',
        'ref_images': ['/fake/ref.fits'],
        'best_ref': '/fake/ref.fits',
        'outdir': str(outdir),
        'repo': repo,
        'nbright': 100,
        'plot': False,
        'verbose': False,
        'cache_dir': None,
        'match_radius_arcsec': 1.0,
        'clip_to_align_footprint': True,
        'refine': False,
        'refine_sigma': 3.0,
        'refine_max_iter': 3,
        'use_filter_calibrators': False,
        'max_nircam_dispersion_mas': 0.0,
        'ref_overlap_frac': 0.5,
        'mode': 'reference',
    }

    with (
        patch.object(align_lib, '_ensure_worker_ready'),
        patch.object(align_lib, 'align_jwst_image'),
        patch.object(align_lib, 'run_alignment'),
    ):
        # No dispersion header → FAILURE for both modes.
        _write_jhat_product(outdir, cal, dispersion_arcsec=None)
        visit_fail = align_lib.run_visit_align_job(visit_job)
        ref_fail = align_lib.run_reference_align_job(ref_job)
        assert visit_fail.ok is False and visit_fail.row['status'] == 'FAILURE'
        assert ref_fail.ok is False and ref_fail.row['status'] == 'FAILURE'

        # Finite JWDISPM → SUCCESS + provenance for both modes.
        _write_jhat_product(outdir, cal, dispersion_arcsec=0.05)
        visit_ok = align_lib.run_visit_align_job(visit_job)
        ref_ok = align_lib.run_reference_align_job(ref_job)
        assert visit_ok.ok and visit_ok.row['status'] == 'SUCCESS'
        assert visit_ok.row['align_mode'] == 'VISIT'
        assert ref_ok.ok and ref_ok.row['status'] == 'SUCCESS'
        assert ref_ok.row['align_mode'] == 'REFERENCE'
        with fits.open(outdir / 'frame_jhat.fits') as hdul:
            assert hdul[0].header['ALGNMODE'] in ('VISIT', 'REFERENCE')


def test_reference_worker_alias():
    assert align_lib.run_nircam_align_job is align_lib.run_reference_align_job
