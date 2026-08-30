"""Tests for the MIRI reference-alignment pipeline helpers and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from astropy.io import fits

from helpers import write_illuminated_fits, write_ref_with_s_region
from st123.stages.alignment import align as align_lib
from st123.stages.alignment.align import (
    FILTER_MAX_REFERENCE_DISPERSION_MAS,
    F770W_CALIBRATOR_SETTINGS,
    AlignWorkerResult,
    SuccessfulAlignment,
    calibrator_settings_for_filter,
    combine_dispersion_mas,
    max_reference_dispersion_mas,
    rank_fallback_parents,
)
from st123.stages.mosaic.image_overlap import (
    BestOverlap,
    MirIFootprint,
    ScienceFootprint,
    compute_cumulative_overlap_fraction,
    compute_overlap,
)
from st123.scripts import align as align_script


def _write_miri_cal(
    path: Path,
    *,
    filter_name: str = 'F560W',
    crval=(150.0, 2.0),
    shape: tuple[int, int] = (1024, 1032),
    subarray: str = 'FULL',
    photplam_angstrom: float | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_illuminated_fits(path, shape=shape, crval=crval, include_s_region=True)
    if photplam_angstrom is None:
        token = str(filter_name).upper().split('_', 1)[0]
        digits = ''.join(ch for ch in token[1:] if ch.isdigit()) if token.startswith('F') else ''
        if digits:
            val = float(digits)
            um = val / 100.0 if val >= 100 else val / 10.0
            photplam_angstrom = um * 1.0e4
        else:
            photplam_angstrom = 1.0e4
    with fits.open(path, mode='update') as hdul:
        hdul[0].header['FILTER'] = filter_name
        hdul[0].header['INSTRUME'] = 'MIRI'
        hdul[0].header['DETECTOR'] = 'MIRIMAGE'
        hdul[0].header['SUBARRAY'] = subarray
        hdul[0].header['PHOTPLAM'] = float(photplam_angstrom)
        if 'SCI' in hdul:
            hdul['SCI'].header['PHOTPLAM'] = float(photplam_angstrom)
    return path


def test_calibrator_settings_and_quality_hold_thresholds():
    assert calibrator_settings_for_filter('F560W').max_residual_arcsec is None
    f770 = calibrator_settings_for_filter('F770W')
    assert f770.nbright == F770W_CALIBRATOR_SETTINGS.nbright
    assert f770.max_residual_arcsec == pytest.approx(0.20)
    assert f770.min_calibrators == F770W_CALIBRATOR_SETTINGS.min_calibrators
    assert f770.min_calibrators == 40
    assert f770.nbright == 200
    assert max_reference_dispersion_mas('F560W') is None
    assert max_reference_dispersion_mas('F770W') == FILTER_MAX_REFERENCE_DISPERSION_MAS['F770W']
    assert max_reference_dispersion_mas('F9999W') == 70.0


def test_assess_field_brightness_flags_bright_sci(tmp_path: Path):
    import numpy as np
    from st123.stages.alignment.align import assess_field_brightness
    from st123.utils.settings import CROWDED_JHAT_NBRIGHT, crowded_jwst_params

    quiet = tmp_path / 'quiet_cal.fits'
    bright = tmp_path / 'bright_cal.fits'
    for path, level in ((quiet, 1.0), (bright, 80.0)):
        data = np.full((128, 128), level, dtype=np.float32)
        fits.HDUList(
            [fits.PrimaryHDU(), fits.ImageHDU(data=data, name='SCI')]
        ).writeto(path, overwrite=True)

    quiet_stats = assess_field_brightness(str(quiet))
    bright_stats = assess_field_brightness(str(bright))
    assert quiet_stats['bright'] is False
    assert bright_stats['bright'] is True
    assert bright_stats['median'] == pytest.approx(80.0)
    assert crowded_jwst_params['SNR_min'] >= 8
    assert crowded_jwst_params['objmag_lim'][1] <= 20
    assert CROWDED_JHAT_NBRIGHT == 100


def test_align_jwst_image_retries_crowded_before_relaxed(tmp_path: Path):
    """Poor strict residual must try crowded/bright cuts before relaxing."""
    import numpy as np
    from st123.utils.settings import CROWDED_JHAT_NBRIGHT, crowded_jwst_params

    cal = tmp_path / 'frame_cal.fits'
    data = np.full((64, 64), 2.0, dtype=np.float32)
    hdr = fits.Header({'CDELT1': 0.062 / 3600.0, 'CDELT2': 0.062 / 3600.0})
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(data=data, header=hdr, name='SCI')]
    ).writeto(cal, overwrite=True)
    outdir = tmp_path / 'out'
    outdir.mkdir()
    phot = tmp_path / 'ref.phot.txt'
    phot.write_text('ra dec mag dmag\n150.0 2.0 18.0 0.01\n')

    kinds: list[str] = []
    nbrights: list[int] = []

    def fake_run_jhat(*, params, Nbright, **_kwargs):
        if params.get('SNR_min') == crowded_jwst_params['SNR_min']:
            kinds.append('crowded')
        elif params.get('d2d_max', 0) >= 2.0:
            kinds.append('relaxed')
        else:
            kinds.append('strict')
        nbrights.append(int(Nbright))

    # Strict and crowded leave residual high; relaxed succeeds.
    disp_seq = iter(
        [
            (0.2, 0.2, 0.2, 0.2),  # strict (~3 px)
            (0.15, 0.15, 0.15, 0.15),  # crowded still high
            (0.01, 0.01, 0.01, 0.01),  # relaxed ok
        ]
    )

    with (
        patch.object(align_lib, 'run_jhat', side_effect=fake_run_jhat),
        patch.object(
            align_lib,
            'jwst_dispersion',
            side_effect=lambda **_k: next(disp_seq),
        ),
        patch.object(align_lib, 'assess_field_brightness', return_value={'bright': False}),
    ):
        align_lib.align_jwst_image(
            str(cal),
            str(outdir),
            gaia=False,
            photfilename=str(phot),
            Nbright=800,
            soft_fail=False,
        )

    assert kinds[:3] == ['strict', 'crowded', 'relaxed']
    assert nbrights[1] == CROWDED_JHAT_NBRIGHT


def test_align_jwst_image_uses_crowded_retry_when_bright_and_strict_fails(tmp_path: Path):
    """Bright SCI still starts strict; crowded is only a retry after failure."""
    import numpy as np
    from st123.utils.settings import crowded_jwst_params

    cal = tmp_path / 'bright_cal.fits'
    data = np.full((64, 64), 100.0, dtype=np.float32)
    hdr = fits.Header({'CDELT1': 0.062 / 3600.0, 'CDELT2': 0.062 / 3600.0})
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(data=data, header=hdr, name='SCI')]
    ).writeto(cal, overwrite=True)
    outdir = tmp_path / 'out'
    outdir.mkdir()
    phot = tmp_path / 'ref.phot.txt'
    phot.write_text('ra dec mag dmag\n150.0 2.0 18.0 0.01\n')

    kinds: list[str] = []

    def fake_run_jhat(*, params, **_kwargs):
        if params.get('SNR_min') == crowded_jwst_params['SNR_min']:
            kinds.append('crowded')
        elif params.get('d2d_max', 0) >= 2.0:
            kinds.append('relaxed')
        else:
            kinds.append('strict')

    disp_seq = iter(
        [
            (0.2, 0.2, 0.2, 0.2),  # strict fails
            (0.01, 0.01, 0.01, 0.01),  # crowded succeeds
        ]
    )

    with (
        patch.object(align_lib, 'run_jhat', side_effect=fake_run_jhat),
        patch.object(
            align_lib,
            'jwst_dispersion',
            side_effect=lambda **_k: next(disp_seq),
        ),
        patch.object(
            align_lib,
            'assess_field_brightness',
            return_value={
                'bright': True,
                'median': 100.0,
                'p99': 100.0,
                'hot_frac': 1.0,
            },
        ),
    ):
        align_lib.align_jwst_image(
            str(cal),
            str(outdir),
            gaia=False,
            photfilename=str(phot),
            Nbright=800,
            soft_fail=False,
        )

    assert kinds == ['strict', 'crowded']


def test_filter_wavelength_um_and_blue_to_red_sort(tmp_path: Path):
    from st123.stages.alignment.align import filter_wavelength_um, sort_frames_blue_to_red

    f560 = _write_miri_cal(tmp_path / 'a_f560.fits', filter_name='F560W')
    f770 = _write_miri_cal(tmp_path / 'b_f770.fits', filter_name='F770W')
    f1000 = _write_miri_cal(tmp_path / 'c_f1000.fits', filter_name='F1000W')
    f200 = tmp_path / 'nircam_f200.fits'
    hdr = fits.Header()
    hdr['TELESCOP'] = 'JWST'
    hdr['INSTRUME'] = 'NIRCAM'
    hdr['FILTER'] = 'F200W'
    hdr['PHOTPLAM'] = 19900.0
    fits.PrimaryHDU(header=hdr).writeto(f200)

    assert filter_wavelength_um(f560) == pytest.approx(5.6)
    assert filter_wavelength_um(f1000) == pytest.approx(10.0)
    assert filter_wavelength_um(f200) == pytest.approx(1.99)
    assert filter_wavelength_um(tmp_path / 'missing.fits') == float('inf')

    frames = [
        {'miri_path': str(f1000)},
        {'miri_path': str(f560)},
        {'miri_path': str(f770)},
    ]
    ordered = sort_frames_blue_to_red(frames)
    assert [f['miri_path'] for f in ordered] == [
        str(f560),
        str(f770),
        str(f1000),
    ]


def test_combine_dispersion_and_rank_parents(tmp_path: Path):
    assert combine_dispersion_mas(30.0, 40.0) == pytest.approx(50.0)
    child = _write_miri_cal(tmp_path / 'child_f770.fits', filter_name='F770W')
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
        'st123.stages.alignment.align.sky_overlap_fraction',
        side_effect=lambda child_path, parent: 0.8 if 'f560' in parent else 0.9,
    ):
        ranked = rank_fallback_parents(
            str(child),
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
    p_flat = (
        '/data/x/download/JWST/MIRI/F560W/144084448/'
        'jw01783007001_02101_00001_mirimage_cal.fits'
    )
    assert align_lib.filter_name_from_miri_path(p_flat) == 'F560W'
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
        / 'download'
        / 'JWST'
        / 'MIRI'
        / 'F560W'
        / '123'
        / 'jw_x_mirimage_cal.fits'
    )
    _write_miri_cal(cal, filter_name='F560W')
    other = cal.parent / 'jw_x_mirimage_rate.fits'
    other.write_bytes(b'')
    # Unsupported cutout / subarray must not enter alignment.
    bad = cal.parent / 'jw_cutout_mirimage_cal.fits'
    _write_miri_cal(
        bad,
        filter_name='F560W',
        shape=(128, 136),
        subarray='SUB64',
    )

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
    _write_miri_cal(miri, filter_name='F560W')
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
    from st123.stages import alignment

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
        # Above F560W min_calibrators (20) so REFERENCE workers stay SUCCESS.
        hdr['JWNCAL'] = 25
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
        # No dispersion header -> FAILURE for both modes.
        _write_jhat_product(outdir, cal, dispersion_arcsec=None)
        visit_fail = align_lib.run_visit_align_job(visit_job)
        ref_fail = align_lib.run_reference_align_job(ref_job)
        assert visit_fail.ok is False and visit_fail.row['status'] == 'FAILURE'
        assert ref_fail.ok is False and ref_fail.row['status'] == 'FAILURE'

        # Finite JWDISPM -> SUCCESS + provenance for both modes.
        _write_jhat_product(outdir, cal, dispersion_arcsec=0.05)
        visit_ok = align_lib.run_visit_align_job(visit_job)
        ref_ok = align_lib.run_reference_align_job(ref_job)
        assert visit_ok.ok and visit_ok.row['status'] == 'SUCCESS'
        assert visit_ok.row['align_mode'] == 'VISIT'
        assert ref_ok.ok and ref_ok.row['status'] == 'SUCCESS'
        assert ref_ok.row['align_mode'] == 'REFERENCE'
        with fits.open(outdir / 'frame_jhat.fits') as hdul:
            assert hdul[0].header['ALGNMODE'] in ('VISIT', 'REFERENCE')


def test_prefer_nircam_reference_paths(tmp_path: Path):
    nircam = tmp_path / 'coadd_nircam_i2d.fits'
    miri = tmp_path / 'coadd_miri_i2d.fits'
    write_illuminated_fits(nircam, crval=(150.0, 2.0), include_s_region=True)
    write_illuminated_fits(miri, crval=(150.0, 2.0), include_s_region=True)
    with fits.open(nircam, mode='update') as hdul:
        hdul[0].header['INSTRUME'] = 'NIRCAM'
        hdul[0].header['FILTER'] = 'F150W2'
    with fits.open(miri, mode='update') as hdul:
        hdul[0].header['INSTRUME'] = 'MIRI'
        hdul[0].header['FILTER'] = 'F560W'

    preferred = align_lib.prefer_nircam_reference_paths([str(miri), str(nircam)])
    assert preferred == [str(nircam)]
    # No NIRCam -> keep MIRI list unchanged.
    assert align_lib.prefer_nircam_reference_paths([str(miri)]) == [str(miri)]


def test_peer_sky_dispersion_mas(tmp_path: Path):
    from astropy.table import Table
    import numpy as np

    a = tmp_path / 'a.phot.txt'
    b = tmp_path / 'b.phot.txt'
    ra = np.array([150.0, 150.001, 150.002])
    dec = np.array([2.0, 2.001, 2.002])
    # B is shifted by ~200 mas in dec relative to A.
    Table({'ra': ra, 'dec': dec}).write(a, format='ascii', overwrite=True)
    Table({'ra': ra, 'dec': dec + (0.2 / 3600.0)}).write(b, format='ascii', overwrite=True)

    med, n = align_lib.peer_sky_dispersion_mas(a, b, match_radius_arcsec=1.0)
    assert n == 3
    assert med == pytest.approx(200.0, rel=0.05)


def test_flag_peer_inconsistent_demotes_worse_frame(tmp_path: Path):
    from astropy.table import Table
    import numpy as np

    def _make_success(name: str, disp: float, *, dec_shift_mas: float = 0.0):
        miri = tmp_path / f'{name}_cal.fits'
        jhat = tmp_path / f'{name}_jhat.fits'
        phot = tmp_path / f'{name}_jhat_cal.phot.txt'
        # Minimal illuminated FITS with S_REGION so overlap works.
        write_illuminated_fits(miri, crval=(150.0, 2.0), include_s_region=True)
        write_illuminated_fits(jhat, crval=(150.0, 2.0), include_s_region=True)
        with fits.open(jhat, mode='update') as hdul:
            hdul[0].header['FILTER'] = 'F770W'
            hdul[0].header['ALGNMODE'] = 'REFERENCE'
            hdul[0].header['JWDISPM'] = disp / 1000.0
        ra = np.linspace(150.0, 150.01, 40)
        dec = np.linspace(2.0, 2.01, 40) + (dec_shift_mas / 3600.0 / 1000.0)
        Table({'ra': ra, 'dec': dec}).write(phot, format='ascii', overwrite=True)
        row = align_lib.AlignmentSummaryRow(
            miri_path=str(miri),
            filter='F770W',
            status='SUCCESS',
            n_calibrators=50,
            dispersion_mas=disp,
            aligned_path=str(jhat),
            align_mode='REFERENCE',
            original_ref='ref.fits',
            aligned_to='ref.fits',
            ref_overlap_frac=1.0,
        )
        success = SuccessfulAlignment(
            miri_path=str(miri),
            jhat_path=str(jhat),
            filter='F770W',
            wavelength_um=7.7,
            dispersion_mas=disp,
            relative_dispersion_mas=disp,
            align_mode='REFERENCE',
            original_ref='ref.fits',
            aligned_to='ref.fits',
            photfile=str(phot),
        )
        return row, success

    good_row, good_s = _make_success('good', disp=20.0, dec_shift_mas=0.0)
    bad_row, bad_s = _make_success('bad', disp=30.0, dec_shift_mas=300.0)
    rows = [good_row, bad_row]
    row_by_miri = {good_row.miri_path: good_row, bad_row.miri_path: bad_row}
    successes = [good_s, bad_s]

    demoted = align_lib.flag_peer_inconsistent_reference_rows(
        filter_name='F770W',
        row_by_miri=row_by_miri,
        rows=rows,
        successes=successes,
        max_peer_dispersion_mas=100.0,
        min_overlap=0.01,
        min_matches=10,
    )
    assert bad_row.miri_path in demoted
    assert row_by_miri[bad_row.miri_path].status == 'PENDING'
    assert row_by_miri[good_row.miri_path].status == 'SUCCESS'
    assert all(s.miri_path != bad_row.miri_path for s in successes)


def test_calc_dispersion_returns_clipped_match_count(tmp_path: Path):
    from astropy.table import Table
    import numpy as np

    # 10 well-matched stars + 2 outliers beyond the match radius.
    ra = np.linspace(150.0, 150.001, 10)
    dec = np.linspace(2.0, 2.001, 10)
    ref = Table({'ra': ra, 'dec': dec})
    sci = tmp_path / 'sci.phot.txt'
    Table(
        {
            'ra': np.concatenate([ra, ra[:2] + 0.01]),
            'dec': np.concatenate([dec, dec[:2]]),
            'x': np.arange(12, dtype=float),
            'y': np.arange(12, dtype=float),
        }
    ).write(sci, format='ascii', overwrite=True)

    mean, med, std, n_cal = align_lib.calc_dispersion(
        ref, str(sci), dist_limit=0.5, sig=2.0, plot=False
    )
    assert n_cal == 10
    assert np.isfinite(mean) and np.isfinite(med) and np.isfinite(std)


def test_jwncal_plausibility_rejects_master_catalog_pollution():
    assert align_lib.jwncal_is_plausible(0)
    assert align_lib.jwncal_is_plausible(32)
    assert not align_lib.jwncal_is_plausible(32875)
    assert not align_lib.jwncal_is_plausible(-1)


def test_harvest_prefers_dispersion_match_count_not_refcat_rows(tmp_path: Path):
    """JWNCAL must not inherit the full unclipped master-catalog length."""
    from astropy.table import Table
    import numpy as np

    outdir = tmp_path / 'alignment_output'
    outdir.mkdir()
    cal = _write_miri_cal(tmp_path / 'frame_cal.fits')
    jhat = outdir / 'frame_jhat.fits'
    write_illuminated_fits(jhat, crval=(150.0, 2.0), include_s_region=True)

    # Science phot: 8 sources. Ref / JWCAT: huge catalog, but only 8 match.
    ra = np.linspace(150.0, 150.002, 8)
    dec = np.linspace(2.0, 2.002, 8)
    Table({'ra': ra, 'dec': dec, 'x': np.arange(8.0), 'y': np.arange(8.0)}).write(
        outdir / 'frame_jhat_cal.phot.txt', format='ascii', overwrite=True
    )
    # Full "master" dump that used to pollute n_calibrators via *.refcat.txt.
    n_master = 500
    ra_big = np.linspace(150.0, 150.1, n_master)
    dec_big = np.linspace(2.0, 2.1, n_master)
    # First 8 coincide with science; the rest are far away.
    ra_big[:8] = ra
    dec_big[:8] = dec
    Table({'ra': ra_big, 'dec': dec_big}).write(
        outdir / 'coadd_ref.phot.txt', format='ascii', overwrite=True
    )
    Table({'ra': ra_big, 'dec': dec_big}).write(
        outdir / 'frame.refcat.txt', format='ascii', overwrite=True
    )

    with fits.open(jhat, mode='update') as hdul:
        hdul[0].header['FILTER'] = 'F560W'
        hdul[0].header['JWDISPM'] = 0.04
        hdul[0].header['JWDISPS'] = 0.01
        # Polluted count: full master length (old bug).
        hdul[0].header['JWNCAL'] = n_master
        hdul[0].header['JWCAT'] = 'coadd_ref.phot.txt'

    row = align_lib.harvest_alignment_metrics(
        str(cal), outdir, ran_ok=True, default_align_mode='REFERENCE'
    )
    assert row.status == 'SUCCESS'
    assert row.n_calibrators == 8
    with fits.open(jhat) as hdul:
        assert int(hdul[0].header['JWNCAL']) == 8


def test_harvest_keeps_soft_fail_zero_calibrators(tmp_path: Path):
    outdir = tmp_path / 'alignment_output'
    outdir.mkdir()
    cal = _write_miri_cal(tmp_path / 'frame_cal.fits')
    jhat = outdir / 'frame_jhat.fits'
    write_illuminated_fits(jhat, crval=(150.0, 2.0), include_s_region=True)
    with fits.open(jhat, mode='update') as hdul:
        hdul[0].header['FILTER'] = 'F2550W'
        hdul[0].header['JWDISPM'] = 0.045
        hdul[0].header['JWDISPS'] = 0.045
        hdul[0].header['JWNCAL'] = 0

    row = align_lib.harvest_alignment_metrics(
        str(cal), outdir, ran_ok=True, default_align_mode='REFERENCE'
    )
    assert row.status == 'SUCCESS'
    assert row.n_calibrators == 0


def test_soft_fail_and_usable_reference_helpers():
    soft = align_lib.AlignmentSummaryRow(
        miri_path='/a_cal.fits',
        filter='F770W',
        status='PENDING',
        n_calibrators=0,
        dispersion_mas=99990.0,
        aligned_path='/a_jhat.fits',
        align_mode='REFERENCE',
        original_ref='/ref.fits',
        aligned_to='/ref.fits',
        ref_overlap_frac=1.0,
    )
    usable_low_n = align_lib.AlignmentSummaryRow(
        miri_path='/b_cal.fits',
        filter='F770W',
        status='PENDING',
        n_calibrators=12,
        dispersion_mas=55.0,
        aligned_path='/b_jhat.fits',
        align_mode='REFERENCE',
        original_ref='/ref.fits',
        aligned_to='/ref.fits',
        ref_overlap_frac=1.0,
    )
    keepable = align_lib.AlignmentSummaryRow(
        miri_path='/c_cal.fits',
        filter='F770W',
        status='PENDING',
        n_calibrators=45,
        dispersion_mas=55.0,
        aligned_path='/c_jhat.fits',
        align_mode='REFERENCE',
        original_ref='/ref.fits',
        aligned_to='/ref.fits',
        ref_overlap_frac=1.0,
    )
    assert align_lib.is_soft_fail_dispersion(99990.0)
    assert align_lib.is_soft_fail_dispersion(99.99 * 1000.0)
    assert not align_lib.is_soft_fail_dispersion(55.0)
    assert not align_lib.reference_solution_usable(soft)
    assert align_lib.reference_solution_usable(usable_low_n)
    # F770W thr=50; disp=55 with low n_cal is not keepable.
    assert not align_lib.reference_solution_keepable(usable_low_n)
    assert align_lib.reference_solution_keepable(keepable)
    # Sparse-but-tight F2100W (below thr=65) is keepable even with n_cal=3.
    sparse_tight = align_lib.AlignmentSummaryRow(
        miri_path='/d_cal.fits',
        filter='F2100W',
        status='PENDING',
        n_calibrators=3,
        dispersion_mas=32.0,
        aligned_path='/d_jhat.fits',
        align_mode='REFERENCE',
        original_ref='/ref.fits',
        aligned_to='/ref.fits',
        ref_overlap_frac=1.0,
    )
    assert align_lib.reference_solution_keepable(sparse_tight)


def test_rank_fallback_prefers_f770w_seed_and_finalized(tmp_path: Path):
    """Redder frames prefer F770W seeds; skip provisional above-threshold parents."""
    child = _write_miri_cal(tmp_path / 'child_f2100.fits', filter_name='F2100W')
    parents = [
        SuccessfulAlignment(
            miri_path='/prov_f2100.fits',
            jhat_path='/prov_jhat.fits',
            filter='F2100W',
            wavelength_um=21.0,
            dispersion_mas=90.0,
            relative_dispersion_mas=90.0,
            align_mode='REFERENCE',
            original_ref='/ref.fits',
            aligned_to='/ref.fits',
            provisional=True,
        ),
        SuccessfulAlignment(
            miri_path='/f1800.fits',
            jhat_path='/f1800_jhat.fits',
            filter='F1800W',
            wavelength_um=18.0,
            dispersion_mas=35.0,
            relative_dispersion_mas=35.0,
            align_mode='MIRI_REL',
            original_ref='/ref.fits',
            aligned_to='/ref.fits',
            provisional=False,
        ),
        SuccessfulAlignment(
            miri_path='/f770.fits',
            jhat_path='/f770_jhat.fits',
            filter='F770W',
            wavelength_um=7.7,
            dispersion_mas=28.0,
            relative_dispersion_mas=28.0,
            align_mode='REFERENCE',
            original_ref='/ref.fits',
            aligned_to='/ref.fits',
            provisional=False,
        ),
    ]
    with patch(
        'st123.stages.alignment.align.sky_overlap_fraction',
        return_value=0.8,
    ):
        ranked = rank_fallback_parents(
            str(child),
            'F2100W',
            parents,
            max_parents=5,
        )
    assert ranked
    assert ranked[0][0].filter == 'F770W'
    assert all(p.filter != 'F2100W' or not p.provisional for p, _ in ranked)


def test_repropagate_miri_rel_absolutes(tmp_path: Path):
    parent_jhat = tmp_path / 'parent_jhat.fits'
    child_jhat = tmp_path / 'child_jhat.fits'
    fits.PrimaryHDU().writeto(parent_jhat)
    fits.PrimaryHDU().writeto(child_jhat)
    align_lib.write_alignment_provenance(
        str(parent_jhat),
        align_mode='MIRI_REL',
        original_ref='/ref.fits',
        aligned_to='/seed.fits',
        relative_dispersion_mas=15.0,
        absolute_dispersion_mas=30.0,
        n_calibrators=20,
    )
    align_lib.write_alignment_provenance(
        str(child_jhat),
        align_mode='MIRI_REL',
        original_ref='/ref.fits',
        aligned_to=str(parent_jhat),
        relative_dispersion_mas=10.0,
        absolute_dispersion_mas=76.0,
        n_calibrators=80,
    )
    parent = SuccessfulAlignment(
        miri_path='/parent_cal.fits',
        jhat_path=str(parent_jhat),
        filter='F770W',
        wavelength_um=7.7,
        dispersion_mas=30.0,
        relative_dispersion_mas=15.0,
        align_mode='MIRI_REL',
        original_ref='/ref.fits',
        aligned_to='/seed.fits',
    )
    child = SuccessfulAlignment(
        miri_path='/child_cal.fits',
        jhat_path=str(child_jhat),
        filter='F770W',
        wavelength_um=7.7,
        dispersion_mas=76.0,
        relative_dispersion_mas=10.0,
        align_mode='MIRI_REL',
        original_ref='/ref.fits',
        aligned_to=str(parent_jhat),
    )
    row = align_lib.AlignmentSummaryRow(
        miri_path='/child_cal.fits',
        filter='F770W',
        status='SUCCESS',
        n_calibrators=80,
        dispersion_mas=76.0,
        aligned_path=str(child_jhat),
        align_mode='MIRI_REL',
        original_ref='/ref.fits',
        aligned_to=str(parent_jhat),
    )
    successes = [parent, child]
    rows = [row]
    row_by = {row.miri_path: row}
    n = align_lib.repropagate_miri_rel_absolutes(successes, row_by, rows)
    assert n == 1
    expected = align_lib.combine_dispersion_mas(30.0, 10.0)
    assert child.dispersion_mas == pytest.approx(expected, abs=0.05)
    assert row.dispersion_mas == pytest.approx(expected, abs=0.05)
    with fits.open(child_jhat) as hdul:
        assert float(hdul[0].header['JWDISPM']) * 1000.0 == pytest.approx(
            expected, abs=0.05
        )


def test_f770w_gate_uses_median_and_skew():
    gate, skew = align_lib.f770w_reference_gate_dispersion_mas(40.0, 35.0)
    assert gate == pytest.approx(40.0)
    assert skew is None
    gate, skew = align_lib.f770w_reference_gate_dispersion_mas(80.0, 40.0)
    assert gate == pytest.approx(80.0)
    assert skew is not None


def test_provisional_parents_from_usable_holds_only(tmp_path: Path):
    good_cal = _write_miri_cal(tmp_path / 'good_cal.fits', filter_name='F770W')
    bad_cal = _write_miri_cal(tmp_path / 'bad_cal.fits', filter_name='F770W')
    good_jhat = tmp_path / 'good_jhat.fits'
    bad_jhat = tmp_path / 'bad_jhat.fits'
    write_illuminated_fits(good_jhat, crval=(150.0, 2.0), include_s_region=True)
    write_illuminated_fits(bad_jhat, crval=(150.0, 2.0), include_s_region=True)

    good_row = align_lib.AlignmentSummaryRow(
        miri_path=str(good_cal),
        filter='F770W',
        status='PENDING',
        n_calibrators=40,
        dispersion_mas=45.0,
        aligned_path=str(good_jhat),
        align_mode='REFERENCE',
        original_ref='/ref.fits',
        aligned_to='/ref.fits',
        ref_overlap_frac=1.0,
    )
    bad_row = align_lib.AlignmentSummaryRow(
        miri_path=str(bad_cal),
        filter='F770W',
        status='PENDING',
        n_calibrators=0,
        dispersion_mas=99990.0,
        aligned_path=str(bad_jhat),
        align_mode='REFERENCE',
        original_ref='/ref.fits',
        aligned_to='/ref.fits',
        ref_overlap_frac=1.0,
    )
    row_by_miri = {
        good_row.miri_path: good_row,
        bad_row.miri_path: bad_row,
    }
    parents = align_lib.provisional_fallback_parents_from_holds(
        'F770W',
        row_by_miri,
        [good_row.miri_path, bad_row.miri_path],
    )
    assert len(parents) == 1
    assert parents[0].miri_path == good_row.miri_path
    assert parents[0].dispersion_mas == pytest.approx(45.0)


def test_quality_hold_no_parent_finalizes_failure(tmp_path: Path):
    """Soft-fail PENDING with no MIRI_REL parent must end as FAILURE."""
    miri = (
        tmp_path
        / 'JWST'
        / 'MIRI'
        / 'F770W'
        / '1'
        / 'mastDownload'
        / 'JWST'
        / 'jw_x_mirimage'
        / 'jw_x_mirimage_cal.fits'
    )
    _write_miri_cal(miri, filter_name='F770W')
    jhat = miri.parent / 'alignment_output' / 'jw_x_mirimage_jhat.fits'
    jhat.parent.mkdir(parents=True)
    write_illuminated_fits(jhat, crval=(150.0, 2.0), include_s_region=True)
    ref = '/fake/ref.fits'
    frames = [
        {
            'miri_path': str(miri),
            'best': {'ref_path': ref, 'overlap_area': {}},
            'overlapping': [{'ref_path': ref, 'overlap_area': {}, 'ref_area': {}}],
            'union_overlap_fraction': 0.5,
        }
    ]
    summary = tmp_path / 'alignment_summary.txt'

    def fake_run(jobs, _fn, *, workers, label, on_result):
        del workers, label
        for job in jobs:
            mode = job.get('mode', 'reference')
            if mode == 'fallback':
                raise AssertionError('fallback should not run without parents')
            on_result(
                AlignWorkerResult(
                    miri_path=job['miri_path'],
                    filter=job['filter'],
                    mode='reference',
                    ok=False,
                    row={
                        'miri_path': job['miri_path'],
                        'filter': job['filter'],
                        'status': 'PENDING',
                        'n_calibrators': 0,
                        'dispersion_mas': 99990.0,
                        'aligned_path': str(jhat),
                        'align_mode': 'REFERENCE',
                        'original_ref': job.get('best_ref', 'NA'),
                        'aligned_to': job.get('best_ref', 'NA'),
                        'ref_overlap_frac': job.get('ref_overlap_frac', 'NA'),
                    },
                    error='REFERENCE soft-failed',
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
            fallback=True,
            summary_outfile=summary,
            workers=1,
            repo=tmp_path,
        )

    assert n_fail == 1
    assert len(rows) == 1
    assert rows[0].status == 'FAILURE'
    assert rows[0].dispersion_mas == pytest.approx(99990.0)
    text = summary.read_text()
    assert 'FAILURE' in text
    # Public summary must not promote the soft-fail hold to SUCCESS.
    assert not any(
        'SUCCESS' in ln and 'jw_x_mirimage_cal.fits' in ln
        for ln in text.splitlines()
    )


def test_usable_hold_kept_after_failed_miri_rel(tmp_path: Path):
    """Keepable REFERENCE hold stays SUCCESS when MIRI_REL does not improve it."""
    # Paths must include filter tokens so blue->red wave order is F560W then F770W.
    parent_cal = _write_miri_cal(
        tmp_path
        / 'JWST'
        / 'MIRI'
        / 'F560W'
        / '1'
        / 'mastDownload'
        / 'JWST'
        / 'jw_parent_mirimage'
        / 'jw_parent_mirimage_cal.fits',
        filter_name='F560W',
    )
    child_cal = _write_miri_cal(
        tmp_path
        / 'JWST'
        / 'MIRI'
        / 'F770W'
        / '1'
        / 'mastDownload'
        / 'JWST'
        / 'jw_child_mirimage'
        / 'jw_child_mirimage_cal.fits',
        filter_name='F770W',
    )
    parent_jhat = parent_cal.parent / 'alignment_output' / 'jw_parent_mirimage_jhat.fits'
    child_jhat = child_cal.parent / 'alignment_output' / 'jw_child_mirimage_jhat.fits'
    parent_jhat.parent.mkdir(parents=True, exist_ok=True)
    child_jhat.parent.mkdir(parents=True, exist_ok=True)
    write_illuminated_fits(parent_jhat, crval=(150.0, 2.0), include_s_region=True)
    write_illuminated_fits(child_jhat, crval=(150.0, 2.0), include_s_region=True)
    ref = '/fake/ref.fits'
    frames = [
        {
            'miri_path': str(parent_cal),
            'best': {'ref_path': ref, 'overlap_area': {}},
            'overlapping': [{'ref_path': ref, 'overlap_area': {}, 'ref_area': {}}],
            'union_overlap_fraction': 0.5,
        },
        {
            'miri_path': str(child_cal),
            'best': {'ref_path': ref, 'overlap_area': {}},
            'overlapping': [{'ref_path': ref, 'overlap_area': {}, 'ref_area': {}}],
            'union_overlap_fraction': 0.5,
        },
    ]

    def fake_run(jobs, _fn, *, workers, label, on_result):
        del workers, label
        for job in jobs:
            mode = job.get('mode', 'reference')
            miri = job['miri_path']
            if mode == 'reference':
                if miri == str(parent_cal):
                    on_result(
                        AlignWorkerResult(
                            miri_path=miri,
                            filter='F560W',
                            mode='reference',
                            ok=True,
                            row={
                                'miri_path': miri,
                                'filter': 'F560W',
                                'status': 'SUCCESS',
                                'n_calibrators': 50,
                                'dispersion_mas': 20.0,
                                'aligned_path': str(parent_jhat),
                                'align_mode': 'REFERENCE',
                                'original_ref': ref,
                                'aligned_to': ref,
                                'ref_overlap_frac': 0.5,
                            },
                            success={
                                'miri_path': miri,
                                'jhat_path': str(parent_jhat),
                                'filter': 'F560W',
                                'wavelength_um': 5.6,
                                'dispersion_mas': 20.0,
                                'relative_dispersion_mas': 20.0,
                                'align_mode': 'REFERENCE',
                                'original_ref': ref,
                                'aligned_to': ref,
                                'photfile': None,
                                'provisional': False,
                            },
                        )
                    )
                else:
                    on_result(
                        AlignWorkerResult(
                            miri_path=miri,
                            filter='F770W',
                            mode='reference',
                            ok=False,
                            row={
                                'miri_path': miri,
                                'filter': 'F770W',
                                'status': 'PENDING',
                                # Meet F770W min_calibrators=40 so keepable.
                                'n_calibrators': 45,
                                'dispersion_mas': 80.0,
                                'aligned_path': str(child_jhat),
                                'align_mode': 'REFERENCE',
                                'original_ref': ref,
                                'aligned_to': ref,
                                'ref_overlap_frac': 0.5,
                            },
                            error='over threshold; try MIRI_REL',
                        )
                    )
            else:
                on_result(
                    AlignWorkerResult(
                        miri_path=miri,
                        filter=job['filter'],
                        mode='fallback',
                        ok=False,
                        row={
                            'miri_path': miri,
                            'filter': job['filter'],
                            'status': 'FAILURE',
                            'n_calibrators': 'NA',
                            'dispersion_mas': 'NA',
                            'aligned_path': 'NA',
                            'align_mode': 'NA',
                            'original_ref': 'NA',
                            'aligned_to': 'NA',
                            'ref_overlap_frac': job.get('ref_overlap_frac', 'NA'),
                        },
                        error='MIRI_REL did not improve',
                    )
                )

    with (
        patch.object(align_lib, '_run_jobs_parallel', side_effect=fake_run),
        patch.object(align_lib, '_resolve_repo_root', return_value=tmp_path),
        patch.object(align_lib, 'sky_overlap_fraction', return_value=0.8),
        patch.object(align_lib, 'flag_peer_inconsistent_reference_rows', return_value=[]),
    ):
        n_fail, rows = align_lib.align_from_frames(
            frames,
            run_alignment=MagicMock(),
            nbright=100,
            plot=False,
            verbose=False,
            fallback=True,
            workers=1,
            repo=tmp_path,
        )

    by_path = {r.miri_path: r for r in rows}
    assert by_path[str(parent_cal)].status == 'SUCCESS'
    assert by_path[str(child_cal)].status == 'SUCCESS'
    assert by_path[str(child_cal)].align_mode == 'REFERENCE'
    assert by_path[str(child_cal)].dispersion_mas == pytest.approx(80.0)
    assert n_fail == 0


def test_tiny_ncal_hold_fails_after_failed_miri_rel(tmp_path: Path):
    """Under-calibrated REFERENCE hold becomes FAILURE when MIRI_REL fails."""
    parent_cal = _write_miri_cal(
        tmp_path
        / 'JWST'
        / 'MIRI'
        / 'F560W'
        / '1'
        / 'mastDownload'
        / 'JWST'
        / 'jw_parent_mirimage'
        / 'jw_parent_mirimage_cal.fits',
        filter_name='F560W',
    )
    child_cal = _write_miri_cal(
        tmp_path
        / 'JWST'
        / 'MIRI'
        / 'F2100W'
        / '1'
        / 'mastDownload'
        / 'JWST'
        / 'jw_child_mirimage'
        / 'jw_child_mirimage_cal.fits',
        filter_name='F2100W',
    )
    parent_jhat = parent_cal.parent / 'alignment_output' / 'jw_parent_mirimage_jhat.fits'
    child_jhat = child_cal.parent / 'alignment_output' / 'jw_child_mirimage_jhat.fits'
    parent_jhat.parent.mkdir(parents=True, exist_ok=True)
    child_jhat.parent.mkdir(parents=True, exist_ok=True)
    write_illuminated_fits(parent_jhat, crval=(150.0, 2.0), include_s_region=True)
    write_illuminated_fits(child_jhat, crval=(150.0, 2.0), include_s_region=True)
    ref = '/fake/ref.fits'
    frames = [
        {
            'miri_path': str(parent_cal),
            'best': {'ref_path': ref, 'overlap_area': {}},
            'overlapping': [{'ref_path': ref, 'overlap_area': {}, 'ref_area': {}}],
            'union_overlap_fraction': 0.5,
        },
        {
            'miri_path': str(child_cal),
            'best': {'ref_path': ref, 'overlap_area': {}},
            'overlapping': [{'ref_path': ref, 'overlap_area': {}, 'ref_area': {}}],
            'union_overlap_fraction': 0.5,
        },
    ]

    def fake_run(jobs, _fn, *, workers, label, on_result):
        del workers, label
        for job in jobs:
            mode = job.get('mode', 'reference')
            miri = job['miri_path']
            if mode == 'reference':
                if miri == str(parent_cal):
                    on_result(
                        AlignWorkerResult(
                            miri_path=miri,
                            filter='F560W',
                            mode='reference',
                            ok=True,
                            row={
                                'miri_path': miri,
                                'filter': 'F560W',
                                'status': 'SUCCESS',
                                'n_calibrators': 50,
                                'dispersion_mas': 20.0,
                                'aligned_path': str(parent_jhat),
                                'align_mode': 'REFERENCE',
                                'original_ref': ref,
                                'aligned_to': ref,
                                'ref_overlap_frac': 0.5,
                            },
                            success={
                                'miri_path': miri,
                                'jhat_path': str(parent_jhat),
                                'filter': 'F560W',
                                'wavelength_um': 5.6,
                                'dispersion_mas': 20.0,
                                'relative_dispersion_mas': 20.0,
                                'align_mode': 'REFERENCE',
                                'original_ref': ref,
                                'aligned_to': ref,
                                'photfile': None,
                                'provisional': False,
                            },
                        )
                    )
                else:
                    on_result(
                        AlignWorkerResult(
                            miri_path=miri,
                            filter='F2100W',
                            mode='reference',
                            ok=False,
                            row={
                                'miri_path': miri,
                                'filter': 'F2100W',
                                'status': 'PENDING',
                                'n_calibrators': 3,
                                'dispersion_mas': 93.7,
                                'aligned_path': str(child_jhat),
                                'align_mode': 'REFERENCE',
                                'original_ref': ref,
                                'aligned_to': ref,
                                'ref_overlap_frac': 0.5,
                            },
                            error='over threshold; try MIRI_REL',
                        )
                    )
            else:
                on_result(
                    AlignWorkerResult(
                        miri_path=miri,
                        filter=job['filter'],
                        mode='fallback',
                        ok=False,
                        row={
                            'miri_path': miri,
                            'filter': job['filter'],
                            'status': 'FAILURE',
                            'n_calibrators': 'NA',
                            'dispersion_mas': 'NA',
                            'aligned_path': 'NA',
                            'align_mode': 'NA',
                            'original_ref': 'NA',
                            'aligned_to': 'NA',
                            'ref_overlap_frac': job.get('ref_overlap_frac', 'NA'),
                        },
                        error='MIRI_REL did not improve',
                    )
                )

    with (
        patch.object(align_lib, '_run_jobs_parallel', side_effect=fake_run),
        patch.object(align_lib, '_resolve_repo_root', return_value=tmp_path),
        patch.object(align_lib, 'sky_overlap_fraction', return_value=0.8),
        patch.object(align_lib, 'flag_peer_inconsistent_reference_rows', return_value=[]),
    ):
        n_fail, rows = align_lib.align_from_frames(
            frames,
            run_alignment=MagicMock(),
            nbright=100,
            plot=False,
            verbose=False,
            fallback=True,
            workers=1,
            repo=tmp_path,
        )

    by_path = {r.miri_path: r for r in rows}
    assert by_path[str(parent_cal)].status == 'SUCCESS'
    assert by_path[str(child_cal)].status == 'FAILURE'
    assert n_fail == 1


def test_write_provenance_jwncal_comment(tmp_path: Path):
    jhat = tmp_path / 'x_jhat.fits'
    fits.PrimaryHDU(header=fits.Header({'JWDISPM': 0.03})).writeto(jhat)
    align_lib.write_alignment_provenance(
        str(jhat),
        align_mode='REFERENCE',
        original_ref='/ref.fits',
        aligned_to='/ref.fits',
        relative_dispersion_mas=30.0,
        absolute_dispersion_mas=30.0,
        n_calibrators=17,
    )
    with fits.open(jhat) as hdul:
        assert int(hdul[0].header['JWNCAL']) == 17
        assert 'calibrator' in hdul[0].header.comments['JWNCAL'].lower()
