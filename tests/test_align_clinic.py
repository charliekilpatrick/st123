"""Local HST alignment clinic: classify salvageable vs unsalvageable frames."""

from __future__ import annotations

from pathlib import Path

from st123.stages.alignment.clinic import (
    L3_PAIRWISE_SOFT_LIMIT,
    apply_salvage,
    assess_clinic_recovery,
    classify_frame,
    cluster_l3_peak_offsets,
    diagnose_base_dir,
    l3_qa_pair_budget,
    plan_visit_relative_salvage,
    prefer_optical_l3_hub,
)
from st123.stages.alignment.frame_qa import (
    QF_EXPFLAG,
    QF_JHAT_NO_SOURCES,
    QF_NO_GUIDE,
    QF_OK,
    QF_RECOVERED_GAIA,
    QF_UV_SPARSE,
    isolated_unmatched_coadds,
)


def test_classify_ngc1350_xcorr_leftover_wants_gaia_shift():
    classed = classify_frame(
        {
            'dispersion_mas': 350.3,
            'n_calibrators': 25,
            'catastrophic': True,
            'image_peak_arcsec': 2.88,
            'image_residual_arcsec': 0.0,
            'image_xcorr': {'abs_arcsec': 0.0, 'n_inner': 2386},
            'gaia_residual_arcsec': 17.5,
            'internal_max_delta_mas': 19.4,
        }
    )
    assert classed['salvageable'] is True
    assert classed['flag'] == QF_RECOVERED_GAIA
    assert classed['success_path'] == 'gaia_visit_shift'
    assert classed['action'] == 'apply_gaia'


def test_classify_uv_sparse_keep_forced():
    classed = classify_frame(
        {'n_calibrators': 1, 'dispersion_mas': None},
        filter_name='F300X',
    )
    assert classed['flag'] == QF_UV_SPARSE
    assert classed['action'] == 'keep_forced'


def test_classify_loss_of_lock_unsalvageable():
    classed = classify_frame({}, expflag=QF_NO_GUIDE)
    assert classed['salvageable'] is False
    assert classed['action'] == 'skip'


def test_classify_tdf_down_rejected():
    classed = classify_frame({}, expflag=QF_EXPFLAG)
    assert classed['salvageable'] is False
    assert classed['flag'] == QF_EXPFLAG
    assert classed['action'] == 'skip'


def test_classify_healthy_catalog_ok():
    classed = classify_frame(
        {
            'dispersion_mas': 41.0,
            'n_calibrators': 80,
            'catastrophic': False,
        }
    )
    assert classed['flag'] == QF_OK
    assert classed['success_path'] == 'jhat_stars'


def test_diagnose_missing_base(tmp_path: Path):
    report = diagnose_base_dir(tmp_path / 'no_such')
    assert report['exists'] is False
    assert report['ok'] is False


def test_diagnose_raw_without_jhat_is_remaining_issue(tmp_path: Path):
    base = tmp_path / 'sn2017bzb'
    raw = base / 'reduction' / 'raw'
    raw.mkdir(parents=True)
    (raw / 'u29r3101t_c0m.fits').write_text('x', encoding='utf-8')
    report = diagnose_base_dir(base)
    assert report['n_frames'] == 0
    assert report['ok'] is False
    assert 'no JHAT products' in str(report.get('error') or '')


def test_diagnose_merges_summary_quality_and_rejected_only(tmp_path: Path):
    import json

    from astropy.io import fits
    import numpy as np

    base = tmp_path / 'ngc1350'
    jhat = base / 'reduction' / 'jhat_hst'
    box = base / 'reduction' / 'reference' / 'group_1' / 'ref_0'
    jhat.mkdir(parents=True)
    box.mkdir(parents=True)
    (base / 'reduction' / 'jhat').symlink_to('jhat_hst')
    path = jhat / 'u29r5201t_jhat.fits'
    fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32)).writeto(path)
    with fits.open(path, mode='update') as hdul:
        hdul[0].header['EXPFLAG'] = 'NORMAL'
        hdul[0].header['ST123CAT'] = True
        hdul.flush()
    summary = {
        'results': [
            {
                'outpath': str(path),
                'status': 'ok',
                'quality': {
                    'dispersion_mas': 350.3,
                    'n_calibrators': 25,
                    'catastrophic': True,
                    'image_peak_arcsec': 2.88,
                    'image_residual_arcsec': 0.0,
                    'image_xcorr': {'abs_arcsec': 0.0, 'n_inner': 2386},
                    'gaia_residual_arcsec': 17.5,
                    'internal_max_delta_mas': 19.4,
                },
            }
        ]
    }
    (jhat / 'alignment_summary.json').write_text(json.dumps(summary), encoding='utf-8')
    (jhat / 'frame_qa.json').write_text(
        json.dumps({'frames': [{'path': str(path), 'status': 'ok'}]}),
        encoding='utf-8',
    )
    rejected = box / 'coadd_1_0_wfc3_f300x_drc.rejected.fits'
    rejected.write_text('bad', encoding='utf-8')
    report = diagnose_base_dir(base)
    assert report['n_frames'] == 1
    row = report['frames'][0]
    assert row['flag'] == QF_RECOVERED_GAIA
    assert row['action'] == 'apply_gaia'
    assert report['rejected_only'] == [str(rejected.resolve())]
    assert report['ok'] is False


def test_align_clinic_parser_accepts_shared_and_stage_flags():
    from st123.scripts import align_clinic as clinic_script

    parser = clinic_script.create_parser()
    args = parser.parse_args(
        [
            '--base-dir',
            '/tmp/ic5267',
            '--instruments',
            'hst',
            '--scan-known',
            '--ra',
            '10',
            '--dec',
            '20',
        ]
    )
    assert args.base_dir == '/tmp/ic5267'
    assert args.instruments == ['hst']
    assert args.scan_known is True
    assert args.apply is False
    assert args.mosaic_after is False


# ---------------------------------------------------------------------------
# Remaining clinic failure modes: recovery assessment (no GB coadds in CI).
# Scenarios mirror sn2021abkm, sn2017bzb, sn2016bkv / sn2020zbv.
# ---------------------------------------------------------------------------


def test_classify_jhat_no_sources_plans_visit_relative():
    """sn2017bzb: 0 JHAT stars -> visit_relative / sibling WCS, not skip."""
    classed = classify_frame(
        {},
        error='Only 0 objects pass the initial cut',
    )
    assert classed['salvageable'] is True
    assert classed['flag'] == QF_JHAT_NO_SOURCES
    assert classed['success_path'] == 'visit_relative'
    assert classed['action'] == 'visit_relative'


def test_plan_visit_relative_salvage_picks_same_visit_donor():
    """Donor is a same-visit OK frame; apply product write still pending."""
    frames = [
        {
            'path': '/tmp/visitA_ok_jhat.fits',
            'visit': 'visitA',
            'flag': QF_OK,
            'action': 'keep',
        },
        {
            'path': '/tmp/visitA_empty_jhat.fits',
            'visit': 'visitA',
            'flag': QF_JHAT_NO_SOURCES,
            'action': 'visit_relative',
        },
        {
            'path': '/tmp/visitB_empty_jhat.fits',
            'visit': 'visitB',
            'flag': QF_JHAT_NO_SOURCES,
            'action': 'visit_relative',
        },
    ]
    plan = plan_visit_relative_salvage(frames)
    assert plan['n_targets'] == 2
    assert plan['n_with_donor'] == 2
    assert plan['apply_implemented'] is True
    by_target = {p['target']: p for p in plan['plans']}
    assert by_target['/tmp/visitA_empty_jhat.fits']['donor'] == (
        '/tmp/visitA_ok_jhat.fits'
    )
    assert by_target['/tmp/visitA_empty_jhat.fits']['method'] == (
        'copy_sibling_wcs'
    )
    # No same-visit donor: fall back to first OK, still planned.
    assert by_target['/tmp/visitB_empty_jhat.fits']['donor'] == (
        '/tmp/visitA_ok_jhat.fits'
    )


def test_apply_salvage_writes_visit_relative_from_raw(tmp_path: Path):
    """JHAT_NO_SOURCES raw -> pipeline JHAT under reduction/jhat_hst."""
    import numpy as np
    from astropy.io import fits

    base = tmp_path / 'field'
    raw_dir = base / 'reduction' / 'raw'
    jhat_dir = base / 'reduction' / 'jhat_hst'
    raw_dir.mkdir(parents=True)
    jhat_dir.mkdir(parents=True)
    def _sci_mef(path: Path) -> None:
        prim = fits.PrimaryHDU()
        sci = fits.ImageHDU(np.zeros((8, 8), dtype=np.float32), name='SCI')
        sci.header['CRVAL1'] = 10.0
        sci.header['CRVAL2'] = 20.0
        sci.header['CRPIX1'] = 4.0
        sci.header['CRPIX2'] = 4.0
        sci.header['CD1_1'] = -1e-4
        sci.header['CD2_2'] = 1e-4
        sci.header['CTYPE1'] = 'RA---TAN'
        sci.header['CTYPE2'] = 'DEC--TAN'
        fits.HDUList([prim, sci]).writeto(path, overwrite=True)

    raw = raw_dir / 'u29r3101t_c0m.fits'
    donor = jhat_dir / 'u29r3102t_jhat.fits'
    donor_raw = raw_dir / 'u29r3102t_c0m.fits'
    _sci_mef(raw)
    _sci_mef(donor)
    _sci_mef(donor_raw)
    report = {
        'base_dir': str(base),
        'frames': [
            {
                'path': str(donor),
                'visit': 'u29r31',
                'flag': QF_OK,
                'action': 'keep',
                'salvageable': True,
            },
            {
                'path': str(raw),
                'visit': 'u29r31',
                'flag': QF_JHAT_NO_SOURCES,
                'action': 'visit_relative',
                'salvageable': True,
            },
        ],
    }
    out = apply_salvage(report, mosaic_after=False)
    assert out['visit_relative_implemented'] is True
    assert out['pending_visit_relative'] == []
    assert len(out['visit_relative']) == 1
    written = Path(out['visit_relative'][0]['jhat'])
    assert written.is_file()
    assert written.name == 'u29r3101t_jhat.fits'
    assert out['visit_relative'][0]['ok'] is True


def test_apply_salvage_records_pending_when_raw_missing(tmp_path: Path):
    """apply_salvage must not silently drop visit_relative frames."""
    import numpy as np
    from astropy.io import fits

    jhat = tmp_path / 'u29r_nosrc_jhat.fits'
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32)).writeto(jhat)
    report = {
        'base_dir': str(tmp_path),
        'frames': [
            {
                'path': str(jhat),
                'flag': QF_JHAT_NO_SOURCES,
                'action': 'visit_relative',
                'salvageable': True,
            }
        ],
    }
    out = apply_salvage(report, mosaic_after=False)
    assert out['visit_relative_implemented'] is True
    assert out['pending_visit_relative'] == [str(jhat)]
    assert out['recovered'] == []


def test_prefer_optical_hub_when_uv_optical_peak_splits():
    """sn2021abkm: ~83\" F275W-F814W peak must not sink F555W; hub=F814W."""
    pairs = [
        {
            'a': 'coadd_0_0_wfc3_f275w_drc.fits',
            'b': 'coadd_0_0_acs_f814w_drc.fits',
            'abs_arcsec': 83.0,
            'n_match': 1,
            'method': 'smooth_peak',
            'ok': False,
        },
        {
            'a': 'coadd_0_0_acs_f555w_drc.fits',
            'b': 'coadd_0_0_acs_f814w_drc.fits',
            'abs_arcsec': 0.12,
            'n_match': 40,
            'method': 'catalog',
            'ok': True,
        },
        {
            'a': 'coadd_0_0_acs_f555w_drc.fits',
            'b': 'coadd_0_0_wfc3_f275w_drc.fits',
            'abs_arcsec': 82.5,
            'n_match': 1,
            'method': 'smooth_peak',
            'ok': False,
        },
    ]
    plan = prefer_optical_l3_hub(pairs)
    assert plan['needs_recovery'] is True
    assert plan['hub'] == 'coadd_0_0_acs_f814w_drc.fits'
    assert 'coadd_0_0_wfc3_f275w_drc.fits' in plan['sideline_uv']
    assert 'coadd_0_0_acs_f555w_drc.fits' in plan['retie_optical']
    # Isolated unmatched would drop UV; optical peers stay matched.
    report = {'pairs': pairs}
    isolated = isolated_unmatched_coadds(
        report, abs_ref_name='coadd_0_0_acs_f814w_drc.fits'
    )
    assert 'coadd_0_0_acs_f555w_drc.fits' not in isolated
    assert 'coadd_0_0_acs_f814w_drc.fits' not in isolated


def test_prefer_optical_hub_noop_when_filters_locked():
    pairs = [
        {
            'a': 'coadd_acs_f555w_drc.fits',
            'b': 'coadd_acs_f814w_drc.fits',
            'abs_arcsec': 0.04,
            'ok': True,
            'n_match': 50,
            'method': 'catalog',
        }
    ]
    plan = prefer_optical_l3_hub(pairs)
    assert plan['needs_recovery'] is False
    assert plan['hub'] is None


def test_l3_qa_pair_budget_recommends_hub_vs_each_for_large_grids():
    """sn2016bkv / sn2020zbv: 16-18 coadds make full pairwise too expensive."""
    small = l3_qa_pair_budget(4)
    assert small['recommended_mode'] == 'pairwise'
    assert small['n_pairs_pairwise'] == 6
    assert small['needs_fast_path'] is False

    large = l3_qa_pair_budget(16)
    assert large['n_pairs_pairwise'] == 120
    assert large['n_pairs_hub_vs_each'] == 15
    assert large['recommended_mode'] == 'hub_vs_each'
    assert large['needs_fast_path'] is True
    assert 16 > L3_PAIRWISE_SOFT_LIMIT

    huge = l3_qa_pair_budget(18)
    assert huge['n_pairs_pairwise'] == 153
    assert huge['n_pairs_hub_vs_each'] == 17


def test_cluster_l3_peak_offsets_detects_zbv_multi_cluster_wcs():
    """sn2020zbv: locked + ~47/65/87\" families need cross-filter unify."""
    offsets = [
        0.05,
        0.12,
        3.88,
        47.16,
        47.14,
        46.64,
        46.59,
        65.21,
        65.28,
        65.26,
        87.86,
        87.83,
        48.53,
        47.19,
    ]
    clustered = cluster_l3_peak_offsets(offsets, width_arcsec=5.0)
    assert clustered['n_clusters'] >= 3
    assert clustered['n_wide_clusters'] >= 2
    assert clustered['needs_cross_filter_unify'] is True
    centers = [c['center_arcsec'] for c in clustered['clusters']]
    # Rough families near 0, ~47, ~65, ~87 (3.88 may join locked or own bin).
    assert any(c < 5.0 for c in centers)
    assert any(40.0 <= c <= 55.0 for c in centers)
    assert any(60.0 <= c <= 70.0 for c in centers)
    assert any(c >= 80.0 for c in centers)


def test_cluster_l3_peak_offsets_single_locked_family():
    clustered = cluster_l3_peak_offsets([0.01, 0.04, 0.08, 0.15])
    assert clustered['n_clusters'] == 1
    assert clustered['needs_cross_filter_unify'] is False


def test_assess_clinic_recovery_bundles_open_actions():
    """End-to-end assessment payload for the four remaining clinic gaps."""
    frames = [
        {
            'path': '/tmp/ok_jhat.fits',
            'visit': 'v1',
            'flag': QF_OK,
            'action': 'keep',
        },
        {
            'path': '/tmp/nosrc_jhat.fits',
            'visit': 'v1',
            'flag': QF_JHAT_NO_SOURCES,
            'action': 'visit_relative',
        },
    ]
    pairs = [
        {
            'a': 'coadd_0_0_wfc3_f275w_drc.fits',
            'b': 'coadd_0_0_acs_f814w_drc.fits',
            'abs_arcsec': 83.0,
        },
        {
            'a': 'coadd_0_0_acs_f555w_drc.fits',
            'b': 'coadd_0_0_acs_f814w_drc.fits',
            'abs_arcsec': 0.1,
        },
    ]
    # Multi-filter grid + multi-cluster offsets (zbv-like).
    offsets = [0.1, 47.2, 47.1, 65.2, 87.8]
    assessment = assess_clinic_recovery(
        frames=frames,
        l3_pairs=pairs,
        n_coadds=16,
        l3_peak_offsets_arcsec=offsets,
    )
    assert 'prefer_optical_hub' in assessment['open_actions']
    assert 'write_visit_relative_jhat' in assessment['open_actions']
    assert 'fast_l3_qa' in assessment['open_actions']
    assert 'cross_filter_crval_unify' in assessment['open_actions']
    assert assessment['optical_hub']['hub'] == (
        'coadd_0_0_acs_f814w_drc.fits'
    )
    assert assessment['visit_relative']['n_planned'] == 1
    assert assessment['l3_qa']['recommended_mode'] == 'hub_vs_each'
    assert assessment['offset_clusters']['needs_cross_filter_unify'] is True


def test_assess_clinic_recovery_clean_field_has_no_open_actions():
    assessment = assess_clinic_recovery(
        frames=[
            {
                'path': '/tmp/a_jhat.fits',
                'visit': 'v1',
                'flag': QF_OK,
                'action': 'keep',
            }
        ],
        l3_pairs=[
            {
                'a': 'coadd_acs_f555w_drc.fits',
                'b': 'coadd_acs_f814w_drc.fits',
                'abs_arcsec': 0.05,
                'ok': True,
                'n_match': 30,
            }
        ],
        n_coadds=2,
        l3_peak_offsets_arcsec=[0.05, 0.08],
    )
    assert assessment['open_actions'] == []


def test_recover_failed_hst_visit_relative_writes_pipeline_jhat(tmp_path: Path):
    """align auto-recovery: JHAT_NO_SOURCES -> sibling/pipeline JHAT product."""
    import numpy as np
    from astropy.io import fits

    from st123.stages.alignment.hst_jhat import recover_failed_hst_visit_relative

    out = tmp_path / 'jhat_hst'
    raw_dir = tmp_path / 'raw'
    out.mkdir()
    raw_dir.mkdir()

    def _sci_mef(path: Path, *, crval1: float = 10.0) -> None:
        prim = fits.PrimaryHDU(
            header=fits.Header(
                {
                    'TELESCOP': 'HST',
                    'INSTRUME': 'WFPC2',
                    'EXPFLAG': 'NORMAL',
                    'ROOTNAME': path.name[:9],
                }
            )
        )
        sci = fits.ImageHDU(np.zeros((8, 8), dtype=np.float32), name='SCI')
        sci.header['CRVAL1'] = crval1
        sci.header['CRVAL2'] = 20.0
        sci.header['CRPIX1'] = 4.0
        sci.header['CRPIX2'] = 4.0
        sci.header['CD1_1'] = -1e-4
        sci.header['CD2_2'] = 1e-4
        sci.header['CTYPE1'] = 'RA---TAN'
        sci.header['CTYPE2'] = 'DEC--TAN'
        fits.HDUList([prim, sci]).writeto(path, overwrite=True)

    raw_fail = raw_dir / 'u29r3101t_c0m.fits'
    raw_ok = raw_dir / 'u29r3102t_c0m.fits'
    donor = out / 'u29r3102t_jhat.fits'
    _sci_mef(raw_fail)
    _sci_mef(raw_ok, crval1=10.001)
    _sci_mef(donor, crval1=10.001)
    results = [
        {
            'path': str(raw_ok),
            'status': 'ok',
            'outpath': str(donor),
            'error': None,
        },
        {
            'path': str(raw_fail),
            'status': 'failed',
            'outpath': None,
            'error': (
                'RuntimeError: Only 0 objects pass the initial cut, '
                'at least 3 required!'
            ),
        },
    ]
    recover_failed_hst_visit_relative(results, out, pipeline_fallback=True)
    failed = [r for r in results if 'u29r3101t' in r['path']][0]
    assert failed['status'] == 'ok'
    assert failed['outpath'] is not None
    assert Path(failed['outpath']).is_file()
    assert failed['align_mode'] in ('HST_REL', 'PIPELINE')

