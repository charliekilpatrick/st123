"""Tests for run-dolphot discovery and launch modes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from st123.scripts import run_dolphot as rd


def _prep_run(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'dolphot.param').write_text('Nimg = 1\n')
    return outdir


def test_discover_dolphot_runs_nircam(tmp_path: Path):
    project = tmp_path / 'SN'
    reduction = project / 'reduction'
    _prep_run(reduction / 'phot_0_0')
    _prep_run(reduction / 'phot_0_1')
    _prep_run(project / 'dolphot' / 'nircam_0_0')
    _prep_run(project / 'dolphot' / 'nircam_miri_0_0')  # warmstart: skip for nircam
    (reduction / 'phot_empty').mkdir(parents=True)  # no param: skip

    runs = rd.discover_dolphot_runs(project, instrument='nircam')
    names = {r.label for r in runs}
    assert names == {'phot_0_0', 'phot_0_1', 'nircam_0_0'}


def test_discover_dolphot_runs_miri_warmstarts(tmp_path: Path):
    """MIRI mode runs every nircam_miri_G_B reduction that has dolphot.param."""
    project = tmp_path / 'SN'
    r0 = _prep_run(project / 'dolphot' / 'nircam_miri_0_0')
    r1 = _prep_run(project / 'dolphot' / 'nircam_miri_0_1')
    _prep_run(project / 'dolphot' / 'miri_0_0')
    _prep_run(project / 'reduction' / 'phot_0_0')  # free NIRCam: skip for miri

    runs = rd.discover_dolphot_runs(project, instrument='miri')
    assert {r.label for r in runs} == {
        'nircam_miri_0_0',
        'nircam_miri_0_1',
        'miri_0_0',
    }
    by_name = {r.label: r for r in runs}
    assert by_name['nircam_miri_0_0'].phot_out == 'nircam_miri_0_0.phot'
    assert by_name['nircam_miri_0_0'].outdir == r0
    assert by_name['nircam_miri_0_1'].outdir == r1
    cmd = rd._format_command(
        by_name['nircam_miri_0_1'], ncores=8, dolphot_bin=None
    )
    assert 'nircam_miri_0_1.phot' in cmd
    assert 'MaxThreads=8' in cmd


def test_discover_dolphot_runs_hst_warmstarts(tmp_path: Path):
    """HST/ACS/WFC3 modes run every nircam_hst_G_B warmstart reduction."""
    project = tmp_path / 'SN'
    r0 = _prep_run(project / 'dolphot' / 'nircam_hst_0_0')
    r1 = _prep_run(project / 'dolphot' / 'nircam_hst_0_1')
    _prep_run(project / 'dolphot' / 'nircam_miri_0_0')  # MIRI warmstart: skip
    _prep_run(project / 'reduction' / 'phot_0_0')

    for inst in ('hst', 'acs', 'wfc3'):
        runs = rd.discover_dolphot_runs(project, instrument=inst)
        assert {r.label for r in runs} == {'nircam_hst_0_0', 'nircam_hst_0_1'}
        by_name = {r.label: r for r in runs}
        assert by_name['nircam_hst_0_0'].phot_out == 'nircam_hst_0_0.phot'
        assert by_name['nircam_hst_0_0'].outdir == r0
        assert by_name['nircam_hst_0_1'].outdir == r1

    cmd = rd._format_command(
        rd.DolphotRun(outdir=r1, phot_out='nircam_hst_0_1.phot'),
        ncores=8,
        dolphot_bin=None,
    )
    assert str(r1) in cmd
    assert 'nircam_hst_0_1.phot' in cmd
    assert 'MaxThreads=8' in cmd


def test_discover_filter_group_box(tmp_path: Path):
    project = tmp_path / 'SN'
    reduction = project / 'reduction'
    _prep_run(reduction / 'phot_0_0')
    _prep_run(reduction / 'phot_0_1')
    _prep_run(reduction / 'phot_1_0')
    runs = rd.discover_dolphot_runs(project, instrument='nircam', group=0, box=1)
    assert [r.label for r in runs] == ['phot_0_1']


def test_parser_background_and_dry_run():
    parser = rd.create_parser()
    args = parser.parse_args(
        [
            '--base-dir',
            '/tmp/p',
            '--instrument',
            'NIRCAM',
            '--ncores',
            '8',
            '--parallel',
            '4',
            '--background',
            '--dry-run',
        ]
    )
    assert args.instruments == ['NIRCAM']
    assert args.ncores == 8
    assert args.parallel == 4
    assert args.background is True
    assert args.dry_run is True


def test_dry_run_lists_without_executing(tmp_path: Path):
    project = tmp_path / 'SN'
    _prep_run(project / 'reduction' / 'phot_0_0')
    with patch.object(rd, '_run_one_wait') as wait, patch.object(
        rd, '_start_background'
    ) as bg:
        rc = rd.main(
            [
                '--base-dir',
                str(project),
                '--instrument',
                'nircam',
                '--ncores',
                '4',
                '--dry-run',
            ]
        )
    assert rc == 0
    wait.assert_not_called()
    bg.assert_not_called()


def test_wait_mode_runs_pool(tmp_path: Path):
    project = tmp_path / 'SN'
    _prep_run(project / 'reduction' / 'phot_0_0')
    _prep_run(project / 'reduction' / 'phot_0_1')

    def _ok(run, **kwargs):
        return run, 0

    with patch.object(rd, '_run_one_wait', side_effect=_ok) as wait:
        rc = rd.main(
            [
                '--base-dir',
                str(project),
                '--instrument',
                'nircam',
                '--ncores',
                '2',
                '--parallel',
                '2',
                '--wait',
            ]
        )
    assert rc == 0
    assert wait.call_count == 2


def test_background_refuses_when_more_runs_than_parallel(tmp_path: Path):
    project = tmp_path / 'SN'
    _prep_run(project / 'reduction' / 'phot_0_0')
    _prep_run(project / 'reduction' / 'phot_0_1')
    with patch.object(rd, '_start_background') as bg:
        rc = rd.main(
            [
                '--base-dir',
                str(project),
                '--instrument',
                'nircam',
                '--parallel',
                '1',
                '--background',
            ]
        )
    assert rc == 2
    bg.assert_not_called()


def test_background_starts_when_fits_parallel(tmp_path: Path):
    project = tmp_path / 'SN'
    _prep_run(project / 'reduction' / 'phot_0_0')
    _prep_run(project / 'reduction' / 'phot_0_1')
    proc = MagicMock()
    proc.pid = 12345
    with patch.object(rd, '_start_background', return_value=proc) as bg:
        rc = rd.main(
            [
                '--base-dir',
                str(project),
                '--instrument',
                'nircam',
                '--parallel',
                '2',
                '--background',
            ]
        )
    assert rc == 0
    assert bg.call_count == 2
