"""Tests for the unified CLI options module."""

from __future__ import annotations

from pathlib import Path

from st123.scripts import align as align_script
from st123.scripts import download as download_script
from st123.scripts import link_raw
from st123.scripts import mosaic as mosaic_script
from st123.scripts.utils.options import (
    configure_logging_from_args,
    create_parser,
    dataset_label,
    default_hst_warmstart_outdir,
    default_instruments_for_telescope,
    default_miri_outdir,
    default_phot_dir,
    default_warmstart_outdir,
    ensure_project_layout,
    ensure_writable_dir,
    instrument_raw_dir,
    resolve_project_root,
    resolve_reduction_dir,
    st123_version_string,
)
from st123.scripts.link_raw import resolve_link_source_dirs


def test_version_flag():
    parser = create_parser('demo')
    try:
        parser.parse_args(['--version'])
    except SystemExit as exc:
        assert exc.code == 0


def test_version_string_contains_st123():
    assert st123_version_string().startswith('st123 ')


def test_default_instruments_for_telescope():
    assert default_instruments_for_telescope('hst') == ['ACS', 'WFC3', 'WFPC2']
    assert default_instruments_for_telescope('jwst') == ['NIRCAM', 'MIRI']
    assert default_instruments_for_telescope(None) is None


def test_ensure_project_layout_creates_standard_dirs(tmp_path: Path):
    root = tmp_path / 'sn1983e'
    out = ensure_project_layout(root)
    assert out == root.resolve()
    assert (root / 'download').is_dir()
    assert (root / 'reduction').is_dir()
    assert (root / 'logs').is_dir()
    # Fresh project now resolves reduction/ correctly for align/mosaic.
    assert resolve_reduction_dir(root) == (root / 'reduction').resolve()


def test_ensure_writable_dir_permission_error(tmp_path: Path, monkeypatch):
    target = tmp_path / 'blocked'

    def _boom(*_a, **_k):
        raise OSError(13, 'Permission denied')

    monkeypatch.setattr(Path, 'mkdir', _boom)
    try:
        ensure_writable_dir(target, label='download')
    except PermissionError as exc:
        msg = str(exc)
        assert 'download' in msg
        assert 'Permission' in msg or 'permission' in msg
    else:
        raise AssertionError('expected PermissionError')


def test_configure_logging_exits_on_permission_error(tmp_path: Path, monkeypatch):
    import argparse

    from st123.utils.logging import shutdown_logging

    args = argparse.Namespace(base_dir=str(tmp_path / 'obj'), verbose=False)

    def _boom(*_a, **_k):
        raise PermissionError('Cannot create project directory ...')

    monkeypatch.setattr(
        'st123.scripts.utils.options.ensure_project_layout', _boom
    )
    try:
        configure_logging_from_args(args, 'download')
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError('expected SystemExit(1)')
    finally:
        shutdown_logging()


def test_ensure_project_layout_legacy_workdir_only_logs(tmp_path: Path):
    (tmp_path / 'raw').mkdir()
    ensure_project_layout(tmp_path)
    assert (tmp_path / 'logs').is_dir()
    assert not (tmp_path / 'download').exists()
    assert not (tmp_path / 'reduction').exists()


def test_resolve_reduction_dir_project_root(tmp_path: Path):
    (tmp_path / 'JWST').mkdir()
    assert resolve_reduction_dir(tmp_path) == (tmp_path / 'reduction').resolve()


def test_resolve_reduction_dir_from_download_marker(tmp_path: Path):
    (tmp_path / 'download').mkdir()
    assert resolve_reduction_dir(tmp_path) == (tmp_path / 'reduction').resolve()


def test_resolve_reduction_dir_legacy_workdir(tmp_path: Path):
    assert resolve_reduction_dir(tmp_path) == tmp_path.resolve()


def test_resolve_reduction_dir_existing_markers(tmp_path: Path):
    (tmp_path / 'raw').mkdir()
    assert resolve_reduction_dir(tmp_path) == tmp_path.resolve()


def test_resolve_reduction_dir_prefers_reduction_over_reference_symlink(
    tmp_path: Path,
):
    """Project-root reference/ symlink must not steal reduction resolution."""
    reduction = tmp_path / 'reduction'
    (tmp_path / 'JWST').mkdir()
    (reduction / 'jhat').mkdir(parents=True)
    (reduction / 'reference').mkdir(parents=True)
    (tmp_path / 'reference').symlink_to(reduction / 'reference')
    assert resolve_reduction_dir(tmp_path) == reduction.resolve()


def test_resolve_project_root_from_reduction(tmp_path: Path):
    red = tmp_path / 'reduction'
    red.mkdir()
    (red / 'raw').mkdir()
    assert resolve_project_root(red) == tmp_path.resolve()


def test_resolve_project_root_from_download(tmp_path: Path):
    dl = tmp_path / 'download'
    (dl / 'HST').mkdir(parents=True)
    assert resolve_project_root(dl) == tmp_path.resolve()


def test_instrument_raw_dir_prefers_download(tmp_path: Path):
    preferred = tmp_path / 'download' / 'HST' / 'ACS'
    preferred.mkdir(parents=True)
    legacy = tmp_path / 'HST' / 'ACS'
    legacy.mkdir(parents=True)
    assert instrument_raw_dir(tmp_path, 'ACS', telescope='HST') == preferred


def test_instrument_raw_dir_legacy_fallback(tmp_path: Path):
    legacy = tmp_path / 'HST' / 'ACS'
    legacy.mkdir(parents=True)
    assert instrument_raw_dir(tmp_path, 'ACS', telescope='HST') == legacy


def test_link_source_all_uses_download_tree(tmp_path: Path):
    acs = tmp_path / 'download' / 'HST' / 'ACS'
    wfc3 = tmp_path / 'download' / 'HST' / 'WFC3'
    acs.mkdir(parents=True)
    wfc3.mkdir(parents=True)
    roots = resolve_link_source_dirs(
        base_dir=tmp_path, telescope='HST', instrument='ALL'
    )
    assert roots == [acs, wfc3]


def test_dataset_label_from_base_dir(tmp_path: Path):
    root = tmp_path / 'NGC3310'
    (root / 'JWST').mkdir(parents=True)
    assert dataset_label(root) == 'NGC3310'
    assert dataset_label(root / 'reduction') == 'NGC3310'


def test_default_phot_dir_has_no_extra_obj_level(tmp_path: Path):
    root = tmp_path / 'NGC3310'
    (root / 'JWST').mkdir(parents=True)
    phot = default_phot_dir(root)
    assert phot == (root / 'reduction' / 'phot_0_0').resolve()
    # Only one NGC3310 in the full path (the project root)
    assert str(phot).count('NGC3310') == 1
    assert phot.parts[-2:] == ('reduction', 'phot_0_0')


def test_default_dolphot_outdirs(tmp_path: Path):
    root = tmp_path / 'NGC3310'
    (root / 'JWST').mkdir(parents=True)
    assert default_miri_outdir(root) == (root / 'dolphot' / 'miri_0_0').resolve()
    assert default_warmstart_outdir(root) == (
        root / 'dolphot' / 'nircam_miri_0_0'
    ).resolve()
    assert default_hst_warmstart_outdir(root) == (
        root / 'dolphot' / 'nircam_hst_0_0'
    ).resolve()
    assert default_miri_outdir(root, group=1, box=2) == (
        root / 'dolphot' / 'miri_1_2'
    ).resolve()


def test_dolphot_warmstart_target_hst_path_defaults(tmp_path: Path):
    from st123.scripts import dolphot_warmstart as ws_script

    root = tmp_path / 'Target'
    (root / 'JWST').mkdir(parents=True)
    (root / 'reduction').mkdir()
    parser = ws_script.create_parser()
    args = parser.parse_args(
        ['--base-dir', str(root), '--instruments', 'HST', '--ncores', '4']
    )
    assert ws_script.resolve_warmstart_targets(args.instruments) == ['hst']
    nircam, outdir, data_root, summary = ws_script._resolve_warmstart_paths(
        args, target='hst'
    )
    assert Path(nircam) == (root / 'reduction' / 'phot_0_0').resolve()
    assert Path(outdir) == (root / 'dolphot' / 'nircam_hst_0_0').resolve()
    assert Path(data_root) == root.resolve()
    assert summary is None

    args_miri = parser.parse_args(['--base-dir', str(root)])
    assert ws_script.resolve_warmstart_targets(
        args_miri.instruments, legacy_target=args_miri.target
    ) == ['miri']
    _, out_miri, _, summary_miri = ws_script._resolve_warmstart_paths(
        args_miri, target='miri'
    )
    assert Path(out_miri) == (root / 'dolphot' / 'nircam_miri_0_0').resolve()
    assert summary_miri is not None


def test_dolphot_warmstart_prep_cli_shape(tmp_path: Path):
    from st123.scripts import dolphot_warmstart as ws_script

    parser = ws_script.create_parser()
    phot = tmp_path / 'phot_0_0'
    args = parser.parse_args(
        [
            '--instruments',
            'MIRI',
            '--base-dir',
            str(tmp_path),
            '--ref-dir',
            str(phot),
            '--prune-xyt',
            '--ncores',
            '8',
            '-v',
        ]
    )
    assert args.instruments == ['MIRI']
    assert args.ref_dir == str(phot)
    assert args.prune_xyt is True
    assert args.ncores == 8
    assert ws_script.resolve_warmstart_targets(args.instruments) == ['miri']
    seed, outdir, _, _ = ws_script._resolve_warmstart_paths(args, target='miri')
    assert Path(seed) == phot
    assert Path(outdir) == (tmp_path / 'dolphot' / 'nircam_miri_0_0')


def test_dolphot_warmstart_ref_dir_inherits_group_box(tmp_path: Path):
    from st123.scripts import dolphot_warmstart as ws_script

    root = tmp_path / 'Target'
    (root / 'JWST').mkdir(parents=True)
    phot = root / 'reduction' / 'phot_0_sn'
    phot.mkdir(parents=True)
    parser = ws_script.create_parser()
    args = parser.parse_args(
        [
            '--instruments',
            'MIRI',
            'HST',
            '--base-dir',
            str(root),
            '--ref-dir',
            str(phot),
        ]
    )
    assert args.ref_dir == str(phot)
    assert ws_script.group_box_from_seed_dirname(phot.name) == (0, 'sn')
    seed_m, out_m, _, _ = ws_script._resolve_warmstart_paths(args, target='miri')
    seed_h, out_h, _, _ = ws_script._resolve_warmstart_paths(args, target='hst')
    assert Path(seed_m) == phot
    assert Path(seed_h) == phot
    assert Path(out_m) == (root / 'dolphot' / 'nircam_miri_0_sn').resolve()
    assert Path(out_h) == (root / 'dolphot' / 'nircam_hst_0_sn').resolve()


def test_dolphot_warmstart_acs_wfc3_plan():
    from st123.scripts import dolphot_warmstart as ws_script

    plan = ws_script.resolve_warmstart_plan(['ACS', 'WFC3'])
    assert plan == [('hst', ['ACS', 'WFC3'])]
    assert ws_script.resolve_warmstart_plan(['HST']) == [('hst', None)]
    assert ws_script.resolve_warmstart_plan(['MIRI', 'ACS', 'WFC3']) == [
        ('miri', None),
        ('hst', ['ACS', 'WFC3']),
    ]


def test_align_accepts_base_dir_without_obj():
    parser = align_script.create_parser()
    args = parser.parse_args(['--base-dir', '/tmp/w', '--ncores', '2'])
    assert args.base_dir == '/tmp/w'
    assert args.ncores == 2
    assert not hasattr(args, 'obj') or getattr(args, 'obj', None) is None


def test_align_accepts_legacy_workdir():
    parser = align_script.create_parser()
    args = parser.parse_args(['--workdir', '/tmp/w', '--workers', '4'])
    assert args.base_dir == '/tmp/w'
    assert args.ncores == 4


def test_mosaic_accepts_base_dir():
    parser = mosaic_script.create_parser()
    args = parser.parse_args(['--base-dir', '/b', '--ncores', '2'])
    assert args.base_dir == '/b'


def test_download_requires_base_dir():
    parser = download_script.create_parser()
    args = parser.parse_args(
        ['--ra', '10', '--dec', '20', '--base-dir', '/data/out']
    )
    assert args.base_dir == '/data/out'


def test_link_raw_base_dir_instrument():
    parser = link_raw.create_parser()
    args = parser.parse_args(['--base-dir', '/data/proj', '--instrument', 'NIRCAM'])
    assert args.base_dir == '/data/proj'
    assert args.instruments == ['NIRCAM']


def test_link_raw_accepts_instruments_list():
    parser = link_raw.create_parser()
    args = parser.parse_args(
        ['--base-dir', '/data/proj', '--instruments', 'ACS', 'WFC3', 'WFPC2']
    )
    assert args.instruments == ['ACS', 'WFC3', 'WFPC2']
    assert link_raw.resolve_cli_instruments(args) == ['ACS', 'WFC3', 'WFPC2']

    hst = parser.parse_args(['--base-dir', '/data/proj', '--instruments', 'hst'])
    assert link_raw.resolve_cli_instruments(hst) == ['ACS', 'WFC3', 'WFPC2']

    all_tel = parser.parse_args(
        ['--base-dir', '/data/proj', '--telescope', 'hst', '--instruments', 'ALL']
    )
    assert link_raw.resolve_cli_instruments(all_tel) == ['ALL']


def test_link_raw_legacy_datadir_symlinkdir():
    parser = link_raw.create_parser()
    args = parser.parse_args(['--datadir', '/d', '--symlinkdir', '/s'])
    assert args.source_dir == '/d'
    assert args.symlink_dir == '/s'


def test_expand_mission_instruments_aliases():
    from st123.scripts.utils.options import (
        expand_mission_instruments,
        resolve_instruments,
        resolve_photometry_instrument,
    )

    assert expand_mission_instruments(['hst']) == ['ACS', 'WFC3', 'WFPC2']
    assert expand_mission_instruments(['jwst']) == ['NIRCAM', 'MIRI']
    assert expand_mission_instruments(['all']) == [
        'NIRCAM',
        'MIRI',
        'ACS',
        'WFC3',
        'WFPC2',
    ]
    assert resolve_instruments(['hst', 'NIRCAM']) == [
        'ACS',
        'WFC3',
        'WFPC2',
        'NIRCAM',
    ]
    assert resolve_photometry_instrument(['hst']) == 'hst'
    assert resolve_photometry_instrument(['ACS', 'WFC3', 'WFPC2']) == 'hst'
    assert resolve_photometry_instrument(['jwst']) == 'nircam'


def test_pipeline_parsers_accept_shared_instruments_hst_and_sky():
    """download/align/mosaic/dolphot-prep/run-dolphot share --instruments hst."""
    from st123.scripts import dolphot as dolphot_script
    from st123.scripts import run_dolphot as run_dolphot_script

    shared = [
        '--base-dir',
        '/tmp/2011ja',
        '--instruments',
        'hst',
        '--ncores',
        '8',
        '-v',
        '--ra',
        '196.296329',
        '--dec',
        '-49.524169',
    ]
    dl = download_script.create_parser().parse_args(shared)
    assert dl.instruments == ['hst']
    assert dl.ra == '196.296329'
    al = align_script.create_parser().parse_args(shared)
    assert al.instruments == ['hst']
    assert al.ra == '196.296329'
    mo = mosaic_script.create_parser().parse_args(shared)
    assert mo.instruments == ['hst']
    assert mo.ra == '196.296329'
    prep = dolphot_script.create_parser().parse_args(shared)
    assert prep.instruments == ['hst']
    assert prep.ra == '196.296329'
    run = run_dolphot_script.create_parser().parse_args(shared)
    assert run.instruments == ['hst']
    assert run.ra == '196.296329'


def test_create_parser_attaches_stamp_box_args():
    """Stamp/box identity lives on the shared stage primitive, not per CLI."""
    parser = create_parser('demo')
    args = parser.parse_args(
        ['--existing-box', 'group_0/ref_sn', '--stamp-id', 'core']
    )
    assert args.existing_box == 'group_0/ref_sn'
    assert args.stamp_id == 'core'
    assert args.center_ra is None
    assert args.center_dec is None
    assert args.stamp_size is None
    assert args.stamp_ref is None


def test_all_stage_parsers_accept_existing_box():
    """Pipeline stages parse --existing-box from the shared primitive dest."""
    from st123.scripts import dolphot as dolphot_script
    from st123.scripts import run_dolphot as run_dolphot_script

    box = ['--existing-box', 'group_0/ref_sn']
    al = align_script.create_parser().parse_args(['--base-dir', '/t', *box])
    assert al.existing_box == 'group_0/ref_sn'
    mo = mosaic_script.create_parser().parse_args(['--base-dir', '/t', *box])
    assert mo.existing_box == 'group_0/ref_sn'
    dl = download_script.create_parser().parse_args(
        ['--ra', '10', '--dec', '20', '--base-dir', '/t', *box]
    )
    assert dl.existing_box == 'group_0/ref_sn'
    prep = dolphot_script.create_parser().parse_args(['--base-dir', '/t', *box])
    assert prep.existing_box == 'group_0/ref_sn'
    run = run_dolphot_script.create_parser().parse_args(['--base-dir', '/t', *box])
    assert run.existing_box == 'group_0/ref_sn'
    lr = link_raw.create_parser().parse_args(['--base-dir', '/t', *box])
    assert lr.existing_box == 'group_0/ref_sn'
