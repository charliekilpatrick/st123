"""Unit tests for DOLPHOT prep and warm-start helpers."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from st123.photometry.dolphot import (
    classify_image_kind,
    dolphot_command,
    phot_to_xyt,
    write_paramfile,
)
from st123.photometry.warmstart import discover_miri_jhat
from st123.utils.helpers import get_detector_chip


def test_get_detector_chip_miri():
    assert get_detector_chip('jw03295_02101_00001_mirimage_jhat.fits') == 'mirimage'


def test_classify_image_kind():
    assert classify_image_kind('x_nrcb1_jhat.fits') == 'short'
    assert classify_image_kind('x_nrcblong_jhat.fits') == 'long'
    assert classify_image_kind('x_mirimage_jhat.fits') == 'miri'


def test_science_fits_paths_excludes_sky(tmp_path: Path):
    from st123.photometry.dolphot import science_fits_paths

    (tmp_path / 'a_jhat.fits').write_text('')
    (tmp_path / 'a_jhat.sky.fits').write_text('')
    (tmp_path / 'b_jhat.fits').write_text('')
    paths = science_fits_paths(tmp_path)
    names = sorted(Path(p).name for p in paths)
    assert names == ['a_jhat.fits', 'b_jhat.fits']


def test_parse_dolphot_frame_list(tmp_path: Path):
    from st123.photometry.dolphot import parse_dolphot_frame_list

    manifest = tmp_path / 'dolphot_frames.txt'
    ref = tmp_path / 'coadd_i2d.fits'
    f1 = tmp_path / 'a_jhat.fits'
    f2 = tmp_path / 'b_jhat.fits'
    for p in (ref, f1, f2):
        p.write_text('')
    manifest.write_text(
        '# group=1 box=2\n'
        f'# ref {ref}\n'
        f'{f1}\n'
        f'{f2}\n'
    )
    refimage, frames, group, box = parse_dolphot_frame_list(manifest)
    assert refimage == ref
    assert frames == [f1, f2]
    assert group == 1
    assert box == 2


def test_parse_param_image_list(tmp_path: Path):
    from st123.photometry.dolphot import parse_param_image_list

    param = tmp_path / 'dolphot.param'
    param.write_text(
        'Nimg = 2\n'
        'img0_file = coadd\n'
        'img1_file = a_nrcb1_jhat\n'
        'img2_file = b_mirimage_jhat\n'
    )
    ref, images = parse_param_image_list(param)
    assert ref == 'coadd'
    assert images == ['a_nrcb1_jhat', 'b_mirimage_jhat']


def test_per_image_params_kinds():
    from st123.photometry.dolphot import per_image_params
    from st123.utils import settings

    assert per_image_params('short') is settings.short_params
    assert per_image_params('long') is settings.long_params
    assert per_image_params('miri') is settings.miri_params
    with pytest.raises(ValueError):
        per_image_params('unknown')


def test_phot_to_xyt(tmp_path: Path):
    phot = tmp_path / 'run.phot'
    # ext Z X Y chi SNR sharp round maj crowd type ...
    phot.write_text(
        '1 1 10.5 20.5 1.0 50.0 0.0 0.0 0.0 0.0 1 1\n'
        '1 1 11.0 21.0 1.0 5.0 0.0 0.0 0.0 0.0 2 1\n'
        '1 1 12.0 22.0 1.0 9.0 0.0 0.0 0.0 0.0 4 1\n'
    )
    xyt = tmp_path / 'warmstart.xyt'
    phot_to_xyt(phot, xyt)
    lines = xyt.read_text().splitlines()
    assert lines[0] == '1 1 10.5 20.5 1 50.0'
    assert lines[1] == '1 1 11.0 21.0 2 5.0'
    assert len(lines) == 3

    phot_to_xyt(phot, xyt, types=[1])
    assert xyt.read_text().splitlines() == ['1 1 10.5 20.5 1 50.0']


def test_phot_to_xyt_miri_prune_and_force(tmp_path: Path):
    phot = tmp_path / 'run.phot'
    # Bright SN, close bright neighbor (should drop via min_sep), faint junk.
    phot.write_text(
        '1 1 100.0 200.0 1.0 100.0 0.01 0.0 0.0 0.2 1 1\n'
        '1 1 103.0 200.0 1.0 40.0 0.01 0.0 0.0 0.2 1 1\n'
        '1 1 150.0 250.0 1.0 5.0 0.01 0.0 0.0 0.2 1 1\n'
        '1 1 180.0 280.0 1.0 30.0 0.5 0.0 0.0 0.2 1 1\n'
        '1 1 200.0 300.0 1.0 25.0 0.01 0.0 0.0 2.0 1 1\n'
    )
    xyt = tmp_path / 'warmstart.xyt'
    phot_to_xyt(
        phot,
        xyt,
        types=[1],
        snr_min=10.0,
        crowd_max=0.5,
        sharp2_max=0.01,
        min_sep_pix=5.0,
        force_xy=[(100.0, 200.0)],
    )
    rows = xyt.read_text().splitlines()
    assert any(r.startswith('1 1 100.0 200.0') for r in rows)
    assert not any(r.startswith('1 1 103.0 200.0') for r in rows)  # too close
    assert not any(r.startswith('1 1 150.0 250.0') for r in rows)  # low SNR
    assert not any(r.startswith('1 1 180.0 280.0') for r in rows)  # sharp^2
    assert not any(r.startswith('1 1 200.0 300.0') for r in rows)  # crowded


def test_phot_to_xyt_max_radius(tmp_path: Path):
    phot = tmp_path / 'run.phot'
    phot.write_text(
        '1 1 100.0 200.0 1.0 100.0 0.01 0.0 0.0 0.2 1 1\n'
        '1 1 110.0 200.0 1.0 50.0 0.01 0.0 0.0 0.2 1 1\n'
        '1 1 300.0 200.0 1.0 80.0 0.01 0.0 0.0 0.2 1 1\n'
    )
    xyt = tmp_path / 'warmstart.xyt'
    phot_to_xyt(
        phot,
        xyt,
        types=[1],
        snr_min=10.0,
        force_xy=[(100.0, 200.0)],
        max_radius_pix=20.0,
    )
    rows = xyt.read_text().splitlines()
    assert any(r.startswith('1 1 100.0 200.0') for r in rows)
    assert any(r.startswith('1 1 110.0 200.0') for r in rows)
    assert not any(r.startswith('1 1 300.0 200.0') for r in rows)


def test_write_paramfile_includes_miri_and_xyt(tmp_path: Path):
    ref = tmp_path / 'coadd.fits'
    nircam = tmp_path / 'a_nrcb1_jhat.fits'
    miri = tmp_path / 'b_mirimage_jhat.fits'
    for p in (ref, nircam, miri):
        p.write_text('')
    xyt = tmp_path / 'warmstart.xyt'
    xyt.write_text('1 1 1.0 2.0 1 10.0\n')
    param = write_paramfile(
        tmp_path / 'dolphot.param',
        refimage=ref,
        images=[nircam, miri],
        xytfile=xyt,
    )
    text = param.read_text()
    assert 'Nimg = 2' in text
    assert 'img0_file = coadd' in text
    assert 'img1_raper = 2' in text  # short NIRCam
    assert 'img2_raper = 3' in text  # MIRI FitSky=2
    assert 'img2_rsky2 = 4 10' in text
    assert 'xytfile = warmstart.xyt' in text
    assert 'UseWCS = 2' in text
    assert 'MIRIvega = 0' in text
    assert 'FlagMask = 4' in text


def test_dolphot_command(tmp_path: Path):
    bin_dir = tmp_path / 'dolphot' / 'bin'
    bin_dir.mkdir(parents=True)
    cmd = dolphot_command(
        '/tmp/run',
        phot_out='out.phot',
        param_file='dolphot.param',
        dolphot_bin=bin_dir,
        ncores=32,
    )
    assert cmd.startswith('cd /tmp/run &&')
    assert 'dolphot out.phot -pdolphot.param MaxThreads=32' in cmd
    assert f'PATH={bin_dir.resolve()}:$PATH' in cmd

    default_cmd = dolphot_command(
        '/tmp/run',
        phot_out='out.phot',
        param_file='dolphot.param',
        dolphot_bin=bin_dir,
    )
    assert 'MaxThreads=1' in default_cmd

    nohup_cmd = dolphot_command(
        '/tmp/run',
        phot_out='out.phot',
        param_file='dolphot.param',
        dolphot_bin=bin_dir,
        ncores=32,
        nohup=True,
    )
    assert 'nohup' in nohup_cmd
    assert '> dolphot.out 2> dolphot.err' in nohup_cmd
    assert nohup_cmd.endswith('&')


def test_filter_frames_and_resolve_coadd_ref(tmp_path: Path):
    from st123.photometry.dolphot import (
        filter_frames_for_instrument,
        resolve_coadd_ref,
    )

    miri = tmp_path / 'a_mirimage_jhat.fits'
    nircam = tmp_path / 'b_nrcb1_jhat.fits'
    miri.write_text('')
    nircam.write_text('')
    assert filter_frames_for_instrument([miri, nircam], 'miri') == [miri]
    assert filter_frames_for_instrument([miri, nircam], 'nircam') == [nircam]

    box = tmp_path / 'ref_0'
    box.mkdir()
    f560 = box / 'coadd_0_0_f560w_i2d.fits'
    f770 = box / 'coadd_0_0_f770w_i2d.fits'
    f560.write_text('')
    f770.write_text('')
    assert resolve_coadd_ref(box, 'F560W', fallback=f770) == f560
    assert resolve_coadd_ref(box, None, fallback=f770) == f770


def test_resolve_dolphot_bin_from_which(tmp_path: Path, monkeypatch):
    from st123.photometry.dolphot import resolve_dolphot_bin

    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    fake_dolphot = fake_bin / 'dolphot'
    fake_dolphot.write_text('#!/bin/sh\n')
    fake_dolphot.chmod(0o755)
    monkeypatch.setenv('PATH', str(fake_bin))
    assert resolve_dolphot_bin(required=False) == fake_bin.resolve()


def test_resolve_dolphot_bin_missing_warns_and_required_raises(monkeypatch, caplog):
    import logging

    from st123.photometry.dolphot import resolve_dolphot_bin

    monkeypatch.setenv('PATH', '')
    with caplog.at_level(logging.WARNING, logger='st123.photometry.dolphot'):
        assert resolve_dolphot_bin(required=False) is None
    assert 'DOLPHOT not found on PATH' in caplog.text
    with pytest.raises(FileNotFoundError, match='DOLPHOT not found'):
        resolve_dolphot_bin(required=True)


def test_discover_miri_jhat_from_summary(tmp_path: Path):
    jhat = tmp_path / 'frame_mirimage_jhat.fits'
    jhat.write_text('x')
    summary = tmp_path / 'obj_alignment_summary.txt'
    summary.write_text(
        'miri_path filter status ref_overlap_frac n_calibrators '
        'dispersion_mas align_mode aligned_path original_ref aligned_to\n'
        f'/cal.fits F560W SUCCESS 0.95 10 1.0 REFERENCE {jhat} /ref.fits /ref.fits\n'
        f'/cal2.fits F560W FAILURE 0.95 10 1.0 REFERENCE {jhat} /ref.fits /ref.fits\n'
        f'/cal3.fits F560W SUCCESS 0.10 10 1.0 REFERENCE {jhat} /ref.fits /ref.fits\n'
    )
    found = discover_miri_jhat(tmp_path, alignment_summary=summary, min_overlap=0.5)
    assert found == [jhat.resolve()] or found == [jhat]
    assert len(found) == 1


@mock.patch('st123.photometry.dolphot.run_logged_subprocess')
def test_prepare_frames_miri_flags(mock_run, tmp_path: Path):
    from st123.photometry.dolphot import prepare_frames

    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    fits = tmp_path / 'x_mirimage_jhat.fits'
    fits.write_text('')
    prepare_frames([fits], instrument='miri', dolphot_bin=bin_dir)
    assert mock_run.call_count == 2
    mask_cmd = mock_run.call_args_list[0].args[0]
    sky_cmd = mock_run.call_args_list[1].args[0]
    assert mask_cmd[0].endswith('mirimask')
    assert '-estnoise' in mask_cmd
    assert '-noetctime' not in mask_cmd
    assert sky_cmd[0].endswith('calcsky')
    assert Path(sky_cmd[1]).name == 'x_mirimage_jhat'
    assert [float(x) for x in sky_cmd[2:7]] == [10.0, 25.0, -64.0, 2.25, 2.0]
    assert mock_run.call_args_list[1].kwargs.get('cwd') == fits.resolve().parent


@mock.patch('st123.photometry.dolphot.run_logged_subprocess')
def test_apply_nircammask_default_flags(mock_run, tmp_path: Path):
    """Installed nircammask has no -etctime; ETC time is the default."""
    from st123.photometry.dolphot import apply_nircammask

    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    fits = tmp_path / 'x_nrcb1_jhat.fits'
    fits.write_text('')
    apply_nircammask([fits], dolphot_bin=bin_dir)
    cmd = mock_run.call_args.args[0]
    assert cmd[0].endswith('nircammask')
    assert '-etctime' not in cmd
    assert cmd[-1] == 'x_nrcb1_jhat.fits'
    assert mock_run.call_args.kwargs.get('cwd') == fits.resolve().parent


def test_parse_and_discover_mosaic_phot_jobs(tmp_path: Path):
    from st123.photometry.dolphot import (
        discover_mosaic_phot_jobs,
        parse_dolphot_frame_list,
    )
    from st123.mosaic.mosaic import write_dolphot_frame_list

    reduction = tmp_path / 'reduction'
    box = reduction / 'reference' / 'group_0' / 'ref_0'
    jhat = reduction / 'jhat'
    box.mkdir(parents=True)
    jhat.mkdir(parents=True)
    ref = box / 'coadd_0_0_f150w2_i2d.fits'
    ref.write_text('ref')
    frames = [jhat / 'a_jhat.fits', jhat / 'b_jhat.fits']
    for path in frames:
        path.write_text('f')
    write_dolphot_frame_list(
        str(box),
        refimage=str(ref),
        frames=[str(p) for p in frames],
        group=0,
        box=0,
    )
    parsed_ref, parsed_frames, group, box_i = parse_dolphot_frame_list(
        box / 'dolphot_frames.txt'
    )
    assert group == 0 and box_i == 0
    assert parsed_ref == ref.resolve()
    assert [p.resolve() for p in parsed_frames] == [p.resolve() for p in frames]

    jobs = discover_mosaic_phot_jobs(reduction)
    assert len(jobs) == 1
    assert jobs[0].phot_outdir == reduction / 'phot_0_0'
    assert len(jobs[0].frames) == 2


def test_discover_mosaic_phot_jobs_miri_only(tmp_path: Path):
    from st123.photometry.dolphot import discover_mosaic_phot_jobs
    from st123.mosaic.mosaic import write_dolphot_frame_list

    project = tmp_path / 'NGC3310'
    reduction = project / 'reduction'
    box = reduction / 'reference' / 'group_0' / 'ref_0'
    jhat = reduction / 'jhat'
    box.mkdir(parents=True)
    jhat.mkdir(parents=True)
    ref_f560 = box / 'coadd_0_0_f560w_i2d.fits'
    ref_f150 = box / 'coadd_0_0_f150w2_i2d.fits'
    miri = jhat / 'a_mirimage_jhat.fits'
    nircam = jhat / 'b_nrcb1_jhat.fits'
    for path in (ref_f560, ref_f150, miri, nircam):
        path.write_text('x')
    write_dolphot_frame_list(
        str(box),
        refimage=str(ref_f150),
        frames=[str(miri), str(nircam)],
        group=0,
        box=0,
    )
    jobs = discover_mosaic_phot_jobs(
        reduction,
        instrument='miri',
        ref_filter='F560W',
        phot_outdir_root=project / 'dolphot',
        outdir_prefix='miri',
    )
    assert len(jobs) == 1
    assert jobs[0].refimage.resolve() == ref_f560.resolve()
    assert list(jobs[0].frames) == [miri]
    assert jobs[0].phot_outdir == project / 'dolphot' / 'miri_0_0'


def test_dolphot_from_mosaic_cli(tmp_path: Path):
    from st123.mosaic.mosaic import write_dolphot_frame_list
    from st123.scripts import dolphot as prep_script

    reduction = tmp_path / 'NGC3310' / 'reduction'
    box = reduction / 'reference' / 'group_0' / 'ref_0'
    jhat = reduction / 'jhat'
    box.mkdir(parents=True)
    jhat.mkdir(parents=True)
    (tmp_path / 'NGC3310' / 'JWST').mkdir(parents=True)
    ref = box / 'coadd_0_0_f150w2_i2d.fits'
    ref.write_bytes(b'')
    frame = jhat / 'x_nrcb1_jhat.fits'
    frame.write_bytes(b'')
    write_dolphot_frame_list(
        str(box), refimage=str(ref), frames=[str(frame)], group=0, box=0
    )

    with mock.patch(
        'st123.scripts.dolphot.prepare_mosaic_phot_job',
        return_value=reduction / 'phot_0_0' / 'dolphot.param',
    ) as prep:
        rc = prep_script.main(
            [
                '--from-mosaic',
                '--base-dir',
                str(tmp_path / 'NGC3310'),
                '--instrument',
                'nircam',
            ]
        )
    assert rc == 0
    prep.assert_called_once()
    job = prep.call_args.args[0]
    assert job.group == 0 and job.box == 0
    assert job.refimage.resolve() == ref.resolve()


@mock.patch('st123.photometry.dolphot.run_logged_subprocess')
def test_prepare_mosaic_phot_job_miri_writes_recommended_params(
    mock_run, tmp_path: Path
):
    """MIRI mosaic prep must stage frames and write dolphotMIRI defaults."""
    from st123.photometry.dolphot import MosaicPhotJob, prepare_mosaic_phot_job
    from st123.utils import settings

    outdir = tmp_path / 'dolphot' / 'miri_0_0'
    ref = tmp_path / 'coadd_0_0_f560w_i2d.fits'
    frame = tmp_path / 'x_mirimage_jhat.fits'
    ref.write_bytes(b'')
    frame.write_bytes(b'')
    job = MosaicPhotJob(
        group=0,
        box=0,
        refimage=ref,
        frames=(frame,),
        phot_outdir=outdir,
        frame_list=tmp_path / 'dolphot_frames.txt',
    )
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    param = prepare_mosaic_phot_job(
        job, instrument='miri', dolphot_bin=bin_dir
    )
    text = param.read_text()
    assert 'img0_file = coadd_0_0_f560w_i2d' in text
    assert 'img1_file = x_mirimage_jhat' in text
    assert f"img1_raper = {settings.miri_params['raper']}" in text
    assert 'UseWCS = 2' in text
    assert 'MIRIvega = 0' in text
    assert 'RCentroid = 1' in text
    assert (outdir / ref.name).is_file()
    assert (outdir / frame.name).is_file()
    # mirimask + calcsky for ref and science frame
    assert mock_run.call_count >= 2


def test_dolphot_prep_parser_miri_defaults():
    from st123.scripts import dolphot as prep_script

    parser = prep_script.create_parser()
    args = parser.parse_args(
        ['--from-mosaic', '--instrument', 'miri', '--base-dir', '/tmp/x', '--ncores', '32']
    )
    assert args.instrument == 'miri'
    assert args.from_mosaic is True
    assert args.ncores == 32
    assert args.ref_filter is None  # runtime default F560W applied in main


def test_dolphot_from_mosaic_cli_miri(tmp_path: Path):
    from st123.mosaic.mosaic import write_dolphot_frame_list
    from st123.scripts import dolphot as prep_script

    project = tmp_path / 'NGC3310'
    reduction = project / 'reduction'
    box = reduction / 'reference' / 'group_0' / 'ref_0'
    jhat = reduction / 'jhat'
    box.mkdir(parents=True)
    jhat.mkdir(parents=True)
    (project / 'JWST').mkdir(parents=True)
    ref = box / 'coadd_0_0_f560w_i2d.fits'
    miri = jhat / 'x_mirimage_jhat.fits'
    nircam = jhat / 'y_nrcb1_jhat.fits'
    for path in (ref, miri, nircam):
        path.write_bytes(b'')
    write_dolphot_frame_list(
        str(box),
        refimage=str(ref),
        frames=[str(miri), str(nircam)],
        group=0,
        box=0,
    )

    with mock.patch(
        'st123.scripts.dolphot.prepare_mosaic_phot_job',
        return_value=project / 'dolphot' / 'miri_0_0' / 'dolphot.param',
    ) as prep:
        rc = prep_script.main(
            [
                '--from-mosaic',
                '--base-dir',
                str(project),
                '--instrument',
                'miri',
                '--ncores',
                '32',
            ]
        )
    assert rc == 0
    prep.assert_called_once()
    job = prep.call_args.args[0]
    assert job.phot_outdir == project / 'dolphot' / 'miri_0_0'
    assert job.refimage.resolve() == ref.resolve()
    assert list(job.frames) == [miri]
    assert prep.call_args.kwargs['instrument'] == 'miri'


def test_chunk_images_equal_and_grouped():
    from st123.photometry.dolphot_split import chunk_images

    imgs = [Path(f'f{i}.fits') for i in range(10)]
    chunks = chunk_images(imgs, max_nimg=4)
    assert len(chunks) == 3
    assert sorted(len(c) for c in chunks) == [3, 3, 4]
    assert sum(len(c) for c in chunks) == 10

    keys = ['A'] * 3 + ['B'] * 3 + ['C'] * 4
    chunks = chunk_images(imgs, max_nimg=5, group_keys=keys)
    # A+B=6 would exceed 5, so A alone or with nothing oversized
    assert all(len(c) <= 5 for c in chunks)
    assert sum(len(c) for c in chunks) == 10
    # Images sharing a key stay together when the group fits.
    for c in chunks:
        names = {p.name for p in c}
        if 'f0.fits' in names:
            assert {'f0.fits', 'f1.fits', 'f2.fits'} <= names


def test_write_split_paramfiles_and_merge(tmp_path: Path):
    from st123.photometry.dolphot_split import (
        SPLIT_MANIFEST_NAME,
        finalize_split_outdir,
        merge_dolphot_phot_catalogs,
        write_split_paramfiles,
    )

    ref = tmp_path / 'coadd.fits'
    ref.write_text('')
    images = []
    for i in range(6):
        p = tmp_path / f'img{i}_nrcb1_jhat.fits'
        p.write_text('')
        images.append(p)
    xyt = tmp_path / 'warmstart.xyt'
    xyt.write_text('1 1 10.0 20.0 1 50.0\n1 1 11.0 21.0 1 40.0\n')

    plan = write_split_paramfiles(
        tmp_path,
        refimage=ref,
        images=images,
        phot_out='run_nircam_miri.phot',
        xytfile=xyt,
        max_nimg=4,
        read_filters=False,
    )
    assert plan.needs_merge
    assert len(plan.parts) == 2
    assert (tmp_path / SPLIT_MANIFEST_NAME).is_file()
    assert (tmp_path / 'dolphot_part00.param').is_file()
    assert (tmp_path / 'dolphot_full.param').is_file()
    assert not (tmp_path / 'dolphot.param').is_file()

    # Synthetic part catalogs: 12 object cols + one 13-col combined block each.
    def _write_part(name: str, filt: str, mags: list[str]) -> Path:
        phot = tmp_path / name
        cols = tmp_path / f'{name}.columns'
        obj = [
            'Extension',
            'Chip',
            'Object X position',
            'Object Y position',
            'Chi for fit',
            'Signal-to-noise',
            'Object sharpness',
            'Object roundness',
            'Direction of major axis',
            'Crowding',
            'Object type',
            'Pass Detected',
        ]
        block = [
            f'Total counts, {filt}',
            f'Total sky level, {filt}',
            f'Normalized count rate, {filt}',
            f'Normalized count rate uncertainty, {filt}',
            f'Instrumental ABMAG magnitude, {filt}',
            f'Transformed UBVRI magnitude, {filt}',
            f'Magnitude uncertainty, {filt}',
            f'Chi, {filt}',
            f'Signal-to-noise, {filt}',
            f'Sharpness, {filt}',
            f'Roundness, {filt}',
            f'Crowding, {filt}',
            f'Photometry quality flag, {filt}',
        ]
        with cols.open('w') as fh:
            for i, n in enumerate(obj + block, start=1):
                fh.write(f'{i}. {n}\n')
        rows = []
        for i, mag in enumerate(mags):
            obj_vals = [
                '1',
                '1',
                f'{10.0 + i:.1f}',
                f'{20.0 + i:.1f}',
                '1.0',
                '50.0',
                '0.0',
                '0.0',
                '0.0',
                '0.1',
                '1',
                '1',
            ]
            phot_vals = ['100', '1', '1', '0.1', mag, mag, '0.01', '1', '50', '0', '0', '0.1', '0']
            rows.append(' '.join(obj_vals + phot_vals))
        phot.write_text('\n'.join(rows) + '\n')
        return phot

    p0 = _write_part(plan.parts[0].phot_out, 'NIRCAM_F115W', ['20.0', '21.0'])
    p1 = _write_part(plan.parts[1].phot_out, 'MIRI_F770W', ['18.0', '19.0'])
    merged = merge_dolphot_phot_catalogs([p0, p1], tmp_path / 'run_nircam_miri.phot')
    text = merged.read_text().splitlines()
    assert len(text) == 2
    # object(12) + F115W(13) + F770W(13) = 38 fields
    assert len(text[0].split()) == 38
    cols = (tmp_path / 'run_nircam_miri.phot.columns').read_text()
    assert 'NIRCAM_F115W' in cols and 'MIRI_F770W' in cols

    # finalize_split_outdir should no-op once merged exists with size>0… rewrite empty
    (tmp_path / 'run_nircam_miri.phot').unlink()
    out = finalize_split_outdir(tmp_path)
    assert out is not None
    assert out.name == 'run_nircam_miri.phot'


def test_setup_paramfile_respects_max_nimg(tmp_path: Path):
    from st123.photometry.dolphot import setup_paramfile
    from st123.photometry.dolphot_split import SPLIT_MANIFEST_NAME

    ref = tmp_path / 'coadd.fits'
    ref.write_text('')
    files = []
    for i in range(5):
        p = tmp_path / f'a{i}_nrcb1_jhat.fits'
        p.write_text('')
        files.append(p)
    plan = setup_paramfile(
        tmp_path / 'phot',
        ref,
        files,
        copy_files=True,
        phot_out='phot.phot',
        max_nimg=3,
        return_plan=True,
    )
    assert plan.needs_merge
    assert (tmp_path / 'phot' / SPLIT_MANIFEST_NAME).is_file()
    assert all(part.nimg <= 3 for part in plan.parts)
