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
    # ext Z X Y chi SNR ... type(at col 11)
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
