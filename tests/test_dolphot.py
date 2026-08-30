"""Unit tests for DOLPHOT prep and warm-start helpers."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from st123.stages.photometry.dolphot import (
    classify_image_kind,
    dolphot_command,
    phot_to_xyt,
    write_paramfile,
)
from st123.stages.photometry.warmstart import discover_miri_jhat
from st123.utils.helpers import get_detector_chip


def test_get_detector_chip_miri():
    assert get_detector_chip('jw03295_02101_00001_mirimage_jhat.fits') == 'mirimage'


def test_classify_image_kind():
    assert classify_image_kind('x_nrcb1_jhat.fits') == 'short'
    assert classify_image_kind('x_nrcblong_jhat.fits') == 'long'
    assert classify_image_kind('x_mirimage_jhat.fits') == 'miri'


def test_science_fits_paths_excludes_sky(tmp_path: Path):
    from st123.stages.photometry.dolphot import science_fits_paths

    (tmp_path / 'a_jhat.fits').write_text('')
    (tmp_path / 'a_jhat.sky.fits').write_text('')
    (tmp_path / 'b_jhat.fits').write_text('')
    paths = science_fits_paths(tmp_path)
    names = sorted(Path(p).name for p in paths)
    assert names == ['a_jhat.fits', 'b_jhat.fits']


def test_parse_dolphot_frame_list(tmp_path: Path):
    from st123.stages.photometry.dolphot import parse_dolphot_frame_list

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


def test_parse_dolphot_frame_list_string_box(tmp_path: Path):
    from st123.stages.photometry.dolphot import parse_dolphot_frame_list

    manifest = tmp_path / 'dolphot_frames.txt'
    ref = tmp_path / 'coadd_i2d.fits'
    f1 = tmp_path / 'a_jhat.fits'
    for p in (ref, f1):
        p.write_text('')
    manifest.write_text(f'# group=0 box=sn\n# ref {ref}\n{f1}\n')
    _, _, group, box = parse_dolphot_frame_list(manifest)
    assert group == 0
    assert box == 'sn'


def test_parse_param_image_list(tmp_path: Path):
    from st123.stages.photometry.dolphot import parse_param_image_list

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
    from st123.stages.photometry.dolphot import per_image_params
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


def test_remap_xyt_extension(tmp_path: Path):
    from st123.stages.photometry.dolphot import remap_xyt_extension

    xyt = tmp_path / 'warmstart.xyt'
    xyt.write_text(
        '1 1 1584.25 2793.24 1 121.5\n'
        '1 1 1772.72 2758.77 1 444.0\n'
    )
    assert remap_xyt_extension(xyt, 0) == 2
    assert xyt.read_text().splitlines() == [
        '0 1 1584.25 2793.24 1 121.5',
        '0 1 1772.72 2758.77 1 444.0',
    ]


def test_ensure_dolphot_cd_matrix(tmp_path: Path):
    from astropy.io import fits
    import numpy as np

    from st123.stages.photometry.dolphot import ensure_dolphot_cd_matrix

    path = tmp_path / 'coadd_i2d.fits'
    hdr = fits.Header(
        {
            'CTYPE1': 'RA---TAN',
            'CTYPE2': 'DEC--TAN',
            'CRPIX1': 10.0,
            'CRPIX2': 10.0,
            'CRVAL1': 150.0,
            'CRVAL2': 2.0,
            'PC1_1': 0.9,
            'PC1_2': 0.4,
            'PC2_1': -0.4,
            'PC2_2': 0.9,
            'CDELT1': -1.0e-5,
            'CDELT2': 1.0e-5,
        }
    )
    fits.PrimaryHDU(data=np.ones((20, 20), dtype=np.float32), header=hdr).writeto(path)
    assert ensure_dolphot_cd_matrix(path) is True
    with fits.open(path) as hdul:
        h = hdul[0].header
        assert 'CD1_1' in h and 'PC1_1' not in h and 'CDELT1' not in h
        assert abs(h['CD1_1'] - (-9.0e-6)) < 1e-12
    assert ensure_dolphot_cd_matrix(path) is False


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
    assert 'nohup env PATH=' in nohup_cmd
    assert '> dolphot.out 2> dolphot.err' in nohup_cmd
    assert nohup_cmd.endswith('&')


def test_flatten_dolphot_fits_coadd_like(tmp_path: Path):
    from astropy.io import fits
    import numpy as np

    from st123.stages.photometry.dolphot import flatten_dolphot_fits

    path = tmp_path / 'coadd_wfc3_f625w_drc.fits'
    primary = fits.PrimaryHDU(header=fits.Header({'INSTRUME': 'WFC3', 'EXPTIME': 360.0}))
    sci = fits.ImageHDU(
        data=np.ones((8, 8), dtype=np.float32),
        name='SCI',
        header=fits.Header({'FILTER': 'F625W', 'GAIN': 1.4}),
    )
    fits.HDUList([primary, sci]).writeto(path)

    assert flatten_dolphot_fits(path) is True
    with fits.open(path) as hdul:
        assert len(hdul) == 1
        assert hdul[0].data.shape == (8, 8)
        assert hdul[0].header['FILTER'] == 'F625W'
        assert hdul[0].header['INSTRUME'] == 'WFC3'
        assert hdul[0].header['EXPTIME'] == 360.0
    assert flatten_dolphot_fits(path) is False


def test_flatten_dolphot_fits_breaks_hardlink(tmp_path: Path):
    """Warmstart hardlinks must not mutate the shared source inode."""
    from astropy.io import fits
    import numpy as np
    import os

    from st123.stages.photometry.dolphot import flatten_dolphot_fits

    src = tmp_path / 'coadd_i2d.fits'
    dst = tmp_path / 'staged_i2d.fits'
    primary = fits.PrimaryHDU()
    sci = fits.ImageHDU(data=np.ones((4, 4), dtype=np.float32), name='SCI')
    fits.HDUList([primary, sci]).writeto(src)
    os.link(src, dst)
    assert src.stat().st_nlink == 2

    assert flatten_dolphot_fits(dst) is True
    with fits.open(dst) as hdul:
        assert len(hdul) == 1
    with fits.open(src) as hdul:
        assert len(hdul) == 2
    assert src.stat().st_ino != dst.stat().st_ino


def test_filter_frames_and_resolve_coadd_ref(tmp_path: Path):
    from st123.stages.photometry.dolphot import (
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

    hst = box / 'coadd_0_0_wfc3_f814w_drc.fits'
    hst.write_text('')
    assert resolve_coadd_ref(box, 'F814W', fallback=f770) == hst


def test_discover_mosaic_phot_jobs_hst_boxed_fallback(tmp_path: Path):
    from astropy.io import fits

    from st123.stages.photometry.dolphot import discover_mosaic_phot_jobs

    reduction = tmp_path / 'reduction'
    box = reduction / 'reference' / 'group_0' / 'ref_5'
    jhat = reduction / 'jhat_hst'
    box.mkdir(parents=True)
    jhat.mkdir(parents=True)
    coadd = box / 'coadd_0_5_wfc3_f814w_drc.fits'
    frame = jhat / 'iey902_jhat.fits'
    coadd.write_text('coadd')
    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = 'WFC3'
    primary.header['DETECTOR'] = 'UVIS'
    fits.HDUList([primary]).writeto(frame, overwrite=True)
    jobs = discover_mosaic_phot_jobs(
        reduction, instrument='wfc3', outdir_prefix='wfc3'
    )
    assert len(jobs) == 1
    assert jobs[0].refimage == coadd
    assert jobs[0].phot_outdir.name == 'wfc3_0_5'
    assert jobs[0].frames[0].name == frame.name


def test_pick_hst_reference_prefers_boxed(tmp_path: Path):
    from st123.scripts.dolphot import _pick_hst_reference

    reduction = tmp_path / 'reduction'
    boxed = (
        reduction
        / 'reference'
        / 'group_0'
        / 'ref_5'
        / 'coadd_0_5_wfc3_f814w_drc.fits'
    )
    flat = reduction / 'reference' / 'coadd_wfc3_f625w_drc.fits'
    boxed.parent.mkdir(parents=True, exist_ok=True)
    flat.parent.mkdir(parents=True, exist_ok=True)
    boxed.write_text('boxed')
    flat.write_text('flat')
    picked = _pick_hst_reference(reduction, [], instrument='wfc3')
    assert picked == boxed


def test_resolve_dolphot_bin_from_which(tmp_path: Path, monkeypatch):
    from st123.stages.photometry.dolphot import resolve_dolphot_bin

    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    fake_dolphot = fake_bin / 'dolphot'
    fake_dolphot.write_text('#!/bin/sh\n')
    fake_dolphot.chmod(0o755)
    monkeypatch.setenv('PATH', str(fake_bin))
    assert resolve_dolphot_bin(required=False) == fake_bin.resolve()


def test_resolve_dolphot_bin_missing_warns_and_required_raises(monkeypatch, caplog):
    import logging

    from st123.stages.photometry.dolphot import resolve_dolphot_bin

    monkeypatch.setenv('PATH', '')
    with caplog.at_level(logging.WARNING, logger='st123.stages.photometry.dolphot'):
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


@mock.patch('st123.stages.photometry.dolphot.run_logged_subprocess')
def test_prepare_frames_miri_flags(mock_run, tmp_path: Path):
    from st123.stages.photometry.dolphot import prepare_frames

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


@mock.patch('st123.stages.photometry.dolphot.run_logged_subprocess')
def test_apply_nircammask_default_flags(mock_run, tmp_path: Path):
    """Installed nircammask has no -etctime; ETC time is the default."""
    from st123.stages.photometry.dolphot import apply_nircammask

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
    from st123.stages.photometry.dolphot import (
        discover_mosaic_phot_jobs,
        parse_dolphot_frame_list,
    )
    from st123.stages.mosaic.mosaic import write_dolphot_frame_list

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


def test_discover_mosaic_phot_jobs_ref_full(tmp_path: Path):
    from st123.stages.mosaic.mosaic import FULL_GROUP_LABEL, write_dolphot_frame_list
    from st123.stages.photometry.dolphot import (
        discover_mosaic_phot_jobs,
        parse_dolphot_frame_list,
    )

    reduction = tmp_path / 'reduction'
    box = reduction / 'reference' / 'group_0' / 'ref_full'
    jhat = reduction / 'jhat'
    box.mkdir(parents=True)
    jhat.mkdir(parents=True)
    ref = box / 'coadd_0_full_f770w_i2d.fits'
    ref.write_text('ref')
    frames = [jhat / 'a_mirimage_jhat.fits', jhat / 'b_mirimage_jhat.fits']
    for path in frames:
        path.write_text('f')
    write_dolphot_frame_list(
        str(box),
        refimage=str(ref),
        frames=[str(p) for p in frames],
        group=0,
        box=FULL_GROUP_LABEL,
    )
    _, _, group, box_id = parse_dolphot_frame_list(box / 'dolphot_frames.txt')
    assert group == 0 and box_id == 'full'

    jobs = discover_mosaic_phot_jobs(reduction)
    assert len(jobs) == 1
    assert jobs[0].box == 'full'
    assert jobs[0].phot_outdir == reduction / 'phot_0_full'


def test_discover_mosaic_phot_jobs_miri_only(tmp_path: Path):
    from st123.stages.photometry.dolphot import discover_mosaic_phot_jobs
    from st123.stages.mosaic.mosaic import write_dolphot_frame_list

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
    from st123.stages.mosaic.mosaic import write_dolphot_frame_list
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
        'st123.stages.photometry.dolphot.prepare_mosaic_phot_job',
        return_value=reduction / 'phot_0_0' / 'dolphot.param',
    ) as prep:
        # Mosaic discovery is the default; --from-mosaic is optional/legacy.
        rc = prep_script.main(
            [
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


@mock.patch('st123.stages.photometry.dolphot.run_logged_subprocess')
def test_prepare_mosaic_phot_job_miri_writes_recommended_params(
    mock_run, tmp_path: Path
):
    """MIRI mosaic prep must stage frames and write dolphotMIRI defaults."""
    from st123.stages.photometry.dolphot import MosaicPhotJob, prepare_mosaic_phot_job
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
    from st123.scripts.utils.options import resolve_photometry_instrument

    parser = prep_script.create_parser()
    args = parser.parse_args(
        ['--instrument', 'miri', '--base-dir', '/tmp/x', '--ncores', '32']
    )
    assert args.instruments == ['miri']
    assert resolve_photometry_instrument(args.instruments) == 'miri'
    assert args.from_mosaic is False  # flag unused; discovery is default
    assert prep_script.use_mosaic_discovery(args) is True
    assert args.ncores == 32
    assert args.ref_filter is None  # runtime default F560W applied in main
    assert args.group is None
    assert args.box is None


def test_dolphot_prep_parser_group_box():
    from st123.scripts import dolphot as prep_script

    parser = prep_script.create_parser()
    args = parser.parse_args(
        [
            '--base-dir',
            '/tmp/x',
            '--instruments',
            'nircam',
            '--from-mosaic',
            '--group',
            '0',
            '--box',
            'sn',
            '--ref-filter',
            'F200W',
        ]
    )
    assert args.group == 0
    assert args.box == 'sn'
    assert args.ref_filter == 'F200W'

def test_use_mosaic_discovery_opt_out_with_refimage_or_files():
    from st123.scripts import dolphot as prep_script

    parser = prep_script.create_parser()
    base = parser.parse_args(['--base-dir', '/tmp/x'])
    assert prep_script.use_mosaic_discovery(base) is True

    with_ref = parser.parse_args(
        ['--base-dir', '/tmp/x', '--refimage', '/tmp/x/ref.fits']
    )
    assert prep_script.use_mosaic_discovery(with_ref) is False

    with_files = parser.parse_args(
        ['--base-dir', '/tmp/x', '--files', '/tmp/x/a_jhat.fits']
    )
    assert prep_script.use_mosaic_discovery(with_files) is False


def test_dolphot_prep_instrument_case_insensitive():
    from st123.scripts import dolphot as prep_script
    from st123.scripts.utils.options import resolve_photometry_instrument

    parser = prep_script.create_parser()
    args = parser.parse_args(
        [
            '--from-mosaic',
            '--base-dir',
            '/tmp/p',
            '--instrument',
            'NIRCAM',
            '--ncores',
            '8',
        ]
    )
    assert args.instruments == ['NIRCAM']
    assert resolve_photometry_instrument(args.instruments) == 'nircam'
    assert prep_script.use_mosaic_discovery(args) is True
    assert args.ncores == 8


def test_dolphot_from_mosaic_cli_miri(tmp_path: Path):
    from st123.stages.mosaic.mosaic import write_dolphot_frame_list
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
        'st123.stages.photometry.dolphot.prepare_mosaic_phot_job',
        return_value=project / 'dolphot' / 'miri_0_0' / 'dolphot.param',
    ) as prep:
        rc = prep_script.main(
            [
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
    assert prep.call_args.kwargs['ncores'] == 32


@mock.patch('st123.stages.photometry.dolphot.run_logged_subprocess')
def test_calc_sky_parallel_invokes_all_frames(mock_run, tmp_path: Path):
    from st123.stages.photometry.dolphot import calc_sky

    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    frames = []
    for i in range(4):
        p = tmp_path / f'f{i}_jhat.chip1.fits'
        p.write_text('')
        frames.append(p)

    calc_sky(frames, instrument='wfc3', dolphot_bin=bin_dir, ncores=4)
    assert mock_run.call_count == 4
    stems = sorted(Path(c.args[0][1]).name for c in mock_run.call_args_list)
    assert stems == [f'f{i}_jhat.chip1' for i in range(4)]


def test_parallel_map_preserves_order_and_fanout():
    import threading
    import time

    from st123.stages.photometry.dolphot import _parallel_map

    active = 0
    peak = 0
    lock = threading.Lock()

    def work(i: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return i * 10

    out = _parallel_map(work, [0, 1, 2, 3], ncores=4, label='test')
    assert out == [0, 10, 20, 30]
    assert peak >= 2


@mock.patch('st123.stages.photometry.dolphot.sanitize_dolphot_wcs')
@mock.patch('st123.stages.photometry.dolphot.apply_hst_mask')
@mock.patch('st123.stages.photometry.dolphot.apply_splitgroups')
@mock.patch('st123.stages.photometry.dolphot.prepare_frames')
@mock.patch('st123.stages.photometry.dolphot.setup_paramfile')
def test_prepare_hst_frames_passes_ncores_and_stage_order(
    mock_setup,
    mock_prepare,
    mock_split,
    mock_mask,
    mock_sanitize,
    tmp_path: Path,
):
    """WFPC2 MEF mask -> splitgroups -> chip prepare_frames, all with ncores."""
    from st123.stages.photometry.dolphot import prepare_hst_frames

    src = tmp_path / 'src'
    out = tmp_path / 'dolphot' / 'hst_0_0'
    src.mkdir()
    mef = src / 'u_wfpc2_jhat.fits'
    mef.write_bytes(b'')
    ref = src / 'coadd_wfc3_f625w_drc.fits'
    ref.write_bytes(b'')
    chip = out / 'u_wfpc2_jhat.chip1.fits'
    param = out / 'dolphot.param'
    mock_setup.return_value = param

    call_order: list[str] = []

    def _mask_side_effect(files, instrument, **kwargs):
        call_order.append('mask')
        return None

    def _split_side_effect(files, **kwargs):
        call_order.append('split')
        chip.parent.mkdir(parents=True, exist_ok=True)
        chip.write_bytes(b'')
        return [chip]

    def _prepare_side_effect(*args, **kwargs):
        call_order.append('prep')
        return None

    mock_mask.side_effect = _mask_side_effect
    mock_split.side_effect = _split_side_effect
    mock_prepare.side_effect = _prepare_side_effect

    with mock.patch(
        'st123.stages.photometry.dolphot._classify_hst_science',
        return_value='wfpc2',
    ):
        prepare_hst_frames(
            [mef],
            out,
            'hst',
            refimage=ref,
            skip_sky=True,
            skip_mask=False,
            ncores=8,
        )

    assert call_order[:3] == ['mask', 'split', 'prep']
    assert mock_mask.call_args_list[0].kwargs.get('ncores') == 8
    assert mock_split.call_args.kwargs.get('ncores') == 8
    assert mock_prepare.call_args.kwargs.get('ncores') == 8
    # Empty stub FITS are skipped; sanitize is still invoked for real frames.


def test_chunk_images_equal_and_grouped():
    from st123.stages.photometry.dolphot_split import chunk_images

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
    from st123.stages.photometry.dolphot_split import (
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

    # finalize_split_outdir should no-op once merged exists with size>0... rewrite empty
    (tmp_path / 'run_nircam_miri.phot').unlink()
    out = finalize_split_outdir(tmp_path)
    assert out is not None
    assert out.name == 'run_nircam_miri.phot'


def test_setup_paramfile_respects_max_nimg(tmp_path: Path):
    from st123.stages.photometry.dolphot import setup_paramfile
    from st123.stages.photometry.dolphot_split import SPLIT_MANIFEST_NAME

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


def test_sanitize_dolphot_wcs_strips_lookup_and_keeps_sip(tmp_path: Path):
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    from st123.stages.photometry.dolphot import sanitize_dolphot_wcs

    data = np.zeros((64, 64), dtype=np.float32)
    hdr = fits.Header()
    hdr['SIMPLE'] = True
    hdr['BITPIX'] = -32
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = 64
    hdr['NAXIS2'] = 64
    hdr['CTYPE1'] = 'RA---TAN-SIP'
    hdr['CTYPE2'] = 'DEC--TAN-SIP'
    hdr['CRPIX1'] = 32.0
    hdr['CRPIX2'] = 32.0
    hdr['CRVAL1'] = 180.0
    hdr['CRVAL2'] = 0.0
    hdr['CD1_1'] = -1.0e-5
    hdr['CD1_2'] = 0.0
    hdr['CD2_1'] = 0.0
    hdr['CD2_2'] = 1.0e-5
    hdr['A_ORDER'] = 2
    hdr['B_ORDER'] = 2
    hdr['A_2_0'] = 1.0e-7
    hdr['A_0_2'] = -1.0e-7
    hdr['A_1_1'] = 0.0
    hdr['B_2_0'] = -1.0e-7
    hdr['B_0_2'] = 1.0e-7
    hdr['B_1_1'] = 0.0
    # Orphaned Lookup distortion (no WCSDVARR HDU).
    hdr['CPDIS1'] = 'Lookup'
    hdr['CPDIS2'] = 'Lookup'
    hdr['D2IMEXT'] = 'iref$dummy_d2i.fits'
    path = tmp_path / 'chip_sip.fits'
    fits.PrimaryHDU(data=data, header=hdr).writeto(path)

    assert sanitize_dolphot_wcs(path) is True
    out = fits.getheader(path)
    assert out['CTYPE1'] == 'RA---TAN-SIP'
    assert 'CPDIS1' not in out
    assert 'D2IMEXT' not in out
    assert out.get('A_ORDER') == 2
    assert abs(float(out['A_2_0']) - 1.0e-7) < 1e-20
    # Round-trip still works under a SIP-only WCS (JHAT forward SIP preserved).
    w = WCS(out, relax=True)
    sky = w.pixel_to_world(31.0, 31.0)  # astropy 0-index; CRPIX=32
    assert abs(sky.ra.deg - 180.0) < 1e-8


def test_is_jwst_dolphot_reference_by_name_and_header(tmp_path: Path):
    from astropy.io import fits

    from st123.stages.photometry.dolphot import _is_jwst_dolphot_reference

    i2d = tmp_path / 'coadd_0_0_f200w_i2d.fits'
    i2d.write_bytes(b'')
    assert _is_jwst_dolphot_reference(i2d) is True

    hst = tmp_path / 'coadd_wfc3_f625w_drc.fits'
    hst.write_bytes(b'')
    assert _is_jwst_dolphot_reference(hst) is False

    nircam = tmp_path / 'ref_custom.fits'
    hdr = fits.Header()
    hdr['TELESCOP'] = 'JWST'
    hdr['INSTRUME'] = 'NIRCAM'
    fits.PrimaryHDU(data=[[1.0]], header=hdr).writeto(nircam)
    assert _is_jwst_dolphot_reference(nircam) is True


@mock.patch('st123.stages.photometry.dolphot.sanitize_dolphot_wcs')
@mock.patch('st123.stages.photometry.dolphot.apply_hst_mask')
@mock.patch('st123.stages.photometry.dolphot.apply_splitgroups')
@mock.patch('st123.stages.photometry.dolphot.prepare_frames')
@mock.patch('st123.stages.photometry.dolphot.calc_sky')
@mock.patch('st123.stages.photometry.dolphot.setup_paramfile')
def test_prepare_hst_frames_jwst_ref_skips_hst_mask_and_passes_xyt(
    mock_setup,
    mock_calc_sky,
    mock_prepare,
    mock_split,
    mock_mask,
    mock_sanitize,
    tmp_path: Path,
):
    """NIRCam img0 must not be HST-masked; xytfile forwarded to setup_paramfile."""
    from st123.stages.photometry.dolphot import prepare_hst_frames

    src = tmp_path / 'src'
    out = tmp_path / 'dolphot' / 'nircam_hst_0_0'
    src.mkdir()
    mef = src / 'u_wfpc2_jhat.fits'
    mef.write_bytes(b'')
    ref = src / 'coadd_0_0_f200w_i2d.fits'
    ref.write_bytes(b'')
    sky = src / 'coadd_0_0_f200w_i2d.sky.fits'
    sky.write_bytes(b'')
    xyt = out / 'warmstart.xyt'
    chip = out / 'u_wfpc2_jhat.chip1.fits'
    param = out / 'dolphot.param'
    mock_setup.return_value = param

    def _split_side_effect(files, **kwargs):
        chip.parent.mkdir(parents=True, exist_ok=True)
        chip.write_bytes(b'')
        return [chip]

    mock_split.side_effect = _split_side_effect

    with mock.patch(
        'st123.stages.photometry.dolphot._classify_hst_science',
        return_value='wfpc2',
    ):
        prepare_hst_frames(
            [mef],
            out,
            'hst',
            refimage=ref,
            skip_sky=False,
            skip_mask=False,
            ncores=2,
            xytfile=xyt,
        )

    # Science MEF may be masked; reference must never get HST mask/calcsky.
    for call in mock_mask.call_args_list:
        paths = [Path(p).name for p in call.args[0]]
        assert 'coadd_0_0_f200w_i2d.fits' not in paths
    mock_calc_sky.assert_not_called()
    assert (out / sky.name).is_file()
    assert mock_setup.call_args.kwargs.get('xytfile') == xyt


def test_setup_hst_warmstart_writes_xyt_and_hst_globals(tmp_path: Path):
    from astropy.io import fits

    from st123.stages.photometry.warmstart import setup_hst_warmstart
    from st123.utils.settings import hst_base_params

    nircam = tmp_path / 'phot_0_0'
    nircam.mkdir()
    ref = nircam / 'coadd_0_0_f200w_i2d.fits'
    hdr = fits.Header()
    hdr['TELESCOP'] = 'JWST'
    hdr['INSTRUME'] = 'NIRCAM'
    hdr['CTYPE1'] = 'RA---TAN'
    hdr['CTYPE2'] = 'DEC--TAN'
    hdr['CRVAL1'] = 180.0
    hdr['CRVAL2'] = 0.0
    hdr['CRPIX1'] = 2.0
    hdr['CRPIX2'] = 2.0
    hdr['CD1_1'] = -1.0e-5
    hdr['CD1_2'] = 0.0
    hdr['CD2_1'] = 0.0
    hdr['CD2_2'] = 1.0e-5
    fits.PrimaryHDU(data=[[1.0, 1.0], [1.0, 1.0]], header=hdr).writeto(ref)
    (nircam / 'coadd_0_0_f200w_i2d.sky.fits').write_bytes(b'')
    (nircam / 'dolphot.param').write_text(
        'Nimg = 1\n'
        'img0_file = coadd_0_0_f200w_i2d\n'
        'img1_file = dummy_nrca1_jhat\n'
    )
    (nircam / 'dummy_nrca1_jhat.fits').write_bytes(b'')
    (nircam / 'run.phot').write_text(
        '0 1 10.0 20.0 1.0 50.0 0.01 0.0 0.0 0.1 1 1\n'
        '0 1 12.0 22.0 1.0 40.0 0.02 0.0 0.0 0.2 1 1\n'
    )

    hst_jhat = tmp_path / 'u_test_jhat.fits'
    hst_jhat.write_bytes(b'')
    out = tmp_path / 'nircam_hst_0_0'

    with mock.patch(
        'st123.stages.photometry.warmstart.prepare_hst_frames'
    ) as mock_prep:
        param = out / 'dolphot.param'
        param.parent.mkdir(parents=True, exist_ok=True)

        def _prep_side_effect(*args, **kwargs):
            out.mkdir(parents=True, exist_ok=True)
            text = (
                'Nimg = 1\n'
                'img0_file = coadd_0_0_f200w_i2d\n'
                'img1_file = u_test_jhat.chip1\n'
                f'UseWCS = {hst_base_params["UseWCS"]}\n'
                f'Align = {hst_base_params["Align"]}\n'
                f'Force1 = {hst_base_params["Force1"]}\n'
                f'PSFres = {hst_base_params["PSFres"]}\n'
                'xytfile = warmstart.xyt\n'
            )
            param.write_text(text)
            (out / 'u_test_jhat.chip1.fits').write_bytes(b'')
            return param

        mock_prep.side_effect = _prep_side_effect
        result = setup_hst_warmstart(
            nircam,
            out,
            hst_jhat=[hst_jhat],
            prune_xyt_for_hst=True,
            ncores=2,
        )

    assert result.xyt_file.is_file()
    assert result.xyt_file.read_text().count('\n') >= 1
    text = result.param_file.read_text()
    assert 'xytfile = warmstart.xyt' in text
    assert 'UseWCS = 2' in text
    assert 'Align = 0' in text
    assert 'Force1 = 1' in text
    assert 'PSFres = 0' in text
    assert mock_prep.call_args.kwargs.get('xytfile') == result.xyt_file
    assert mock_prep.call_args.kwargs.get('refimage').name == ref.name


def test_discover_hst_jhat(tmp_path: Path):
    from astropy.io import fits
    import numpy as np

    from st123.stages.photometry.warmstart import discover_hst_jhat

    jhat_dir = tmp_path / 'reduction' / 'jhat_hst'
    jhat_dir.mkdir(parents=True)

    def _write(name: str, instrument: str) -> Path:
        path = jhat_dir / name
        primary = fits.PrimaryHDU()
        primary.header['INSTRUME'] = instrument
        primary.header['TELESCOP'] = 'HST'
        sci = fits.ImageHDU(np.ones((4, 4), dtype=np.float32), name='SCI')
        fits.HDUList([primary, sci]).writeto(path)
        return path

    acs = _write('j9acs_jhat.fits', 'ACS')
    wfc3 = _write('iewfc3_jhat.fits', 'WFC3')
    wfpc2 = _write('u2wfpc2_jhat.fits', 'WFPC2')
    jw = jhat_dir / 'jw012345_jhat.fits'
    jw.write_bytes(b'')

    found = discover_hst_jhat(tmp_path)
    assert acs.resolve() in found
    assert wfc3.resolve() in found
    assert wfpc2.resolve() in found
    assert jw.resolve() not in found

    filtered = discover_hst_jhat(tmp_path, instruments=['ACS', 'WFC3'])
    assert {p.resolve() for p in filtered} == {acs.resolve(), wfc3.resolve()}
