"""Regression for issue #17: never put unsplit JHAT MEFs in dolphot.param."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
from astropy.io import fits

import pytest

from st123.scripts.dolphot import _collect_hst_frames, _pick_hst_reference
from st123.stages.photometry.dolphot import (
    _dedupe_paths_by_resolve,
    _is_dolphot_chip_product,
    _is_unsplit_hst_jhat,
    apply_splitgroups,
    prepare_hst_frames,
)


def _write_acs_mef(path: Path, shape: tuple[int, int] = (512, 512)) -> Path:
    sci = np.ones(shape, dtype=np.float32)
    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = 'ACS'
    primary.header['FILTER1'] = 'F814W'
    primary.header['EXPTIME'] = 100.0
    primary.header['EXPFLAG'] = 'NORMAL'
    hdul = fits.HDUList(
        [
            primary,
            fits.ImageHDU(data=sci, name='SCI'),
            fits.ImageHDU(data=sci, name='ERR'),
            fits.ImageHDU(data=np.zeros(sci.shape, dtype=np.int16), name='DQ'),
        ]
    )
    hdul.writeto(path, overwrite=True)
    return path


def test_dedupe_paths_follows_symlink_jhat_dirs(tmp_path: Path):
    real = tmp_path / 'jhat_hst'
    real.mkdir()
    frame = real / 'abc_jhat.fits'
    frame.write_bytes(b'x')
    link = tmp_path / 'jhat'
    link.symlink_to(real)
    dupes = [link / 'abc_jhat.fits', real / 'abc_jhat.fits']
    kept = _dedupe_paths_by_resolve(dupes)
    assert len(kept) == 1
    assert kept[0].resolve() == frame.resolve()


def test_collect_hst_frames_dedupes_symlink_jhat(tmp_path: Path):
    reduction = tmp_path / 'reduction'
    jhat_hst = reduction / 'jhat_hst'
    jhat_hst.mkdir(parents=True)
    _write_acs_mef(jhat_hst / 'abc_jhat.fits')
    (reduction / 'jhat').symlink_to(jhat_hst)
    frames = _collect_hst_frames(reduction, 'hst')
    assert len(frames) == 1
    assert frames[0].name == 'abc_jhat.fits'


def test_is_dolphot_chip_product():
    assert _is_dolphot_chip_product('foo_jhat.chip1.fits')
    assert _is_dolphot_chip_product('foo_jhat.chip4.fits')
    assert not _is_dolphot_chip_product('foo_jhat.fits')
    assert not _is_dolphot_chip_product('coadd_acs_f814w_drc.fits')


def test_is_unsplit_hst_jhat():
    assert _is_unsplit_hst_jhat('jey322pnq_jhat.fits')
    assert not _is_unsplit_hst_jhat('jey322pnq_jhat.chip1.fits')
    assert not _is_unsplit_hst_jhat('coadd_0_0_acs_f555w_drc.fits')


def test_pick_hst_reference_refuses_jhat_fallback(tmp_path: Path):
    reduction = tmp_path / 'reduction'
    jhat = reduction / 'jhat_hst'
    jhat.mkdir(parents=True)
    mef = jhat / 'abc_jhat.fits'
    mef.write_bytes(b'x')
    with pytest.raises(FileNotFoundError, match='refusing JHAT'):
        _pick_hst_reference(reduction, [mef], instrument='hst')


def test_prepare_hst_frames_refuses_jhat_img0(tmp_path: Path):
    mef = _write_acs_mef(tmp_path / 'abc_jhat.fits')
    outdir = tmp_path / 'dolphot'
    outdir.mkdir()
    with pytest.raises(ValueError, match='refusing unsplit JHAT as img0'):
        prepare_hst_frames(
            [mef],
            outdir,
            'hst',
            refimage=mef,
            skip_mask=True,
            skip_sky=True,
            skip_split=True,
            ncores=1,
        )


def test_apply_splitgroups_never_returns_unsplit_mef(tmp_path: Path):
    mef = _write_acs_mef(tmp_path / 'abc_jhat.fits')

    def _fake_one(src: Path):
        # Pretend splitgroups failed after a racing sibling deleted chips.
        return []

    with mock.patch(
        'st123.stages.photometry.dolphot.resolve_dolphot_bin',
        return_value=tmp_path,
    ), mock.patch(
        'st123.stages.photometry.dolphot.shutil.which',
        return_value=str(tmp_path / 'splitgroups'),
    ), mock.patch(
        'st123.stages.photometry.dolphot.run_logged_subprocess',
    ), mock.patch(
        'st123.stages.photometry.dolphot._parallel_map',
        side_effect=lambda fn, paths, **kw: [_fake_one(p) for p in paths],
    ):
        (tmp_path / 'splitgroups').write_text('#!/bin/sh\n')
        out = apply_splitgroups([mef, mef])
    assert out == []
    assert all(_is_dolphot_chip_product(p) for p in out)


def test_prepare_hst_frames_rejects_duplicate_unsplit_leak(tmp_path: Path):
    """Duplicate MEF paths must not put unsplit JHAT into dolphot.param."""
    outdir = tmp_path / 'dolphot' / 'hst_0_0'
    outdir.mkdir(parents=True)
    mef = _write_acs_mef(tmp_path / 'abc_jhat.fits')
    ref = tmp_path / 'coadd_0_0_acs_f814w_drc.fits'
    fits.PrimaryHDU(np.ones((64, 64), dtype=np.float32)).writeto(ref)

    chip1 = outdir / 'abc_jhat.chip1.fits'
    chip4 = outdir / 'abc_jhat.chip4.fits'

    def _fake_split(files, **kwargs):
        # Simulate a race that would previously return chips + unsplit MEF.
        for chip in (chip1, chip4):
            hdu = fits.PrimaryHDU(np.ones((512, 512), dtype=np.float32))
            hdu.header['EXTNAME'] = 'SCI'
            hdu.header['INSTRUME'] = 'ACS'
            hdu.writeto(chip, overwrite=True)
        return [chip1, Path(files[0]), chip4]

    with mock.patch(
        'st123.stages.photometry.dolphot.apply_splitgroups',
        side_effect=_fake_split,
    ), mock.patch(
        'st123.stages.photometry.dolphot.prepare_frames',
    ), mock.patch(
        'st123.stages.photometry.dolphot.sanitize_dolphot_wcs',
    ), mock.patch(
        'st123.stages.photometry.dolphot.apply_hst_mask',
        side_effect=lambda paths, *a, **k: list(paths),
    ), mock.patch(
        'st123.stages.photometry.dolphot.calc_sky',
    ), mock.patch(
        'st123.stages.photometry.dolphot._flatten_dolphot_ref_and_sky',
    ), mock.patch(
        'st123.datamodels.hst.filter_paths_for_stage',
        side_effect=lambda paths, **k: list(paths),
    ), mock.patch(
        'st123.datamodels.hst.restore_hst_identity_cards',
    ):
        param = prepare_hst_frames(
            [mef, mef],
            outdir,
            'hst',
            refimage=ref,
            skip_mask=True,
            skip_sky=True,
            ncores=1,
        )
    text = param.read_text()
    assert 'abc_jhat.chip1' in text
    assert 'abc_jhat.chip4' in text
    sci_lines = [
        line
        for line in text.splitlines()
        if line.startswith('img') and '_file' in line and not line.startswith('img0_')
    ]
    assert sci_lines
    for line in sci_lines:
        assert '.chip' in line, line
        assert not line.rstrip().endswith('jhat'), line
