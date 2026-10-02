"""Tests for relative-alignment library helpers and align pair-mode CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from astropy.table import Table

from helpers import make_wcs_header
from st123.stages.alignment import align as align_lib
from st123.scripts import align as align_script
from st123.scripts.utils.options import add_pair_align_args, create_parser as build_parser


def test_phot_catalog_path_and_resolve_outdir(tmp_path: Path):
    assert align_lib.phot_catalog_path('/a/b/c_i2d.fits', str(tmp_path)).endswith('c_i2d.phot.txt')
    out = align_lib.resolve_outdir(str(tmp_path / 'align_out'))
    assert Path(out).is_dir()


def test_reference_align_job_alias():
    """Historical NIRCam worker name remains an alias of the REFERENCE worker."""
    assert align_lib.run_nircam_align_job is align_lib.run_reference_align_job


def test_has_jwst_gwcs(tmp_path: Path):
    plain = tmp_path / 'plain.fits'
    fits.PrimaryHDU(np.ones((10, 10))).writeto(plain)
    assert align_lib.has_jwst_gwcs(str(plain)) is False

    with_asdf = tmp_path / 'with_asdf.fits'
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(name='ASDF')]
    ).writeto(with_asdf)
    assert align_lib.has_jwst_gwcs(str(with_asdf)) is True


def test_is_level3_i2d():
    assert align_lib.is_level3_i2d('coadd_0_0_f150w2_i2d.fits')
    assert align_lib.is_level3_i2d('/tmp/foo_i2d.fits.gz')
    assert not align_lib.is_level3_i2d('jw_x_mirimage_cal.fits')


def test_build_ref_catalog_uses_fix_phot_for_i2d_even_with_asdf(tmp_path: Path):
    """Level-3 i2d references must use fix_phot, not native jwst_phot."""
    i2d = tmp_path / 'coadd_0_0_f150w2_i2d.fits'
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.ones((8, 8), dtype=np.float32), name='SCI'),
            fits.ImageHDU(name='ASDF'),
        ]
    ).writeto(i2d)
    outdir = tmp_path / 'out'
    outdir.mkdir()
    fixed = tmp_path / 'coadd_0_0_f150w2_i2d.corr.phot.txt'
    fixed.write_text('ra dec mag dmag\n1 2 20 0.1\n')

    with (
        patch.object(align_lib, 'fix_phot', return_value=str(fixed)) as mock_fix,
        patch.object(align_lib, 'jwst_phot') as mock_jwst,
        patch.object(align_lib, 'photutils_phot') as mock_pu,
    ):
        staged = align_lib.build_ref_catalog(str(i2d), str(outdir))

    mock_fix.assert_called_once()
    assert mock_fix.call_args.args[0] == str(i2d)
    assert 'workdir' in mock_fix.call_args.kwargs
    mock_jwst.assert_not_called()
    mock_pu.assert_not_called()
    assert Path(staged).is_file()


def test_write_jhat_phot_table_and_stage(tmp_path: Path):
    table = Table({'ra': [1.0], 'dec': [2.0], 'mag': [20.0], 'dmag': [0.1]})
    phot = tmp_path / 'cat.phot.txt'
    written = align_lib.write_jhat_phot_table(table, str(phot))
    assert Path(written).is_file()
    outdir = tmp_path / 'out'
    outdir.mkdir()
    staged = align_lib.stage_photfile(written, str(outdir))
    assert Path(staged).is_file()


def test_load_sci_data_wcs(tmp_path: Path):
    path = tmp_path / 'sci.fits'
    data = np.ones((20, 20), dtype=np.float32)
    header = make_wcs_header(shape=data.shape)
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(data=data, header=header, name='SCI')]
    ).writeto(path)
    arr, wcs = align_lib.load_sci_data_wcs(str(path))
    assert arr.shape == (20, 20)
    assert wcs.wcs.crval[0] == 150.0


def test_add_pair_align_args_on_options_api():
    parser = build_parser('pair opts')
    add_pair_align_args(parser)
    args = parser.parse_args(
        ['--ref', 'r.fits', '--image', 'a.fits', '--photfile', 'c.phot.txt']
    )
    assert args.ref == 'r.fits'
    assert args.image == 'a.fits'
    assert args.photfile == 'c.phot.txt'


def test_align_pair_parser():
    parser = align_script.create_parser()
    args = parser.parse_args(
        [
            '--ref',
            'r.fits',
            '--image',
            'a.fits',
            '--photfile',
            'c.phot.txt',
            '--nbright',
            '100',
            '--outdir',
            '/tmp/out',
        ]
    )
    assert args.ref == 'r.fits'
    assert args.image == 'a.fits'
    assert args.photfile == 'c.phot.txt'
    assert args.nbright == 100
    assert args.base_dir == '/tmp/out'
    assert args.mode == 'visit'  # auto-resolved to pair in main()


def test_align_pair_main_missing_file(tmp_path: Path):
    rc = align_script.main(
        ['--ref', str(tmp_path / 'missing.fits'), '--image', str(tmp_path / 'also.fits')]
    )
    assert rc == 1


def test_align_pair_main_success(tmp_path: Path):
    ref = tmp_path / 'ref.fits'
    image = tmp_path / 'sci.fits'
    fits.PrimaryHDU(np.ones((5, 5))).writeto(ref)
    fits.PrimaryHDU(np.ones((5, 5))).writeto(image)
    with patch(
        'st123.stages.alignment.align.run_alignment',
        return_value=((0.1, -0.2), str(tmp_path / 'out')),
    ):
        rc = align_script.main(
            [
                '--ref',
                str(ref),
                '--image',
                str(image),
                '--outdir',
                str(tmp_path / 'out'),
            ]
        )
    assert rc == 0

def test_align_pair_mode_requires_both_paths():
    rc = align_script.main(['--mode', 'pair', '--ref', 'only_ref.fits'])
    assert rc == 2
