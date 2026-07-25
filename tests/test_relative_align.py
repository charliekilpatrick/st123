"""Tests for relative-align script helpers and entry point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from astropy.table import Table

from helpers import make_wcs_header
from st123.scripts import relative_align as rel


def test_phot_catalog_path_and_resolve_outdir(tmp_path: Path):
    assert rel.phot_catalog_path('/a/b/c_i2d.fits', str(tmp_path)).endswith('c_i2d.phot.txt')
    out = rel.resolve_outdir(str(tmp_path / 'align_out'))
    assert Path(out).is_dir()


def test_has_jwst_gwcs(tmp_path: Path):
    plain = tmp_path / 'plain.fits'
    fits.PrimaryHDU(np.ones((10, 10))).writeto(plain)
    assert rel.has_jwst_gwcs(str(plain)) is False

    with_asdf = tmp_path / 'with_asdf.fits'
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(name='ASDF')]
    ).writeto(with_asdf)
    assert rel.has_jwst_gwcs(str(with_asdf)) is True


def test_write_jhat_phot_table_and_stage(tmp_path: Path):
    table = Table({'ra': [1.0], 'dec': [2.0], 'mag': [20.0], 'dmag': [0.1]})
    phot = tmp_path / 'cat.phot.txt'
    written = rel.write_jhat_phot_table(table, str(phot))
    assert Path(written).is_file()
    outdir = tmp_path / 'out'
    outdir.mkdir()
    staged = rel.stage_photfile(written, str(outdir))
    assert Path(staged).is_file()


def test_load_sci_data_wcs(tmp_path: Path):
    path = tmp_path / 'sci.fits'
    data = np.ones((20, 20), dtype=np.float32)
    header = make_wcs_header(shape=data.shape)
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(data=data, header=header, name='SCI')]
    ).writeto(path)
    arr, wcs = rel.load_sci_data_wcs(str(path))
    assert arr.shape == (20, 20)
    assert wcs.wcs.crval[0] == 150.0


def test_relative_align_main_missing_file(tmp_path: Path):
    rc = rel.main(
        ['--ref', str(tmp_path / 'missing.fits'), '--align', str(tmp_path / 'also.fits')]
    )
    assert rc == 1


def test_relative_align_main_success(tmp_path: Path):
    ref = tmp_path / 'ref.fits'
    align = tmp_path / 'align.fits'
    fits.PrimaryHDU(np.ones((5, 5))).writeto(ref)
    fits.PrimaryHDU(np.ones((5, 5))).writeto(align)
    with patch(
        'st123.alignment.relative_align.run_alignment',
        return_value=((0.1, -0.2), str(tmp_path / 'out')),
    ):
        rc = rel.main(
            ['--ref', str(ref), '--align', str(align), '--outdir', str(tmp_path / 'out')]
        )
    assert rc == 0


def test_relative_align_parser():
    parser = rel.create_parser()
    args = parser.parse_args(['--ref', 'r.fits', '--align', 'a.fits', '--nbright', '100'])
    assert args.nbright == 100
    assert args.plot is False
