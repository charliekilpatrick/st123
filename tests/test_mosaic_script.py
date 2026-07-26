"""Tests for mosaic script helpers and required GWCS pipeline wiring."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from st123.mosaic.mosaic import (
    create_coadd_mosaic,
    create_dirs,
    create_gwcs,
    edit_spec_groups,
    ensure_dataset_reference_link,
    mp_init,
    update_path,
    write_dolphot_frame_list,
)
from st123.scripts import mosaic as mosaic_script


def test_mosaic_create_dirs(tmp_path: Path):
    out = create_dirs(str(tmp_path), n=2)
    assert set(out.keys()) == {0, 1}
    assert (tmp_path / 'reference' / 'group_0').is_dir()
    assert (tmp_path / 'reference' / 'group_1').is_dir()


def test_ensure_dataset_reference_link_from_reduction(tmp_path: Path):
    reduce = tmp_path / 'NGC3310' / 'reduction'
    reduce.mkdir(parents=True)
    linked = ensure_dataset_reference_link(str(reduce))
    assert linked is not None
    dataset_ref = tmp_path / 'NGC3310' / 'reference'
    assert dataset_ref.is_symlink()
    assert dataset_ref.resolve() == (reduce / 'reference').resolve()
    # Idempotent when already present.
    assert ensure_dataset_reference_link(str(reduce)) == str(dataset_ref)


def test_update_path():
    table = Table({'image': ['/old/a.fits', '/old/b.fits']})
    filter_table = {'F200W': table}
    updated = update_path(filter_table, '/new')
    assert list(updated['F200W']['image']) == ['/new/a.fits', '/new/b.fits']


def test_edit_spec_groups(tmp_path: Path):
    table = Table(
        {
            'image': ['/data/a.fits', '/data/b.fits', '/data/c.fits'],
            'group': [0, 0, 1],
        }
    )
    spec = tmp_path / 'spec.txt'
    # np.loadtxt with dtype=str on a single token can return a 0-d array;
    # write two names so iteration is stable.
    spec.write_text('a.fits\nb.fits\n')
    edited = edit_spec_groups(table, str(spec))
    assert edited['group'][0] == 2
    assert edited['group'][1] == 2
    assert edited['group'][2] == 1


def test_mp_init_sets_globals():
    mp_init(1, 2, ['a.fits'])
    import st123.mosaic.mosaic as mosaic_mod

    assert mosaic_mod.success == 1
    assert mosaic_mod.failed == 2
    assert mosaic_mod.success_files == ['a.fits']


def test_mosaic_parser():
    parser = mosaic_script.create_parser()
    args = parser.parse_args(
        ['--basedir', '/b', '--nmax', '10', '--ncores', '2']
    )
    assert args.nmax == 10
    assert args.base_dir == '/b'


def _sample_wcs_header() -> fits.Header:
    return fits.Header(
        {
            'CRPIX1': 100.0,
            'CRPIX2': 200.0,
            'CRVAL1': 160.0,
            'CRVAL2': 45.0,
            'CDELT1': -8.64e-6,
            'CDELT2': 8.64e-6,
            'PC1_1': 1.0,
            'PC1_2': 0.0,
            'PC2_1': 0.0,
            'PC2_2': 1.0,
            'NAXIS1': 300,
            'NAXIS2': 400,
        }
    )


def test_create_gwcs_supports_jwst_update_fits_wcsinfo(tmp_path: Path):
    """Custom output_wcs must expose .crpix for jwst 1.20 resample finalize."""
    from gwcs import FITSImagingWCSTransform
    from jwst.resample.resample import ResampleImage
    from stdatamodels.jwst import datamodels

    hdr = _sample_wcs_header()
    gwcs_path = create_gwcs(outdir=str(tmp_path), sci_header=hdr)
    assert Path(gwcs_path).is_file()

    wcsobj = create_gwcs(outdir=str(tmp_path), sci_header=hdr, return_gwcs=True)
    assert isinstance(wcsobj.forward_transform, FITSImagingWCSTransform)

    model = datamodels.ImageModel((10, 10))
    model.meta.wcs = wcsobj
    ResampleImage.update_fits_wcsinfo(model)
    assert model.meta.wcsinfo.crpix1 == 100.0
    assert model.meta.wcsinfo.crpix2 == 200.0
    assert model.meta.wcsinfo.crval1 == 160.0
    assert model.meta.wcsinfo.crval2 == 45.0


def test_create_coadd_mosaic_requires_gwcs_inputs(tmp_path: Path):
    table = Table({'image': ['a.fits']})
    with pytest.raises(ValueError, match='header or wcs'):
        create_coadd_mosaic(table, outdir=str(tmp_path), filt='f150w')


def test_create_coadd_mosaic_builds_and_sets_output_wcs(tmp_path: Path):
    table = Table({'image': [str(tmp_path / 'a.fits')]})
    hdr = _sample_wcs_header()
    image3 = MagicMock()
    image3.resample = MagicMock()
    image3.tweakreg = MagicMock()
    image3.skymatch = MagicMock()
    image3.source_catalog = MagicMock()

    with (
        patch('st123.mosaic.mosaic.patch_jwst_for_photutils3'),
        patch('st123.mosaic.mosaic.asn_from_list') as asn_mod,
        patch('st123.mosaic.mosaic.calwebb_image3.Image3Pipeline', return_value=image3),
    ):
        asn = MagicMock()
        asn.dump.return_value = ('name', '{}')
        asn_mod.asn_from_list.return_value = asn
        out = create_coadd_mosaic(
            table, outdir=str(tmp_path), filt='f150w', sci_header=hdr
        )

    gwcs_path = tmp_path / 'mosaic_gwcs.asdf'
    assert gwcs_path.is_file()
    assert image3.resample.output_wcs == str(gwcs_path)
    assert out == str(tmp_path / 'out_f150w' / 'f150w_i2d.fits')
    image3.run.assert_called_once()


def test_mosaic_main_no_jhat_files(tmp_path: Path):
    rc = mosaic_script.main(['--basedir', str(tmp_path), '--ncores', '1'])
    assert rc == 1


def test_write_dolphot_frame_list(tmp_path: Path):
    ref = tmp_path / 'coadd_0_0_f150w2_i2d.fits'
    ref.write_text('x')
    frames = [tmp_path / 'a_jhat.fits', tmp_path / 'b_jhat.fits']
    for path in frames:
        path.write_text('y')
    out = write_dolphot_frame_list(
        str(tmp_path),
        refimage=str(ref),
        frames=[str(p) for p in frames],
        group=0,
        box=0,
    )
    text = Path(out).read_text()
    assert 'group=0' in text and 'box=0' in text
    assert str(ref.resolve()) in text
    assert str(frames[0].resolve()) in text


def test_apply_gwcs_removed_from_scripts_package():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('st123.scripts.apply_gwcs')