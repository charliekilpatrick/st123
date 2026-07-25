"""Tests for align script helpers used by st123.scripts.align."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from astropy.table import Table

from st123.alignment.align import (
    create_dirs,
    get_input_images,
    pick_deepest_image,
    visit_filter_dict,
)
from st123.scripts import align as align_script


def test_create_dirs(tmp_path: Path):
    out = create_dirs(str(tmp_path), 'm92')
    assert Path(out).is_dir()
    assert (tmp_path / 'align').is_dir()
    assert (tmp_path / 'reference').is_dir()
    assert (tmp_path / 'm92').is_dir()


def test_get_input_images(tmp_path: Path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    (raw / 'jwst_nrca1_cal.fits').write_bytes(b'x')
    (raw / 'jwst_nrcb1_cal.fits').write_bytes(b'x')
    (raw / 'other.fits').write_bytes(b'x')
    images = get_input_images(workdir=str(tmp_path))
    assert len(images) == 2
    assert all('nrc' in Path(p).name for p in images)


def test_pick_deepest_image():
    table = Table(
        {
            'image': ['a.fits', 'b.fits'],
            'exptime': [100.0, 500.0],
        }
    )
    deepest = pick_deepest_image(table)
    assert deepest['image'] == 'b.fits'


def _fits_with_s_region(path: Path, offset: float = 0.0) -> Path:
    """Write a file whose S_REGION string matches align.visit_filter_dict parsing."""
    # visit_filter_dict splits on 'POLYGON ICRS  ' (two spaces) then reshape(4,2)
    ra0, dec0 = 150.0 + offset, 2.0
    coords = [
        ra0,
        dec0,
        ra0 + 0.01,
        dec0,
        ra0 + 0.01,
        dec0 + 0.01,
        ra0,
        dec0 + 0.01,
    ]
    region = 'POLYGON ICRS  ' + ' '.join(str(c) for c in coords)
    header = fits.Header()
    header['S_REGION'] = region
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.ones((10, 10), dtype=np.float32), header=header, name='SCI'),
        ]
    ).writeto(path, overwrite=True)
    return path


def test_visit_filter_dict(tmp_path: Path):
    image_small = _fits_with_s_region(tmp_path / 'small_footprint.fits', offset=0.0)
    # Larger polygon → should be preferred for alignment
    image_large = _fits_with_s_region(tmp_path / 'large_footprint.fits', offset=0.0)
    coords = [150.0, 2.0, 150.05, 2.0, 150.05, 2.05, 150.0, 2.05]
    region = 'POLYGON ICRS  ' + ' '.join(str(c) for c in coords)
    with fits.open(image_large, mode='update') as hdul:
        hdul['SCI'].header['S_REGION'] = region

    table = Table(
        {
            'visit': ['v1', 'v1'],
            'filter': ['F200W', 'F150W'],
            'image': [str(image_small), str(image_large)],
            'pupil': ['CLEAR', 'CLEAR'],
        }
    )
    result = visit_filter_dict(table)
    assert result['v1'] == 'F150W'


def test_align_parser():
    parser = align_script.create_parser()
    args = parser.parse_args(['--workdir', '/tmp/w', '--object', 'obj', '--ncores', '2'])
    assert args.workdir == '/tmp/w'
    assert args.object == 'obj'
    assert args.ncores == 2


def test_align_main_empty_workdir(tmp_path: Path):
    """Align main with no images should create dirs and exit cleanly (no groups)."""
    empty = Table({'group': np.array([], dtype=int), 'visit': np.array([], dtype='U8')})
    with (
        patch('st123.scripts.align.get_input_images', return_value=[]),
        patch('st123.scripts.align.input_list', return_value=empty),
        patch('st123.scripts.align.visit_filter_dict', return_value={}),
    ):
        rc = align_script.main(
            ['--workdir', str(tmp_path), '--object', 'testobj', '--ncores', '1']
        )
    assert rc == 0
    assert (tmp_path / 'align').is_dir()
