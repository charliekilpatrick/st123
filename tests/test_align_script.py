"""Tests for align script helpers used by st123.scripts.align."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from st123.alignment.align import (
    create_dirs,
    get_input_images,
    pick_deepest_image,
    run_jhat,
    visit_filter_dict,
)
from st123.scripts import align as align_script


def test_create_dirs(tmp_path: Path):
    out = create_dirs(str(tmp_path))
    assert Path(out).is_dir()
    assert (tmp_path / 'align').is_dir()
    assert (tmp_path / 'reference').is_dir()
    assert (tmp_path / 'jhat').is_dir()
    # No extra object subdirectory under the reduction root
    assert not (tmp_path / 'm92').exists()


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
    args = parser.parse_args(['--workdir', '/tmp/w', '--ncores', '2'])
    assert args.base_dir == '/tmp/w'
    assert args.ncores == 2
    assert args.mode == 'visit'
    assert args.instrument is None

    ref = parser.parse_args(
        [
            '--base-dir',
            '/tmp/d',
            '--mode',
            'reference',
            '--instrument',
            'MIRI',
            '--ncores',
            '4',
        ]
    )
    assert ref.mode == 'reference'
    assert ref.instrument == 'MIRI'
    assert ref.ncores == 4
    # Removed from the unified CLI.
    with pytest.raises(SystemExit):
        parser.parse_args(['--continue-on-error'])


def test_run_jhat_passes_absolute_outrootdir(tmp_path: Path):
    """JHAT must receive outrootdir=abs(outdir), not outsubdir=abs path."""
    outdir = tmp_path / 'align' / 'group_0' / 'visit_0'
    outdir.mkdir(parents=True)
    phot = tmp_path / 'ref.phot.txt'
    phot.write_text('ra dec\n')

    with patch('st123.alignment.align.st_wcs_align') as mock_cls:
        instance = mock_cls.return_value
        run_jhat(
            align_image=str(tmp_path / 'img_cal.fits'),
            outdir=str(outdir),
            params={},
            gaia=False,
            photfilename=str(phot),
        )
        kwargs = instance.run_all.call_args.kwargs
        assert kwargs['outrootdir'] == str(outdir.resolve())
        assert 'outsubdir' not in kwargs or kwargs.get('outsubdir') in (None, '')


def test_align_main_empty_workdir(tmp_path: Path):
    """Visit-mode align with no images should create dirs and exit cleanly."""
    empty = Table({'group': np.array([], dtype=int), 'visit': np.array([], dtype='U8')})
    with (
        patch('st123.scripts.align.get_input_images', return_value=[]),
        patch('st123.scripts.align.input_list', return_value=empty),
        patch('st123.scripts.align.visit_filter_dict', return_value={}),
    ):
        rc = align_script.main(
            ['--workdir', str(tmp_path), '--ncores', '1']
        )
    assert rc == 0
    assert (tmp_path / 'align').is_dir()


def test_align_main_visit_mode_explicit(tmp_path: Path):
    empty = Table({'group': np.array([], dtype=int), 'visit': np.array([], dtype='U8')})
    with (
        patch('st123.scripts.align.get_input_images', return_value=[]) as get_imgs,
        patch('st123.scripts.align.input_list', return_value=empty),
        patch('st123.scripts.align.visit_filter_dict', return_value={}),
    ):
        rc = align_script.main(
            [
                '--base-dir',
                str(tmp_path),
                '--mode',
                'visit',
                '--instrument',
                'NIRCAM',
                '--ncores',
                '1',
            ]
        )
    assert rc == 0
    patterns = get_imgs.call_args.kwargs.get('pattern') or get_imgs.call_args[0][0]
    assert any('nrca' in p for p in patterns)


def test_resolve_instrument_defaults():
    assert align_script._resolve_instrument('visit', None) == 'NIRCAM'
    assert align_script._resolve_instrument('reference', None) == 'MIRI'
    assert align_script._resolve_instrument('visit', 'miri') == 'MIRI'


def test_run_visit_alignment_nonzero_on_worker_failure(tmp_path: Path):
    """Visit mode exits 1 when pooled JHAT workers report failures."""
    table = Table(
        {
            'group': [0],
            'visit': ['v1'],
            'filter': ['F200W'],
            'image': [str(tmp_path / 'a_cal.fits')],
            'pupil': ['CLEAR'],
        }
    )
    filter_table = {'F200W': table}

    with (
        patch('st123.scripts.align.get_input_images', return_value=['a.fits']),
        patch('st123.scripts.align.input_list', return_value=table),
        patch('st123.scripts.align.visit_filter_dict', return_value={'v1': 'F200W'}),
        patch('st123.scripts.align.get_visit_geoms', return_value={'v1': object()}),
        patch('st123.scripts.align.pick_visit', return_value=('v1', 0.5)),
        patch('st123.scripts.align.create_filter_table', return_value=filter_table),
        patch(
            'st123.scripts.align.create_alignment_mosaic',
            return_value=('mosaic.fits', (0.0, 0.0), 1),
        ),
        patch('st123.scripts.align.fix_phot', return_value='mosaic.phot.txt'),
        patch('st123.scripts.align.update_refcat', return_value=None),
        patch('st123.scripts.align.align_to_mosaic', return_value=0),
    ):
        rc = align_script.run_visit_alignment(
            base_dir=tmp_path,
            instrument='NIRCAM',
            ncores=1,
            verbose=False,
        )
    assert rc == 1


def test_run_visit_alignment_zero_when_workers_ok(tmp_path: Path):
    table = Table(
        {
            'group': [0],
            'visit': ['v1'],
            'filter': ['F200W'],
            'image': [str(tmp_path / 'a_cal.fits')],
            'pupil': ['CLEAR'],
        }
    )
    filter_table = {'F200W': table}

    with (
        patch('st123.scripts.align.get_input_images', return_value=['a.fits']),
        patch('st123.scripts.align.input_list', return_value=table),
        patch('st123.scripts.align.visit_filter_dict', return_value={'v1': 'F200W'}),
        patch('st123.scripts.align.get_visit_geoms', return_value={'v1': object()}),
        patch('st123.scripts.align.pick_visit', return_value=('v1', 0.5)),
        patch('st123.scripts.align.create_filter_table', return_value=filter_table),
        patch(
            'st123.scripts.align.create_alignment_mosaic',
            return_value=('mosaic.fits', (0.0, 0.0), 0),
        ),
        patch('st123.scripts.align.fix_phot', return_value='mosaic.phot.txt'),
        patch('st123.scripts.align.update_refcat', return_value=None),
        patch('st123.scripts.align.align_to_mosaic', return_value=0),
    ):
        rc = align_script.run_visit_alignment(
            base_dir=tmp_path,
            instrument='NIRCAM',
            ncores=1,
            verbose=False,
        )
    assert rc == 0
