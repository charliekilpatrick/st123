"""Tests for align script helpers used by st123.scripts.align."""

from __future__ import annotations

import argparse
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
    assert args.instruments is None

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
    assert ref.instruments == ['MIRI']
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
        patch('st123.alignment.align.get_input_images', return_value=[]),
        patch('st123.scripts.align.input_list', return_value=empty),
        patch('st123.alignment.align.visit_filter_dict', return_value={}),
        patch('st123.alignment.align.create_dirs', return_value=str(tmp_path)),
    ):
        rc = align_script.main(
            ['--workdir', str(tmp_path), '--ncores', '1']
        )
    assert rc == 0


def test_align_main_visit_mode_explicit(tmp_path: Path):
    empty = Table({'group': np.array([], dtype=int), 'visit': np.array([], dtype='U8')})
    with (
        patch(
            'st123.alignment.align.get_input_images', return_value=[]
        ) as get_imgs,
        patch('st123.scripts.align.input_list', return_value=empty),
        patch('st123.alignment.align.visit_filter_dict', return_value={}),
        patch('st123.alignment.align.create_dirs', return_value=str(tmp_path)),
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


def test_resolve_align_instruments_all_and_nircam():
    assert align_script.resolve_align_instruments(None) is None
    assert align_script.resolve_align_instruments(['NIRCAM']) == ['NIRCAM']
    assert align_script.resolve_align_instruments(['ALL']) == [
        'NIRCAM',
        'MIRI',
        'ACS',
        'WFC3',
        'WFPC2',
    ]
    assert align_script.resolve_align_instruments(['hst']) == [
        'ACS',
        'WFC3',
        'WFPC2',
    ]
    assert align_script.resolve_align_instruments(
        ['NIRCAM', 'MIRI', 'ACS', 'WFC3']
    ) == ['NIRCAM', 'MIRI', 'ACS', 'WFC3']
    assert align_script.resolve_align_instruments(
        None, instrument='nircam'
    ) == ['NIRCAM']


def test_needs_orchestration():
    assert not align_script.needs_orchestration(None)
    assert not align_script.needs_orchestration(['NIRCAM'])
    assert align_script.needs_orchestration(['NIRCAM', 'MIRI', 'ACS', 'WFC3'])
    assert align_script.needs_orchestration(['ACS', 'WFC3'])
    assert align_script.needs_orchestration(['MIRI'])


def test_telescope_equals_default_instruments():
    from st123.scripts.utils.options import (
        default_instruments_for_telescope,
        resolve_instruments_with_telescope,
    )
    from st123.utils.settings import (
        DEFAULT_HST_INSTRUMENTS,
        DEFAULT_JWST_INSTRUMENTS,
    )

    assert default_instruments_for_telescope('hst') == list(
        DEFAULT_HST_INSTRUMENTS
    )
    assert default_instruments_for_telescope('jwst') == list(
        DEFAULT_JWST_INSTRUMENTS
    )
    assert default_instruments_for_telescope(None) is None

    hst = resolve_instruments_with_telescope(
        None,
        'hst',
        resolve_instruments_fn=align_script.resolve_align_instruments,
    )
    jwst = resolve_instruments_with_telescope(
        None,
        'jwst',
        resolve_instruments_fn=align_script.resolve_align_instruments,
    )
    assert hst == ['ACS', 'WFC3', 'WFPC2']
    assert jwst == ['NIRCAM', 'MIRI']
    # Explicit --instruments wins over --telescope.
    assert resolve_instruments_with_telescope(
        ['ACS'],
        'hst',
        resolve_instruments_fn=align_script.resolve_align_instruments,
    ) == ['ACS']
    assert align_script.needs_orchestration(hst)
    assert align_script.needs_orchestration(jwst)


def test_align_parser_instruments_and_all():
    parser = align_script.create_parser()
    args = parser.parse_args(
        [
            '--base-dir',
            '/tmp/p',
            '--instruments',
            'NIRCAM',
            'MIRI',
            'ACS',
            'WFC3',
            '--ncores',
            '4',
            '--skip-intermediate-mosaic',
        ]
    )
    assert args.instruments == ['NIRCAM', 'MIRI', 'ACS', 'WFC3']
    assert args.skip_intermediate_mosaic is True
    assert args.nmax == 150

    args_all = parser.parse_args(
        ['--base-dir', '/tmp/p', '--instruments', 'ALL']
    )
    assert args_all.instruments == ['ALL']


def test_orchestrated_alignment_stage_order(tmp_path: Path):
    """NIRCam → HST → mosaic → MIRI, in that order."""
    calls: list[str] = []

    def _nircam(**kwargs):
        calls.append('nircam')
        return 0

    def _hst(**kwargs):
        calls.append('hst')
        assert kwargs.get('instruments') == ['ACS', 'WFC3']
        return 0

    def _mosaic(**kwargs):
        calls.append('mosaic')
        return 0

    def _miri(args):
        calls.append('miri')
        assert args.instrument == 'MIRI'
        assert args.mode == 'reference'
        return 0

    ns = argparse.Namespace(
        base_dir=str(tmp_path),
        ncores=2,
        verbose=False,
        skip_intermediate_mosaic=False,
        force_miri=False,
        nmax=150,
        filters=None,
        # reference-mode leftovers
        plot=False,
        nbright=800,
        match_radius=0.1,
        no_clip_footprint=False,
        no_refine=False,
        refine_sigma=2.0,
        refine_max_iter=5,
        no_filter_calibrators=False,
        no_fallback=False,
        max_nircam_dispersion_mas=None,
        min_ref_overlap_frac=0.02,
        overlap_only=False,
        align_only=False,
        overlap_json=None,
        overlap_outdir=None,
        limit=None,
        repo=None,
        data_root=None,
        legacy_overlap_file=None,
        legacy_outdir=None,
        _default_reference_data_dir=str(tmp_path),
    )

    with (
        patch.object(align_script, 'run_visit_alignment', side_effect=_nircam),
        patch.object(align_script, 'run_hst_visit_alignment', side_effect=_hst),
        patch.object(
            align_script, 'run_intermediate_nircam_mosaic', side_effect=_mosaic
        ),
        patch.object(align_script, 'run_reference_alignment', side_effect=_miri),
        patch.object(
            align_script,
            '_existing_reference_coadds',
            side_effect=[[], [str(tmp_path / 'coadd.fits')], [str(tmp_path / 'coadd.fits')]],
        ),
        patch(
            'st123.utils.jwst_coverage.count_jwst_frames_on_disk',
            return_value=(1, 1),
        ),
    ):
        # Import argparse into test module namespace for Namespace above
        rc = align_script.run_orchestrated_alignment(
            ns, ['NIRCAM', 'MIRI', 'ACS', 'WFC3']
        )
    assert rc == 0
    assert calls == ['nircam', 'hst', 'mosaic', 'miri']


def test_orchestrated_align_skips_miri_only_jwst(tmp_path: Path):
    """MIRI-only on disk → skip JWST stages; HST still runs."""
    import argparse

    calls: list[str] = []

    def _nircam(**kwargs):
        calls.append('nircam')
        return 0

    def _hst(**kwargs):
        calls.append('hst')
        return 0

    def _mosaic(**kwargs):
        calls.append('mosaic')
        return 0

    def _miri(args):
        calls.append('miri')
        return 0

    ns = argparse.Namespace(
        base_dir=str(tmp_path),
        ncores=1,
        verbose=False,
        skip_intermediate_mosaic=False,
        force_miri=False,
        nmax=150,
        filters=None,
        plot=False,
        nbright=800,
        match_radius=0.1,
        no_clip_footprint=False,
        no_refine=False,
        refine_sigma=2.0,
        refine_max_iter=5,
        no_filter_calibrators=False,
        no_fallback=False,
        max_nircam_dispersion_mas=None,
        min_ref_overlap_frac=0.02,
        overlap_only=False,
        align_only=False,
        overlap_json=None,
        overlap_outdir=None,
        limit=None,
        repo=None,
        data_root=None,
        legacy_overlap_file=None,
        legacy_outdir=None,
        _default_reference_data_dir=str(tmp_path),
    )

    with (
        patch.object(align_script, 'run_visit_alignment', side_effect=_nircam),
        patch.object(align_script, 'run_hst_visit_alignment', side_effect=_hst),
        patch.object(
            align_script, 'run_intermediate_nircam_mosaic', side_effect=_mosaic
        ),
        patch.object(align_script, 'run_reference_alignment', side_effect=_miri),
        patch(
            'st123.utils.jwst_coverage.count_jwst_frames_on_disk',
            return_value=(0, 4),
        ),
    ):
        rc = align_script.run_orchestrated_alignment(
            ns, ['NIRCAM', 'MIRI', 'ACS', 'WFC3']
        )
    assert rc == 0
    assert calls == ['hst']

    calls.clear()
    ns.force_miri = True
    with (
        patch.object(align_script, 'run_visit_alignment', side_effect=_nircam),
        patch.object(align_script, 'run_hst_visit_alignment', side_effect=_hst),
        patch.object(
            align_script, 'run_intermediate_nircam_mosaic', side_effect=_mosaic
        ),
        patch.object(align_script, 'run_reference_alignment', side_effect=_miri),
        patch.object(
            align_script,
            '_existing_reference_coadds',
            return_value=[str(tmp_path / 'coadd.fits')],
        ),
        patch(
            'st123.utils.jwst_coverage.count_jwst_frames_on_disk',
            return_value=(0, 4),
        ),
    ):
        rc = align_script.run_orchestrated_alignment(
            ns, ['NIRCAM', 'MIRI']
        )
    # Forced: do not soft-skip; NIRCam visit is attempted.
    assert 'nircam' in calls


def test_align_main_instruments_dispatches_orchestrator(tmp_path: Path):
    with patch.object(
        align_script, 'run_orchestrated_alignment', return_value=0
    ) as mock_orch:
        rc = align_script.main(
            [
                '--base-dir',
                str(tmp_path),
                '--instruments',
                'NIRCAM',
                'MIRI',
                'ACS',
                'WFC3',
                '--ncores',
                '1',
            ]
        )
    assert rc == 0
    mock_orch.assert_called_once()
    assert mock_orch.call_args.args[1] == ['NIRCAM', 'MIRI', 'ACS', 'WFC3']


def test_align_main_instruments_nircam_only_skips_orchestrator(tmp_path: Path):
    empty = Table({'group': np.array([], dtype=int), 'visit': np.array([], dtype='U8')})
    with (
        patch.object(align_script, 'run_orchestrated_alignment') as mock_orch,
        patch(
            'st123.alignment.align.get_input_images', return_value=[]
        ),
        patch('st123.scripts.align.input_list', return_value=empty),
        patch(
            'st123.alignment.align.visit_filter_dict', return_value={}
        ),
        patch('st123.alignment.align.create_dirs', return_value=str(tmp_path)),
    ):
        rc = align_script.main(
            [
                '--base-dir',
                str(tmp_path),
                '--instruments',
                'NIRCAM',
                '--ncores',
                '1',
            ]
        )
    assert rc == 0
    mock_orch.assert_not_called()


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
        patch('st123.alignment.align.get_input_images', return_value=['a.fits']),
        patch('st123.scripts.align.input_list', return_value=table),
        patch(
            'st123.alignment.align.visit_filter_dict', return_value={'v1': 'F200W'}
        ),
        patch(
            'st123.alignment.align.get_visit_geoms', return_value={'v1': object()}
        ),
        patch('st123.alignment.align.pick_visit', return_value=('v1', 0.5)),
        patch('st123.scripts.align.create_filter_table', return_value=filter_table),
        patch(
            'st123.alignment.align.create_alignment_mosaic',
            return_value=('mosaic.fits', (0.0, 0.0), 1),
        ),
        patch('st123.alignment.align.fix_phot', return_value='mosaic.phot.txt'),
        patch('st123.alignment.align.update_refcat', return_value=None),
        patch('st123.alignment.align.align_to_mosaic', return_value=0),
        patch('st123.alignment.align.create_dirs', return_value=str(tmp_path)),
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
        patch('st123.alignment.align.get_input_images', return_value=['a.fits']),
        patch('st123.scripts.align.input_list', return_value=table),
        patch(
            'st123.alignment.align.visit_filter_dict', return_value={'v1': 'F200W'}
        ),
        patch(
            'st123.alignment.align.get_visit_geoms', return_value={'v1': object()}
        ),
        patch('st123.alignment.align.pick_visit', return_value=('v1', 0.5)),
        patch('st123.scripts.align.create_filter_table', return_value=filter_table),
        patch(
            'st123.alignment.align.create_alignment_mosaic',
            return_value=('mosaic.fits', (0.0, 0.0), 0),
        ),
        patch('st123.alignment.align.fix_phot', return_value='mosaic.phot.txt'),
        patch('st123.alignment.align.update_refcat', return_value=None),
        patch('st123.alignment.align.align_to_mosaic', return_value=0),
        patch('st123.alignment.align.create_dirs', return_value=str(tmp_path)),
    ):
        rc = align_script.run_visit_alignment(
            base_dir=tmp_path,
            instrument='NIRCAM',
            ncores=1,
            verbose=False,
        )
    assert rc == 0
