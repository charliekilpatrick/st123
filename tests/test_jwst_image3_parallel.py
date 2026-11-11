"""Tests for JWST per-filter Image3 parallelism."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from st123.stages.mosaic.mosaic import (
    _parallel_image3_budget,
    run_jwst_filter_image3_jobs,
)


def test_parallel_image3_budget():
    assert _parallel_image3_budget(4, 8) == 4
    assert _parallel_image3_budget(10, 8) == 8
    assert _parallel_image3_budget(1, 8) == 1
    assert _parallel_image3_budget(0, 8) == 1


def test_run_jwst_filter_image3_jobs_serial(monkeypatch):
    calls: list[str] = []

    def _worker(job):
        calls.append(job['filter_name'])
        return {
            'filter': job['filter_name'],
            'status': 'ok',
            'coadd': f"{job['filter_name']}_i2d.fits",
            'error': None,
        }

    monkeypatch.setattr(
        'st123.stages.mosaic.mosaic.jwst_filter_image3_worker', _worker
    )
    jobs = [
        {'filter_name': 'f150w'},
        {'filter_name': 'f200w'},
    ]
    out = run_jwst_filter_image3_jobs(jobs, ncores=1, parallel=True)
    assert [r['filter'] for r in out] == ['f150w', 'f200w']
    assert calls == ['f150w', 'f200w']


def test_run_jwst_filter_image3_jobs_uses_process_pool(monkeypatch):
    """ncores>1 with multiple filters submits to ProcessPoolExecutor."""
    submitted: list[dict] = []

    class _Fut:
        def __init__(self, job):
            self._job = job

        def result(self):
            return {
                'filter': self._job['filter_name'],
                'status': 'ok',
                'coadd': f"{self._job['filter_name']}_i2d.fits",
                'error': None,
            }

    class _Pool:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, job):
            del fn
            submitted.append(job)
            return _Fut(job)

    monkeypatch.setattr(
        'concurrent.futures.ProcessPoolExecutor', _Pool
    )
    monkeypatch.setattr(
        'concurrent.futures.as_completed',
        lambda futures: list(futures),
    )

    jobs = [
        {'filter_name': 'f150w'},
        {'filter_name': 'f200w'},
        {'filter_name': 'f444w'},
    ]
    out = run_jwst_filter_image3_jobs(jobs, ncores=8, parallel=True)
    assert len(submitted) == 3
    assert [r['filter'] for r in out] == ['f150w', 'f200w', 'f444w']
    assert all(r['status'] == 'ok' for r in out)


def test_run_forced_filter_coadds_passes_ncores(tmp_path: Path):
    from astropy.wcs import WCS
    from st123.scripts import mosaic as mosaic_script

    hdr = {
        'CRPIX1': 1.0,
        'CRPIX2': 1.0,
        'CRVAL1': 180.0,
        'CRVAL2': 0.0,
        'CD1_1': -1e-5,
        'CD1_2': 0.0,
        'CD2_1': 0.0,
        'CD2_2': 1e-5,
        'CTYPE1': 'RA---TAN',
        'CTYPE2': 'DEC--TAN',
        'NAXIS1': 16,
        'NAXIS2': 16,
    }
    w = WCS(hdr)
    w.pixel_shape = (16, 16)

    from astropy.table import Table

    filter_table = {
        'f150w': Table(
            {
                'image': [str(tmp_path / 'a_jhat.fits')],
                'instrument': ['nircam'],
                'filter': ['f150w'],
            }
        ),
        'f200w': Table(
            {
                'image': [str(tmp_path / 'b_jhat.fits')],
                'instrument': ['nircam'],
                'filter': ['f200w'],
            }
        ),
    }
    for name in ('a_jhat.fits', 'b_jhat.fits'):
        (tmp_path / name).write_bytes(b'x')

    seen: dict = {}

    def _fake_copy(ftable, outdir):
        del ftable, outdir

    def _fake_update(ftable, outdir):
        del outdir
        return ftable

    def _fake_run(jobs, *, ncores=1, parallel=True):
        seen['ncores'] = ncores
        seen['n_jobs'] = len(jobs)
        seen['parallel'] = parallel
        return [
            {
                'filter': j['filter_name'],
                'status': 'ok',
                'coadd': str(tmp_path / f"{j['filter_name']}_i2d.fits"),
                'error': None,
            }
            for j in jobs
        ]

    with (
        patch(
            'st123.stages.mosaic.mosaic.copy_files', side_effect=_fake_copy
        ),
        patch('st123.stages.mosaic.mosaic.update_path', side_effect=_fake_update),
        patch(
            'st123.stages.mosaic.mosaic.run_jwst_filter_image3_jobs',
            side_effect=_fake_run,
        ),
        patch(
            'st123.stages.mosaic.mosaic.write_dolphot_frame_list',
            return_value=str(tmp_path / 'dolphot_frames.txt'),
        ),
        patch(
            'st123.stages.mosaic.mosaic.unify_jwst_astrometric_frame',
            return_value={'ok': True, 'final_max_abs_arcsec': 0.0},
        ) as mock_unify,
    ):
        written = mosaic_script._run_forced_filter_coadds(
            filter_table=filter_table,
            box_wcs=w,
            box_outdir=str(tmp_path / 'box'),
            group_id=0,
            box_index=0,
            subimages=[str(tmp_path / 'a_jhat.fits')],
            base_dir_arg=str(tmp_path),
            verbose=False,
            ncores=8,
        )

    assert seen['ncores'] == 8
    assert seen['n_jobs'] == 2
    assert len(written) == 2
    assert mock_unify.call_args.kwargs.get('apply_shifts') is False
    assert mock_unify.call_args.kwargs.get('remosaic') is False
