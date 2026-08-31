"""Tests for HST JHAT / mosaic parallelization helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from astropy.io import fits

from st123.stages.alignment.hst_jhat import align_hst_raw_dir, _parallel_worker_budget
from st123.stages.mosaic.hst_drizzle import (
    _frame_sets_disjoint,
    _parallel_drizzle_budget,
    _run_filter_drizzle_jobs,
)


def test_parallel_budgets():
    assert _parallel_worker_budget(16, 8) == 8
    assert _parallel_worker_budget(3, 8) == 3
    assert _parallel_worker_budget(0, 8) == 1
    assert _parallel_drizzle_budget(4, 8) == (4, 2)
    assert _parallel_drizzle_budget(2, 8) == (2, 4)
    assert _parallel_drizzle_budget(1, 8) == (1, 8)


def test_frame_sets_disjoint(tmp_path: Path):
    a = tmp_path / 'a.fits'
    b = tmp_path / 'b.fits'
    c = tmp_path / 'c.fits'
    for p in (a, b, c):
        p.write_bytes(b'x')
    assert _frame_sets_disjoint([[a, b], [c]]) is True
    assert _frame_sets_disjoint([[a, b], [b, c]]) is False


def test_align_hst_raw_dir_parallel_workers(tmp_path: Path):
    raw = tmp_path / 'raw'
    jhat = tmp_path / 'jhat'
    raw.mkdir()
    for name, inst in (('a_flc.fits', 'ACS'), ('b_flc.fits', 'WFC3')):
        primary = fits.PrimaryHDU()
        primary.header['INSTRUME'] = inst
        primary.header['FILTER'] = 'F814W'
        sci = fits.ImageHDU(np.ones((8, 8), dtype=np.float32), name='SCI')
        fits.HDUList([primary, sci]).writeto(raw / name)

    aligned: list[str] = []

    def _fake_align(frame, outdir, **kwargs):
        del kwargs
        aligned.append(Path(frame).name)
        out = Path(outdir) / f'{Path(frame).stem}_jhat.fits'
        out.write_bytes(b'x')
        return out

    # Serial path (workers=1) must still work with patches.
    with (
        patch(
            'st123.stages.alignment.hst_jhat.align_hst_image', side_effect=_fake_align
        ),
        patch(
            'st123.stages.alignment.hst_jhat.harmonize_hst_jhat_dir', return_value=[]
        ),
    ):
        results = align_hst_raw_dir(
            raw,
            jhat,
            instruments=['ACS', 'WFC3'],
            soft_fail=True,
            gaia=True,
            workers=1,
        )
    assert len(results) == 2
    assert set(aligned) == {'a_flc.fits', 'b_flc.fits'}

    # Parallel path uses ProcessPoolExecutor; stub it to run inline.
    aligned.clear()
    fake_future = MagicMock()
    fake_future.result.side_effect = lambda: {
        'path': str(raw / 'a_flc.fits'),
        'status': 'ok',
        'error': None,
        'outpath': str(jhat / 'a_jhat.fits'),
    }

    class _InlinePool:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, job):
            fut = MagicMock()
            fut.result.return_value = fn(job)
            return fut

    with (
        patch(
            'st123.stages.alignment.hst_jhat.align_hst_image', side_effect=_fake_align
        ),
        patch(
            'st123.stages.alignment.hst_jhat.harmonize_hst_jhat_dir', return_value=[]
        ),
        patch(
            'concurrent.futures.ProcessPoolExecutor', _InlinePool
        ),
        patch(
            'concurrent.futures.as_completed',
            side_effect=lambda futures: list(futures),
        ),
    ):
        results = align_hst_raw_dir(
            raw,
            jhat,
            instruments=['ACS', 'WFC3'],
            soft_fail=True,
            gaia=True,
            workers=2,
        )
    assert len(results) == 2
    assert all(r.get('status') == 'ok' for r in results)


def test_run_filter_drizzle_jobs_serial_and_parallel():
    calls: list[int] = []

    def _worker(job):
        calls.append(int(job['num_cores']))
        rec = dict(job['record'])
        rec['status'] = 'ok'
        return rec

    jobs = [
        {'record': {'status': 'ready', 'instrument': 'wfc3', 'filter': 'f555w'}},
        {'record': {'status': 'ready', 'instrument': 'wfc3', 'filter': 'f814w'}},
    ]
    with patch(
        'st123.stages.mosaic.hst_drizzle._filter_drizzle_worker', side_effect=_worker
    ):
        out = _run_filter_drizzle_jobs(jobs, num_cores=8, parallel=False)
    assert len(out) == 2
    assert calls == [8, 8]

    calls.clear()

    class _InlinePool:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, job):
            fut = MagicMock()
            fut.result.return_value = fn(job)
            return fut

    with (
        patch(
            'st123.stages.mosaic.hst_drizzle._filter_drizzle_worker',
            side_effect=_worker,
        ),
        patch('concurrent.futures.ProcessPoolExecutor', _InlinePool),
        patch(
            'concurrent.futures.as_completed',
            side_effect=lambda futures: list(futures),
        ),
    ):
        out = _run_filter_drizzle_jobs(jobs, num_cores=8, parallel=True)
    assert len(out) == 2
    # 2 workers -> 4 cores each
    assert calls == [4, 4]
