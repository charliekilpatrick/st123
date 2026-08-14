"""Tests for DOLPHOT catalog HDF5 export and dolphot-hdf5 CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


def _write_minimal_catalog(outdir: Path, *, name: str | None = None) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    label = name or outdir.name
    base = outdir / f'{label}.phot'
    (outdir / 'dolphot.param').write_text('Nimg = 1\nFitSky = 2\n', encoding='utf-8')
    Path(str(base) + '.columns').write_text(
        '1. Extension (zero for base image)\n'
        '2. Chip (for three-dimensional FITS image)\n'
        '3. Object X position\n'
        '4. Object Y position\n',
        encoding='utf-8',
    )
    cat = np.array([[0, 1, 100.0, 50.0], [0, 1, 200.0, 60.0]], dtype=np.float64)
    np.savetxt(base, cat)
    Path(str(base) + '.info').write_text('1 sets of output data\n', encoding='utf-8')
    Path(str(base) + '.data').write_text('WCS image 1: 1\n', encoding='utf-8')
    Path(str(base) + '.warnings').write_text('', encoding='utf-8')
    return base


def test_hdf5_path_for_phot_base():
    from st123.photometry.dolphot_catalog_hdf5 import hdf5_path_for_phot_base

    assert hdf5_path_for_phot_base('/tmp/phot_0_0.phot').name == 'phot_0_0.h5'
    assert hdf5_path_for_phot_base('/tmp/dp0000').name == 'dp0000.h5'


def test_ensure_dolphot_catalog_hdf5_roundtrip(tmp_path: Path):
    h5py = pytest.importorskip('h5py')
    from st123.photometry.dolphot_catalog_hdf5 import (
        ensure_dolphot_catalog_hdf5,
        read_dolphot_catalog_hdf5,
    )

    outdir = tmp_path / 'phot_0_0'
    _write_minimal_catalog(outdir)
    out = ensure_dolphot_catalog_hdf5(
        outdir, compression=False, include_directory_manifest=False
    )
    assert out is not None
    assert out.name == 'phot_0_0.h5'
    assert out.is_file()

    # Second call skips existing.
    again = ensure_dolphot_catalog_hdf5(outdir, compression=False)
    assert again == out

    t = read_dolphot_catalog_hdf5(out)
    assert len(t) == 2
    assert len(t.colnames) == 4

    with h5py.File(out, 'r') as hf:
        assert 'metadata' in hf
        assert hf['photometry'].attrs.get('st123_dolphot_hdf5_format') == 1


def test_ensure_uses_dolphot_param(tmp_path: Path):
    """st123 run dirs store the input param as dolphot.param, not <base>.param."""
    pytest.importorskip('h5py')
    from st123.photometry.dolphot_catalog_hdf5 import (
        ensure_dolphot_catalog_hdf5,
        _load_json_payload,
    )
    import h5py

    outdir = tmp_path / 'nircam_hst_0_0'
    _write_minimal_catalog(outdir)
    out = ensure_dolphot_catalog_hdf5(
        outdir, compression=False, include_directory_manifest=False
    )
    assert out is not None
    with h5py.File(out, 'r') as hf:
        merged = _load_json_payload(hf['metadata'], 'dolphot_merged_metadata_json')
        assert merged['global_param'].get('Nimg') == '1'


def test_discover_and_cli_skip_existing(tmp_path: Path):
    pytest.importorskip('h5py')
    from st123.scripts import dolphot_hdf5 as dh

    project = tmp_path / 'SN'
    _write_minimal_catalog(project / 'reduction' / 'phot_0_0')
    _write_minimal_catalog(project / 'dolphot' / 'nircam_hst_0_0')
    (project / 'reduction' / 'phot_empty').mkdir(parents=True)

    jobs = dh.discover_dolphot_hdf5_jobs(project)
    assert {j.label for j in jobs} == {'phot_0_0', 'nircam_hst_0_0'}

    rc = dh.main(['--base-dir', str(project), '-v'])
    assert rc == 0
    h5 = project / 'reduction' / 'phot_0_0' / 'phot_0_0.h5'
    assert h5.is_file()

    # Re-run skips without --force.
    mtime = h5.stat().st_mtime
    rc2 = dh.main(['--base-dir', str(project)])
    assert rc2 == 0
    assert h5.stat().st_mtime == mtime


def test_run_dolphot_writes_hdf5_after_success(tmp_path: Path):
    pytest.importorskip('h5py')
    from st123.scripts import run_dolphot as rd

    project = tmp_path / 'SN'
    outdir = project / 'reduction' / 'phot_0_0'
    outdir.mkdir(parents=True)
    (outdir / 'dolphot.param').write_text('Nimg = 1\n')

    def _fake_run(run, *, ncores, dolphot_bin):
        _write_minimal_catalog(run.outdir)
        return run, 0

    with patch.object(rd, '_run_one_wait', side_effect=_fake_run):
        rc = rd.main(
            [
                '--base-dir',
                str(project),
                '--instruments',
                'nircam',
                '--ncores',
                '1',
            ]
        )
    assert rc == 0
    assert (outdir / 'phot_0_0.h5').is_file()


def test_run_dolphot_can_skip_hdf5(tmp_path: Path):
    from st123.scripts import run_dolphot as rd

    project = tmp_path / 'SN'
    outdir = project / 'reduction' / 'phot_0_0'
    outdir.mkdir(parents=True)
    (outdir / 'dolphot.param').write_text('Nimg = 1\n')

    def _fake_run(run, *, ncores, dolphot_bin):
        _write_minimal_catalog(run.outdir)
        return run, 0

    with patch.object(rd, '_run_one_wait', side_effect=_fake_run):
        rc = rd.main(
            [
                '--base-dir',
                str(project),
                '--instruments',
                'nircam',
                '--no-write-dolphot-hdf5',
            ]
        )
    assert rc == 0
    assert not (outdir / 'phot_0_0.h5').is_file()


def test_dolphot_hdf5_parser_help():
    from st123.scripts import dolphot_hdf5 as dh

    parser = dh.create_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(['--help'])
    assert exc.value.code == 0
