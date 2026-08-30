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
    from st123.stages.photometry.dolphot_catalog_hdf5 import hdf5_path_for_phot_base

    assert hdf5_path_for_phot_base('/tmp/phot_0_0.phot').name == 'phot_0_0.h5'
    assert hdf5_path_for_phot_base('/tmp/dp0000').name == 'dp0000.h5'


def test_ensure_dolphot_catalog_hdf5_roundtrip(tmp_path: Path):
    h5py = pytest.importorskip('h5py')
    from st123.stages.photometry.dolphot_catalog_hdf5 import (
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
    from st123.stages.photometry.dolphot_catalog_hdf5 import (
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


def _write_filter_catalog(
    outdir: Path,
    *,
    name: str | None = None,
    filt: str,
    xy_mags: list[tuple[float, float, str]],
) -> Path:
    """Write a minimal DOLPHOT .phot with one combined-filter block."""
    outdir.mkdir(parents=True, exist_ok=True)
    label = name or outdir.name
    base = outdir / f'{label}.phot'
    (outdir / 'dolphot.param').write_text('Nimg = 1\nFitSky = 2\n', encoding='utf-8')
    obj = [
        'Extension',
        'Chip',
        'Object X position',
        'Object Y position',
        'Chi for fit',
        'Signal-to-noise',
        'Object sharpness',
        'Object roundness',
        'Direction of major axis',
        'Crowding',
        'Object type',
        'Pass Detected',
    ]
    block = [
        f'Total counts, {filt}',
        f'Total sky level, {filt}',
        f'Normalized count rate, {filt}',
        f'Normalized count rate uncertainty, {filt}',
        f'Instrumental ABMAG magnitude, {filt}',
        f'Transformed UBVRI magnitude, {filt}',
        f'Magnitude uncertainty, {filt}',
        f'Chi, {filt}',
        f'Signal-to-noise, {filt}',
        f'Sharpness, {filt}',
        f'Roundness, {filt}',
        f'Crowding, {filt}',
        f'Photometry quality flag, {filt}',
    ]
    Path(str(base) + '.columns').write_text(
        ''.join(f'{i}. {n}\n' for i, n in enumerate(obj + block, start=1)),
        encoding='utf-8',
    )
    lines = []
    for x, y, mag in xy_mags:
        obj_vals = [
            '0',
            '1',
            f'{x:.3f}',
            f'{y:.3f}',
            '1.0',
            '10.0',
            '0.0',
            '0.0',
            '0.0',
            '0.0',
            '1',
            '1',
        ]
        phot_vals = ['100'] * 4 + [mag, '99.999', '0.01'] + ['0'] * 6
        lines.append(' '.join(obj_vals + phot_vals))
    base.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    Path(str(base) + '.info').write_text('1 sets of output data\n', encoding='utf-8')
    Path(str(base) + '.data').write_text('WCS image 1: 1\n', encoding='utf-8')
    Path(str(base) + '.warnings').write_text('', encoding='utf-8')
    return base


def test_discover_megacatalog_phot_files(tmp_path: Path):
    from st123.stages.photometry.megacatalog import discover_megacatalog_phot_files

    project = tmp_path / 'SN'
    nircam = _write_filter_catalog(
        project / 'reduction' / 'phot_0_sn',
        filt='NIRCAM_F150W',
        xy_mags=[(10.0, 20.0, '20.0'), (11.0, 21.0, '21.0')],
    )
    miri = _write_filter_catalog(
        project / 'dolphot' / 'nircam_miri_0_sn',
        filt='MIRI_F770W',
        xy_mags=[(10.0, 20.0, '18.0')],
    )
    hst = _write_filter_catalog(
        project / 'dolphot' / 'nircam_hst_0_sn',
        filt='WFC3_F814W',
        xy_mags=[(10.0, 20.0, '19.0')],
    )
    found = discover_megacatalog_phot_files(project, group=0, box='sn')
    assert found == [nircam, miri, hst]


def test_build_megacatalog_nircam_master_fill(tmp_path: Path):
    pytest.importorskip('h5py')
    from st123.stages.photometry.megacatalog import build_megacatalog

    project = tmp_path / 'SN'
    nircam = _write_filter_catalog(
        project / 'reduction' / 'phot_0_sn',
        filt='NIRCAM_F150W',
        xy_mags=[(10.0, 20.0, '20.0'), (11.0, 21.0, '21.0')],
    )
    miri = _write_filter_catalog(
        project / 'dolphot' / 'nircam_miri_0_sn',
        filt='MIRI_F770W',
        xy_mags=[(10.0, 20.0, '18.5')],
    )
    hst = _write_filter_catalog(
        project / 'dolphot' / 'nircam_hst_0_sn',
        filt='WFC3_F814W',
        xy_mags=[(10.0, 20.0, '19.5')],
    )
    out_h5 = project / 'dolphot' / 'megacatalog_0_sn' / 'megacatalog_0_sn.h5'
    written = build_megacatalog(
        [nircam, miri, hst],
        out_h5,
        compression=False,
        force=True,
    )
    assert written == out_h5.resolve()
    assert out_h5.is_file()
    phot = out_h5.with_suffix('.phot')
    cols = Path(str(phot) + '.columns')
    assert phot.is_file() and cols.is_file()
    text = cols.read_text()
    assert 'NIRCAM_F150W' in text
    assert 'MIRI_F770W' in text
    assert 'WFC3_F814W' in text
    rows = [ln.split() for ln in phot.read_text().splitlines() if ln.strip()]
    assert len(rows) == 2
    # Combined blocks are 13 cols each after 12 object cols; mag is index 4 in block.
    assert float(rows[0][16]) == pytest.approx(20.0)  # F150W mag
    assert float(rows[0][16 + 13]) == pytest.approx(18.5)  # F770W
    assert float(rows[0][16 + 26]) == pytest.approx(19.5)  # F814W
    assert float(rows[1][16 + 13]) == pytest.approx(99.999)
    assert float(rows[1][16 + 26]) == pytest.approx(99.999)


def test_dolphot_hdf5_merge_requires_outfile(tmp_path: Path):
    from st123.scripts import dolphot_hdf5 as dh

    project = tmp_path / 'SN'
    project.mkdir()
    rc = dh.main(['--base-dir', str(project), '--merge', '-v'])
    assert rc == 2


def test_dolphot_hdf5_merge_cli(tmp_path: Path):
    pytest.importorskip('h5py')
    from st123.scripts import dolphot_hdf5 as dh

    project = tmp_path / 'SN'
    nircam = _write_filter_catalog(
        project / 'reduction' / 'phot_0_sn',
        filt='NIRCAM_F150W',
        xy_mags=[(10.0, 20.0, '20.0')],
    )
    miri = _write_filter_catalog(
        project / 'dolphot' / 'nircam_miri_0_sn',
        filt='MIRI_F770W',
        xy_mags=[(10.0, 20.0, '18.0')],
    )
    out_h5 = project / 'dolphot' / 'megacatalog_0_sn' / 'megacatalog_0_sn.h5'
    rc = dh.main(
        [
            '--base-dir',
            str(project),
            '--merge',
            '--phot',
            str(nircam),
            str(miri),
            '--outfile',
            str(out_h5),
            '--no-compression',
            '-v',
        ]
    )
    assert rc == 0
    assert out_h5.is_file()
