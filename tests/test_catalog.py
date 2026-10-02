"""Tests for catalog script and st123.stages.photometry.catalog helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from st123.stages.photometry.catalog import (
    combine_visit_photometry,
    get_filters,
    inverse_variance_abmag,
    map_columns,
    parse_per_image_phot_columns,
    source_image_photometry,
    source_lightcurve,
    three_sigma_limit_abmag,
)
from st123.scripts import catalog as catalog_script


def _write_columns_file(path: Path) -> Path:
    # Minimal DOLPHOT-style column description used by get_filters / map_columns.
    lines = [
        '1. Object X position\n',
        '2. Object Y position\n',
        '3. Signal-to-noise\n',
        '4. Object sharpness\n',
        '5. Crowding\n',
        '6. Object type\n',
        '7. Normalized count rate, NIRCAM_F115W\n',
        '8. Normalized count rate uncertainty, NIRCAM_F115W\n',
        '9. Instrumental VEGAMAG magnitude, NIRCAM_F115W\n',
        '10. Magnitude uncertainty, NIRCAM_F115W\n',
        '11. Normalized count rate, NIRCAM_F115W\n',
        '12. Normalized count rate uncertainty, NIRCAM_F115W\n',
        '13. unused\n',
        '14. unused\n',
        '15. unused\n',
        '16. unused\n',
        '17. unused\n',
        '18. unused\n',
        '19. unused\n',
        '20. unused\n',
        '21. Flag column placeholder for F115W\n',
    ]
    path.write_text(''.join(lines))
    return path


def _image_block_names(image: str, instfilt: str) -> list[str]:
    tag = f'{image} ({instfilt}, 10.0 sec)'
    return [
        f'Measured counts, {tag}',
        f'Measured sky level, {tag}',
        f'Normalized count rate, {tag}',
        f'Normalized count rate uncertainty, {tag}',
        f'Instrumental ABMAG magnitude, {tag}',
        f'Transformed UBVRI magnitude, {tag}',
        f'Magnitude uncertainty, {tag}',
        f'Chi, {tag}',
        f'Signal-to-noise, {tag}',
        f'Sharpness, {tag}',
        f'Roundness, {tag}',
        f'Crowding, {tag}',
        f'Photometry quality flag, {tag}',
    ]


def _write_cal(path: Path, mjd: float) -> Path:
    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = 'NIRCAM'
    primary.header['TELESCOP'] = 'JWST'
    sci = fits.ImageHDU(np.ones((4, 4), dtype=np.float32), name='SCI')
    sci.header['MJD-AVG'] = mjd
    fits.HDUList([primary, sci]).writeto(path, overwrite=True)
    return path


def _write_visit_catalog(tmp_path: Path) -> Path:
    """One source, four frames: two F200W same day, one F200W +5d, one F150W."""
    frames = [
        ('img_a', 'NIRCAM_F200W', 20.00, 0.10, 0, 60000.0),
        ('img_b', 'NIRCAM_F200W', 20.00, 0.10, 0, 60000.4),
        ('img_c', 'NIRCAM_F200W', 21.00, 0.20, 0, 60005.0),
        ('img_d', 'NIRCAM_F150W', 22.00, 0.05, 0, 60000.1),
    ]
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
    names = list(obj)
    for image, instfilt, _mag, _err, _flag, mjd in frames:
        names.extend(_image_block_names(image, instfilt))
        _write_cal(tmp_path / f'{image}.fits', mjd)
    phot = tmp_path / 'run.phot'
    cols = Path(str(phot) + '.columns')
    cols.write_text(
        ''.join(f'{i}. {n}\n' for i, n in enumerate(names, start=1))
    )
    obj_vals = [
        '0', '1', '10.0', '20.0', '1.0', '50.0', '0.0', '0.0', '0.0', '0.1', '1', '1',
    ]
    phot_vals: list[str] = []
    for _image, _instfilt, mag, err, flag, _mjd in frames:
        phot_vals.extend(
            [
                '100',
                '1',
                '10',
                '1',
                f'{mag:.4f}',
                '99.999',
                f'{err:.4f}',
                '1',
                '20',
                '0',
                '0',
                '0.1',
                str(flag),
            ]
        )
    phot.write_text(' '.join(obj_vals + phot_vals) + '\n')
    return phot


def test_get_filters_and_map_columns(tmp_path: Path):
    columns = _write_columns_file(tmp_path / 'phot.columns')
    lines, filters = get_filters(str(columns))
    assert 'F115W' in filters
    assert len(lines) > 0
    col_dict, filters2, filter_cols = map_columns(str(columns))
    assert filters2 == filters
    assert col_dict['X'] == 0
    assert 'F115W' in filter_cols


def test_inverse_variance_abmag_two_equal_frames():
    out = inverse_variance_abmag([20.0, 20.0], [0.1, 0.1], mjd=[10.0, 12.0])
    assert out['n'] == 2
    assert out['mag'] == pytest.approx(20.0, abs=1e-8)
    assert out['magerr'] == pytest.approx(0.1 / np.sqrt(2.0), rel=1e-6)
    assert out['mjd'] == pytest.approx(11.0)


def test_parse_per_image_phot_columns_and_lightcurve(tmp_path: Path):
    phot = _write_visit_catalog(tmp_path)
    blocks = parse_per_image_phot_columns(str(phot) + '.columns')
    assert [b.filter for b in blocks] == ['F200W', 'F200W', 'F200W', 'F150W']
    per = source_image_photometry(phot, 0)
    assert len(per) == 4
    assert set(per['filter']) == {'F200W', 'F150W'}
    assert per['mjd'][0] == pytest.approx(60000.0)
    lc = source_lightcurve(phot, 0, max_gap_days=1.0)
    assert list(lc.colnames)[:4] == ['mjd', 'filter', 'mag', 'magerr']
    # F200W: two visits (same-day pair + isolated +5d); F150W: one visit.
    assert len(lc) == 3
    f200 = lc[lc['filter'] == 'F200W']
    assert len(f200) == 2
    assert f200['n'][0] == 2
    assert f200['mag'][0] == pytest.approx(20.0)
    assert f200['magerr'][0] == pytest.approx(0.1 / np.sqrt(2.0), rel=1e-5)
    assert f200['n'][1] == 1
    assert f200['mag'][1] == pytest.approx(21.0)
    f150 = lc[lc['filter'] == 'F150W']
    assert len(f150) == 1
    assert f150['mag'][0] == pytest.approx(22.0)
    assert f150['n'][0] == 1
    assert np.all(np.asarray(lc['detection'], dtype=bool))


def test_combine_visit_respects_max_gap(tmp_path: Path):
    phot = _write_visit_catalog(tmp_path)
    per = source_image_photometry(phot, 0)
    wide = combine_visit_photometry(per, max_gap_days=10.0)
    f200 = wide[wide['filter'] == 'F200W']
    assert len(f200) == 1
    assert f200['n'][0] == 3


def test_sub_three_sigma_epochs_become_neighbor_limits(tmp_path: Path):
    """SNR < 3 uses a 3-sigma limit from nearby sources; detections stay."""
    frames = [
        ('img_a', 'NIRCAM_F335M', 60000.0),
        ('img_b', 'NIRCAM_F335M', 60000.2),
    ]
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
    names = list(obj)
    for image, instfilt, mjd in frames:
        names.extend(_image_block_names(image, instfilt))
        _write_cal(tmp_path / f'{image}.fits', mjd)
    phot = tmp_path / 'run.phot'
    Path(str(phot) + '.columns').write_text(
        ''.join(f'{i}. {n}\n' for i, n in enumerate(names, start=1))
    )

    def _row(x: float, y: float, mag: float, err: float) -> str:
        obj_vals = [
            '0', '1', f'{x:.1f}', f'{y:.1f}', '1.0', '50.0', '0.0', '0.0',
            '0.0', '0.1', '1', '1',
        ]
        phot_vals: list[str] = []
        for _ in frames:
            phot_vals.extend(
                [
                    '100', '1', '10', '1', f'{mag:.4f}', '99.999', f'{err:.4f}',
                    '1', '20', '0', '0', '0.1', '0',
                ]
            )
        return ' '.join(obj_vals + phot_vals)

    # Target at (10,20): faint, SNR ~ 1. Neighbor 5 pix away: sky-like ferr.
    phot.write_text(
        '\n'.join(
            [
                _row(10.0, 20.0, 28.00, 1.00),
                _row(15.0, 20.0, 26.00, 0.25),
            ]
        )
        + '\n'
    )
    lc = source_lightcurve(phot, 0, max_gap_days=1.0)
    assert len(lc) == 1
    assert lc['filter'][0] == 'F335M'
    assert bool(lc['detection'][0]) is False
    assert not np.isfinite(lc['magerr'][0])
    neigh = inverse_variance_abmag([26.0, 26.0], [0.25, 0.25])
    expect = three_sigma_limit_abmag(neigh['flux_err_ujy'], nsigma=3.0)
    assert lc['mag'][0] == pytest.approx(expect, rel=1e-5)
    assert int(lc['n_neighbors'][0]) == 1


def test_catalog_main_lightcurve(tmp_path: Path):
    phot = _write_visit_catalog(tmp_path)
    outfile = tmp_path / 'lc.ecsv'
    rc = catalog_script.main(
        [
            '--photfile',
            str(phot),
            '--index',
            '0',
            '--outfile',
            str(outfile),
        ]
    )
    assert rc == 0
    assert outfile.is_file()
    from astropy.table import Table

    table = Table.read(outfile)
    assert 'mjd' in table.colnames and 'filter' in table.colnames
    assert 'mag' in table.colnames and 'magerr' in table.colnames
    assert len(table) == 3


def test_catalog_main_no_csv(tmp_path: Path):
    rc = catalog_script.main(
        ['--photdir', str(tmp_path), '--outfile', str(tmp_path / 'out.csv')]
    )
    assert rc == 1


def test_catalog_main_calls_create_common(tmp_path: Path):
    csv_path = tmp_path / 'rsg_f115w.csv'
    pd.DataFrame({'idx': [1, 2], 'x': [10.0, 11.0]}).to_csv(csv_path, index=False)
    outfile = tmp_path / 'combined.csv'
    with patch('st123.stages.photometry.catalog.create_common_catalog') as mock_create:
        rc = catalog_script.main(
            ['--photdir', str(tmp_path), '--outfile', str(outfile)]
        )
    assert rc == 0
    mock_create.assert_called_once()
    args = mock_create.call_args[0]
    assert np.array_equal(args[0], np.array([1, 2]))
    assert args[3] == str(outfile)


def test_catalog_parser():
    parser = catalog_script.create_parser()
    args = parser.parse_args(['--photdir', 'p', '--outfile', 'o.csv'])
    assert args.base_dir == 'p'
    assert args.outfile == 'o.csv'
    assert args.max_gap_days == 1.0
    args2 = parser.parse_args(
        ['--photfile', 'run.phot', '--index', '3', '--outfile', 'lc.ecsv']
    )
    assert args2.photfile == 'run.phot'
    assert args2.source_index == 3
