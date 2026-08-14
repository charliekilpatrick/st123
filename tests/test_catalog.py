"""Tests for catalog script and st123.photometry.catalog helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from st123.photometry.catalog import get_filters, map_columns
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


def test_get_filters_and_map_columns(tmp_path: Path):
    columns = _write_columns_file(tmp_path / 'phot.columns')
    lines, filters = get_filters(str(columns))
    assert 'F115W' in filters
    assert len(lines) > 0
    col_dict, filters2, filter_cols = map_columns(str(columns))
    assert filters2 == filters
    assert col_dict['X'] == 0
    assert 'F115W' in filter_cols


def test_catalog_main_no_csv(tmp_path: Path):
    rc = catalog_script.main(['--photdir', str(tmp_path), '--outfile', str(tmp_path / 'out.csv')])
    assert rc == 1


def test_catalog_main_calls_create_common(tmp_path: Path):
    csv_path = tmp_path / 'rsg_f115w.csv'
    pd.DataFrame({'idx': [1, 2], 'x': [10.0, 11.0]}).to_csv(csv_path, index=False)
    outfile = tmp_path / 'combined.csv'
    with patch('st123.photometry.catalog.create_common_catalog') as mock_create:
        rc = catalog_script.main(['--photdir', str(tmp_path), '--outfile', str(outfile)])
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
