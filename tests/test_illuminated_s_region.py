"""Tests for illuminated S_REGION library API and region CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from st123.mosaic.region import (
    SRegionPolygon,
    default_adjacency_pixels,
    illuminated_mask_from_dq,
    illuminated_s_region_from_fits,
    select_right_illuminated_component,
)
from st123.scripts import region as illum_script
from helpers import write_illuminated_fits


def test_s_region_polygon_roundtrip():
    text = 'POLYGON ICRS 10.0 20.0 10.1 20.0 10.1 20.1 10.0 20.1'
    poly = SRegionPolygon.parse(text)
    assert poly.frame == 'ICRS'
    assert poly.vertices.shape == (4, 2)
    assert 'POLYGON ICRS' in poly.to_string()
    with pytest.raises(ValueError):
        SRegionPolygon.parse('CIRCLE ICRS 1 2 3')


def test_illuminated_mask_and_right_component():
    dq = np.full((40, 60), 1024, dtype=np.uint16)
    dq[:, 40:] = 0
    mask = illuminated_mask_from_dq(dq, dq_threshold=512)
    assert mask[:, 40:].all()
    assert not mask[:, :40].any()
    component = select_right_illuminated_component(mask, min_pixels=10)
    assert component[:, 40:].all()


def test_default_adjacency_pixels():
    assert default_adjacency_pixels((100, 1000)) == pytest.approx(80.0)
    assert default_adjacency_pixels((100, 2000)) == pytest.approx(160.0)


def test_illuminated_s_region_from_fits(tmp_path: Path):
    path = write_illuminated_fits(tmp_path / 'science.fits')
    s_region, region_mask, wcs, header, data, _ = illuminated_s_region_from_fits(
        path,
        simplify_tolerance=1.0,
        adjacency_pixels=5.0,
        bridge_pixels=2.0,
    )
    assert isinstance(s_region, SRegionPolygon)
    assert region_mask.sum() > 0
    assert s_region.vertices.shape[0] >= 3
    assert data.shape == region_mask.shape
    assert wcs is not None
    assert 'NAXIS1' in header or header.get('NAXIS', 0) == 2


def test_illuminated_script_main(tmp_path: Path, monkeypatch):
    path = write_illuminated_fits(tmp_path / 'science.fits')
    out = tmp_path / 'plot.png'
    monkeypatch.setattr(illum_script.plt, 'close', lambda *a, **k: None)
    with patch('st123.scripts.region.save_illuminated_region_plot') as mock_plot:
        rc = illum_script.main(
            [str(path), '--output', str(out), '--simplify', '1.0', '--adjacency', '5']
        )
    assert rc == 0
    mock_plot.assert_called_once()


def test_illuminated_parser_defaults():
    parser = illum_script.create_parser()
    args = parser.parse_args(['image.fits'])
    assert args.hdu is None
    assert args.simplify == 2.0
    assert args.dq_threshold == 512
