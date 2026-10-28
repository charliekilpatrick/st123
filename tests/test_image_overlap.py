"""Tests for image-overlap library API and script entry point."""

from __future__ import annotations

from pathlib import Path

import pytest
from shapely.geometry import Polygon

from helpers import write_illuminated_fits, write_ref_with_s_region
from st123.stages.mosaic.image_overlap import (
    AreaMetrics,
    ScienceFootprint,
    compute_overlap,
    find_best_refs,
    load_header_s_region,
    polygon_area,
)
from st123.scripts import image_overlap as overlap_script


def test_area_metrics_and_polygon_area():
    metrics = AreaMetrics.from_pixels(100.0, 0.01, 200.0)
    assert metrics.pixels2 == 100.0
    assert metrics.arcmin2 == pytest.approx(1.0)
    assert metrics.fraction_of_roi == pytest.approx(0.5)
    assert 'pixels^2' in metrics.format()
    assert polygon_area(Polygon()) == 0.0
    assert polygon_area(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])) == pytest.approx(1.0)


def test_load_header_s_region_and_compute_overlap(tmp_path: Path):
    science_path = write_illuminated_fits(tmp_path / 'science.fits')
    ref_path = write_ref_with_s_region(tmp_path / 'ref_i2d.fits')
    ref_region = load_header_s_region(str(ref_path))
    assert ref_region.frame == 'ICRS'

    science = ScienceFootprint.from_fits(str(science_path))
    result = compute_overlap(science, str(ref_path))
    assert result.overlap_area.pixels2 > 0
    assert result.ref_path == str(ref_path)


def test_find_best_refs(tmp_path: Path):
    science_path = write_illuminated_fits(tmp_path / 'science.fits')
    good_ref = write_ref_with_s_region(tmp_path / 'good_i2d.fits')
    bad_ref = write_ref_with_s_region(
        tmp_path / 'bad_i2d.fits', crval=(160.0, 10.0)
    )
    outfile = tmp_path / 'summary.txt'
    results = find_best_refs(
        [str(science_path)],
        [str(bad_ref), str(good_ref)],
        outfile=str(outfile),
    )
    assert len(results) == 1
    assert results[0].ref_path == str(good_ref)
    assert results[0].science_path == str(science_path)
    assert outfile.is_file()


def test_image_overlap_script_main(tmp_path: Path):
    science_path = write_illuminated_fits(tmp_path / 'science.fits')
    ref_path = write_ref_with_s_region(tmp_path / 'ref_i2d.fits')
    rc = overlap_script.main(
        ['--image', str(science_path), '--ref', str(ref_path)]
    )
    assert rc == 0


def test_image_overlap_parser():
    parser = overlap_script.create_parser()
    args = parser.parse_args(['--image', 'a.fits', '--ref', 'b.fits', 'c.fits'])
    assert args.image == ['a.fits']
    assert args.ref == ['b.fits', 'c.fits']
