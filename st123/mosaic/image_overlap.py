#!/usr/bin/env python
"""Compute science / reference image footprint overlap from S_REGION polygons."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from shapely.geometry import Polygon
from shapely.ops import unary_union

from st123.mosaic.region import SRegionPolygon, illuminated_s_region_from_fits

warnings.filterwarnings('ignore')


@dataclass(frozen=True)
class AreaMetrics:
    """Area in science-image pixel units, with solid-angle and ROI-fraction context."""

    pixels2: float
    arcmin2: float
    fraction_of_roi: float

    @classmethod
    def from_pixels(
        cls,
        area_pix2: float,
        pix_area_arcmin2: float,
        roi_pix2: float,
    ) -> AreaMetrics:
        frac = area_pix2 / roi_pix2 if roi_pix2 > 0 else float('nan')
        return cls(
            pixels2=float(area_pix2),
            arcmin2=float(area_pix2) * pix_area_arcmin2,
            fraction_of_roi=frac,
        )

    @property
    def fraction_of_miri_roi(self) -> float:
        """Alias used by the MIRI alignment pipeline."""
        return self.fraction_of_roi

    def format(self) -> str:
        return (
            f'{self.pixels2:.3f} pixels^2 | {self.arcmin2:.6f} arcmin^2 | '
            f'{self.fraction_of_roi:.4f} of illuminated ROI'
        )


@dataclass(frozen=True)
class ScienceFootprint:
    """Illuminated science footprint (sky S_REGION + on-detector pixel polygon)."""

    path: str
    s_region: SRegionPolygon
    wcs: WCS
    polygon: Polygon
    pixel_area_arcmin2: float
    center_ra_deg: float
    center_dec_deg: float
    sky_polygon_arcsec: Polygon

    @property
    def area(self) -> AreaMetrics:
        return AreaMetrics.from_pixels(
            self.polygon.area,
            self.pixel_area_arcmin2,
            self.polygon.area,
        )

    @property
    def pixel_area_arcsec2(self) -> float:
        return self.pixel_area_arcmin2 * 3600.0

    @classmethod
    def from_fits(cls, path: str) -> ScienceFootprint:
        s_region, _, wcs, _, _, _ = illuminated_s_region_from_fits(path)
        verts = np.asarray(s_region.vertices, dtype=float)
        center_ra = float(np.mean(verts[:, 0]))
        center_dec = float(np.mean(verts[:, 1]))
        return cls(
            path=path,
            s_region=s_region,
            wcs=wcs,
            polygon=s_region.to_pixel_polygon(wcs),
            pixel_area_arcmin2=float(
                wcs.proj_plane_pixel_area().to(u.arcmin**2).value
            ),
            center_ra_deg=center_ra,
            center_dec_deg=center_dec,
            sky_polygon_arcsec=s_region.to_tangent_polygon(center_ra, center_dec),
        )

    def metrics(self, area_pix2: float) -> AreaMetrics:
        return AreaMetrics.from_pixels(
            area_pix2,
            self.pixel_area_arcmin2,
            float(self.polygon.area),
        )

    def metrics_from_sky_arcsec2(self, area_arcsec2: float) -> AreaMetrics:
        """Convert a tangent-plane area (arcsec²) into science-pixel AreaMetrics."""
        pix_area = self.pixel_area_arcsec2
        area_pix2 = float(area_arcsec2) / pix_area if pix_area > 0 else 0.0
        roi_pix2 = (
            float(self.sky_polygon_arcsec.area) / pix_area if pix_area > 0 else 0.0
        )
        return AreaMetrics.from_pixels(
            area_pix2,
            self.pixel_area_arcmin2,
            roi_pix2,
        )


# Backward-compatible alias for older call sites / notebooks.
MirIFootprint = ScienceFootprint


@dataclass(frozen=True)
class OverlapResult:
    """Overlap of one reference footprint with a science illuminated footprint."""

    ref_path: str
    ref_s_region: SRegionPolygon
    ref_area: AreaMetrics
    overlap_area: AreaMetrics


@dataclass(frozen=True)
class BestOverlap:
    """Best-matching reference for a single science frame."""

    science_path: str
    ref_path: str | None
    overlap_area: AreaMetrics

    @property
    def miri_path(self) -> str:
        """Alias used by the MIRI alignment pipeline."""
        return self.science_path


def load_header_s_region(fits_path: str, extname: str = 'SCI') -> SRegionPolygon:
    """Parse S_REGION from a FITS science header."""
    with fits.open(fits_path) as hdul:
        return SRegionPolygon.parse(hdul[extname].header['S_REGION'])


def polygon_area(polygon: Polygon) -> float:
    """Return Shapely polygon area, or 0 for an empty geometry."""
    return 0.0 if polygon.is_empty else float(polygon.area)


def compute_overlap(science: ScienceFootprint, ref_path: str) -> OverlapResult:
    """
    Compute footprint overlap in a local sky tangent plane.

    Intersection is performed on ``S_REGION`` polygons expressed as
    arcsecond offsets from the science footprint center. Reported areas are
    converted to science pixels² via the science pixel solid angle.
    """
    ref_s_region = load_header_s_region(ref_path)
    ref_sky = ref_s_region.to_tangent_polygon(
        science.center_ra_deg,
        science.center_dec_deg,
    )
    overlap_sky = science.sky_polygon_arcsec.intersection(ref_sky)
    return OverlapResult(
        ref_path=ref_path,
        ref_s_region=ref_s_region,
        ref_area=science.metrics_from_sky_arcsec2(polygon_area(ref_sky)),
        overlap_area=science.metrics_from_sky_arcsec2(polygon_area(overlap_sky)),
    )


def compute_cumulative_overlap_fraction(
    science: ScienceFootprint,
    ref_paths: list[str],
) -> float:
    """
    Fraction of the science illuminated ROI covered by the union of references.

    Overlapping reference footprints are merged (unique area only) before
    dividing by the science sky footprint area. Returns 0.0 when there is no
    overlap.
    """
    science_area = polygon_area(science.sky_polygon_arcsec)
    if science_area <= 0.0 or not ref_paths:
        return 0.0

    pieces = []
    for ref_path in ref_paths:
        try:
            ref_s_region = load_header_s_region(ref_path)
            ref_sky = ref_s_region.to_tangent_polygon(
                science.center_ra_deg,
                science.center_dec_deg,
            )
            overlap_sky = science.sky_polygon_arcsec.intersection(ref_sky)
        except Exception:
            continue
        if not overlap_sky.is_empty and polygon_area(overlap_sky) > 0.0:
            pieces.append(overlap_sky)

    if not pieces:
        return 0.0
    return float(polygon_area(unary_union(pieces)) / science_area)


def overlap_area_pixels(
    science_image: str,
    ref_image: str,
) -> tuple[float, Polygon, Polygon]:
    """
    Return overlap area (science pixels²) and the two sky-tangent polygons.

    Polygons are in local tangent-plane arcseconds (not detector pixels).
    """
    science = ScienceFootprint.from_fits(science_image)
    ref_s_region = load_header_s_region(ref_image)
    ref_sky = ref_s_region.to_tangent_polygon(
        science.center_ra_deg,
        science.center_dec_deg,
    )
    overlap = science.sky_polygon_arcsec.intersection(ref_sky)
    return (
        science.metrics_from_sky_arcsec2(polygon_area(overlap)).pixels2,
        science.sky_polygon_arcsec,
        ref_sky,
    )


def find_best_refs(
    science_images: list[str],
    refs: list[str],
    outfile: str | None = None,
) -> list[BestOverlap]:
    """For each science image, find the reference with maximum overlap area."""
    results: list[BestOverlap] = []
    out_path = Path(outfile) if outfile else None
    out_lines: list[str] = []

    for image in science_images:
        print(f'Science: {image}')
        science = ScienceFootprint.from_fits(image)
        print(f'  illuminated S_REGION: {science.s_region.to_string()}')
        print(
            f'  WCS pixel solid angle: {science.pixel_area_arcmin2:.8e} '
            f'arcmin^2 / pixel'
        )
        print(f'  illuminated area: {science.area.format()}')

        best: BestOverlap | None = None
        for ref in refs:
            try:
                result = compute_overlap(science, ref)
            except Exception as exc:
                msg = f'{image} {ref}\nimage failed: {exc}\n\n'
                print(f'  FAILED for ref {ref}: {exc}')
                out_lines.append(msg)
                continue

            print(f'  ref: {ref}')
            print(f'    S_REGION: {result.ref_s_region.to_string()}')
            print(f'    ref area: {result.ref_area.format()}')
            print(f'    overlap area: {result.overlap_area.format()}')

            if best is None or result.overlap_area.pixels2 > best.overlap_area.pixels2:
                best = BestOverlap(
                    science_path=image,
                    ref_path=result.ref_path,
                    overlap_area=result.overlap_area,
                )

        if best is None:
            best = BestOverlap(
                science_path=image,
                ref_path=None,
                overlap_area=science.metrics(0.0),
            )

        line = (
            f'Overlap maximized: Science image: {best.science_path}, '
            f'Reference image: {best.ref_path}, '
            f'Max overlap area: {best.overlap_area.pixels2:.3f} pixels^2 '
            f'({best.overlap_area.arcmin2:.6f} arcmin^2, '
            f'{best.overlap_area.fraction_of_roi:.4f} of illuminated ROI)'
        )
        print(line)
        print()
        out_lines.append(line + '\n\n')
        results.append(best)

    if out_path is not None:
        out_path.write_text(''.join(out_lines))

    return results
