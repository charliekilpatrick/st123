#!/usr/bin/env python
"""Compute science / reference image footprint overlap from S_REGION polygons."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from shapely.geometry import Polygon
from shapely.ops import unary_union

from st123.stages.mosaic.region import SRegionPolygon, illuminated_s_region_from_fits

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AreaMetrics:
    """Area in science-image pixel units, with solid-angle and ROI-fraction context.

    Attributes
    ----------
    pixels2 : float
        Area in science detector pixels squared.
    arcmin2 : float
        Solid angle in square arcminutes.
    fraction_of_roi : float
        Fraction of the illuminated science ROI (0-1, or NaN if ROI area is zero).
    """

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
        """Build metrics from a pixel area and calibration constants.

        Parameters
        ----------
        area_pix2 : float
            Area in science detector pixels squared.
        pix_area_arcmin2 : float
            Solid angle of one science pixel in square arcminutes.
        roi_pix2 : float
            Total illuminated ROI area in science pixels squared.

        Returns
        -------
        AreaMetrics
            Computed area metrics.
        """
        frac = area_pix2 / roi_pix2 if roi_pix2 > 0 else float('nan')
        return cls(
            pixels2=float(area_pix2),
            arcmin2=float(area_pix2) * pix_area_arcmin2,
            fraction_of_roi=frac,
        )

    @property
    def fraction_of_miri_roi(self) -> float:
        """Alias for :attr:`fraction_of_roi` used by the MIRI alignment pipeline.

        Returns
        -------
        float
            Same value as :attr:`fraction_of_roi`.
        """
        return self.fraction_of_roi

    def format(self) -> str:
        """Return a human-readable summary of all area fields.

        Returns
        -------
        str
            Formatted string with pixels^2, arcmin^2, and ROI fraction.
        """
        return (
            f'{self.pixels2:.3f} pixels^2 | {self.arcmin2:.6f} arcmin^2 | '
            f'{self.fraction_of_roi:.4f} of illuminated ROI'
        )


@dataclass(frozen=True)
class ScienceFootprint:
    """Illuminated science footprint (sky S_REGION + on-detector pixel polygon).

    See also :data:`MirIFootprint`, an alias for this class used in MIRI notebooks
    and alignment call sites.
    """

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
        """Area metrics for the full illuminated science footprint.

        Returns
        -------
        AreaMetrics
            Metrics for the on-detector pixel polygon.
        """
        return AreaMetrics.from_pixels(
            self.polygon.area,
            self.pixel_area_arcmin2,
            self.polygon.area,
        )

    @property
    def pixel_area_arcsec2(self) -> float:
        """Solid angle of one science pixel in square arcseconds.

        Returns
        -------
        float
            Pixel solid angle derived from :attr:`pixel_area_arcmin2`.
        """
        return self.pixel_area_arcmin2 * 3600.0

    @classmethod
    def from_fits(cls, image) -> ScienceFootprint:
        """Load an illuminated footprint from a science datamodel.

        Parameters
        ----------
        image
            Science datamodel or FITS path with ``S_REGION`` and WCS headers.

        Returns
        -------
        ScienceFootprint
            Parsed footprint with sky and pixel polygons.
        """
        from st123.datamodels.instrument import as_datamodel, path_of

        model = as_datamodel(image)
        path = str(path_of(model))
        s_region, _, wcs, _, _, _ = illuminated_s_region_from_fits(model)
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
        """Convert a pixel area into :class:`AreaMetrics` for this footprint.

        Parameters
        ----------
        area_pix2 : float
            Area in science detector pixels squared.

        Returns
        -------
        AreaMetrics
            Metrics normalized to the illuminated ROI of this footprint.
        """
        return AreaMetrics.from_pixels(
            area_pix2,
            self.pixel_area_arcmin2,
            float(self.polygon.area),
        )

    def metrics_from_sky_arcsec2(self, area_arcsec2: float) -> AreaMetrics:
        """Convert a tangent-plane area (arcsec^2) into science-pixel metrics.

        Parameters
        ----------
        area_arcsec2 : float
            Area in square arcseconds on the local sky tangent plane.

        Returns
        -------
        AreaMetrics
            Equivalent area expressed in science pixels and ROI fraction.
        """
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


# Alias for :class:`ScienceFootprint` (MIRI pipeline / notebook naming).
MirIFootprint = ScienceFootprint


@dataclass(frozen=True)
class OverlapResult:
    """Overlap of one reference footprint with a science illuminated footprint.

    Attributes
    ----------
    ref_path : str
        Path to the reference FITS file.
    ref_s_region : SRegionPolygon
        Parsed ``S_REGION`` polygon from the reference header.
    ref_area : AreaMetrics
        Reference footprint area expressed in science-pixel units.
    overlap_area : AreaMetrics
        Intersection area expressed in science-pixel units.
    """

    ref_path: str
    ref_s_region: SRegionPolygon
    ref_area: AreaMetrics
    overlap_area: AreaMetrics


@dataclass(frozen=True)
class BestOverlap:
    """Best-matching reference for a single science frame.

    Attributes
    ----------
    science_path : str
        Path to the science FITS file.
    ref_path : str or None
        Path to the best-matching reference, or ``None`` if no overlap was found.
    overlap_area : AreaMetrics
        Overlap area for the chosen reference (zero if ``ref_path`` is ``None``).
    """

    science_path: str
    ref_path: str | None
    overlap_area: AreaMetrics

    @property
    def miri_path(self) -> str:
        """Alias for :attr:`science_path` used by the MIRI alignment pipeline.

        Returns
        -------
        str
            Same value as :attr:`science_path`.
        """
        return self.science_path


def load_header_s_region(image, extname: str = 'SCI') -> SRegionPolygon:
    """Parse ``S_REGION`` from a science datamodel.

    Parameters
    ----------
    image
        Datamodel or FITS path containing an ``S_REGION`` keyword.
    extname : str, optional
        HDU extension name to read (default ``'SCI'``).

    Returns
    -------
    SRegionPolygon
        Parsed sky polygon from the header.
    """
    from st123.datamodels.instrument import as_datamodel

    model = as_datamodel(image)
    if str(extname).upper() == 'SCI':
        text = model.s_region
        if not text:
            raise KeyError(f'No S_REGION in {model.path}')
        return SRegionPolygon.parse(text)
    with model.open() as hdul:
        return SRegionPolygon.parse(hdul[extname].header['S_REGION'])


def polygon_area(polygon: Polygon) -> float:
    """Return Shapely polygon area, or 0 for an empty geometry.

    Parameters
    ----------
    polygon : Polygon
        Shapely polygon (typically in arcsecond offsets or pixel coordinates).

    Returns
    -------
    float
        Polygon area, or ``0.0`` when ``polygon`` is empty.
    """
    return 0.0 if polygon.is_empty else float(polygon.area)


def compute_overlap(science: ScienceFootprint, ref) -> OverlapResult:
    """Compute footprint overlap in a local sky tangent plane.

    Intersection is performed on ``S_REGION`` polygons expressed as
    arcsecond offsets from the science footprint center. Reported areas are
    converted to science pixels^2 via the science pixel solid angle.

    Parameters
    ----------
    science : ScienceFootprint
        Illuminated science footprint (from :meth:`ScienceFootprint.from_fits`).
    ref
        Reference datamodel or FITS path with an ``S_REGION`` header keyword.

    Returns
    -------
    OverlapResult
        Reference and overlap areas in science-pixel units.
    """
    from st123.datamodels.instrument import as_datamodel, path_of

    ref_model = as_datamodel(ref)
    ref_path = str(path_of(ref_model))
    ref_s_region = load_header_s_region(ref_model)
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
    """Fraction of the science illuminated ROI covered by the union of references.

    Overlapping reference footprints are merged (unique area only) before
    dividing by the science sky footprint area.

    Parameters
    ----------
    science : ScienceFootprint
        Illuminated science footprint.
    ref_paths : list of str
        Paths to reference FITS files with ``S_REGION`` headers.

    Returns
    -------
    float
        Covered fraction of the science sky footprint (0.0 when there is no
        overlap or the science area is zero).
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
    """Return overlap area (science pixels^2) and the two sky-tangent polygons.

    Parameters
    ----------
    science_image : str
        Path to the science FITS file.
    ref_image : str
        Path to the reference FITS file.

    Returns
    -------
    overlap_pixels2 : float
        Intersection area in science detector pixels squared.
    science_sky : Polygon
        Science footprint in local tangent-plane arcseconds.
    ref_sky : Polygon
        Reference footprint in the same tangent-plane frame.
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
    """For each science image, find the reference with maximum overlap area.

    Parameters
    ----------
    science_images : list of str
        Paths to science FITS files.
    refs : list of str
        Paths to candidate reference FITS files.
    outfile : str or None, optional
        If given, write a human-readable log of overlap results to this path.

    Returns
    -------
    list of BestOverlap
        One best-match record per science image (``ref_path`` may be ``None``).
    """
    results: list[BestOverlap] = []
    out_path = Path(outfile) if outfile else None
    out_lines: list[str] = []

    for image in science_images:
        logger.info('Science: %s', image)
        science = ScienceFootprint.from_fits(image)
        logger.info('  illuminated S_REGION: %s', science.s_region.to_string())
        logger.info(
            '  WCS pixel solid angle: %.8e arcmin^2 / pixel',
            science.pixel_area_arcmin2,
        )
        logger.info('  illuminated area: %s', science.area.format())

        best: BestOverlap | None = None
        for ref in refs:
            try:
                result = compute_overlap(science, ref)
            except Exception as exc:
                msg = f'{image} {ref}\nimage failed: {exc}\n\n'
                logger.error('  FAILED for ref %s: %s', ref, exc)
                out_lines.append(msg)
                continue

            logger.info('  ref: %s', ref)
            logger.info('    S_REGION: %s', result.ref_s_region.to_string())
            logger.info('    ref area: %s', result.ref_area.format())
            logger.info('    overlap area: %s', result.overlap_area.format())

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
        logger.info(line)
        out_lines.append(line + '\n\n')
        results.append(best)

    if out_path is not None:
        out_path.write_text(''.join(out_lines))

    return results
