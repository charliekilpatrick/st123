"""Derive an S_REGION polygon for the illuminated portion of a FITS image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage
from shapely.geometry import Polygon
from shapely import simplify
from shapely.ops import unary_union
from skimage import measure


@dataclass(frozen=True)
class SRegionPolygon:
    """Sky polygon in S_REGION format."""

    frame: str
    vertices: np.ndarray  # shape (N, 2), columns are (ra, dec) in degrees

    @classmethod
    def parse(cls, s_region: str) -> SRegionPolygon:
        tokens = s_region.strip().split()
        if len(tokens) < 4 or tokens[0].upper() != "POLYGON":
            raise ValueError(f"Unsupported S_REGION format: {s_region!r}")
        frame = tokens[1].upper()
        values = [float(token) for token in tokens[2:]]
        if len(values) % 2:
            raise ValueError(f"S_REGION has an odd number of coordinate values: {s_region!r}")
        vertices = np.asarray(values, dtype=float).reshape(-1, 2)
        return cls(frame=frame, vertices=vertices)

    def to_string(self, precision: int = 9) -> str:
        coord_text = " ".join(
            f"{ra:.{precision}f} {dec:.{precision}f}" for ra, dec in self.vertices
        )
        return f"POLYGON {self.frame} {coord_text}"

    def to_tangent_polygon(
        self,
        center_ra_deg: float,
        center_dec_deg: float,
    ) -> Polygon:
        """
        Project vertices into a local tangent plane centered on ``(ra, dec)``.

        Coordinates are offsets in arcseconds (east, north). This is the safe
        geometry for footprint intersection: unlike WCS pixel projection, it
        does not invent detector-plane coordinates for far off-axis sky positions.
        """
        ra = np.asarray(self.vertices[:, 0], dtype=float)
        dec = np.asarray(self.vertices[:, 1], dtype=float)
        dra = (ra - float(center_ra_deg)) * np.cos(np.deg2rad(float(center_dec_deg)))
        ddec = dec - float(center_dec_deg)
        return Polygon(np.column_stack([dra * 3600.0, ddec * 3600.0]))

    def to_pixel_polygon(
        self,
        wcs: WCS,
        *,
        max_roundtrip_arcsec: float = 1.0,
    ) -> Polygon:
        """
        Project vertices into ``wcs`` pixel coordinates.

        Rejects projections whose sky→pixel→sky round-trip exceeds
        ``max_roundtrip_arcsec`` so far off-axis vertices are not treated as
        valid detector-plane footprint corners.
        """
        ra0 = np.asarray(self.vertices[:, 0], dtype=float)
        dec0 = np.asarray(self.vertices[:, 1], dtype=float)
        xy = np.asarray(wcs.all_world2pix(ra0, dec0, 0, quiet=True))
        if xy.ndim != 2:
            raise ValueError("Unexpected all_world2pix return shape for S_REGION")
        if xy.shape[0] == 2 and xy.shape[1] != 2:
            x, y = np.asarray(xy[0], dtype=float), np.asarray(xy[1], dtype=float)
        else:
            x, y = np.asarray(xy[:, 0], dtype=float), np.asarray(xy[:, 1], dtype=float)

        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError(
                "S_REGION vertices fall outside the WCS validity domain "
                "(non-finite pixel coordinates)"
            )

        ra1, dec1 = wcs.all_pix2world(x, y, 0)
        dra = (np.asarray(ra1, dtype=float) - ra0) * np.cos(np.deg2rad(dec0))
        ddec = np.asarray(dec1, dtype=float) - dec0
        err_arcsec = np.hypot(dra, ddec) * 3600.0
        if np.any(err_arcsec > max_roundtrip_arcsec):
            raise ValueError(
                "S_REGION vertices fall outside the WCS validity domain "
                f"(max sky round-trip {float(np.nanmax(err_arcsec)):.1f} arcsec "
                f"> {max_roundtrip_arcsec} arcsec)"
            )
        return Polygon(np.column_stack([x, y]))


def find_image_hdu(hdulist: fits.HDUList) -> tuple[int, fits.ImageHDU | fits.PrimaryHDU]:
    """Return the HDU index and HDU for a 2D science image with a usable WCS."""
    for index, hdu in enumerate(hdulist):
        if hdu.data is None or hdu.data.ndim != 2:
            continue
        header = hdu.header
        if "CTYPE1" in header and "CTYPE2" in header:
            return index, hdu

    for index, hdu in enumerate(hdulist):
        if hdu.data is not None and hdu.data.ndim == 2:
            return index, hdu

    raise ValueError("No 2D image HDU found in FITS file.")


def find_dq_hdu(
    hdulist: fits.HDUList,
    sci_hdu: fits.ImageHDU | fits.PrimaryHDU,
    sci_index: int,
) -> fits.ImageHDU | fits.PrimaryHDU:
    """Return the DQ HDU matching the science image shape."""
    if "DQ" in hdulist:
        dq_hdu = hdulist["DQ"]
        if dq_hdu.data is not None and dq_hdu.data.shape == sci_hdu.data.shape:
            return dq_hdu

    for hdu in hdulist:
        if (
            hdu.name == "DQ"
            and hdu.data is not None
            and hdu.data.ndim == 2
            and hdu.data.shape == sci_hdu.data.shape
        ):
            return hdu

    for offset in (2, 1, -1, 3):
        candidate_index = sci_index + offset
        if 0 <= candidate_index < len(hdulist):
            candidate = hdulist[candidate_index]
            if (
                candidate.data is not None
                and candidate.data.ndim == 2
                and candidate.data.shape == sci_hdu.data.shape
            ):
                return candidate

    raise ValueError("No DQ extension with matching shape found in FITS file.")


def illuminated_mask_from_dq(dq: np.ndarray, *, dq_threshold: int = 512) -> np.ndarray:
    """Pixels considered illuminated based on the DQ frame."""
    return dq < dq_threshold


def illuminated_mask(data: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for finite-value masking."""
    return np.isfinite(data)


def select_right_illuminated_component(
    mask: np.ndarray,
    *,
    min_pixels: int | None = None,
) -> np.ndarray:
    """Return a mask for the largest illuminated component on the right side."""
    labeled, component_count = ndimage.label(mask)
    if component_count == 0:
        raise ValueError("No illuminated pixels found in image data.")

    if min_pixels is None:
        min_pixels = max(100, int(0.001 * mask.size))

    image_center_x = (mask.shape[1] - 1) / 2.0
    right_candidates: list[tuple[int, int, float]] = []
    for component_id in range(1, component_count + 1):
        component_mask = labeled == component_id
        pixel_count = int(component_mask.sum())
        if pixel_count < min_pixels:
            continue
        mean_x = float(np.where(component_mask)[1].mean())
        if mean_x > image_center_x:
            right_candidates.append((component_id, pixel_count, mean_x))

    if not right_candidates:
        raise ValueError(
            "No illuminated component found on the right-hand side of the image."
        )

    right_candidates.sort(key=lambda item: item[1], reverse=True)
    selected_id = right_candidates[0][0]
    return labeled == selected_id


def default_adjacency_pixels(shape: tuple[int, ...]) -> float:
    """Heuristic reach for merging nearby illuminated fragments into the ROI."""
    width = shape[-1]
    return max(80.0, 0.08 * width)


def expand_illuminated_region(
    valid_mask: np.ndarray,
    primary_mask: np.ndarray,
    *,
    adjacency_pixels: float | None = None,
) -> np.ndarray:
    """
    Include illuminated pixels within adjacency_pixels of the primary region.

    Distance is measured through bad pixels, so components on both sides of a
    bad column can be merged into the ROI when they lie near the primary area.
    """
    if adjacency_pixels is None:
        adjacency_pixels = default_adjacency_pixels(valid_mask.shape)
    distance = ndimage.distance_transform_edt(~primary_mask)
    return valid_mask & (distance <= adjacency_pixels)


def auto_bridge_pixels(mask: np.ndarray, max_bridge: int = 50) -> int:
    """Return a dilation radius that merges disconnected ROI fragments."""
    _, component_count = ndimage.label(mask)
    if component_count <= 1:
        return 0

    structure = ndimage.generate_binary_structure(2, 2)
    for iterations in range(1, max_bridge + 1):
        dilated = ndimage.binary_dilation(mask, structure=structure, iterations=iterations)
        if ndimage.label(dilated)[1] == 1:
            return iterations
    return max_bridge


def _contour_polygon(component_mask: np.ndarray) -> Polygon:
    contours = measure.find_contours(component_mask.astype(float), 0.5)
    if not contours:
        raise ValueError("Could not trace a contour for the illuminated region.")

    contour = max(contours, key=len)
    polygon = Polygon([(column, row) for row, column in contour])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def mask_to_pixel_polygon(
    mask: np.ndarray,
    *,
    simplify_tolerance: float = 2.0,
    bridge_pixels: float | None = None,
    min_component_pixels: int = 50,
) -> Polygon:
    """Trace the illuminated mask boundary and return a simplified pixel polygon."""
    labeled, component_count = ndimage.label(mask)
    component_polygons: list[Polygon] = []
    for component_id in range(1, component_count + 1):
        component_mask = labeled == component_id
        if int(component_mask.sum()) < min_component_pixels:
            continue
        component_polygons.append(_contour_polygon(component_mask))

    if not component_polygons:
        raise ValueError("Could not trace a contour for the illuminated region.")

    if len(component_polygons) == 1 and bridge_pixels in (None, 0):
        polygon = component_polygons[0]
    else:
        polygon = unary_union(component_polygons)
        if bridge_pixels is None:
            bridge_pixels = auto_bridge_pixels(mask)
        if bridge_pixels > 0:
            polygon = polygon.buffer(bridge_pixels, join_style=2)
        if polygon.geom_type == "MultiPolygon":
            polygon = max(polygon.geoms, key=lambda geom: geom.area)

    if simplify_tolerance > 0:
        polygon = simplify(polygon, tolerance=simplify_tolerance, preserve_topology=True)
        if polygon.geom_type == "MultiPolygon":
            polygon = max(polygon.geoms, key=lambda geom: geom.area)
    return polygon


def pixel_polygon_to_s_region(
    polygon: Polygon,
    wcs: WCS,
    *,
    frame: str = "ICRS",
) -> SRegionPolygon:
    """Convert a pixel-space polygon to sky coordinates."""
    x_coords, y_coords = polygon.exterior.coords.xy
    ra, dec = wcs.pixel_to_world_values(np.asarray(x_coords), np.asarray(y_coords))
    vertices = np.column_stack([ra, dec])
    return SRegionPolygon(frame=frame, vertices=vertices)


def infer_coordinate_frame(header: fits.Header, s_region: str | None = None) -> str:
    if s_region:
        return SRegionPolygon.parse(s_region).frame
    if "RADESYS" in header:
        return str(header["RADESYS"]).upper()
    return "ICRS"


def illuminated_s_region_from_fits(
    fits_path: str | Path,
    *,
    hdu_index: int | None = None,
    simplify_tolerance: float = 2.0,
    adjacency_pixels: float | None = None,
    bridge_pixels: float | None = None,
    dq_threshold: int = 512,
) -> tuple[SRegionPolygon, np.ndarray, WCS, fits.Header, np.ndarray, np.ndarray]:
    """
    Build an S_REGION polygon for the right-hand illuminated region.

    Illuminated pixels are selected from the DQ extension (DQ < dq_threshold),
    not from NaNs in the science image.

    Returns
    -------
    s_region_polygon
        Sky polygon for the illuminated region.
    region_mask
        Boolean mask of the selected illuminated component.
    wcs
        WCS for the image HDU.
    header
        Header for the image HDU.
    data
        Image data array.
    valid_mask
        Boolean mask of pixels with DQ below the threshold.
    """
    with fits.open(fits_path) as hdulist:
        if hdu_index is None:
            hdu_index, hdu = find_image_hdu(hdulist)
        else:
            hdu = hdulist[hdu_index]

        dq_hdu = find_dq_hdu(hdulist, hdu, hdu_index)
        data = np.asarray(hdu.data, dtype=float)
        dq = np.asarray(dq_hdu.data)
        header = hdu.header
        wcs = WCS(header)
        frame = infer_coordinate_frame(header, header.get("S_REGION"))
        valid_mask = illuminated_mask_from_dq(dq, dq_threshold=dq_threshold)
        primary_mask = select_right_illuminated_component(valid_mask)
        region_mask = expand_illuminated_region(
            valid_mask,
            primary_mask,
            adjacency_pixels=adjacency_pixels,
        )
        pixel_polygon = mask_to_pixel_polygon(
            region_mask,
            simplify_tolerance=simplify_tolerance,
            bridge_pixels=bridge_pixels,
        )
        s_region_polygon = pixel_polygon_to_s_region(
            pixel_polygon,
            wcs,
            frame=frame,
        )
        return s_region_polygon, region_mask, wcs, header, data, valid_mask


def illuminated_s_region_string(
    fits_path: str | Path,
    *,
    hdu_index: int | None = None,
    simplify_tolerance: float = 2.0,
    adjacency_pixels: float | None = None,
    bridge_pixels: float | None = None,
    dq_threshold: int = 512,
    precision: int = 9,
) -> str:
    """Return the S_REGION string bounding the illuminated region of interest."""
    s_region_polygon, _, _, _, _, _ = illuminated_s_region_from_fits(
        fits_path,
        hdu_index=hdu_index,
        simplify_tolerance=simplify_tolerance,
        adjacency_pixels=adjacency_pixels,
        bridge_pixels=bridge_pixels,
        dq_threshold=dq_threshold,
    )
    return s_region_polygon.to_string(precision=precision)


def save_illuminated_region_plot(
    data: np.ndarray,
    illuminated_polygon: SRegionPolygon,
    wcs: WCS,
    output_path: str | Path,
    *,
    original_s_region: SRegionPolygon | None = None,
    title: str | None = None,
) -> Path:
    """Plot the illuminated-region polygon over the original image and save a PNG."""
    plot_data = np.array(data, copy=True)
    plot_data[~np.isfinite(plot_data)] = np.nan

    pixel_polygon = illuminated_polygon.to_pixel_polygon(wcs)
    illum_x, illum_y = pixel_polygon.exterior.xy

    fig, ax = plt.subplots(figsize=(10, 10))
    vmin, vmax = np.nanpercentile(plot_data, [2, 98])
    ax.imshow(
        plot_data,
        origin="lower",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.plot(
        illum_x,
        illum_y,
        color="lime",
        linewidth=2.0,
        label="Illuminated S_REGION",
    )
    ax.plot(
        [illum_x[0]],
        [illum_y[0]],
        color="lime",
        marker="o",
        markersize=5,
    )

    if original_s_region is not None:
        original_polygon = original_s_region.to_pixel_polygon(wcs)
        orig_x, orig_y = original_polygon.exterior.xy
        ax.plot(
            orig_x,
            orig_y,
            color="dodgerblue",
            linewidth=1.5,
            linestyle="--",
            label="Header S_REGION",
        )

    ax.set_xlim(-0.5, data.shape[1] - 0.5)
    ax.set_ylim(-0.5, data.shape[0] - 0.5)
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.set_title(title or "Illuminated region polygon")
    ax.legend(loc="upper right")
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
