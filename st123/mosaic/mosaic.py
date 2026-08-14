from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shapely
import shapely.ops
import stpsf
from asdf import AsdfFile
from astropy import coordinates as coord
from astropy import nddata
from astropy import units as u
from astropy import wcs
from astropy.convolution import convolve_fft
from astropy.io import fits
from astropy.modeling import models
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats as scs
from astropy.table import Table
from ccdproc import Combiner
from gwcs import FITSImagingWCSTransform, coordinate_frames as cf
from gwcs import WCS as g_wcs
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
from jwst.pipeline import calwebb_image3
from matplotlib.patches import Polygon
from photutils.psf.matching import SplitCosineBellWindow, create_matching_kernel
from reproject.mosaicking import find_optimal_celestial_wcs

from st123.mast import parse_s_region
from st123.utils.compatibility import (
    ensure_local_crds_context,
    patch_jwst_for_photutils3,
)
from st123.utils.helpers import create_filter_table, input_list
from st123.utils.logging import capture_output

logger = logging.getLogger(__name__)


def _shape_and_wcs_for_mosaic(path: str | Path) -> tuple[tuple[int, int], wcs.WCS]:
    """
    Return ``(ny, nx)`` and a celestial WCS for mosaic footprint planning.

    Uses the SCI extension when present. Distortion lookup-table keywords
    (``CPDIS``) are resolved via the open HDUList, or stripped on failure so
    ``find_optimal_celestial_wcs`` can accept ``(shape, wcs)`` pairs for HST.
    """
    with fits.open(path) as hdul:
        sci = hdul['SCI'] if 'SCI' in hdul else hdul[1]
        try:
            w = wcs.WCS(sci.header, fobj=hdul, naxis=2)
        except Exception:
            hdr = sci.header.copy()
            for key in list(hdr.keys()):
                ku = str(key).upper()
                if ku.startswith(('CPDIS', 'DP', 'CPERR', 'DTOD')):
                    del hdr[key]
            w = wcs.WCS(hdr, naxis=2, relax=True)
        if sci.data is not None:
            shape = tuple(int(x) for x in sci.data.shape[-2:])
        else:
            shape = (int(sci.header['NAXIS2']), int(sci.header['NAXIS1']))
        return shape, w


def _sky_footprint_coords(path: str | Path) -> np.ndarray:
    """Return ``(N, 2)`` RA/Dec vertices from ``S_REGION`` or WCS footprint."""
    with fits.open(path) as hdul:
        sci = hdul['SCI'] if 'SCI' in hdul else hdul[1]
        region = sci.header.get('S_REGION')
        if not region:
            for hdu in hdul:
                if hdu.header.get('S_REGION'):
                    region = hdu.header['S_REGION']
                    break
        if region:
            return np.asarray(parse_s_region(region).exterior.coords[:-1])
        try:
            w = wcs.WCS(sci.header, fobj=hdul, naxis=2)
        except Exception:
            w = wcs.WCS(sci.header, naxis=2, relax=True)
        return np.asarray(w.calc_footprint(center=False), dtype=float)


# JWST imaging native / recommended mosaic pixel scales (arcsec / pixel).
# https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-observing-modes/nircam-imaging
NIRCAM_SW_PIXEL_SCALE = 0.031
NIRCAM_LW_PIXEL_SCALE = 0.063
MIRI_PIXEL_SCALE = 0.11

# Directory / filename token for mosaics that skip overlap box-splitting and
# combine every frame in a group+filter (see ``split_observations.as_full_group``).
FULL_GROUP_LABEL = 'full'


def mosaic_box_dirname(box_id: int | str) -> str:
    """Return ``ref_<box_id>`` (e.g. ``ref_0`` or ``ref_full``)."""
    return f'ref_{box_id}'


# Persisted shared sky stamp for each ``ref_*`` box (JWST + HST inherit this).
STAMP_WCS_BASENAME = 'stamp_wcs.fits'


def mosaic_coadd_basename(
    group_id: int | str,
    box_id: int | str,
    filter_name: str,
) -> str:
    """Return ``coadd_{group}_{box}_{filter}_i2d.fits``."""
    filt = str(filter_name).lower().replace(' ', '')
    return f'coadd_{group_id}_{box_id}_{filt}_i2d.fits'


def mosaic_hst_coadd_basename(
    group_id: int | str,
    box_id: int | str,
    instrument: str,
    filter_name: str,
    *,
    suffix: str = 'drc',
    visit: str | None = None,
) -> str:
    """
    Return boxed HST coadd basename.

    Pattern: ``coadd_{group}_{box}_{inst}_{filter}[_{visit}]_{drc|drz}.fits``.
    """
    inst = str(instrument).lower().replace(' ', '').split('_')[0]
    filt = str(filter_name).lower().replace(' ', '')
    suf = str(suffix).lower().replace('.fits', '').lstrip('_')
    if visit:
        vid = str(visit).lower().replace(' ', '')
        return f'coadd_{group_id}_{box_id}_{inst}_{filt}_{vid}_{suf}.fits'
    return f'coadd_{group_id}_{box_id}_{inst}_{filt}_{suf}.fits'


def is_jwst_mosaic_instrument(instrument: str | None) -> bool:
    """True for NIRCam / MIRI (and NRC alias)."""
    key = str(instrument or '').lower().split('_')[0]
    return key in {'nircam', 'nrc', 'miri'}


def is_hst_mosaic_instrument(instrument: str | None) -> bool:
    """True for ACS / WFC3 / WFPC2."""
    key = str(instrument or '').lower().split('_')[0]
    return key in {'acs', 'wfc3', 'wfpc2', 'wfc'}


def nircam_sw_filter_code(filter_name: str) -> int:
    """
    Return the numeric JWST filter code used for NIRCam SW selection.

    Parameters
    ----------
    filter_name : str
        Filter name such as ``F150W2`` or ``f1130w``.

    Returns
    -------
    int
        Full numeric code (``F1130W`` → ``1130``). Returns a large sentinel
        when the name cannot be parsed.
    """
    match = re.match(r'f(\d+)', str(filter_name).lower())
    return int(match.group(1)) if match else 10**9


def is_nircam_sw_broadband(filter_name: str) -> bool:
    """
    True for NIRCam short-wavelength broadband filters (excludes narrowbands
    and MIRI, whose codes are ≥560).
    """
    name = str(filter_name).lower()
    if 'n' in name:
        return False
    return nircam_sw_filter_code(name) < 215


def mosaic_pixel_scale_arcsec(
    filter_name: str,
    instrument: str | None = None,
) -> float:
    """
    Recommended mosaic pixel scale (arcsec) for a JWST filter / instrument.

    Parameters
    ----------
    filter_name : str
        Filter name (e.g. ``F150W``, ``f444w``, ``F560W``).
    instrument : str, optional
        Instrument name when known (``NIRCAM`` / ``MIRI``). Used to
        disambiguate when the filter code alone is insufficient.

    Returns
    -------
    float
        Pixel scale in arcseconds per pixel (NIRCam SW 0.031, NIRCam LW
        0.063, MIRI 0.11).
    """
    filt = str(filter_name).strip().lower()
    inst = str(instrument or '').strip().lower()
    match = re.match(r'f(\d+)', filt)
    code = int(match.group(1)) if match else 0
    if 'miri' in inst or code >= 560:
        return MIRI_PIXEL_SCALE
    if code >= 240:
        return NIRCAM_LW_PIXEL_SCALE
    return NIRCAM_SW_PIXEL_SCALE


def _ensure_pc_cdelt_header(hdr: fits.Header, w: wcs.WCS | None = None) -> fits.Header:
    """Ensure ``PC*`` + ``CDELT*`` keywords exist for :func:`create_gwcs`."""
    out = hdr.copy()
    if w is None:
        w = wcs.WCS(out)
    if 'PC1_1' in out and 'CDELT1' in out:
        return out
    cd = np.asarray(w.pixel_scale_matrix, dtype=float)
    cdelt = np.array(
        [np.hypot(cd[0, 0], cd[1, 0]), np.hypot(cd[0, 1], cd[1, 1])],
        dtype=float,
    )
    if cd[0, 0] < 0:
        cdelt[0] *= -1.0
    if cd[1, 1] < 0:
        cdelt[1] *= -1.0
    # Avoid divide-by-zero for degenerate matrices.
    cdelt = np.where(np.abs(cdelt) > 0, cdelt, np.array([-1.0, 1.0]) * np.abs(cdelt).max())
    pc = cd / cdelt
    out['CDELT1'] = float(cdelt[0])
    out['CDELT2'] = float(cdelt[1])
    out['PC1_1'] = float(pc[0, 0])
    out['PC1_2'] = float(pc[0, 1])
    out['PC2_1'] = float(pc[1, 0])
    out['PC2_2'] = float(pc[1, 1])
    for key in ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'):
        if key in out:
            del out[key]
    return out


def rescale_wcs_to_pixel_scale(
    box_wcs: wcs.WCS,
    pixel_scale_arcsec: float,
) -> fits.Header:
    """
    Rebuild a mosaic WCS header at a new pixel scale, keeping the same sky footprint.

    The output covers the same RA/Dec extent and orientation as ``box_wcs``, but
    with ``pixel_scale_arcsec`` sampling (so NIRCam SW / LW / MIRI can share a
    bounding box while using their recommended mosaic scales).

    Parameters
    ----------
    box_wcs : astropy.wcs.WCS
        Existing box WCS (typically a slice of the group mosaic WCS).
    pixel_scale_arcsec : float
        Desired output pixel scale in arcseconds.

    Returns
    -------
    fits.Header
        FITS WCS header with ``NAXIS*``, ``CRPIX*``, ``CRVAL*``, ``PC*``, and
        ``CDELT*`` suitable for :func:`create_gwcs`.
    """
    celestial = box_wcs.celestial
    # wcs.utils.proj_plane_pixel_scales returns degrees (plain floats).
    old_scale = float(np.asarray(wcs.utils.proj_plane_pixel_scales(celestial))[0]) * 3600.0
    if not np.isfinite(old_scale) or old_scale <= 0:
        raise ValueError(f'Invalid existing mosaic pixel scale: {old_scale}')
    scale_factor = float(pixel_scale_arcsec) / old_scale

    if box_wcs.pixel_shape is not None:
        naxis1, naxis2 = (int(box_wcs.pixel_shape[0]), int(box_wcs.pixel_shape[1]))
    else:
        naxis1, naxis2 = (int(box_wcs._naxis[0]), int(box_wcs._naxis[1]))

    new_naxis1 = max(1, int(np.ceil(naxis1 / scale_factor)))
    new_naxis2 = max(1, int(np.ceil(naxis2 / scale_factor)))
    crpix = celestial.wcs.crpix
    new_crpix = [
        (float(crpix[0]) - 0.5) / scale_factor + 0.5,
        (float(crpix[1]) - 0.5) / scale_factor + 0.5,
    ]

    new = wcs.WCS(naxis=2)
    new.wcs.crpix = new_crpix
    new.wcs.crval = celestial.wcs.crval.copy()
    new.wcs.ctype = list(celestial.wcs.ctype)
    new.wcs.cd = celestial.pixel_scale_matrix * scale_factor
    new.pixel_shape = (new_naxis1, new_naxis2)
    hdr = new.to_header()
    hdr['NAXIS1'] = new_naxis1
    hdr['NAXIS2'] = new_naxis2
    return _ensure_pc_cdelt_header(hdr, new)


def slice_box_wcs(mosaic_wcs: wcs.WCS, bbox) -> wcs.WCS:
    """
    Slice a group mosaic WCS to the pixel bounds of one overlap box.

    Parameters
    ----------
    mosaic_wcs : astropy.wcs.WCS
        Full-group WCS used for box planning (``MosaicBox.wcs``).
    bbox : shapely polygon
        Box footprint in *mosaic_wcs* pixel coordinates.

    Returns
    -------
    astropy.wcs.WCS
        WCS covering only this box's sky footprint (pixel shape set).
    """
    minx = int(np.abs(np.floor(min(bbox.exterior.xy[0]))))
    maxx = int(np.abs(np.ceil(max(bbox.exterior.xy[0]))))
    miny = int(np.abs(np.floor(min(bbox.exterior.xy[1]))))
    maxy = int(np.abs(np.ceil(max(bbox.exterior.xy[1]))))
    box_wcs = mosaic_wcs.slice((slice(miny, maxy), slice(minx, maxx)))
    if getattr(box_wcs, '_naxis', None) is not None:
        box_wcs.pixel_shape = (int(box_wcs._naxis[0]), int(box_wcs._naxis[1]))
    return box_wcs


def local_bbox_for_wcs(box_wcs: wcs.WCS):
    """Shapely pixel bbox covering the full array of a sliced stamp WCS."""
    import shapely

    nx, ny = box_wcs.pixel_shape or (
        int(box_wcs._naxis[0]),
        int(box_wcs._naxis[1]),
    )
    return shapely.box(0.0, 0.0, float(nx), float(ny))


def stamp_sky_center(box_wcs: wcs.WCS) -> tuple[float, float]:
    """RA/Dec (degrees) of the stamp array center."""
    nx, ny = box_wcs.pixel_shape or (
        int(box_wcs._naxis[0]),
        int(box_wcs._naxis[1]),
    )
    ra, dec = box_wcs.pixel_to_world_values(0.5 * (nx - 1), 0.5 * (ny - 1))
    return float(ra), float(dec)


def stamp_sky_polygon(box_wcs: wcs.WCS):
    """Shapely polygon of the stamp footprint in RA/Dec degrees."""
    import shapely

    return shapely.Polygon(box_wcs.calc_footprint(center=False))


def stamp_diagonal_deg(box_wcs: wcs.WCS) -> float:
    """Approximate stamp diagonal on-sky in degrees."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    corners = np.asarray(box_wcs.calc_footprint(center=False), dtype=float)
    coords = SkyCoord(ra=corners[:, 0] * u.deg, dec=corners[:, 1] * u.deg)
    return float(coords[0].separation(coords[2]).deg)


def write_stamp_wcs(outdir: str | Path, box_wcs: wcs.WCS) -> Path:
    """
    Persist the shared stamp WCS for a ``ref_*`` box.

    Stores WCS keywords plus ``MOSNX``/``MOSNY`` (no full-size image data).
    JWST and HST remosaics load this file so products stay on one sky grid.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / STAMP_WCS_BASENAME
    nx, ny = box_wcs.pixel_shape or (
        int(box_wcs._naxis[0]),
        int(box_wcs._naxis[1]),
    )
    hdr = box_wcs.to_header(relax=True)
    hdr['MOSNX'] = (int(nx), 'Stamp NAXIS1 (pixels)')
    hdr['MOSNY'] = (int(ny), 'Stamp NAXIS2 (pixels)')
    hdr['STAMPWCS'] = (True, 'Shared mosaic stamp WCS')
    fits.PrimaryHDU(header=hdr).writeto(path, overwrite=True)
    return path


def load_stamp_wcs(outdir: str | Path) -> wcs.WCS | None:
    """
    Load a persisted stamp WCS from ``stamp_wcs.fits`` or a boxed ``*_i2d``.

    Returns ``None`` when neither is available.
    """
    out = Path(outdir)
    stamp_path = out / STAMP_WCS_BASENAME
    if stamp_path.is_file():
        with fits.open(stamp_path) as hdul:
            hdr = hdul[0].header
            stamp = wcs.WCS(hdr, naxis=2)
            nx = int(hdr.get('MOSNX') or hdr.get('NAXIS1') or 0)
            ny = int(hdr.get('MOSNY') or hdr.get('NAXIS2') or 0)
            if nx > 0 and ny > 0:
                stamp.pixel_shape = (nx, ny)
                stamp._naxis = [nx, ny]
            return stamp

    for pattern in (
        'coadd_*_f150w2_i2d.fits',
        'coadd_*_f200w_i2d.fits',
        'coadd_*_f150w_i2d.fits',
        'coadd_*_i2d.fits',
    ):
        cands = sorted(out.glob(pattern))
        if not cands:
            continue
        with fits.open(cands[0]) as hdul:
            sci = hdul['SCI'] if 'SCI' in hdul else hdul[1]
            stamp = wcs.WCS(sci.header, naxis=2)
            ny, nx = sci.data.shape[-2:]
        stamp.pixel_shape = (int(nx), int(ny))
        stamp._naxis = [int(nx), int(ny)]
        return stamp
    return None


def ensure_box_stamp_wcs(
    box: 'MosaicBox',
    *,
    mosaic_wcs: wcs.WCS | None = None,
    bbox=None,
) -> wcs.WCS:
    """
    Resolve the shared stamp WCS for *box*, writing ``stamp_wcs.fits`` if needed.

    Prefers an on-disk stamp (stable across replans), else slices *mosaic_wcs*
    / ``box.wcs`` by *bbox* / ``box.bbox``. Updates ``box.wcs`` and ``box.bbox``
    to the local stamp frame.
    """
    loaded = load_stamp_wcs(box.outdir)
    if loaded is not None:
        box.wcs = loaded
        box.bbox = local_bbox_for_wcs(loaded)
        return loaded

    mw = mosaic_wcs if mosaic_wcs is not None else box.wcs
    bb = bbox if bbox is not None else box.bbox
    if mw is None:
        raise ValueError(f'No WCS available for stamp under {box.outdir}')
    if bb is None:
        stamp = mw
    else:
        stamp = slice_box_wcs(mw, bb)
    write_stamp_wcs(box.outdir, stamp)
    box.wcs = stamp
    box.bbox = local_bbox_for_wcs(stamp)
    return stamp


def assign_stable_box_ids(
    group_outdir: str | Path,
    candidate_wcs_list: Sequence[wcs.WCS],
    *,
    match_frac: float = 0.35,
) -> list[int]:
    """
    Match planned stamp centers to existing ``ref_*`` directories.

    When an on-disk stamp center lies within ``match_frac`` of the candidate
    stamp diagonal, reuse that directory's integer id so replans cannot
    renumber a sky region into a different ``ref_N``. Unmatched candidates
    receive the lowest unused non-negative integer ids.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    group_out = Path(group_outdir)
    existing: list[tuple[int, float, float, float]] = []
    if group_out.is_dir():
        for child in sorted(group_out.glob('ref_*')):
            if not child.is_dir():
                continue
            suffix = child.name[len('ref_') :]
            try:
                eid = int(suffix)
            except ValueError:
                continue
            loaded = load_stamp_wcs(child)
            if loaded is None:
                continue
            ra, dec = stamp_sky_center(loaded)
            existing.append((eid, ra, dec, stamp_diagonal_deg(loaded)))

    used: set[int] = set()
    assigned: list[int] = []
    for cand in candidate_wcs_list:
        cra, cdec = stamp_sky_center(cand)
        cdiag = max(stamp_diagonal_deg(cand), 1e-8)
        c_coord = SkyCoord(ra=cra * u.deg, dec=cdec * u.deg)
        best_id: int | None = None
        best_sep = np.inf
        for eid, era, edec, ediag in existing:
            if eid in used:
                continue
            sep = float(
                c_coord.separation(
                    SkyCoord(ra=era * u.deg, dec=edec * u.deg)
                ).deg
            )
            thresh = float(match_frac) * min(cdiag, max(ediag, 1e-8))
            if sep <= thresh and sep < best_sep:
                best_sep = sep
                best_id = eid
        if best_id is None:
            claimed = set(used) | {e[0] for e in existing}
            nid = 0
            while nid in claimed:
                nid += 1
            best_id = nid
        used.add(int(best_id))
        assigned.append(int(best_id))
    return assigned


def filter_frames_overlapping_box(
    paths: Sequence[str | Path],
    bbox,
    *,
    mosaic_wcs: wcs.WCS | None = None,
    min_overlap: float = 0.01,
) -> list[str]:
    """
    Keep frames whose sky footprint overlaps *bbox* by ≥ *min_overlap*.

    When *mosaic_wcs* is given, *bbox* is interpreted in that WCS's pixel
    space (as produced by :class:`split_observations`). Otherwise *bbox*
    must already be a sky-coordinate polygon (RA/Dec degrees).
    """
    import shapely

    if bbox is None:
        return [str(p) for p in paths]

    if mosaic_wcs is not None:
        box_poly = shapely.geometry.shape(bbox) if not hasattr(bbox, 'intersection') else bbox
    else:
        box_poly = bbox

    kept: list[str] = []
    for path in paths:
        try:
            sky = _sky_footprint_coords(path)
            if mosaic_wcs is not None:
                px, py = mosaic_wcs.world_to_pixel_values(sky[:, 0], sky[:, 1])
                frame_poly = shapely.Polygon(
                    np.column_stack([np.asarray(px, dtype=float), np.asarray(py, dtype=float)])
                )
            else:
                frame_poly = shapely.Polygon(sky)
            if not frame_poly.is_valid:
                frame_poly = frame_poly.buffer(0)
            if frame_poly.is_empty or frame_poly.area <= 0:
                continue
            frac = box_poly.intersection(frame_poly).area / frame_poly.area
            if frac >= float(min_overlap):
                kept.append(str(path))
        except Exception as exc:
            logger.warning(
                'Could not test box overlap for %s (%s); dropping',
                path,
                exc,
            )
    return kept


def mp_init(
    init_success: int = 0,
    init_failed: int = 0,
    init_success_files: list[str] | None = None,
) -> None:
    """
    Initialize module-level counters for multiprocessing mosaic workers.

    Parameters
    ----------
    init_success : int, optional
        Initial success count.
    init_failed : int, optional
        Initial failure count.
    init_success_files : list of str, optional
        Paths of successfully processed files.

    Returns
    -------
    None
    """
    if init_success_files is None:
        init_success_files = []
    global success
    global failed
    global success_files
    success = init_success
    failed = init_failed
    success_files = init_success_files

class split_observations(object):
    """
    Split an input list into spatial boxes for Level-3 mosaic / coadd groups.

    Footprints are derived from ``S_REGION`` polygons (or caller-supplied
    shapely geometries) and recursively subdivided until each box contains at
    most ``N_max`` images.
    """

    def __init__(
        self,
        table: Table,
        N_max: int = 150,
        min_overlap: float = 0.2,
        pad: float = 15,
        polygons: Sequence[Any] | None = None,
        wcs_opt: wcs.WCS | None = None,
        weight_pgons: Sequence[Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        table : Table
            Input list with an ``image`` column.
        N_max : int, optional
            Maximum images per mosaic box (science + weight footprints).
        min_overlap : float, optional
            Minimum fractional overlap required to assign an image to a box.
        pad : float, optional
            Pixel padding applied to each accepted box.
        polygons : sequence, optional
            Precomputed shapely polygons (one per table row).
        wcs_opt : astropy.wcs.WCS, optional
            Pixel WCS matching ``polygons`` when supplied.
        weight_pgons : sequence, optional
            Extra footprints (e.g. MIRI + HST ACS/WFC3 chips) that count
            toward ``N_max`` but are not mosaicked as JWST science frames.
        """
        self.table = table
        self.N_max = N_max
        self.min_overlap = min_overlap
        self.pad = pad

        if polygons is None:
            self.wcs, self.pgons, self.centroids = self.get_pgons(table)
        else:
            self.pgons = np.array(polygons)
            self.wcs = wcs_opt
            self.centroids = np.array([i.centroid for i in polygons])

        if weight_pgons is None:
            self.weight_pgons = np.array([], dtype=object)
        else:
            self.weight_pgons = np.asarray(weight_pgons, dtype=object)

        self.split_boxes = []
        self.subimages = []
        self.reftables = []
        self.refpgons = []
        self.filtertables = []
        self.physical_split = True
        self.check_box = None

    def count_weight_overlaps(self, bbox, *, min_overlap: float | None = None) -> int:
        """Number of weight footprints overlapping *bbox* above *min_overlap*."""
        if self.weight_pgons is None or len(self.weight_pgons) == 0:
            return 0
        tol = self.min_overlap if min_overlap is None else float(min_overlap)
        n = 0
        for p in self.weight_pgons:
            try:
                if p is None or getattr(p, 'area', 0) <= 0:
                    continue
                frac = bbox.intersection(p).area / p.area
                if frac > tol:
                    n += 1
            except Exception:
                continue
        return n

    def as_full_group(self) -> split_observations:
        """
        Collapse to one box covering every image in the table.

        Ignores ``N_max`` / overlap partitioning used by :meth:`boxsplit`.
        Intended for whole-group mosaics (e.g. all MIRI frames of one filter
        in a visit group) written under ``ref_full/``.

        Returns
        -------
        split_observations
            ``self``, for chaining.
        """
        bbox = self.get_bbox(pgons=self.pgons)
        bbox = self.pad_box(bbox, pad=self.pad)
        self.split_boxes = [bbox]
        self.subimages = [np.asarray(self.table['image'])]
        self.reftables = [self.table]
        self.refpgons = [self.pgons]
        self.filtertables = []
        return self

    def get_pgons(self, table):
        inputs = [_shape_and_wcs_for_mosaic(i) for i in table['image']]
        wcs_out, shape_out = find_optimal_celestial_wcs(inputs, auto_rotate=True)

        wcs_header = wcs_out.to_header()
        wcs_header['NAXIS1'] = shape_out[1]
        wcs_header['NAXIS2'] = shape_out[0]
        wcs_opt = wcs.WCS(wcs_header)
        
        pgons, centroids = [], []
        for im in table['image']:
            sky = _sky_footprint_coords(im)
            x, y = wcs_opt.all_world2pix(sky[:, 0], sky[:, 1], 0)
            xy_coords = np.column_stack((x, y))
            pgons.append(shapely.Polygon(xy_coords))
            centroids.append(shapely.Polygon(xy_coords).centroid)

        pgons, centroids = np.array(pgons), np.array(centroids)

        return wcs_opt, pgons, centroids
    
    def line_split(self, bounds, xs, ys, split_size=150, split_by='x', physical=False):
        lstrings = []
        
        if split_by == 'x':
            if physical:
                i = (bounds[0]+bounds[2])/2
                lstrings.append(shapely.LineString([[i, bounds[1]], [i, bounds[3]]]))
            else:
                cst = np.argsort(xs)
                xs, ys = np.array(xs)[cst], np.array(ys)[cst]
                split_lines = xs[0::split_size][1:]
                for i in split_lines:
                    lstrings.append(shapely.LineString([[i, bounds[1]], [i, bounds[3]]]))

        elif split_by == 'y':
            if physical:
                i = (bounds[1]+bounds[3])/2
                lstrings.append(shapely.LineString([[bounds[0], i], [bounds[2], i]]))
            else:
                cst = np.argsort(ys)
                xs, ys = np.array(xs)[cst], np.array(ys)[cst]
                split_lines = ys[0::split_size][1:]
                for i in split_lines:
                    lstrings.append(shapely.LineString([[bounds[0], i], [bounds[2], i]]))

        else:
            raise ValueError("Must be split by x or y")
        
        lstrings = shapely.MultiLineString(lstrings)

        return lstrings
    
    def find_intersections(self, bbox=None, min_overlap=0.05):

        if bbox is None:
            bbox = self.split_boxes

        int_area = np.array([bbox.intersection(p).area/p.area for p in self.pgons])
        mask = int_area > min_overlap
        return mask
    
    def get_bbox(self, pgons=None):
        
        if pgons is None:
            pgons = self.pgons

        bounds = shapely.unary_union(pgons).bounds
        bbox_coords = [[bounds[0], bounds[1]], [bounds[0], bounds[3]], [bounds[2], bounds[3]], [bounds[2], bounds[1]]]
        bbox = shapely.Polygon(bbox_coords)

        return bbox
    
    def pad_box(self, box, pad=0):

        bounds = box.bounds
        padded_box_coords = [[bounds[0]-pad, bounds[1]-pad], [bounds[0]-pad, bounds[3]+pad], [bounds[2]+pad, bounds[3]+pad], [bounds[2]+pad, bounds[1]-pad]]
        padded_box_coords = np.array(padded_box_coords)
        padded_box_coords[padded_box_coords < 0] = 0
        bbox = shapely.Polygon(padded_box_coords)

        return bbox
    
    def boxsplit(self, bbox=None, split_n=None):

        self.check_box = bbox

        if bbox is None:
            bbox = self.get_bbox(pgons=self.pgons)

        mask = self.find_intersections(bbox, min_overlap=self.min_overlap)
        n_sci = int(mask.sum())
        n_weight = self.count_weight_overlaps(bbox)
        n_total = n_sci + n_weight
        if n_total < self.N_max:
            bbox = self.pad_box(bbox, pad=self.pad)
            self.split_boxes.append(bbox)
            self.subimages.append(self.table[mask]['image'])
            ref_mask = self.find_intersections(bbox, min_overlap=0.0)
            self.reftables.append(self.table[ref_mask])
            self.refpgons.append(self.pgons[ref_mask])
            return None

        else:
            spl_pgons, spl_centroids  = self.pgons[mask], self.centroids[mask]
            bounds = bbox.bounds
            width, height = bounds[3] - bounds[1], bounds[2] - bounds[0]
            split_by = 'x' if height > width else 'y'
            if split_n is None:
                # Prefer splitting on science centroids; if a box is heavy only
                # from weights, still bisect geometrically.
                split_n = max(len(spl_pgons) // 2 + 1, 2)

            xs = [i.x for i in spl_centroids] if len(spl_centroids) else [
                (bounds[0] + bounds[2]) / 2.0
            ]
            ys = [i.y for i in spl_centroids] if len(spl_centroids) else [
                (bounds[1] + bounds[3]) / 2.0
            ]
            lstrings = self.line_split(bounds, xs, ys, split_size=split_n, split_by=split_by, physical=self.physical_split)
            
            split_bbox = []
            for ln in lstrings.geoms:
                linesplit = shapely.ops.split(bbox, ln)
                if len(linesplit.geoms) > 1:
                    split_bbox.append(linesplit.geoms[0])
                    bbox = linesplit.geoms[1]
                else:
                    bbox = linesplit.geoms[0]
            split_bbox.append(bbox)

            for split_box in split_bbox:
                if self.check_box == split_box:
                    logger.warning(
                        'Cannot split further; science=%d weight=%d '
                        '(N_max=%d)',
                        n_sci,
                        n_weight,
                        self.N_max,
                    )
                    split_box = self.pad_box(split_box, pad=self.pad)
                    self.split_boxes.append(split_box)
                    self.subimages.append(self.table[mask]['image'])
                    ref_mask = self.find_intersections(split_box, min_overlap=0.0)
                    self.reftables.append(self.table[ref_mask])
                    self.refpgons.append(self.pgons[ref_mask])
                    continue
                _ = self.boxsplit(split_box)
            
            return None
        
    def get_sw_filter_tables(self, tol = 0.05):
        for b in range(len(self.split_boxes)):
            bbox, reftable, refpgons = self.split_boxes[b], self.reftables[b], self.refpgons[b]
            filters = np.unique(reftable['filter'])
            # Full numeric codes (F1130W → 1130). Truncating to three digits
            # previously misclassified MIRI F1130W as NIRCam SW.
            swmask = np.array([is_nircam_sw_broadband(i) for i in filters])

            filter_footprints = []
            for filter_name in filters[swmask]:
                filter_footprints.append(shapely.unary_union(refpgons[reftable['filter'] == filter_name]))
            filter_footprints = np.array(filter_footprints)
            shortwave_union = shapely.unary_union(filter_footprints)
            
            net_ref_pgon = shapely.unary_union(refpgons).intersection(bbox)
            best_tol = net_ref_pgon.difference(shortwave_union).area/net_ref_pgon.area
            tol = max(tol, best_tol)
            total_ref_area, uncovered_frac = net_ref_pgon.area, 1
            ref_filters = []

            while uncovered_frac > tol:
                max_int = np.argmax([i.intersection(net_ref_pgon).area/net_ref_pgon.area for i in filter_footprints])
                footprint = filter_footprints[max_int]
                ref_filters.append(filters[swmask][max_int])
                uncovered_frac = net_ref_pgon.difference(footprint).area/total_ref_area
                net_ref_pgon = net_ref_pgon.difference(footprint)

            filter_table = create_filter_table(reftable, ref_filters)
            self.filtertables.append(filter_table)

        return self.filtertables
    
    def add_plt_patch(self, pgon, ax, facecolor = 'lightblue', edgecolor = 'blue', alpha = 0.3):
        vertices = np.array(pgon.exterior.xy)
        polygon = Polygon(vertices.T, closed=True, facecolor=facecolor, edgecolor=edgecolor, alpha = alpha)
        ax.add_patch(polygon)

    def plot_obs(self, cents = False):
        fig, ax = plt.subplots(1, 1)
        for polygon in self.pgons:
            self.add_plt_patch(polygon, ax)
        for polygon in self.split_boxes:
            self.add_plt_patch(polygon, ax, facecolor = 'none', edgecolor = 'black', alpha = 1)
        if cents:
            ax.scatter([i.x for i in self.centroids], [i.y for i in self.centroids], color = 'mediumvioletred', s = 2)
        xmin, xmax = min([min(i.exterior.xy[0]) for i in self.pgons]), max([max(i.exterior.xy[0]) for i in self.pgons])
        ymin, ymax = min([min(i.exterior.xy[1]) for i in self.pgons]), max([max(i.exterior.xy[1]) for i in self.pgons])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('x (pix)')
        ax.set_ylabel('y (pix)')
        
        
def get_pgons(
    table: Table,
) -> tuple[wcs.WCS, np.ndarray, np.ndarray]:
    """
    Build pixel-space shapely footprints for each row in an input list.

    Parameters
    ----------
    table : Table
        Input list with an ``image`` column.

    Returns
    -------
    wcs_opt : astropy.wcs.WCS
        Optimal celestial WCS projected to pixels.
    pgons : numpy.ndarray
        Shapely polygons, one per image.
    centroids : numpy.ndarray
        Polygon centroids in pixel coordinates.
    """
    inputs = [_shape_and_wcs_for_mosaic(i) for i in table['image']]
    wcs_out, shape_out = find_optimal_celestial_wcs(inputs, auto_rotate=True)

    wcs_header = wcs_out.to_header()
    wcs_header['NAXIS1'] = shape_out[1]
    wcs_header['NAXIS2'] = shape_out[0]
    wcs_opt = wcs.WCS(wcs_header)
    
    pgons, centroids = [], []
    for im in table['image']:
        sky = _sky_footprint_coords(im)
        x, y = wcs_opt.all_world2pix(sky[:, 0], sky[:, 1], 0)
        xy_coords = np.column_stack((x, y))
        pgons.append(shapely.Polygon(xy_coords))
        centroids.append(shapely.Polygon(xy_coords).centroid)

    pgons, centroids = np.array(pgons), np.array(centroids)

    return wcs_opt, pgons, centroids
        
def create_default_mosaic(
    inputfiles: Sequence[str],
    outdir: str,
    filt: str,
) -> None:
    """
    Create a Level-3 drizzled mosaic from Level-2 inputs with default JWST options.

    Parameters
    ----------
    inputfiles : sequence of str
        Level-2 science FITS paths.
    outdir : str
        Output directory for association and pipeline products.
    filt : str
        Filter name used to select inputs and name outputs.

    Returns
    -------
    None
    """
    patch_jwst_for_photutils3()
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    table = input_list(inputfiles)
    table = table[table['filter'] == filt]
    preferred_ctx = None
    if len(table) > 0:
        try:
            preferred_ctx = fits.getval(str(table['image'][0]), 'CRDS_CTX', ext=0)
        except Exception:
            preferred_ctx = None
    ensure_local_crds_context(preferred=preferred_ctx)
    asn_file = f'{outdir}/{filt}.json'
    base_filenames = np.array([os.path.basename(r['image']) for r in table])
    asn3 = asn_from_list.asn_from_list(base_filenames,
        rule=DMS_Level3_Base, product_name=f'{filt}')

    with open(asn_file, 'w') as outfile:
        name, serialized = asn3.dump(format='json')
        outfile.write(serialized)

    image3 = calwebb_image3.Image3Pipeline()

    outdir_level3 = os.path.join(outdir, f'out_{filt}')
    if not os.path.exists(outdir_level3):
        os.makedirs(outdir_level3)

    image3.output_dir = outdir_level3
    image3.save_results = True
    image3.tweakreg.skip = True
    image3.skymatch.skip = True
    image3.skymatch.match_down = False
    image3.source_catalog.skip=False
    image3.resample.pixfrac = 1.0
    image3.pixel_scale = mosaic_pixel_scale_arcsec(filt)
    image3.weight_type = 'ivm'

    with capture_output():
        image3.run(asn_file)

def create_coadd_mosaic(
    table: Table,
    outdir: str,
    filt: str,
    *,
    sci_header: fits.Header | None = None,
    wcs_out: wcs.WCS | None = None,
    shape_out: tuple[int, int] | None = None,
    gwcs_file: str | None = None,
    pixel_scale: float | None = None,
) -> str:
    """
    Create a Level-3 drizzled mosaic with a shared output GWCS.

    A ``FITSImagingWCSTransform``-based output WCS is always applied
    (via :func:`create_gwcs`) so resample can write FITS WCS keywords
    under jwst>=1.20. Provide an existing ``gwcs_file``, or inputs to
    build one (``sci_header``, or ``wcs_out`` + ``shape_out``).

    Parameters
    ----------
    table : Table
        Input list of images to resample.
    outdir : str
        Output directory for association and pipeline products.
    filt : str
        Filter name for the resampled product.
    sci_header : fits.Header, optional
        FITS WCS header used to build ``mosaic_gwcs_<filt>.asdf`` when
        ``gwcs_file`` is not given.
    wcs_out : astropy.wcs.WCS, optional
        Astropy WCS used with ``shape_out`` when ``gwcs_file`` /
        ``sci_header`` are not given.
    shape_out : tuple of int, optional
        ``(NAXIS2, NAXIS1)`` for ``wcs_out``.
    gwcs_file : str, optional
        Path to an existing GWCS asdf file (from :func:`create_gwcs`).
    pixel_scale : float, optional
        Absolute mosaic pixel scale in arcsec. Defaults to the JWST
        recommended scale for ``filt`` (NIRCam SW/LW or MIRI). Ignored by
        resample when ``output_wcs`` / ``gwcs_file`` is set.

    Returns
    -------
    str
        Path to the resampled ``*_i2d.fits`` image.
    """
    patch_jwst_for_photutils3()
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if pixel_scale is None:
        inst = None
        if 'instrument' in table.colnames and len(table) > 0:
            inst = str(table['instrument'][0])
        pixel_scale = mosaic_pixel_scale_arcsec(filt, inst)

    # Prefer the CRDS context used to calibrate the input frames when cached.
    preferred_ctx = None
    if len(table) > 0:
        try:
            preferred_ctx = fits.getval(str(table['image'][0]), 'CRDS_CTX', ext=0)
        except Exception:
            preferred_ctx = None
    ensure_local_crds_context(preferred=preferred_ctx)

    if gwcs_file is None:
        gwcs_file = create_gwcs(
            outdir=outdir,
            sci_header=sci_header,
            wcs_out=wcs_out,
            shape_out=shape_out,
            filename=f'mosaic_gwcs_{str(filt).lower()}.asdf',
        )
    elif not os.path.exists(gwcs_file):
        raise FileNotFoundError(f'gwcs_file not found: {gwcs_file}')

    asn_file = f'{outdir}/{filt}.json'
    base_filenames = np.array([os.path.basename(r['image']) for r in table])
    asn3 = asn_from_list.asn_from_list(base_filenames,
        rule=DMS_Level3_Base, product_name=f'{filt}')

    with open(asn_file, 'w') as outfile:
        name, serialized = asn3.dump(format='json')
        outfile.write(serialized)

    image3 = calwebb_image3.Image3Pipeline()

    outdir_level3 = os.path.join(outdir, f'out_{filt}')
    if not os.path.exists(outdir_level3):
        os.makedirs(outdir_level3)

    image3.output_dir = outdir_level3
    image3.save_results = True
    image3.tweakreg.skip = True
    image3.skymatch.skip = True
    image3.skymatch.match_down = False
    image3.source_catalog.skip=False
    image3.resample.pixfrac = 1.0
    image3.pixel_scale = float(pixel_scale)
    image3.weight_type = 'ivm'
    image3.resample.output_wcs = gwcs_file

    with capture_output():
        image3.run(asn_file)

    filepath = f'{outdir}/out_{filt}/{filt}_i2d.fits'
    return filepath


def create_gwcs(
    outdir: str,
    sci_header: fits.Header | None = None,
    wcs_out: wcs.WCS | None = None,
    shape_out: tuple[int, int] | None = None,
    return_gwcs: bool = False,
    filename: str = 'mosaic_gwcs.asdf',
) -> str | g_wcs:
    """
    Convert an astropy WCS to GWCS and write or return it.

    Parameters
    ----------
    outdir : str
        Directory for the asdf product when ``return_gwcs`` is False.
    sci_header : fits.Header, optional
        FITS WCS header defining the output mosaic grid.
    wcs_out : astropy.wcs.WCS, optional
        Astropy WCS converted with ``shape_out`` when ``sci_header`` is omitted.
    shape_out : tuple of int, optional
        ``(NAXIS2, NAXIS1)`` shape paired with ``wcs_out``.
    return_gwcs : bool, optional
        When True, return the in-memory GWCS object instead of writing asdf.
    filename : str, optional
        Basename of the asdf file written under ``outdir``. Default
        ``mosaic_gwcs.asdf``.

    Returns
    -------
    str or gwcs.wcs.WCS
        Path to the asdf file, or the GWCS object when ``return_gwcs`` is True.
    """

    if sci_header:
        sci_header = _ensure_pc_cdelt_header(sci_header)
    elif wcs_out:
        sci_header = wcs_out.to_header()
        sci_header['NAXIS1'] = shape_out[1]
        sci_header['NAXIS2'] = shape_out[0]
        sci_header = _ensure_pc_cdelt_header(sci_header, wcs_out)
    else:
        raise ValueError("Please provide header or wcs object")

    # jwst>=1.20 ResampleImage.update_fits_wcsinfo expects a
    # FITSImagingWCSTransform (with .crpix/.cdelt/.crval/.pc). A plain
    # CompoundModel of Shift|Affine|Scale|TAN|Rotate raises
    # AttributeError: Attribute "crpix" not found (jwst#10377).
    # crpix here is 0-indexed detector pixels (FITS CRPIX minus 1).
    matrix = np.array(
        [
            [sci_header['PC1_1'], sci_header['PC1_2']],
            [sci_header['PC2_1'], sci_header['PC2_2']],
        ]
    )
    det2sky = FITSImagingWCSTransform(
        models.Pix2Sky_TAN(),
        crpix=[sci_header['CRPIX1'] - 1, sci_header['CRPIX2'] - 1],
        crval=[sci_header['CRVAL1'], sci_header['CRVAL2']],
        cdelt=[sci_header['CDELT1'], sci_header['CDELT2']],
        pc=matrix,
    )
    det2sky.name = 'linear_transform'

    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"),
                                unit=(u.pix, u.pix))
    sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), name='world',
                                unit=(u.deg, u.deg))

    pipeline = [(detector_frame, det2sky),
                (sky_frame, None)
            ]
    wcsobj = g_wcs(pipeline)
    wcsobj.bounding_box = ((0, sci_header['NAXIS1']), (0, sci_header['NAXIS2']))

    if return_gwcs:
        return wcsobj

    else:
        #write gwcs to asdf file
        tree = {"wcs": wcsobj}
        wcs_file = AsdfFile(tree)
        gwcs_path = os.path.join(outdir, filename)
        wcs_file.write_to(gwcs_path)

    return gwcs_path

def find_optimal_wcs(
    filter_table: dict[str, Table],
) -> tuple[wcs.WCS, tuple[int, int]]:
    """
    Find the optimal celestial WCS spanning all images in a filter table dict.

    Parameters
    ----------
    filter_table : dict
        Filter name mapped to an input-list :class:`~astropy.table.Table`.

    Returns
    -------
    wcs_out : astropy.wcs.WCS
        Optimal output WCS.
    shape_out : tuple of int
        ``(NAXIS2, NAXIS1)`` shape for the mosaic grid.
    """
    images = np.hstack([filter_table[i]['image'].value for i in filter_table.keys()])
    inputs = [_shape_and_wcs_for_mosaic(i) for i in images]
    wcs_out, shape_out = find_optimal_celestial_wcs(inputs, auto_rotate=True)

    return wcs_out, shape_out

def create_psf_kernel(
    ref_filter: str,
    in_filter: str,
    ovs: int = 5,
    fov: int = 81,
) -> np.ndarray:
    """
    Build a photutils PSF-matching kernel between two NIRCam filters.

    Parameters
    ----------
    ref_filter : str
        Reference filter name (e.g. ``F277W``).
    in_filter : str
        Source filter to match to ``ref_filter``.
    ovs : int, optional
        STPSF oversampling factor.
    fov : int, optional
        STPSF field of view in pixels.

    Returns
    -------
    numpy.ndarray
        Matching kernel for :func:`astropy.convolution.convolve_fft`.
    """
    nrc = stpsf.NIRCam()
    nrc.filter = in_filter.upper()
    if nrc.filter == 'F150W2':
        nrc.SHORT_WAVELENGTH_MAX = 2.39e-6
    nrc.detector = 'NRCA3'
    psf_src = nrc.calc_psf(oversample=ovs, fov_pixels=fov) 

    #use detector distorted version
    psf_src_dat = psf_src[3].data/psf_src[3].data.sum()

    nrc.filter = ref_filter.upper()
    if nrc.filter == 'F150W2':
        nrc.SHORT_WAVELENGTH_MAX = 2.39e-6
    psf_ref = nrc.calc_psf(oversample=ovs, fov_pixels=fov)
    psf_ref_dat = psf_ref[3].data/psf_ref[3].data.sum()

    window = SplitCosineBellWindow(1.5, 1.3)
    psf_kernel = create_matching_kernel(psf_src_dat, psf_ref_dat, window=window) 

    return psf_kernel

def convolve_images(
    filter_table: dict[str, Table],
    target_filter: str,
) -> None:
    """
    PSF-match and overwrite science images to ``target_filter`` in place.

    Parameters
    ----------
    filter_table : dict
        Filter name mapped to an input-list :class:`~astropy.table.Table`.
    target_filter : str
        Reference filter for PSF matching.

    Returns
    -------
    None
    """
    for filt in filter_table.keys():
        if filt.upper() == target_filter.upper():
            continue
        psf_kernel = create_psf_kernel(target_filter, filt)
        tbl = filter_table[filt]
        for im in tbl['image']:
            hdu = fits.open(im)
            sci, err = hdu['SCI'].data, hdu['ERR'].data
            sci_con = convolve_fft(sci, psf_kernel, normalize_kernel=True)
            err_con = convolve_fft(err, psf_kernel, normalize_kernel=True)
            sci_header = hdu['SCI'].header
            sci_header['filter'] = target_filter.upper()

            hdu['SCI'].header, hdu['SCI'].data = sci_header, sci_con
            hdu['ERR'].data = err_con
            hdu.writeto(im, overwrite=True)

def create_ccddata(file: str) -> CCDData:
    """
    Load a JWST i2d FITS file as :class:`~astropy.nddata.CCDData`.

    Parameters
    ----------
    file : str
        Path to a Level-3 or coadd FITS file with SCI/ERR extensions.

    Returns
    -------
    CCDData
        Science data, uncertainty, WCS, and zero mask.
    """
    hdu = fits.open(file)
    sci_data = hdu['SCI'].data
    
    uncertainty = nddata.StdDevUncertainty(array = hdu['ERR'].data)
    data_unit = u.MJy/u.sr
    w = wcs.WCS(hdu['SCI'].header)
    mask = sci_data == 0
    ccd_data = CCDData(data = sci_data, uncertainty = uncertainty, 
                       wcs = w, unit = data_unit)
    
    return ccd_data

def update_photmjsr(
    ccddata: Sequence[CCDData],
    phots: Sequence[float],
) -> float:
    """
    Estimate a combined PHOTMJSR from weighted coadd inputs.

    Parameters
    ----------
    ccddata : sequence of CCDData
        Per-image science arrays in MJy/sr.
    phots : sequence of float
        Per-image ``PHOTMJSR`` header values.

    Returns
    -------
    float
        Sigma-clipped median conversion factor (MJy/sr per count).
    """
    ccd_mjsr = np.sum([ccd.data for ccd in ccddata], axis = 0)
    ccd_cps = np.sum([ccd.data/phot for ccd, phot in list(zip(ccddata, phots))], axis = 0)
    mjsr = ccd_mjsr/ccd_cps
    _, mjsr_med, _ = scs(mjsr)

    return mjsr_med

def coadd(
    ref_files: Sequence[str],
    filt: str,
    filename: str = 'coadd_i2d.fits',
) -> None:
    """
    Inverse-variance coadd Level-3 images with ccdproc.

    Parameters
    ----------
    ref_files : sequence of str
        Input ``*_i2d.fits`` paths to combine.
    filt : str
        Filter name written to the coadd primary header.
    filename : str, optional
        Output coadd FITS path.

    Returns
    -------
    None
    """
    #edit specific header keys
    hdu_template = fits.open(ref_files[0])
    hdr_update = {'EFFEXPTM': [], 'TMEASURE': [], 'DURATION': []}
    filters, phots = [], []
    #WHT data for coadded image
    wht_data = []
    
    for file in ref_files:
        hdul = fits.open(file)
        for key in list(hdr_update.keys()):
            hdr_update[key].append(fits.getval(file, key, ext = 0))
        filter_name = fits.getval(file, 'FILTER', ext = 0)
        filters.append(filter_name)
        phots.append(fits.getval(file, 'PHOTMJSR', ext = 1))
        # #inverse variance weighting
        wht_data.append(hdul['WHT'].data/fits.getval(file, 'DURATION', ext = 0))
        hdul.close()

    combiner_weights = np.array(wht_data)
    combiner_weights /= np.sum(combiner_weights, axis = 0)
    
    for i, weight in enumerate(combiner_weights):
        invalid = np.isnan(weight) | np.isinf(weight)
        weight[invalid] = 0
        combiner_weights[i] = weight 
    combiner_weights = np.array(combiner_weights)
    
    #coadd images using ccdproc
    ccddata_ = []
    for file in ref_files:
        ccddata_.append(create_ccddata(file))
        
    combiner = Combiner(ccddata_)
    combiner.weights = combiner_weights
    combined_sum = combiner.sum_combine()

    #SCI and ERR data for coadded image
    coadd_data = combined_sum.data
    det_mask = coadd_data == 0
    quad_err = np.sqrt(np.sum([(wht_*ccd.uncertainty.array)**2 for ccd, wht_ in zip(ccddata_, combiner_weights)], axis = 0))
        
    primary_header, sci_header = hdu_template['PRIMARY'].header, hdu_template['SCI'].header
    err_header, wht_header = hdu_template['ERR'].header, hdu_template['WHT'].header
    primary_header['FILENAME'] = filename
    primary_header['FILTER'] = filt.upper()
    
    exptime_wt = [np.nanmean(i) for i in combiner_weights]
    for key in list(hdr_update.keys()):
        hdr_update[key] = np.sum(hdr_update[key])
        primary_header[key] = hdr_update[key]

    sci_header['PHOTMJSR'] = update_photmjsr(ccddata_, phots)
    sci_header['XPOSURE'] = hdr_update['EFFEXPTM']
    sci_header['TELAPSE'] = hdr_update['DURATION']

    primary_hdu = fits.PrimaryHDU(header = primary_header)
    sci_hdu = fits.ImageHDU(data = coadd_data, header = sci_header, name = 'SCI')
    err_hdu = fits.ImageHDU(data = quad_err, header = err_header, name = 'ERR')
    wht_hdu = fits.ImageHDU(data = np.sum(wht_data, axis = 0), header = wht_header, name = 'WHT')
    
    coadd_hdul = fits.HDUList([primary_hdu, sci_hdu, err_hdu, wht_hdu])
    coadd_hdul.writeto(filename, overwrite = True)
    hdu_template.close()

def create_dirs(base_dir: str, n: int = 1) -> dict[int, str]:
    """
    Create ``reference/group_*`` directories under a mosaic base directory.

    When ``base_dir`` is named ``reduction``, also symlink ``../reference`` to
    ``reduction/reference`` for align reference mode.

    Parameters
    ----------
    base_dir : str
        Mosaic reduction root (typically ``.../reduction``).
    n : int, optional
        Number of group directories to create.

    Returns
    -------
    dict
        Group index mapped to ``reference/group_<index>`` path.
    """
    out_dict = dict.fromkeys(range(n))
    os.makedirs(base_dir, exist_ok=True)
    for i in range(n):
        outdir = os.path.join(base_dir, f'reference/group_{i}')
        out_dict[i] = outdir
        os.makedirs(outdir, exist_ok=True)

    # When reducing under <data-root>/reduction, expose reference/ at the
    # dataset root so align --mode reference can find coadds without a manual ln/mkdir.
    ensure_dataset_reference_link(base_dir)

    return out_dict


def ensure_dataset_reference_link(base_dir: str) -> str | None:
    """
    Symlink dataset-root ``reference`` to ``reduction/reference``.

    Parameters
    ----------
    base_dir : str
        Reduction directory; no-op unless its basename is ``reduction``.

    Returns
    -------
    str or None
        Dataset-root reference path when linked or already present, else
        ``None``.
    """
    base = Path(base_dir).resolve()
    if base.name != 'reduction':
        return None
    src = base / 'reference'
    os.makedirs(src, exist_ok=True)
    dst = base.parent / 'reference'
    if dst.exists() or dst.is_symlink():
        return str(dst)
    try:
        os.symlink(src, dst)
    except OSError:
        return None
    return str(dst)

def copy_files(filter_table: dict[str, Table], outdir: str) -> None:
    """
    Copy all science images listed in a filter table dict into ``outdir``.

    Parameters
    ----------
    filter_table : dict
        Filter name mapped to an input-list :class:`~astropy.table.Table`.
    outdir : str
        Destination directory.

    Returns
    -------
    None
    """
    infiles = np.hstack([filter_table[i]['image'].value for i in filter_table.keys()])
    for file in infiles:
        shutil.copy(file, outdir)

def update_path(
    filter_table: dict[str, Table],
    outdir: str,
) -> dict[str, Table]:
    """
    Rewrite ``image`` paths in a filter table dict to basenames under ``outdir``.

    Parameters
    ----------
    filter_table : dict
        Filter name mapped to an input-list :class:`~astropy.table.Table`.
    outdir : str
        Directory containing copied FITS files.

    Returns
    -------
    dict
        Updated filter table (same object, mutated in place).
    """
    for flt in filter_table.keys():
        tbl = filter_table[flt]
        filenames = [os.path.basename(i['image']) for i in tbl]
        tbl['image'] = [os.path.join(outdir, i) for i in filenames]
        filter_table[flt] = tbl
    
    return filter_table

def write_dolphot_frame_list(
    box_outdir: str,
    *,
    refimage: str,
    frames: Sequence[str],
    group: int = 0,
    box: int | str = 0,
) -> str:
    """
    Write a manifest of coadd + JHAT frames for ``dolphot-prep`` mosaic discovery.

    Parameters
    ----------
    box_outdir : str
        Mosaic box directory (``reference/group_G/ref_B`` or ``…/ref_full``).
    refimage : str
        Path to the coadd ``*_i2d.fits`` reference.
    frames : sequence
        JHAT frame paths belonging to this box.
    group : int, optional
        Group index used by ``dolphot-prep`` for ``phot_{group}_{box}``.
    box : int or str, optional
        Box index (``0``, ``1``, …) or :data:`FULL_GROUP_LABEL` (``'full'``).

    Returns
    -------
    str
        Path to ``dolphot_frames.txt``.
    """
    out = os.path.join(box_outdir, 'dolphot_frames.txt')
    with open(out, 'w') as fh:
        fh.write(f'# group={int(group)} box={box}\n')
        fh.write(f'# ref {os.path.abspath(refimage)}\n')
        for path in frames:
            fh.write(f'{os.path.abspath(path)}\n')
    return out


@dataclass
class MosaicBox:
    """One overlap box under ``reference/group_G/ref_B``."""

    group_id: int
    box_id: int | str
    outdir: Path
    bbox: Any
    frames: list[str] = field(default_factory=list)
    wcs: Any | None = None

    def frames_for_mission(self, mission: str) -> list[str]:
        """Filter box frames to ``jwst`` or ``hst`` by FITS ``INSTRUME``."""
        from st123.utils.helpers import get_instrument

        want = str(mission).lower()
        out: list[str] = []
        for path in self.frames:
            try:
                inst = get_instrument(path)
            except Exception:
                continue
            if want == 'jwst' and is_jwst_mosaic_instrument(inst):
                out.append(path)
            elif want == 'hst' and is_hst_mosaic_instrument(inst):
                out.append(path)
        return out


@dataclass
class MosaicPlan:
    """Shared footprint plan for JWST and/or HST mosaics."""

    base_dir: Path
    reference_dir: Path
    table: Table
    boxes: list[MosaicBox] = field(default_factory=list)
    group_dirs: dict[int, str] = field(default_factory=dict)


def plan_mosaic_boxes(
    base_dir: str | Path,
    inputfiles: Sequence[str | Path],
    *,
    nmax: int = 150,
    full_group: bool = False,
    footprint_weights: str = 'auto',
    spec_group_file: str | None = None,
    verbose: bool = False,
) -> MosaicPlan:
    """
    Build a joint ``group_*/ref_*`` mosaic plan from JHAT science frames.

    Groups come from footprint connectivity (:func:`~st123.utils.helpers.edit_visits_groups`);
    boxes are recursive ``N_max`` splits (or one ``ref_full`` when *full_group*).
    Each box stores a sliced shared stamp WCS (``stamp_wcs.fits``) so JWST and
    HST remosaics share one sky grid. Integer ``ref_N`` ids are matched to
    existing on-disk stamps by sky center so replans do not renumber regions.
    When *footprint_weights* is ``auto`` and not *full_group*, overlapping
    footprints from missions **not** already in *inputfiles* still count toward
    ``N_max`` (legacy JWST-only weighting of MIRI/HST).
    """
    base = Path(base_dir).expanduser().resolve()
    paths = [str(Path(p).expanduser().resolve()) for p in inputfiles]
    if not paths:
        raise ValueError(f'No input JHAT frames for mosaic plan under {base}')

    table = input_list(paths)
    if spec_group_file:
        table = edit_spec_groups(table, spec_group_file)
    ngroups = np.unique(table['group'])
    group_dirs = create_dirs(str(base), len(ngroups))
    plan = MosaicPlan(
        base_dir=base,
        reference_dir=base / 'reference',
        table=table,
        group_dirs={int(k): v for k, v in group_dirs.items()},
    )

    weights_mode = str(footprint_weights or 'none').lower()
    for group_id in ngroups:
        gid = int(group_id)
        group_table = table[table['group'] == group_id]
        outdir = group_dirs[gid]
        weight_pgons = None
        if weights_mode == 'auto' and not full_group:
            from st123.mosaic.footprints import (
                build_weight_pgons,
                collect_footprint_weight_paths,
            )

            split_seed = split_observations(table=group_table, N_max=nmax)
            weight_files = collect_footprint_weight_paths(
                base,
                exclude=list(group_table['image']),
            )
            if weight_files and split_seed.wcs is not None:
                weight_pgons = build_weight_pgons(weight_files, split_seed.wcs)
                if verbose:
                    logger.info(
                        'Group %s: %d footprint weight polygon(s) '
                        'count toward N_max=%d',
                        gid,
                        len(weight_pgons),
                        nmax,
                    )
            split_obs = split_observations(
                table=group_table,
                N_max=nmax,
                polygons=split_seed.pgons,
                wcs_opt=split_seed.wcs,
                weight_pgons=weight_pgons,
            )
        else:
            split_obs = split_observations(table=group_table, N_max=nmax)

        if full_group:
            split_obs.as_full_group()
            box_ids: list[int | str] = [FULL_GROUP_LABEL]
            if verbose:
                logger.info(
                    'Full-group mosaic: group %s (%d frames) → %s/',
                    gid,
                    len(group_table),
                    mosaic_box_dirname(FULL_GROUP_LABEL),
                )
        else:
            split_obs.boxsplit()
            if verbose:
                logger.info(
                    'Group %s: %d mosaic box(es) after split '
                    '(science frames in table=%d)',
                    gid,
                    len(split_obs.split_boxes),
                    len(group_table),
                )

        group_frames = [str(p) for p in group_table['image']]
        n_boxes = len(split_obs.split_boxes)
        sliced_wcs = [
            slice_box_wcs(split_obs.wcs, split_obs.split_boxes[i])
            for i in range(n_boxes)
        ]
        if full_group:
            box_ids = [FULL_GROUP_LABEL]
        else:
            box_ids = assign_stable_box_ids(outdir, sliced_wcs)
            if verbose and box_ids != list(range(n_boxes)):
                logger.info(
                    'Group %s: stable box ids %s (matched existing stamps when possible)',
                    gid,
                    box_ids,
                )

        for i, box_id in enumerate(box_ids):
            box_outdir = Path(outdir) / mosaic_box_dirname(box_id)
            box_outdir.mkdir(parents=True, exist_ok=True)
            # Prefer an on-disk stamp WCS so remosaics keep the same sky grid.
            stamp = load_stamp_wcs(box_outdir)
            if stamp is None:
                stamp = sliced_wcs[i]
                write_stamp_wcs(box_outdir, stamp)
            sky_poly = stamp_sky_polygon(stamp)
            # Filter against stamp sky (not just the latest split index) so
            # stable-id rematches still select the right JHAT frames.
            frames = filter_frames_overlapping_box(
                group_frames,
                sky_poly,
                mosaic_wcs=None,
                min_overlap=0.01,
            )
            if not frames:
                # Fall back to the planner's split membership.
                frames = [str(p) for p in split_obs.subimages[i]]
            plan.boxes.append(
                MosaicBox(
                    group_id=gid,
                    box_id=box_id,
                    outdir=box_outdir,
                    bbox=local_bbox_for_wcs(stamp),
                    frames=frames,
                    wcs=stamp,
                )
            )
    return plan


def plan_existing_box(
    base_dir: str | Path,
    box_outdir: str | Path,
    inputfiles: Sequence[str | Path],
    *,
    min_overlap: float = 0.01,
    verbose: bool = False,
) -> MosaicPlan:
    """
    Build a one-box mosaic plan from an existing coadd stamp directory.

    Uses ``stamp_wcs.fits`` when present, else a boxed ``*_i2d.fits`` (prefer
    F150W2/F200W) as the shared sky WCS so remosaics keep the on-disk stamp
    without re-running overlap box-splitting.
    """
    base = Path(base_dir)
    out = Path(box_outdir)
    if not out.is_dir():
        raise FileNotFoundError(f'existing box directory not found: {out}')

    stamp_wcs = load_stamp_wcs(out)
    if stamp_wcs is None:
        raise FileNotFoundError(
            f'no {STAMP_WCS_BASENAME} or coadd_*_i2d.fits stamp in {out}'
        )
    stamp_label = STAMP_WCS_BASENAME
    if not (out / STAMP_WCS_BASENAME).is_file():
        for pattern in (
            'coadd_*_f150w2_i2d.fits',
            'coadd_*_f200w_i2d.fits',
            'coadd_*_f150w_i2d.fits',
            'coadd_*_i2d.fits',
        ):
            cands = sorted(out.glob(pattern))
            if cands:
                stamp_label = cands[0].name
                break
        write_stamp_wcs(out, stamp_wcs)

    sky_poly = stamp_sky_polygon(stamp_wcs)
    bbox = local_bbox_for_wcs(stamp_wcs)

    kept = filter_frames_overlapping_box(
        [str(p) for p in inputfiles],
        sky_poly,
        mosaic_wcs=None,
        min_overlap=float(min_overlap),
    )
    if verbose:
        logger.info(
            'Existing-box plan %s: stamp=%s frames=%d/%d',
            out,
            stamp_label,
            len(kept),
            len(list(inputfiles)),
        )

    # Infer group/box ids from directory names when possible.
    group_id = 0
    box_id: int | str = out.name.replace('ref_', '') if out.name.startswith('ref_') else out.name
    try:
        box_id = int(box_id)
    except ValueError:
        pass
    if out.parent.name.startswith('group_'):
        try:
            group_id = int(out.parent.name.replace('group_', ''))
        except ValueError:
            group_id = 0

    table = input_list(kept) if kept else Table()
    plan = MosaicPlan(
        base_dir=base,
        reference_dir=out.parent.parent if out.parent.name.startswith('group_') else out.parent,
        table=table,
    )
    plan.boxes.append(
        MosaicBox(
            group_id=group_id,
            box_id=box_id,
            outdir=out,
            bbox=bbox,
            frames=kept,
            wcs=stamp_wcs,
        )
    )
    return plan


def build_centered_stamp_wcs(
    ra: float,
    dec: float,
    *,
    size_x_arcsec: float,
    size_y_arcsec: float,
    pixel_scale_arcsec: float = 0.031,
    orientat_deg: float = 0.0,
    ref_wcs: wcs.WCS | None = None,
) -> wcs.WCS:
    """
    Build a TAN stamp WCS centered on ``(ra, dec)`` with a fixed on-sky size.

    When *ref_wcs* is given, copy its orientation and (optionally) pixel scale
    so the custom stamp matches an existing JWST coadd grid style.
    """
    if ref_wcs is not None:
        celestial = ref_wcs.celestial
        pixel_scale_arcsec = float(
            np.asarray(wcs.utils.proj_plane_pixel_scales(celestial))[0] * 3600.0
        )
        cd = np.asarray(celestial.pixel_scale_matrix, dtype=float)
        # STWCS / DrizzlePac orientat
        orientat_deg = float(np.degrees(np.arctan2(cd[0, 1], cd[1, 1])))

    nx = max(1, int(np.ceil(float(size_x_arcsec) / float(pixel_scale_arcsec))))
    ny = max(1, int(np.ceil(float(size_y_arcsec) / float(pixel_scale_arcsec))))
    # North-up CD with PA = orientat (STWCS buildRotMatrix convention).
    pa = np.deg2rad(float(orientat_deg))
    pscale = float(pixel_scale_arcsec) / 3600.0
    rot = np.array(
        [[-np.cos(pa), np.sin(pa)], [np.sin(pa), np.cos(pa)]],
        dtype=float,
    )
    stamp = wcs.WCS(naxis=2)
    stamp.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    stamp.wcs.crval = [float(ra), float(dec)]
    stamp.wcs.crpix = [(nx + 1) * 0.5, (ny + 1) * 0.5]
    stamp.wcs.cd = rot * pscale
    stamp.pixel_shape = (nx, ny)
    stamp._naxis = [nx, ny]
    return stamp


def plan_centered_box(
    base_dir: str | Path,
    inputfiles: Sequence[str | Path],
    ra: float,
    dec: float,
    *,
    size_arcsec: float | tuple[float, float] | None = None,
    box_id: int | str = 'sn',
    group_id: int = 0,
    ref_wcs: wcs.WCS | None = None,
    pixel_scale_arcsec: float = 0.031,
    min_overlap: float = 0.01,
    verbose: bool = False,
) -> MosaicPlan:
    """
    Build a one-box plan for a stamp centered on specific sky coordinates.

    Useful when a target sits near the edge of an auto-split ``ref_*`` stamp.
    The stamp size defaults to ~the NIRCam SW stamp (~68″×51″) or, when
    *ref_wcs* is supplied, that WCS's on-sky width/height.
    """
    base = Path(base_dir).expanduser().resolve()
    if size_arcsec is None:
        if ref_wcs is not None and getattr(ref_wcs, 'pixel_shape', None):
            scales = np.asarray(
                wcs.utils.proj_plane_pixel_scales(ref_wcs.celestial),
                dtype=float,
            ) * 3600.0
            nx, ny = ref_wcs.pixel_shape
            size_x = float(nx) * float(scales[0])
            size_y = float(ny) * float(scales[1])
        else:
            size_x, size_y = 68.0, 51.0
    elif isinstance(size_arcsec, (tuple, list)):
        size_x, size_y = float(size_arcsec[0]), float(size_arcsec[1])
    else:
        size_x = size_y = float(size_arcsec)

    stamp = build_centered_stamp_wcs(
        float(ra),
        float(dec),
        size_x_arcsec=size_x,
        size_y_arcsec=size_y,
        pixel_scale_arcsec=float(pixel_scale_arcsec),
        ref_wcs=ref_wcs,
    )
    out = base / 'reference' / f'group_{int(group_id)}' / mosaic_box_dirname(box_id)
    out.mkdir(parents=True, exist_ok=True)
    write_stamp_wcs(out, stamp)

    sky_poly = stamp_sky_polygon(stamp)
    kept = filter_frames_overlapping_box(
        [str(p) for p in inputfiles],
        sky_poly,
        mosaic_wcs=None,
        min_overlap=float(min_overlap),
    )
    if verbose:
        logger.info(
            'Centered-box plan %s: RA=%.6f Dec=%.6f size=%.1f"×%.1f" '
            'shape=%sx%s frames=%d/%d',
            out,
            float(ra),
            float(dec),
            size_x,
            size_y,
            stamp.pixel_shape[0],
            stamp.pixel_shape[1],
            len(kept),
            len(list(inputfiles)),
        )

    table = input_list(kept) if kept else Table()
    plan = MosaicPlan(
        base_dir=base,
        reference_dir=base / 'reference',
        table=table,
    )
    plan.boxes.append(
        MosaicBox(
            group_id=int(group_id),
            box_id=box_id,
            outdir=out,
            bbox=local_bbox_for_wcs(stamp),
            frames=kept,
            wcs=stamp,
        )
    )
    return plan


def edit_spec_groups(
    table: Table,
    spec_group_file: str,
) -> Table:
    """
    Assign a new mosaic group index to files listed in a text manifest.

    Parameters
    ----------
    table : Table
        Input list with ``image`` and ``group`` columns.
    spec_group_file : str
        Text file of basenames to move into group ``max(group)+1``.

    Returns
    -------
    Table
        Updated input list.
    """
    files = np.loadtxt(spec_group_file, dtype=str)
    ngrp = np.max(table['group'])
    basenames = np.array([os.path.basename(i) for i in table['image']])
    for fl in files:
        table['group'][basenames == fl] = ngrp+1

    return table

def assign_gwcs(box_outdir: str, wcs_hdr: fits.Header) -> g_wcs:
    """
    Build a GWCS object for a mosaic box from a FITS WCS header.

    Parameters
    ----------
    box_outdir : str
        Mosaic box directory passed to :func:`create_gwcs`.
    wcs_hdr : fits.Header
        SCI extension WCS header from a coadd product.

    Returns
    -------
    gwcs.wcs.WCS
        GWCS object suitable for JWST datamodel ``meta.wcs``.
    """
    wcsobj = create_gwcs(outdir=box_outdir, sci_header=wcs_hdr, return_gwcs=True)

    return wcsobj


def apply_wcs_to_coadd(coadd_file: str, output: str | None = None) -> str:
    """
    Attach a :func:`create_gwcs` GWCS object to a coadd datamodel.

    By default the coadd is updated in place so mosaic products already
    carry a pipeline-compatible GWCS (no separate ``apply-gwcs`` step).

    Parameters
    ----------
    coadd_file : str
        Path to a coadd ``*_i2d.fits`` product
    output : str or None
        Optional alternate save path; default overwrites ``coadd_file``

    Returns
    -------
    str
        Path to the saved coadd with GWCS attached
    """
    from jwst import datamodels

    out_path = output or coadd_file
    with fits.open(coadd_file) as hdul:
        wcs_hdr = hdul['SCI'].header
    im = datamodels.open(coadd_file)
    wcsobj = assign_gwcs(box_outdir=os.path.dirname(coadd_file), wcs_hdr=wcs_hdr)
    im.meta.wcs = wcsobj
    im.save(out_path)
    return out_path


# Target NIRCam relative/absolute frame consistency (arcsec).
JWST_ALIGN_MAX_ARCSEC = 0.020


def _jwst_asn_members(outdir: Path, filt: str) -> list[Path]:
    """Return JHAT paths listed in ``{outdir}/{filt}.json`` when present."""
    import json

    asn = Path(outdir) / f'{str(filt).lower()}.json'
    if not asn.is_file():
        return []
    try:
        data = json.loads(asn.read_text())
    except Exception:
        return []
    out: list[Path] = []
    for prod in data.get('products') or []:
        for mem in prod.get('members') or []:
            name = mem.get('expname')
            if not name:
                continue
            cand = Path(outdir) / Path(name).name
            if cand.is_file():
                out.append(cand)
    return out


def _pick_jwst_coadd_abs_ref(coadds: Sequence[Path]) -> Path | None:
    """Prefer F200W / F150W2 / F150W i2d products as the absolute frame."""

    def score(p: Path) -> tuple[int, int]:
        name = p.name.lower()
        pref = 0
        if 'f200w' in name:
            pref = 500
        elif 'f150w2' in name:
            pref = 450
        elif 'f150w' in name:
            pref = 400
        elif 'f444w' in name:
            pref = 200
        try:
            size = int(p.stat().st_size)
        except OSError:
            size = 0
        return (pref, size)

    existing = [Path(p) for p in coadds if Path(p).is_file()]
    if not existing:
        return None
    return max(existing, key=score)


def harmonize_jwst_frames_to_ref(
    frames: Sequence[str | Path],
    abs_ref: str | Path,
    *,
    max_residual_arcsec: float = JWST_ALIGN_MAX_ARCSEC,
    max_search_arcsec: float = 2.0,
    bin_arcsec: float = 0.02,
) -> dict[str, Any]:
    """
    Shift JHAT CRVALs onto a common absolute reference (coadd or frame).

    Uses the same 2-D histogram sky-offset estimator as HST L3 unify. Designed
    to remove cross-program / cross-filter NIRCam systematics before mosaicing.
    """
    from st123.alignment.hst_jhat import measure_hst_sky_offset_2dhist
    from st123.mosaic.hst_drizzle import apply_sky_shift_to_fits

    ref = Path(abs_ref)
    report: dict[str, Any] = {
        'ok': True,
        'abs_ref': str(ref),
        'max_residual_arcsec': float(max_residual_arcsec),
        'n_shifted': 0,
        'n_ok': 0,
        'n_fail_measure': 0,
        'frames': [],
    }
    if not ref.is_file():
        report['ok'] = False
        report['error'] = f'abs_ref missing: {ref}'
        return report

    max_abs = 0.0
    for path in frames:
        fp = Path(path)
        row: dict[str, Any] = {'frame': str(fp), 'applied': False}
        if not fp.is_file():
            row['error'] = 'missing'
            report['frames'].append(row)
            report['ok'] = False
            continue
        if fp.resolve() == ref.resolve():
            row['skipped'] = 'abs_ref'
            report['frames'].append(row)
            report['n_ok'] += 1
            continue
        off = measure_hst_sky_offset_2dhist(
            fp,
            ref,
            max_offset_arcsec=float(max_search_arcsec),
            bin_arcsec=float(bin_arcsec),
            nbright=500,
            min_peak=3,
            exclude_zero_arcsec=min(0.35, max(0.08, 2.0 * float(bin_arcsec))),
        )
        row['measure'] = {
            'ok': bool(off.get('ok')),
            'abs_arcsec': off.get('abs_arcsec'),
            'dra_arcsec': off.get('dra_arcsec'),
            'ddec_arcsec': off.get('ddec_arcsec'),
            'n_pairs': off.get('n_pairs'),
            'peak_count': off.get('peak_count'),
        }
        if not off.get('ok'):
            report['n_fail_measure'] += 1
            report['frames'].append(row)
            continue
        abs_as = float(off['abs_arcsec'])
        max_abs = max(max_abs, abs_as)
        if abs_as <= float(max_residual_arcsec):
            report['n_ok'] += 1
            report['frames'].append(row)
            continue
        # img−ref → apply −Δ so the frame moves onto abs_ref.
        dra_deg = -float(off['dra_deg'])
        ddec_deg = -float(off['ddec_deg'])
        apply_sky_shift_to_fits(
            fp,
            dra_deg,
            ddec_deg,
            comment='st123: JWST box harmonize to abs_ref',
        )
        row['applied'] = True
        row['dra_arcsec'] = -float(off['dra_arcsec'])
        row['ddec_arcsec'] = -float(off['ddec_arcsec'])
        report['n_shifted'] += 1
        report['n_ok'] += 1
        report['frames'].append(row)
        logger.info(
            'JWST harmonize %s → %s: dRA=%+.1f mas dDec=%+.1f mas',
            fp.name,
            ref.name,
            -float(off['dra_arcsec']) * 1000.0,
            -float(off['ddec_arcsec']) * 1000.0,
        )

    report['max_abs_arcsec'] = max_abs
    if report['n_fail_measure'] and not report['n_shifted'] and report['n_ok'] == 0:
        report['ok'] = False
    return report


def unify_jwst_astrometric_frame(
    outdir: str | Path,
    *,
    max_residual_arcsec: float = JWST_ALIGN_MAX_ARCSEC,
    max_search_arcsec: float = 2.0,
    remosaic: bool = False,
    box_wcs=None,
) -> dict[str, Any]:
    """
    Put in-box JWST ``*_i2d.fits`` coadds (and their ASN JHAT inputs) on one frame.

    Prefers F200W as the absolute reference. Applies CRVAL shifts to each
    non-reference coadd and its Level-2 JHAT members. Optional remosaic rebuilds
    each shifted filter onto *box_wcs* (same sky stamp).
    """
    import json
    import re

    from st123.alignment.hst_jhat import measure_hst_sky_offset_2dhist
    from st123.mosaic.hst_drizzle import apply_sky_shift_to_fits

    out = Path(outdir)
    report: dict[str, Any] = {
        'ok': False,
        'outdir': str(out),
        'max_residual_arcsec': float(max_residual_arcsec),
        'groups': [],
        'abs_ref': None,
    }
    coadds = sorted(out.glob('coadd_*_i2d.fits'))
    # Skip visit-like / scratch names if any.
    coadds = [p for p in coadds if p.is_file()]
    abs_ref = _pick_jwst_coadd_abs_ref(coadds)
    if abs_ref is None:
        report['error'] = 'no JWST i2d coadds found'
        return report
    report['abs_ref'] = str(abs_ref)

    filt_re = re.compile(
        r'coadd_\d+_[^_]+_(?P<filt>[a-z0-9]+)_i2d\.fits$', re.IGNORECASE
    )

    max_abs = 0.0
    any_fail = False
    for coadd in coadds:
        m = filt_re.search(coadd.name)
        filt = m.group('filt').lower() if m else None
        row: dict[str, Any] = {
            'coadd': str(coadd),
            'filter': filt,
            'is_abs_ref': coadd.resolve() == abs_ref.resolve(),
            'applied': False,
        }
        if row['is_abs_ref']:
            report['groups'].append(row)
            continue
        off = measure_hst_sky_offset_2dhist(
            coadd,
            abs_ref,
            max_offset_arcsec=float(max_search_arcsec),
            bin_arcsec=0.05,
            nbright=800,
            min_peak=3,
            exclude_zero_arcsec=0.15,
        )
        row['measure'] = {
            'ok': bool(off.get('ok')),
            'abs_arcsec': off.get('abs_arcsec'),
            'dra_arcsec': off.get('dra_arcsec'),
            'ddec_arcsec': off.get('ddec_arcsec'),
            'n_pairs': off.get('n_pairs'),
            'peak_count': off.get('peak_count'),
        }
        if not off.get('ok'):
            any_fail = True
            report['groups'].append(row)
            logger.warning(
                'JWST unify: could not measure %s vs %s (pairs=%s)',
                coadd.name,
                abs_ref.name,
                off.get('n_pairs'),
            )
            continue
        abs_as = float(off['abs_arcsec'])
        max_abs = max(max_abs, abs_as)
        if abs_as <= float(max_residual_arcsec):
            report['groups'].append(row)
            continue
        dra_deg = -float(off['dra_deg'])
        ddec_deg = -float(off['ddec_deg'])
        frames = _jwst_asn_members(out, filt) if filt else []
        n_l2 = 0
        for fp in frames:
            n_l2 += apply_sky_shift_to_fits(
                fp, dra_deg, ddec_deg, comment='st123: JWST L2 unify'
            )
        # Keep shared stamp WCS locked: never walk coadd CRVAL when box_wcs
        # is set; remosaic onto the stamp instead.
        do_remosaic = bool(remosaic) or box_wcs is not None
        if box_wcs is None:
            apply_sky_shift_to_fits(
                coadd, dra_deg, ddec_deg, comment='st123: JWST L3 unify'
            )
        row['applied'] = True
        row['n_l2'] = n_l2
        row['dra_arcsec'] = -float(off['dra_arcsec'])
        row['ddec_arcsec'] = -float(off['ddec_arcsec'])
        logger.info(
            'JWST unify %s → %s: dRA=%+.1f mas dDec=%+.1f mas (%d L2)%s',
            coadd.name,
            abs_ref.name,
            row['dra_arcsec'] * 1000.0,
            row['ddec_arcsec'] * 1000.0,
            n_l2,
            '; stamp WCS locked' if box_wcs is not None else '',
        )

        if do_remosaic and filt and frames and box_wcs is not None:
            try:
                if filt in {
                    'f560w', 'f770w', 'f1000w', 'f1130w', 'f1280w',
                    'f1500w', 'f1800w', 'f2100w', 'f2550w',
                }:
                    inst = 'miri'
                else:
                    inst = 'nircam'
                pixscale = mosaic_pixel_scale_arcsec(filt, inst)
                filt_hdr = rescale_wcs_to_pixel_scale(box_wcs, pixscale)
                table = input_list([str(p) for p in frames])
                gwcs_path = create_gwcs(
                    outdir=str(out),
                    sci_header=filt_hdr,
                    filename=f'mosaic_gwcs_{filt}_unify.asdf',
                )
                single = {filt: table}
                copy_files(single, str(out))
                single = update_path(single, str(out))
                driz = create_coadd_mosaic(
                    single[filt],
                    outdir=str(out),
                    filt=filt,
                    gwcs_file=gwcs_path,
                )
                # Rename pipeline product onto the canonical coadd path.
                target = coadd
                if Path(driz).resolve() != target.resolve():
                    shutil.copy2(driz, target)
                apply_wcs_to_coadd(str(target))
                row['remosaicked'] = str(target)
            except Exception as exc:
                row['remosaic_error'] = str(exc)
                any_fail = True
                logger.exception('JWST unify remosaic failed for %s: %s', filt, exc)

        report['groups'].append(row)

    report['max_abs_arcsec'] = max_abs
    # Re-measure after shifts for the pass/fail gate.
    final_max = 0.0
    for coadd in coadds:
        if coadd.resolve() == abs_ref.resolve():
            continue
        off = measure_hst_sky_offset_2dhist(
            coadd,
            abs_ref,
            max_offset_arcsec=float(max_search_arcsec),
            bin_arcsec=0.05,
            nbright=800,
            min_peak=3,
            exclude_zero_arcsec=0.15,
        )
        if off.get('ok'):
            final_max = max(final_max, float(off['abs_arcsec']))
    report['final_max_abs_arcsec'] = final_max
    report['ok'] = final_max <= float(max_residual_arcsec)
    qa_path = out / 'jwst_astrometric_frame_qa.json'
    qa_path.write_text(json.dumps(report, indent=2, default=str))
    report['qa_path'] = str(qa_path)
    logger.info(
        'JWST unify %s: final max |Δ|=%.1f mas (limit %.1f mas) → %s',
        out.name,
        final_max * 1000.0,
        float(max_residual_arcsec) * 1000.0,
        qa_path,
    )
    return report

