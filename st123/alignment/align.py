"""
Unified JWST alignment library.

Everything needed to align JWST imaging lives here, ordered from low-level
knobs to high-level orchestration:

1. Calibrator settings — per-filter JHAT / refine tuning and quality-hold cuts
2. Fallback helpers — MIRI→MIRI parent ranking, provenance headers
3. JHAT core — photometry, dispersion, ``align_jwst_image``, visit mosaics
4. Relative API — ``run_alignment`` plus reference-catalog construction
5. Overlap discovery — footprint matching and alignment summary tables
6. Parallel workers — spawn-safe REFERENCE / MIRI_REL jobs
7. Orchestration — ``align_from_frames`` / ``run_overlaps``

The CLI wrapper lives in :mod:`st123.scripts.align` (``visit`` / ``reference``
/ ``pair`` modes).
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import traceback
import warnings
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

warnings.filterwarnings('ignore')
# stpipe/pysiaf log handlers raise on multi-argument warnings during spawn
# worker imports; never let a logging failure abort alignment.
logging.raiseExceptions = False

import matplotlib

matplotlib.use('Agg')

import astropy.wcs as wcs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from astropy import units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.stats import sigma_clipped_stats  # noqa: E402
from astropy.table import Column, Table, vstack  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
from photutils.detection import DAOStarFinder  # noqa: E402


@contextmanager
def suppress_output():
    """
    Silence stdout, stderr, and logging for the duration of the block.

    JHAT and the JWST pipeline are extremely chatty. Workers wrap every
    alignment call in this context so the parent process only emits its own
    ``START`` / ``DONE`` lines.
    """
    devnull = open(os.devnull, 'w')
    try:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            previous_disable = logging.root.manager.disable
            logging.disable(logging.CRITICAL)
            try:
                yield
            finally:
                logging.disable(previous_disable)
    finally:
        devnull.close()


# Import the science stack quietly: these packages print banners and emit
# import-time log records that break stpipe handlers inside spawn workers.
with suppress_output():
    from astroquery.gaia import Gaia  # noqa: E402
    from jhat import jwst_photclass, st_wcs_align  # noqa: E402
    from jwst.associations import asn_from_list  # noqa: E402
    from jwst.associations.lib.rules_level3_base import (  # noqa: E402
        DMS_Level3_Base,
    )
    from jwst.datamodels import ImageModel  # noqa: E402
    from jwst.pipeline import calwebb_image3  # noqa: E402

from st123.mosaic.region import SRegionPolygon  # noqa: E402
from st123.utils.helpers import input_list, xmatch_common  # noqa: E402
from st123.utils.jwst_compat import patch_jwst_for_photutils3  # noqa: E402
from st123.utils.settings import *  # noqa: E402,F403


# ---------------------------------------------------------------------------
# 1. Calibrator settings
# ---------------------------------------------------------------------------
#
# F770W fields often yield hundreds of detections (PAH / arm structure). Across
# frames, higher ``n_calibrators`` correlates with worse dispersion because busy
# fields are harder — but *within* a frame the brightest JHAT matches are
# typically the most coherent calibrators. Severely trimming ``Nbright`` and
# hard-clipping refine residuals therefore improves F770W solutions more than
# magnitude / morphology cuts that reject bright sources.


@dataclass(frozen=True)
class CalibratorSettings:
    """Per-filter knobs for JHAT plus the iterative refine loop."""

    nbright: int = 800
    refine_sigma: float = 2.0
    refine_max_iter: int = 5
    refine_dist_limit_arcsec: float = 0.5
    # Hard residual ceiling during refine (None disables).
    max_residual_arcsec: float | None = None
    # Optional MIRI photometry morphology / magnitude cuts applied when those
    # columns exist on the aligned photometry catalog (refine stage).
    miri_mag_min: float | None = None
    miri_mag_max: float | None = None
    miri_round_max: float | None = None
    miri_sharp_min: float | None = None
    miri_sharp_max: float | None = None
    min_calibrators: int = 20
    # JHAT native source cuts (applied before Nbright). None = keep the
    # ``strict_jwst_params`` defaults.
    sharpness_lim: tuple[float, float] | None = None
    roundness1_lim: tuple[float, float] | None = None
    objmag_lim: tuple[float, float] | None = None
    dmag_max: float | None = None

    def jhat_param_overrides(self) -> dict[str, Any]:
        """
        Build JHAT ``run_all`` overrides for the non-default source cuts.

        Returns
        -------
        dict
            Keyword arguments merged into the JHAT parameter dictionary.
        """
        out: dict[str, Any] = {}
        if self.sharpness_lim is not None:
            out['sharpness_lim'] = self.sharpness_lim
        if self.roundness1_lim is not None:
            out['roundness1_lim'] = self.roundness1_lim
        if self.objmag_lim is not None:
            out['objmag_lim'] = self.objmag_lim
        if self.dmag_max is not None:
            out['dmag_max'] = self.dmag_max
        return out

    def as_run_kwargs(self) -> dict[str, Any]:
        """
        Build the keyword arguments consumed by :func:`run_alignment`.

        Returns
        -------
        dict
            Calibrator knobs plus nested ``jhat_params`` overrides.
        """
        return {
            'nbright': self.nbright,
            'refine_sigma': self.refine_sigma,
            'refine_max_iter': self.refine_max_iter,
            'refine_dist_limit_arcsec': self.refine_dist_limit_arcsec,
            'max_residual_arcsec': self.max_residual_arcsec,
            'miri_mag_min': self.miri_mag_min,
            'miri_mag_max': self.miri_mag_max,
            'miri_round_max': self.miri_round_max,
            'miri_sharp_min': self.miri_sharp_min,
            'miri_sharp_max': self.miri_sharp_max,
            'min_calibrators': self.min_calibrators,
            'jhat_params': self.jhat_param_overrides(),
        }


# Default (non-F770W) pipeline settings — use CLI / strict_jwst_params defaults.
DEFAULT_CALIBRATOR_SETTINGS = CalibratorSettings()

# F770W: keep JHAT's default bright-source preference, but severely trim how
# many enter the fit and hard-clip refine residuals. Avoid objmag cuts that
# reject the brightest MIRI detections — those are the best within-frame
# calibrators on PAH-heavy fields.
F770W_CALIBRATOR_SETTINGS = CalibratorSettings(
    nbright=100,
    refine_sigma=1.5,
    refine_max_iter=5,
    refine_dist_limit_arcsec=0.50,
    max_residual_arcsec=0.08,
    min_calibrators=15,
)

# Per-filter REFERENCE→MIRI_REL quality thresholds (mas).
#
# Empirically, MIRI_REL absolute dispersion (quadrature of parent absolute +
# relative) is typically *worse* than REFERENCE below these cuts and better
# above them, when a good overlapping parent is available. F560W must stay on
# REFERENCE (``None`` disables the quality-hold fallback for that filter).
FILTER_MAX_REFERENCE_DISPERSION_MAS: dict[str, float | None] = {
    'F560W': None,
    'F770W': 50.0,
    'F1000W': 35.0,
    'F1130W': 55.0,
    'F1280W': 50.0,
    'F1500W': 50.0,
    'F1800W': 50.0,
    'F2100W': 65.0,
}

# Default when a filter is absent from the map.
DEFAULT_MAX_REFERENCE_DISPERSION_MAS = 70.0


def calibrator_settings_for_filter(filter_name: str | None) -> CalibratorSettings:
    """
    Return the calibrator settings for a MIRI filter.

    Parameters
    ----------
    filter_name : str or None
        Filter name such as ``'F770W'``; suffixes after ``_`` are ignored.

    Returns
    -------
    CalibratorSettings
        Filter-specific settings, or the shared defaults.
    """
    key = str(filter_name or '').upper().split('_', 1)[0]
    if key == 'F770W':
        return F770W_CALIBRATOR_SETTINGS
    return DEFAULT_CALIBRATOR_SETTINGS


def max_reference_dispersion_mas(filter_name: str | None) -> float | None:
    """
    Return the REFERENCE quality-hold threshold for a filter.

    Parameters
    ----------
    filter_name : str or None
        Filter name such as ``'F1000W'``.

    Returns
    -------
    float or None
        Threshold in mas, or ``None`` to never quality-hold (F560W).
    """
    key = str(filter_name or '').upper().split('_', 1)[0]
    if key in FILTER_MAX_REFERENCE_DISPERSION_MAS:
        return FILTER_MAX_REFERENCE_DISPERSION_MAS[key]
    return DEFAULT_MAX_REFERENCE_DISPERSION_MAS


def describe_calibrator_settings(settings: CalibratorSettings) -> str:
    """
    Summarize calibrator settings on one line for logging.

    Parameters
    ----------
    settings : CalibratorSettings
        Settings to describe.

    Returns
    -------
    str
        Comma-separated ``key=value`` summary.
    """
    parts = [
        f'nbright={settings.nbright}',
        f'refine_sigma={settings.refine_sigma}',
        f'dist_limit={settings.refine_dist_limit_arcsec}"',
    ]
    if settings.objmag_lim is not None:
        parts.append(f'objmag={settings.objmag_lim}')
    if settings.sharpness_lim is not None:
        parts.append(f'sharp={settings.sharpness_lim}')
    if settings.roundness1_lim is not None:
        parts.append(f'round={settings.roundness1_lim}')
    if settings.max_residual_arcsec is not None:
        parts.append(f'max_resid={settings.max_residual_arcsec}"')
    return ', '.join(parts)


# ---------------------------------------------------------------------------
# 2. Fallback (MIRI→MIRI relative) helpers
# ---------------------------------------------------------------------------
#
# When direct reference alignment fails, align the failed frame to a
# successfully aligned MIRI image that is closest in wavelength and has the
# largest footprint overlap. Absolute dispersion is the quadrature sum of the
# parent absolute dispersion and the new relative dispersion.


# Approximate MIRI filter central wavelengths (microns), blue → red.
MIRI_FILTER_WAVELENGTH_UM: dict[str, float] = {
    'F560W': 5.6,
    'F770W': 7.7,
    'F1000W': 10.0,
    'F1130W': 11.3,
    'F1280W': 12.8,
    'F1500W': 15.0,
    'F1800W': 18.0,
    'F2100W': 21.0,
    'F2550W': 25.5,
}


@dataclass
class SuccessfulAlignment:
    """A MIRI frame that has been successfully placed on an absolute frame."""

    miri_path: str
    jhat_path: str
    filter: str
    wavelength_um: float
    dispersion_mas: float
    relative_dispersion_mas: float
    align_mode: str  # 'REFERENCE' or 'MIRI_REL'
    original_ref: str
    aligned_to: str
    photfile: str | None = None


def filter_wavelength_um(filter_name: str) -> float:
    """
    Return the central wavelength of a MIRI filter.

    Parameters
    ----------
    filter_name : str
        Filter name such as ``'F1130W'``.

    Returns
    -------
    float
        Central wavelength in microns, or ``inf`` when unparseable.
    """
    key = str(filter_name).upper()
    if key in MIRI_FILTER_WAVELENGTH_UM:
        return MIRI_FILTER_WAVELENGTH_UM[key]
    # Parse FnnnnW / FnnnW names when not in the table.
    token = key.split('_', 1)[0]
    if token.startswith('F') and token.endswith('W'):
        digits = ''.join(ch for ch in token[1:-1] if ch.isdigit())
        if digits:
            # F560W → 5.60, F1000W → 10.00, F1130W → 11.30
            val = float(digits)
            return val / 100.0 if val >= 100 else val / 10.0
    return float('inf')


def sort_frames_blue_to_red(frames: list, *, filter_from_path) -> list:
    """
    Sort overlap frames by increasing filter wavelength, then path.

    Parameters
    ----------
    frames : list
        Overlap frames (``FrameOverlaps`` instances or dicts).
    filter_from_path : callable
        Maps a MIRI path to a filter name.

    Returns
    -------
    list
        Frames ordered blue → red.
    """

    def key(frame) -> tuple:
        path = frame['miri_path'] if isinstance(frame, dict) else frame.miri_path
        filt = filter_from_path(path) or 'UNKNOWN'
        return (filter_wavelength_um(filt), str(path))

    return sorted(frames, key=key)


def load_s_region(fits_path: str) -> SRegionPolygon:
    """
    Parse ``S_REGION`` from the first HDU that defines it.

    Parameters
    ----------
    fits_path : str
        FITS file to read.

    Returns
    -------
    SRegionPolygon
        Sky polygon for the image footprint.

    Raises
    ------
    KeyError
        If no HDU carries an ``S_REGION`` keyword.
    """
    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            if 'S_REGION' in hdu.header:
                return SRegionPolygon.parse(hdu.header['S_REGION'])
    raise KeyError(f'No S_REGION in {fits_path}')


def sky_overlap_fraction(miri_a: str, miri_b: str) -> float:
    """
    Return the fraction of ``miri_a``'s footprint overlapped by ``miri_b``.

    Parameters
    ----------
    miri_a, miri_b : str
        FITS files carrying ``S_REGION`` headers.

    Returns
    -------
    float
        Overlap fraction in a local tangent plane, in ``[0, 1]``.
    """
    a = load_s_region(miri_a)
    b = load_s_region(miri_b)
    verts = np.asarray(a.vertices, dtype=float)
    cen_ra = float(np.mean(verts[:, 0]))
    cen_dec = float(np.mean(verts[:, 1]))
    pa = a.to_tangent_polygon(cen_ra, cen_dec)
    pb = b.to_tangent_polygon(cen_ra, cen_dec)
    if pa.is_empty or pa.area <= 0:
        return 0.0
    inter = pa.intersection(pb)
    if inter.is_empty:
        return 0.0
    return float(inter.area) / float(pa.area)


def combine_dispersion_mas(parent_mas: float, relative_mas: float) -> float:
    """
    Combine a parent absolute dispersion with a relative dispersion.

    Parameters
    ----------
    parent_mas : float
        Absolute dispersion of the parent frame, in mas.
    relative_mas : float
        Relative dispersion of the new alignment step, in mas.

    Returns
    -------
    float
        Quadrature sum, in mas.
    """
    return float(math.sqrt(parent_mas**2 + relative_mas**2))


def rank_fallback_parents(
    miri_path: str,
    filter_name: str,
    successes: list[SuccessfulAlignment],
    *,
    min_overlap_fraction: float = 0.05,
    assume_relative_mas: float = 25.0,
    wavelength_penalty_mas_per_um: float = 3.5,
    max_parents: int = 5,
) -> list[tuple[SuccessfulAlignment, float]]:
    """
    Rank already-aligned MIRI parents for relative fallback.

    Among parents with sky overlap at or above ``min_overlap_fraction``, sort by
    lowest ``sqrt(parent_abs**2 + assume_relative_mas**2) +
    wavelength_penalty_mas_per_um * |dlambda|``, then closest wavelength, then
    largest overlap. This prefers high-quality parents without favouring
    arbitrarily blue ones that often fail matching across large wavelength gaps.

    Parameters
    ----------
    miri_path : str
        Frame that needs a parent.
    filter_name : str
        Filter of ``miri_path``.
    successes : list of SuccessfulAlignment
        Candidate parents.
    min_overlap_fraction : float, optional
        Minimum sky overlap fraction to consider a parent.
    assume_relative_mas : float, optional
        Assumed relative dispersion when scoring candidates.
    wavelength_penalty_mas_per_um : float, optional
        Score penalty per micron of wavelength separation.
    max_parents : int, optional
        Maximum number of ranked parents to return.

    Returns
    -------
    list of (SuccessfulAlignment, float)
        Parents paired with their sky overlap fraction, best first.
    """
    if not successes:
        return []

    target_wl = filter_wavelength_um(filter_name)
    ranked: list[tuple[float, float, float, SuccessfulAlignment]] = []
    for parent in successes:
        if parent.miri_path == miri_path:
            continue
        try:
            frac = sky_overlap_fraction(miri_path, parent.miri_path)
        except Exception:
            continue
        if frac < min_overlap_fraction:
            continue
        est_abs = combine_dispersion_mas(
            float(parent.dispersion_mas), float(assume_relative_mas)
        )
        dlam = abs(parent.wavelength_um - target_wl)
        score = est_abs + float(wavelength_penalty_mas_per_um) * dlam
        ranked.append((score, dlam, -frac, parent))

    if not ranked:
        return []
    ranked.sort()
    out: list[tuple[SuccessfulAlignment, float]] = []
    for _score, _dlam, neg_frac, parent in ranked[: max(1, int(max_parents))]:
        out.append((parent, -neg_frac))
    return out


def select_fallback_parent(
    miri_path: str,
    filter_name: str,
    successes: list[SuccessfulAlignment],
    *,
    min_overlap_fraction: float = 0.05,
    assume_relative_mas: float = 25.0,
    wavelength_penalty_mas_per_um: float = 3.5,
) -> tuple[SuccessfulAlignment | None, float]:
    """
    Choose the single best fallback parent.

    Parameters
    ----------
    miri_path : str
        Frame that needs a parent.
    filter_name : str
        Filter of ``miri_path``.
    successes : list of SuccessfulAlignment
        Candidate parents.
    min_overlap_fraction, assume_relative_mas, wavelength_penalty_mas_per_um
        See :func:`rank_fallback_parents`.

    Returns
    -------
    tuple
        ``(parent, overlap_fraction)``, or ``(None, 0.0)`` when none qualify.
    """
    ranked = rank_fallback_parents(
        miri_path,
        filter_name,
        successes,
        min_overlap_fraction=min_overlap_fraction,
        assume_relative_mas=assume_relative_mas,
        wavelength_penalty_mas_per_um=wavelength_penalty_mas_per_um,
        max_parents=1,
    )
    if not ranked:
        return None, 0.0
    return ranked[0]


def find_aligned_photfile(jhat_path: str) -> str | None:
    """
    Locate post-alignment photometry next to a JHAT product.

    Parameters
    ----------
    jhat_path : str
        Path to a ``*_jhat.fits`` product.

    Returns
    -------
    str or None
        Resolved catalog path, or ``None`` when no candidate exists.
    """
    jhat = Path(jhat_path)
    stem = jhat.name.replace('_jhat.fits', '')
    for name in (
        f'{stem}_jhat_cal.phot.txt',
        f'{stem}_jhat_i2d.phot.txt',
        f'{stem}.phot.txt',
    ):
        cand = jhat.parent / name
        if cand.is_file():
            return str(cand.resolve())
    return None


def write_alignment_provenance(
    jhat_path: str,
    *,
    align_mode: str,
    original_ref: str,
    aligned_to: str,
    relative_dispersion_mas: float,
    absolute_dispersion_mas: float,
    n_calibrators: int | None = None,
) -> None:
    """
    Record alignment provenance and dispersions on the JHAT primary header.

    Parameters
    ----------
    jhat_path : str
        JHAT product to update in place.
    align_mode : str
        ``REFERENCE`` (absolute align to a reference image) or ``MIRI_REL``
        (relative align to another MIRI frame). Legacy ``NIRCAM`` is mapped to
        ``REFERENCE``.
    original_ref : str
        Original (root) reference image defining the absolute frame.
    aligned_to : str
        Image or catalog this frame was aligned to in this step.
    relative_dispersion_mas : float
        Relative dispersion of this step, in mas.
    absolute_dispersion_mas : float
        Absolute dispersion, in mas; for ``MIRI_REL`` the quadrature
        combination of the parent absolute and relative terms.
    n_calibrators : int, optional
        Astrometric calibrator count to store in ``JWNCAL``.
    """
    with fits.open(jhat_path, mode='update') as hdul:
        hdr = hdul[0].header
        mode = str(align_mode).upper()
        if mode == 'NIRCAM':
            mode = 'REFERENCE'
        hdr['ALGNMODE'] = (mode, 'REFERENCE or MIRI_REL')
        # Astropy stores long paths via CONTINUE cards.
        hdr['ALGNREF'] = (str(original_ref), 'Original abs reference')
        hdr['ALGNTO'] = (str(aligned_to), 'Aligned-to image/catalog')
        hdr['JWDISPR'] = (
            float(relative_dispersion_mas) / 1000.0,
            '[arcsec] relative dispersion',
        )
        hdr['JWDISPM'] = (
            float(absolute_dispersion_mas) / 1000.0,
            '[arcsec] absolute dispersion',
        )
        if n_calibrators is not None:
            hdr['JWNCAL'] = (int(n_calibrators), 'Astrometric calibrators')


# ---------------------------------------------------------------------------
# 3. JHAT core and visit-mode helpers
# ---------------------------------------------------------------------------


def get_input_images(pattern=None, workdir=None):
    """
    Collect images to process from the ``raw`` subdirectory of a work directory.

    Parameters
    ----------
    pattern : list, optional
        Glob patterns to search for (default: NIRCam A/B ``*_cal.fits``).
    workdir : str, optional
        Work directory (default: current directory).

    Returns
    -------
    list
        Matching image paths.
    """
    if workdir is None:
        workdir = '.'
    if pattern is None:
        pattern = ['*nrca*_cal.fits', '*nrcb*_cal.fits']
    return [s for p in pattern for s in glob.glob(os.path.join(workdir, 'raw', p))]


def pick_deepest_image(table):
    """
    Pick the longest-exposure image from a table.

    Parameters
    ----------
    table : astropy.table.Table
        Table with an ``exptime`` column.

    Returns
    -------
    astropy.table.Row
        Row with the maximum exposure time.
    """
    exptimes = [r['exptime'] for r in table]
    return table[exptimes.index(max(exptimes))]


def add_alignment_groups(table, use_shapely=False):
    """
    Add alignment groups derived from image polygon overlap area.

    Parameters
    ----------
    table : astropy.table.Table
        Input list table.
    use_shapely : bool, optional
        Group by spatially distinct shapely geometries instead of detector
        chip. This fails for certain dither patterns.

    Returns
    -------
    astropy.table.Table
        Table with ``guide_star`` and ``ref_img`` columns added.
    """
    pgons, guide_star_id = [], []
    table_indices = np.arange(len(table))
    for im in table['image']:
        region = fits.open(im)['SCI'].header['S_REGION']
        coords = np.array(region.split('POLYGON ICRS  ')[1].split(' '), dtype=float)
        pgons.append(shapely.Polygon(coords.reshape(4, 2)))
        guide_star_id.append(fits.getval(im, 'GDSTARID', ext=0))

    # For each guide star, find a reference image for each image and record it.
    guide_star_id, pgons = np.array(guide_star_id), np.array(pgons)
    table.add_column(Column(name='guide_star', data=guide_star_id))
    table.add_column(Column(name='ref_img', data=[None] * len(table)))
    if use_shapely:
        for guide_star in np.unique(guide_star_id):
            align_groups = np.array([])
            guide_star_mask = guide_star_id == guide_star
            guide_star_polygons = pgons[guide_star_mask]
            footprint = shapely.unary_union(guide_star_polygons)
            # Convert to multipolygon for a single-observation footprint.
            if isinstance(footprint, shapely.geometry.polygon.Polygon):
                footprint = shapely.MultiPolygon([footprint])

            for component in footprint.geoms:
                align_groups = np.append(align_groups, component)

            align_idx, overlap = [], []
            for chip_polygon in guide_star_polygons:
                intersect_area = np.array(
                    [
                        shapely.intersection(chip_polygon, group_geom).area
                        / chip_polygon.area
                        for group_geom in align_groups
                    ]
                )
                align_idx.append(np.argmax(intersect_area))
                overlap.append(np.max(intersect_area))

            for aln_idx in np.unique(align_idx):
                aln_mask = np.array(align_idx) == aln_idx
                ref_image = pick_deepest_image(
                    table[guide_star_mask][aln_mask]
                )['image']
                table['ref_img'][table_indices[guide_star_mask][aln_mask]] = ref_image
    else:
        for guide_star in np.unique(guide_star_id):
            guide_star_mask = guide_star_id == guide_star
            for chip in np.unique(table[guide_star_mask]['chip']):
                chip_mask = table[guide_star_mask]['chip'] == chip
                ref_image = pick_deepest_image(
                    table[guide_star_mask][chip_mask]
                )['image']
                table['ref_img'][table_indices[guide_star_mask][chip_mask]] = ref_image

    return table


def visit_filter_dict(table):
    """
    Find the broadband filter with maximum spatial coverage in each visit.

    Parameters
    ----------
    table : astropy.table.Table
        Input list table.

    Returns
    -------
    dict
        Visit identifier mapped to the filter used for its alignment mosaic.
    """
    visits = np.unique(table['visit']).value
    visit_filter = dict.fromkeys(visits)
    for vis in visits:
        tbl = table[table['visit'] == vis]
        net_polygon = []
        filters = np.unique(tbl['filter']).value
        is_broadband = []
        for filt in filters:
            if 'N' in filt.upper():
                is_broadband.append(False)
                continue
            is_broadband.append(True)
            pgons = []
            filter_rows = tbl[tbl['filter'] == filt]
            for im, pupil in zip(filter_rows['image'], filter_rows['pupil']):
                if 'N' in pupil.upper():
                    continue
                region = fits.open(im)['SCI'].header['S_REGION']
                coords = np.array(
                    region.split('POLYGON ICRS  ')[1].split(' '), dtype=float
                )
                pgons.append(shapely.Polygon(coords.reshape(4, 2)))
            net_polygon.append(shapely.unary_union(pgons))

        area = [polygon.area for polygon in net_polygon]
        visit_filter[vis] = filters[is_broadband][np.argmax(area)]

    return visit_filter


def get_visit_geoms(table):
    """
    Build the union sky footprint of each visit.

    Parameters
    ----------
    table : astropy.table.Table
        Input list table.

    Returns
    -------
    dict
        Visit identifier mapped to a shapely geometry.
    """
    visits = np.unique(table['visit']).value
    field = []
    for vis in visits:
        tbl = table[table['visit'] == vis]
        net_polygon = []
        for filt in np.unique(tbl['filter']).value:
            pgons = []
            filter_rows = tbl[tbl['filter'] == filt]
            for im in filter_rows['image']:
                region = fits.open(im)['SCI'].header['S_REGION']
                coords = np.array(
                    region.split('POLYGON ICRS  ')[1].split(' '), dtype=float
                )
                pgons.append(shapely.Polygon(coords.reshape(4, 2)))
            net_polygon.append(shapely.unary_union(pgons))
        field.append(shapely.unary_union(net_polygon))

    return dict(zip(visits, field))


def order_visits(table):
    """
    Order visits from largest to smallest sky footprint.

    Parameters
    ----------
    table : astropy.table.Table
        Input list table.

    Returns
    -------
    numpy.ndarray
        Indices that sort visits by decreasing footprint area.
    """
    geoms = get_visit_geoms(table)
    return np.argsort([geom.area for geom in geoms.values()])[::-1]


def pick_visit(align_pgon, visit_geoms, visit_filter):
    """
    Pick the next visit to align against the current alignment footprint.

    Parameters
    ----------
    align_pgon : shapely geometry or None
        Footprint already aligned; ``None`` selects the largest broadband visit.
    visit_geoms : dict
        Visit identifier mapped to a shapely geometry (mutated when
        ``align_pgon`` is ``None``).
    visit_filter : dict
        Visit identifier mapped to its alignment filter.

    Returns
    -------
    tuple
        ``(visit_id, overlap_fraction)``.
    """
    if align_pgon is None:
        narrowband_visits = np.array(list(visit_filter.keys()))[
            np.array(['N' in i.upper() for i in visit_filter.values()])
        ]
        for visit_id in narrowband_visits:
            if visit_id in visit_geoms and len(visit_geoms) > 1:
                visit_geoms.pop(visit_id)
        vis_area = [visit_geoms[i].area for i in visit_geoms]
        arg = np.argmax(vis_area)
        overlap_frac = vis_area[arg]
    else:
        intersect_area = [
            align_pgon.intersection(visit_geoms[i]).area / visit_geoms[i].area
            for i in visit_geoms
        ]
        arg = np.argmax(intersect_area)
        overlap_frac = intersect_area[arg]

    return list(visit_geoms.keys())[arg], overlap_frac


def jwst_phot(phot_img):
    """
    Run JHAT ``jwst_photclass`` photometry on an image.

    Parameters
    ----------
    phot_img : str
        Image to photometer.

    Returns
    -------
    tuple
        ``(refcat, photfilename)`` where ``refcat`` is an
        :class:`astropy.table.Table`.
    """
    patch_jwst_for_photutils3()
    photometry = jwst_photclass()
    photfilename = phot_img.replace('.fits', '.phot.txt')
    photometry.run_phot(
        imagename=phot_img,
        photfilename=photfilename,
        overwrite=True,
        ee_radius=70,
    )
    refcat = Table.read(photfilename, format='ascii')
    return refcat, photfilename


def fix_phot(mosaic):
    """
    Rewrite JHAT photometry sky coordinates for an i2d mosaic.

    Parameters
    ----------
    mosaic : str
        Mosaic file name.

    Returns
    -------
    str
        Path to the corrected photometry catalog.
    """
    refcat, photfile = jwst_phot(mosaic)
    w = wcs.WCS(fits.open(mosaic)['SCI'].header)
    sky_xy = w.all_pix2world(refcat['x'], refcat['y'], 0)
    refcat['ra'], refcat['dec'] = np.array(sky_xy[0]), np.array(sky_xy[1])
    corrected = photfile.replace('i2d', 'i2d.corr')
    refcat.write(corrected, format='ascii', overwrite=True)
    return corrected


def generate_level3_mosaic(inputfiles, outdir):
    """
    Create a Level-3 drizzled mosaic from Level-2 inputs.

    Parameters
    ----------
    inputfiles : list
        Level-2 input files.
    outdir : str
        Output directory; products land in ``<outdir>/out``.

    Returns
    -------
    str
        Path to the Level-3 mosaic.
    """
    # jwst 1.20.x + photutils>=3: SourceFinder requires n_pixels, not npixels.
    patch_jwst_for_photutils3()
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    table = input_list(inputfiles)
    filters = np.unique([r['filter'] for r in table])
    filter_name = '_'.join(filters)
    asn_file = f'{outdir}/{filter_name}.json'
    base_filenames = np.array([os.path.basename(r['image']) for r in table])
    asn3 = asn_from_list.asn_from_list(
        base_filenames, rule=DMS_Level3_Base, product_name=f'{filter_name}'
    )

    with open(asn_file, 'w') as outfile:
        _name, serialized = asn3.dump(format='json')
        outfile.write(serialized)

    image3 = calwebb_image3.Image3Pipeline()

    outdir_level3 = os.path.join(outdir, 'out')
    if not os.path.exists(outdir_level3):
        os.makedirs(outdir_level3)

    image3.output_dir = outdir_level3
    image3.save_results = True
    image3.tweakreg.skip = True
    image3.skymatch.skip = True
    image3.skymatch.match_down = False
    image3.source_catalog.skip = False
    image3.resample.pixfrac = 1.0
    image3.weight_type = 'ivm'

    image3.run(asn_file)
    return f'{outdir_level3}/{filter_name}_i2d.fits'


def create_alignment_mosaic(
    filter_table,
    outdir,
    align_filter=None,
    align_to='gaia',
    ncores=10,
    Nbright=800,
):
    """
    Build an i2d alignment mosaic and align it to Gaia or a catalog.

    Parameters
    ----------
    filter_table : dict
        Filter name mapped to an input list table.
    outdir : str
        Output directory for JHAT products.
    align_filter : str, optional
        Filter to mosaic; picked from a preferred list when omitted.
    align_to : str, optional
        ``'gaia'`` or a photometry catalog path.
    ncores : int, optional
        Pool size for per-image relative alignment.
    Nbright : int, optional
        Number of bright sources for the mosaic alignment.

    Returns
    -------
    tuple
        ``(aligned_mosaic, guess_offset)``.
    """
    if align_to is None:
        raise ValueError('Invalid alignment phot file')

    # Best filters for Gaia alignment, in order.
    good_filters = ['f277w', 'f322w2', 'f356w', 'f250m', 'f300m']

    align_table = None
    if align_filter:
        if align_filter in filter_table:
            align_table = filter_table[align_filter]
    else:
        for filt in good_filters:
            if filt in filter_table:
                align_table = filter_table[filt]
                break

    if align_table is None:
        raise ValueError('No compatible alignment filter found')

    # Alignment groups pick the correct reference image for each module.
    align_table = add_alignment_groups(align_table)
    repo = str(_resolve_repo_root())
    jobs = []
    for row in align_table:
        image = row['image']
        jhat_dest = os.path.join(
            outdir, os.path.basename(image.replace('cal.fits', 'jhat.fits'))
        )
        if os.path.exists(jhat_dest):
            continue
        ref_image = row['ref_img']
        if not os.path.exists(ref_image.replace('.fits', '.phot.txt')):
            _, photfilename = jwst_phot(ref_image)
        else:
            photfilename = ref_image.replace('.fits', '.phot.txt')
        jobs.append(
            {
                'miri_path': image,  # START/DONE label key shared with wrap runner
                'align_image': image,
                'outdir': outdir,
                'gaia': False,
                'photfilename': photfilename,
                'xshift': 0.0,
                'yshift': 0.0,
                'Nbright': 800,
                'sig': 2,
                'filter': str(row.get('filter', '')),
                'mode': 'VISIT',
                'repo': repo,
            }
        )
    _run_jobs_parallel(
        jobs,
        run_visit_align_job,
        workers=ncores,
        label='VISIT',
        on_result=lambda result: print(_format_worker_done(result), flush=True),
    )

    # Create an i2d mosaic from the relatively aligned images.
    inputfiles = [
        os.path.join(outdir, os.path.basename(i.replace('cal.fits', 'jhat.fits')))
        for i in align_table['image']
    ]
    mosaic_name = generate_level3_mosaic(inputfiles, outdir)

    guess_offset = align_jwst_image(
        align_image=mosaic_name,
        outdir=outdir,
        gaia=align_to == 'gaia',
        photfilename=None if align_to == 'gaia' else align_to,
        Nbright=Nbright,
    )
    aligned_image = os.path.join(
        outdir, os.path.basename(mosaic_name).replace('i2d.fits', 'jhat.fits')
    )
    aligned_mosaic = aligned_image.replace('jhat.fits', 'jhat_i2d.fits')
    shutil.move(aligned_image, aligned_mosaic)
    with fits.open(aligned_mosaic, mode='update') as filehandle:
        filehandle[0].header['JHATX'] = guess_offset[0]
        filehandle[0].header['JHATY'] = guess_offset[1]

    return aligned_mosaic, guess_offset


def cut_gaia_sources(image, table_gaia):
    """
    Drop Gaia sources that fall outside an image.

    Parameters
    ----------
    image : str
        Image file name.
    table_gaia : astropy.table.Table
        Gaia table with ``ra`` / ``dec`` columns.

    Returns
    -------
    astropy.table.Table
        Sources inside the image bounds.
    """
    im = fits.open(image)
    hdr = im['SCI'].header
    w = wcs.WCS(hdr)
    nx, ny = hdr['NAXIS1'], hdr['NAXIS2']

    pix_coords = w.all_world2pix(
        np.array(table_gaia['ra']), np.array(table_gaia['dec']), 0
    )
    im_x, im_y = pix_coords[0], pix_coords[1]
    mask = (im_x > 0) & (im_x < nx) & (im_y > 0) & (im_y < ny)

    return table_gaia[mask]


def query_gaia(image, dr='gaiadr3', telescope='jwst', save_file=False):
    """
    Query Gaia for sources covering an image.

    Parameters
    ----------
    image : str
        Image file name.
    dr : str, optional
        Gaia data release table prefix.
    telescope : str, optional
        ``'jwst'`` uses the ImageModel GWCS; ``'hst'`` uses the SCI WCS.
    save_file : str or bool, optional
        Path to save the ``ra``/``dec`` list, or ``False`` to skip.

    Returns
    -------
    astropy.table.Table
        Gaia sources inside the image.
    """
    im = fits.open(image)
    hdr = im['SCI'].header
    nx = hdr['NAXIS1']
    ny = hdr['NAXIS2']

    if telescope == 'jwst':
        image_model = ImageModel(im)

        def pix_to_world(x, y):
            return image_model.meta.wcs(x, y)

        ra0, dec0 = pix_to_world(nx / 2.0 - 1, ny / 2.0 - 1)
    elif telescope == 'hst':
        w = wcs.WCS(hdr)

        def pix_to_world(x, y):
            return w.pixel_to_world_values(x, y)

        ra0, dec0 = pix_to_world(nx / 2.0 - 1, ny / 2.0 - 1)
    else:
        raise ValueError(f'Unsupported telescope: {telescope}')

    coord0 = SkyCoord(ra0, dec0, unit=(u.deg, u.deg), frame='icrs')
    radius_deg = []
    for x in [0, nx - 1]:
        for y in [0, ny - 1]:
            ra, dec = pix_to_world(x, y)
            radius_deg.append(
                coord0.separation(
                    SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
                ).deg
            )
    radius_deg = np.amax(radius_deg) * 1.1

    query = (
        "SELECT * FROM {}.gaia_source WHERE CONTAINS(POINT('ICRS',"
        '{}.gaia_source.ra,{}.gaia_source.dec),'
        "CIRCLE('ICRS',{},{} ,{}))=1;".format(dr, dr, dr, ra0, dec0, radius_deg)
    )

    job = Gaia.launch_job_async(query)
    tb_gaia = job.get_results()
    pm_cols = ('pmra', 'pmdec', 'pmra_error', 'pmdec_error')
    if all(col in tb_gaia.colnames for col in pm_cols):
        tb_gaia['pm/pmerr'] = (tb_gaia['pmra'] ** 2 + tb_gaia['pmdec'] ** 2) / (
            tb_gaia['pmra_error'] ** 2 + tb_gaia['pmdec_error'] ** 2
        )
    tb_gaia = cut_gaia_sources(image, tb_gaia)
    print('Number of Gaia stars:', len(tb_gaia))

    if save_file:
        print(f'Saving Gaia query to {save_file}')
        np.savetxt(save_file, np.array(tb_gaia[['ra', 'dec']]), fmt='%s')

    return tb_gaia


def expand_mask(mask, size=40, mask_shape='square'):
    """
    Expand a binary mask with square or circular dilation.

    Parameters
    ----------
    mask : numpy.ndarray
        Mask where ``1`` marks kept pixels.
    size : int, optional
        Structuring-element size in pixels.
    mask_shape : {'square', 'circle'}, optional
        Structuring-element shape.

    Returns
    -------
    numpy.ndarray
        Dilated mask.
    """
    from scipy.ndimage import binary_dilation

    binary_mask = mask == 1
    if mask_shape == 'square':
        structuring_element = np.ones((size, size), dtype=bool)
    elif mask_shape == 'circle':
        y, x = np.ogrid[:size, :size]
        center = (size - 1) / 2
        structuring_element = (x - center) ** 2 + (y - center) ** 2 <= center**2
    else:
        raise ValueError(f'Unsupported mask_shape: {mask_shape}')

    expanded_mask = binary_dilation(binary_mask, structure=structuring_element)
    return np.where(expanded_mask, 1, mask)


def add_bin_dq(filename, outfile=None, mask_shape='circle'):
    """
    Write a ``*_masked.fits`` copy with an expanded binary DQ extension.

    DQ flags 1/2/3 are treated as keep; all other values are expanded and
    masked.

    Parameters
    ----------
    filename : str
        Input FITS file with a ``DQ`` extension.
    outfile : str, optional
        Output path (default: ``<filename>_masked.fits``).
    mask_shape : {'circle', 'square'}, optional
        Dilation shape for the mask.

    Returns
    -------
    str
        Path to the written file.
    """
    im = fits.open(filename)
    dq_mask = copy.deepcopy(im['DQ'].data)

    flag_sat = (dq_mask != 1) & (dq_mask != 2) & (dq_mask != 3)
    dq_mask[flag_sat] = 10
    dq_mask[~flag_sat] = 1

    expmask = expand_mask(dq_mask, mask_shape=mask_shape)
    expmask = np.where(expmask == 10, False, True)

    _data, hdr = fits.getdata(filename, ext=3, header=True)
    hdr['EXTNAME'] = 'BIN_DQ'
    image_hdu = fits.ImageHDU(data=expmask.astype(np.uint8), name='BIN_DQ', header=hdr)
    im.insert(8, image_hdu)

    if outfile is None:
        outfile = filename.replace('.fits', '_masked.fits')
    im.writeto(outfile, overwrite=True)
    im.close()
    return outfile


def calc_dispersion(ref_table, phot_file, w=False, dist_limit=1, sig=2, plot=False):
    """
    Measure the dispersion between a photometry catalog and a reference table.

    Parameters
    ----------
    ref_table : astropy.table.Table
        Reference photometry table with ``ra`` / ``dec`` columns.
    phot_file : str
        Photometry catalog to compare.
    w : astropy.wcs.WCS or False, optional
        WCS used to convert catalog ``x``/``y`` to sky; ``False`` uses the
        catalog's own ``ra``/``dec``.
    dist_limit : float, optional
        Cross-match radius in arcsec.
    sig : float, optional
        Upper sigma-clip threshold.
    plot : bool, optional
        Show a histogram of matched separations.

    Returns
    -------
    tuple
        ``(mean, median, std)`` dispersion in arcsec.
    """
    phot_df = pd.read_csv(phot_file, sep=r'\s+')

    if w:
        sky_xy = w.all_pix2world(phot_df['x'], phot_df['y'], 0)
        cat_ra, cat_dec = np.array(sky_xy[0]) * u.degree, np.array(sky_xy[1]) * u.degree
    else:
        cat_ra = phot_df['ra'].to_numpy() * u.degree
        cat_dec = phot_df['dec'].to_numpy() * u.degree
    cat_skycoord = SkyCoord(ra=cat_ra, dec=cat_dec)
    ref_ra, ref_dec = (
        np.array(ref_table['ra']) * u.degree,
        np.array(ref_table['dec']) * u.degree,
    )
    ref_skycoord = SkyCoord(ra=ref_ra, dec=ref_dec)

    dist_matched_df = xmatch_common(cat_skycoord, ref_skycoord, dist_limit=dist_limit)

    clip_mean, clip_median, clip_std = sigma_clipped_stats(
        dist_matched_df['d2d'] ** 2, sigma_lower=None, sigma_upper=sig
    )
    mean_dispersion = np.sqrt(clip_mean)
    median_dispersion = np.sqrt(clip_median)
    std_dispersion = np.sqrt(clip_std)

    if plot:
        plt.hist(
            dist_matched_df['d2d'],
            bins=40,
            histtype='step',
            linestyle='--',
            color='cornflowerblue',
        )
        plt.axvline(
            mean_dispersion,
            linestyle='--',
            alpha=0.8,
            color='royalblue',
            label='mean',
        )
        plt.axvline(
            mean_dispersion + sig * std_dispersion,
            linestyle='--',
            alpha=0.8,
            color='purple',
            label=rf'mean+{sig}$\sigma$',
        )
        plt.legend()
        plt.xlabel('d2d (arcsec)')
        plt.ylabel('frequency')
        plt.title('JWST-Gaia xmatch')
        plt.grid(alpha=0.2, linestyle='--')
        plt.show()

    return mean_dispersion, median_dispersion, std_dispersion


def jwst_dispersion(align_image, outdir, photfile=None, gaia=False, plot=False, sig=2):
    """
    Measure dispersion before and after JHAT alignment and store it in headers.

    Parameters
    ----------
    align_image : str
        Original science image (``*cal.fits`` or ``*i2d.fits``).
    outdir : str
        Directory holding the JHAT product.
    photfile : str, optional
        Reference photometry catalog; required unless ``gaia`` is True.
    gaia : bool, optional
        Compare against Gaia instead of ``photfile``.
    plot : bool, optional
        Show diagnostic histograms.
    sig : float, optional
        Sigma-clip threshold.

    Returns
    -------
    tuple
        ``(initial_mean, initial_median, final_mean, final_median)`` in arcsec.
    """
    if 'cal.fits' in align_image:
        aligned_image = os.path.join(
            outdir, os.path.basename(align_image.replace('cal.fits', 'jhat.fits'))
        )
        temp_cal_name = aligned_image.replace('jhat.fits', 'jhat_cal.fits')
        phot_image = False
    elif 'i2d.fits' in align_image:
        aligned_image = os.path.join(
            outdir, os.path.basename(align_image.replace('i2d.fits', 'jhat.fits'))
        )
        temp_cal_name = aligned_image.replace('jhat.fits', 'jhat_i2d.fits')
        phot_image = temp_cal_name
    else:
        raise ValueError('Invalid image type')

    if gaia:
        refcat = query_gaia(align_image, save_file=None)
    else:
        if photfile is None:
            raise ValueError('Input photometric catalog is required')
        refcat = Table.read(photfile, format='ascii')

    disp_in_mean, disp_in_median, _ = calc_dispersion(
        refcat,
        aligned_image.replace('_jhat.fits', '.phot.txt'),
        dist_limit=0.5,
        sig=sig,
        plot=plot,
    )
    print(f'Initial mean dispersion: {disp_in_mean * 1000} mas')
    print(f'Initial median dispersion: {disp_in_median * 1000} mas')

    os.rename(aligned_image, temp_cal_name)
    _align_cat, align_photfile = jwst_phot(temp_cal_name)
    wcs_in = wcs.WCS(fits.getheader(phot_image, ext=1)) if phot_image else False
    disp_fn_mean, disp_fn_median, disp_fn_std = calc_dispersion(
        refcat, align_photfile, w=wcs_in, sig=sig, dist_limit=0.5, plot=plot
    )
    print(f'Final mean dispersion: {disp_fn_mean * 1000} mas')
    print(f'Final median dispersion: {disp_fn_median * 1000} mas')
    os.rename(temp_cal_name, aligned_image)

    with fits.open(aligned_image, mode='update') as filehandle:
        if gaia:
            filehandle[0].header['GADISPM'] = disp_fn_mean
            filehandle[0].header['GADISPD'] = disp_fn_median
            filehandle[0].header['GADISPS'] = disp_fn_std
        else:
            filehandle[0].header['JWDISPM'] = disp_fn_mean
            filehandle[0].header['JWDISPD'] = disp_fn_median
            filehandle[0].header['JWDISPS'] = disp_fn_std
            filehandle[0].header['JWCAT'] = os.path.basename(photfile)

    return disp_in_mean, disp_in_median, disp_fn_mean, disp_fn_median


def guess_shift(align_image, ref_table, radius_px=50, res=5, sig=2, plot=False):
    """
    Grid-search the pixel shift that minimizes dispersion to a reference table.

    Parameters
    ----------
    align_image : str
        Image to shift.
    ref_table : astropy.table.Table
        Reference catalog.
    radius_px : int, optional
        Half-width of the shift search grid, in pixels.
    res : int, optional
        Grid step, in pixels.
    sig : float, optional
        Sigma-clip threshold used when scoring each shift.
    plot : bool, optional
        Show the shift landscape.

    Returns
    -------
    tuple
        ``(best_x, best_y)`` shift in pixels.
    """
    sci_hdr = copy.copy(wcs.WCS(fits.getheader(align_image, ext=1)))
    align_photfile = fix_phot(align_image)
    crpix1, crpix2 = sci_hdr.wcs.crpix

    off, xshift, yshift = [], [], []
    xsh = np.arange(-radius_px, radius_px + res, res)
    ysh = np.arange(-radius_px, radius_px + res, res)
    if 0 not in xsh:
        xsh, ysh = np.append(xsh, 0), np.append(ysh, 0)
    ng = int(len(xsh))
    for xs in xsh:
        for ys in ysh:
            in_wcs = copy.copy(sci_hdr)
            in_wcs.wcs.crpix = [crpix1 + xs, crpix2 + ys]
            _, disp, _ = calc_dispersion(
                ref_table, align_photfile, w=in_wcs, dist_limit=1, sig=sig, plot=False
            )
            off.append(disp)
            xshift.append(xs)
            yshift.append(ys)

    best_x, best_y = -xshift[np.argmin(off)], -yshift[np.argmin(off)]
    print(f'Best guess for {align_image}: ({best_x}, {best_y})')

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(24, 6))
        ax[0].scatter(np.array(xshift), np.array(off), s=5)
        ax[0].grid(linestyle='--', alpha=0.5)
        ax[0].set_xlabel('XSHIFT')
        ax[0].set_ylabel('Offset (arcsec)')

        ax[1].scatter(np.array(yshift), np.array(off), s=5)
        ax[1].grid(linestyle='--', alpha=0.5)
        ax[1].set_xlabel('YSHIFT')
        ax[1].set_ylabel('Offset (arcsec)')

        def fmt(x):
            s = f'{x * 100:.1f}'
            if s.endswith('0'):
                s = f'{x * 100:.0f}'
            return rf'{s} \%' if plt.rcParams['text.usetex'] else f'{s} %'

        im = ax[2].imshow(np.array(off).reshape(ng, ng).T)
        cs = ax[2].contour(
            np.arange(0, ng, 1),
            np.arange(0, ng, 1),
            np.array(off).reshape(ng, ng).T,
            colors='white',
            levels=3,
        )
        ax[2].clabel(cs, cs.levels, fmt=fmt, fontsize=9)
        fig.colorbar(im, ax=ax[2], pad=0.02)
        ax[2].set_xticks(
            np.linspace(0, ng - 1, 8), np.round(np.linspace(-radius_px, radius_px, 8))
        )
        ax[2].set_yticks(
            np.linspace(0, ng - 1, 8), np.round(np.linspace(-radius_px, radius_px, 8))
        )
        ax[2].set_xlabel('XSHIFT')
        ax[2].set_ylabel('YSHIFT')
        plt.show()

    return best_x, best_y


def run_jhat(
    align_image,
    outdir,
    params,
    gaia=False,
    photfilename=None,
    xshift=0,
    yshift=0,
    Nbright=800,
    verbose=False,
):
    """
    Run JHAT to align a JWST image to Gaia or a source catalog.

    JHAT's own stdout/stderr is suppressed unless ``verbose`` is set.

    Parameters
    ----------
    align_image : str
        Image to align.
    outdir : str
        Output directory; passed to JHAT as ``outrootdir``.
    params : dict
        JHAT ``run_all`` parameters.
    gaia : bool, optional
        Align to Gaia instead of ``photfilename``.
    photfilename : str, optional
        Reference photometry catalog; required unless ``gaia`` is True.
    xshift, yshift : float, optional
        Initial pixel shift seeds.
    Nbright : int, optional
        Number of bright sources to fit (catalog mode only).
    verbose : bool, optional
        Let JHAT print its own progress.
    """
    # JHAT joins outrootdir + '/' + outsubdir, defaulting outrootdir to '.'.
    # Passing an absolute path as outsubdir therefore writes under
    # cwd/.//abs/path (e.g. <repo>/data/...), while callers look in outdir.
    # Always pass the destination as outrootdir so products land correctly.
    outroot = os.path.abspath(outdir) if outdir else '.'
    wcs_align = st_wcs_align()
    if gaia:
        run_kwargs = dict(
            outrootdir=outroot,
            refcatname='Gaia',
            pmflag=True,  # propagate proper motion to observation time
            use_dq=False,
            verbose=verbose,
            xshift=xshift,
            yshift=yshift,
            **params,
        )
    else:
        if photfilename is None:
            raise ValueError('Input photometric catalog is required')
        run_kwargs = dict(
            outrootdir=outroot,
            refcatname=photfilename,
            use_dq=False,
            verbose=verbose,
            xshift=xshift,
            yshift=yshift,
            Nbright=Nbright,
            **params,
        )

    if verbose:
        wcs_align.run_all(align_image, **run_kwargs)
    else:
        with suppress_output():
            wcs_align.run_all(align_image, **run_kwargs)


def align_jwst_image(
    align_image,
    outdir,
    gaia=False,
    photfilename=None,
    xshift=0,
    yshift=0,
    Nbright=800,
    sig=2,
    verbose=False,
    plot=False,
    soft_fail=True,
    soft_fail_pix=2.0,
    jhat_params=None,
):
    """
    Align an image with JHAT, retrying with relaxed parameters when needed.

    Three attempts are made in order: strict parameters, relaxed parameters,
    and relaxed parameters seeded with a grid-searched pixel shift. The first
    solution within one pixel of the reference wins.

    Parameters
    ----------
    align_image : str
        Image to align.
    outdir : str
        Output directory for JHAT products.
    gaia : bool, optional
        Align to Gaia instead of ``photfilename``.
    photfilename : str, optional
        Reference photometry catalog.
    xshift, yshift : float, optional
        Initial pixel shift seeds.
    Nbright : int, optional
        Number of bright sources to fit.
    sig : float, optional
        Sigma-clip threshold for dispersion measurement.
    verbose : bool, optional
        Let JHAT print its own progress.
    plot : bool, optional
        Enable JHAT diagnostic plots.
    soft_fail : bool
        If True, discard a JHAT product whose median residual exceeds
        ``soft_fail_pix`` pixels and replace it with an unaligned copy.
        Set False during iterative refinement so a usable WCS is never wiped.
    soft_fail_pix : float
        Median-residual threshold in detector pixels (default 2.0).
    jhat_params : dict or None
        Optional overrides merged into ``strict_jwst_params`` /
        ``strict_gaia_params`` (e.g. tighter ``objmag_lim`` /
        ``sharpness_lim`` for F770W).

    Returns
    -------
    tuple
        ``(xshift, yshift)`` guess offset applied to the accepted solution.
    """
    print(
        f'Aligning {os.path.basename(align_image)} to '
        f"{'Gaia' if gaia else photfilename.replace('.phot.txt', '.fits')}"
    )
    params = dict(strict_gaia_params if gaia else strict_jwst_params)  # noqa: F405
    if jhat_params:
        params.update(jhat_params)
    if plot:
        params['showplots'] = 2
    try:
        wv = float(os.path.basename(align_image).split('_jhat')[0][1:4])
        factor = 2 if wv > 220 else 1
    except Exception:
        factor = 2 if 'long' in align_image else 1

    try:
        pixscale = np.abs(fits.getval(align_image, 'CDELT1', ext=1) * 3600)
    except Exception:
        pixscale = np.abs(fits.getval(align_image, 'CD1_2', ext=1) * 3600)
    pixscale = 0.031 if pixscale < 0.032 else 0.062
    retry_pix = min(float(soft_fail_pix), 1.0)

    # Initialize so failed JHAT/dispersion attempts cannot raise UnboundLocalError.
    disp_in_mu = disp_in_med = disp_fn_mu = disp_fn_med = 99.99
    guess_offset = (0, 0)

    try:
        run_jhat(
            align_image=align_image,
            outdir=outdir,
            params=params,
            gaia=gaia,
            photfilename=photfilename,
            xshift=xshift,
            yshift=yshift,
            Nbright=Nbright,
            verbose=verbose,
        )
        disp_in_mu, disp_in_med, disp_fn_mu, disp_fn_med = jwst_dispersion(
            align_image=align_image,
            outdir=outdir,
            photfile=photfilename,
            gaia=gaia,
            plot=plot,
            sig=sig,
        )
        guess_offset = (0, 0)
    except Exception:
        print(traceback.format_exc())
        disp_fn_med = 99.99

    # Retry with relaxed params / guess shift when the strict solution is still
    # worse than one MIRI/NIRCam pixel. Preserve caller JHAT overrides.
    if disp_fn_med / pixscale > retry_pix:
        params = dict(relaxed_gaia_params if gaia else relaxed_jwst_params)  # noqa: F405
        if jhat_params:
            params.update(jhat_params)
        if plot:
            params['showplots'] = 2
        try:
            run_jhat(
                align_image=align_image,
                outdir=outdir,
                params=params,
                gaia=gaia,
                photfilename=photfilename,
                xshift=xshift,
                yshift=yshift,
                Nbright=Nbright,
                verbose=verbose,
            )
            disp_in_mu, disp_in_med, disp_fn_mu, disp_fn_med = jwst_dispersion(
                align_image=align_image,
                outdir=outdir,
                photfile=photfilename,
                gaia=gaia,
                plot=plot,
                sig=1,
            )
            guess_offset = (0, 0)
        except Exception:
            print(traceback.format_exc())
            disp_fn_med = 99.99

    if disp_fn_med / pixscale > retry_pix:
        if gaia:
            ref_table = query_gaia(align_image)
        else:
            ref_table = Table.read(photfilename, format='ascii')
        xsh, ysh = guess_shift(
            align_image, ref_table, radius_px=50, res=4, sig=1, plot=plot
        )
        try:
            run_jhat(
                align_image=align_image,
                outdir=outdir,
                params=params,
                gaia=gaia,
                photfilename=photfilename,
                xshift=xsh,
                yshift=ysh,
                Nbright=Nbright,
                verbose=verbose,
            )
            disp_in_mu, disp_in_med, disp_fn_mu, disp_fn_med = jwst_dispersion(
                align_image=align_image,
                outdir=outdir,
                photfile=photfilename,
                gaia=gaia,
                plot=plot,
                sig=1,
            )
            guess_offset = (xsh * factor, ysh * factor)
        except Exception:
            print(traceback.format_exc())
            disp_fn_med = 99.99

    print(
        f"Final {'Gaia' if gaia else 'JWST'} dispersion for {align_image}: "
        f'{disp_fn_mu * 1000} mas'
    )

    def finite_arcsec(val, default=99.99):
        try:
            v = float(val)
            if np.isfinite(v):
                return v
        except Exception:
            pass
        return float(default)

    if soft_fail and disp_fn_med / pixscale > soft_fail_pix:
        print(
            f'Copying unaligned {align_image} to output, redo alignment '
            f'(median {disp_fn_med * 1000:.1f} mas > {soft_fail_pix}*pixscale)'
        )
        if 'cal.fits' in align_image:
            jhat_image = os.path.join(
                outdir, os.path.basename(align_image.replace('cal.fits', 'jhat.fits'))
            )
        elif 'i2d.fits' in align_image:
            jhat_image = os.path.join(
                outdir, os.path.basename(align_image.replace('i2d.fits', 'jhat.fits'))
            )
        else:
            jhat_image = os.path.join(
                outdir, os.path.basename(align_image).replace('.fits', '_jhat.fits')
            )
        guess_offset = (0, 0)

        shutil.copy(align_image, jhat_image)
        with fits.open(jhat_image, mode='update') as filehandle:
            d_mu = finite_arcsec(disp_in_mu)
            d_med = finite_arcsec(disp_in_med)
            if gaia:
                filehandle[0].header['GADISPM'] = d_mu
                filehandle[0].header['GADISPD'] = d_med
                filehandle[0].header['GADISPS'] = d_med
                filehandle[0].header['GANCAL'] = 0
            else:
                filehandle[0].header['JWDISPM'] = d_mu
                filehandle[0].header['JWDISPD'] = d_med
                filehandle[0].header['JWDISPS'] = d_med
                filehandle[0].header['JWNCAL'] = 0
    elif disp_fn_med / pixscale > retry_pix:
        print(
            f'Keeping JHAT solution despite median '
            f'{disp_fn_med * 1000:.1f} mas ({disp_fn_med / pixscale:.2f} pix); '
            f'below soft-fail cut ({soft_fail_pix} pix) for refine / summary'
        )

    return guess_offset


def update_refcat(mosaic_name, photfile, out_refcat, align_pgon):
    """
    Merge a visit photometry catalog into the running master reference catalog.

    Parameters
    ----------
    mosaic_name : str
        Visit mosaic whose ``FILTER`` / ``PUPIL`` decide narrowband skipping.
    photfile : str
        Photometry catalog for this visit.
    out_refcat : str
        Master catalog to create or extend.
    align_pgon : shapely geometry or None
        Already-aligned footprint; sources inside it are dropped.

    Returns
    -------
    int
        Always ``0``.
    """
    flt = fits.getval(mosaic_name, keyword='FILTER', ext=0)
    pupil = fits.getval(mosaic_name, keyword='PUPIL', ext=0)

    if not os.path.exists(out_refcat):
        refcat = Table.read(photfile, format='ascii')
        refcat[['ra', 'dec', 'mag', 'dmag']].write(
            out_refcat, format='ascii', overwrite=True
        )
        return 0

    if ('N' in flt) or ('N' in pupil):
        return 0

    mastercat = Table.read(out_refcat, format='ascii')
    refcat = Table.read(photfile, format='ascii')

    pts = shapely.points(refcat['ra'], refcat['dec'])
    refcat = refcat[~shapely.contains(align_pgon, pts)]

    mastercat = vstack([mastercat, refcat[['ra', 'dec', 'mag', 'dmag']]])
    mastercat.write(out_refcat, format='ascii', overwrite=True)

    return 0


def align_to_mosaic(
    mosaic_photfile,
    cal_images,
    outdir,
    guess_offset=(0, 0),
    verbose=False,
    ncores=10,
):
    """
    Align a set of images to the visit alignment mosaic.

    Parameters
    ----------
    mosaic_photfile : str
        Mosaic photometry catalog to align against.
    cal_images : list
        Images to align.
    outdir : str
        Output directory for JHAT products.
    guess_offset : tuple, optional
        Initial ``(xshift, yshift)`` seed; halved for long-wavelength frames.
    verbose : bool, optional
        Let JHAT print its own progress.
    ncores : int, optional
        Pool size.
    """
    repo = str(_resolve_repo_root())
    jobs = []
    for im in cal_images:
        jhat_dest = os.path.join(
            outdir, os.path.basename(im.replace('cal.fits', 'jhat.fits'))
        )
        if os.path.exists(jhat_dest):
            continue
        xs, ys = guess_offset
        if 'long' in im:
            xs = xs * 0.5
            ys = ys * 0.5
        jobs.append(
            {
                'miri_path': im,
                'align_image': im,
                'outdir': outdir,
                'gaia': False,
                'photfilename': mosaic_photfile,
                'xshift': xs,
                'yshift': ys,
                'Nbright': 800,
                'sig': 2,
                'filter': '',
                'mode': 'VISIT',
                'repo': repo,
                'verbose': bool(verbose),
            }
        )
    _run_jobs_parallel(
        jobs,
        run_visit_align_job,
        workers=ncores,
        label='VISIT',
        on_result=lambda result: print(_format_worker_done(result), flush=True),
    )


def create_dirs(work_dir, obj=None):
    """
    Create the reduction subdirectories used by visit alignment.

    Parameters
    ----------
    work_dir : str
        Reduction work directory.
    obj : str or None
        Deprecated and ignored. Photometry runs live directly under
        ``work_dir/phot_*``; the object name comes from ``--base-dir``.

    Returns
    -------
    str
        The reduction work directory.
    """
    del obj  # legacy signature compatibility
    for name in ('', 'align', 'reference', 'jhat'):
        os.makedirs(os.path.join(work_dir, name) if name else work_dir, exist_ok=True)
    return work_dir


# ---------------------------------------------------------------------------
# 4. Relative single-frame alignment API
# ---------------------------------------------------------------------------


def has_jwst_gwcs(image: str) -> bool:
    """
    Report whether a FITS file carries a JWST ASDF / GWCS extension.

    Parameters
    ----------
    image : str
        FITS file to inspect.

    Returns
    -------
    bool
        True when an ``ASDF`` extension is present.
    """
    with fits.open(image) as hdul:
        return any(hdu.name == 'ASDF' for hdu in hdul)


def phot_catalog_path(image: str, outdir: str) -> str:
    """
    Return the default catalog path for an image inside an output directory.

    Parameters
    ----------
    image : str
        Science or reference image.
    outdir : str
        Output directory.

    Returns
    -------
    str
        ``<outdir>/<image-stem>.phot.txt``.
    """
    return str(Path(outdir) / f'{Path(image).stem}.phot.txt')


def write_jhat_phot_table(table: Table, photfilename: str) -> str:
    """
    Write a catalog in the plain space-separated format JHAT can load.

    JHAT's pdastro loader does not treat ``#`` as a comment header, so the
    table is written via pandas rather than astropy ASCII.

    Parameters
    ----------
    table : astropy.table.Table
        Catalog to write.
    photfilename : str
        Destination path.

    Returns
    -------
    str
        The destination path.
    """
    Path(photfilename).parent.mkdir(parents=True, exist_ok=True)
    table.to_pandas().to_string(photfilename, index=False)
    return photfilename


def read_jhat_phot_table(photfilename: str) -> Table:
    """
    Load a JHAT-style space-separated photometry catalog.

    Parameters
    ----------
    photfilename : str
        Catalog path.

    Returns
    -------
    astropy.table.Table
        Parsed catalog.
    """
    return Table.read(photfilename, format='ascii')


def load_sci_data_wcs(image: str) -> tuple[np.ndarray, WCS]:
    """
    Load the science array and WCS from ``SCI``, else the first 2-D HDU.

    Parameters
    ----------
    image : str
        FITS file to read.

    Returns
    -------
    tuple
        ``(data, wcs)``.
    """
    with fits.open(image) as hdul:
        if 'SCI' in hdul and hdul['SCI'].data is not None:
            hdu = hdul['SCI']
        else:
            hdu = next(
                (
                    h
                    for h in hdul
                    if h.data is not None and getattr(h.data, 'ndim', 0) == 2
                ),
                None,
            )
            if hdu is None:
                raise ValueError(f'No 2-D image HDU found in {image}')
        return hdu.data.astype(float), WCS(hdu.header)


def photutils_phot(
    image: str,
    photfilename: str,
    nsigma: float = 5.0,
    fwhm: float = 3.0,
) -> str:
    """
    Run fallback DAOStarFinder photometry using the science WCS.

    Parameters
    ----------
    image : str
        Image to photometer.
    photfilename : str
        Destination catalog path.
    nsigma : float, optional
        Detection threshold in units of the background sigma.
    fwhm : float, optional
        Assumed source FWHM in pixels.

    Returns
    -------
    str
        The written catalog path.
    """
    data, image_wcs = load_sci_data_wcs(image)
    _, median, std = sigma_clipped_stats(data, sigma=3.0)
    sources = DAOStarFinder(fwhm=fwhm, threshold=nsigma * std)(data - median)
    if sources is None or len(sources) == 0:
        raise RuntimeError(f'No sources found in {image}')

    ra, dec = image_wcs.all_pix2world(sources['xcentroid'], sources['ycentroid'], 0)
    flux = np.asarray(sources['flux'], dtype=float)
    flux = np.where(flux > 0, flux, np.nan)
    mag = -2.5 * np.log10(flux)

    catalog = Table(
        {
            'x': sources['xcentroid'],
            'y': sources['ycentroid'],
            'ra': ra,
            'dec': dec,
            'mag': mag,
            'dmag': np.full(len(mag), 0.05),
        }
    )
    return write_jhat_phot_table(catalog, photfilename)


def resolve_outdir(outdir: str) -> str:
    """
    Create an output directory and return its absolute path.

    Parameters
    ----------
    outdir : str
        Directory to create.

    Returns
    -------
    str
        Resolved absolute path.
    """
    path = Path(outdir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def stage_photfile(
    photfile: str,
    outdir: str,
    dest_name: str | None = None,
) -> str:
    """
    Copy a catalog into an output directory.

    Parameters
    ----------
    photfile : str
        Catalog to stage.
    outdir : str
        Destination directory.
    dest_name : str, optional
        Destination file name (default: the source basename).

    Returns
    -------
    str
        Path to the staged catalog; a no-op when it is already there.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    dest = os.path.join(outdir, dest_name or os.path.basename(photfile))
    if os.path.exists(dest) and os.path.samefile(photfile, dest):
        return dest
    shutil.copy2(photfile, dest)
    print(f'Copied reference catalog → {dest}')
    return dest


def build_ref_catalog(
    image: str,
    outdir: str,
    photfile: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """
    Build or stage a JHAT-compatible reference photometry catalog.

    Preference order for new catalogs is :func:`jwst_phot` when JWST GWCS /
    ASDF is present, :func:`fix_phot` for i2d mosaics, then photutils
    DAOStarFinder for custom coadds without a pipeline WCS.

    Parameters
    ----------
    image : str
        Reference image to photometer.
    outdir : str
        Output directory for the staged catalog.
    photfile : str, optional
        Existing catalog to reuse instead of running photometry.
    cache_dir : str, optional
        Directory for reusing per-image catalogs across MIRI frames.

    Returns
    -------
    str
        Path to the staged catalog inside ``outdir``.
    """
    if photfile is not None:
        photfile = str(Path(photfile).expanduser().resolve())
        if not os.path.exists(photfile):
            raise FileNotFoundError(
                f'Reference photometry catalog not found: {photfile}'
            )
        print(f'Using existing reference catalog: {photfile}')
        return stage_photfile(photfile, outdir)

    dest_name = Path(phot_catalog_path(image, outdir)).name
    if cache_dir is not None:
        cache_path = Path(cache_dir).expanduser().resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        cached = cache_path / dest_name
        if cached.is_file():
            print(f'Reusing cached reference catalog: {cached}')
            return stage_photfile(str(cached), outdir, dest_name=dest_name)

    dest = phot_catalog_path(image, outdir)
    print(f'Running photometry on reference: {image}')

    if has_jwst_gwcs(image):
        print('  detected JWST ASDF/GWCS → jwst_phot')
        _, src = jwst_phot(image)
        staged = stage_photfile(src, outdir, dest_name=Path(dest).name)
    elif image.endswith(('i2d.fits', 'i2d.fits.gz')):
        try:
            print('  no JWST ASDF/GWCS; trying fix_phot')
            src = fix_phot(image)
            staged = stage_photfile(src, outdir, dest_name=Path(dest).name)
        except Exception as exc:
            print(f'  fix_phot failed ({exc}); falling back to photutils')
            staged = photutils_phot(image, dest)
    else:
        print('  falling back to photutils DAOStarFinder')
        staged = photutils_phot(image, dest)

    if cache_dir is not None:
        cache_dest = Path(cache_dir).expanduser().resolve() / Path(staged).name
        if not cache_dest.exists():
            shutil.copy2(staged, cache_dest)
            print(f'Cached reference catalog → {cache_dest}')

    return staged


def normalize_phot_columns(table: Table) -> Table:
    """
    Keep the JHAT sky-matching columns, synthesizing any that are missing.

    Parameters
    ----------
    table : astropy.table.Table
        Catalog with at least ``ra``, ``dec``, and ``mag`` columns.

    Returns
    -------
    astropy.table.Table
        Table with ``x``, ``y``, ``ra``, ``dec``, ``mag``, ``dmag`` columns.
    """
    required = ('ra', 'dec', 'mag')
    lower = {c.lower(): c for c in table.colnames}
    for name in required:
        if name not in lower:
            raise ValueError(f'Photometry table missing required column {name!r}')

    ra = np.asarray(table[lower['ra']], dtype=float)
    dec = np.asarray(table[lower['dec']], dtype=float)
    mag = np.asarray(table[lower['mag']], dtype=float)
    if 'dmag' in lower:
        dmag = np.asarray(table[lower['dmag']], dtype=float)
    else:
        dmag = np.full(len(mag), 0.05)

    # x/y are image-specific; JHAT sky matching uses ra/dec. Fill placeholders.
    if 'x' in lower and 'y' in lower:
        x = np.asarray(table[lower['x']], dtype=float)
        y = np.asarray(table[lower['y']], dtype=float)
    else:
        x = np.zeros(len(mag))
        y = np.zeros(len(mag))

    return Table({'x': x, 'y': y, 'ra': ra, 'dec': dec, 'mag': mag, 'dmag': dmag})


def merge_phot_catalogs(
    tables: list[Table],
    match_radius_arcsec: float = 0.1,
) -> Table:
    """
    Merge photometry tables on sky, keeping the brightest source on conflicts.

    All catalogs are assumed to share the same astrometric frame.

    Parameters
    ----------
    tables : list of astropy.table.Table
        Catalogs to merge.
    match_radius_arcsec : float, optional
        Sky radius within which two detections are the same source.

    Returns
    -------
    astropy.table.Table
        Merged catalog sorted brightest first.
    """
    if not tables:
        raise ValueError('No photometry tables to merge')

    merged = normalize_phot_columns(tables[0])
    for table in tables[1:]:
        incoming = normalize_phot_columns(table)
        if len(merged) == 0:
            merged = incoming
            continue
        if len(incoming) == 0:
            continue

        ref_coord = SkyCoord(ra=merged['ra'] * u.deg, dec=merged['dec'] * u.deg)
        new_coord = SkyCoord(ra=incoming['ra'] * u.deg, dec=incoming['dec'] * u.deg)
        idx, sep, _ = new_coord.match_to_catalog_sky(ref_coord)
        matched = sep < (match_radius_arcsec * u.arcsec)

        # Replace existing entries with brighter matches.
        for i in np.where(matched)[0]:
            j = int(idx[i])
            if incoming['mag'][i] < merged['mag'][j]:
                for col in ('x', 'y', 'ra', 'dec', 'mag', 'dmag'):
                    merged[col][j] = incoming[col][i]

        # Append unmatched sources.
        if np.any(~matched):
            merged = vstack([merged, incoming[~matched]], join_type='exact')

    # Brightest first so JHAT's Nbright cut prefers useful stars.
    merged.sort('mag')
    return merged


def clip_catalog_to_miri_footprint(table: Table, miri_image: str) -> Table:
    """
    Keep only sources inside the MIRI illuminated region of interest.

    Clipping is done in sky coordinates using the illuminated ``S_REGION``
    polygon. This avoids ``WCS.all_world2pix`` ``NoConvergence`` errors that
    occur when the merged master catalog contains stars far outside the MIRI
    WCS validity domain (common once several large reference coadds are
    merged).

    Parameters
    ----------
    table : astropy.table.Table
        Catalog with ``ra`` / ``dec`` columns.
    miri_image : str
        MIRI frame whose illuminated footprint defines the clip.

    Returns
    -------
    astropy.table.Table
        Clipped catalog.
    """
    from matplotlib.path import Path as MplPath

    from st123.mosaic.image_overlap import MirIFootprint

    miri = MirIFootprint.from_fits(miri_image)
    ra = np.asarray(table['ra'], dtype=float)
    dec = np.asarray(table['dec'], dtype=float)

    # Illuminated S_REGION vertices are (lon, lat) in degrees.
    verts = np.asarray(miri.s_region.vertices, dtype=float)
    if len(verts) < 3:
        raise RuntimeError(
            f'Illuminated S_REGION has too few vertices for {miri_image}'
        )

    # Handle simple RA wrap near 0/360 if the footprint crosses the branch cut.
    ra_v = verts[:, 0].copy()
    if ra_v.max() - ra_v.min() > 180.0:
        ra_v = np.where(ra_v < 180.0, ra_v + 360.0, ra_v)
        ra_use = np.where(ra < 180.0, ra + 360.0, ra)
    else:
        ra_use = ra

    sky_path = MplPath(np.column_stack([ra_v, verts[:, 1]]))
    keep = sky_path.contains_points(np.column_stack([ra_use, dec]))

    # Cross-check with the pixel-plane illuminated polygon when the sky clip
    # yields nothing (should be rare; quiet=True avoids hard WCS failures).
    if not np.any(keep):
        print(
            'WARNING: sky-footprint clip kept 0 sources; '
            'retrying with quiet WCS pixel clip'
        )
        from shapely.geometry import Point

        x, y = miri.wcs.all_world2pix(ra, dec, 0, quiet=True)
        keep = np.array(
            [
                np.isfinite(xi)
                and np.isfinite(yi)
                and miri.polygon.contains(Point(float(xi), float(yi)))
                for xi, yi in zip(x, y)
            ],
            dtype=bool,
        )

    clipped = table[keep]
    print(
        f'Clipped master catalog to MIRI footprint: '
        f'{len(table)} → {len(clipped)} sources'
    )
    return clipped


def build_master_ref_catalog(
    ref_images: list[str],
    outdir: str,
    *,
    align_image: str | None = None,
    cache_dir: str | None = None,
    match_radius_arcsec: float = 0.1,
    clip_to_align_footprint: bool = True,
    dest_name: str = 'master_ref.phot.txt',
) -> str:
    """
    Build a master JHAT catalog from every overlapping reference image.

    Photometry is run (or loaded from ``cache_dir``) for each reference, merged
    on sky, optionally clipped to the MIRI illuminated footprint, and written
    to ``outdir/dest_name``.

    Parameters
    ----------
    ref_images : list of str
        Reference images to photometer.
    outdir : str
        Output directory.
    align_image : str, optional
        Science frame used for footprint clipping.
    cache_dir : str, optional
        Cache directory for per-reference catalogs.
    match_radius_arcsec : float, optional
        Sky radius for merging duplicate detections.
    clip_to_align_footprint : bool, optional
        Clip the merged catalog to the ``align_image`` footprint.
    dest_name : str, optional
        Output file name.

    Returns
    -------
    str
        Path to the master catalog.
    """
    if not ref_images:
        raise ValueError('ref_images is empty')

    outdir = resolve_outdir(outdir)
    tables: list[Table] = []
    for ref in ref_images:
        ref = str(Path(ref).expanduser().resolve())
        try:
            phot = build_ref_catalog(ref, outdir, cache_dir=cache_dir)
            tables.append(read_jhat_phot_table(phot))
            print(f'  + {len(tables[-1])} sources from {ref}')
        except Exception as exc:
            print(f'  WARNING: photometry failed for {ref}: {exc}')

    if not tables:
        raise RuntimeError('No reference photometry catalogs could be built')

    master = merge_phot_catalogs(tables, match_radius_arcsec=match_radius_arcsec)
    print(
        f'Merged {len(tables)} reference catalog(s) → {len(master)} unique sources '
        f'(match radius {match_radius_arcsec}" )'
    )

    if clip_to_align_footprint and align_image is not None:
        master = clip_catalog_to_miri_footprint(master, align_image)
        if len(master) == 0:
            raise RuntimeError(
                'Master catalog has no sources inside the MIRI illuminated footprint'
            )

    dest = str(Path(outdir) / dest_name)
    write_jhat_phot_table(master, dest)
    print(f'Wrote master reference catalog → {dest} ({len(master)} sources)')
    return dest


@contextmanager
def working_directory(path: str):
    """
    Temporarily change the process working directory.

    Parameters
    ----------
    path : str
        Directory to enter for the duration of the block.
    """
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


@contextmanager
def plot_saver(outdir: str, prefix: str):
    """
    Save JHAT figures that only call ``plt.show()`` into a directory.

    Parameters
    ----------
    outdir : str
        Directory for the saved PNGs.
    prefix : str
        File-name prefix for each diagnostic figure.
    """
    counter = {'n': 0}
    already_saved: set[int] = set()
    original_show = plt.show
    original_savefig = plt.Figure.savefig

    def savefig_track(self, *args, **kwargs):
        already_saved.add(self.number)
        return original_savefig(self, *args, **kwargs)

    def show_and_save(*args, **kwargs):
        for num in plt.get_fignums():
            if num in already_saved:
                continue
            fig = plt.figure(num)
            counter['n'] += 1
            out = os.path.join(outdir, f'{prefix}.diag_{counter["n"]:02d}.png')
            # Use the original savefig so we do not mark this as a JHAT savefig.
            original_savefig(fig, out, dpi=150, bbox_inches='tight')
            print(f'Saved diagnostic plot: {out}')
        already_saved.clear()
        plt.close('all')

    plt.Figure.savefig = savefig_track
    plt.show = show_and_save
    try:
        yield
    finally:
        plt.show = original_show
        plt.Figure.savefig = original_savefig


def jhat_product_paths(align_image: str, outdir: str) -> tuple[str, str, str]:
    """
    Return the JHAT product paths for a science frame.

    Parameters
    ----------
    align_image : str
        Science image.
    outdir : str
        Directory holding JHAT products.

    Returns
    -------
    tuple
        ``(jhat_fits, align_phot, stem)``. ``align_phot`` prefers
        post-alignment photometry written by :func:`jwst_dispersion`.
    """
    base = Path(align_image).name
    if base.endswith('_cal.fits'):
        stem = base.replace('_cal.fits', '')
        jhat = str(Path(outdir) / base.replace('_cal.fits', '_jhat.fits'))
    elif base.endswith('_i2d.fits'):
        stem = base.replace('_i2d.fits', '')
        jhat = str(Path(outdir) / base.replace('_i2d.fits', '_jhat.fits'))
    else:
        stem = Path(base).stem
        jhat = str(Path(outdir) / f'{stem}_jhat.fits')
    phot_candidates = [
        str(Path(outdir) / f'{stem}_jhat_cal.phot.txt'),
        str(Path(outdir) / f'{stem}_jhat_i2d.phot.txt'),
        str(Path(outdir) / f'{stem}.phot.txt'),
    ]
    align_phot = next(
        (p for p in phot_candidates if os.path.exists(p)), phot_candidates[0]
    )
    return jhat, align_phot, stem


def read_dispersion_mas(jhat_image: str) -> tuple[float | None, int | None]:
    """
    Read the final dispersion and calibrator count from a JHAT product.

    Parameters
    ----------
    jhat_image : str
        JHAT product to read.

    Returns
    -------
    tuple
        ``(dispersion_mas, n_calibrators)``; either may be ``None``.
    """
    with fits.open(jhat_image) as hdul:
        hdr = hdul[0].header
        disp = hdr.get('JWDISPM', hdr.get('GADISPM'))
        ncal = hdr.get('JWNCAL', hdr.get('GANCAL'))
    disp_mas = float(disp) * 1000.0 if disp is not None else None
    n_cal = int(ncal) if ncal is not None else None
    return disp_mas, n_cal


def read_jhat_pixel_offset(jhat_image: str) -> tuple[float, float]:
    """
    Read the pixel offsets applied by JHAT.

    Parameters
    ----------
    jhat_image : str
        JHAT product to read.

    Returns
    -------
    tuple
        ``(xshift, yshift)`` in pixels.
    """
    with fits.open(jhat_image) as hdul:
        hdr = hdul[0].header
        xoff = hdr.get('XOFFSET', 0.0)
        yoff = hdr.get('YOFFSET', 0.0)
    return float(xoff or 0.0), float(yoff or 0.0)


def miri_calibrator_mask(
    jhat_df,
    *,
    miri_mag_min: float | None = None,
    miri_mag_max: float | None = None,
    miri_round_max: float | None = None,
    miri_sharp_min: float | None = None,
    miri_sharp_max: float | None = None,
):
    """
    Build a boolean mask of star-like MIRI detections suitable as calibrators.

    Parameters
    ----------
    jhat_df : pandas.DataFrame
        Aligned MIRI photometry.
    miri_mag_min, miri_mag_max : float, optional
        Magnitude bounds applied when a ``mag`` column exists.
    miri_round_max : float, optional
        Maximum absolute ``roundness1``.
    miri_sharp_min, miri_sharp_max : float, optional
        ``sharpness`` bounds.

    Returns
    -------
    numpy.ndarray
        Boolean mask of rows to keep.
    """
    if not isinstance(jhat_df, pd.DataFrame):
        jhat_df = pd.DataFrame(jhat_df)
    keep = np.ones(len(jhat_df), dtype=bool)
    if miri_mag_min is not None and 'mag' in jhat_df.columns:
        keep &= np.asarray(jhat_df['mag'], dtype=float) >= float(miri_mag_min)
    if miri_mag_max is not None and 'mag' in jhat_df.columns:
        keep &= np.asarray(jhat_df['mag'], dtype=float) <= float(miri_mag_max)
    if miri_round_max is not None and 'roundness1' in jhat_df.columns:
        keep &= np.abs(np.asarray(jhat_df['roundness1'], dtype=float)) <= float(
            miri_round_max
        )
    if miri_sharp_min is not None and 'sharpness' in jhat_df.columns:
        keep &= np.asarray(jhat_df['sharpness'], dtype=float) >= float(miri_sharp_min)
    if miri_sharp_max is not None and 'sharpness' in jhat_df.columns:
        keep &= np.asarray(jhat_df['sharpness'], dtype=float) <= float(miri_sharp_max)
    return keep


def iterative_sigma_clip_matches(
    align_phot: str,
    ref_table: Table,
    *,
    dist_limit_arcsec: float = 0.5,
    sigma: float = 2.0,
    max_clip_iter: int = 10,
    max_residual_arcsec: float | None = None,
    miri_mag_min: float | None = None,
    miri_mag_max: float | None = None,
    miri_round_max: float | None = None,
    miri_sharp_min: float | None = None,
    miri_sharp_max: float | None = None,
) -> tuple[Table, dict]:
    """
    Cross-match aligned photometry to a reference table and clip outliers.

    Optional MIRI morphology / magnitude cuts are applied before matching when
    those columns exist. An optional hard residual ceiling is applied after
    sigma-clipping.

    Parameters
    ----------
    align_phot : str
        Aligned MIRI photometry catalog.
    ref_table : astropy.table.Table
        Reference catalog.
    dist_limit_arcsec : float, optional
        Cross-match radius.
    sigma : float, optional
        Sigma-clip threshold.
    max_clip_iter : int, optional
        Maximum clipping iterations.
    max_residual_arcsec : float, optional
        Hard residual ceiling.
    miri_mag_min, miri_mag_max, miri_round_max, miri_sharp_min, miri_sharp_max
        See :func:`miri_calibrator_mask`.

    Returns
    -------
    tuple
        ``(cleaned_ref_table, stats)``.
    """
    jhat_df = pd.read_csv(align_phot, sep=r'\s+')
    morph_keep = miri_calibrator_mask(
        jhat_df,
        miri_mag_min=miri_mag_min,
        miri_mag_max=miri_mag_max,
        miri_round_max=miri_round_max,
        miri_sharp_min=miri_sharp_min,
        miri_sharp_max=miri_sharp_max,
    )
    n_morph = int(morph_keep.sum())
    if n_morph == 0:
        raise RuntimeError('No MIRI sources survive calibrator morphology/mag cuts')
    if n_morph < len(jhat_df):
        print(f'  calibrator morph/mag cut: kept {n_morph}/{len(jhat_df)} MIRI sources')
    jhat_df = jhat_df.loc[morph_keep].reset_index(drop=True)

    jh = SkyCoord(
        ra=np.asarray(jhat_df['ra'], dtype=float) * u.deg,
        dec=np.asarray(jhat_df['dec'], dtype=float) * u.deg,
    )
    rf = SkyCoord(
        ra=np.asarray(ref_table['ra'], dtype=float) * u.deg,
        dec=np.asarray(ref_table['dec'], dtype=float) * u.deg,
    )
    matched = xmatch_common(jh, rf, dist_limit=dist_limit_arcsec)
    if len(matched) == 0:
        raise RuntimeError('No matches for iterative outlier clipping')

    d2d = np.asarray(matched['d2d'], dtype=float)
    keep = np.ones(len(matched), dtype=bool)
    for i in range(max_clip_iter):
        di = d2d[keep]
        _mn, med, std = sigma_clipped_stats(di, sigma=sigma)
        thr = float(med + sigma * std)
        if max_residual_arcsec is not None:
            thr = min(thr, float(max_residual_arcsec))
        new_keep = keep & (d2d <= thr)
        print(
            f'  clip iter {i}: n={int(new_keep.sum())}/{len(matched)} '
            f'med={np.median(d2d[new_keep]) * 1000:.2f} mas '
            f'thr={thr * 1000:.2f} mas'
        )
        if new_keep.sum() == keep.sum():
            keep = new_keep
            break
        keep = new_keep
        if keep.sum() < 10:
            break

    if max_residual_arcsec is not None:
        hard = d2d <= float(max_residual_arcsec)
        if hard.sum() < keep.sum():
            print(
                f'  hard residual cut ({max_residual_arcsec * 1000:.1f} mas): '
                f'{int(keep.sum())} → {int((keep & hard).sum())}'
            )
        keep = keep & hard

    good_ref_idx = np.unique(np.asarray(matched['idx_2'], dtype=int)[keep])
    cleaned = ref_table[good_ref_idx]
    stats = {
        'n_match_initial': int(len(matched)),
        'n_match_clipped': int(keep.sum()),
        'n_ref_kept': int(len(cleaned)),
        'median_d2d_mas': (
            float(np.median(d2d[keep]) * 1000.0) if keep.any() else float('nan')
        ),
        'mean_d2d_mas': (
            float(np.mean(d2d[keep]) * 1000.0) if keep.any() else float('nan')
        ),
        'n_miri_morph': n_morph,
    }
    return cleaned, stats


def refine_alignment_iteratively(
    align_image: str,
    outdir: str,
    ref_phot: str,
    *,
    nbright: int = 800,
    plot: bool = False,
    verbose: bool = False,
    sigma: float = 2.0,
    max_iter: int = 5,
    tol_mas: float = 1.0,
    min_calibrators: int = 20,
    dist_limit_arcsec: float = 0.5,
    max_residual_arcsec: float | None = None,
    miri_mag_min: float | None = None,
    miri_mag_max: float | None = None,
    miri_round_max: float | None = None,
    miri_sharp_min: float | None = None,
    miri_sharp_max: float | None = None,
    jhat_params: dict | None = None,
) -> tuple[object, float | None, int | None]:
    """
    Prune outlier reference stars and re-run JHAT until dispersion converges.

    Each iteration cross-matches the current aligned photometry to the
    reference catalog, sigma-clips residual outliers, rewrites a cleaned
    catalog, re-runs JHAT from the original science image, and stops when
    ``|delta dispersion| < tol_mas`` or no further stars are clipped.

    Parameters
    ----------
    align_image : str
        Science image being aligned.
    outdir : str
        Directory holding JHAT products.
    ref_phot : str
        Starting reference catalog.
    nbright : int, optional
        Number of bright sources for JHAT.
    plot, verbose : bool, optional
        Diagnostics and JHAT verbosity.
    sigma : float, optional
        Sigma-clip threshold.
    max_iter : int, optional
        Maximum refine iterations.
    tol_mas : float, optional
        Convergence tolerance in mas.
    min_calibrators : int, optional
        Stop when fewer reference stars survive.
    dist_limit_arcsec : float, optional
        Cross-match radius.
    max_residual_arcsec : float, optional
        Hard residual ceiling; also enables pixel-offset seeding.
    miri_mag_min, miri_mag_max, miri_round_max, miri_sharp_min, miri_sharp_max
        See :func:`miri_calibrator_mask`.
    jhat_params : dict, optional
        JHAT parameter overrides.

    Returns
    -------
    tuple
        ``(guess_offset, dispersion_mas, n_calibrators)``.
    """
    outdir = resolve_outdir(outdir)
    align_image = str(Path(align_image).expanduser().resolve())
    ref_phot = str(Path(ref_phot).expanduser().resolve())
    stem = Path(align_image).stem.replace('_cal', '').replace('_i2d', '')

    jhat, align_phot, _ = jhat_product_paths(align_image, outdir)
    if not os.path.exists(jhat):
        raise FileNotFoundError(f'JHAT product not found for refinement: {jhat}')

    disp_mas, n_cal = read_dispersion_mas(jhat)
    xshift, yshift = read_jhat_pixel_offset(jhat)
    guess_offset: object = (xshift, yshift)
    print(
        f'Iterative refinement starting from dispersion={disp_mas} mas, '
        f'n_calibrators={n_cal}, xshift={xshift:.3f}, yshift={yshift:.3f}'
    )

    current_ref = ref_phot
    for it in range(1, max_iter + 1):
        if not os.path.exists(align_phot):
            print(
                f'  refine iter {it}: missing align photometry {align_phot}; stopping'
            )
            break

        ref_table = read_jhat_phot_table(current_ref)
        try:
            cleaned, stats = iterative_sigma_clip_matches(
                align_phot,
                ref_table,
                dist_limit_arcsec=dist_limit_arcsec,
                sigma=sigma,
                max_residual_arcsec=max_residual_arcsec,
                miri_mag_min=miri_mag_min,
                miri_mag_max=miri_mag_max,
                miri_round_max=miri_round_max,
                miri_sharp_min=miri_sharp_min,
                miri_sharp_max=miri_sharp_max,
            )
        except Exception as exc:
            print(f'  refine iter {it}: clipping failed ({exc}); stopping')
            break

        if stats['n_ref_kept'] < min_calibrators:
            print(
                f'  refine iter {it}: only {stats["n_ref_kept"]} ref stars left '
                f'(<{min_calibrators}); stopping'
            )
            break

        # Only re-run JHAT when the matched set actually shrank (sigma-clip /
        # hard residual). Do NOT treat "matched subset << full master_ref" as
        # progress — that is always true on the first pass and was forcing a
        # destructive catalog rewrite that hurt F560W.
        #
        # Optional MIRI morph/mag cuts are applied *before* matching, so when
        # they are active force one cleaned-catalog re-run on iter 1 even if
        # sigma-clip finds no further outliers.
        morph_active = any(
            v is not None
            for v in (
                miri_mag_min,
                miri_mag_max,
                miri_round_max,
                miri_sharp_min,
                miri_sharp_max,
            )
        )
        no_clip_progress = stats['n_match_clipped'] >= stats['n_match_initial']
        force_morph_pass = (
            morph_active
            and it == 1
            and int(stats.get('n_miri_morph', 0)) > 0
            and current_ref == ref_phot
        )
        if no_clip_progress and not force_morph_pass:
            print(f'  refine iter {it}: no outliers clipped; converged')
            break

        cleaned_path = str(Path(outdir) / f'master_ref_refined_iter{it:02d}.phot.txt')
        write_jhat_phot_table(cleaned, cleaned_path)
        print(
            f'  refine iter {it}: wrote cleaned catalog {cleaned_path} '
            f'({len(cleaned)} stars; match median {stats["median_d2d_mas"]:.2f} mas)'
        )

        # Snapshot current JHAT products so a worse iteration can be rolled back.
        jhat, align_phot, _ = jhat_product_paths(align_image, outdir)
        backup_jhat = jhat + '.refine_bak'
        backup_phot = align_phot + '.refine_bak'
        if os.path.exists(jhat):
            shutil.copy2(jhat, backup_jhat)
        if os.path.exists(align_phot):
            shutil.copy2(align_phot, backup_phot)

        # Only seed pixel offsets for F770W-style tight refine (hard residual
        # ceiling). Seeding large F560W XOFFSET/YOFFSET (~50–130 px) into JHAT
        # with a cleaned catalog routinely fails matching and destroys the
        # ~10 mas solutions refine would otherwise recover.
        if max_residual_arcsec is not None:
            xshift, yshift = read_jhat_pixel_offset(jhat)
        else:
            xshift, yshift = 0.0, 0.0

        with working_directory(outdir):
            # soft_fail=False: never replace a refine JHAT product with an
            # unaligned copy; rollback handles worsened iterations instead.
            align_kw = dict(
                align_image=align_image,
                outdir='.',
                gaia=False,
                photfilename=cleaned_path,
                Nbright=min(nbright, len(cleaned)),
                verbose=verbose,
                soft_fail=False,
                jhat_params=jhat_params,
                xshift=xshift,
                yshift=yshift,
            )
            if plot:
                with plot_saver(outdir, prefix=f'{stem}.refine{it:02d}'):
                    guess_offset = align_jwst_image(plot=True, **align_kw)
            else:
                guess_offset = align_jwst_image(plot=False, **align_kw)

        jhat, align_phot, _ = jhat_product_paths(align_image, outdir)
        new_disp, new_ncal = read_dispersion_mas(jhat)
        print(
            f'  refine iter {it}: dispersion {disp_mas} → {new_disp} mas '
            f'(n_calibrators={new_ncal})'
        )

        def restore_backup_and_stop(reason: str) -> None:
            print(
                f'  refine iter {it}: {reason}; '
                f'restoring previous JHAT products and stopping'
            )
            if os.path.exists(backup_jhat):
                shutil.copy2(backup_jhat, jhat)
            if os.path.exists(backup_phot):
                shutil.copy2(backup_phot, align_phot)
            for bak in (backup_jhat, backup_phot):
                if os.path.exists(bak):
                    os.remove(bak)

        # Failed / soft-failed refine attempts can leave absurd dispersions.
        if new_disp is not None and new_disp > 500.0:
            restore_backup_and_stop(
                f'refine product unusable (dispersion={new_disp:.1f} mas)'
            )
            break

        if disp_mas is not None and new_disp is not None:
            if abs(new_disp - disp_mas) < tol_mas:
                print(
                    f'  refine iter {it}: |Δdispersion|='
                    f'{abs(new_disp - disp_mas):.3f} mas < {tol_mas} mas; converged'
                )
                disp_mas, n_cal = new_disp, new_ncal
                current_ref = cleaned_path
                for bak in (backup_jhat, backup_phot):
                    if os.path.exists(bak):
                        os.remove(bak)
                break
            if new_disp > disp_mas + tol_mas:
                restore_backup_and_stop(
                    f'dispersion worsened ({disp_mas:.3f} → {new_disp:.3f} mas)'
                )
                break

        disp_mas, n_cal = new_disp, new_ncal
        current_ref = cleaned_path
        for bak in (backup_jhat, backup_phot):
            if os.path.exists(bak):
                os.remove(bak)

    # Stage final cleaned catalog under a stable name when available.
    if current_ref != ref_phot and os.path.exists(current_ref):
        final_clean = str(Path(outdir) / 'master_ref_refined.phot.txt')
        if os.path.abspath(current_ref) != os.path.abspath(final_clean):
            shutil.copy2(current_ref, final_clean)
            print(f'Final refined reference catalog → {final_clean}')

    print(
        f'Iterative refinement finished: dispersion_mas={disp_mas}, '
        f'n_calibrators={n_cal}'
    )
    return guess_offset, disp_mas, n_cal


def run_alignment(
    ref_image: str | None = None,
    align_image: str | None = None,
    outdir: str = 'alignment_output',
    photfile: str | None = None,
    ref_images: list[str] | None = None,
    nbright: int = 800,
    plot: bool = False,
    verbose: bool = False,
    cache_dir: str | None = None,
    match_radius_arcsec: float = 0.1,
    clip_to_align_footprint: bool = True,
    refine: bool = True,
    refine_sigma: float = 2.0,
    refine_max_iter: int = 5,
    refine_tol_mas: float = 1.0,
    refine_dist_limit_arcsec: float = 0.5,
    max_residual_arcsec: float | None = None,
    miri_mag_min: float | None = None,
    miri_mag_max: float | None = None,
    miri_round_max: float | None = None,
    miri_sharp_min: float | None = None,
    miri_sharp_max: float | None = None,
    min_calibrators: int = 20,
    jhat_params: dict | None = None,
) -> tuple[object, str]:
    """
    Build a reference catalog and align one science image to it.

    The catalog source is the first of ``photfile`` (reuse an existing
    catalog), ``ref_images`` (merge photometry from all listed references into
    a master catalog), or ``ref_image`` (single-reference photometry).

    Parameters
    ----------
    ref_image : str, optional
        Single reference image.
    align_image : str
        Science image to align. Required.
    outdir : str, optional
        Output directory.
    photfile : str, optional
        Existing reference catalog.
    ref_images : list of str, optional
        Multiple reference images to merge.
    nbright : int, optional
        Number of bright sources for JHAT.
    plot, verbose : bool, optional
        Diagnostics and JHAT verbosity.
    cache_dir : str, optional
        Reference photometry cache directory.
    match_radius_arcsec : float, optional
        Sky radius for merging reference catalogs.
    clip_to_align_footprint : bool, optional
        Clip the master catalog to the science illuminated footprint.
    refine : bool, optional
        Run :func:`refine_alignment_iteratively` after the first JHAT pass.
    refine_sigma, refine_max_iter, refine_tol_mas, refine_dist_limit_arcsec
        Refine loop controls.
    max_residual_arcsec, miri_mag_min, miri_mag_max, miri_round_max,
    miri_sharp_min, miri_sharp_max, min_calibrators
        Calibrator selection knobs, usually supplied per filter by
        :func:`calibrator_settings_for_filter`.
    jhat_params : dict, optional
        JHAT parameter overrides.

    Returns
    -------
    tuple
        ``(guess_offset, outdir)``.
    """
    if align_image is None:
        raise ValueError('align_image is required')

    align_image = str(Path(align_image).expanduser().resolve())
    outdir = resolve_outdir(outdir)

    refs = list(ref_images) if ref_images else []
    if not refs and ref_image is not None:
        refs = [ref_image]
    refs = [str(Path(r).expanduser().resolve()) for r in refs]

    print(f'Output directory: {outdir}')
    print(f'Align image:      {align_image}')
    if jhat_params:
        print(f'JHAT param overrides: {jhat_params}')
    print(
        f'Calibrator knobs: nbright={nbright}, refine_sigma={refine_sigma}, '
        f'dist_limit={refine_dist_limit_arcsec}", '
        f'max_resid={max_residual_arcsec}"'
    )

    if photfile is not None:
        ref_phot = build_ref_catalog(
            refs[0] if refs else align_image, outdir, photfile=photfile
        )
        print(f'Reference catalog: {ref_phot}')
    elif len(refs) > 1:
        print(f'Reference images ({len(refs)}): building master catalog')
        for path in refs:
            print(f'  {path}')
        ref_phot = build_master_ref_catalog(
            refs,
            outdir,
            align_image=align_image,
            cache_dir=cache_dir,
            match_radius_arcsec=match_radius_arcsec,
            clip_to_align_footprint=clip_to_align_footprint,
        )
    elif len(refs) == 1:
        print(f'Reference image:  {refs[0]}')
        ref_phot = build_ref_catalog(refs[0], outdir, cache_dir=cache_dir)
    else:
        raise ValueError('Provide photfile, ref_image, or ref_images')

    # Run from outdir so relative JHAT artifacts (plots, etc.) land beside
    # products. Paths above are absolute so chdir is safe.
    stem = Path(align_image).stem.replace('_cal', '').replace('_i2d', '')
    align_kw = dict(
        align_image=align_image,
        outdir='.',
        gaia=False,
        photfilename=ref_phot,
        Nbright=nbright,
        verbose=verbose,
        jhat_params=jhat_params,
    )
    with working_directory(outdir):
        if plot:
            with plot_saver(outdir, prefix=stem):
                guess_offset = align_jwst_image(plot=True, **align_kw)
        else:
            guess_offset = align_jwst_image(plot=False, **align_kw)

    if refine:
        guess_offset, disp_mas, n_cal = refine_alignment_iteratively(
            align_image=align_image,
            outdir=outdir,
            ref_phot=ref_phot,
            nbright=nbright,
            plot=plot,
            verbose=verbose,
            sigma=refine_sigma,
            max_iter=refine_max_iter,
            tol_mas=refine_tol_mas,
            min_calibrators=min_calibrators,
            dist_limit_arcsec=refine_dist_limit_arcsec,
            max_residual_arcsec=max_residual_arcsec,
            miri_mag_min=miri_mag_min,
            miri_mag_max=miri_mag_max,
            miri_round_max=miri_round_max,
            miri_sharp_min=miri_sharp_min,
            miri_sharp_max=miri_sharp_max,
            jhat_params=jhat_params,
        )
        print(f'Refined dispersion_mas={disp_mas}, n_calibrators={n_cal}')

    return guess_offset, outdir


# ---------------------------------------------------------------------------
# 5. Overlap discovery, metric harvesting, and summary tables
# ---------------------------------------------------------------------------


def _resolve_repo_root(explicit: Path | None = None) -> Path:
    """
    Locate the installable st123 repo root used to bootstrap spawn workers.

    Parameters
    ----------
    explicit : pathlib.Path, optional
        Caller-supplied repo root (``--repo``).

    Returns
    -------
    pathlib.Path
        Directory containing the ``st123`` package.
    """
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not (root / 'st123' / 'alignment' / 'align.py').is_file() and not (
            root / 'alignment' / 'align.py'
        ).is_file():
            raise FileNotFoundError(f'--repo does not look like st123: {root}')
        return root

    here = Path(__file__).resolve().parent  # <repo>/st123/alignment
    repo_root = here.parents[1]
    for cand in (repo_root, Path.cwd(), Path('/data/rwisenbaker/st123')):
        if (cand / 'st123' / 'alignment' / 'align.py').is_file():
            return cand.resolve()
    return repo_root.resolve()


def _bootstrap_imports(repo: Path) -> None:
    """
    Ensure the repo root is importable.

    Parameters
    ----------
    repo : pathlib.Path
        Repo root to prepend to ``sys.path``.
    """
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


@dataclass(frozen=True)
class FrameOverlaps:
    """Best reference plus every reference with nonzero overlap for one frame."""

    miri_path: str
    best: object  # image_overlap.BestOverlap
    overlapping: list  # list[image_overlap.OverlapResult]
    # Unique MIRI-ROI fraction covered by the union of all overlapping refs.
    union_overlap_fraction: float = 0.0

    def to_dict(self) -> dict:
        """
        Serialize to a JSON-friendly dictionary.

        Returns
        -------
        dict
            Nested overlap metrics for ``overlap_summary.json``.
        """
        return {
            'miri_path': self.miri_path,
            'best': {
                'ref_path': self.best.ref_path,
                'overlap_area': asdict(self.best.overlap_area),
            },
            'overlapping': [
                {
                    'ref_path': r.ref_path,
                    'overlap_area': asdict(r.overlap_area),
                    'ref_area': asdict(r.ref_area),
                }
                for r in self.overlapping
            ],
            'union_overlap_fraction': float(self.union_overlap_fraction),
        }


@dataclass
class AlignmentSummaryRow:
    """One row of the dataset alignment summary table."""

    miri_path: str
    filter: str
    status: str
    n_calibrators: int | str
    dispersion_mas: float | str
    aligned_path: str = 'NA'
    align_mode: str = 'NA'
    original_ref: str = 'NA'
    aligned_to: str = 'NA'
    # Cumulative unique fraction of MIRI ROI covered by all overlapping refs.
    ref_overlap_frac: float | str = 'NA'

    def format_line(self, widths: dict[str, int]) -> str:
        """
        Render this row as a fixed-width text line.

        Parameters
        ----------
        widths : dict
            Column name mapped to field width.

        Returns
        -------
        str
            Formatted line without a trailing newline.
        """
        disp = (
            f'{self.dispersion_mas:.3f}'
            if isinstance(self.dispersion_mas, float)
            else str(self.dispersion_mas)
        )
        ov = (
            f'{self.ref_overlap_frac:.4f}'
            if isinstance(self.ref_overlap_frac, float)
            else str(self.ref_overlap_frac)
        )
        ncal = str(self.n_calibrators)
        return (
            f'{self.miri_path:<{widths["miri_path"]}}  '
            f'{self.filter:<{widths["filter"]}}  '
            f'{self.status:<{widths["status"]}}  '
            f'{ov:>{widths["ref_overlap_frac"]}}  '
            f'{ncal:>{widths["n_calibrators"]}}  '
            f'{disp:>{widths["dispersion_mas"]}}  '
            f'{self.align_mode:<{widths["align_mode"]}}  '
            f'{self.aligned_path:<{widths["aligned_path"]}}  '
            f'{self.original_ref:<{widths["original_ref"]}}  '
            f'{self.aligned_to:<{widths["aligned_to"]}}'
        )


def read_miri_filter(miri_path: str) -> str:
    """
    Return ``FILTER`` from a MIRI primary (or SCI) header.

    Parameters
    ----------
    miri_path : str
        MIRI FITS file.

    Returns
    -------
    str
        Filter name, or ``'UNKNOWN'``.
    """
    with fits.open(miri_path) as hdul:
        filt = hdul[0].header.get('FILTER')
        if not filt and 'SCI' in hdul:
            filt = hdul['SCI'].header.get('FILTER')
    return str(filt) if filt else 'UNKNOWN'


def find_jhat_product(outdir: Path, miri_path: str) -> Path | None:
    """
    Locate the JHAT FITS product for a science frame.

    Parameters
    ----------
    outdir : pathlib.Path
        Directory holding JHAT products.
    miri_path : str
        Original science frame.

    Returns
    -------
    pathlib.Path or None
        The matching product, or the first ``*jhat*.fits`` found.
    """
    stem = (
        Path(miri_path)
        .name.replace('_cal.fits', '_jhat.fits')
        .replace('_i2d.fits', '_jhat.fits')
    )
    candidate = outdir / stem
    if candidate.is_file():
        return candidate
    matches = sorted(outdir.glob('*jhat*.fits'))
    return matches[0] if matches else None


def _normalize_align_mode(align_mode: str | None) -> str:
    """
    Map legacy ``NIRCAM`` labels to ``REFERENCE``; otherwise uppercase.

    Parameters
    ----------
    align_mode : str or None
        Raw mode label.

    Returns
    -------
    str
        Normalized mode.
    """
    mode = str(align_mode or 'NA').upper()
    if mode == 'NIRCAM':
        return 'REFERENCE'
    return mode


def _is_reference_quality_hold(row: AlignmentSummaryRow) -> bool:
    """
    Report whether a row is a REFERENCE solution held pending MIRI_REL.

    These rows keep finite metrics and JHAT paths so MIRI_REL can be tried.
    They are omitted from the live alignment summary until MIRI_REL finishes
    (kept if improved) or the REFERENCE solution is restored as SUCCESS.

    Parameters
    ----------
    row : AlignmentSummaryRow
        Row to test.

    Returns
    -------
    bool
        True for a pending REFERENCE quality hold.
    """
    return (
        row.status == 'PENDING'
        and _normalize_align_mode(row.align_mode) == 'REFERENCE'
        and isinstance(row.dispersion_mas, float)
        and bool(row.aligned_path)
        and str(row.aligned_path) != 'NA'
    )


def harvest_alignment_metrics(
    miri_path: str,
    outdir: Path,
    *,
    ran_ok: bool,
    default_align_mode: str = 'NA',
    default_original_ref: str = 'NA',
    default_aligned_to: str = 'NA',
    ref_overlap_frac: float | str = 'NA',
) -> AlignmentSummaryRow:
    """
    Build a summary row from alignment products and headers.

    SUCCESS requires a JHAT product with a finite final mean dispersion
    (``JWDISPM`` / ``GADISPM``, stored in arcsec, reported in mas). The method
    is recorded separately in ``align_mode``.

    Parameters
    ----------
    miri_path : str
        Science frame.
    outdir : pathlib.Path
        Directory holding JHAT products.
    ran_ok : bool
        Whether the alignment call completed without raising.
    default_align_mode, default_original_ref, default_aligned_to : str, optional
        Values used when the JHAT header lacks provenance keywords.
    ref_overlap_frac : float or str, optional
        Unique MIRI-ROI fraction covered by all overlapping references.

    Returns
    -------
    AlignmentSummaryRow
        Populated summary row.
    """
    filt = read_miri_filter(miri_path)
    empty = dict(
        miri_path=miri_path,
        filter=filt,
        status='FAILURE',
        n_calibrators='NA',
        dispersion_mas='NA',
        aligned_path='NA',
        align_mode='NA',
        original_ref='NA',
        aligned_to='NA',
        ref_overlap_frac=ref_overlap_frac,
    )
    if not ran_ok:
        return AlignmentSummaryRow(**empty)

    jhat = find_jhat_product(outdir, miri_path)
    if jhat is None:
        return AlignmentSummaryRow(**empty)
    aligned_path = str(jhat.resolve())

    with fits.open(jhat) as hdul:
        hdr = hdul[0].header
        if hdr.get('FILTER'):
            filt = str(hdr['FILTER'])
        disp_std = hdr.get('JWDISPS', hdr.get('GADISPS'))
        disp_mean = hdr.get('JWDISPM', hdr.get('GADISPM'))
        n_cal = hdr.get('JWNCAL', hdr.get('GANCAL'))
        align_mode = _normalize_align_mode(
            hdr.get('ALGNMODE', default_align_mode) or default_align_mode
        )
        original_ref = str(
            hdr.get('ALGNREF', default_original_ref) or default_original_ref
        )
        aligned_to = str(hdr.get('ALGNTO', default_aligned_to) or default_aligned_to)

    # The soft-failure path in align_jwst_image writes JWDISPS as 'NaN'.
    rejected = isinstance(disp_std, str) and disp_std.upper() == 'NAN'

    dispersion_mas: float | str = 'NA'
    try:
        if disp_mean is not None and not (
            isinstance(disp_mean, str) and str(disp_mean).upper() == 'NAN'
        ):
            disp_val = float(disp_mean)
            if math.isfinite(disp_val):
                dispersion_mas = disp_val * 1000.0
    except (TypeError, ValueError):
        dispersion_mas = 'NA'

    n_calibrators: int | str = 'NA'
    try:
        if n_cal is not None and not (
            isinstance(n_cal, str) and str(n_cal).upper() == 'NAN'
        ):
            n_calibrators = int(n_cal)
    except (TypeError, ValueError):
        n_calibrators = 'NA'

    if rejected or dispersion_mas == 'NA':
        return AlignmentSummaryRow(**empty)

    return AlignmentSummaryRow(
        miri_path=miri_path,
        filter=filt,
        status='SUCCESS',
        n_calibrators=n_calibrators,
        dispersion_mas=dispersion_mas,
        aligned_path=aligned_path,
        align_mode=align_mode,
        original_ref=original_ref,
        aligned_to=aligned_to,
        ref_overlap_frac=ref_overlap_frac,
    )


def write_alignment_summary(rows: list[AlignmentSummaryRow], outfile: Path) -> Path:
    """
    Write a plain ASCII alignment summary table.

    The file is written to a temporary name and atomically replaced so a live
    reader never sees a partial table.

    Parameters
    ----------
    rows : list of AlignmentSummaryRow
        Rows to write.
    outfile : pathlib.Path
        Destination path.

    Returns
    -------
    pathlib.Path
        The resolved destination path.
    """
    outfile = Path(outfile).expanduser().resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        'miri_path': 'miri_path',
        'filter': 'filter',
        'status': 'status',
        'ref_overlap_frac': 'ref_overlap_frac',
        'n_calibrators': 'n_calibrators',
        'dispersion_mas': 'dispersion_mas',
        'align_mode': 'align_mode',
        'aligned_path': 'aligned_path',
        'original_ref': 'original_ref',
        'aligned_to': 'aligned_to',
    }

    def disp_len(r: AlignmentSummaryRow) -> int:
        if isinstance(r.dispersion_mas, float):
            return len(f'{r.dispersion_mas:.3f}')
        return len(str(r.dispersion_mas))

    def ov_len(r: AlignmentSummaryRow) -> int:
        if isinstance(r.ref_overlap_frac, float):
            return len(f'{r.ref_overlap_frac:.4f}')
        return len(str(r.ref_overlap_frac))

    def width(key: str, lengths) -> int:
        return max([len(headers[key])] + list(lengths) + [1])

    widths = {
        'miri_path': width('miri_path', (len(r.miri_path) for r in rows)),
        'filter': width('filter', (len(r.filter) for r in rows)),
        'status': width('status', (len(r.status) for r in rows)),
        'ref_overlap_frac': width('ref_overlap_frac', (ov_len(r) for r in rows)),
        'n_calibrators': width(
            'n_calibrators', (len(str(r.n_calibrators)) for r in rows)
        ),
        'dispersion_mas': width('dispersion_mas', (disp_len(r) for r in rows)),
        'align_mode': width('align_mode', (len(r.align_mode) for r in rows)),
        'aligned_path': width('aligned_path', (len(r.aligned_path) for r in rows)),
        'original_ref': width('original_ref', (len(r.original_ref) for r in rows)),
        'aligned_to': width('aligned_to', (len(r.aligned_to) for r in rows)),
    }

    header = (
        f'{headers["miri_path"]:<{widths["miri_path"]}}  '
        f'{headers["filter"]:<{widths["filter"]}}  '
        f'{headers["status"]:<{widths["status"]}}  '
        f'{headers["ref_overlap_frac"]:>{widths["ref_overlap_frac"]}}  '
        f'{headers["n_calibrators"]:>{widths["n_calibrators"]}}  '
        f'{headers["dispersion_mas"]:>{widths["dispersion_mas"]}}  '
        f'{headers["align_mode"]:<{widths["align_mode"]}}  '
        f'{headers["aligned_path"]:<{widths["aligned_path"]}}  '
        f'{headers["original_ref"]:<{widths["original_ref"]}}  '
        f'{headers["aligned_to"]:<{widths["aligned_to"]}}'
    )
    lines = [header, '-' * len(header)]
    lines.extend(r.format_line(widths) for r in rows)
    lines.append('')
    tmp = outfile.with_name(outfile.name + '.tmp')
    tmp.write_text('\n'.join(lines))
    tmp.replace(outfile)
    return outfile


def _looks_like_filter(token: str) -> bool:
    """
    Report whether a path token looks like a filter name.

    Parameters
    ----------
    token : str
        Path segment such as ``F560W`` or ``F150W2``.

    Returns
    -------
    bool
        True for filter-shaped tokens.
    """
    return bool(re.fullmatch(r'F\d+[WMN]\d*', str(token).upper()))


# Path segments that are telescope/instrument roots, not filters.
_PATH_SKIP_TOKENS = frozenset(
    {
        'JWST',
        'HST',
        'ROMAN',
        'EUCLID',
        'MIRI',
        'NIRCAM',
        'NIRISS',
        'ACS',
        'WFC3',
        'WFPC2',
        'WFI',
        'VIS',
        'NISP',
        'MASTDOWNLOAD',
    }
)


def discover_miri_images(data_dir: Path) -> list[str]:
    """
    Find MIRI cal images under a dataset root.

    The preferred layout is
    ``<data-dir>/JWST/MIRI/<FILTER>/<obsid>/mastDownload/JWST/*_mirimage/*_cal.fits``;
    older ``<FILTER>/<obsid>/…`` and ``<FILTER>_<obsid>/…`` trees also match.

    Parameters
    ----------
    data_dir : pathlib.Path
        Dataset root.

    Returns
    -------
    list of str
        Sorted resolved paths.
    """
    data_dir = Path(data_dir)
    found: set[str] = set()
    for path in data_dir.glob('**/mastDownload/JWST/*_mirimage/*_cal.fits'):
        found.add(str(path.resolve()))
    return sorted(found)


def discover_ref_images(data_dir: Path) -> list[str]:
    """
    Find reference coadds under a dataset root.

    Searches ``<data-dir>/reference/group_*/ref_*/coadd*i2d.fits`` and the
    ``reduction/`` variant produced by ``mosaic --basedir <data-dir>/reduction``.

    Parameters
    ----------
    data_dir : pathlib.Path
        Dataset root.

    Returns
    -------
    list of str
        Sorted resolved paths.
    """
    root = Path(data_dir)
    patterns = (
        'reference/group_*/ref_*/coadd*i2d.fits',
        'reduction/reference/group_*/ref_*/coadd*i2d.fits',
    )
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            found[str(path.resolve())] = path
    return sorted(found)


def ensure_overlap_outdir(data_dir: Path, overlap_outdir: Path | None = None) -> Path:
    """
    Create and return the overlap summary directory.

    Parameters
    ----------
    data_dir : pathlib.Path
        Dataset root.
    overlap_outdir : pathlib.Path, optional
        Explicit output directory (default: ``<data-dir>/overlap``).

    Returns
    -------
    pathlib.Path
        The created directory.
    """
    out = (
        Path(overlap_outdir)
        if overlap_outdir is not None
        else Path(data_dir) / 'overlap'
    )
    out.mkdir(parents=True, exist_ok=True)
    return out


def parse_filters_arg(filters: str | None) -> list[str] | None:
    """
    Parse a comma-separated filter list.

    Parameters
    ----------
    filters : str or None
        Value such as ``'F560W,F770W'``.

    Returns
    -------
    list of str or None
        Uppercased filter names, or ``None`` when unrestricted.
    """
    if filters is None or not str(filters).strip():
        return None
    parsed = []
    for part in str(filters).split(','):
        name = part.strip().upper()
        if not name:
            continue
        parsed.append(name)
    return parsed or None


def filter_name_from_miri_path(miri_path: str) -> str | None:
    """
    Infer the MIRI filter from a MAST download path.

    Supports ``.../JWST/MIRI/<FILTER>/<obsid>/mastDownload/JWST/...``,
    ``.../<FILTER>/<obsid>/mastDownload/JWST/...``, and
    ``.../<FILTER>_<obsid>/mastDownload/JWST/...``.

    Parameters
    ----------
    miri_path : str
        MIRI cal file path.

    Returns
    -------
    str or None
        Filter name, or ``None`` when the path has no filter segment.
    """
    parts = Path(miri_path).parts
    for i, part in enumerate(parts):
        if part != 'mastDownload' or i < 1:
            continue
        # Walk upward from the directory containing mastDownload.
        for j in range(i - 1, -1, -1):
            tok = str(parts[j])
            if tok.isdigit():
                continue
            if tok.upper() in _PATH_SKIP_TOKENS:
                continue
            if _looks_like_filter(tok):
                return tok.upper()
            # Legacy: <FILTER>_<obsid>
            head = tok.split('_', 1)[0]
            if _looks_like_filter(head):
                return head.upper()
        break
    return None


def filter_miri_images(
    miri_images: list[str],
    filters: list[str] | None,
) -> list[str]:
    """
    Keep only images whose path filter is in a requested set.

    Parameters
    ----------
    miri_images : list of str
        Candidate images.
    filters : list of str or None
        Requested filters; ``None`` keeps everything.

    Returns
    -------
    list of str
        Selected images.
    """
    if not filters:
        return list(miri_images)
    wanted = {f.upper() for f in filters}
    selected = []
    for path in miri_images:
        name = filter_name_from_miri_path(path)
        if name is not None and name in wanted:
            selected.append(path)
    return selected


def find_frame_overlaps(
    miri_images: list[str],
    refs: list[str],
    *,
    MirIFootprint,
    BestOverlap,
    compute_overlap,
) -> list[FrameOverlaps]:
    """
    Compute the best and all-nonzero-overlap references for each frame.

    Parameters
    ----------
    miri_images : list of str
        Science frames.
    refs : list of str
        Reference coadds.
    MirIFootprint, BestOverlap, compute_overlap : callable
        Injected from :mod:`st123.mosaic.image_overlap`.

    Returns
    -------
    list of FrameOverlaps
        One entry per science frame.
    """
    from st123.mosaic.image_overlap import compute_cumulative_overlap_fraction

    results: list[FrameOverlaps] = []

    for image in miri_images:
        print(f'MIRI: {image}')
        miri = MirIFootprint.from_fits(image)
        print(f'  illuminated S_REGION: {miri.s_region.to_string()}')
        print(
            f'  WCS pixel solid angle: {miri.pixel_area_arcmin2:.8e} '
            f'arcmin^2 / pixel'
        )
        print(f'  illuminated area: {miri.area.format()}')

        overlapping = []
        best = None

        for ref in refs:
            try:
                result = compute_overlap(miri, ref)
            except Exception as exc:
                print(f'  FAILED for ref {ref}: {exc}')
                continue

            print(f'  ref: {ref}')
            print(f'    S_REGION: {result.ref_s_region.to_string()}')
            print(f'    ref area: {result.ref_area.format()}')
            print(f'    overlap area: {result.overlap_area.format()}')

            if result.overlap_area.pixels2 > 0.0:
                overlapping.append(result)

            if best is None or result.overlap_area.pixels2 > best.overlap_area.pixels2:
                best = BestOverlap(
                    science_path=image,
                    ref_path=result.ref_path,
                    overlap_area=result.overlap_area,
                )

        if best is None or best.overlap_area.pixels2 <= 0.0:
            best = BestOverlap(
                science_path=image,
                ref_path=None,
                overlap_area=miri.metrics(0.0),
            )

        overlapping.sort(key=lambda r: r.overlap_area.pixels2, reverse=True)

        union_frac = compute_cumulative_overlap_fraction(
            miri, [r.ref_path for r in overlapping]
        )

        print(
            f'Overlap maximized: MIRI image: {best.miri_path}, '
            f'Reference image: {best.ref_path}, '
            f'Max overlap area: {best.overlap_area.pixels2:.3f} pixels^2 '
            f'({best.overlap_area.arcmin2:.6f} arcmin^2, '
            f'{best.overlap_area.fraction_of_miri_roi:.4f} of MIRI illuminated ROI); '
            f'{len(overlapping)} reference(s) with any overlap; '
            f'union coverage {union_frac:.4f} of MIRI ROI'
        )
        if overlapping:
            print('  References with any overlap (largest first):')
            for result in overlapping:
                print(
                    f'    {result.ref_path}: '
                    f'{result.overlap_area.pixels2:.3f} pixels^2 '
                    f'({result.overlap_area.fraction_of_miri_roi:.4f} of MIRI ROI)'
                )
        print()

        results.append(
            FrameOverlaps(
                miri_path=image,
                best=best,
                overlapping=overlapping,
                union_overlap_fraction=union_frac,
            )
        )

    return results


def write_overlap_summaries(
    frames: list[FrameOverlaps],
    outdir: Path,
) -> tuple[Path, Path]:
    """
    Write text and JSON summaries of best and any-overlap references.

    Parameters
    ----------
    frames : list of FrameOverlaps
        Overlap results.
    outdir : pathlib.Path
        Destination directory.

    Returns
    -------
    tuple
        ``(txt_path, json_path)``.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    txt_path = outdir / 'overlap_summary.txt'
    json_path = outdir / 'overlap_summary.json'

    lines: list[str] = []
    for frame in frames:
        best = frame.best
        lines.append(
            f'Overlap maximized: MIRI image: {best.miri_path}, '
            f'Reference image: {best.ref_path}, '
            f'Max overlap area: {best.overlap_area.pixels2:.3f} pixels^2 '
            f'({best.overlap_area.arcmin2:.6f} arcmin^2, '
            f'{best.overlap_area.fraction_of_miri_roi:.4f} of MIRI illuminated ROI); '
            f'{len(frame.overlapping)} reference(s) with any overlap; '
            f'union coverage {frame.union_overlap_fraction:.4f} of MIRI ROI\n'
        )
        for result in frame.overlapping:
            lines.append(
                f'  any-overlap: {result.ref_path} '
                f'{result.overlap_area.pixels2:.3f} pixels^2 '
                f'({result.overlap_area.fraction_of_miri_roi:.4f} of MIRI ROI)\n'
            )
        lines.append('\n')

    txt_path.write_text(''.join(lines))
    payload = {
        'n_miri': len(frames),
        'n_with_overlap': sum(1 for f in frames if f.best.ref_path is not None),
        'frames': [f.to_dict() for f in frames],
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f'Wrote overlap summary: {txt_path}')
    print(f'Wrote overlap JSON:    {json_path}')
    return txt_path, json_path


def alignment_outdir_for(miri_path: str) -> Path:
    """
    Return the per-frame alignment output directory.

    Parameters
    ----------
    miri_path : str
        Science frame.

    Returns
    -------
    pathlib.Path
        ``<directory of the cal file>/alignment_output``.
    """
    return Path(miri_path).resolve().parent / 'alignment_output'


def resolve_data_dir(
    *,
    data_dir: Path | None,
    data_root: Path | None,
    galaxy: str | None,
    default_data_dir: Path,
) -> tuple[Path, str]:
    """
    Resolve the dataset root and its label.

    Preference order is ``data_dir``, then ``data_root`` / ``galaxy`` (legacy),
    then ``default_data_dir``.

    Parameters
    ----------
    data_dir : pathlib.Path or None
        Explicit dataset root.
    data_root : pathlib.Path or None
        Parent directory of a dataset subdirectory.
    galaxy : str or None
        Dataset subdirectory / label.
    default_data_dir : pathlib.Path
        Fallback root.

    Returns
    -------
    tuple
        ``(data_dir, dataset_label)``.
    """
    if data_dir is not None:
        root = Path(data_dir).expanduser().resolve()
    elif data_root is not None:
        label = galaxy or 'M51'
        root = (Path(data_root).expanduser().resolve() / label).resolve()
    else:
        root = Path(default_data_dir).expanduser().resolve()

    label = galaxy or root.name
    return root, label


def run_legacy_overlap_file_pipeline(
    overlap_file: Path,
    *,
    outdir: Path = Path('alignment_output_auto'),
    success_file: Path = Path('successful_alignments.txt'),
    fail_file: Path = Path('failed_alignments.txt'),
    filters: tuple[str, ...] = ('F560W', 'F770W'),
    alignment_script: str = 'alignment_dispersion.py',
    plot: bool = True,
    verbose: bool = True,
) -> int:
    """
    Align each MIRI/reference pair listed in a legacy overlap text file.

    Parameters
    ----------
    overlap_file : pathlib.Path
        Text file with ``Overlap maximized:`` lines.
    outdir : pathlib.Path, optional
        Output root for per-pair directories.
    success_file, fail_file : pathlib.Path, optional
        Destinations for the pair result logs.
    filters : tuple of str, optional
        Only process lines mentioning one of these filters.
    alignment_script : str, optional
        Script invoked per pair.
    plot, verbose : bool, optional
        Flags forwarded to the alignment script.

    Returns
    -------
    int
        ``0`` when every pair succeeded, else ``1``.
    """
    overlap_file = Path(overlap_file).expanduser().resolve()
    if not overlap_file.is_file():
        print(f'ERROR: legacy overlap file not found: {overlap_file}', file=sys.stderr)
        return 1

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    success_file = Path(success_file)
    fail_file = Path(fail_file)
    filt_tokens = tuple(f.upper() for f in filters)

    n_ok = 0
    n_fail = 0
    with open(success_file, 'w') as success, open(fail_file, 'w') as failed:
        with open(overlap_file) as file:
            for line in file:
                if 'Overlap maximized' not in line:
                    continue
                if not any(tok in line for tok in filt_tokens):
                    continue

                align_image = line.split('MIRI image: ')[1].split(
                    ', Reference image: '
                )[0]
                ref_image = line.split(', Reference image: ')[1].split(
                    ', Max overlap area'
                )[0]

                align_name = Path(align_image).stem
                ref_name = Path(ref_image).stem
                pair_outdir = os.path.join(
                    str(outdir), f'{align_name}_aligned_to_{ref_name}'
                )

                command = [
                    sys.executable,
                    alignment_script,
                    '--ref',
                    ref_image,
                    '--image',
                    align_image,
                    '--outdir',
                    pair_outdir,
                ]
                if plot:
                    command.append('--plot')
                if verbose:
                    command.append('--verbose')

                result = subprocess.run(command)
                if result.returncode == 0:
                    success.write(
                        f'MIRI image: {align_image}, \n'
                        f'Reference image: {ref_image}\n\n'
                    )
                    success.flush()
                    n_ok += 1
                else:
                    failed.write(
                        f'MIRI image: {align_image}, \n'
                        f'Reference image: {ref_image}\n\n'
                    )
                    failed.flush()
                    n_fail += 1

    print('Done')
    print(f'Successful pairs written to {success_file} ({n_ok})')
    print(f'Failed pairs written to {fail_file} ({n_fail})')
    return 1 if n_fail else 0


def _frame_ref_images(frame: FrameOverlaps | dict) -> tuple[str, list[str], str | None]:
    """
    Return the science path and ordered reference images for one frame.

    Parameters
    ----------
    frame : FrameOverlaps or dict
        Overlap record.

    Returns
    -------
    tuple
        ``(miri_path, ordered_ref_images, best_ref)``.
    """
    if isinstance(frame, dict):
        miri_path = frame['miri_path']
        best_ref = frame['best']['ref_path']
        overlapping = [r['ref_path'] for r in frame.get('overlapping', [])]
    else:
        miri_path = frame.miri_path
        best_ref = frame.best.ref_path
        overlapping = [r.ref_path for r in frame.overlapping]

    ref_images = overlapping or ([best_ref] if best_ref else [])
    ordered: list[str] = []
    for path in ([best_ref] if best_ref else []) + list(ref_images):
        if path and path not in ordered:
            ordered.append(path)
    return miri_path, ordered, best_ref


def _frame_ref_overlap_frac(frame: FrameOverlaps | dict) -> float:
    """
    Return the unique MIRI-ROI fraction covered by all overlapping references.

    Uses a cached ``union_overlap_fraction`` when present; otherwise recomputes
    from the overlapping reference paths.

    Parameters
    ----------
    frame : FrameOverlaps or dict
        Overlap record.

    Returns
    -------
    float
        Union coverage fraction.
    """
    if isinstance(frame, dict):
        cached = frame.get('union_overlap_fraction')
        if cached is not None:
            try:
                return float(cached)
            except (TypeError, ValueError):
                pass
        miri_path, ref_images, _best = _frame_ref_images(frame)
    else:
        try:
            return float(frame.union_overlap_fraction)
        except (TypeError, ValueError, AttributeError):
            pass
        miri_path, ref_images, _best = _frame_ref_images(frame)

    if not ref_images:
        return 0.0
    from st123.mosaic.image_overlap import (
        MirIFootprint,
        compute_cumulative_overlap_fraction,
    )

    return compute_cumulative_overlap_fraction(
        MirIFootprint.from_fits(miri_path), ref_images
    )


def frame_has_nircam_overlap(
    frame: FrameOverlaps | dict,
    *,
    min_ref_overlap_frac: float = 0.02,
) -> bool:
    """
    Report whether cumulative reference coverage meets the threshold.

    Parameters
    ----------
    frame : FrameOverlaps or dict
        Overlap record.
    min_ref_overlap_frac : float, optional
        Minimum unique fraction of the science footprint that references must
        cover.

    Returns
    -------
    bool
        True when the frame should be aligned.
    """
    _miri_path, ref_images, best_ref = _frame_ref_images(frame)
    if not ref_images or best_ref is None:
        return False
    return _frame_ref_overlap_frac(frame) >= float(min_ref_overlap_frac)


def reject_zero_nircam_overlap_frames(
    frames: list[FrameOverlaps] | list[dict],
    *,
    min_ref_overlap_frac: float = 0.02,
) -> tuple[list[FrameOverlaps] | list[dict], int]:
    """
    Drop frames with insufficient cumulative reference footprint overlap.

    Rejected frames remain in the overlap summaries only — they are not written
    to ``alignment_summary.txt`` and never reach JHAT, including MIRI→MIRI
    fallback.

    Parameters
    ----------
    frames : list
        Overlap records.
    min_ref_overlap_frac : float, optional
        Minimum union coverage fraction.

    Returns
    -------
    tuple
        ``(kept_frames, n_rejected)``.
    """
    min_frac = float(min_ref_overlap_frac)
    kept: list[FrameOverlaps | dict] = []
    n_rejected = 0

    for frame in frames:
        miri_path, _ref_images, best_ref = _frame_ref_images(frame)
        filt = filter_name_from_miri_path(miri_path) or read_miri_filter(miri_path)
        ov_frac = _frame_ref_overlap_frac(frame)
        if best_ref and ov_frac >= min_frac:
            kept.append(frame)
            continue

        n_rejected += 1
        print(
            f'REJECT {Path(miri_path).name}  {filt}  '
            f'ref_overlap_frac={ov_frac:.4f} < {min_frac:.4f} '
            f'(excluded from alignment)',
            flush=True,
        )

    if n_rejected:
        print(
            f'Rejected {n_rejected} MIRI frame(s) with '
            f'ref_overlap_frac < {min_frac:.4f}; '
            f'{len(kept)} frame(s) remain for alignment',
            flush=True,
        )
    return kept, n_rejected


def _group_frames_by_filter(
    frames: list[FrameOverlaps] | list[dict],
) -> OrderedDict[str, list[FrameOverlaps | dict]]:
    """
    Group frames by filter, preserving blue→red order of first appearance.

    Parameters
    ----------
    frames : list
        Overlap records.

    Returns
    -------
    collections.OrderedDict
        Filter name mapped to its frames.
    """
    ordered = sort_frames_blue_to_red(
        frames, filter_from_path=filter_name_from_miri_path
    )
    groups: OrderedDict[str, list[FrameOverlaps | dict]] = OrderedDict()
    for frame in ordered:
        miri_path, _, _ = _frame_ref_images(frame)
        filt = filter_name_from_miri_path(miri_path) or read_miri_filter(miri_path)
        groups.setdefault(filt, []).append(frame)
    return groups


def _format_worker_done(result) -> str:
    """
    Render a one-line DONE status for a worker result.

    Parameters
    ----------
    result : AlignWorkerResult
        Finished worker result.

    Returns
    -------
    str
        Status line for the parent process log.
    """
    base = Path(result.miri_path).name
    filt = result.row.get('filter') or getattr(result, 'filter', None) or 'UNKNOWN'
    status = str(result.row.get('status', 'FAILURE'))
    mode = _normalize_align_mode(result.row.get('align_mode'))
    disp = result.row.get('dispersion_mas', 'NA')
    disp_s = f'{disp:.3f}' if isinstance(disp, float) else str(disp)
    if status == 'SUCCESS':
        return (
            f'DONE  {base}  {filt}  SUCCESS  align_mode={mode}  '
            f'dispersion_mas={disp_s}'
        )
    if status == 'PENDING':
        return (
            f'DONE  {base}  {filt}  PENDING  align_mode={mode}  '
            f'dispersion_mas={disp_s} (over threshold; try MIRI_REL)'
        )
    if status in ('SKIP', 'REJECTED'):
        return f'DONE  {base}  {filt}  {status}'
    return f'DONE  {base}  {filt}  FAILURE'


def _needs_miri_fallback(row: AlignmentSummaryRow) -> bool:
    """
    Report whether a frame should enter MIRI_REL after the REFERENCE pass.

    Parameters
    ----------
    row : AlignmentSummaryRow
        Current row for the frame.

    Returns
    -------
    bool
        True when fallback should be attempted.
    """
    return row.status not in ('SUCCESS', 'REJECTED', 'SKIP')


# ---------------------------------------------------------------------------
# 6. Parallel workers
# ---------------------------------------------------------------------------
#
# These must stay module-level so ``ProcessPoolExecutor`` can pickle them for
# spawn children. JHAT / pipeline chatter is silenced with ``suppress_output``;
# the parent process only prints START / DONE lines.

_STACK_READY = False


@dataclass
class AlignWorkerResult:
    """Picklable result returned by a worker process."""

    miri_path: str
    filter: str
    mode: str  # 'nircam' | 'fallback' | 'skip' | 'visit'
    ok: bool
    row: dict[str, Any]
    success: dict[str, Any] | None = None
    error: str | None = None
    message: str = ''


def _bootstrap(repo: str) -> None:
    """
    Ensure the installable package root is importable in spawn workers.

    Parameters
    ----------
    repo : str
        Repo root path.
    """
    root = str(Path(repo).expanduser().resolve())
    # Editable installs already expose st123; still allow repo-root on sys.path.
    if root not in sys.path:
        sys.path.insert(0, root)


def _sanitize_logging() -> None:
    """
    Neutralize stpipe / jwst log handlers that raise on pysiaf warnings.

    ``pysiaf`` emits a multi-argument ``logger.warning(...)`` at import time.
    ``stpipe.log.LogHandler`` then raises during ``emit``, which can abort
    worker imports under spawn. Replace those handlers with NullHandlers.
    """
    logging.raiseExceptions = False

    def quiet(logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.ERROR)

    quiet(logging.getLogger())
    for name in (
        'stpipe',
        'stpipe.pipeline',
        'jwst',
        'jwst.associations',
        'pysiaf',
        'CRDS',
    ):
        quiet(logging.getLogger(name))


def _preload_science_stack() -> None:
    """Import the jwst / jhat / pysiaf stack once per process, quietly."""
    global _STACK_READY
    if _STACK_READY:
        return

    _sanitize_logging()
    last_exc: Exception | None = None
    for _attempt in range(2):
        try:
            with suppress_output():
                _sanitize_logging()
                import st123  # noqa: F401
                import st123.alignment.align  # noqa: F401
            _STACK_READY = True
            return
        except Exception as exc:  # pragma: no cover - environment-dependent
            last_exc = exc
            _sanitize_logging()
    raise RuntimeError(f'Failed to preload science stack in worker: {last_exc}')


def worker_initializer(repo: str) -> None:
    """
    Prepare a spawn worker: path, timeouts, quiet logs, warm imports.

    Called once per worker process before any alignment jobs run.

    Parameters
    ----------
    repo : str
        Repo root to make importable.
    """
    _bootstrap(repo)
    if socket.getdefaulttimeout() is None:
        socket.setdefaulttimeout(15)
    # Limit BLAS/OpenMP oversubscription when many workers run together.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    _sanitize_logging()
    _preload_science_stack()


def _ensure_worker_ready(repo: str) -> None:
    """
    Run the idempotent worker bootstrap at the start of each job.

    Parameters
    ----------
    repo : str
        Repo root to make importable.
    """
    _bootstrap(repo)
    if socket.getdefaulttimeout() is None:
        socket.setdefaulttimeout(15)
    if not _STACK_READY:
        _sanitize_logging()
        _preload_science_stack()


def _run_alignment_kwargs(job: dict[str, Any], filt: str) -> dict[str, Any]:
    """
    Build :func:`run_alignment` kwargs with filter-specific calibrator settings.

    F770W overrides the CLI ``nbright`` / refine knobs unless
    ``job['use_filter_calibrators']`` is False.

    Parameters
    ----------
    job : dict
        Worker job description.
    filt : str
        Filter name.

    Returns
    -------
    dict
        Keyword arguments for :func:`run_alignment`.
    """
    kwargs: dict[str, Any] = {
        'nbright': job['nbright'],
        'plot': job['plot'],
        'verbose': False,
        'refine': job['refine'],
        'refine_sigma': job['refine_sigma'],
        'refine_max_iter': job['refine_max_iter'],
    }
    if not job.get('use_filter_calibrators', True):
        return kwargs

    settings = calibrator_settings_for_filter(filt)
    if settings is DEFAULT_CALIBRATOR_SETTINGS:
        return kwargs

    kwargs.update(settings.as_run_kwargs())
    # Preserve plot from the job; the refine flag stays under CLI control.
    kwargs['plot'] = job['plot']
    kwargs['refine'] = job['refine']
    kwargs['verbose'] = False
    print(
        f'  {filt} calibrator settings: {describe_calibrator_settings(settings)}',
        flush=True,
    )
    return kwargs


def run_visit_align_job(job: dict[str, Any]) -> AlignWorkerResult:
    """
    Align one visit-mode frame with JHAT (stdout suppressed).

    Parameters
    ----------
    job : dict
        Picklable job with ``align_image``, ``outdir``, and JHAT knobs.
        ``miri_path`` is the START/DONE label (same path as ``align_image``).

    Returns
    -------
    AlignWorkerResult
        Outcome for the parent process summary line.
    """
    _ensure_worker_ready(job.get('repo', ''))
    image = job['align_image']
    filt = str(job.get('filter', ''))
    try:
        # Quiet JHAT / pipeline chatter; parent only sees START / DONE.
        with suppress_output():
            align_jwst_image(
                align_image=image,
                outdir=job['outdir'],
                gaia=bool(job.get('gaia', False)),
                photfilename=job.get('photfilename'),
                xshift=float(job.get('xshift', 0.0)),
                yshift=float(job.get('yshift', 0.0)),
                Nbright=int(job.get('Nbright', 800)),
                sig=float(job.get('sig', 2)),
                verbose=False,
                plot=False,
            )
        row = AlignmentSummaryRow(
            miri_path=image,
            filter=filt,
            status='SUCCESS',
            n_calibrators='NA',
            dispersion_mas='NA',
            aligned_path=str(
                Path(job['outdir'])
                / Path(image).name.replace('cal.fits', 'jhat.fits')
            ),
        )
        return AlignWorkerResult(
            miri_path=image,
            filter=filt,
            mode='visit',
            ok=True,
            row=asdict(row),
            message='ok',
        )
    except Exception as exc:
        err = str(exc)
        if job.get('verbose'):
            err = f'{err}\n{traceback.format_exc()}'
        row = AlignmentSummaryRow(
            miri_path=image,
            filter=filt,
            status='FAILURE',
            n_calibrators='NA',
            dispersion_mas='NA',
            aligned_path='NA',
        )
        return AlignWorkerResult(
            miri_path=image,
            filter=filt,
            mode='visit',
            ok=False,
            row=asdict(row),
            error=err,
            message=f'Worker crashed: {exc}',
        )


def run_nircam_align_job(job: dict[str, Any]) -> AlignWorkerResult:
    """
    Align one science frame to its overlapping reference images.

    Parameters
    ----------
    job : dict
        Picklable job description built by :func:`align_from_frames`.

    Returns
    -------
    AlignWorkerResult
        Outcome, including the summary row and any success record.
    """
    _ensure_worker_ready(job['repo'])

    miri_path = job['miri_path']
    filt = job['filter']
    ref_images = list(job['ref_images'])
    best_ref = job.get('best_ref')
    outdir = Path(job['outdir'])
    ref_overlap_frac = job.get('ref_overlap_frac', 'NA')

    def fail(msg: str, exc: Exception | None = None) -> AlignWorkerResult:
        err = msg if exc is None else f'{msg}: {exc}'
        if job.get('verbose') and exc is not None:
            err = f'{err}\n{traceback.format_exc()}'
        row = harvest_alignment_metrics(
            miri_path,
            outdir,
            ran_ok=False,
            ref_overlap_frac=ref_overlap_frac,
        )
        row.filter = filt
        return AlignWorkerResult(
            miri_path=miri_path,
            filter=filt,
            mode='nircam',
            ok=False,
            row=asdict(row),
            error=err,
        )

    if not ref_images:
        row = AlignmentSummaryRow(
            miri_path=miri_path,
            filter=filt,
            status='SKIP',
            n_calibrators='NA',
            dispersion_mas='NA',
            aligned_path='NA',
            ref_overlap_frac=ref_overlap_frac,
        )
        return AlignWorkerResult(
            miri_path=miri_path,
            filter=filt,
            mode='skip',
            ok=False,
            row=asdict(row),
        )

    try:
        align_kw = _run_alignment_kwargs(job, filt)
        with suppress_output():
            run_alignment(
                ref_images=ref_images,
                align_image=miri_path,
                outdir=str(outdir),
                cache_dir=job.get('cache_dir'),
                match_radius_arcsec=job['match_radius_arcsec'],
                clip_to_align_footprint=job['clip_to_align_footprint'],
                **align_kw,
            )
    except Exception as exc:
        return fail('NIRCam alignment failed', exc)

    original_ref = best_ref or ref_images[0]
    aligned_to = original_ref
    row = harvest_alignment_metrics(
        miri_path,
        outdir,
        ran_ok=True,
        default_align_mode='REFERENCE',
        default_original_ref=original_ref,
        default_aligned_to=aligned_to,
        ref_overlap_frac=ref_overlap_frac,
    )
    if row.status != 'SUCCESS' or not isinstance(row.dispersion_mas, float):
        return AlignWorkerResult(
            miri_path=miri_path,
            filter=filt,
            mode='nircam',
            ok=False,
            row=asdict(row),
            error='REFERENCE alignment soft-failed',
        )

    write_alignment_provenance(
        row.aligned_path,
        align_mode='REFERENCE',
        original_ref=original_ref,
        aligned_to=aligned_to,
        relative_dispersion_mas=row.dispersion_mas,
        absolute_dispersion_mas=row.dispersion_mas,
        n_calibrators=(row.n_calibrators if isinstance(row.n_calibrators, int) else None),
    )
    row = harvest_alignment_metrics(
        miri_path,
        outdir,
        ran_ok=True,
        default_align_mode='REFERENCE',
        default_original_ref=original_ref,
        default_aligned_to=aligned_to,
        ref_overlap_frac=ref_overlap_frac,
    )
    if not row.filter or row.filter == 'UNKNOWN':
        row.filter = filt or read_miri_filter(miri_path)

    # ``None`` in the job means use the per-filter map; a positive float is a
    # uniform CLI override; <=0 disables the quality hold.
    max_disp = job.get('max_nircam_dispersion_mas')
    if max_disp is None:
        max_disp = max_reference_dispersion_mas(filt)
    elif float(max_disp) <= 0:
        max_disp = None
    if (
        max_disp is not None
        and float(max_disp) > 0
        and float(row.dispersion_mas) > float(max_disp)
    ):
        # Keep REFERENCE products on disk, but mark PENDING (not FAILURE) so
        # the parent can try MIRI_REL before recording a final status.
        # PENDING rows are omitted from the live alignment summary.
        row.status = 'PENDING'
        return AlignWorkerResult(
            miri_path=miri_path,
            filter=row.filter,
            mode='nircam',
            ok=False,
            row=asdict(row),
            error=(
                f'REFERENCE dispersion {float(row.dispersion_mas):.3f} mas '
                f'exceeds quality threshold {float(max_disp):.3f} mas; '
                f'trying MIRI_REL'
            ),
        )

    success = SuccessfulAlignment(
        miri_path=miri_path,
        jhat_path=row.aligned_path,
        filter=row.filter,
        wavelength_um=filter_wavelength_um(row.filter),
        dispersion_mas=float(row.dispersion_mas),
        relative_dispersion_mas=float(row.dispersion_mas),
        align_mode='REFERENCE',
        original_ref=original_ref,
        aligned_to=aligned_to,
        photfile=find_aligned_photfile(row.aligned_path),
    )
    return AlignWorkerResult(
        miri_path=miri_path,
        filter=row.filter,
        mode='nircam',
        ok=True,
        row=asdict(row),
        success=asdict(success),
    )


def _backup_alignment_products(outdir: Path, miri_path: str) -> list[tuple[Path, Path]]:
    """
    Copy existing JHAT / phot products aside so a failed MIRI_REL can restore them.

    Parameters
    ----------
    outdir : pathlib.Path
        Directory holding the products.
    miri_path : str
        Science frame.

    Returns
    -------
    list of (pathlib.Path, pathlib.Path)
        ``(source, backup)`` pairs.
    """
    backups: list[tuple[Path, Path]] = []
    jhat = find_jhat_product(outdir, miri_path)
    candidates: list[Path] = []
    if jhat is not None:
        candidates.append(jhat)
        stem = jhat.name.replace('_jhat.fits', '')
        for pattern in (f'{stem}*phot*.fits', f'{stem}*phot*.ecsv', f'{stem}*.reg'):
            candidates.extend(outdir.glob(pattern))
    seen: set[Path] = set()
    for src in candidates:
        src = src.resolve()
        if src in seen or not src.is_file():
            continue
        seen.add(src)
        bak = src.with_name(src.name + '.nircam_bak')
        shutil.copy2(src, bak)
        backups.append((src, bak))
    return backups


def _restore_alignment_products(backups: list[tuple[Path, Path]]) -> None:
    """
    Restore products from their backups.

    Parameters
    ----------
    backups : list of (pathlib.Path, pathlib.Path)
        ``(source, backup)`` pairs from :func:`_backup_alignment_products`.
    """
    for src, bak in backups:
        if bak.is_file():
            shutil.copy2(bak, src)
            bak.unlink(missing_ok=True)


def _cleanup_alignment_backups(backups: list[tuple[Path, Path]]) -> None:
    """
    Delete product backups.

    Parameters
    ----------
    backups : list of (pathlib.Path, pathlib.Path)
        ``(source, backup)`` pairs from :func:`_backup_alignment_products`.
    """
    for _, bak in backups:
        bak.unlink(missing_ok=True)


def run_fallback_align_job(job: dict[str, Any]) -> AlignWorkerResult:
    """
    Relative-align one frame, trying ranked parent JHAT products in order.

    Parameters
    ----------
    job : dict
        Picklable job description with a ``parents`` list of serialized
        :class:`SuccessfulAlignment` records.

    Returns
    -------
    AlignWorkerResult
        Outcome; the first parent that improves on REFERENCE wins.
    """
    _ensure_worker_ready(job['repo'])

    miri_path = job['miri_path']
    filt = job['filter']
    outdir = Path(job['outdir'])
    backups = _backup_alignment_products(outdir, miri_path)
    ref_overlap_frac = job.get('ref_overlap_frac', 'NA')

    parent_dicts = list(job.get('parents') or [])
    if not parent_dicts and job.get('parent') is not None:
        parent_dicts = [job['parent']]

    def fail(msg: str, exc: Exception | None = None) -> AlignWorkerResult:
        _restore_alignment_products(backups)
        row = harvest_alignment_metrics(
            miri_path,
            outdir,
            ran_ok=False,
            ref_overlap_frac=ref_overlap_frac,
        )
        row.filter = filt
        err = msg if exc is None else f'{msg}: {exc}'
        if job.get('verbose') and exc is not None:
            err = f'{err}\n{traceback.format_exc()}'
        return AlignWorkerResult(
            miri_path=miri_path,
            filter=filt,
            mode='fallback',
            ok=False,
            row=asdict(row),
            error=err,
        )

    if not parent_dicts:
        return fail('no MIRI_REL parents provided')

    ref_disp_f = None
    ref_disp = job.get('reference_dispersion_mas')
    if ref_disp is not None:
        try:
            ref_disp_f = float(ref_disp)
        except (TypeError, ValueError):
            ref_disp_f = None

    align_kw = _run_alignment_kwargs(job, filt)
    errors: list[str] = []

    for parent_dict in parent_dicts:
        parent = SuccessfulAlignment(**parent_dict)
        # Always restart from the backed-up REFERENCE / prior products.
        _restore_alignment_products(backups)
        backups = _backup_alignment_products(outdir, miri_path)
        try:
            parent_phot = parent.photfile or find_aligned_photfile(parent.jhat_path)
            common = dict(
                align_image=miri_path,
                outdir=str(outdir),
                **align_kw,
            )
            with suppress_output():
                if parent_phot is not None:
                    run_alignment(
                        photfile=parent_phot,
                        ref_image=parent.jhat_path,
                        **common,
                    )
                else:
                    run_alignment(
                        ref_image=parent.jhat_path,
                        **common,
                    )
        except Exception as exc:
            errors.append(f'{Path(parent.miri_path).name}: align failed ({exc})')
            continue

        row = harvest_alignment_metrics(
            miri_path,
            outdir,
            ran_ok=True,
            default_align_mode='MIRI_REL',
            default_original_ref=parent.original_ref,
            default_aligned_to=parent.jhat_path,
            ref_overlap_frac=ref_overlap_frac,
        )
        if (
            row.status != 'SUCCESS'
            or not isinstance(row.dispersion_mas, float)
            or float(row.dispersion_mas) > 500.0
        ):
            errors.append(
                f'{Path(parent.miri_path).name}: soft-failed '
                f'(disp={row.dispersion_mas})'
            )
            continue

        rel_mas = float(row.dispersion_mas)
        abs_mas = combine_dispersion_mas(parent.dispersion_mas, rel_mas)

        # Quality-hold: keep MIRI_REL only when it improves absolute dispersion.
        if ref_disp_f is not None and abs_mas >= ref_disp_f:
            errors.append(
                f'{Path(parent.miri_path).name}: abs {abs_mas:.3f} mas '
                f'not better than REFERENCE {ref_disp_f:.3f} mas'
            )
            continue

        write_alignment_provenance(
            row.aligned_path,
            align_mode='MIRI_REL',
            original_ref=parent.original_ref,
            aligned_to=parent.jhat_path,
            relative_dispersion_mas=rel_mas,
            absolute_dispersion_mas=abs_mas,
            n_calibrators=(
                row.n_calibrators if isinstance(row.n_calibrators, int) else None
            ),
        )
        row = harvest_alignment_metrics(
            miri_path,
            outdir,
            ran_ok=True,
            default_align_mode='MIRI_REL',
            default_original_ref=parent.original_ref,
            default_aligned_to=parent.jhat_path,
            ref_overlap_frac=ref_overlap_frac,
        )
        success = SuccessfulAlignment(
            miri_path=miri_path,
            jhat_path=row.aligned_path,
            filter=row.filter,
            wavelength_um=filter_wavelength_um(row.filter),
            dispersion_mas=float(row.dispersion_mas),
            relative_dispersion_mas=rel_mas,
            align_mode='MIRI_REL',
            original_ref=parent.original_ref,
            aligned_to=parent.jhat_path,
            photfile=find_aligned_photfile(row.aligned_path),
        )
        _cleanup_alignment_backups(backups)
        return AlignWorkerResult(
            miri_path=miri_path,
            filter=row.filter,
            mode='fallback',
            ok=True,
            row=asdict(row),
            success=asdict(success),
            message=f'MIRI_REL via {Path(parent.miri_path).name}',
        )

    detail = '; '.join(errors[:5]) if errors else 'no viable parent'
    return fail(f'MIRI_REL failed for all candidate parents ({detail})')


# ---------------------------------------------------------------------------
# 7. Orchestration
# ---------------------------------------------------------------------------


def _run_jobs_parallel(
    jobs: list[dict],
    worker,
    *,
    workers: int,
    label: str,
    on_result=None,
) -> list:
    """
    Run picklable worker jobs across processes (or serially).

    Emits a brief ``START`` line when each job begins and invokes ``on_result``
    as each job finishes, so the caller can log ``DONE`` and update the summary
    without interleaving JHAT chatter.

    Parameters
    ----------
    jobs : list of dict
        Job descriptions.
    worker : callable
        Module-level worker function.
    workers : int
        Maximum worker processes; ``1`` runs serially in this process.
    label : str
        Prefix used in the plan line.
    on_result : callable, optional
        Called with each finished :class:`AlignWorkerResult`.

    Returns
    -------
    list of AlignWorkerResult
        Results in submission order.
    """
    import multiprocessing as mp

    if not jobs:
        return []

    n_workers = max(1, int(workers))
    print(
        f'{label}: {len(jobs)} job(s), workers={min(n_workers, len(jobs))}',
        flush=True,
    )

    def start_line(job: dict) -> None:
        print(f'START {Path(job["miri_path"]).name}', flush=True)

    def handle(result) -> None:
        if on_result is not None:
            on_result(result)

    if n_workers == 1 or len(jobs) == 1:
        serial_results = []
        for job in jobs:
            start_line(job)
            result = worker(job)
            serial_results.append(result)
            handle(result)
        return serial_results

    results: list[AlignWorkerResult | None] = [None] * len(jobs)
    # spawn avoids fork+OpenMP/BLAS deadlocks after heavy scientific imports.
    repo = str(jobs[0].get('repo') or '')
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(
        max_workers=min(n_workers, len(jobs)),
        mp_context=ctx,
        initializer=worker_initializer,
        initargs=(repo,),
    ) as pool:
        future_map = {}
        for i, job in enumerate(jobs):
            start_line(job)
            future_map[pool.submit(worker, job)] = i
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                job = jobs[idx]
                results[idx] = AlignWorkerResult(
                    miri_path=job['miri_path'],
                    filter=job['filter'],
                    mode=job.get('mode', label),
                    ok=False,
                    row=asdict(
                        AlignmentSummaryRow(
                            miri_path=job['miri_path'],
                            filter=job['filter'],
                            status='FAILURE',
                            n_calibrators='NA',
                            dispersion_mas='NA',
                            aligned_path='NA',
                            ref_overlap_frac=job.get('ref_overlap_frac', 'NA'),
                        )
                    ),
                    error=str(exc),
                    message=f'Worker crashed: {exc}',
                )
            handle(results[idx])
    return [r for r in results if r is not None]


def align_from_frames(
    frames: list[FrameOverlaps] | list[dict],
    *,
    run_alignment=None,
    nbright: int,
    plot: bool,
    verbose: bool,
    cache_dir: Path | None = None,
    match_radius_arcsec: float = 0.1,
    clip_to_align_footprint: bool = True,
    refine: bool = True,
    refine_sigma: float = 2.0,
    refine_max_iter: int = 5,
    use_filter_calibrators: bool = True,
    fallback: bool = True,
    max_nircam_dispersion_mas: float | None = None,
    min_ref_overlap_frac: float = 0.02,
    summary_outfile: Path | None = None,
    workers: int = 1,
    repo: Path | None = None,
) -> tuple[int, list[AlignmentSummaryRow]]:
    """
    Align frames filter-by-filter (blue→red), in parallel within each filter.

    For each filter wave the frames are first aligned to their overlapping
    reference images (``align_mode=REFERENCE`` on success). MIRI→MIRI fallback
    then runs in parallel for hard failures and for REFERENCE solutions whose
    dispersion exceeds the per-filter (or uniform CLI) quality-hold threshold;
    MIRI_REL is kept only when its absolute dispersion improves on REFERENCE.
    The fallback pass repeats once so same-filter MIRI_REL successes can parent
    remaining hard failures.

    Summary ``status`` is binary SUCCESS / FAILURE; the method is recorded in
    ``align_mode``. Per-frame failures are always recorded and processing
    continues.

    Parameters
    ----------
    frames : list
        Overlap records from :func:`run_overlaps` or ``overlap_summary.json``.
    run_alignment : callable, optional
        Accepted for API compatibility and ignored; workers call the
        module-level :func:`run_alignment`.
    nbright : int
        Number of bright sources for JHAT.
    plot, verbose : bool
        Diagnostics and verbose error detail.
    cache_dir : pathlib.Path, optional
        Reference photometry cache directory.
    match_radius_arcsec : float, optional
        Sky radius for merging reference catalogs.
    clip_to_align_footprint : bool, optional
        Clip master catalogs to the science illuminated footprint.
    refine, refine_sigma, refine_max_iter
        Iterative refine controls.
    use_filter_calibrators : bool, optional
        Apply :func:`calibrator_settings_for_filter` overrides.
    fallback : bool, optional
        Enable MIRI→MIRI relative fallback.
    max_nircam_dispersion_mas : float, optional
        Uniform REFERENCE quality hold in mas; ``None`` uses the per-filter
        map and values at or below zero disable the hold.
    min_ref_overlap_frac : float, optional
        Minimum union reference coverage required to attempt alignment.
    summary_outfile : pathlib.Path, optional
        Summary table rewritten after each finished frame.
    workers : int, optional
        Worker processes per filter wave.
    repo : pathlib.Path, optional
        Repo root passed to spawn workers.

    Returns
    -------
    tuple
        ``(n_failures, summary_rows)``.
    """
    del run_alignment  # workers call the module-level run_alignment

    repo_str = str((repo or _resolve_repo_root(None)).resolve())
    workers = max(1, int(workers))
    # CLI: None → per-filter map; <=0 → disable; >0 → uniform override.
    if max_nircam_dispersion_mas is not None and max_nircam_dispersion_mas <= 0:
        max_nircam_dispersion_mas = 0.0  # sentinel: disabled for all filters

    # Drop low-overlap frames before any alignment work. These remain in
    # overlap_summary* only and are omitted from alignment_summary.txt.
    frames, n_rejected = reject_zero_nircam_overlap_frames(
        frames, min_ref_overlap_frac=min_ref_overlap_frac
    )
    groups = _group_frames_by_filter(frames)

    failures = 0
    n_ok = 0
    n_fallback = 0
    rows: list[AlignmentSummaryRow] = []
    row_by_miri: dict[str, AlignmentSummaryRow] = {}
    successes: list[SuccessfulAlignment] = []

    def flush_summary() -> None:
        if summary_outfile is None:
            return
        # Omit PENDING quality-holds until MIRI_REL finishes or is exhausted.
        public = [r for r in rows if r.status != 'PENDING']
        write_alignment_summary(public, summary_outfile)

    def finalize_quality_hold_keep_reference(
        prev: AlignmentSummaryRow, *, reason: str
    ) -> None:
        """
        Keep the REFERENCE solution after MIRI_REL does not improve it.

        The per-filter dispersion cut is a *try MIRI_REL* trigger, not a hard
        reject: a usable REFERENCE WCS remains SUCCESS when fallback cannot
        beat it.
        """
        nonlocal n_ok
        final = AlignmentSummaryRow(
            miri_path=prev.miri_path,
            filter=prev.filter,
            status='SUCCESS',
            n_calibrators=prev.n_calibrators,
            dispersion_mas=prev.dispersion_mas,
            aligned_path=prev.aligned_path,
            align_mode=_normalize_align_mode(prev.align_mode),
            original_ref=prev.original_ref,
            aligned_to=prev.aligned_to,
            ref_overlap_frac=prev.ref_overlap_frac,
        )
        idx = rows.index(prev)
        rows[idx] = final
        row_by_miri[prev.miri_path] = final
        if isinstance(final.dispersion_mas, float) and final.aligned_path not in (
            None,
            'NA',
        ):
            successes.append(
                SuccessfulAlignment(
                    miri_path=final.miri_path,
                    jhat_path=str(final.aligned_path),
                    filter=final.filter,
                    wavelength_um=filter_wavelength_um(final.filter),
                    dispersion_mas=float(final.dispersion_mas),
                    relative_dispersion_mas=float(final.dispersion_mas),
                    align_mode='REFERENCE',
                    original_ref=str(final.original_ref),
                    aligned_to=str(final.aligned_to),
                    photfile=None,
                )
            )
            n_ok += 1
        print(
            f'DONE  {Path(prev.miri_path).name}  {prev.filter}  SUCCESS  '
            f'align_mode=REFERENCE  dispersion_mas={final.dispersion_mas:.3f} '
            f'({reason})',
            flush=True,
        )
        flush_summary()

    def record_result(result, *, count_fallback: bool = False) -> None:
        nonlocal n_ok, n_fallback
        row = AlignmentSummaryRow(**result.row)
        prev = row_by_miri.get(result.miri_path)

        # MIRI_REL did not improve a REFERENCE quality hold: keep REFERENCE.
        if (
            not result.ok
            and result.mode == 'fallback'
            and prev is not None
            and _is_reference_quality_hold(prev)
        ):
            why = 'MIRI_REL did not improve REFERENCE'
            if result.error:
                why = f'{why}: {result.error}'
            finalize_quality_hold_keep_reference(prev, reason=why)
            if verbose and result.error:
                print(f'  detail: {result.error}', file=sys.stderr, flush=True)
            return

        if prev is None:
            rows.append(row)
        else:
            idx = rows.index(prev)
            rows[idx] = row
        row_by_miri[result.miri_path] = row

        if result.ok and result.success is not None:
            successes.append(SuccessfulAlignment(**result.success))
            if prev is None or prev.status != 'SUCCESS':
                n_ok += 1
                if count_fallback or result.mode == 'fallback':
                    n_fallback += 1

        print(_format_worker_done(result), flush=True)
        if verbose and result.error:
            print(f'  detail: {result.error}', file=sys.stderr, flush=True)
        flush_summary()

    if summary_outfile is not None:
        summary_outfile = Path(summary_outfile).expanduser().resolve()
        write_alignment_summary(rows, summary_outfile)
        print(f'Live alignment summary → {summary_outfile}')

    print(
        f'Alignment plan: {len(groups)} filter wave(s), '
        f'{sum(len(v) for v in groups.values())} frame(s) after rejecting '
        f'{n_rejected} low-overlap '
        f'(ref_overlap_frac < {min_ref_overlap_frac:.4f}), workers={workers}'
    )
    for filt, group in groups.items():
        print(f'  {filt}: {len(group)} frame(s)')

    if not groups:
        print('No science frames with reference overlap remain to align.')
        return 0, rows

    common_job = dict(
        repo=repo_str,
        nbright=nbright,
        plot=plot,
        verbose=verbose,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        match_radius_arcsec=match_radius_arcsec,
        clip_to_align_footprint=clip_to_align_footprint,
        refine=refine,
        refine_sigma=refine_sigma,
        refine_max_iter=refine_max_iter,
        use_filter_calibrators=use_filter_calibrators,
        max_nircam_dispersion_mas=max_nircam_dispersion_mas,
    )

    if use_filter_calibrators:
        print(
            'Filter calibrators: F770W → '
            f'{describe_calibrator_settings(F770W_CALIBRATOR_SETTINGS)}'
        )
    else:
        print('Filter calibrators: disabled (--no-filter-calibrators)')

    if max_nircam_dispersion_mas == 0.0:
        print('REFERENCE quality hold: disabled')
    elif max_nircam_dispersion_mas is not None:
        print(
            f'REFERENCE quality hold: uniform > '
            f'{max_nircam_dispersion_mas:.1f} mas → try MIRI_REL '
            f'(keep only if improved)'
        )
    else:
        parts = []
        for name, thr in FILTER_MAX_REFERENCE_DISPERSION_MAS.items():
            parts.append(f'{name}:{"off" if thr is None else f"{thr:.0f}"}')
        print(
            'REFERENCE quality hold (per filter, mas → try MIRI_REL; '
            f'keep only if improved): {", ".join(parts)}'
        )

    for filt, group in groups.items():
        print()
        print('=' * 72)
        print(f'Filter wave {filt}: {len(group)} frame(s)')
        print('=' * 72)

        pending: dict[str, dict] = {}
        for frame in group:
            miri_path, ref_images, best_ref = _frame_ref_images(frame)
            # Safety: low-overlap frames are rejected above and must not reach
            # run_alignment / JHAT.
            if not ref_images or not best_ref:
                continue
            outdir = alignment_outdir_for(miri_path)
            pending[miri_path] = {
                **common_job,
                'miri_path': miri_path,
                'filter': filt,
                'ref_images': ref_images,
                'best_ref': best_ref,
                'outdir': str(outdir),
                'ref_overlap_frac': _frame_ref_overlap_frac(frame),
            }

        # --- Pass 1: parallel REFERENCE alignment ---
        nircam_jobs = [{**job, 'mode': 'nircam'} for job in pending.values()]

        _run_jobs_parallel(
            nircam_jobs,
            run_nircam_align_job,
            workers=workers,
            label=f'{filt} REFERENCE',
            on_result=record_result,
        )

        # --- Passes 2+: parallel MIRI fallback ---
        if fallback:
            for pass_idx in (1, 2):
                need_fallback = [
                    miri
                    for miri, row in row_by_miri.items()
                    if miri in pending and _needs_miri_fallback(row)
                ]
                for miri in pending:
                    if miri not in row_by_miri:
                        need_fallback.append(miri)
                seen: set[str] = set()
                ordered_need: list[str] = []
                for miri in need_fallback:
                    if miri not in seen:
                        seen.add(miri)
                        ordered_need.append(miri)

                fb_jobs = []
                for miri in ordered_need:
                    ranked = rank_fallback_parents(miri, filt, successes, max_parents=5)
                    if not ranked:
                        continue
                    prev_row = row_by_miri.get(miri)
                    ref_disp = (
                        float(prev_row.dispersion_mas)
                        if prev_row is not None
                        and _is_reference_quality_hold(prev_row)
                        and isinstance(prev_row.dispersion_mas, float)
                        else None
                    )
                    fb_jobs.append(
                        {
                            **pending[miri],
                            'mode': 'fallback',
                            'parent': asdict(ranked[0][0]),
                            'parents': [asdict(p) for p, _ov in ranked],
                            'overlap_fraction': ranked[0][1],
                            'reference_dispersion_mas': ref_disp,
                        }
                    )

                if not fb_jobs:
                    break

                any_new = False

                def on_fallback(result) -> None:
                    nonlocal any_new
                    before = row_by_miri.get(result.miri_path)
                    record_result(result, count_fallback=True)
                    if result.ok and (before is None or before.status != 'SUCCESS'):
                        any_new = True

                _run_jobs_parallel(
                    fb_jobs,
                    run_fallback_align_job,
                    workers=workers,
                    label=f'{filt} fallback pass {pass_idx}',
                    on_result=on_fallback,
                )
                if not any_new:
                    break

        # Finalize remaining REFERENCE quality-holds when no MIRI_REL parent
        # was available (or fallback was disabled): keep the REFERENCE WCS.
        for miri in list(pending):
            row = row_by_miri.get(miri)
            if row is None or not _is_reference_quality_hold(row):
                continue
            finalize_quality_hold_keep_reference(
                row,
                reason='no MIRI_REL parent; keeping REFERENCE solution',
            )

        # Final failure tally for this filter wave.
        wave_failures = [
            miri
            for miri in pending
            if row_by_miri.get(miri) is not None
            and row_by_miri[miri].status not in ('SUCCESS', 'REJECTED')
        ]
        # Ensure every pending frame has a row.
        for miri, job in pending.items():
            if miri in row_by_miri:
                continue
            row = harvest_alignment_metrics(
                miri,
                Path(job['outdir']),
                ran_ok=False,
                ref_overlap_frac=job.get('ref_overlap_frac', 'NA'),
            )
            row.filter = filt
            rows.append(row)
            row_by_miri[miri] = row
            wave_failures.append(miri)
            flush_summary()

        failures += len(wave_failures)

        print(
            f'Filter wave {filt} done: '
            f'{sum(1 for m in pending if row_by_miri[m].status == "SUCCESS")} ok, '
            f'{len(wave_failures)} failed'
        )

    print()
    print(
        f'Alignment finished: {n_ok} ok ({n_fallback} via MIRI fallback), '
        f'{n_rejected} rejected (ref_overlap_frac < {min_ref_overlap_frac:.4f}), '
        f'{failures} failed'
    )
    return failures, rows


def run_overlaps(
    args: argparse.Namespace,
    *,
    MirIFootprint,
    BestOverlap,
    compute_overlap,
) -> list[FrameOverlaps]:
    """
    Discover science / reference overlaps and write the overlap summaries.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (``data_dir``, ``filters``, ``limit``,
        ``overlap_outdir``, ``repo``, ``galaxy``).
    MirIFootprint, BestOverlap, compute_overlap : callable
        Injected from :mod:`st123.mosaic.image_overlap`.

    Returns
    -------
    list of FrameOverlaps
        One entry per discovered science frame.
    """
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Data directory not found: {data_dir}')

    filters = parse_filters_arg(getattr(args, 'filters', None))
    miri_images = filter_miri_images(discover_miri_images(data_dir), filters)
    refs = discover_ref_images(data_dir)
    if args.limit is not None:
        miri_images = miri_images[: args.limit]

    if not miri_images:
        msg = (
            f'No MIRI *_cal.fits found under '
            f'{data_dir}/JWST/MIRI/<FILTER>/<obsid>/mastDownload/JWST/'
        )
        if filters:
            msg += f' for filters {",".join(filters)}'
        raise FileNotFoundError(msg)
    if not refs:
        raise FileNotFoundError(
            f'No reference coadd*i2d.fits found under {data_dir}/reference/ '
            f'or {data_dir}/reduction/reference/'
        )

    print(f'Repo:              {args.repo}')
    print(f'Data dir:          {data_dir}')
    print(f'Dataset label:     {args.galaxy}')
    print(f'Filters:           {", ".join(filters) if filters else "ALL"}')
    print(f'MIRI images:       {len(miri_images)}')
    print(f'Reference images:  {len(refs)}')
    print()

    frames = find_frame_overlaps(
        miri_images,
        refs,
        MirIFootprint=MirIFootprint,
        BestOverlap=BestOverlap,
        compute_overlap=compute_overlap,
    )
    overlap_outdir = ensure_overlap_outdir(
        data_dir,
        Path(args.overlap_outdir).expanduser() if args.overlap_outdir else None,
    )
    write_overlap_summaries(frames, overlap_outdir)
    return frames
