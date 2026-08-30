"""
Unified JWST alignment library.

Everything needed to align JWST imaging lives here, ordered from low-level
knobs to high-level orchestration:

1. Calibrator settings - per-filter JHAT / refine tuning and quality-hold cuts
2. Fallback helpers - MIRI->MIRI parent ranking, provenance headers
3. JHAT core - photometry, dispersion, ``align_jwst_image``, visit mosaics
4. Relative API - ``run_alignment`` plus reference-catalog construction
5. Overlap discovery - footprint matching and alignment summary tables
6. Parallel workers - spawn-safe visit / REFERENCE / MIRI_REL jobs
7. Orchestration - ``align_from_frames`` / ``run_overlaps``

Visit and reference modes share the same parallel runner
(``_run_jobs_parallel``), logging contract (``capture_output`` for JHAT /
Image3 + parent START/DONE), JHAT entry (``align_jwst_image``), and post-JHAT
metrics (``harvest_alignment_metrics`` / provenance). They diverge on topology:
visit uses a mosaic/Gaia cascade with chip-relative alignment; reference uses
overlap discovery, filter-wave ordering, MIRI_REL fallback, and quality holds.

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
import sys
import traceback
import warnings
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

warnings.filterwarnings('ignore')
# stpipe/pysiaf log handlers raise on multi-argument warnings during spawn
# worker imports; never let a logging failure abort alignment.
logging.raiseExceptions = False
logger = logging.getLogger(__name__)

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
from st123.datamodels import DataModelLike, as_datamodel
from astropy.stats import sigma_clipped_stats  # noqa: E402
from astropy.table import Column, Row, Table, vstack  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
from photutils.detection import DAOStarFinder  # noqa: E402

from st123.utils.logging import (  # noqa: E402
    capture_output,
    configure_worker_logging,
    run_logged_subprocess,
    suppress_output,
)

# Import the science stack quietly: these packages print banners and emit
# import-time log records that break stpipe handlers inside spawn workers.
with suppress_output():
    from jhat import jwst_photclass, st_wcs_align  # noqa: E402
    from jwst.associations import asn_from_list  # noqa: E402
    from jwst.associations.lib.rules_level3_base import (  # noqa: E402
        DMS_Level3_Base,
    )
    from jwst.pipeline import calwebb_image3  # noqa: E402

# JHAT's stock Gaia path uses ESA TAP; force Vizier for all JWST align paths.
from st123.stages.alignment.gaia_catalog import install_jhat_gaia_vizier_patch  # noqa: E402

install_jhat_gaia_vizier_patch()

from st123.stages.mosaic.region import SRegionPolygon  # noqa: E402
from st123.utils.helpers import (  # noqa: E402
    input_list,
    is_full_frame_miri,
    xmatch_common,
)
from st123.utils.compatibility import patch_jwst_for_photutils3  # noqa: E402
from st123.utils.settings import (  # noqa: E402
    CROWDED_JHAT_NBRIGHT,
    DEFAULT_MAX_REFERENCE_DISPERSION_MAS,
    FILTER_MAX_REFERENCE_DISPERSION_MAS,
    crowded_jwst_params,
    relaxed_gaia_params,
    relaxed_jwst_params,
    strict_gaia_params,
    strict_jwst_params,
)


# ---------------------------------------------------------------------------
# 1. Calibrator settings
# ---------------------------------------------------------------------------
#
# F770W fields often yield hundreds of detections (PAH / arm structure). Across
# frames, higher ``n_calibrators`` correlates with worse dispersion because busy
# fields are harder - but *within* a frame the brightest JHAT matches are
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


# Default (non-F770W) pipeline settings - use CLI / strict_jwst_params defaults.
DEFAULT_CALIBRATOR_SETTINGS = CalibratorSettings()

# F770W: prefer bright calibrators but do *not* over-clip to a tiny residual
# floor. Hard 80 mas residuals with min_calibrators=15 previously produced
# ~20-star solutions that looked good vs the refined subset (JWDISPM ~25 mas)
# while destroying dither-to-dither consistency (~300-600 mas peer offsets).
# Pre-align pipeline WCSs already agree at ~20 mas; peer QA below recovers
# that when REFERENCE overfits. Crowded/bright nuclei (e.g. M82) additionally
# use the ``crowded_jwst_params`` retry in :func:`align_jwst_image`.
# Keep global F770W cuts close to the proven baseline. Bright/crowded nuclei
# are handled by the ``crowded_jwst_params`` retry in :func:`align_jwst_image`
# rather than starving all F770W frames of calibrators.
F770W_CALIBRATOR_SETTINGS = CalibratorSettings(
    nbright=200,
    refine_sigma=2.0,
    refine_max_iter=5,
    refine_dist_limit_arcsec=0.50,
    max_residual_arcsec=0.20,
    min_calibrators=40,
)

# After REFERENCE, overlapping same-filter JHAT products must agree on sky.
# Peer median separation above this (with enough matches) marks the worse
# frame PENDING so MIRI_REL can restore relative consistency.
DEFAULT_PEER_DISPERSION_MAX_MAS: float = 100.0  # ~1 MIRI pixel
DEFAULT_PEER_MIN_OVERLAP: float = 0.15
DEFAULT_PEER_MIN_MATCHES: int = 20
DEFAULT_PEER_MATCH_RADIUS_ARCSEC: float = 0.5

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
# 2. Fallback (MIRI->MIRI relative) helpers
# ---------------------------------------------------------------------------
#
# When direct reference alignment fails, align the failed frame to a
# successfully aligned MIRI image that is closest in wavelength and has the
# largest footprint overlap. Absolute dispersion is the quadrature sum of the
# parent absolute dispersion and the new relative dispersion.


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
    # True when this parent is a PENDING REFERENCE quality-hold used only to
    # bootstrap MIRI_REL (not yet a finalized SUCCESS absolute).
    provisional: bool = False


# Ranking offsets for MIRI_REL parent selection (mas-equivalent score terms).
_SAME_FILTER_PARENT_BONUS_MAS: float = -35.0
_F770W_SEED_BONUS_MAS: float = -50.0
_PROVISIONAL_PARENT_PENALTY_MAS: float = 55.0
# Reject provisional parents above the filter REFERENCE threshold whenever at
# least one finalized SUCCESS parent is available.
_PROVISIONAL_ABOVE_THRESHOLD_PENALTY_MAS: float = 200.0
# F770W mean/median skew: hold REFERENCE when mean is inflated vs median.
F770W_MEAN_MEDIAN_SKEW_RATIO: float = 1.5
F770W_SKEW_MEAN_FLOOR_FRAC: float = 0.7


def filter_wavelength_um(image: DataModelLike) -> float:
    """
    Return the pivot wavelength of a science frame in microns.

    Reads photometric header cards on the datamodel (``PHOTPLAM`` in
    Angstroms on HST / JWST imaging). Filter-name tables are not used.

    Parameters
    ----------
    image
        Datamodel or FITS path.

    Returns
    -------
    float
        Pivot wavelength in microns, or ``inf`` when the header has no
        usable wavelength card.
    """
    return float(as_datamodel(image).wavelength_um)


def sort_frames_blue_to_red(frames: list) -> list:
    """
    Sort overlap frames by increasing header wavelength, then path.

    Parameters
    ----------
    frames : list
        Overlap frames (``FrameOverlaps`` instances or dicts).

    Returns
    -------
    list
        Frames ordered blue -> red.
    """

    def key(frame) -> tuple:
        path = frame['miri_path'] if isinstance(frame, dict) else frame.miri_path
        return (filter_wavelength_um(path), str(path))

    return sorted(frames, key=key)


def load_s_region(image) -> SRegionPolygon:
    """
    Parse ``S_REGION`` from a science datamodel.

    Parameters
    ----------
    image
        Datamodel or FITS path.

    Returns
    -------
    SRegionPolygon
        Sky polygon for the image footprint.

    Raises
    ------
    KeyError
        If no HDU carries an ``S_REGION`` keyword.
    """
    model = as_datamodel(image)
    text = model.s_region
    if not text:
        raise KeyError(f'No S_REGION in {model.path}')
    return SRegionPolygon.parse(text)


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
    lowest estimated absolute score with these preferences:
    - same-filter SUCCESS parents
    - F770W seed parents for redder targets
    - finalized (non-provisional) parents over PENDING REFERENCE holds
    - then ``sqrt(parent_abs**2 + assume_relative_mas**2) +
      wavelength_penalty * |dlambda|``, closest wavelength, largest overlap

    Parameters
    ----------
    miri_path : str
        Frame that needs a parent.
    filter_name : str
        Filter of ``miri_path`` (same-filter ranking and dispersion cuts).
    successes : list of SuccessfulAlignment
        Candidate parents. Wavelength ranking uses each parent's stored
        ``wavelength_um`` (from that frame's photometric headers) and the
        target frame's datamodel.
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

    target_filt = str(filter_name or '').upper().split('_', 1)[0]
    target_wl = filter_wavelength_um(miri_path)
    thr = max_reference_dispersion_mas(filter_name)
    has_finalized = any(not p.provisional for p in successes)
    f770_parents = [
        p
        for p in successes
        if str(p.filter or '').upper().split('_', 1)[0] == 'F770W'
    ]
    f770_wl = (
        float(f770_parents[0].wavelength_um) if f770_parents else float('nan')
    )
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
        # Skip provisional parents above the filter threshold when a finalized
        # SUCCESS parent exists - avoids inheriting inflated absolute scores.
        if (
            parent.provisional
            and has_finalized
            and thr is not None
            and float(parent.dispersion_mas) > float(thr)
        ):
            continue
        est_abs = combine_dispersion_mas(
            float(parent.dispersion_mas), float(assume_relative_mas)
        )
        dlam = abs(parent.wavelength_um - target_wl)
        score = est_abs + float(wavelength_penalty_mas_per_um) * dlam
        parent_filt = str(parent.filter or '').upper().split('_', 1)[0]
        if parent_filt == target_filt:
            score += _SAME_FILTER_PARENT_BONUS_MAS
        # Redder-than-F770W frames should prefer an F770W absolute seed.
        if (
            math.isfinite(target_wl)
            and math.isfinite(f770_wl)
            and target_wl > f770_wl + 0.05
        ):
            if parent_filt == 'F770W':
                score += _F770W_SEED_BONUS_MAS
            elif parent.wavelength_um + 0.05 < f770_wl:
                # Prefer F770W over much-bluer seeds when both overlap.
                score += 8.0
        if parent.provisional:
            score += _PROVISIONAL_PARENT_PENALTY_MAS
            if thr is not None and float(parent.dispersion_mas) > float(thr):
                score += _PROVISIONAL_ABOVE_THRESHOLD_PENALTY_MAS
        ranked.append((score, dlam, -frac, parent))

    if not ranked:
        return []
    ranked.sort()
    out: list[tuple[SuccessfulAlignment, float]] = []
    for _score, _dlam, neg_frac, parent in ranked[: max(1, int(max_parents))]:
        out.append((parent, -neg_frac))
    return out


def build_miri_rel_parent_pool(
    filter_name: str,
    successes: list[SuccessfulAlignment],
    row_by_miri: dict[str, AlignmentSummaryRow],
    pending_paths: set[str] | list[str],
) -> list[SuccessfulAlignment]:
    """
    Build the MIRI_REL parent pool: finalized SUCCESS, then safe provisionals.

    Provisional PENDING REFERENCE holds are included only when they meet the
    filter dispersion threshold, or when no finalized parents exist yet
    (same-filter bootstrap).

    Parameters
    ----------
    filter_name : str
        Current filter wave.
    successes : list of SuccessfulAlignment
        Finalized SUCCESS parents (any filter wave so far).
    row_by_miri : dict
        Live summary rows keyed by MIRI path.
    pending_paths : set or list of str
        MIRI paths in the current filter wave.

    Returns
    -------
    list of SuccessfulAlignment
        Deduplicated parent pool (finalized entries win on path clashes).
    """
    thr = max_reference_dispersion_mas(filter_name)
    finalized = [p for p in successes if not p.provisional]
    # Refresh provisional flag on any SUCCESS that somehow still carries it.
    for parent in finalized:
        parent.provisional = False

    provisionals = provisional_fallback_parents_from_holds(
        filter_name, row_by_miri, pending_paths
    )
    safe_prov: list[SuccessfulAlignment] = []
    for parent in provisionals:
        parent.provisional = True
        if thr is None or float(parent.dispersion_mas) <= float(thr):
            safe_prov.append(parent)
        elif not finalized:
            # Bootstrap only: whole wave held / no SUCCESS yet.
            safe_prov.append(parent)

    seen: dict[str, SuccessfulAlignment] = {}
    for parent in list(finalized) + safe_prov:
        seen.setdefault(parent.miri_path, parent)
    return list(seen.values())


def repropagate_miri_rel_absolutes(
    successes: list[SuccessfulAlignment],
    row_by_miri: dict[str, AlignmentSummaryRow],
    rows: list[AlignmentSummaryRow],
) -> int:
    """
    Recompute MIRI_REL absolute dispersions after parent absolutes improve.

    Children that aligned to a provisional / early parent keep an inflated
    ``JWDISPM`` = sqrt(parent0^2 + rel^2) even after the parent later lands a better
    absolute. This rewrites headers, summary rows, and ``successes`` from the
    stored ``JWDISPR`` / ``relative_dispersion_mas`` and the parent's current
    absolute - WCS is unchanged.

    Parameters
    ----------
    successes : list of SuccessfulAlignment
        Live SUCCESS parents (updated in place).
    row_by_miri : dict
        Summary rows keyed by MIRI path (updated in place).
    rows : list of AlignmentSummaryRow
        Ordered summary rows (updated in place).

    Returns
    -------
    int
        Number of child frames whose absolute dispersion decreased.
    """
    by_jhat: dict[str, SuccessfulAlignment] = {}
    by_miri: dict[str, SuccessfulAlignment] = {}
    for parent in successes:
        by_jhat[str(Path(parent.jhat_path).resolve())] = parent
        by_miri[parent.miri_path] = parent

    n_updated = 0
    # Iterate to convergence for short MIRI_REL chains (parent->child->...).
    for _ in range(max(1, len(successes))):
        changed = False
        for child in list(successes):
            if str(child.align_mode).upper() != 'MIRI_REL':
                continue
            parent = by_jhat.get(str(Path(child.aligned_to).resolve()))
            if parent is None:
                continue
            rel = float(child.relative_dispersion_mas)
            # Prefer JWDISPR from the product when present.
            try:
                with as_datamodel(child.jhat_path).open(memmap=True) as hdul:
                    jwdispr = hdul[0].header.get('JWDISPR')
                if jwdispr is not None:
                    rel_hdr = float(jwdispr) * 1000.0
                    if math.isfinite(rel_hdr) and rel_hdr > 0:
                        rel = rel_hdr
            except Exception:
                pass
            new_abs = combine_dispersion_mas(float(parent.dispersion_mas), rel)
            old_abs = float(child.dispersion_mas)
            if not math.isfinite(new_abs) or new_abs >= old_abs - 0.05:
                continue
            write_alignment_provenance(
                child.jhat_path,
                align_mode='MIRI_REL',
                original_ref=str(parent.original_ref or child.original_ref),
                aligned_to=child.aligned_to,
                relative_dispersion_mas=rel,
                absolute_dispersion_mas=new_abs,
                n_calibrators=(
                    row_by_miri[child.miri_path].n_calibrators
                    if child.miri_path in row_by_miri
                    and isinstance(row_by_miri[child.miri_path].n_calibrators, int)
                    else None
                ),
            )
            child.dispersion_mas = new_abs
            child.relative_dispersion_mas = rel
            child.original_ref = str(parent.original_ref or child.original_ref)
            child.provisional = False
            row = row_by_miri.get(child.miri_path)
            if row is not None and isinstance(row.dispersion_mas, float):
                row.dispersion_mas = new_abs
                row.original_ref = child.original_ref
                row_by_miri[child.miri_path] = row
                for idx, existing in enumerate(rows):
                    if existing.miri_path == child.miri_path:
                        rows[idx] = row
                        break
            by_jhat[str(Path(child.jhat_path).resolve())] = child
            by_miri[child.miri_path] = child
            n_updated += 1
            changed = True
            logger.info(
                f'Repropagated MIRI_REL absolute {Path(child.miri_path).name}: '
                f'{old_abs:.2f} -> {new_abs:.2f} mas '
                f'(parent {Path(parent.miri_path).name} '
                f'abs={parent.dispersion_mas:.2f}, rel={rel:.2f})'
            )
        if not changed:
            break
    return n_updated


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


def find_aligned_refcat(jhat_path: str) -> str | None:
    """
    Locate the refined reference catalog written next to a JHAT product.

    Parameters
    ----------
    jhat_path : str
        Path to a ``*_jhat.fits`` product.

    Returns
    -------
    str or None
        Path to ``*.refcat.txt`` when present.
    """
    jhat = Path(jhat_path)
    stem = jhat.name.replace('_jhat.fits', '')
    cand = jhat.parent / f'{stem}.refcat.txt'
    if cand.is_file():
        return str(cand.resolve())
    return None


def count_refcat_calibrators(jhat_path: str) -> int | None:
    """
    Count rows in the JHAT ``*.refcat.txt`` beside a product.

    .. warning::
        This is **not** the number of calibrators used in the WCS fit or
        final dispersion. JHAT dumps the full loaded reference catalog into
        ``*.refcat.txt`` (often the unclipped master, tens of thousands of
        rows). Prefer :func:`count_alignment_calibrators` / header ``JWNCAL``.

    Parameters
    ----------
    jhat_path : str
        Path to a ``*_jhat.fits`` product.

    Returns
    -------
    int or None
        Row count, or ``None`` when no refcat exists / cannot be read.
    """
    path = find_aligned_refcat(jhat_path)
    if path is None:
        return None
    try:
        table = Table.read(path, format='ascii')
    except Exception:
        return None
    return int(len(table))


def find_dispersion_refcat(jhat_path: str) -> str | None:
    """
    Locate the reference catalog used for the final dispersion measurement.

    Preference order (per-frame ``alignment_output`` only - never shared
    cache paths, so parallel workers cannot cross-pollute counts):

    1. ``JWCAT`` basename beside the JHAT product
    2. ``master_ref_refined.phot.txt`` / latest refine iter in that directory
    3. ``*.refcat.txt`` for this frame stem

    Parameters
    ----------
    jhat_path : str
        Path to a ``*_jhat.fits`` product.

    Returns
    -------
    str or None
        Catalog path, or ``None`` when none can be resolved.
    """
    jhat = Path(jhat_path)
    parent = jhat.parent
    try:
        with as_datamodel(jhat).open() as hdul:
            jwcat = hdul[0].header.get('JWCAT')
    except Exception:
        jwcat = None
    if jwcat:
        cand = parent / Path(str(jwcat)).name
        if cand.is_file():
            return str(cand.resolve())

    refined = parent / 'master_ref_refined.phot.txt'
    if refined.is_file():
        return str(refined.resolve())
    iters = sorted(parent.glob('master_ref_refined_iter*.phot.txt'))
    if iters:
        return str(iters[-1].resolve())

    return find_aligned_refcat(str(jhat))


def count_alignment_calibrators(
    jhat_path: str,
    *,
    dist_limit: float = 0.5,
    sig: float = 2.0,
) -> int | None:
    """
    Count calibrators used for the final alignment dispersion of a JHAT product.

    Cross-matches the post-alignment science photometry to the dispersion
    reference catalog with the same radius / sigma-clip as
    :func:`calc_dispersion` / :func:`jwst_dispersion`, and returns the number
    of matches that survive clipping. This is the authoritative value for
    ``JWNCAL`` / summary ``n_calibrators`` when the header is missing or
    untrustworthy (e.g. a stale dump of the full master catalog length).

    Parameters
    ----------
    jhat_path : str
        Path to a ``*_jhat.fits`` product.
    dist_limit : float, optional
        Cross-match radius in arcsec (must match ``jwst_dispersion``).
    sig : float, optional
        Sigma-clip threshold (must match ``jwst_dispersion``).

    Returns
    -------
    int or None
        Clipped match count, or ``None`` when catalogs are unavailable.
    """
    phot = find_aligned_photfile(jhat_path)
    refcat_path = find_dispersion_refcat(jhat_path)
    if phot is None or refcat_path is None:
        return None
    try:
        refcat = Table.read(refcat_path, format='ascii')
        _mean, _med, _std, n_cal = calc_dispersion(
            refcat, phot, dist_limit=dist_limit, sig=sig, plot=False
        )
    except Exception:
        return None
    return int(n_cal)


def jwncal_is_plausible(n_cal: int, jhat_path: str | None = None) -> bool:
    """
    Return whether a ``JWNCAL`` value looks like a real calibrator count.

    Rejects the common pollution mode where ``JWNCAL`` was set to the full
    unclipped master-catalog length (tens of thousands) instead of the
    clipped match count used for ``JWDISPM``.

    Parameters
    ----------
    n_cal : int
        Candidate calibrator count.
    jhat_path : str, optional
        When provided, also require ``n_cal`` not greatly exceed the science
        photometry row count for that frame.

    Returns
    -------
    bool
        ``True`` when the value is usable as ``n_calibrators``.
    """
    if n_cal < 0:
        return False
    # Soft-fail / empty match sets legitimately write 0.
    if n_cal == 0:
        return True
    # Master F150W2-style catalogs are ~1e4-1e5; real MIRI/NIRCam match
    # counts for a single frame are orders of magnitude smaller.
    if n_cal >= 5000:
        return False
    if jhat_path is not None:
        phot = find_aligned_photfile(jhat_path)
        if phot is not None:
            try:
                n_phot = int(len(pd.read_csv(phot, sep=r'\s+')))
            except Exception:
                n_phot = None
            # Matched calibrators cannot exceed science detections.
            if n_phot is not None and n_cal > n_phot:
                return False
    return True


def peer_sky_dispersion_mas(
    phot_a: str | Path,
    phot_b: str | Path,
    *,
    match_radius_arcsec: float = DEFAULT_PEER_MATCH_RADIUS_ARCSEC,
) -> tuple[float | None, int]:
    """
    Median sky separation between two aligned photometry catalogs.

    Used to QA that overlapping same-filter JHAT frames remain consistent
    with each other after independent REFERENCE alignments. A low
    ``JWDISPM`` vs a tiny refined refcat can still hide large peer offsets.

    Parameters
    ----------
    phot_a, phot_b : str or Path
        JHAT photometry catalogs with ``ra`` / ``dec`` columns.
    match_radius_arcsec : float, optional
        Maximum match radius for the nearest-neighbour cross-match.

    Returns
    -------
    tuple
        ``(median_separation_mas, n_matches)``. Median is ``None`` when there
        are no matches inside the radius.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    def _load(path: str | Path) -> Table:
        table = Table.read(str(path), format='ascii')
        cols = {c.lower(): c for c in table.colnames}
        if 'ra' not in cols or 'dec' not in cols:
            raise ValueError(f'Catalog missing ra/dec: {path}')
        return Table(
            {
                'ra': np.asarray(table[cols['ra']], dtype=float),
                'dec': np.asarray(table[cols['dec']], dtype=float),
            }
        )

    a = _load(phot_a)
    b = _load(phot_b)
    if len(a) == 0 or len(b) == 0:
        return None, 0
    ca = SkyCoord(a['ra'] * u.deg, a['dec'] * u.deg)
    cb = SkyCoord(b['ra'] * u.deg, b['dec'] * u.deg)
    _idx, sep, _ = ca.match_to_catalog_sky(cb)
    mask = sep < (match_radius_arcsec * u.arcsec)
    n_match = int(np.count_nonzero(mask))
    if n_match == 0:
        return None, 0
    return float(np.median(sep[mask].to(u.mas).value)), n_match


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
        Number of calibrators used for the final alignment dispersion
        (clipped science<->reference matches that produce ``JWDISPM``).
        Stored in ``JWNCAL``. Pass the value written by :func:`jwst_dispersion`;
        do not pass JHAT ``*.refcat.txt`` row counts.
    """
    with as_datamodel(jhat_path).open(mode='update') as hdul:
        hdr = hdul[0].header
        mode = str(align_mode).upper()
        if mode == 'NIRCAM':
            mode = 'REFERENCE'
        hdr['ALGNMODE'] = (mode, 'VISIT, REFERENCE, or MIRI_REL')
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
            hdr['JWNCAL'] = (
                int(n_calibrators),
                'N calibrators for JWDISPM / align solution',
            )


# ---------------------------------------------------------------------------
# 3. JHAT core and visit-mode helpers
# ---------------------------------------------------------------------------


def get_input_images(
    pattern: Sequence[str] | None = None,
    workdir: str | None = None,
) -> list[str]:
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


def pick_deepest_image(table: Table) -> Row:
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


def add_alignment_groups(table: Table, use_shapely: bool = False) -> Table:
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
        region = as_datamodel(im).s_region
        coords = np.array(region.split('POLYGON ICRS  ')[1].split(' '), dtype=float)
        pgons.append(shapely.Polygon(coords.reshape(4, 2)))
        guide_star_id.append(as_datamodel(im).keyword('GDSTARID', ext=0))

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


# Absolute-hub filter preference for NIRCam visit seeding / mosaic filter.
# Deep F200W / F150W hubs beat large but shallow or narrowband-led visits.
_HUB_FILTER_RANK: dict[str, float] = {
    'f200w': 100.0,
    'f150w2': 90.0,
    'f150w': 85.0,
    'f277w': 70.0,
    'f322w2': 68.0,
    'f356w': 65.0,
    'f444w': 60.0,
    'f250m': 45.0,
    'f300m': 45.0,
    'f335m': 42.0,
    'f360m': 42.0,
    'f430m': 40.0,
}

# Seed mosaic must beat this Gaia residual (mas) with n_cal > 0.
HUB_GAIA_MAX_MAS: float = 40.0
# Post-pass abs retie search (arcsec); covers ~2" wrong-island cases.
HUB_ABS_SEARCH_ARCSEC: float = 5.0
# Post-pass: shift frames worse than this vs hub (arcsec).
HUB_ABS_RETIE_TOL_ARCSEC: float = 0.050
# Do not apply CRVAL shifts larger than this (false multimodal peaks).
HUB_ABS_RETIE_MAX_APPLY_ARCSEC: float = 0.40
# Require a stronger 2-D histogram peak before applying an abs retie shift.
HUB_ABS_RETIE_MIN_PEAK: int = 8
# Frames farther than this from the hub (with a confident peak) are re-JHAT'd
# to the hub catalog rather than CRVAL-shifted.
HUB_FRAME_ABS_MAX_MAS: float = 200.0


def _hub_filter_rank(filt: str) -> float:
    key = str(filt or '').lower()
    if key in _HUB_FILTER_RANK:
        return float(_HUB_FILTER_RANK[key])
    if 'N' in key.upper():
        return 5.0
    return 25.0


def visit_filter_dict(table: Table) -> dict[Any, str]:
    """
    Choose each visit's alignment-mosaic filter.

    Prefers deep broadband hubs (F200W > F150W2 > F150W > ...) over merely
    largest footprint so shallow / large SW filters do not seed weak mosaics.
    Narrowband filters are never selected when any broadband exists.
    Footprint area and depth (frame count / exposure) break remaining ties.

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
        filters = np.unique(tbl['filter']).value
        best_filt: str | None = None
        best_score = float('-inf')
        for filt in filters:
            if 'N' in str(filt).upper():
                continue
            pgons = []
            filter_rows = tbl[tbl['filter'] == filt]
            for im, pupil in zip(filter_rows['image'], filter_rows['pupil']):
                if 'N' in str(pupil).upper():
                    continue
                region = as_datamodel(im).s_region
                coords = np.array(
                    region.split('POLYGON ICRS  ')[1].split(' '), dtype=float
                )
                pgons.append(shapely.Polygon(coords.reshape(4, 2)))
            if not pgons:
                continue
            area = float(shapely.unary_union(pgons).area)
            n_frames = int(len(filter_rows))
            try:
                exptime = float(
                    np.nansum(np.asarray(filter_rows['exptime'], dtype=float))
                )
            except Exception:
                exptime = 0.0
            # Rank dominates; area / depth are tie-breaks only.
            score = (
                _hub_filter_rank(str(filt)) * 1.0e6
                + area * 1.0e3
                + float(n_frames)
                + 1.0e-3 * exptime
            )
            if score > best_score:
                best_score = score
                best_filt = str(filt)
        if best_filt is None:
            # All-narrowband visit: fall back to largest footprint filter.
            net_polygon = []
            for filt in filters:
                pgons = []
                filter_rows = tbl[tbl['filter'] == filt]
                for im, pupil in zip(filter_rows['image'], filter_rows['pupil']):
                    region = as_datamodel(im).s_region
                    coords = np.array(
                        region.split('POLYGON ICRS  ')[1].split(' '), dtype=float
                    )
                    pgons.append(shapely.Polygon(coords.reshape(4, 2)))
                net_polygon.append(shapely.unary_union(pgons) if pgons else None)
            areas = [p.area if p is not None else -1.0 for p in net_polygon]
            best_filt = str(filters[int(np.argmax(areas))])
        visit_filter[vis] = best_filt

    return visit_filter


def get_visit_geoms(table: Table) -> dict[Any, shapely.geometry.base.BaseGeometry]:
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
                region = as_datamodel(im).s_region
                coords = np.array(
                    region.split('POLYGON ICRS  ')[1].split(' '), dtype=float
                )
                pgons.append(shapely.Polygon(coords.reshape(4, 2)))
            net_polygon.append(shapely.unary_union(pgons))
        field.append(shapely.unary_union(net_polygon))

    return dict(zip(visits, field))


def order_visits(table: Table) -> np.ndarray:
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


def jhat_product_needs_realign(
    jhat_path: str | Path,
    *,
    catalog_path: str | Path | None = None,
) -> tuple[bool, str]:
    """
    Decide whether an existing JHAT product should be rebuilt.

    Soft-fail copies (``JWNCAL`` / ``GANCAL`` == 0) and products older than
    the alignment catalog are always rebuilt so a new hub catalog is not
    skipped by a stale on-disk ``*_jhat.fits``.
    """
    path = Path(jhat_path)
    if not path.is_file():
        return True, 'missing'
    try:
        _disp_mas, n_cal = read_dispersion_mas(str(path))
    except Exception:
        return True, 'unreadable'
    if n_cal is None or int(n_cal) <= 0:
        return True, 'soft_fail'
    if catalog_path is not None:
        cat = Path(catalog_path)
        if cat.is_file():
            try:
                if path.stat().st_mtime + 1.0e-6 < cat.stat().st_mtime:
                    return True, 'stale_vs_catalog'
            except OSError:
                pass
    return False, 'ok'


def remove_jhat_for_realign(jhat_path: str | Path) -> None:
    """Delete a JHAT product (and break a symlink) so JHAT can rewrite it."""
    path = Path(jhat_path)
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
    except OSError as exc:
        logger.warning('Could not remove %s for realign: %s', path, exc)


def is_jwst_science_jhat(path: str | Path) -> bool:
    """
    Return True for JWST science JHAT products (excludes HST in mixed trees).
    """
    p = Path(path)
    name = p.name.lower()
    if name.startswith('jw'):
        return True
    try:
        telescop = str(as_datamodel(str(p)).keyword('TELESCOP', ext=0) or '').upper()
        if telescop == 'JWST':
            return True
        instrume = str(as_datamodel(str(p)).keyword('INSTRUME', ext=0) or '').upper()
        return instrume in {'NIRCAM', 'MIRI', 'NIRISS', 'NIRSPEC'}
    except Exception:
        return False


def cal_image_for_jhat(
    jhat_path: str | Path,
    raw_dir: str | Path,
) -> Path | None:
    """Map ``*_jhat.fits`` back to ``reduction/raw/*_cal.fits`` when present."""
    stem = Path(jhat_path).name.replace('_jhat.fits', '')
    raw = Path(raw_dir)
    for name in (f'{stem}_cal.fits', f'{stem}.fits'):
        cand = raw / name
        if cand.exists():
            return cand
    return None


def measure_frame_abs_offset_mas(
    frame: str | Path,
    abs_ref: str | Path,
    *,
    max_search_arcsec: float = HUB_ABS_SEARCH_ARCSEC,
    min_peak: int = 3,
    bin_arcsec: float = 0.05,
) -> dict[str, Any]:
    """Measure absolute sky offset of *frame* relative to *abs_ref* (mas)."""
    from st123.stages.alignment.hst_jhat import measure_hst_sky_offset_2dhist

    off = measure_hst_sky_offset_2dhist(
        Path(frame),
        Path(abs_ref),
        max_offset_arcsec=float(max_search_arcsec),
        bin_arcsec=float(bin_arcsec),
        nbright=800,
        min_peak=int(min_peak),
        exclude_zero_arcsec=min(0.35, max(0.08, 2.0 * float(bin_arcsec))),
    )
    abs_as = float(off.get('abs_arcsec') or 0.0)
    return {
        'ok': bool(off.get('ok')),
        'abs_mas': 1000.0 * abs_as,
        'abs_arcsec': abs_as,
        'peak_count': int(off.get('peak_count') or 0),
        'dra_arcsec': off.get('dra_arcsec'),
        'ddec_arcsec': off.get('ddec_arcsec'),
    }


def select_jwst_jhats_for_abs_redo(
    jhat_paths: Sequence[str | Path],
    abs_ref: str | Path,
    *,
    catalog_path: str | Path | None = None,
    check_stale: bool = False,
    measure_abs: bool = False,
    max_abs_mas: float = HUB_FRAME_ABS_MAX_MAS,
    min_peak: int = 5,
    ncores: int = 1,
) -> list[tuple[Path, str]]:
    """
    Select JWST JHAT products that should be re-aligned to the hub catalog.

    Soft-fails (and optional catalog-mtime staleness) are always selected from
    headers only. Expensive absolute 2-D histogram checks run only when
    *measure_abs* is True; prefer catching wrong-island WCSs from the capped
    abs-retie pass (``skip_large``) instead of measuring every frame twice.
    """
    selected: list[tuple[Path, str]] = []
    candidates: list[Path] = []
    ref = Path(abs_ref)
    cat = catalog_path if check_stale else None
    for path in jhat_paths:
        fp = Path(path)
        if not is_jwst_science_jhat(fp):
            continue
        need, reason = jhat_product_needs_realign(fp, catalog_path=cat)
        if need:
            selected.append((fp, reason))
            continue
        if measure_abs and ref.is_file():
            candidates.append(fp)

    if not measure_abs or not candidates:
        return selected

    # Parallel abs-offset screen for remaining good products only.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _check(fp: Path) -> tuple[Path, str] | None:
        meas = measure_frame_abs_offset_mas(fp, ref, min_peak=int(min_peak))
        if (
            meas['ok']
            and int(meas['peak_count']) >= int(min_peak)
            and float(meas['abs_mas']) > float(max_abs_mas)
        ):
            return (fp, f'abs_qa_{meas["abs_mas"]:.0f}mas')
        return None

    n_workers = max(1, min(int(ncores), len(candidates)))
    if n_workers == 1:
        for fp in candidates:
            hit = _check(fp)
            if hit is not None:
                selected.append(hit)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_check, fp) for fp in candidates]
            for fut in as_completed(futures):
                hit = fut.result()
                if hit is not None:
                    selected.append(hit)
    return selected


def realign_jwst_jhats_to_catalog(
    jhat_reasons: Sequence[tuple[Path, str]],
    *,
    catalog_path: str | Path,
    raw_dir: str | Path,
    jhat_outdir: str | Path,
    guess_offset: tuple[float, float] = (0.0, 0.0),
    ncores: int = 1,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Delete selected JHAT products and re-run JHAT against *catalog_path*.
    """
    cals: list[str] = []
    raw = Path(raw_dir)
    for jhat, reason in jhat_reasons:
        cal = cal_image_for_jhat(jhat, raw)
        if cal is None:
            logger.warning(
                'Abs QA redo skipped %s (%s): no cal under %s',
                Path(jhat).name,
                reason,
                raw,
            )
            continue
        logger.info(
            'Abs QA redo %s (%s) -> %s',
            Path(jhat).name,
            reason,
            Path(catalog_path).name,
        )
        remove_jhat_for_realign(jhat)
        cals.append(str(cal))
    report: dict[str, Any] = {
        'n_selected': len(jhat_reasons),
        'n_queued': len(cals),
        'n_failures': 0,
    }
    if not cals:
        return report
    report['n_failures'] = int(
        align_to_mosaic(
            str(catalog_path),
            cals,
            str(jhat_outdir),
            guess_offset=guess_offset,
            verbose=verbose,
            ncores=ncores,
        )
    )
    return report


def score_visit_abs_hub(
    table: Table,
    visit_id: Any,
    align_filter: str,
    *,
    footprint_area: float = 0.0,
) -> float:
    """
    Score a visit as an absolute-alignment hub (higher is better).

    Prefers deep F200W/F150W alignment filters with many frames / exposure
    time over merely large footprints (which can seed a weak Gaia solution).
    """
    vis = table[table['visit'] == visit_id]
    if len(vis) == 0:
        return -1.0
    filt = str(align_filter or '').lower()
    rank = _hub_filter_rank(filt)
    align_rows = vis[np.array([str(f).lower() == filt for f in vis['filter']])]
    if len(align_rows) == 0:
        align_rows = vis
    n_frames = int(len(align_rows))
    try:
        exptime = float(np.nansum(np.asarray(align_rows['exptime'], dtype=float)))
    except Exception:
        exptime = 0.0
    area = max(0.0, float(footprint_area))
    # Rank dominates; then depth; area is a weak tie-break only.
    return rank * 1.0e9 + n_frames * 1.0e6 + exptime * 1.0e3 + area


def rank_visits_as_abs_hubs(
    table: Table,
    visit_filter: dict[Any, str],
    visit_geoms: dict[Any, shapely.geometry.base.BaseGeometry],
) -> list[Any]:
    """
    Return visit IDs sorted best-first as absolute Gaia / catalog hubs.
    """
    scored: list[tuple[float, Any]] = []
    for vid in list(visit_geoms.keys()):
        filt = visit_filter.get(vid, '')
        area = float(visit_geoms[vid].area) if vid in visit_geoms else 0.0
        scored.append(
            (
                score_visit_abs_hub(
                    table, vid, str(filt), footprint_area=area
                ),
                vid,
            )
        )
    scored.sort(key=lambda t: (-t[0], str(t[1])))
    return [vid for _, vid in scored]


def mosaic_abs_quality(
    mosaic_path: str | Path,
    *,
    max_mas: float = HUB_GAIA_MAX_MAS,
) -> dict[str, Any]:
    """
    Summarize whether a visit mosaic is usable as an absolute hub seed.

    Uses ``GADISPM`` / ``JWDISPM`` (arcsec in header) and ``GANCAL`` /
    ``JWNCAL``. Soft-fail products have ``n_calibrators == 0``.
    """
    path = Path(mosaic_path)
    disp_mas, n_cal = read_dispersion_mas(str(path))
    n_cal_i = int(n_cal) if n_cal is not None else 0
    ok = (
        path.is_file()
        and n_cal_i > 0
        and disp_mas is not None
        and math.isfinite(float(disp_mas))
        and float(disp_mas) <= float(max_mas)
    )
    return {
        'ok': bool(ok),
        'path': str(path),
        'dispersion_mas': None if disp_mas is None else float(disp_mas),
        'n_calibrators': n_cal_i,
        'max_mas': float(max_mas),
    }


def retie_jwst_jhat_to_abs_ref(
    jhat_paths: Sequence[str | Path],
    abs_ref: str | Path,
    *,
    max_residual_arcsec: float = HUB_ABS_RETIE_TOL_ARCSEC,
    max_search_arcsec: float = HUB_ABS_SEARCH_ARCSEC,
    max_apply_arcsec: float = HUB_ABS_RETIE_MAX_APPLY_ARCSEC,
    min_peak: int = HUB_ABS_RETIE_MIN_PEAK,
    bin_arcsec: float = 0.05,
    jwst_only: bool = True,
    ncores: int = 1,
) -> dict[str, Any]:
    """
    Shift JWST JHAT products onto a common absolute hub mosaic / frame.

    Filters out HST products when *jwst_only* is True (safe for mixed legacy
    ``jhat/`` trees). Caps applied CRVAL shifts at *max_apply_arcsec* so
    multimodal false peaks (~1-2") are not written; larger offsets should be
    fixed by :func:`realign_jwst_jhats_to_catalog` instead.
    """
    from st123.stages.mosaic.mosaic import harmonize_jwst_frames_to_ref

    paths: list[Path] = []
    n_skipped_non_jwst = 0
    for path in jhat_paths:
        fp = Path(path)
        if jwst_only and not is_jwst_science_jhat(fp):
            n_skipped_non_jwst += 1
            continue
        paths.append(fp)
    report = harmonize_jwst_frames_to_ref(
        paths,
        abs_ref,
        max_residual_arcsec=float(max_residual_arcsec),
        max_search_arcsec=float(max_search_arcsec),
        bin_arcsec=float(bin_arcsec),
        max_apply_arcsec=float(max_apply_arcsec),
        min_peak=int(min_peak),
        ncores=max(1, int(ncores)),
    )
    report['n_skipped_non_jwst'] = int(n_skipped_non_jwst)
    return report


def pick_visit(
    align_pgon: shapely.geometry.base.BaseGeometry | None,
    visit_geoms: dict[Any, shapely.geometry.base.BaseGeometry],
    visit_filter: dict[Any, str],
    *,
    hub_order: Sequence[Any] | None = None,
) -> tuple[Any, float]:
    """
    Pick the next visit to align against the current alignment footprint.

    Parameters
    ----------
    align_pgon : shapely geometry or None
        Footprint already aligned; ``None`` selects the absolute hub visit
        (via *hub_order* when provided, else largest broadband footprint).
    visit_geoms : dict
        Visit identifier mapped to a shapely geometry (mutated when
        ``align_pgon`` is ``None``).
    visit_filter : dict
        Visit identifier mapped to its alignment filter.
    hub_order : sequence, optional
        Best-first absolute-hub visit IDs from :func:`rank_visits_as_abs_hubs`.

    Returns
    -------
    tuple
        ``(visit_id, overlap_fraction)``.
    """
    if align_pgon is None:
        if hub_order:
            for visit_id in hub_order:
                if visit_id in visit_geoms:
                    return visit_id, float(visit_geoms[visit_id].area)
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


def jwst_phot(
    phot_img: str,
    photfilename: str | None = None,
) -> tuple[Table, str]:
    """
    Run JHAT ``jwst_photclass`` photometry on an image.

    Parameters
    ----------
    phot_img : str
        Image to photometer.
    photfilename : str, optional
        Destination catalog path. Defaults to ``<image>.phot.txt`` beside the
        FITS file. Parallel jobs should pass a unique path so workers never
        overwrite each other's catalogs.

    Returns
    -------
    tuple
        ``(refcat, photfilename)`` where ``refcat`` is an
        :class:`astropy.table.Table`.
    """
    patch_jwst_for_photutils3()
    photometry = jwst_photclass()
    if photfilename is None:
        photfilename = phot_img.replace('.fits', '.phot.txt')
    else:
        photfilename = str(Path(photfilename).expanduser().resolve())
        Path(photfilename).parent.mkdir(parents=True, exist_ok=True)
    photometry.run_phot(
        imagename=phot_img,
        photfilename=photfilename,
        overwrite=True,
        ee_radius=70,
    )
    refcat = Table.read(photfilename, format='ascii')
    return refcat, photfilename


def is_level3_i2d(image: str) -> bool:
    """
    Return True when ``image`` is a Level-3 / coadd ``*i2d*.fits`` product.

    These frames need :func:`fix_phot` because JHAT GWCS and FITS WCS can
    disagree on pixel<->sky transforms for the same sky coordinate.
    """
    name = Path(image).name.lower()
    return name.endswith(('.fits', '.fits.gz')) and 'i2d' in name


def fix_phot(mosaic: str, *, workdir: str | None = None) -> str:
    """
    Photometer an i2d mosaic and rewrite sky coords with the FITS SCI WCS.

    JHAT's native GWCS transform can disagree with the Level-3 FITS WCS for
    the same sky coordinate. Detection still uses :func:`jwst_phot`, but RA/Dec
    are recomputed from pixel positions via ``astropy.wcs.WCS`` on the SCI
    header so reference catalogs stay consistent with the coadd WCS used for
    alignment inspection.

    Parameters
    ----------
    mosaic : str
        Mosaic / coadd ``*i2d*.fits`` file name.
    workdir : str, optional
        Directory for photometry outputs. Defaults to the mosaic's parent.
        Parallel REFERENCE workers must pass a unique directory so coadd
        sidecars are never shared across processes.

    Returns
    -------
    str
        Path to the corrected photometry catalog (``*i2d.corr*.phot.txt``).
    """
    mosaic = str(Path(mosaic).expanduser().resolve())
    if workdir is None:
        workdir = str(Path(mosaic).parent)
    else:
        workdir = resolve_outdir(workdir)
    stem = Path(mosaic).stem  # e.g. coadd_0_0_f150w2_i2d
    raw = os.path.join(workdir, f'{stem}.phot.txt')
    refcat, _photfile = jwst_phot(mosaic, photfilename=raw)
    w = as_datamodel(mosaic).sci_wcs()
    sky_xy = w.all_pix2world(refcat['x'], refcat['y'], 0)
    refcat['ra'], refcat['dec'] = np.array(sky_xy[0]), np.array(sky_xy[1])
    corr_stem = stem.replace('i2d', 'i2d.corr') if 'i2d' in stem.lower() else f'{stem}.corr'
    corrected = os.path.join(workdir, f'{corr_stem}.phot.txt')
    refcat.write(corrected, format='ascii', overwrite=True)
    return corrected


def generate_level3_mosaic(
    inputfiles: Sequence[str],
    outdir: str,
) -> str:
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

    mosaic_path = f'{outdir_level3}/{filter_name}_i2d.fits'
    logger.info(
        'Image3Pipeline.run(%s) -> %s (resample + source catalog)...',
        asn_file,
        mosaic_path,
    )
    with capture_output():
        image3.run(asn_file)
    logger.info('Image3Pipeline finished: %s', mosaic_path)
    return mosaic_path


def create_alignment_mosaic(
    filter_table: dict[str, Table],
    outdir: str,
    align_filter: str | None = None,
    align_to: str = 'gaia',
    ncores: int = 10,
    Nbright: int = 800,
) -> tuple[str, tuple[float, float], int]:
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
        ``(aligned_mosaic, guess_offset, n_failures)``.
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

    logger.info(
        'Visit mosaic: filter=%s, %d frame(s), align_to=%s, outdir=%s',
        align_filter or '(auto)',
        len(align_table),
        'Gaia' if align_to == 'gaia' else align_to,
        outdir,
    )

    # Alignment groups pick the correct reference image for each module.
    logger.info('Building alignment groups / reference images...')
    align_table = add_alignment_groups(align_table)
    repo = str(_resolve_repo_root())
    jobs = []
    n_skip = 0
    n_redo = 0
    for row in align_table:
        image = row['image']
        jhat_dest = os.path.join(
            outdir, os.path.basename(image.replace('cal.fits', 'jhat.fits'))
        )
        need, reason = jhat_product_needs_realign(jhat_dest)
        ref_image = row['ref_img']
        phot_sidecar = ref_image.replace('.fits', '.phot.txt')
        if not need and os.path.exists(jhat_dest) and os.path.exists(phot_sidecar):
            # Cheap stale check against an existing parent catalog only.
            need, reason = jhat_product_needs_realign(
                jhat_dest, catalog_path=phot_sidecar
            )
        if not need:
            n_skip += 1
            continue
        if is_level3_i2d(ref_image):
            # Always rebuild i2d catalogs via fix_phot (GWCS vs FITS WCS bug).
            photfilename = fix_phot(ref_image)
        elif not os.path.exists(phot_sidecar):
            _, photfilename = jwst_phot(ref_image)
        else:
            photfilename = phot_sidecar
        if os.path.exists(jhat_dest):
            n_redo += 1
            logger.info(
                'Replacing relative JHAT %s (%s)',
                os.path.basename(jhat_dest),
                reason,
            )
            remove_jhat_for_realign(jhat_dest)
        jobs.append(
            _build_visit_jhat_job(
                image=image,
                outdir=outdir,
                photfilename=photfilename,
                repo=repo,
                filter=str(row.get('filter', '')),
            )
        )
    logger.info(
        'Relative JHAT: %d to run, %d already present, %d redo (workers=%d)',
        len(jobs),
        n_skip,
        n_redo,
        max(1, int(ncores)),
    )
    results = _run_jobs_parallel(
        jobs,
        run_visit_align_job,
        workers=ncores,
        label='VISIT',
        on_result=lambda result: logger.info(_format_worker_done(result)),
    )
    n_failures = _count_worker_failures(results)

    # Create an i2d mosaic from the relatively aligned images.
    inputfiles = [
        os.path.join(outdir, os.path.basename(i.replace('cal.fits', 'jhat.fits')))
        for i in align_table['image']
    ]
    logger.info(
        'Starting Level-3 Image3 mosaic from %d JHAT frame(s) '
        '(JWST pipeline stdout suppressed; this often takes several minutes)...',
        len(inputfiles),
    )
    mosaic_name = generate_level3_mosaic(inputfiles, outdir)
    logger.info('Level-3 mosaic written: %s', mosaic_name)

    logger.info(
        'Aligning Level-3 mosaic to %s (JHAT; can take several minutes)...',
        'Gaia' if align_to == 'gaia' else align_to,
    )
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
    with as_datamodel(aligned_mosaic).open(mode='update') as filehandle:
        filehandle[0].header['JHATX'] = guess_offset[0]
        filehandle[0].header['JHATY'] = guess_offset[1]

    return aligned_mosaic, guess_offset, n_failures


# Gaia catalog helpers live in :mod:`st123.stages.alignment.gaia_catalog` (Vizier
# default) so HST scoring does not import the JWST stack just to query Gaia.
from st123.stages.alignment.gaia_catalog import (  # noqa: E402
    query_gaia,
)


def expand_mask(
    mask: np.ndarray,
    size: int = 40,
    mask_shape: str = 'square',
) -> np.ndarray:
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
    im = as_datamodel(filename).open()
    dq_mask = copy.deepcopy(im['DQ'].data)

    flag_sat = (dq_mask != 1) & (dq_mask != 2) & (dq_mask != 3)
    dq_mask[flag_sat] = 10
    dq_mask[~flag_sat] = 1

    expmask = expand_mask(dq_mask, mask_shape=mask_shape)
    expmask = np.where(expmask == 10, False, True)

    hdr = im[3].header.copy()
    hdr['EXTNAME'] = 'BIN_DQ'
    image_hdu = fits.ImageHDU(data=expmask.astype(np.uint8), name='BIN_DQ', header=hdr)
    im.insert(8, image_hdu)

    if outfile is None:
        outfile = filename.replace('.fits', '_masked.fits')
    im.writeto(outfile, overwrite=True)
    im.close()
    return outfile


def calc_dispersion(
    ref_table: Table,
    photfile: str,
    w: WCS | bool = False,
    dist_limit: float = 1,
    sig: float = 2,
    plot: bool = False,
) -> tuple[float, float, float, int]:
    """
    Measure the dispersion between a photometry catalog and a reference table.

    Parameters
    ----------
    ref_table : astropy.table.Table
        Reference photometry table with ``ra`` / ``dec`` columns.
    photfile : str
        Photometry catalog to compare.
    w : astropy.wcs.WCS or bool, optional
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
    mean : float
        Mean matched separation in arcsec (sigma-clipped).
    median : float
        Median matched separation in arcsec (sigma-clipped).
    std : float
        Standard deviation of matched separations in arcsec (sigma-clipped).
    n_calibrators : int
        Number of matched pairs that survive the sigma-clip and contribute to
        the reported dispersion. This is the value stored as ``JWNCAL``.
    """
    phot_df = pd.read_csv(photfile, sep=r'\s+')

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
    if len(dist_matched_df) == 0:
        return float('nan'), float('nan'), float('nan'), 0

    d2d = np.asarray(dist_matched_df['d2d'], dtype=float)
    d2d_sq = d2d ** 2
    # Preserve historical dispersion statistics (sigma-clip on d2d**2).
    clip_mean, clip_median, clip_std = sigma_clipped_stats(
        d2d_sq, sigma_lower=None, sigma_upper=sig
    )
    mean_dispersion = float(np.sqrt(clip_mean))
    median_dispersion = float(np.sqrt(clip_median))
    std_dispersion = float(np.sqrt(clip_std))

    # Calibrator count = matched pairs retained by an equivalent residual cut
    # on d2d. Floor the scatter so near-perfect alignments (machine-zero
    # residuals) are not spuriously rejected by sigma_clip.
    _mn, med_d2d, std_d2d = sigma_clipped_stats(d2d, sigma_lower=None, sigma_upper=sig)
    std_floor = max(float(std_d2d), 1e-9)  # arcsec; ~0.001 mas
    n_calibrators = int(np.count_nonzero(d2d <= float(med_d2d) + sig * std_floor))
    if n_calibrators == 0:
        # Degenerate clip - fall back to the raw in-radius match count.
        n_calibrators = int(len(d2d))

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

    return mean_dispersion, median_dispersion, std_dispersion, n_calibrators


def jwst_dispersion(align_image, outdir, photfile=None, gaia=False, plot=False, sig=2):
    """
    Measure dispersion before and after JHAT alignment and store it in headers.

    Writes ``JWDISPM`` / ``JWDISPD`` / ``JWDISPS`` together with ``JWNCAL``
    (clipped match count used for the final dispersion) in one header update
    so parallel workers cannot observe a JHAT with dispersion but a stale or
    missing calibrator count.

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

    disp_in_mean, disp_in_median, _, _ = calc_dispersion(
        refcat,
        aligned_image.replace('_jhat.fits', '.phot.txt'),
        dist_limit=0.5,
        sig=sig,
        plot=plot,
    )
    logger.info(f'Initial mean dispersion: {disp_in_mean * 1000} mas')
    logger.info(f'Initial median dispersion: {disp_in_median * 1000} mas')

    os.rename(aligned_image, temp_cal_name)
    _align_cat, align_photfile = jwst_phot(temp_cal_name)
    wcs_in = wcs.WCS(as_datamodel(phot_image).header(1)) if phot_image else False
    disp_fn_mean, disp_fn_median, disp_fn_std, n_calibrators = calc_dispersion(
        refcat, align_photfile, w=wcs_in, sig=sig, dist_limit=0.5, plot=plot
    )
    logger.info(f'Final mean dispersion: {disp_fn_mean * 1000} mas')
    logger.info(f'Final median dispersion: {disp_fn_median * 1000} mas')
    logger.info(f'Final n_calibrators (JWNCAL): {n_calibrators}')
    os.rename(temp_cal_name, aligned_image)

    with as_datamodel(aligned_image).open(mode='update') as filehandle:
        hdr = filehandle[0].header
        if gaia:
            hdr['GADISPM'] = disp_fn_mean
            hdr['GADISPD'] = disp_fn_median
            hdr['GADISPS'] = disp_fn_std
            hdr['GANCAL'] = (
                int(n_calibrators),
                'N calibrators for GADISPM / align solution',
            )
        else:
            hdr['JWDISPM'] = disp_fn_mean
            hdr['JWDISPD'] = disp_fn_median
            hdr['JWDISPS'] = disp_fn_std
            hdr['JWNCAL'] = (
                int(n_calibrators),
                'N calibrators for JWDISPM / align solution',
            )
            hdr['JWCAT'] = os.path.basename(photfile)

    from st123.datamodels.jwst import sanitize_jwst_l2

    sanitize_jwst_l2(aligned_image, materialize_headers=True)
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
    sci_hdr = copy.copy(wcs.WCS(as_datamodel(align_image).header(1)))
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
            _, disp, _, _ = calc_dispersion(
                ref_table, align_photfile, w=in_wcs, dist_limit=1, sig=sig, plot=False
            )
            off.append(disp)
            xshift.append(xs)
            yshift.append(ys)

    best_x, best_y = -xshift[np.argmin(off)], -yshift[np.argmin(off)]
    logger.info(f'Best guess for {align_image}: ({best_x}, {best_y})')

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


def assess_field_brightness(
    align_image: str,
    *,
    bright_median: float = 80.0,
    bright_p99: float = 500.0,
    bright_frac: float = 0.20,
) -> dict[str, Any]:
    """
    Estimate whether a JWST/MIRI SCI frame is extremely bright or crowded.

    Used to prefer a stricter / brighter-source JHAT retry before the existing
    relaxed-parameter fallback (which otherwise adds faint PAH / arm structure
    as false calibrators in nuclei like M82 F770W).

    Parameters
    ----------
    align_image : str
        Path to a ``*_cal.fits`` (or similar) with a SCI extension.
    bright_median, bright_p99 : float
        SCI thresholds (MJy/sr-like native units) flagging a bright field.
    bright_frac : float
        Minimum fraction of finite pixels above ``max(50, 5*median)`` to flag
        extended bright structure.

    Returns
    -------
    dict
        ``bright`` (bool) plus diagnostic ``median``, ``p99``, ``hot_frac``.
    """
    try:
        with as_datamodel(align_image).open(memmap=True) as hdul:
            sci = np.asarray(hdul['SCI'].data, dtype=np.float64)
    except Exception:
        return {
            'bright': False,
            'median': float('nan'),
            'p99': float('nan'),
            'hot_frac': float('nan'),
        }
    finite = np.isfinite(sci)
    if finite.sum() < 1000:
        return {
            'bright': False,
            'median': float('nan'),
            'p99': float('nan'),
            'hot_frac': float('nan'),
        }
    vals = sci[finite]
    med = float(np.median(vals))
    p99 = float(np.percentile(vals, 99.0))
    thr = max(50.0, 5.0 * max(med, 0.0))
    hot_frac = float(np.mean(vals > thr))
    bright = bool(
        med >= float(bright_median)
        or p99 >= float(bright_p99)
        or hot_frac >= float(bright_frac)
    )
    return {
        'bright': bright,
        'median': med,
        'p99': p99,
        'hot_frac': hot_frac,
    }


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

    # Always mediate JHAT stdout/stderr through logging (DEBUG -> log file;
    # console only when --verbose raises the stream handler to DEBUG).
    with capture_output():
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
    Align an image with JHAT, retrying when the first solution is poor.

    Attempts, in order:

    1. Strict parameters (always first so globally bright galaxies do not
       starve calibrators off-nucleus).
    2. Crowded/bright fallback when the strict residual is still high -
       fewer, brighter, high-SNR calibrators (before relaxing, which would
       add faint PAH / arm structure).
    3. Relaxed parameters.
    4. Relaxed parameters seeded with a grid-searched pixel shift.

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
        Optional overrides merged into the JHAT parameter dictionaries
        (e.g. tighter ``objmag_lim`` / ``sharpness_lim`` for F770W).

    Returns
    -------
    tuple
        ``(xshift, yshift)`` guess offset applied to the accepted solution.
    """
    logger.info(
        f'Aligning {os.path.basename(align_image)} to '
        f"{'Gaia' if gaia else photfilename.replace('.phot.txt', '.fits')}"
    )
    field = assess_field_brightness(align_image)
    if field.get('bright'):
        logger.info(
            'Bright/crowded SCI detected for %s '
            '(median=%.3g p99=%.3g hot_frac=%.3f); prefer bright-calibrator cuts',
            os.path.basename(align_image),
            field.get('median', float('nan')),
            field.get('p99', float('nan')),
            field.get('hot_frac', float('nan')),
        )

    def _base_params(kind: str) -> dict:
        if kind == 'crowded' and not gaia:
            p = dict(crowded_jwst_params)
        elif kind == 'relaxed':
            p = dict(relaxed_gaia_params if gaia else relaxed_jwst_params)
        else:
            p = dict(strict_gaia_params if gaia else strict_jwst_params)
        if jhat_params:
            # Crowded retry keeps its brighter objmag / SNR floor unless the
            # caller override is itself stricter on those keys.
            if kind == 'crowded':
                for key, val in jhat_params.items():
                    if key in ('objmag_lim', 'SNR_min', 'find_stars_threshold', 'd2d_max'):
                        continue
                    p[key] = val
            else:
                p.update(jhat_params)
        if plot:
            p['showplots'] = 2
        return p

    try:
        wv = float(os.path.basename(align_image).split('_jhat')[0][1:4])
        factor = 2 if wv > 220 else 1
    except Exception:
        factor = 2 if 'long' in align_image else 1

    try:
        pixscale = np.abs(as_datamodel(align_image).keyword('CDELT1', ext=1) * 3600)
    except Exception:
        pixscale = np.abs(as_datamodel(align_image).keyword('CD1_2', ext=1) * 3600)
    pixscale = 0.031 if pixscale < 0.032 else 0.062
    retry_pix = min(float(soft_fail_pix), 1.0)

    # Initialize so failed JHAT/dispersion attempts cannot raise UnboundLocalError.
    disp_in_mu = disp_in_med = disp_fn_mu = disp_fn_med = 99.99
    guess_offset = (0, 0)

    def _attempt(
        kind: str,
        *,
        nbright: int,
        x0: float | None = None,
        y0: float | None = None,
        sig_clip: float | None = None,
        track_guess: bool = False,
    ):
        nonlocal disp_in_mu, disp_in_med, disp_fn_mu, disp_fn_med, guess_offset
        params = _base_params(kind)
        xs = float(xshift if x0 is None else x0)
        ys = float(yshift if y0 is None else y0)
        run_jhat(
            align_image=align_image,
            outdir=outdir,
            params=params,
            gaia=gaia,
            photfilename=photfilename,
            xshift=xs,
            yshift=ys,
            Nbright=nbright,
            verbose=verbose,
        )
        disp_in_mu, disp_in_med, disp_fn_mu, disp_fn_med = jwst_dispersion(
            align_image=align_image,
            outdir=outdir,
            photfile=photfilename,
            gaia=gaia,
            plot=plot,
            sig=sig if sig_clip is None else sig_clip,
        )
        guess_offset = (xs * factor, ys * factor) if track_guess else (0, 0)

    # 1) Strict first (consistent across outer / nucleus fields)
    try:
        _attempt('strict', nbright=int(Nbright))
    except Exception:
        logger.error(traceback.format_exc())
        disp_fn_med = 99.99

    # 2) Crowded/bright fallback before relaxing - only when needed.
    # Use for bright/crowded SCI *or* any high residual; never as the default
    # first attempt on globally bright galaxies.
    if disp_fn_med / pixscale > retry_pix and not gaia:
        why = (
            'bright/crowded SCI'
            if field.get('bright')
            else f'strict residual {float(disp_fn_med) * 1000.0:.1f} mas'
        )
        logger.info(
            'Retrying %s with crowded/bright calibrator cuts (%s)',
            os.path.basename(align_image),
            why,
        )
        try:
            _attempt(
                'crowded',
                nbright=min(int(Nbright), int(CROWDED_JHAT_NBRIGHT)),
                sig_clip=1,
            )
        except Exception:
            logger.error(traceback.format_exc())
            disp_fn_med = 99.99

    # 3) Relaxed params when crowded/bright still fails
    if disp_fn_med / pixscale > retry_pix:
        logger.info(
            'Retrying %s with relaxed JHAT parameters',
            os.path.basename(align_image),
        )
        try:
            _attempt('relaxed', nbright=int(Nbright), sig_clip=1)
        except Exception:
            logger.error(traceback.format_exc())
            disp_fn_med = 99.99

    # 4) Guess-shift + relaxed
    if disp_fn_med / pixscale > retry_pix:
        if gaia:
            ref_table = query_gaia(align_image)
        else:
            ref_table = Table.read(photfilename, format='ascii')
        xsh, ysh = guess_shift(
            align_image, ref_table, radius_px=50, res=4, sig=1, plot=plot
        )
        try:
            _attempt(
                'relaxed',
                nbright=int(Nbright),
                x0=float(xsh),
                y0=float(ysh),
                sig_clip=1,
                track_guess=True,
            )
        except Exception:
            logger.error(traceback.format_exc())
            disp_fn_med = 99.99

    logger.info(
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
        logger.info(
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

        safe_copy(align_image, jhat_image)
        with as_datamodel(jhat_image).open(mode='update') as filehandle:
            d_mu = finite_arcsec(disp_in_mu)
            d_med = finite_arcsec(disp_in_med)
            if gaia:
                filehandle[0].header['GADISPM'] = d_mu
                filehandle[0].header['GADISPD'] = d_med
                filehandle[0].header['GADISPS'] = d_med
                filehandle[0].header['GANCAL'] = (
                    0,
                    'N calibrators for GADISPM / align solution',
                )
            else:
                filehandle[0].header['JWDISPM'] = d_mu
                filehandle[0].header['JWDISPD'] = d_med
                filehandle[0].header['JWDISPS'] = d_med
                # Soft-fail: no alignment solution was accepted, so no
                # calibrators contributed to a derived WCS / final dispersion.
                filehandle[0].header['JWNCAL'] = (
                    0,
                    'N calibrators for JWDISPM / align solution',
                )
        from st123.datamodels.jwst import sanitize_jwst_l2

        sanitize_jwst_l2(jhat_image, materialize_headers=True)
    elif disp_fn_med / pixscale > retry_pix:
        logger.info(
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
    flt = as_datamodel(mosaic_name).keyword('FILTER', ext=0)
    pupil = as_datamodel(mosaic_name).keyword('PUPIL', ext=0)

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
        Forwarded to worker crash formatting (JHAT stays muted).
    ncores : int, optional
        Pool size.

    Returns
    -------
    int
        Number of failed worker jobs.
    """
    repo = str(_resolve_repo_root())
    jobs = []
    n_skip = 0
    n_redo = 0
    for im in cal_images:
        jhat_dest = os.path.join(
            outdir, os.path.basename(im.replace('cal.fits', 'jhat.fits'))
        )
        need, reason = jhat_product_needs_realign(
            jhat_dest, catalog_path=mosaic_photfile
        )
        if not need:
            n_skip += 1
            continue
        if os.path.exists(jhat_dest):
            n_redo += 1
            logger.info(
                'Replacing mosaic JHAT %s (%s)',
                os.path.basename(jhat_dest),
                reason,
            )
            remove_jhat_for_realign(jhat_dest)
        xs, ys = guess_offset
        if 'long' in im:
            xs = xs * 0.5
            ys = ys * 0.5
        jobs.append(
            _build_visit_jhat_job(
                image=im,
                outdir=outdir,
                photfilename=mosaic_photfile,
                repo=repo,
                xshift=xs,
                yshift=ys,
                verbose=bool(verbose),
            )
        )
    logger.info(
        'Align-to-mosaic JHAT: %d to run, %d already present, %d redo '
        '(workers=%d)',
        len(jobs),
        n_skip,
        n_redo,
        max(1, int(ncores)),
    )
    results = _run_jobs_parallel(
        jobs,
        run_visit_align_job,
        workers=ncores,
        label='VISIT',
        on_result=lambda result: logger.info(_format_worker_done(result)),
    )
    return _count_worker_failures(results)


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
    with as_datamodel(image).open() as hdul:
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
    with as_datamodel(image).open() as hdul:
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


def safe_copy(src: str, dest: str) -> None:
    """
    Copy ``src`` -> ``dest``, tolerating EPERM on metadata/utime.

    Shared ``alignment_output`` trees are often owned by another user: ``copy2``
    can fail on utime/chmod even when the content is group-writable.
    """
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        try:
            dest_path.unlink()
        except OSError:
            pass
    try:
        shutil.copy2(src, dest)
    except OSError:
        shutil.copyfile(src, dest)


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
    safe_copy(photfile, dest)
    logger.info(f'Copied reference catalog -> {dest}')
    return dest


def build_ref_catalog(
    image: str,
    outdir: str,
    photfile: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """
    Build or stage a JHAT-compatible reference photometry catalog.

    Preference order for new catalogs:

    1. :func:`fix_phot` for Level-3 / coadd ``*i2d*`` frames (even when ASDF /
       GWCS is present), so sky coordinates use the FITS SCI WCS.
    2. :func:`jwst_phot` for Level-2 ``*_cal.fits`` (and other GWCS products).
    3. photutils DAOStarFinder when neither path applies.

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
        logger.info(f'Using existing reference catalog: {photfile}')
        return stage_photfile(photfile, outdir)

    dest_name = Path(phot_catalog_path(image, outdir)).name
    # Prefer a reusable cache when present. For i2d, only reuse catalogs that
    # look like fix_phot products (enough sources); tiny/corrupt caches must
    # not short-circuit a rebuild.
    if cache_dir is not None:
        cache_path = Path(cache_dir).expanduser().resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        cached = cache_path / dest_name
        if cached.is_file():
            reuse = True
            if is_level3_i2d(image):
                try:
                    n_cached = len(read_jhat_phot_table(str(cached)))
                except Exception:
                    n_cached = 0
                reuse = n_cached >= 50
                if not reuse:
                    logger.info(
                        f'Ignoring thin/corrupt i2d cache ({n_cached} rows): {cached}'
                    )
            if reuse:
                logger.info(f'Reusing cached reference catalog: {cached}')
                return stage_photfile(str(cached), outdir, dest_name=dest_name)

    dest = phot_catalog_path(image, outdir)
    logger.info(f'Running photometry on reference: {image}')
    # Always write JHAT intermediates under this worker's outdir - never beside
    # shared coadds under reduction/reference/ (parallel workers race there).
    workdir = resolve_outdir(os.path.join(outdir, '_phot_work', Path(image).stem))

    if is_level3_i2d(image):
        try:
            logger.info('  level-3/i2d -> fix_phot (FITS WCS sky coords)')
            src = fix_phot(image, workdir=workdir)
            staged = stage_photfile(src, outdir, dest_name=Path(dest).name)
        except Exception as exc:
            logger.info(f'  fix_phot failed ({exc}); falling back to photutils')
            staged = photutils_phot(image, dest)
    elif has_jwst_gwcs(image):
        logger.info('  detected JWST ASDF/GWCS -> jwst_phot')
        raw = os.path.join(workdir, f'{Path(image).stem}.phot.txt')
        _, src = jwst_phot(image, photfilename=raw)
        staged = stage_photfile(src, outdir, dest_name=Path(dest).name)
    else:
        logger.info('  falling back to photutils DAOStarFinder')
        staged = photutils_phot(image, dest)

    if cache_dir is not None:
        cache_path = Path(cache_dir).expanduser().resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_dest = cache_path / Path(staged).name
        try:
            safe_copy(staged, str(cache_dest))
            logger.info(f'Cached reference catalog -> {cache_dest}')
        except OSError as exc:
            # Shared caches owned by another user often reject overwrite.
            logger.warning(
                f'Could not update phot cache {cache_dest}: {exc}; '
                f'continuing with staged catalog {staged}'
            )

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
    missing = [name for name in required if name not in lower]
    if missing:
        raise ValueError(
            f'Photometry table missing required column(s) {missing!r}; '
            f'have {list(table.colnames)!r}. This often means a parallel '
            f'worker read a partially written / corrupt shared catalog.'
        )

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

    from st123.stages.mosaic.image_overlap import MirIFootprint

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
        logger.warning(
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
    logger.info(
        f'Clipped master catalog to MIRI footprint: '
        f'{len(table)} -> {len(clipped)} sources'
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
            logger.info(f'  + {len(tables[-1])} sources from {ref}')
        except Exception as exc:
            logger.warning(f'  WARNING: photometry failed for {ref}: {exc}')

    if not tables:
        raise RuntimeError('No reference photometry catalogs could be built')

    master = merge_phot_catalogs(tables, match_radius_arcsec=match_radius_arcsec)
    logger.info(
        f'Merged {len(tables)} reference catalog(s) -> {len(master)} unique sources '
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
    logger.info(f'Wrote master reference catalog -> {dest} ({len(master)} sources)')
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
            logger.info(f'Saved diagnostic plot: {out}')
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
        ``n_calibrators`` is ``JWNCAL`` / ``GANCAL`` when present and
        plausible; otherwise recomputed from this frame's dispersion match.
    """
    with as_datamodel(jhat_image).open() as hdul:
        hdr = hdul[0].header
        disp = hdr.get('JWDISPM', hdr.get('GADISPM'))
        ncal = hdr.get('JWNCAL', hdr.get('GANCAL'))
    disp_mas = float(disp) * 1000.0 if disp is not None else None
    n_cal: int | None
    try:
        n_cal = int(ncal) if ncal is not None else None
    except (TypeError, ValueError):
        n_cal = None
    if n_cal is None or not jwncal_is_plausible(n_cal, jhat_image):
        n_recomputed = count_alignment_calibrators(jhat_image)
        if n_recomputed is not None:
            n_cal = n_recomputed
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
    with as_datamodel(jhat_image).open() as hdul:
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
        logger.info(f'  calibrator morph/mag cut: kept {n_morph}/{len(jhat_df)} MIRI sources')
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
        logger.info(
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
            logger.info(
                f'  hard residual cut ({max_residual_arcsec * 1000:.1f} mas): '
                f'{int(keep.sum())} -> {int((keep & hard).sum())}'
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
    logger.info(
        f'Iterative refinement starting from dispersion={disp_mas} mas, '
        f'n_calibrators={n_cal}, xshift={xshift:.3f}, yshift={yshift:.3f}'
    )

    current_ref = ref_phot
    for it in range(1, max_iter + 1):
        if not os.path.exists(align_phot):
            logger.info(
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
            logger.info(f'  refine iter {it}: clipping failed ({exc}); stopping')
            break

        if stats['n_ref_kept'] < min_calibrators:
            logger.info(
                f'  refine iter {it}: only {stats["n_ref_kept"]} ref stars left '
                f'(<{min_calibrators}); stopping'
            )
            break

        # Only re-run JHAT when the matched set actually shrank (sigma-clip /
        # hard residual). Do NOT treat "matched subset << full master_ref" as
        # progress - that is always true on the first pass and was forcing a
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
            logger.info(f'  refine iter {it}: no outliers clipped; converged')
            break

        cleaned_path = str(Path(outdir) / f'master_ref_refined_iter{it:02d}.phot.txt')
        write_jhat_phot_table(cleaned, cleaned_path)
        logger.info(
            f'  refine iter {it}: wrote cleaned catalog {cleaned_path} '
            f'({len(cleaned)} stars; match median {stats["median_d2d_mas"]:.2f} mas)'
        )

        # Snapshot current JHAT products so a worse iteration can be rolled back.
        jhat, align_phot, _ = jhat_product_paths(align_image, outdir)
        backup_jhat = jhat + '.refine_bak'
        backup_phot = align_phot + '.refine_bak'
        if os.path.exists(jhat):
            safe_copy(jhat, backup_jhat)
        if os.path.exists(align_phot):
            safe_copy(align_phot, backup_phot)

        # Only seed pixel offsets for F770W-style tight refine (hard residual
        # ceiling). Seeding large F560W XOFFSET/YOFFSET (~50-130 px) into JHAT
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
        logger.info(
            f'  refine iter {it}: dispersion {disp_mas} -> {new_disp} mas '
            f'(n_calibrators={new_ncal})'
        )

        def restore_backup_and_stop(reason: str) -> None:
            logger.info(
                f'  refine iter {it}: {reason}; '
                f'restoring previous JHAT products and stopping'
            )
            if os.path.exists(backup_jhat):
                safe_copy(backup_jhat, jhat)
            if os.path.exists(backup_phot):
                safe_copy(backup_phot, align_phot)
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
                logger.info(
                    f'  refine iter {it}: |Deltadispersion|='
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
                    f'dispersion worsened ({disp_mas:.3f} -> {new_disp:.3f} mas)'
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
            logger.info(f'Final refined reference catalog -> {final_clean}')

    logger.info(
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

    logger.info(f'Output directory: {outdir}')
    logger.info(f'Align image:      {align_image}')
    if jhat_params:
        logger.info(f'JHAT param overrides: {jhat_params}')
    logger.info(
        f'Calibrator knobs: nbright={nbright}, refine_sigma={refine_sigma}, '
        f'dist_limit={refine_dist_limit_arcsec}", '
        f'max_resid={max_residual_arcsec}"'
    )

    if photfile is not None:
        ref_phot = build_ref_catalog(
            refs[0] if refs else align_image, outdir, photfile=photfile
        )
        logger.info(f'Reference catalog: {ref_phot}')
    elif len(refs) > 1:
        logger.info(f'Reference images ({len(refs)}): building master catalog')
        for path in refs:
            logger.info(f'  {path}')
        ref_phot = build_master_ref_catalog(
            refs,
            outdir,
            align_image=align_image,
            cache_dir=cache_dir,
            match_radius_arcsec=match_radius_arcsec,
            clip_to_align_footprint=clip_to_align_footprint,
        )
    elif len(refs) == 1:
        logger.info(f'Reference image:  {refs[0]}')
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
        logger.info(f'Refined dispersion_mas={disp_mas}, n_calibrators={n_cal}')

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
        if not (
            (root / 'st123' / 'stages' / 'alignment' / 'align.py').is_file()
            or (root / 'st123' / 'alignment' / 'align.py').is_file()
            or (root / 'alignment' / 'align.py').is_file()
        ):
            raise FileNotFoundError(f'--repo does not look like st123: {root}')
        return root

    here = Path(__file__).resolve().parent  # <repo>/st123/stages/alignment
    repo_root = here.parents[2]
    marker = Path('st123') / 'stages' / 'alignment' / 'align.py'
    for cand in (repo_root, here.parents[1], Path.cwd()):
        if (cand / marker).is_file() or (
            cand / 'st123' / 'alignment' / 'align.py'
        ).is_file():
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
    """One row of the dataset alignment summary table.

    ``n_calibrators`` is the number of science<->reference matches that survive
    the sigma-clip used for ``dispersion_mas`` / ``JWDISPM`` (header
    ``JWNCAL``). It is never the raw JHAT ``*.refcat.txt`` row count.
    """

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
    with as_datamodel(miri_path).open() as hdul:
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


# Soft-fail sentinel written by align_jwst_image (99.99 arcsec -> mas).
SOFT_FAIL_DISPERSION_MAS: float = 99990.0


def is_soft_fail_dispersion(dispersion_mas: float | str | None) -> bool:
    """
    Return whether a dispersion is the soft-fail / unusable sentinel.

    Parameters
    ----------
    dispersion_mas : float or str or None
        Dispersion in milliarcseconds (or non-numeric placeholder).

    Returns
    -------
    bool
        True when missing, non-finite, or at/above the soft-fail sentinel.
    """
    try:
        value = float(dispersion_mas)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True
    return (not math.isfinite(value)) or value >= SOFT_FAIL_DISPERSION_MAS


def reference_solution_usable(row: AlignmentSummaryRow) -> bool:
    """
    Return whether a REFERENCE row may be used as a provisional MIRI_REL parent.

    Usable means a real JHAT product with a positive calibrator count and a
    non-sentinel finite dispersion. Soft-fail copies (``JWNCAL=0`` /
    ``99990`` mas) are never usable. Meeting ``min_calibrators`` is *not*
    required here - see :func:`reference_solution_keepable` for final SUCCESS.

    Parameters
    ----------
    row : AlignmentSummaryRow
        Row to test.

    Returns
    -------
    bool
        True when the REFERENCE solution may parent MIRI_REL attempts.
    """
    if not isinstance(row.dispersion_mas, float):
        return False
    if is_soft_fail_dispersion(row.dispersion_mas):
        return False
    if not isinstance(row.n_calibrators, int) or int(row.n_calibrators) <= 0:
        return False
    if not row.aligned_path or str(row.aligned_path) == 'NA':
        return False
    return True


def reference_solution_keepable(row: AlignmentSummaryRow) -> bool:
    """
    Return whether a REFERENCE hold may be finalized as SUCCESS.

    Requires :func:`reference_solution_usable` plus either:
    - ``n_calibrators >= min_calibrators`` for the filter, or
    - absolute dispersion at or below the per-filter REFERENCE threshold

    The second clause keeps sparse-but-tight solutions (common at F2100W:
    ``n_cal`` of a few with ~20 mas residuals). High-dispersion tiny-n_cal
    solutions (e.g. 94 mas with ``n_cal=3``) remain non-keepable so they
    finalize as FAILURE when MIRI_REL cannot improve them.

    Parameters
    ----------
    row : AlignmentSummaryRow
        Row to test.

    Returns
    -------
    bool
        True when the REFERENCE solution may be kept as final SUCCESS.
    """
    if not reference_solution_usable(row):
        return False
    min_cal = calibrator_settings_for_filter(row.filter).min_calibrators
    if int(row.n_calibrators) >= int(min_cal):
        return True
    thr = max_reference_dispersion_mas(row.filter)
    if thr is None:
        # F560W: no quality-hold threshold - allow sparse usable REFERENCE.
        return True
    return float(row.dispersion_mas) <= float(thr)


def read_jhat_dispersion_median_mas(jhat_path: str | Path) -> float | None:
    """
    Read ``JWDISPD`` (median dispersion) from a JHAT product, in mas.

    Parameters
    ----------
    jhat_path : str or pathlib.Path
        JHAT product path.

    Returns
    -------
    float or None
        Median dispersion in milliarcseconds, or ``None`` if missing.
    """
    path = Path(jhat_path)
    if not path.is_file():
        return None
    try:
        with as_datamodel(path).open(memmap=True) as hdul:
            val = hdul[0].header.get('JWDISPD', hdul[0].header.get('GADISPD'))
        if val is None:
            return None
        med = float(val) * 1000.0
        if not math.isfinite(med):
            return None
        return med
    except Exception:
        return None


def f770w_reference_gate_dispersion_mas(
    mean_mas: float,
    median_mas: float | None,
) -> tuple[float, str | None]:
    """
    Dispersion value used for F770W REFERENCE quality-hold decisions.

    Uses ``max(mean, median)`` when median is available, and reports a skew
    reason when mean >> median (busy PAH fields inflate the mean).

    Parameters
    ----------
    mean_mas : float
        ``JWDISPM`` in mas.
    median_mas : float or None
        ``JWDISPD`` in mas.

    Returns
    -------
    tuple
        ``(gate_dispersion_mas, skew_reason_or_None)``.
    """
    mean = float(mean_mas)
    if median_mas is None or not math.isfinite(float(median_mas)):
        return mean, None
    med = float(median_mas)
    gate = max(mean, med)
    thr = max_reference_dispersion_mas('F770W') or 50.0
    skew_reason = None
    if (
        med > 0
        and mean > F770W_MEAN_MEDIAN_SKEW_RATIO * med
        and mean > F770W_SKEW_MEAN_FLOOR_FRAC * float(thr)
    ):
        skew_reason = (
            f'F770W mean/median skew mean={mean:.2f} med={med:.2f} mas '
            f'(ratio>{F770W_MEAN_MEDIAN_SKEW_RATIO:.1f})'
        )
    return gate, skew_reason


def _is_reference_quality_hold(row: AlignmentSummaryRow) -> bool:
    """
    Report whether a row is a REFERENCE solution held pending MIRI_REL.

    These rows keep finite metrics and JHAT paths so MIRI_REL can be tried.
    They are omitted from the live alignment summary until MIRI_REL finishes
    (kept as SUCCESS only when improved or when a usable REFERENCE remains
    after a failed MIRI_REL attempt). Exhausted holds with no MIRI_REL parent
    finalize as FAILURE.

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


def provisional_fallback_parents_from_holds(
    filter_name: str,
    row_by_miri: dict[str, AlignmentSummaryRow],
    pending_paths: set[str] | list[str],
) -> list[SuccessfulAlignment]:
    """
    Build provisional MIRI_REL parents from usable PENDING REFERENCE holds.

    When an entire filter wave is quality-held, ``successes`` is empty and
    same-filter relative alignment would otherwise never run. Usable PENDING
    REFERENCE products (positive calibrators, non-sentinel dispersion) are
    exposed as provisional parents so siblings can try MIRI_REL.

    Parameters
    ----------
    filter_name : str
        Current filter wave.
    row_by_miri : dict
        Live summary rows keyed by MIRI path.
    pending_paths : set or list of str
        MIRI paths in the current filter wave.

    Returns
    -------
    list of SuccessfulAlignment
        Provisional parents (not yet final SUCCESS rows).
    """
    filt = str(filter_name).upper()
    out: list[SuccessfulAlignment] = []
    for miri in pending_paths:
        row = row_by_miri.get(miri)
        if row is None or not _is_reference_quality_hold(row):
            continue
        if str(row.filter).upper() != filt:
            continue
        if not reference_solution_usable(row):
            continue
        out.append(
            SuccessfulAlignment(
                miri_path=row.miri_path,
                jhat_path=str(row.aligned_path),
                filter=row.filter,
                wavelength_um=filter_wavelength_um(row.miri_path),
                dispersion_mas=float(row.dispersion_mas),
                relative_dispersion_mas=float(row.dispersion_mas),
                align_mode='REFERENCE',
                original_ref=str(row.original_ref),
                aligned_to=str(row.aligned_to),
                photfile=find_aligned_photfile(str(row.aligned_path)),
                provisional=True,
            )
        )
    return out


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

    with as_datamodel(jhat).open() as hdul:
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

    # JWNCAL must be the clipped match count used for JWDISPM - never the
    # JHAT *.refcat.txt row count (often the full unclipped master catalog).
    # Recompute from this frame's photometry when the header is missing or
    # looks polluted (e.g. JWNCAL == len(master_ref) from an older bug).
    needs_recompute = n_calibrators == 'NA' or (
        isinstance(n_calibrators, int)
        and not jwncal_is_plausible(n_calibrators, aligned_path)
    )
    if needs_recompute:
        n_from_match = count_alignment_calibrators(aligned_path)
        if n_from_match is not None:
            if (
                isinstance(n_calibrators, int)
                and n_calibrators != n_from_match
            ):
                logger.info(
                    f'{Path(aligned_path).name}: replacing implausible '
                    f'JWNCAL={n_calibrators} with dispersion match count '
                    f'n_calibrators={n_from_match}'
                )
            n_calibrators = n_from_match
            # Persist the corrected count so later provenance / summary
            # passes do not re-inherit a polluted header value.
            try:
                with as_datamodel(aligned_path).open(mode='update') as hdul:
                    hdul[0].header['JWNCAL'] = (
                        int(n_from_match),
                        'N calibrators for JWDISPM / align solution',
                    )
            except Exception as exc:
                logger.info(
                    f'Could not rewrite JWNCAL on {aligned_path}: {exc}'
                )

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
    reader never sees a partial table. Also writes unified ``frame_qa.json``
    next to the summary when rows are available.

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
    try:
        _write_jwst_frame_qa_from_summary(rows, outfile.parent)
    except Exception as exc:
        logger.warning('Could not write JWST frame_qa.json: %s', exc)
    return outfile


def _write_jwst_frame_qa_from_summary(
    rows: list[AlignmentSummaryRow],
    outdir: Path,
) -> Path | None:
    """Build unified ``frame_qa.json`` from JWST alignment summary rows."""
    from st123.stages.alignment.frame_qa import (
        build_frame_qa,
        warn_if_frame_qa_soft,
        write_alignment_summary_table,
        write_frame_qa,
    )

    if not rows:
        return None
    frame_rows: list[dict] = []
    abs_vals: list[float] = []
    n_cals: list[int] = []
    for r in rows:
        disp = r.dispersion_mas if isinstance(r.dispersion_mas, (int, float)) else None
        ncal = r.n_calibrators if isinstance(r.n_calibrators, int) else None
        if disp is not None and str(r.status).upper() == 'SUCCESS':
            abs_vals.append(float(disp))
        if ncal is not None and str(r.status).upper() == 'SUCCESS':
            n_cals.append(int(ncal))
        frame_rows.append(
            {
                'path': Path(r.aligned_path or r.miri_path).name,
                'filter': r.filter,
                'status': r.status,
                'n_calibrators': ncal if ncal is not None else 'NA',
                'dispersion_mas': float(disp) if disp is not None else 'NA',
                'internal_max_delta_mas': 'NA',
                'align_mode': r.align_mode,
                'algnref': r.original_ref,
                'aligned_to': r.aligned_to,
                'mission': 'jwst',
            }
        )
    qa = build_frame_qa(
        mission='jwst',
        align_mode='REFERENCE',
        abs_ref=next((r.original_ref for r in rows if r.original_ref), None),
        residual_mas=max(abs_vals) if abs_vals else None,
        n_calibrators=min(n_cals) if n_cals else None,
        abs_method='catalog_dispersion',
        max_delta_mas=None,
        frames=frame_rows,
    )
    path = write_frame_qa(outdir, qa)
    write_alignment_summary_table(
        [
            {
                'path': fr['path'],
                'filter': fr['filter'],
                'status': fr['status'],
                'n_calibrators': fr['n_calibrators'],
                'dispersion_mas': fr['dispersion_mas'],
                'internal_max_delta_mas': fr['internal_max_delta_mas'],
                'align_mode': fr['align_mode'],
                'algnref': fr['algnref'],
                'aligned_to': fr['aligned_to'],
            }
            for fr in frame_rows
        ],
        outdir / 'frame_qa_alignment_summary.txt',
    )
    warn_if_frame_qa_soft(qa, log=logger, context=outdir.name)
    return path


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

    Preferred layout is
    ``<data-dir>/download/JWST/MIRI/<FILTER>/<obsid>/*mirimage*_cal.fits``
    (flattened MAST products). Also matches legacy nested
    ``.../mastDownload/JWST/*_mirimage/*_cal.fits`` trees and older
    ``<FILTER>/<obsid>/...`` layouts.

    Non-full-frame MIRI products (subarrays / cutouts) are skipped - they are
    unsupported by ``mirimask`` and the alignment->DOLPHOT path.

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
    n_skipped = 0
    patterns = (
        '**/mastDownload/JWST/*_mirimage/*_cal.fits',
        '**/MIRI/*/*/*mirimage*_cal.fits',
    )
    for pattern in patterns:
        for path in data_dir.glob(pattern):
            if not is_full_frame_miri(path):
                n_skipped += 1
                logger.info(
                    'Skipping non-full-frame MIRI cal (unsupported): %s',
                    path,
                )
                continue
            found.add(str(path.resolve()))
    if n_skipped:
        logger.info(
            'Skipped %d non-full-frame MIRI cal frame(s) during discovery',
            n_skipped,
        )
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

    Supports flattened
    ``.../JWST/MIRI/<FILTER>/<obsid>/<filename>``, nested
    ``.../JWST/MIRI/<FILTER>/<obsid>/mastDownload/JWST/...``,
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
    # Flat canonical: .../MIRI/<FILTER>/<obsid>/<file>
    for i, part in enumerate(parts):
        if part.upper() != 'MIRI' or i + 1 >= len(parts):
            continue
        tok = str(parts[i + 1])
        if _looks_like_filter(tok):
            return tok.upper()
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
        Injected from :mod:`st123.stages.mosaic.image_overlap`.

    Returns
    -------
    list of FrameOverlaps
        One entry per science frame.
    """
    from st123.stages.mosaic.image_overlap import compute_cumulative_overlap_fraction

    results: list[FrameOverlaps] = []

    for image in miri_images:
        logger.info(f'MIRI: {image}')
        miri = MirIFootprint.from_fits(image)
        logger.info(f'  illuminated S_REGION: {miri.s_region.to_string()}')
        logger.info(
            f'  WCS pixel solid angle: {miri.pixel_area_arcmin2:.8e} '
            f'arcmin^2 / pixel'
        )
        logger.info(f'  illuminated area: {miri.area.format()}')

        overlapping = []
        best = None

        for ref in refs:
            try:
                result = compute_overlap(miri, ref)
            except Exception as exc:
                logger.warning(f'  FAILED for ref {ref}: {exc}')
                continue

            logger.info(f'  ref: {ref}')
            logger.info(f'    S_REGION: {result.ref_s_region.to_string()}')
            logger.info(f'    ref area: {result.ref_area.format()}')
            logger.info(f'    overlap area: {result.overlap_area.format()}')

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

        logger.info(
            f'Overlap maximized: MIRI image: {best.miri_path}, '
            f'Reference image: {best.ref_path}, '
            f'Max overlap area: {best.overlap_area.pixels2:.3f} pixels^2 '
            f'({best.overlap_area.arcmin2:.6f} arcmin^2, '
            f'{best.overlap_area.fraction_of_miri_roi:.4f} of MIRI illuminated ROI); '
            f'{len(overlapping)} reference(s) with any overlap; '
            f'union coverage {union_frac:.4f} of MIRI ROI'
        )
        if overlapping:
            logger.info('  References with any overlap (largest first):')
            for result in overlapping:
                logger.info(
                    f'    {result.ref_path}: '
                    f'{result.overlap_area.pixels2:.3f} pixels^2 '
                    f'({result.overlap_area.fraction_of_miri_roi:.4f} of MIRI ROI)'
                )

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
    logger.info(f'Wrote overlap summary: {txt_path}')
    logger.info(f'Wrote overlap JSON:    {json_path}')
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
        logger.error(f'ERROR: legacy overlap file not found: {overlap_file}')
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

                result = run_logged_subprocess(command, check=False, logger=logger)
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

    logger.info('Done')
    logger.info(f'Successful pairs written to {success_file} ({n_ok})')
    logger.info(f'Failed pairs written to {fail_file} ({n_fail})')
    return 1 if n_fail else 0


def ref_instrument(path: str | Path) -> str:
    """
    Return the primary-header ``INSTRUME`` for a reference FITS file.

    Parameters
    ----------
    path : str or Path
        Reference coadd / image.

    Returns
    -------
    str
        Uppercased instrument name, or ``''`` when missing / unreadable.
    """
    try:
        return str(as_datamodel(str(path)).keyword('INSTRUME', ext=0) or '').strip().upper()
    except Exception:
        return ''


def prefer_nircam_reference_paths(ref_paths: list[str]) -> list[str]:
    """
    Prefer NIRCam coadds when present so MIRI is not aligned to MIRI mosaics.

    Parameters
    ----------
    ref_paths : list of str
        Candidate reference paths (best-first).

    Returns
    -------
    list of str
        NIRCam-only subset when any NIRCam refs exist; otherwise the input list.
    """
    nircam = [p for p in ref_paths if ref_instrument(p) == 'NIRCAM']
    if nircam:
        return nircam
    return list(ref_paths)


def _frame_ref_images(frame: FrameOverlaps | dict) -> tuple[str, list[str], str | None]:
    """
    Return the science path and ordered reference images for one frame.

    When any overlapping NIRCam coadd exists, MIRI self-coadds are dropped so
    MIRI frames align to the absolute NIRCam frame (e.g. F150W2) rather than
    to previously mosaicked MIRI products.

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
    ordered = prefer_nircam_reference_paths(ordered)
    best_ref = ordered[0] if ordered else None
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
    from st123.stages.mosaic.image_overlap import (
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


def reject_non_full_frame_miri_frames(
    frames: list[FrameOverlaps] | list[dict],
) -> tuple[list[FrameOverlaps] | list[dict], int]:
    """
    Drop MIRI frames that are not full-frame imager products.

    Subarrays and cutouts are unsupported downstream (``mirimask`` / DOLPHOT).
    Rejected frames are omitted from ``alignment_summary.txt``.

    Parameters
    ----------
    frames : list
        Overlap records.

    Returns
    -------
    tuple
        ``(kept_frames, n_rejected)``.
    """
    kept: list[FrameOverlaps | dict] = []
    n_rejected = 0
    for frame in frames:
        miri_path, _ref_images, _best_ref = _frame_ref_images(frame)
        if is_full_frame_miri(miri_path):
            kept.append(frame)
            continue
        n_rejected += 1
        filt = filter_name_from_miri_path(miri_path) or read_miri_filter(miri_path)
        logger.info(
            'REJECT %s  %s  non-full-frame MIRI (excluded from alignment)',
            Path(miri_path).name,
            filt,
        )
    if n_rejected:
        logger.info(
            'Rejected %d non-full-frame MIRI frame(s); %d remain for alignment',
            n_rejected,
            len(kept),
        )
    return kept, n_rejected


def reject_zero_nircam_overlap_frames(
    frames: list[FrameOverlaps] | list[dict],
    *,
    min_ref_overlap_frac: float = 0.02,
) -> tuple[list[FrameOverlaps] | list[dict], int]:
    """
    Drop frames with insufficient cumulative reference footprint overlap.

    Rejected frames remain in the overlap summaries only - they are not written
    to ``alignment_summary.txt`` and never reach JHAT, including MIRI->MIRI
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
        logger.info(
            f'REJECT {Path(miri_path).name}  {filt}  '
            f'ref_overlap_frac={ov_frac:.4f} < {min_frac:.4f} '
            f'(excluded from alignment)',
        )

    if n_rejected:
        logger.info(
            f'Rejected {n_rejected} MIRI frame(s) with '
            f'ref_overlap_frac < {min_frac:.4f}; '
            f'{len(kept)} frame(s) remain for alignment',
        )
    return kept, n_rejected


def _group_frames_by_filter(
    frames: list[FrameOverlaps] | list[dict],
) -> OrderedDict[str, list[FrameOverlaps | dict]]:
    """
    Group frames by filter, preserving blue->red order of first appearance.

    Parameters
    ----------
    frames : list
        Overlap records.

    Returns
    -------
    collections.OrderedDict
        Filter name mapped to its frames.
    """
    ordered = sort_frames_blue_to_red(frames)
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


def _log_align_worker_detail(
    result,
    *,
    kept_reference: bool = False,
) -> None:
    """
    Log a worker ``error`` detail string at an appropriate level.

    REFERENCE quality-holds and other deferred fallbacks are INFO (recoverable).
    ERROR is reserved for terminal FAILURE after fallbacks are exhausted.
    """
    if not result.error:
        return
    status = str(result.row.get('status', '')).upper()
    err = str(result.error)
    recoverable = (
        kept_reference
        or status == 'PENDING'
        or status in {'SUCCESS', 'SKIP', 'REJECTED'}
        or result.ok
        or 'trying MIRI_REL' in err
    )
    if recoverable:
        logger.info('  detail: %s', err)
        return
    logger.error('  detail: %s', err)


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


def flag_peer_inconsistent_reference_rows(
    *,
    filter_name: str,
    row_by_miri: dict[str, AlignmentSummaryRow],
    rows: list[AlignmentSummaryRow],
    successes: list[SuccessfulAlignment],
    max_peer_dispersion_mas: float = DEFAULT_PEER_DISPERSION_MAX_MAS,
    min_overlap: float = DEFAULT_PEER_MIN_OVERLAP,
    min_matches: int = DEFAULT_PEER_MIN_MATCHES,
    match_radius_arcsec: float = DEFAULT_PEER_MATCH_RADIUS_ARCSEC,
) -> list[str]:
    """
    Demote REFERENCE SUCCESS frames that disagree with overlapping peers.

    Independent REFERENCE alignments can each achieve a low ``JWDISPM`` against
    a small refined refcat while disagreeing by several MIRI pixels on shared
    sky. Those peers are marked ``PENDING`` so same-filter MIRI_REL can restore
    relative consistency onto the better absolute solution.

    Parameters
    ----------
    filter_name : str
        Current filter wave.
    row_by_miri : dict
        Live summary rows keyed by MIRI path (mutated in place).
    rows : list
        Ordered summary rows (mutated in place).
    successes : list of SuccessfulAlignment
        Live SUCCESS list (mutated in place).
    max_peer_dispersion_mas : float, optional
        Median peer separation above which a pair is inconsistent.
    min_overlap : float, optional
        Minimum footprint overlap fraction required to compare a pair.
    min_matches : int, optional
        Minimum cross-matched sources required for a peer measurement.
    match_radius_arcsec : float, optional
        Match radius for the photometry cross-match.

    Returns
    -------
    list of str
        MIRI paths demoted to PENDING.
    """
    filt = str(filter_name).upper()
    success_rows = [
        row_by_miri[s.miri_path]
        for s in successes
        if s.miri_path in row_by_miri
        and str(row_by_miri[s.miri_path].filter).upper() == filt
        and row_by_miri[s.miri_path].status == 'SUCCESS'
        and _normalize_align_mode(row_by_miri[s.miri_path].align_mode) == 'REFERENCE'
        and row_by_miri[s.miri_path].aligned_path not in ('NA', None, '')
    ]
    # Unique by path, preserve order.
    seen: set[str] = set()
    unique_rows: list[AlignmentSummaryRow] = []
    for row in success_rows:
        if row.miri_path in seen:
            continue
        seen.add(row.miri_path)
        unique_rows.append(row)

    if len(unique_rows) < 2:
        return []

    phot_by_miri: dict[str, str] = {}
    for row in unique_rows:
        phot = find_aligned_photfile(str(row.aligned_path))
        if phot is not None:
            phot_by_miri[row.miri_path] = phot

    demote: set[str] = set()
    for i, a in enumerate(unique_rows):
        for b in unique_rows[i + 1 :]:
            try:
                overlap = sky_overlap_fraction(a.miri_path, b.miri_path)
            except Exception as exc:
                logger.info(
                    f'Peer QA skip {Path(a.miri_path).name} vs '
                    f'{Path(b.miri_path).name}: overlap failed ({exc})'
                )
                continue
            if overlap < min_overlap:
                continue
            phot_a = phot_by_miri.get(a.miri_path)
            phot_b = phot_by_miri.get(b.miri_path)
            if not phot_a or not phot_b:
                logger.info(
                    f'Peer QA skip {Path(a.miri_path).name} vs '
                    f'{Path(b.miri_path).name}: missing aligned phot catalogs'
                )
                continue
            try:
                peer_med, n_match = peer_sky_dispersion_mas(
                    phot_a,
                    phot_b,
                    match_radius_arcsec=match_radius_arcsec,
                )
            except Exception as exc:
                logger.info(
                    f'Peer QA skip {Path(a.miri_path).name} vs '
                    f'{Path(b.miri_path).name}: phot match failed ({exc})'
                )
                continue
            if peer_med is None or n_match < min_matches:
                logger.info(
                    f'Peer QA {Path(a.miri_path).name} vs '
                    f'{Path(b.miri_path).name}: overlap={overlap:.3f} '
                    f'n_match={n_match} (need >={min_matches}); skipping'
                )
                continue
            logger.info(
                f'Peer QA {Path(a.miri_path).name} vs '
                f'{Path(b.miri_path).name}: overlap={overlap:.3f} '
                f'n_match={n_match} peer_med={peer_med:.1f} mas '
                f'(limit {max_peer_dispersion_mas:.1f} mas)'
            )
            if peer_med <= max_peer_dispersion_mas:
                continue
            # Demote the worse absolute solution; keep the better as MIRI_REL parent.
            disp_a = float(a.dispersion_mas) if isinstance(a.dispersion_mas, float) else np.inf
            disp_b = float(b.dispersion_mas) if isinstance(b.dispersion_mas, float) else np.inf
            worse = a if disp_a >= disp_b else b
            demote.add(worse.miri_path)

    demoted_paths: list[str] = []
    for miri_path in sorted(demote):
        prev = row_by_miri.get(miri_path)
        if prev is None or prev.status != 'SUCCESS':
            continue
        pending = AlignmentSummaryRow(
            miri_path=prev.miri_path,
            filter=prev.filter,
            status='PENDING',
            n_calibrators=prev.n_calibrators,
            dispersion_mas=prev.dispersion_mas,
            aligned_path=prev.aligned_path,
            align_mode=_normalize_align_mode(prev.align_mode),
            original_ref=prev.original_ref,
            aligned_to=prev.aligned_to,
            ref_overlap_frac=prev.ref_overlap_frac,
        )
        idx = rows.index(prev)
        rows[idx] = pending
        row_by_miri[miri_path] = pending
        successes[:] = [s for s in successes if s.miri_path != miri_path]
        demoted_paths.append(miri_path)
        logger.info(
            f'DONE  {Path(miri_path).name}  {prev.filter}  PENDING  '
            f'align_mode=REFERENCE  dispersion_mas='
            f'{prev.dispersion_mas if isinstance(prev.dispersion_mas, float) else prev.dispersion_mas} '
            f'(peer inconsistency; try MIRI_REL)'
        )
    return demoted_paths


# ---------------------------------------------------------------------------
# 6. Parallel workers
# ---------------------------------------------------------------------------
#
# These must stay module-level so ``ProcessPoolExecutor`` can pickle them for
# spawn children. JHAT / pipeline chatter is mediated with ``capture_output``
# into the shared log file; the parent process emits START / DONE. Visit and
# reference workers share job construction, crash handling, and post-JHAT
# finalize helpers below.

_STACK_READY = False


@dataclass
class AlignWorkerResult:
    """Picklable result returned by a worker process."""

    miri_path: str
    filter: str
    mode: str  # 'visit' | 'reference' | 'fallback' | 'skip'
    ok: bool
    row: dict[str, Any]
    success: dict[str, Any] | None = None
    error: str | None = None
    message: str = ''


def _build_visit_jhat_job(
    *,
    image: str,
    outdir: str,
    photfilename: str | None,
    repo: str,
    filter: str = '',
    xshift: float = 0.0,
    yshift: float = 0.0,
    Nbright: int = 800,
    sig: float = 2.0,
    gaia: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Build a picklable visit-mode JHAT job dict for ``_run_jobs_parallel``."""
    return {
        'miri_path': image,  # START/DONE label key shared with all workers
        'align_image': image,
        'outdir': outdir,
        'gaia': bool(gaia),
        'photfilename': photfilename,
        'xshift': float(xshift),
        'yshift': float(yshift),
        'Nbright': int(Nbright),
        'sig': float(sig),
        'filter': str(filter or ''),
        'mode': 'VISIT',
        'repo': str(repo),
        'verbose': bool(verbose),
    }


def _worker_crash_result(
    job: dict[str, Any],
    exc: Exception | None,
    *,
    mode: str,
    message: str,
    ran_ok: bool = False,
) -> AlignWorkerResult:
    """Shared FAILURE result when a worker raises or cannot run JHAT."""
    miri_path = job['miri_path']
    filt = str(job.get('filter', ''))
    outdir = Path(job['outdir'])
    ref_overlap_frac = job.get('ref_overlap_frac', 'NA')
    err = message if exc is None else f'{message}: {exc}'
    if job.get('verbose') and exc is not None:
        err = f'{err}\n{traceback.format_exc()}'
    row = harvest_alignment_metrics(
        miri_path,
        outdir,
        ran_ok=ran_ok,
        ref_overlap_frac=ref_overlap_frac,
    )
    if filt:
        row.filter = filt
    return AlignWorkerResult(
        miri_path=miri_path,
        filter=row.filter or filt,
        mode=mode,
        ok=False,
        row=asdict(row),
        error=err,
        message=message,
    )


def _finalize_jhat_worker(
    job: dict[str, Any],
    *,
    mode: str,
    align_mode: str,
    ran_ok: bool,
    original_ref: str = 'NA',
    aligned_to: str = 'NA',
    write_provenance: bool = True,
    soft_fail_error: str | None = None,
) -> AlignWorkerResult:
    """
    Shared post-JHAT path: harvest metrics, optional provenance, AlignWorkerResult.

    SUCCESS requires a finite mean dispersion in the JHAT header (same contract
    for visit and reference).
    """
    miri_path = job['miri_path']
    filt = str(job.get('filter', ''))
    outdir = Path(job['outdir'])
    ref_overlap_frac = job.get('ref_overlap_frac', 'NA')
    row = harvest_alignment_metrics(
        miri_path,
        outdir,
        ran_ok=ran_ok,
        default_align_mode=align_mode,
        default_original_ref=original_ref,
        default_aligned_to=aligned_to,
        ref_overlap_frac=ref_overlap_frac,
    )
    if filt and (not row.filter or row.filter == 'UNKNOWN'):
        row.filter = filt

    if row.status != 'SUCCESS' or not isinstance(row.dispersion_mas, float):
        return AlignWorkerResult(
            miri_path=miri_path,
            filter=row.filter or filt,
            mode=mode,
            ok=False,
            row=asdict(row),
            error=soft_fail_error or f'{align_mode} alignment soft-failed',
        )

    if write_provenance and row.aligned_path not in ('NA', None, ''):
        write_alignment_provenance(
            row.aligned_path,
            align_mode=align_mode,
            original_ref=original_ref,
            aligned_to=aligned_to,
            relative_dispersion_mas=float(row.dispersion_mas),
            absolute_dispersion_mas=float(row.dispersion_mas),
            n_calibrators=(
                row.n_calibrators if isinstance(row.n_calibrators, int) else None
            ),
        )
        row = harvest_alignment_metrics(
            miri_path,
            outdir,
            ran_ok=True,
            default_align_mode=align_mode,
            default_original_ref=original_ref,
            default_aligned_to=aligned_to,
            ref_overlap_frac=ref_overlap_frac,
        )
        if filt and (not row.filter or row.filter == 'UNKNOWN'):
            row.filter = filt

    return AlignWorkerResult(
        miri_path=miri_path,
        filter=row.filter or filt,
        mode=mode,
        ok=True,
        row=asdict(row),
        message='ok',
    )


def _count_worker_failures(results: list[AlignWorkerResult]) -> int:
    """Return how many worker results are not ``ok``."""
    return sum(1 for result in results if not result.ok)


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
                import st123.stages.alignment.align  # noqa: F401
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
    configure_worker_logging()
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
    configure_worker_logging()
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
    logger.info(
        f'  {filt} calibrator settings: {describe_calibrator_settings(settings)}',
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
        Outcome for the parent process summary line (metrics via harvest).
    """
    _ensure_worker_ready(job.get('repo', ''))
    image = job['align_image']
    photfilename = job.get('photfilename')
    aligned_to = (
        str(photfilename)
        if photfilename
        else ('gaia' if job.get('gaia') else 'NA')
    )
    try:
        # Quiet JHAT / pipeline chatter; parent only sees START / DONE.
        with capture_output():
            align_jwst_image(
                align_image=image,
                outdir=job['outdir'],
                gaia=bool(job.get('gaia', False)),
                photfilename=photfilename,
                xshift=float(job.get('xshift', 0.0)),
                yshift=float(job.get('yshift', 0.0)),
                Nbright=int(job.get('Nbright', 800)),
                sig=float(job.get('sig', 2)),
                verbose=False,
                plot=False,
            )
        return _finalize_jhat_worker(
            job,
            mode='visit',
            align_mode='VISIT',
            ran_ok=True,
            original_ref=aligned_to,
            aligned_to=aligned_to,
            write_provenance=True,
        )
    except Exception as exc:
        return _worker_crash_result(
            job, exc, mode='visit', message='Worker crashed'
        )


def run_reference_align_job(job: dict[str, Any]) -> AlignWorkerResult:
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
    ref_overlap_frac = job.get('ref_overlap_frac', 'NA')

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
        with capture_output():
            run_alignment(
                ref_images=ref_images,
                align_image=miri_path,
                outdir=str(job['outdir']),
                cache_dir=job.get('cache_dir'),
                match_radius_arcsec=job['match_radius_arcsec'],
                clip_to_align_footprint=job['clip_to_align_footprint'],
                **align_kw,
            )
    except Exception as exc:
        return _worker_crash_result(
            job, exc, mode='reference', message='REFERENCE alignment failed'
        )

    original_ref = best_ref or ref_images[0]
    aligned_to = original_ref
    result = _finalize_jhat_worker(
        job,
        mode='reference',
        align_mode='REFERENCE',
        ran_ok=True,
        original_ref=original_ref,
        aligned_to=aligned_to,
        write_provenance=True,
        soft_fail_error='REFERENCE alignment soft-failed',
    )
    if not result.ok:
        return result

    row = AlignmentSummaryRow(**result.row)

    # Too few calibrators in the final dispersion match set -> quality-hold
    # even when JWDISPM looks good (tiny overfitted subsets / soft-fail
    # JWNCAL=0 are common false SUCCESS modes).
    min_cal = calibrator_settings_for_filter(filt).min_calibrators
    n_cal = row.n_calibrators if isinstance(row.n_calibrators, int) else None
    if n_cal is None and row.aligned_path not in ('NA', None, ''):
        n_cal = count_alignment_calibrators(str(row.aligned_path))
        if n_cal is not None:
            row.n_calibrators = n_cal
            if row.aligned_path not in ('NA', None, ''):
                write_alignment_provenance(
                    str(row.aligned_path),
                    align_mode='REFERENCE',
                    original_ref=original_ref,
                    aligned_to=aligned_to,
                    relative_dispersion_mas=float(row.dispersion_mas),
                    absolute_dispersion_mas=float(row.dispersion_mas),
                    n_calibrators=n_cal,
                )
    elif (
        isinstance(n_cal, int)
        and row.aligned_path not in ('NA', None, '')
        and not jwncal_is_plausible(n_cal, str(row.aligned_path))
    ):
        n_fixed = count_alignment_calibrators(str(row.aligned_path))
        if n_fixed is not None:
            logger.info(
                f'{Path(row.aligned_path).name}: correcting JWNCAL '
                f'{n_cal} -> {n_fixed} before min_calibrators gate'
            )
            n_cal = n_fixed
            row.n_calibrators = n_fixed
            write_alignment_provenance(
                str(row.aligned_path),
                align_mode='REFERENCE',
                original_ref=original_ref,
                aligned_to=aligned_to,
                relative_dispersion_mas=float(row.dispersion_mas),
                absolute_dispersion_mas=float(row.dispersion_mas),
                n_calibrators=n_fixed,
            )
    if n_cal is not None and n_cal < min_cal:
        row.status = 'PENDING'
        return AlignWorkerResult(
            miri_path=miri_path,
            filter=row.filter,
            mode='reference',
            ok=False,
            row=asdict(row),
            error=(
                f'REFERENCE n_calibrators={n_cal} below min_calibrators={min_cal} '
                f'for {filt}; trying MIRI_REL'
            ),
        )

    # ``None`` in the job means use the per-filter map; a positive float is a
    # uniform CLI override; <=0 disables the quality hold.
    max_disp = job.get('max_nircam_dispersion_mas')
    if max_disp is None:
        max_disp = max_reference_dispersion_mas(filt)
    elif float(max_disp) <= 0:
        max_disp = None
    gate_disp = float(row.dispersion_mas)
    skew_reason = None
    filt_key = str(filt or '').upper().split('_', 1)[0]
    if (
        filt_key == 'F770W'
        and row.aligned_path not in ('NA', None, '')
        and isinstance(row.dispersion_mas, float)
    ):
        med_mas = read_jhat_dispersion_median_mas(str(row.aligned_path))
        gate_disp, skew_reason = f770w_reference_gate_dispersion_mas(
            float(row.dispersion_mas), med_mas
        )
    hold_for_disp = (
        max_disp is not None
        and float(max_disp) > 0
        and gate_disp > float(max_disp)
    )
    hold_for_skew = skew_reason is not None
    if hold_for_disp or hold_for_skew:
        # Keep REFERENCE products on disk, but mark PENDING (not FAILURE) so
        # the parent can try MIRI_REL before recording a final status.
        # PENDING rows are omitted from the live alignment summary.
        row.status = 'PENDING'
        if hold_for_skew and not hold_for_disp:
            err = f'{skew_reason}; trying MIRI_REL'
        else:
            err = (
                f'REFERENCE dispersion {gate_disp:.3f} mas '
                f'exceeds quality threshold {float(max_disp):.3f} mas'
            )
            if skew_reason:
                err = f'{err}; {skew_reason}'
            err = f'{err}; trying MIRI_REL'
        return AlignWorkerResult(
            miri_path=miri_path,
            filter=row.filter,
            mode='reference',
            ok=False,
            row=asdict(row),
            error=err,
        )

    success = SuccessfulAlignment(
        miri_path=miri_path,
        jhat_path=row.aligned_path,
        filter=row.filter,
        wavelength_um=filter_wavelength_um(row.miri_path),
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
        mode='reference',
        ok=True,
        row=asdict(row),
        success=asdict(success),
    )


# Backward-compatible alias (REFERENCE worker was historically misnamed).
run_nircam_align_job = run_reference_align_job


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
        return _worker_crash_result(
            job, exc, mode='fallback', message=msg, ran_ok=False
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
            with capture_output():
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

        # Keep MIRI_REL when it improves absolute dispersion. If the held
        # REFERENCE is under-calibrated, also accept a finalized parent chain
        # whose absolute is only moderately worse - prefer a real relative
        # tie over a tiny-n_cal REFERENCE SUCCESS.
        ref_under_cal = bool(job.get('reference_under_calibrated'))
        parent_finalized = not bool(getattr(parent, 'provisional', False))
        if ref_disp_f is not None and abs_mas >= ref_disp_f:
            accept_under_cal = (
                ref_under_cal
                and parent_finalized
                and abs_mas < max(float(ref_disp_f) * 1.25, float(ref_disp_f) + 15.0)
                and abs_mas < 150.0
            )
            if not accept_under_cal:
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
            wavelength_um=filter_wavelength_um(row.miri_path),
            dispersion_mas=float(row.dispersion_mas),
            relative_dispersion_mas=rel_mas,
            align_mode='MIRI_REL',
            original_ref=parent.original_ref,
            aligned_to=parent.jhat_path,
            photfile=find_aligned_photfile(row.aligned_path),
            provisional=False,
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
    logger.info(
        f'{label}: {len(jobs)} job(s), workers={min(n_workers, len(jobs))}',
    )

    def start_line(job: dict) -> None:
        logger.info(f'START {Path(job["miri_path"]).name}')

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
    Align frames filter-by-filter (blue->red), in parallel within each filter.

    For each filter wave the frames are first aligned to their overlapping
    reference images (``align_mode=REFERENCE`` on success). MIRI->MIRI fallback
    then runs in parallel for hard failures and for REFERENCE solutions whose
    dispersion exceeds the per-filter (or uniform CLI) quality-hold threshold;
    MIRI_REL is kept only when its absolute dispersion improves on REFERENCE.
    The fallback pass repeats once so same-filter MIRI_REL successes can parent
    remaining hard failures.

    Summary ``status`` is binary SUCCESS / FAILURE; the method is recorded in
    ``align_mode``. REFERENCE quality-holds are omitted from the live summary
    while MIRI_REL is pending. Soft-fail / tiny-``n_cal`` REFERENCE holds are
    never promoted to SUCCESS; keepable holds (enough calibrators) may be kept
    when MIRI_REL cannot improve them. After each MIRI_REL pass, absolute
    dispersions are re-propagated from finalized parents so children do not
    inherit stale provisional-parent scores. Per-frame failures are always
    recorded and processing continues.

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
        Enable MIRI->MIRI relative fallback.
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
    # CLI: None -> per-filter map; <=0 -> disable; >0 -> uniform override.
    if max_nircam_dispersion_mas is not None and max_nircam_dispersion_mas <= 0:
        max_nircam_dispersion_mas = 0.0  # sentinel: disabled for all filters

    # Drop unsupported / low-overlap frames before any alignment work. These
    # remain in overlap_summary* only and are omitted from alignment_summary.txt.
    frames, n_bad = reject_non_full_frame_miri_frames(frames)
    frames, n_rejected = reject_zero_nircam_overlap_frames(
        frames, min_ref_overlap_frac=min_ref_overlap_frac
    )
    del n_bad  # logged inside reject_non_full_frame_miri_frames
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
        Keep a usable REFERENCE solution after MIRI_REL does not improve it.

        The per-filter dispersion cut is a *try MIRI_REL* trigger, not a hard
        reject: a *keepable* REFERENCE WCS (usable + enough calibrators)
        remains SUCCESS when fallback cannot beat it. Soft-fail / tiny-n_cal
        / unusable holds must not use this path.
        """
        nonlocal n_ok
        if not reference_solution_keepable(prev):
            finalize_quality_hold_as_failure(
                prev,
                reason=f'{reason}; REFERENCE not keepable',
            )
            return
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
        successes.append(
            SuccessfulAlignment(
                miri_path=final.miri_path,
                jhat_path=str(final.aligned_path),
                filter=final.filter,
                wavelength_um=filter_wavelength_um(final.miri_path),
                dispersion_mas=float(final.dispersion_mas),
                relative_dispersion_mas=float(final.dispersion_mas),
                align_mode='REFERENCE',
                original_ref=str(final.original_ref),
                aligned_to=str(final.aligned_to),
                photfile=find_aligned_photfile(str(final.aligned_path)),
                provisional=False,
            )
        )
        n_ok += 1
        logger.info(
            f'DONE  {Path(prev.miri_path).name}  {prev.filter}  SUCCESS  '
            f'align_mode=REFERENCE  dispersion_mas={final.dispersion_mas:.3f} '
            f'({reason})',
        )
        flush_summary()

    def finalize_quality_hold_as_failure(
        prev: AlignmentSummaryRow, *, reason: str
    ) -> None:
        """
        Finalize a REFERENCE quality hold as FAILURE (no usable MIRI_REL).

        Soft-fail / quality-held frames must not appear as SUCCESS in the
        alignment summary when relative alignment never lands.
        """
        disp = prev.dispersion_mas
        disp_txt = (
            f'{disp:.3f}' if isinstance(disp, float) else str(disp)
        )
        final = AlignmentSummaryRow(
            miri_path=prev.miri_path,
            filter=prev.filter,
            status='FAILURE',
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
        # Never seed later waves from a failed quality hold.
        successes[:] = [s for s in successes if s.miri_path != prev.miri_path]
        logger.info(
            f'DONE  {Path(prev.miri_path).name}  {prev.filter}  FAILURE  '
            f'align_mode=REFERENCE  dispersion_mas={disp_txt} ({reason})',
        )
        flush_summary()

    def record_result(result, *, count_fallback: bool = False) -> None:
        nonlocal n_ok, n_fallback
        row = AlignmentSummaryRow(**result.row)
        prev = row_by_miri.get(result.miri_path)

        # MIRI_REL did not improve a REFERENCE quality hold: keep keepable
        # REFERENCE only; soft-fail / tiny-n_cal / unusable -> FAILURE.
        if (
            not result.ok
            and result.mode == 'fallback'
            and prev is not None
            and _is_reference_quality_hold(prev)
        ):
            why = 'MIRI_REL did not improve REFERENCE'
            if result.error:
                why = f'{why}: {result.error}'
            if reference_solution_keepable(prev):
                finalize_quality_hold_keep_reference(prev, reason=why)
                if verbose:
                    _log_align_worker_detail(result, kept_reference=True)
            else:
                detail = 'REFERENCE not keepable'
                if reference_solution_usable(prev) and not reference_solution_keepable(
                    prev
                ):
                    min_cal = calibrator_settings_for_filter(prev.filter).min_calibrators
                    detail = (
                        f'REFERENCE n_calibrators={prev.n_calibrators} '
                        f'< min_calibrators={min_cal}'
                    )
                finalize_quality_hold_as_failure(
                    prev,
                    reason=f'{why}; {detail}',
                )
                if verbose:
                    _log_align_worker_detail(result)
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

        logger.info(_format_worker_done(result))
        if verbose:
            _log_align_worker_detail(result)
        flush_summary()

    if summary_outfile is not None:
        summary_outfile = Path(summary_outfile).expanduser().resolve()
        write_alignment_summary(rows, summary_outfile)
        logger.info(f'Live alignment summary -> {summary_outfile}')

    logger.info(
        f'Alignment plan: {len(groups)} filter wave(s), '
        f'{sum(len(v) for v in groups.values())} frame(s) after rejecting '
        f'{n_rejected} low-overlap '
        f'(ref_overlap_frac < {min_ref_overlap_frac:.4f}), workers={workers}'
    )
    for filt, group in groups.items():
        logger.info(f'  {filt}: {len(group)} frame(s)')

    if not groups:
        logger.info('No science frames with reference overlap remain to align.')
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
        logger.info(
            'Filter calibrators: F770W -> '
            f'{describe_calibrator_settings(F770W_CALIBRATOR_SETTINGS)}'
        )
    else:
        logger.info('Filter calibrators: disabled (--no-filter-calibrators)')

    if max_nircam_dispersion_mas == 0.0:
        logger.info('REFERENCE quality hold: disabled')
    elif max_nircam_dispersion_mas is not None:
        logger.info(
            f'REFERENCE quality hold: uniform > '
            f'{max_nircam_dispersion_mas:.1f} mas -> try MIRI_REL '
            f'(keep only if improved)'
        )
    else:
        parts = []
        for name, thr in FILTER_MAX_REFERENCE_DISPERSION_MAS.items():
            parts.append(f'{name}:{"off" if thr is None else f"{thr:.0f}"}')
        logger.info(
            'REFERENCE quality hold (per filter, mas -> try MIRI_REL; '
            f'keep only if improved): {", ".join(parts)}'
        )

    for filt, group in groups.items():
        logger.info('=' * 72)
        logger.info(f'Filter wave {filt}: {len(group)} frame(s)')
        logger.info('=' * 72)

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
        reference_jobs = [{**job, 'mode': 'reference'} for job in pending.values()]

        _run_jobs_parallel(
            reference_jobs,
            run_reference_align_job,
            workers=workers,
            label=f'{filt} REFERENCE',
            on_result=record_result,
        )

        # Peer consistency: independent REFERENCE solutions that disagree on
        # overlapping sky are demoted to PENDING so same-filter MIRI_REL can
        # restore relative alignment onto the better absolute frame.
        demoted = flag_peer_inconsistent_reference_rows(
            filter_name=filt,
            row_by_miri=row_by_miri,
            rows=rows,
            successes=successes,
        )
        if demoted:
            logger.info(
                f'{filt}: demoted {len(demoted)} REFERENCE frame(s) for peer '
                f'inconsistency -> MIRI_REL'
            )
            flush_summary()

        # --- Passes 2+: parallel MIRI fallback ---
        if fallback:
            for pass_idx in (1, 2, 3):
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

                # Finalized SUCCESS first; provisional holds only if under
                # threshold (or as bootstrap when no SUCCESS exists yet).
                parent_pool = build_miri_rel_parent_pool(
                    filt, successes, row_by_miri, pending.keys()
                )

                fb_jobs = []
                for miri in ordered_need:
                    ranked = rank_fallback_parents(
                        miri, filt, parent_pool, max_parents=5
                    )
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
                    min_cal = calibrator_settings_for_filter(filt).min_calibrators
                    ref_under_cal = bool(
                        prev_row is not None
                        and _is_reference_quality_hold(prev_row)
                        and isinstance(prev_row.n_calibrators, int)
                        and int(prev_row.n_calibrators) < int(min_cal)
                    )
                    fb_jobs.append(
                        {
                            **pending[miri],
                            'mode': 'fallback',
                            'parent': asdict(ranked[0][0]),
                            'parents': [asdict(p) for p, _ov in ranked],
                            'overlap_fraction': ranked[0][1],
                            'reference_dispersion_mas': ref_disp,
                            'reference_under_calibrated': ref_under_cal,
                        }
                    )

                if not fb_jobs:
                    # Still re-propagate in case earlier passes left stale abs.
                    n_reprop = repropagate_miri_rel_absolutes(
                        successes, row_by_miri, rows
                    )
                    if n_reprop:
                        flush_summary()
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
                n_reprop = repropagate_miri_rel_absolutes(
                    successes, row_by_miri, rows
                )
                if n_reprop:
                    any_new = True
                    flush_summary()
                if not any_new:
                    break

        # Final absolute re-propagation across the wave (parent chains).
        n_reprop = repropagate_miri_rel_absolutes(successes, row_by_miri, rows)
        if n_reprop:
            flush_summary()

        # Finalize remaining REFERENCE quality-holds when no MIRI_REL parent
        # was available (or fallback was disabled): FAILURE, not SUCCESS.
        for miri in list(pending):
            row = row_by_miri.get(miri)
            if row is None or not _is_reference_quality_hold(row):
                continue
            if reference_solution_keepable(row) and not fallback:
                finalize_quality_hold_keep_reference(
                    row,
                    reason='fallback disabled; keepable REFERENCE retained',
                )
                continue
            if reference_solution_keepable(row):
                # Had parents but MIRI_REL never scheduled / never returned -
                # still keep keepable REFERENCE (dispersion hold only).
                finalize_quality_hold_keep_reference(
                    row,
                    reason='no successful MIRI_REL; keepable REFERENCE retained',
                )
                continue
            finalize_quality_hold_as_failure(
                row,
                reason='no MIRI_REL parent; REFERENCE quality hold not kept',
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

        logger.info(
            f'Filter wave {filt} done: '
            f'{sum(1 for m in pending if row_by_miri[m].status == "SUCCESS")} ok, '
            f'{len(wave_failures)} failed'
        )

    logger.info(
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
        Injected from :mod:`st123.stages.mosaic.image_overlap`.

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
    existing_box = getattr(args, 'existing_box', None)
    if existing_box:
        from st123.stages.mosaic.mosaic import box_coadd_i2d_paths, resolve_existing_box_dir

        box_dir = resolve_existing_box_dir(data_dir, existing_box)
        box_resolved = box_dir.resolve()
        refs = [r for r in refs if Path(r).resolve().parent == box_resolved]
        if not refs:
            refs = box_coadd_i2d_paths(box_dir)
        logger.info(
            'Restricting MIRI reference coadds to --existing-box %s (%d file(s))',
            existing_box,
            len(refs),
        )
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

    logger.info(f'Repo:              {args.repo}')
    logger.info(f'Data dir:          {data_dir}')
    logger.info(f'Dataset label:     {args.galaxy}')
    logger.info(f'Filters:           {", ".join(filters) if filters else "ALL"}')
    logger.info(f'MIRI images:       {len(miri_images)}')
    logger.info(f'Reference images:  {len(refs)}')

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
