"""
HST JHAT alignment helpers (Gaia by default) with a WFPC2 ``*_c0m.fits`` patch.

Upstream ``jhat.simple_jwst_phot.hst_photclass`` assumes ACS/WFC3-style headers
(``FILTER1`` is a string; filenames are flt/flc/drz/drc only). WFPC2 calibrated
products use ``FILTNAM1``/``FILTNAM2`` and ``*_c0m.fits``, and often store a
numeric ``FILTER1`` that crashes ``'CLEAR' not in FILTER1``.

Also patches JHAT's encircled-energy helper: SciPy >=1.14 removed ``interp2d``,
which upstream ``hst_get_ee_corr`` still calls.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Sequence

from st123.utils.logging import capture_output, configure_worker_logging
from st123.datamodels import as_datamodel
from st123.stages.alignment.frame_qa import (
    FRAME_ABS_RETIE_MAX_APPLY_ARCSEC,
    FRAME_ABS_RETIE_MIN_PEAK,
    FRAME_ABS_RETIE_TOL_ARCSEC,
    FRAME_ABS_TOL_ARCSEC,
    FRAME_INTERNAL_TOL_ARCSEC,
    FRAME_L3_SOFT_TOL_ARCSEC,
    FRAME_MIN_MATCH_HEALTHY,
    FRAME_SPARSE_TOL_ARCSEC,
    build_frame_qa,
    coherent_tol_arcsec,
    stamp_quality_headers,
    warn_if_frame_qa_soft,
    write_alignment_summary_table,
    write_frame_qa,
)

log = logging.getLogger(__name__)

_PATCH_MARKER = 'load_image_st123_wfpc2'
_EE_PATCH_MARKER = '_hst_get_ee_corr_st123'

# JHAT defaults for Gaia-anchoring a deep level-3 coadd (multi-star fit).
HST_JHAT_GAIA_L3_PARAMS: dict[str, Any] = {
    'overwrite': True,
    'find_stars_threshold': 2.5,
    'SNR_min': 3.0,
    'd2d_max': 1.0,
    'dmag_max': 1.5,
    'objmag_lim': (14, 24),
    'refmag_lim': (14, 21),
    'sharpness_lim': (0.2, 1.0),
    'roundness1_lim': (-1.0, 1.0),
    'Nbright4match': 2000,
    'Nbright': 1500,
    'Nfwhm': 3.5,
    'histocut_order': 'dxdy',
    'iterate_with_xyshifts': True,
    # Allow larger initial offsets than JHAT's 0.8 px default cap.
    'rough_cut_px_min': 0.5,
    'rough_cut_px_max': 8.0,
    'd_rotated_Nsigma': 3.0,
}

# JHAT defaults when aligning science frames to the L3 phot catalog.
# Explicit ra/dec/mag columns are required: JHAT's run_all default
# refcat_racol='auto' is a literal column name for file-based refcats.
HST_JHAT_L3REF_PARAMS: dict[str, Any] = {
    'overwrite': True,
    'find_stars_threshold': 2.5,
    'SNR_min': 3.0,
    'd2d_max': 2.0,
    'dmag_max': 2.0,
    'objmag_lim': (12, 26),
    'sharpness_lim': (0.2, 1.0),
    'roundness1_lim': (-1.0, 1.0),
    'Nbright4match': 3000,
    'Nbright': 2000,
    'Nfwhm': 4.0,
    'histocut_order': 'dxdy',
    'iterate_with_xyshifts': True,
    'rough_cut_px_min': 0.5,
    'rough_cut_px_max': 15.0,
    'd_rotated_Nsigma': 3.0,
    'refcat_racol': 'ra',
    'refcat_deccol': 'dec',
    'refcat_magcol': 'mag',
}

# Narrowband JHAT retry: continuum-poor frames often fail broadband Deltamag /
# tight d2d cuts against F814W/F606W L3 catalogs. Loosen matching only for
# filters classified by :func:`is_hst_narrowband_filter`.
HST_JHAT_NARROWBAND_PARAMS: dict[str, Any] = {
    **HST_JHAT_L3REF_PARAMS,
    'find_stars_threshold': 2.0,
    'SNR_min': 2.5,
    'd2d_max': 5.0,
    'dmag_max': 5.0,
    'objmag_lim': (10, 28),
    'sharpness_lim': (0.15, 1.2),
    'Nbright4match': 4000,
    'Nbright': 2500,
    'rough_cut_px_max': 25.0,
    'd_rotated_Nsigma': 4.0,
}

# Preferred broadband parents when tying narrowband frames (HST_REL).
HST_NARROWBAND_PARENT_FILTERS: tuple[str, ...] = (
    'f814w',
    'f606w',
    'f625w',
    'f555w',
    'f775w',
    'f850lp',
    'f475w',
    'f438w',
    'f336w',
    'f110w',
    'f160w',
)

# Relative-match knobs for sparse narrowband -> broadband ties.
HST_NARROWBAND_REL_MIN_MATCHES: int = 5
HST_NARROWBAND_REL_MAX_MATCH_PIX: float = 25.0
HST_NARROWBAND_REL_NBRIGHT: int = 400


def is_hst_narrowband_filter(filter_name: str | None) -> bool:
    """
    Return whether an HST filter should use the narrowband mitigation path.

    Matches ``F###N`` / ``F####N`` style names and ``FQ*`` quad filters. Medium
    (``M``) and wide (``W`` / ``LP`` / ``X``) bands are excluded.

    Parameters
    ----------
    filter_name : str or None
        Filter string from :func:`st123.utils.helpers.get_filter`.

    Returns
    -------
    bool
        True for narrow / quad narrowband-like filters.
    """
    key = str(filter_name or '').strip().upper().split('_', 1)[0]
    if not key:
        return False
    if key.startswith('FQ'):
        return True
    return bool(re.fullmatch(r'F\d{3,4}N', key))


def write_hst_jhat_from_raw(
    raw_path: str | Path,
    outdir: str | Path,
) -> Path:
    """
    Copy a calibrated HST frame to the expected ``*_jhat.fits`` product path.

    Parameters
    ----------
    raw_path : str or pathlib.Path
        Calibrated ``flc`` / ``flt`` / ``c0m`` frame.
    outdir : str or pathlib.Path
        JHAT output directory.

    Returns
    -------
    pathlib.Path
        Destination JHAT path.
    """
    import shutil

    raw = Path(raw_path).expanduser().resolve()
    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    dest = _jhat_hst_output_path(raw, out)
    shutil.copy2(raw, dest)
    return dest


def write_hst_narrowband_provenance(
    jhat_path: str | Path,
    *,
    align_mode: str,
    aligned_to: str = 'NA',
    n_match: int | None = None,
    abs_arcsec: float | None = None,
    dra_arcsec: float | None = None,
    ddec_arcsec: float | None = None,
) -> None:
    """
    Record narrowband mitigation provenance on the JHAT primary header.

    Parameters
    ----------
    jhat_path : str or pathlib.Path
        JHAT product to update.
    align_mode : str
        ``HST_REL`` (relative to broadband) or ``PIPELINE`` (uncorrected copy).
    aligned_to : str, optional
        Parent JHAT / catalog path for ``HST_REL``.
    n_match, abs_arcsec, dra_arcsec, ddec_arcsec
        Optional relative-match metrics.
    """

    mode = str(align_mode).upper()
    with as_datamodel(jhat_path).open(mode='update', memmap=False) as hdul:
        hdr = hdul[0].header
        hdr['ALGNMODE'] = (mode, 'JHAT, HST_REL, or PIPELINE')
        hdr['ALGNTO'] = (str(aligned_to), 'Narrowband aligned-to parent')
        hdr['ST123NAR'] = (True, 'st123: HST narrowband mitigation product')
        if n_match is not None:
            hdr['ST123NM'] = (int(n_match), 'st123: narrowband relative n_match')
        if abs_arcsec is not None:
            hdr['ST123NAS'] = (
                float(abs_arcsec),
                '[arcsec] narrowband abs offset vs parent',
            )
        if dra_arcsec is not None:
            hdr['ST123NRA'] = (
                float(dra_arcsec),
                '[arcsec] applied dRA cos(Dec)',
            )
        if ddec_arcsec is not None:
            hdr['ST123NDE'] = (float(ddec_arcsec), '[arcsec] applied dDec')
        hdul.flush()


def _hst_parent_filter_rank(filter_name: str) -> int:
    """Lower is better; unknown broadband filters rank after the preferred list."""
    key = str(filter_name or '').strip().lower()
    try:
        return HST_NARROWBAND_PARENT_FILTERS.index(key)
    except ValueError:
        if is_hst_narrowband_filter(key):
            return 10_000
        return 100 + len(HST_NARROWBAND_PARENT_FILTERS)


def rank_hst_narrowband_parents(
    raw_path: str | Path,
    ok_results: list[dict],
    *,
    max_parents: int = 5,
) -> list[Path]:
    """
    Rank aligned broadband JHAT parents for a narrowband frame.

    Prefers same visit key and instrument, then preferred broadband filters
    (F814W, F606W, ...). Narrowband SUCCESS products are never used as parents.

    Parameters
    ----------
    raw_path : str or pathlib.Path
        Failed / pending narrowband calibrated frame.
    ok_results : list of dict
        Batch rows with ``status='ok'`` and ``outpath`` set.
    max_parents : int, optional
        Maximum parents to return.

    Returns
    -------
    list of pathlib.Path
        Ranked parent JHAT paths (best first).
    """
    from st123.utils.helpers import get_filter, get_instrument

    raw = Path(raw_path)
    try:
        child_inst = get_instrument(raw).split('_')[0].lower()
    except Exception:
        child_inst = ''
    child_visit = _hst_visit_key(raw)

    candidates: list[tuple[int, int, int, Path]] = []
    for row in ok_results:
        outpath = row.get('outpath')
        if not outpath:
            continue
        parent = Path(outpath)
        if not parent.is_file():
            continue
        try:
            pfilt = get_filter(parent)
        except Exception:
            continue
        if is_hst_narrowband_filter(pfilt):
            continue
        try:
            pinst = get_instrument(parent).split('_')[0].lower()
        except Exception:
            try:
                pinst = get_instrument(row.get('path') or parent).split('_')[0].lower()
            except Exception:
                pinst = ''
        same_visit = int(_hst_visit_key(parent) != child_visit)
        same_inst = int(pinst != child_inst)
        filt_rank = _hst_parent_filter_rank(pfilt)
        candidates.append((same_visit, same_inst, filt_rank, parent))

    if not candidates:
        return []
    candidates.sort(key=lambda t: (t[0], t[1], t[2], str(t[3])))
    return [c[3] for c in candidates[: max(1, int(max_parents))]]


def select_hst_narrowband_parent(
    raw_path: str | Path,
    ok_results: list[dict],
    *,
    jhat_outdir: str | Path | None = None,
) -> Path | None:
    """
    Choose the best aligned broadband JHAT parent for a narrowband frame.

    Parameters
    ----------
    raw_path : str or pathlib.Path
        Failed / pending narrowband calibrated frame.
    ok_results : list of dict
        Batch rows with ``status='ok'`` and ``outpath`` set.
    jhat_outdir : str or pathlib.Path, optional
        Unused; kept for API symmetry with callers.

    Returns
    -------
    pathlib.Path or None
        Parent JHAT path, or ``None`` when no usable parent exists.
    """
    del jhat_outdir
    ranked = rank_hst_narrowband_parents(raw_path, ok_results, max_parents=1)
    return ranked[0] if ranked else None


def measure_hst_relative_offset_from_phot(
    path_img: str | Path,
    path_ref: str | Path,
    *,
    jhat_outdir: str | Path | None = None,
    match_radius_arcsec: float = 2.0,
    min_matches: int = HST_NARROWBAND_REL_MIN_MATCHES,
    nbright: int = HST_NARROWBAND_REL_NBRIGHT,
) -> dict[str, Any]:
    """
    Relative sky offset using JHAT ``*.phot.txt`` catalogs (fast path).

    Prefers existing photometry next to the frames (or under *jhat_outdir*).
    Falls back to an empty failed stats dict when catalogs are missing.

    Parameters
    ----------
    path_img, path_ref : str or pathlib.Path
        Image and reference FITS (raw or JHAT).
    jhat_outdir : str or pathlib.Path, optional
        Extra directory to search for phot catalogs.
    match_radius_arcsec : float, optional
        Sky match radius.
    min_matches, nbright
        Match / bright-source controls.

    Returns
    -------
    dict
        Same offset fields as :func:`measure_hst_frame_relative_offset_pixel`.
    """
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    import astropy.units as u

    stats: dict[str, Any] = {
        'path_img': str(Path(path_img)),
        'path_ref': str(Path(path_ref)),
        'method': 'phot_sky_match',
        'n_match': 0,
        'ok': False,
        'dra_deg': 0.0,
        'ddec_deg': 0.0,
        'dra_arcsec': 0.0,
        'ddec_arcsec': 0.0,
        'abs_arcsec': 0.0,
        'med_sep_arcsec': 0.0,
        'med_dpix': 0.0,
    }

    def _load_radec(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        search_dirs = []
        if jhat_outdir is not None:
            search_dirs.append(Path(jhat_outdir))
        search_dirs.append(path.parent)
        phot = find_jhat_phot(path, path.parent, search_dirs=search_dirs)
        if phot is None and jhat_outdir is not None:
            phot = find_jhat_phot(path, Path(jhat_outdir))
        if phot is None:
            # Also try stem without product suffix.
            stem = path.name
            for suf in ('_jhat.fits', '_flc.fits', '_flt.fits', '_c0m.fits', '.fits'):
                if stem.lower().endswith(suf):
                    stem = stem[: -len(suf)]
                    break
            for directory in search_dirs:
                cand = Path(directory) / f'{stem}.phot.txt'
                if cand.is_file() and cand.stat().st_size > 0:
                    phot = cand
                    break
        if phot is None:
            return None
        try:
            tbl = Table.read(phot, format='ascii')
        except Exception:
            return None
        if 'ra' not in tbl.colnames or 'dec' not in tbl.colnames:
            return None
        if 'mag' in tbl.colnames:
            tbl.sort('mag')
        elif 'flux' in tbl.colnames:
            tbl.sort('flux')
            tbl.reverse()
        tbl = tbl[: int(nbright)]
        ra = np.asarray(tbl['ra'], dtype=float)
        dec = np.asarray(tbl['dec'], dtype=float)
        good = np.isfinite(ra) & np.isfinite(dec)
        if int(np.count_nonzero(good)) < 3:
            return None
        return ra[good], dec[good]

    img = Path(path_img)
    ref = Path(path_ref)
    coords_i = _load_radec(img)
    coords_r = _load_radec(ref)
    if coords_i is None or coords_r is None:
        return stats
    ra_i, dec_i = coords_i
    ra_r, dec_r = coords_r
    # Narrowband pipeline WCS can be arcminutes off. Bootstrap a coarse
    # centroid shift, then refine with a tight match radius.
    dra0 = float(np.median(ra_i) - np.median(ra_r))
    dra0 = (dra0 + 180.0) % 360.0 - 180.0
    ddec0 = float(np.median(dec_i) - np.median(dec_r))
    ra_shift = ra_i - dra0
    dec_shift = dec_i - ddec0
    c_i = SkyCoord(ra_shift * u.deg, dec_shift * u.deg)
    c_r = SkyCoord(ra_r * u.deg, dec_r * u.deg)
    idx, sep, _ = c_i.match_to_catalog_sky(c_r)
    good = sep.arcsec <= float(match_radius_arcsec)
    n = int(np.count_nonzero(good))
    stats['n_match'] = n
    if n < int(min_matches):
        # One more try with a wider fine radius after the same bootstrap.
        wide = max(float(match_radius_arcsec), 5.0)
        good = sep.arcsec <= wide
        n = int(np.count_nonzero(good))
        stats['n_match'] = n
        if n < int(min_matches):
            return stats
    # Total img-ref offset = bootstrap + residual of shifted coords.
    c_im = SkyCoord(ra_i[good] * u.deg, dec_i[good] * u.deg)
    c_rm = c_r[idx[good]]
    dra = (c_im.ra - c_rm.ra).to(u.deg).value
    dra = (dra + 180.0) % 360.0 - 180.0
    ddec = (c_im.dec - c_rm.dec).to(u.deg).value
    dra_m = float(np.median(dra))
    ddec_m = float(np.median(ddec))
    dec0 = float(np.median(c_im.dec.degree))
    dra_as = dra_m * 3600.0 * float(np.cos(np.radians(dec0)))
    ddec_as = ddec_m * 3600.0
    stats.update(
        {
            'ok': True,
            'dra_deg': dra_m,
            'ddec_deg': ddec_m,
            'dra_arcsec': dra_as,
            'ddec_arcsec': ddec_as,
            'abs_arcsec': float(np.hypot(dra_as, ddec_as)),
            'med_sep_arcsec': float(np.median(sep.arcsec[good])),
            'bootstrap_dra_arcsec': dra0
            * 3600.0
            * float(np.cos(np.radians(dec0))),
            'bootstrap_ddec_arcsec': ddec0 * 3600.0,
        }
    )
    return stats


def align_hst_narrowband_relative(
    raw_path: str | Path,
    parent_jhat: str | Path,
    outdir: str | Path,
    *,
    min_matches: int = HST_NARROWBAND_REL_MIN_MATCHES,
    max_match_pix: float = HST_NARROWBAND_REL_MAX_MATCH_PIX,
    nbright: int = HST_NARROWBAND_REL_NBRIGHT,
    match_radius_arcsec: float = 2.0,
) -> dict[str, Any]:
    """
    Tie a narrowband frame to an aligned broadband JHAT via catalog / pixel match.

    Copies the calibrated raw to ``*_jhat.fits``, measures the sky offset to
    *parent_jhat* (phot-catalog match first, then pixel detection fallback),
    and applies the inverse CRVAL shift (same convention as sibling harmonize).

    Parameters
    ----------
    raw_path : str or pathlib.Path
        Narrowband calibrated frame.
    parent_jhat : str or pathlib.Path
        Aligned broadband JHAT parent.
    outdir : str or pathlib.Path
        JHAT output directory.
    min_matches, max_match_pix, nbright, match_radius_arcsec
        Relative-match controls (looser than broadband harmonize defaults).

    Returns
    -------
    dict
        Result with ``ok``, ``outpath``, ``n_match``, offset fields, and
        ``error`` when the relative match fails (no product kept on failure).
    """

    raw = Path(raw_path).expanduser().resolve()
    parent = Path(parent_jhat).expanduser().resolve()
    out = Path(outdir).expanduser().resolve()
    result: dict[str, Any] = {
        'ok': False,
        'outpath': None,
        'align_mode': 'HST_REL',
        'aligned_to': str(parent),
        'n_match': 0,
        'abs_arcsec': None,
        'error': None,
    }
    if not parent.is_file():
        result['error'] = f'parent missing: {parent}'
        return result

    jhat = write_hst_jhat_from_raw(raw, out)
    result['outpath'] = str(jhat)
    # Fast path: reuse JHAT phot catalogs when present (avoids full-frame DAO).
    off = measure_hst_relative_offset_from_phot(
        jhat,
        parent,
        jhat_outdir=out,
        match_radius_arcsec=float(match_radius_arcsec),
        min_matches=int(min_matches),
        nbright=int(nbright),
    )
    if not off.get('ok'):
        # Also try matching raw-stem phot (written before JHAT failed) to parent.
        off = measure_hst_relative_offset_from_phot(
            raw,
            parent,
            jhat_outdir=out,
            match_radius_arcsec=float(match_radius_arcsec),
            min_matches=int(min_matches),
            nbright=int(nbright),
        )
    # Pixel DAO fallback only when phot catalogs are unavailable. Full-frame
    # detection on ACS/WFC is too slow for the narrowband recovery path.
    if not off.get('ok') and off.get('n_match', 0) == 0:
        child_phot = find_jhat_phot(raw, out) or find_jhat_phot(jhat, out)
        parent_phot = find_jhat_phot(parent, out)
        if child_phot is None or parent_phot is None:
            off = measure_hst_frame_relative_offset_pixel(
                jhat,
                parent,
                min_matches=int(min_matches),
                max_match_pix=float(max_match_pix),
                nbright=int(nbright),
            )
    result['n_match'] = int(off.get('n_match') or 0)
    if not off.get('ok'):
        try:
            jhat.unlink(missing_ok=True)
        except Exception:
            pass
        result['outpath'] = None
        result['error'] = (
            f'relative match failed vs {parent.name} '
            f'(n_match={result["n_match"]}, min={min_matches}, '
            f'method={off.get("method")})'
        )
        return result

    dra_deg = -float(off['dra_deg'])
    ddec_deg = -float(off['ddec_deg'])
    with as_datamodel(jhat).open(mode='update', memmap=False) as hdul:
        apply_sky_translation_to_sci(
            hdul,
            dra_deg,
            ddec_deg,
            comment='st123: narrowband HST_REL vs broadband',
        )
        hdul.flush()
    # Applied shift is the inverse of the measured img-ref offset.
    write_hst_narrowband_provenance(
        jhat,
        align_mode='HST_REL',
        aligned_to=str(parent),
        n_match=int(off['n_match']),
        abs_arcsec=float(off['abs_arcsec']),
        dra_arcsec=-float(off['dra_arcsec']),
        ddec_arcsec=-float(off['ddec_arcsec']),
    )
    try:
        propagate_jhat_wcs_to_all_sci(jhat, raw)
    except Exception as exc:
        log.warning(
            'narrowband multi-SCI WCS propagation failed for %s: %s',
            jhat.name,
            exc,
        )
    result.update(
        {
            'ok': True,
            'abs_arcsec': float(off['abs_arcsec']),
            'dra_arcsec': -float(off['dra_arcsec']),
            'ddec_arcsec': -float(off['ddec_arcsec']),
            'error': None,
        }
    )
    return result


def align_hst_narrowband_pipeline_fallback(
    raw_path: str | Path,
    outdir: str | Path,
) -> dict[str, Any]:
    """
    Last-resort narrowband product: calibrated WCS copied to ``*_jhat.fits``.

    Parameters
    ----------
    raw_path : str or pathlib.Path
        Narrowband calibrated frame.
    outdir : str or pathlib.Path
        JHAT output directory.

    Returns
    -------
    dict
        Result with ``ok``, ``outpath``, and ``align_mode='PIPELINE'``.
    """
    jhat = write_hst_jhat_from_raw(raw_path, outdir)
    write_hst_narrowband_provenance(
        jhat,
        align_mode='PIPELINE',
        aligned_to='NA',
    )
    return {
        'ok': True,
        'outpath': str(jhat),
        'align_mode': 'PIPELINE',
        'aligned_to': 'NA',
        'n_match': 0,
        'abs_arcsec': None,
        'residual_arcsec': None,
        'error': None,
    }


def measure_hst_narrowband_residual_vs_refcat(
    jhat_path: str | Path,
    refcat: str | Path,
    *,
    match_radius_arcsec: float = 0.5,
    nbright: int = 500,
) -> dict[str, Any]:
    """
    Post-alignment residual of a JHAT product vs an L3 / broadband refcat.

    Projects the frame's ``*.phot.txt`` x/y through the current SCI WCS and
    matches to *refcat* ra/dec. Reports median separation of good matches -
    this is the quality metric (dispersion), not the bootstrap offset.

    Parameters
    ----------
    jhat_path : str or pathlib.Path
        Narrowband JHAT product.
    refcat : str or pathlib.Path
        Reference catalog with ``ra`` / ``dec`` columns.
    match_radius_arcsec : float, optional
        Match radius for residual measurement.
    nbright : int, optional
        Brightest sources to keep from the science phot table.

    Returns
    -------
    dict
        ``ok``, ``n_match``, ``residual_arcsec`` (median sep), offset fields.
    """
    import numpy as np
    import pandas as pd
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    import astropy.units as u

    stats: dict[str, Any] = {
        'ok': False,
        'n_match': 0,
        'residual_arcsec': None,
        'residual_mean_arcsec': None,
        'residual_std_arcsec': None,
        'dra_arcsec': 0.0,
        'ddec_arcsec': 0.0,
        'dra_deg': 0.0,
        'ddec_deg': 0.0,
    }
    jhat = Path(jhat_path).expanduser().resolve()
    ref_path = Path(refcat).expanduser().resolve()
    if not jhat.is_file() or not ref_path.is_file():
        return stats
    stem = jhat.name.replace('_jhat.fits', '').replace('.fits', '')
    phot_path = jhat.parent / f'{stem}.phot.txt'
    if not phot_path.is_file():
        return stats
    try:
        phot = pd.read_csv(phot_path, sep=r'\s+', engine='python')
        ref = pd.read_csv(ref_path, sep=r'\s+', engine='python')
    except Exception:
        return stats
    if 'x' not in phot.columns or 'y' not in phot.columns:
        return stats
    if 'ra' not in ref.columns or 'dec' not in ref.columns:
        return stats
    if 'mag' in phot.columns:
        phot = phot.sort_values('mag').head(int(nbright))
    else:
        phot = phot.head(int(nbright))

    with as_datamodel(jhat).open(memmap=True) as hdul:
        idxs = _sci_hdu_indices(hdul)
        if not idxs:
            return stats
        w = WCS(hdul[idxs[0]].header, hdul, naxis=2)
        ny, nx = np.asarray(hdul[idxs[0]].data).shape[-2:]
    x = np.asarray(phot['x'], dtype=float)
    y = np.asarray(phot['y'], dtype=float)
    on = (x >= 0) & (x < nx) & (y >= 0) & (y < ny) & np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(on)) < 3:
        return stats
    ra_img, dec_img = w.pixel_to_world_values(x[on], y[on])
    c_img = SkyCoord(
        np.asarray(ra_img, dtype=float) * u.deg,
        np.asarray(dec_img, dtype=float) * u.deg,
    )
    c_ref = SkyCoord(
        np.asarray(ref['ra'], dtype=float) * u.deg,
        np.asarray(ref['dec'], dtype=float) * u.deg,
    )
    idx, sep, _ = c_img.match_to_catalog_sky(c_ref)
    good = sep.arcsec < float(match_radius_arcsec)
    n = int(np.count_nonzero(good))
    stats['n_match'] = n
    if n < 3:
        return stats
    # ref - img (apply to CRVAL to move image onto ref)
    dra = (c_ref.ra[idx] - c_img.ra).to(u.deg).value
    ddec = (c_ref.dec[idx] - c_img.dec).to(u.deg).value
    dra_m = float(np.median(dra[good]))
    ddec_m = float(np.median(ddec[good]))
    dec0 = float(np.median(np.asarray(dec_img, dtype=float)[good]))
    sep_good = np.asarray(sep.arcsec[good], dtype=float)
    stats.update(
        {
            'ok': True,
            'residual_arcsec': float(np.median(sep_good)),
            'residual_mean_arcsec': float(np.mean(sep_good)),
            'residual_std_arcsec': float(np.std(sep_good)),
            'dra_deg': dra_m,
            'ddec_deg': ddec_m,
            'dra_arcsec': dra_m * 3600.0 * float(np.cos(np.radians(dec0))),
            'ddec_arcsec': ddec_m * 3600.0,
            'abs_arcsec': float(
                np.hypot(
                    dra_m * 3600.0 * float(np.cos(np.radians(dec0))),
                    ddec_m * 3600.0,
                )
            ),
        }
    )
    return stats


def refine_hst_narrowband_to_refcat(
    jhat_path: str | Path,
    refcat: str | Path,
    *,
    stages: tuple[tuple[float, float], ...] = (
        (2.0, 5.0),
        (1.0, 1.5),
        (0.5, 0.6),
        (0.25, 0.35),
        (0.15, 0.25),
    ),
    min_matches: int = 8,
    per_chip: bool = True,
) -> dict[str, Any]:
    """
    Multi-pass refine of a narrowband JHAT onto an L3 / broadband refcat.

    Sequence:
    1. Iterative CRVAL stages ``(match_radius_arcsec, max_apply_arcsec)``
       using phot x/y projected through the current SCI WCS (not stale phot
       ra/dec, which still reflect the pipeline WCS after HST_REL).
    2. Per-SCI CRPIX centroid refine vs the same refcat (chip-2 / ACS-WFC).
    3. A final tight CRVAL polish, then residual measurement at 0.15-0.25".

    Parameters
    ----------
    jhat_path : str or pathlib.Path
        Narrowband JHAT product (updated in place).
    refcat : str or pathlib.Path
        Gaia-anchored L3 or broadband science catalog.
    stages : tuple of (float, float)
        Match radius and max residual to apply per pass.
    min_matches : int, optional
        Minimum matches required for a pass to apply.
    per_chip : bool, optional
        Run :func:`refine_hst_wcs_per_chip_from_refcat` after CRVAL passes.

    Returns
    -------
    dict
        ``ok``, ``n_passes``, final residual / n_match, per-pass log.
    """

    jhat = Path(jhat_path).expanduser().resolve()
    ref_path = Path(refcat).expanduser().resolve()
    report: dict[str, Any] = {
        'ok': False,
        'path': str(jhat),
        'refcat': str(ref_path),
        'n_passes': 0,
        'passes': [],
        'per_chip': None,
        'n_match': 0,
        'residual_arcsec': None,
    }
    if not jhat.is_file() or not ref_path.is_file():
        report['error'] = 'jhat or refcat missing'
        return report

    def _crval_stages(stage_list: tuple[tuple[float, float], ...]) -> None:
        for match_rad, max_apply in stage_list:
            meas = measure_hst_narrowband_residual_vs_refcat(
                jhat,
                ref_path,
                match_radius_arcsec=float(match_rad),
            )
            pass_row = {
                'match_radius_arcsec': float(match_rad),
                'max_apply_arcsec': float(max_apply),
                'n_match': meas.get('n_match'),
                'residual_arcsec': meas.get('residual_arcsec'),
                'abs_arcsec': meas.get('abs_arcsec'),
                'applied': False,
            }
            if not meas.get('ok') or meas.get('abs_arcsec') is None:
                report['passes'].append(pass_row)
                continue
            shift = float(meas['abs_arcsec'])
            # Prefer denser matches at tight radii, but still allow a small
            # coherent polish when n is modest (narrowband cores are sparse).
            need = int(min_matches)
            if float(match_rad) <= 0.15:
                need = 4 if shift <= 0.10 else max(need, 10)
            elif float(match_rad) <= 0.25:
                need = max(need, 8)
            if int(meas.get('n_match') or 0) < need:
                pass_row['skipped_sparse'] = True
                report['passes'].append(pass_row)
                continue
            if shift < 0.005 or shift > float(max_apply):
                report['passes'].append(pass_row)
                report['n_match'] = int(meas['n_match'])
                report['residual_arcsec'] = float(meas['residual_arcsec'])
                continue
            # Guard: only apply if a tight (0.15") residual does not get worse.
            pre_tight = measure_hst_narrowband_residual_vs_refcat(
                jhat, ref_path, match_radius_arcsec=0.15
            )
            with as_datamodel(jhat).open(mode='update', memmap=False) as hdul:
                apply_sky_translation_to_sci(
                    hdul,
                    float(meas['dra_deg']),
                    float(meas['ddec_deg']),
                    comment='st123: narrowband L3 refine',
                )
                hdul[0].header['ST123NRF'] = (
                    True,
                    'st123: narrowband iterative refine vs L3/refcat',
                )
                hdul.flush()
            post_tight = measure_hst_narrowband_residual_vs_refcat(
                jhat, ref_path, match_radius_arcsec=0.15
            )
            pre_r = pre_tight.get('residual_arcsec')
            post_r = post_tight.get('residual_arcsec')
            # Revert if we lose the tight core or inflate its residual.
            revert = False
            if pre_tight.get('ok') and pre_r is not None:
                if not post_tight.get('ok'):
                    revert = True
                elif post_r is not None and post_r > pre_r + 0.015:
                    revert = True
                elif int(post_tight.get('n_match') or 0) + 2 < int(
                    pre_tight.get('n_match') or 0
                ):
                    revert = True
            if revert:
                with as_datamodel(jhat).open(mode='update', memmap=False) as hdul:
                    apply_sky_translation_to_sci(
                        hdul,
                        -float(meas['dra_deg']),
                        -float(meas['ddec_deg']),
                        comment='st123: revert narrowband refine',
                    )
                    hdul.flush()
                pass_row['reverted'] = True
                report['passes'].append(pass_row)
                continue
            pass_row['applied'] = True
            report['n_passes'] += 1
            report['passes'].append(pass_row)
            report['n_match'] = int(
                (post_tight if post_tight.get('ok') else meas).get('n_match') or 0
            )
            report['residual_arcsec'] = float(
                (post_tight if post_tight.get('ok') else meas)['residual_arcsec']
            )
            log.info(
                'Narrowband refine %s: apply |Delta|=%.3f" (n=%d, rad=%.2f")',
                jhat.name,
                shift,
                meas['n_match'],
                match_rad,
            )

    # Choose stage ladder from the current residual - avoid re-opening a
    # wide match radius once the solution is already near-broadband.
    probe = measure_hst_narrowband_residual_vs_refcat(
        jhat, ref_path, match_radius_arcsec=0.5
    )
    probe_abs = float(probe.get('abs_arcsec') or 99.0)
    probe_res = float(probe.get('residual_arcsec') or 99.0)
    if probe.get('ok') and probe_abs < 0.04 and probe_res < 0.08:
        active_stages: tuple[tuple[float, float], ...] = (
            (0.15, 0.12),
            (0.10, 0.08),
        )
    elif probe.get('ok') and probe_abs < 0.25 and probe_res < 0.35:
        active_stages = (
            (0.5, 0.35),
            (0.25, 0.25),
            (0.15, 0.15),
            (0.10, 0.10),
        )
    else:
        active_stages = stages
    _crval_stages(active_stages)

    if per_chip:
        # Wider search than broadband: narrowband continuum is sparse and the
        # sibling bootstrap can leave ~0.1-0.5" chip residuals.
        chip = refine_hst_wcs_per_chip_from_refcat(
            jhat,
            ref_path,
            search_radius_arcsec=1.5,
            min_matches=5,
            max_shift_arcsec=2.0,
        )
        report['per_chip'] = {
            'n_updated': chip.get('n_updated'),
            'n_sci': chip.get('n_sci'),
            'chips': chip.get('chips'),
        }
        # Only re-polish CRVAL when CRPIX actually moved - otherwise the
        # extra medium-radius stages can walk a good solution off L3.
        if int(chip.get('n_updated') or 0) > 0:
            _crval_stages(((0.25, 0.20), (0.15, 0.12), (0.10, 0.08)))

    final = measure_hst_narrowband_residual_vs_refcat(
        jhat, ref_path, match_radius_arcsec=0.15
    )
    if not final.get('ok'):
        final = measure_hst_narrowband_residual_vs_refcat(
            jhat, ref_path, match_radius_arcsec=0.25
        )
    if not final.get('ok'):
        final = measure_hst_narrowband_residual_vs_refcat(
            jhat, ref_path, match_radius_arcsec=0.5
        )
    if final.get('ok'):
        # Treat residual quality as "ok" when median sep is at most moderately
        # worse than typical broadband (~0.05"); allow up to ~0.12" (~3 UVIS px).
        report['ok'] = float(final['residual_arcsec']) <= 0.12
        report['n_match'] = int(final['n_match'])
        report['residual_arcsec'] = float(final['residual_arcsec'])
        report['dra_arcsec'] = float(final['dra_arcsec'])
        report['ddec_arcsec'] = float(final['ddec_arcsec'])
        with as_datamodel(jhat).open(mode='update', memmap=False) as hdul:
            hdul[0].header['ST123NRM'] = (
                float(final['residual_arcsec']),
                '[arcsec] narrowband residual vs refcat',
            )
            hdul[0].header['ST123NNM'] = (
                int(final['n_match']),
                'n_match for ST123NRM residual',
            )
            hdul[0].header['ALGNTO'] = (
                str(ref_path),
                'Narrowband aligned-to parent',
            )
            hdul.flush()
    return report


def finalize_hst_narrowband_group(
    jhat_paths: list[Path],
    *,
    abs_ref: str | Path | None = None,
    refcat: str | Path | None = None,
    max_internal_arcsec: float = 0.20,
) -> dict[str, Any]:
    """
    L3-first finalize for a narrowband visit/filter group.

    Order matters for sparse narrowbands:
    1. Per-frame iterative refine onto *refcat* (primary absolute tie).
    2. Optional small internal relative CRVAL tweaks (pixel match only;
       capped - never use stale pipeline phot ra/dec).
    3. Optional common abs retie vs *abs_ref* image only (no Gaia fallback;
       Gaia is too easy to mismatch on emission-line frames).
    4. Final per-frame L3 polish + residual measurement.

    Parameters
    ----------
    jhat_paths : list of pathlib.Path
        Narrowband JHAT products (same filter / visit).
    abs_ref : str or pathlib.Path, optional
        Deep L3 / broadband image for 2-D-hist absolute retie.
    refcat : str or pathlib.Path, optional
        L3 / broadband phot catalog (required for science-grade residuals).
    max_internal_arcsec : float, optional
        Reject sibling relative corrections larger than this (bad matches).

    Returns
    -------
    dict
        Harmonize / abs-retie / residual summary.
    """

    paths = [Path(p).resolve() for p in jhat_paths if Path(p).is_file()]
    report: dict[str, Any] = {
        'ok': False,
        'n_frames': len(paths),
        'internal': None,
        'abs_retie': None,
        'residuals': [],
    }
    if len(paths) < 1:
        return report

    refcat_path = Path(refcat).resolve() if refcat else None
    if refcat_path is not None and not refcat_path.is_file():
        refcat_path = None

    # 1) Primary absolute: per-frame L3 refine (before any group shifts).
    if refcat_path is not None:
        for path in paths:
            polish = refine_hst_narrowband_to_refcat(path, refcat_path)
            report['residuals'].append(
                {
                    'path': str(path),
                    'stage': 'pre_group',
                    'ok': polish.get('ok'),
                    'n_match': polish.get('n_match'),
                    'residual_arcsec': polish.get('residual_arcsec'),
                    'n_passes': polish.get('n_passes'),
                }
            )

    # 2) Small internal relative (pixel match only; reject large jumps).
    if len(paths) >= 2:
        anchor = max(paths, key=_frame_exptime)
        corrections = []
        rejected = []
        for path in paths:
            if path == anchor:
                continue
            off = measure_hst_frame_relative_offset_pixel(
                path, anchor, min_matches=8, max_match_pix=8.0
            )
            if not off.get('ok'):
                rejected.append(
                    {
                        'path': str(path),
                        'reason': 'pixel_match_failed',
                        'n_match': off.get('n_match'),
                    }
                )
                continue
            shift = float(off.get('abs_arcsec') or 0.0)
            if shift > float(max_internal_arcsec):
                rejected.append(
                    {
                        'path': str(path),
                        'reason': 'shift_too_large',
                        'abs_arcsec': shift,
                        'n_match': off.get('n_match'),
                    }
                )
                continue
            with as_datamodel(path).open(mode='update', memmap=False) as hdul:
                apply_sky_translation_to_sci(
                    hdul,
                    -float(off['dra_deg']),
                    -float(off['ddec_deg']),
                    comment='st123: narrowband internal relative',
                )
                hdul.flush()
            corrections.append(
                {
                    'path': str(path),
                    'dra_arcsec': -float(off['dra_arcsec']),
                    'ddec_arcsec': -float(off['ddec_arcsec']),
                    'n_match': off.get('n_match'),
                }
            )
        report['internal'] = {
            'anchor': str(anchor),
            'corrections': corrections,
            'rejected': rejected,
            'post': validate_hst_group_internal_alignment(
                paths,
                min_matches=5,
                max_coherent_arcsec=max(HST_INTERNAL_ALIGN_MAX_ARCSEC, 0.15),
            ),
        }

    # 3) Common abs vs deep image only - never Gaia on narrowbands.
    abs_path = Path(abs_ref).resolve() if abs_ref else None
    if abs_path is not None and abs_path.is_file():
        report['abs_retie'] = apply_common_abs_shift_vs_ref(
            paths,
            abs_path,
            max_abs_offset_arcsec=1.0,
            allow_gaia_fallback=False,
        )

    # 4) Final L3 polish only if group-level shifts actually moved anything.
    n_group_moves = 0
    if report.get('internal'):
        n_group_moves += len(report['internal'].get('corrections') or [])
    if (report.get('abs_retie') or {}).get('applied'):
        n_group_moves += 1
    final_residuals = []
    if refcat_path is not None:
        if n_group_moves:
            for path in paths:
                polish = refine_hst_narrowband_to_refcat(path, refcat_path)
                final_residuals.append(
                    {
                        'path': str(path),
                        'stage': 'final',
                        'ok': polish.get('ok'),
                        'n_match': polish.get('n_match'),
                        'residual_arcsec': polish.get('residual_arcsec'),
                        'n_passes': polish.get('n_passes'),
                    }
                )
        else:
            # Re-measure only; do not re-run wide-stage refine.
            for path in paths:
                meas = measure_hst_narrowband_residual_vs_refcat(
                    path, refcat_path, match_radius_arcsec=0.15
                )
                if not meas.get('ok'):
                    meas = measure_hst_narrowband_residual_vs_refcat(
                        path, refcat_path, match_radius_arcsec=0.25
                    )
                final_residuals.append(
                    {
                        'path': str(path),
                        'stage': 'final',
                        'ok': bool(
                            meas.get('ok')
                            and meas.get('residual_arcsec') is not None
                            and float(meas['residual_arcsec']) <= 0.12
                        ),
                        'n_match': meas.get('n_match'),
                        'residual_arcsec': meas.get('residual_arcsec'),
                        'n_passes': 0,
                    }
                )
        report['residuals'] = final_residuals

    res_ok = [
        float(r['residual_arcsec'])
        for r in report['residuals']
        if r.get('residual_arcsec') is not None and r.get('stage', 'final') == 'final'
    ]
    if not res_ok:
        res_ok = [
            float(r['residual_arcsec'])
            for r in report['residuals']
            if r.get('residual_arcsec') is not None
        ]
    # ~3 UVIS pixels; broadband is typically tighter (~0.05").
    report['ok'] = bool(res_ok) and max(res_ok) <= 0.12
    report['max_residual_arcsec'] = max(res_ok) if res_ok else None
    report['med_residual_arcsec'] = (
        float(sorted(res_ok)[len(res_ok) // 2]) if res_ok else None
    )
    return report


def polish_hst_narrowband_products(
    jhat_dir: str | Path,
    *,
    refcat: str | Path | None = None,
    abs_ref: str | Path | None = None,
    paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """
    Re-refine existing narrowband JHAT products onto L3 (post-hoc polish).

    Use after an earlier HST_REL bootstrap left ~arcsecond match residuals vs
    the sibling parent: this runs internal relative + abs retie + iterative
    L3 refine so residuals approach broadband quality.

    Parameters
    ----------
    jhat_dir : str or pathlib.Path
        JHAT product directory (also searched for ``l3_ref`` / abs ref).
    refcat, abs_ref : str or pathlib.Path, optional
        Override L3 phot catalog / deep abs image.
    paths : sequence of paths, optional
        Explicit JHAT products; default = all ``*_jhat.fits`` with
        ``ST123NAR`` / narrowband filter.

    Returns
    -------
    dict
        Per-visit/filter finalize reports under ``groups``.
    """
    from st123.utils.helpers import get_filter

    out = Path(jhat_dir).expanduser().resolve()
    refcat_path = Path(refcat).resolve() if refcat else find_hst_l3_refcat(out)
    abs_ref_path = Path(abs_ref).resolve() if abs_ref else find_hst_abs_ref_image(out)
    if refcat_path is not None and not Path(refcat_path).is_file():
        refcat_path = None
    if abs_ref_path is not None and not Path(abs_ref_path).is_file():
        abs_ref_path = None

    candidates: list[Path] = []
    if paths is not None:
        candidates = [Path(p).resolve() for p in paths if Path(p).is_file()]
    else:
        for path in sorted(out.glob('*_jhat.fits')):
            try:
                with as_datamodel(path).open(memmap=True) as hdul:
                    if hdul[0].header.get('ST123NAR'):
                        candidates.append(path.resolve())
                        continue
                if is_hst_narrowband_filter(get_filter(path)):
                    candidates.append(path.resolve())
            except Exception:
                continue

    by_key: dict[tuple[str, str], list[Path]] = {}
    for path in candidates:
        try:
            filt = str(get_filter(path)).lower()
        except Exception:
            continue
        if not is_hst_narrowband_filter(filt):
            continue
        by_key.setdefault((_hst_visit_key(path), filt), []).append(path)

    report: dict[str, Any] = {
        'ok': True,
        'refcat': str(refcat_path) if refcat_path else None,
        'abs_ref': str(abs_ref_path) if abs_ref_path else None,
        'n_frames': len(candidates),
        'groups': [],
    }
    if not by_key:
        report['ok'] = False
        report['error'] = 'no narrowband JHAT products found'
        return report

    for (visit, filt), group_paths in sorted(by_key.items()):
        uniq = sorted({p.resolve() for p in group_paths})
        log.info(
            'Polish HST narrowband visit=%s filter=%s n=%d refcat=%s',
            visit,
            filt,
            len(uniq),
            Path(refcat_path).name if refcat_path else 'none',
        )
        grp = finalize_hst_narrowband_group(
            uniq, abs_ref=abs_ref_path, refcat=refcat_path
        )
        grp['visit'] = visit
        grp['filter'] = filt
        report['groups'].append(grp)
        if not grp.get('ok'):
            report['ok'] = False
    return report


def recover_failed_hst_narrowbands(
    results: list[dict],
    outdir: str | Path,
    *,
    pipeline_fallback: bool = True,
    refcat: str | Path | None = None,
    abs_ref: str | Path | None = None,
) -> list[dict]:
    """
    Recover failed narrowband frames via HST_REL + L3 refine, else PIPELINE.

    Workflow per failed narrowband:
    1. Bootstrap HST_REL onto a broadband JHAT parent (large CRVAL shift).
    2. Iterative refine onto *refcat* (Gaia-anchored L3 phot) when provided.
    3. After all recoveries, visit/filter groups get internal relative
       harmonize + common absolute retie to *abs_ref* / *refcat*.
    4. PIPELINE copy only if relative+refine both fail.

    Mutates *results* in place for recovered rows and returns the same list.

    Parameters
    ----------
    results : list of dict
        Per-frame batch results from :func:`align_hst_raw_dir`.
    outdir : str or pathlib.Path
        JHAT output directory.
    pipeline_fallback : bool, optional
        When relative matching fails, write a pipeline-WCS JHAT so DOLPHOT
        still has a product.
    refcat : str or pathlib.Path, optional
        L3 / broadband phot catalog for iterative refine.
    abs_ref : str or pathlib.Path, optional
        Deep L3 image for group absolute retie (2-D hist).

    Returns
    -------
    list of dict
        The updated *results* list.
    """
    from st123.utils.helpers import get_filter

    out = Path(outdir).expanduser().resolve()
    refcat_path = Path(refcat).resolve() if refcat else None
    if refcat_path is not None and not refcat_path.is_file():
        refcat_path = None
    abs_ref_path = Path(abs_ref).resolve() if abs_ref else None
    if abs_ref_path is not None and not abs_ref_path.is_file():
        abs_ref_path = None

    ok_rows = [
        r
        for r in results
        if r.get('status') == 'ok' and r.get('outpath')
    ]
    recovered_nb: list[Path] = []
    for entry in results:
        if entry.get('status') == 'ok' and entry.get('outpath'):
            continue
        raw = Path(entry.get('path') or '')
        if not raw.is_file():
            continue
        try:
            filt = get_filter(raw)
        except Exception:
            continue
        if not is_hst_narrowband_filter(filt):
            continue

        parents = rank_hst_narrowband_parents(raw, ok_rows, max_parents=5)
        recovered: dict[str, Any] | None = None
        rel_errors: list[str] = []
        for parent in parents:
            log.info(
                'HST narrowband HST_REL: %s (%s) -> parent %s',
                raw.name,
                filt,
                parent.name,
            )
            recovered = align_hst_narrowband_relative(raw, parent, out)
            if recovered.get('ok'):
                break
            if recovered.get('error'):
                rel_errors.append(str(recovered['error']))
            log.warning(
                'HST narrowband HST_REL failed for %s: %s',
                raw.name,
                recovered.get('error'),
            )
        else:
            recovered = None

        if recovered is not None and recovered.get('ok'):
            jhat_path = Path(recovered['outpath'])
            residual = None
            n_match = recovered.get('n_match')
            if refcat_path is not None:
                polish = refine_hst_narrowband_to_refcat(jhat_path, refcat_path)
                if polish.get('ok'):
                    residual = polish.get('residual_arcsec')
                    n_match = polish.get('n_match')
                    # Keep ST123NAS as the large bootstrap offset; residual
                    # quality lives in ST123NRM (set by refine).
                    log.info(
                        'HST narrowband L3 refine ok %s residual=%.3f" (n=%s)',
                        raw.name,
                        float(residual or 0.0),
                        n_match,
                    )
                else:
                    log.warning(
                        'HST narrowband L3 refine weak for %s (n=%s residual=%s)',
                        raw.name,
                        polish.get('n_match'),
                        polish.get('residual_arcsec'),
                    )
                    residual = polish.get('residual_arcsec')
            entry['status'] = 'ok'
            entry['outpath'] = str(jhat_path)
            entry['error'] = None
            entry['align_mode'] = 'HST_REL'
            entry['aligned_to'] = (
                str(refcat_path)
                if refcat_path is not None
                else recovered.get('aligned_to')
            )
            entry['n_match'] = n_match
            # Quality metric: post-refine residual when available; else bootstrap.
            entry['residual_arcsec'] = residual
            entry['abs_arcsec'] = (
                residual
                if residual is not None
                else recovered.get('abs_arcsec')
            )
            ok_rows.append(entry)
            recovered_nb.append(jhat_path)
            continue

        if not pipeline_fallback:
            if rel_errors:
                entry['error'] = (
                    f'{entry.get("error") or "JHAT failed"}; '
                    + '; '.join(rel_errors[:3])
                )
            continue

        log.info(
            'HST narrowband PIPELINE fallback: %s (%s)',
            raw.name,
            filt,
        )
        pipe = align_hst_narrowband_pipeline_fallback(raw, out)
        entry['status'] = 'ok'
        entry['outpath'] = pipe['outpath']
        entry['align_mode'] = 'PIPELINE'
        entry['aligned_to'] = 'NA'
        entry['n_match'] = 0
        entry['abs_arcsec'] = None
        entry['residual_arcsec'] = None
        prior = entry.get('error')
        entry['error'] = None
        entry['warning'] = (
            f'PIPELINE WCS kept after JHAT/HST_REL failure'
            + (f' ({prior})' if prior else '')
        )
        ok_rows.append(entry)
        log.warning(
            'HST narrowband PIPELINE product written for %s (not science-grade abs)',
            raw.name,
        )

    # Visit/filter group finalize for all HST_REL narrowband products in this batch.
    nb_by_key: dict[tuple[str, str], list[Path]] = {}
    for entry in results:
        if entry.get('align_mode') != 'HST_REL' or not entry.get('outpath'):
            continue
        path = Path(entry['outpath'])
        if not path.is_file():
            continue
        try:
            filt = get_filter(path)
        except Exception:
            try:
                filt = get_filter(entry['path'])
            except Exception:
                filt = 'unknown'
        key = (_hst_visit_key(path), str(filt).lower())
        nb_by_key.setdefault(key, []).append(path)

    for (visit, filt), paths in nb_by_key.items():
        # Include any already-ok narrowbands of the same visit/filter.
        for entry in results:
            if entry.get('status') != 'ok' or not entry.get('outpath'):
                continue
            p = Path(entry['outpath'])
            if p in paths or not p.is_file():
                continue
            try:
                if str(get_filter(p)).lower() != filt:
                    continue
            except Exception:
                continue
            if _hst_visit_key(p) != visit:
                continue
            if not is_hst_narrowband_filter(filt):
                continue
            paths.append(p)
        uniq = sorted({p.resolve() for p in paths})
        if not uniq:
            continue
        log.info(
            'HST narrowband group finalize visit=%s filter=%s n=%d',
            visit,
            filt,
            len(uniq),
        )
        grp = finalize_hst_narrowband_group(
            uniq, abs_ref=abs_ref_path, refcat=refcat_path
        )
        if grp.get('med_residual_arcsec') is not None:
            log.info(
                'HST narrowband group %s/%s residual med=%.3f" max=%.3f" (ok=%s)',
                visit,
                filt,
                grp['med_residual_arcsec'],
                grp.get('max_residual_arcsec') or -1.0,
                grp.get('ok'),
            )
        # Refresh per-row residual metrics from headers / group report.
        res_by = {
            Path(r['path']).resolve(): r
            for r in grp.get('residuals') or []
            if r.get('path')
        }
        for entry in results:
            if not entry.get('outpath'):
                continue
            rp = Path(entry['outpath']).resolve()
            if rp in res_by and res_by[rp].get('residual_arcsec') is not None:
                entry['residual_arcsec'] = res_by[rp]['residual_arcsec']
                entry['n_match'] = res_by[rp].get('n_match')
                entry['abs_arcsec'] = res_by[rp]['residual_arcsec']
                entry['aligned_to'] = (
                    str(refcat_path) if refcat_path is not None else entry.get('aligned_to')
                )
    return results


def install_jhat_pandas_read_table_compat() -> None:
    """
    JHAT calls ``pandas.read_table(..., delim_whitespace=...)`` via pdastro.

    pandas 2.2+ removed ``delim_whitespace``; patch ``pd.read_table`` once per process.
    """
    try:
        import inspect

        import pandas as pd  # type: ignore

        sig = inspect.signature(pd.read_table)
        if 'delim_whitespace' not in sig.parameters and not hasattr(
            pd, '_st123_read_table_compat'
        ):
            _orig_read_table = pd.read_table

            def _read_table_compat(*args, delim_whitespace=None, **kwargs):
                kwargs.pop('delim_whitespace', None)
                if delim_whitespace:
                    kwargs.setdefault('sep', r'\s+')
                    return pd.read_csv(*args, **kwargs)
                return _orig_read_table(*args, **kwargs)

            pd.read_table = _read_table_compat  # type: ignore[assignment]
            pd._st123_read_table_compat = True  # type: ignore[attr-defined]
    except Exception:
        pass


def install_scipy_interp2d_compat() -> None:
    """
    Provide a minimal ``scipy.interpolate.interp2d`` shim via RectBivariateSpline.

    SciPy 1.14+ keeps an ``interp2d`` stub that raises ``NotImplementedError``;
    vendored JHAT still calls it in ``hst_get_ee_corr``. Idempotent.
    """
    try:
        import numpy as np
        import scipy.interpolate as si
    except Exception:
        return
    if getattr(si, '_st123_interp2d_shim', False):
        return

    # Detect a working legacy interp2d (pre-1.14). If construction succeeds,
    # leave SciPy alone.
    try:
        probe = si.interp2d([0.0, 1.0], [0.0, 1.0], [[0.0, 1.0], [1.0, 2.0]])
        _ = probe(0.5, 0.5)
        return
    except Exception:
        pass

    class _Interp2dCompat:
        def __init__(self, x, y, z, *args, **kwargs):
            x = np.asarray(x, dtype=float).ravel()
            y = np.asarray(y, dtype=float).ravel()
            z = np.asarray(z, dtype=float)
            # Match legacy interp2d: z shaped (len(y), len(x)).
            if z.ndim == 2 and z.shape == (len(x), len(y)):
                z = z.T
            xidx = np.argsort(x)
            yidx = np.argsort(y)
            xs = x[xidx]
            ys = y[yidx]
            zs = z[yidx][:, xidx]
            kx = int(min(3, max(1, len(xs) - 1)))
            ky = int(min(3, max(1, len(ys) - 1)))
            self._spline = si.RectBivariateSpline(xs, ys, zs, kx=kx, ky=ky)

        def __call__(self, x, y, *args, **kwargs):
            return np.asarray(self._spline(x, y), dtype=float)

    si.interp2d = _Interp2dCompat  # type: ignore[attr-defined]
    si._st123_interp2d_shim = True  # type: ignore[attr-defined]


def _jhat_ee_calibration_dir() -> str:
    base = os.environ.get('XDG_CACHE_HOME') or os.path.join(
        os.path.expanduser('~'), '.cache'
    )
    d = os.path.join(base, 'st123', 'jhat_ee')
    os.makedirs(d, exist_ok=True)
    return d


def _install_hst_get_ee_corr_patch(sjp: Any) -> None:
    """Replace ``hst_get_ee_corr`` with a SciPy-1.14-safe, cache-dir version."""
    if getattr(getattr(sjp, 'hst_get_ee_corr', None), '__name__', '') == _EE_PATCH_MARKER:
        return
    _orig = getattr(sjp, 'hst_get_ee_corr', None)

    def _hst_get_ee_corr_st123(ap, pxscale, filt, inst):
        try:
            import urllib.request

            import numpy as np
            import scipy
            from astropy.table import Table

            ee_base = _jhat_ee_calibration_dir()
            if str(inst).lower() == 'ir':
                ir_path = os.path.join(ee_base, 'ir_ee_corrections.csv')
                if not os.path.exists(ir_path):
                    urllib.request.urlretrieve(
                        'https://www.stsci.edu/files/live/sites/www/files/home/hst/'
                        'instrumentation/wfc3/data-analysis/photometric-calibration/'
                        'ir-encircled-energy/_documents/ir_ee_corrections.csv',
                        ir_path,
                    )
                ee = Table.read(ir_path, format='ascii')
                ee.rename_column('PIVOT', 'WAVELENGTH')
            else:
                uvis_path = os.path.join(ee_base, 'wfc3uvis2_aper_007_syn.csv')
                if not os.path.exists(uvis_path):
                    urllib.request.urlretrieve(
                        'https://www.stsci.edu/files/live/sites/www/files/home/hst/'
                        'instrumentation/wfc3/data-analysis/photometric-calibration/'
                        'uvis-encircled-energy/_documents/wfc3uvis2_aper_007_syn.csv',
                        uvis_path,
                    )
                ee = Table.read(uvis_path, format='ascii')
                if str(filt).upper() not in [str(x).upper() for x in ee['FILTER']]:
                    bohlin_path = os.path.join(ee_base, 'bohlin2016_wfc_ee-1.txt')
                    if not os.path.exists(bohlin_path):
                        urllib.request.urlretrieve(
                            'https://www.stsci.edu/files/live/sites/www/files/home/hst/'
                            'instrumentation/acs/data-analysis/aperture-corrections/'
                            '_documents/bohlin2016_wfc_ee-1.txt',
                            bohlin_path,
                        )
                    ee = Table.read(bohlin_path, format='ascii', data_start=1)
                    ee.rename_column('col1', 'FILTER')
                    ee['WAVELENGTH'] = [
                        float(x[1:-1]) * 10 if len(x) == 5 else float(x[1:-2]) * 10
                        for x in ee['FILTER']
                    ]
                    px_cols = [
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 40.0
                    ]
                    n = 0
                    for col in list(ee.colnames):
                        if col in ('FILTER', 'WAVELENGTH'):
                            continue
                        ee.rename_column(col, '#' + str(pxscale * px_cols[n]))
                        n += 1

            filts = np.asarray(ee['FILTER'])
            ee.remove_column('FILTER')
            waves = np.asarray(ee['WAVELENGTH'], dtype=float)
            ee.remove_column('WAVELENGTH')
            colnames = list(ee.colnames)
            apps = np.asarray([float(x.split('#')[1]) for x in colnames], dtype=float)
            ee_arr = np.asarray(
                [np.asarray(ee[col], dtype=float) for col in colnames], dtype=float
            )
            aidx = np.argsort(apps)
            widx = np.argsort(waves)
            apps_s = apps[aidx]
            waves_s = waves[widx]
            ee_arr_s = ee_arr[aidx][:, widx]
            filts_s = filts[widx]
            if np.any(np.diff(waves_s) <= 0) or np.any(np.diff(apps_s) <= 0):
                return np.asarray([1.0], dtype=float)
            interp = scipy.interpolate.RectBivariateSpline(
                waves_s, apps_s, ee_arr_s.T
            )
            m = np.where(np.char.upper(filts_s.astype(str)) == str(filt).upper())[0]
            if m.size == 0:
                return np.asarray([1.0], dtype=float)
            filt_wave = float(waves_s[m[0]])
            return interp(filt_wave, float(ap) * float(pxscale)).flatten()
        except Exception:
            try:
                if _orig is not None:
                    return _orig(ap, pxscale, filt, inst)
            except Exception:
                pass
            import numpy as np

            return np.asarray([1.0], dtype=float)

    _hst_get_ee_corr_st123.__name__ = _EE_PATCH_MARKER
    try:
        sjp.hst_get_ee_corr = _hst_get_ee_corr_st123  # type: ignore[attr-defined]
    except Exception:
        pass


def _mjd_avg_from_primary(primaryhdr: Any) -> float | None:
    """Representative MJD: PRIMARY ``MJD-AVG``, else mid-exposure from EXPSTART/EXPTIME."""
    try:
        if 'MJD-AVG' in primaryhdr:
            return float(primaryhdr['MJD-AVG'])
    except Exception:
        pass
    try:
        if 'EXPSTART' in primaryhdr:
            start = float(primaryhdr['EXPSTART'])
            if 'EXPTIME' in primaryhdr:
                try:
                    return start + 0.5 * float(primaryhdr['EXPTIME']) / 86400.0
                except Exception:
                    pass
            if 'EXPEND' in primaryhdr:
                try:
                    return 0.5 * (start + float(primaryhdr['EXPEND']))
                except Exception:
                    pass
            return start
    except Exception:
        pass
    return None


def _ensure_jhat_scihdr_mjd_avg(primaryhdr: Any, scihdr: Any) -> None:
    """Inject ``MJD-AVG`` into the SCI header when missing (Gaia proper motion)."""
    try:
        if scihdr is None or 'MJD-AVG' in scihdr:
            return
    except Exception:
        return
    mjd = _mjd_avg_from_primary(primaryhdr)
    if mjd is None:
        return
    try:
        scihdr['MJD-AVG'] = (float(mjd), 'Representative MJD (mid-exposure, st123)')
    except Exception:
        return


def wfpc2_filter_key_and_name(primaryhdr: Any) -> tuple[str, str]:
    """
    Return ``(header_key, filter_string)`` for WFPC2 primary headers.

    Avoids ``'CLEAR' in FILTER1`` when ``FILTER1`` is not a string (JHAT bug).
    """
    for key in ('FILTNAM1', 'FILTNAM2', 'FILTER'):
        if key not in primaryhdr:
            continue
        raw = primaryhdr[key]
        if raw is None:
            continue
        s = str(raw).strip()
        if not s or s.upper() in ('N/A', 'NONE', ''):
            continue
        if s.upper() == 'CLEAR' and key.startswith('FILTNAM'):
            continue
        return key, s
    return 'FILTNAM1', 'CLEAR'


def _wfpc2_hst_photclass_load_image(
    self,
    imagename: str,
    imagetype: str | None = None,
    DNunits: bool = False,
    use_dq: bool = False,
    skip_preparing: bool = False,
) -> None:
    """Replacement for ``hst_photclass.load_image`` when ``INSTRUME`` is WFPC2."""
    from astropy.io import fits as fits_mod
    from astropy import wcs as wcs_mod

    self.imagename = imagename
    self.im = fits_mod.open(imagename)
    self.primaryhdr = self.im['PRIMARY'].header
    try:
        self.scihdr = self.im['SCI'].header
    except KeyError as exc:
        self.im.close()
        raise RuntimeError(
            f'JHAT WFPC2 patch: no SCI extension in {imagename!r} '
            '(expected MEF with at least one SCI HDU).'
        ) from exc

    self.NAXIS1 = self.scihdr['NAXIS1']
    self.NAXIS2 = self.scihdr['NAXIS2']
    self.instrument = str(self.primaryhdr.get('INSTRUME', 'WFPC2')).strip()
    _ensure_jhat_scihdr_mjd_avg(self.primaryhdr, self.scihdr)

    fk, fn = wfpc2_filter_key_and_name(self.primaryhdr)
    self.filterkey = fk
    self.filtername = fn

    ap_raw = self.primaryhdr.get('APERTURE', 'LARGE')
    ap = str(ap_raw).replace('-', '')
    if 'ACS' in self.instrument.upper():
        self.aperture = 'J' + ap
    else:
        self.aperture = 'I' + ap

    psf_scalar = self.psf_fwhm
    self.filters = {self.instrument: [self.filtername]}
    self.psf_fwhm = {self.instrument: [psf_scalar]}
    self.dict_utils = {}
    for instrument in self.filters:
        self.dict_utils[instrument.upper()] = {
            self.filters[instrument.upper()][i]: {
                'psf fwhm': self.psf_fwhm[instrument.upper()][i]
            }
            for i in range(len(self.filters[instrument]))
        }

    self.sci_wcs = wcs_mod.WCS(self.scihdr, self.im)
    try:
        self.err = self.im['ERR'].data
    except Exception:
        self.err = None
    self.pixel_scale = (
        wcs_mod.utils.proj_plane_pixel_scales(self.sci_wcs)[0]
        * self.sci_wcs.wcs.cunit[0].to('arcsec')
    )

    if getattr(self, 'verbose', False):
        log.info(
            'JHAT WFPC2 patch: instrument=%s filter=%s aperture=%s',
            self.instrument,
            self.filtername,
            self.aperture,
        )

    if imagetype is None:
        if re.search(
            r'flt\.fits$|flc\.fits$|tweakregstep\.fits$|assignwcsstep\.fits$',
            imagename,
            re.I,
        ):
            self.imagetype = 'flc'
        elif re.search(r'drz\.fits$|drc\.fits$', imagename, re.I):
            self.imagetype = 'drz'
        elif re.search(r'c0m\.fits$', imagename, re.I):
            self.imagetype = 'wfpc2_c0m'
        else:
            self.im.close()
            raise RuntimeError(
                f'JHAT WFPC2 patch: unknown image type for file {imagename!r}'
            )
        # Skip ACS/WFC3 PAM / AstroDrizzle: use direct SCI data.
        self.pipeline_level = 3
        self.do_driz = False
    else:
        self.imagetype = imagetype
        self.pipeline_level = 3
        self.do_driz = False

    if not skip_preparing:
        (self.data, self.mask) = self.prepare_image(
            self.im['SCI'].data,
            self.im['SCI'].header,
            self.do_driz,
        )


def _install_match_refcat_bounds_patch(sjp: Any) -> None:
    """
    Retry ``match_refcat`` with expanded image bounds for HST when the first
    pass finds no in-bounds Gaia sources (common for poorly WCS'd WFPC2).
    """
    marker = '_match_refcat_st123'
    cur = getattr(sjp.hst_photclass, 'match_refcat', None)
    if not callable(cur) or getattr(cur, '__name__', '') == marker:
        return
    _orig = cur

    def _match_refcat_st123(self, *args, **kwargs):
        out = _orig(self, *args, **kwargs)
        try:
            tel = str(getattr(self, 'primaryhdr', {}).get('TELESCOP', '')).strip().upper()
        except Exception:
            tel = ''
        if tel != 'HST':
            return out
        if out not in (0, None):
            return out
        try:
            kw = dict(kwargs)
            kw['borderpadding'] = -10000
            kw.setdefault('max_sep', 5.0)
            return _orig(self, *args, **kw)
        except Exception:
            return out

    _match_refcat_st123.__name__ = marker
    sjp.hst_photclass.match_refcat = _match_refcat_st123  # type: ignore[assignment]


def ensure_wfpc2_jhat_patch() -> None:
    """
    Replace ``jhat.simple_jwst_phot.hst_photclass.load_image`` with a WFPC2-safe
    wrapper (idempotent; safe if ``jhat`` is re-imported).

    Also installs SciPy ``interp2d`` / EE-correction shims needed on SciPy >=1.14,
    a tolerant HST ``match_refcat`` retry for poor initial WCS, and the
    Gaia->Vizier patch (ESA TAP disabled).
    """
    install_scipy_interp2d_compat()
    from st123.stages.alignment.gaia_catalog import install_jhat_gaia_vizier_patch

    install_jhat_gaia_vizier_patch()
    import jhat.simple_jwst_phot as sjp

    _install_hst_get_ee_corr_patch(sjp)
    _install_match_refcat_bounds_patch(sjp)

    cur = sjp.hst_photclass.load_image
    if getattr(cur, '__name__', '') == _PATCH_MARKER:
        return

    _orig_load_image = cur

    def load_image_st123_wfpc2(
        self,
        imagename: str,
        imagetype: str | None = None,
        DNunits: bool = False,
        use_dq: bool = False,
        skip_preparing: bool = False,
    ) -> None:
        from astropy.io import fits as fits_mod

        try:
            ph = fits_mod.getheader(imagename, 0)
        except Exception:
            return _orig_load_image(
                self,
                imagename,
                imagetype,
                DNunits,
                use_dq,
                skip_preparing,
            )
        inst = str(ph.get('INSTRUME', '')).strip().upper()
        if inst != 'WFPC2':
            # Avoid internal AstroDrizzle for multi-chip HST FLC/FLT (unstable
            # in some environments). Load with skip_preparing, force do_driz
            # off, then prepare SCI ourselves.
            out = _orig_load_image(
                self,
                imagename,
                imagetype,
                DNunits,
                use_dq,
                True,  # skip_preparing
            )
            _ensure_jhat_scihdr_mjd_avg(
                getattr(self, 'primaryhdr', ph), getattr(self, 'scihdr', None)
            )
            tel = str(getattr(self, 'primaryhdr', ph).get('TELESCOP', '')).strip().upper()
            if tel == 'HST' and hasattr(self, 'do_driz'):
                self.do_driz = False
            if not skip_preparing:
                dq = None
                if use_dq:
                    try:
                        dq = self.im['DQ'].data  # type: ignore[attr-defined]
                    except Exception:
                        dq = None
                area = None
                try:
                    from stsci.skypac import pamutils  # type: ignore

                    area = pamutils.pam_from_file(
                        self.imagename, ('sci', 1), self.imagename + '_pam.fits'
                    )
                except Exception:
                    area = None
                data_original = self.im['SCI'].data  # type: ignore[attr-defined]
                imhdr = self.im['SCI'].header  # type: ignore[attr-defined]
                (self.data, self.mask) = self.prepare_image(  # type: ignore[attr-defined]
                    data_original,
                    imhdr,
                    area=area,
                    dq=dq,
                )
            return out
        return _wfpc2_hst_photclass_load_image(
            self,
            imagename,
            imagetype,
            DNunits,
            use_dq,
            skip_preparing,
        )

    load_image_st123_wfpc2.__name__ = _PATCH_MARKER
    sjp.hst_photclass.load_image = load_image_st123_wfpc2
    log.debug('Applied JHAT hst_photclass.load_image WFPC2 monkey-patch')
    ensure_hst_rshift_fitgeometry_patch()


def ensure_hst_rshift_fitgeometry_patch() -> None:
    """
    Force ``fitgeometry='rshift'`` and ``minobj=3`` on JHAT TweakReg for HST.

    Patches whatever ``jhat`` is importable (site-packages or vendored), so
    WFPC2 ``do_driz=True`` no longer falls into upstream's ``general`` fit.
    """
    try:
        from jhat.st_wcs_align import st_wcs_align
    except ImportError:
        return
    cur = st_wcs_align.run_align2refcat
    if getattr(cur, '_st123_hst_rshift', False):
        return
    _orig = cur

    def run_align2refcat_st123(self, *args, **kwargs):
        # Intercept TweakRegStep instances created inside _orig by wrapping
        # the class __setattr__ / post-init via a temporary subclass hook.
        try:
            from jwst.tweakreg.tweakreg_step import TweakRegStep
        except Exception:
            return _orig(self, *args, **kwargs)

        _real_call = TweakRegStep.__call__

        def _call_force_rshift(step_self, *a, **k):
            tel = str(getattr(self, 'telescope', '') or '').lower()
            if tel == 'hst' or int(getattr(step_self, 'pipeline_level', 2) or 2) == 2:
                try:
                    step_self.fitgeometry = 'rshift'
                except Exception:
                    pass
                try:
                    step_self.minobj = 3
                except Exception:
                    pass
            return _real_call(step_self, *a, **k)

        TweakRegStep.__call__ = _call_force_rshift  # type: ignore[method-assign]
        try:
            return _orig(self, *args, **kwargs)
        finally:
            TweakRegStep.__call__ = _real_call  # type: ignore[method-assign]

    run_align2refcat_st123._st123_hst_rshift = True  # type: ignore[attr-defined]
    st_wcs_align.run_align2refcat = run_align2refcat_st123  # type: ignore[assignment]
    log.debug('Applied JHAT HST rshift/minobj monkey-patch')


def _jhat_hst_output_path(image: str | Path, outdir: str | Path) -> Path:
    """Expected JHAT HST product path (``*_jhat.fits``)."""
    base = os.path.basename(os.fspath(image))
    short = re.sub(r'_([a-zA-Z0-9]+)\.fits$', '_jhat.fits', base)
    if short == base:
        stem = re.sub(r'\.fits$', '', base, flags=re.I)
        short = f'{stem}_jhat.fits'
    return Path(outdir) / short


def _recover_jhat_product(image: str | Path, outdir: str | Path) -> Path | None:
    """
    Locate / normalize JHAT products when tweakreg writes ``*_tweakregstep.fits``.

    Upstream JHAT renames ``{stem}_tweakregstep.fits`` -> ``{stem}_jhat.fits``, but
    tweakreg-hack often emits ``{inputstem}_tweakregstep.fits`` (e.g.
    ``iey902sdq_flc_tweakregstep.fits``), which the rename misses.
    """
    image_path = Path(image)
    out = Path(outdir)
    expected = _jhat_hst_output_path(image_path, out)
    if expected.is_file():
        return expected

    full_stem = re.sub(r'\.fits$', '', image_path.name, flags=re.I)
    short_stem = re.sub(r'_([a-zA-Z0-9]+)$', '', full_stem)
    candidates = [
        out / f'{full_stem}_tweakregstep.fits',
        out / f'{short_stem}_tweakregstep.fits',
        out / f'{full_stem}_jhat.fits',
        out / f'{short_stem}_jhat.fits',
    ]
    for cand in candidates:
        if not cand.is_file():
            continue
        if cand.resolve() != expected.resolve():
            if expected.exists():
                expected.unlink()
            cand.rename(expected)
        return expected

    # Last resort: newest matching tweakreg / jhat product for this root.
    globs = sorted(
        list(out.glob(f'{short_stem}*tweakregstep.fits'))
        + list(out.glob(f'{short_stem}*_jhat.fits')),
        key=lambda p: p.stat().st_mtime,
    )
    if globs:
        cand = globs[-1]
        if cand.resolve() != expected.resolve():
            if expected.exists():
                expected.unlink()
            cand.rename(expected)
        return expected
    return None


def _sci_hdu_indices(hdul) -> list[int]:
    """Return indices of 2-D SCI (or first science) HDUs in *hdul*."""
    out: list[int] = []
    for i, hdu in enumerate(hdul):
        if getattr(hdu, 'name', '') == 'SCI' and hdu.data is not None:
            if getattr(hdu.data, 'ndim', 0) >= 2:
                out.append(i)
    return out


def measure_sci_sky_translation(
    wcs_before,
    wcs_after,
    shape: tuple[int, int],
    *,
    ngrid: int = 5,
    margin: int = 50,
) -> tuple[float, float]:
    """
    Median sky translation (DeltaRA, DeltaDec) in degrees from *wcs_before* -> *wcs_after*.

    Samples a grid of detector pixels and differences the world coordinates.
    """
    import numpy as np

    ny, nx = int(shape[0]), int(shape[1])
    m = min(int(margin), nx // 4, ny // 4)
    xs = np.linspace(m, nx - 1 - m, int(ngrid))
    ys = np.linspace(m, ny - 1 - m, int(ngrid))
    xx, yy = np.meshgrid(xs, ys)
    ra0, dec0 = wcs_before.pixel_to_world_values(xx, yy)
    ra1, dec1 = wcs_after.pixel_to_world_values(xx, yy)
    # Handle RA wrap near 0/360.
    dra = (np.asarray(ra1, dtype=float) - np.asarray(ra0, dtype=float) + 180.0) % 360.0 - 180.0
    ddec = np.asarray(dec1, dtype=float) - np.asarray(dec0, dtype=float)
    return float(np.median(dra)), float(np.median(ddec))


def apply_sky_translation_to_sci(
    hdul,
    dra_deg: float,
    ddec_deg: float,
    *,
    sci_indices: list[int] | None = None,
    comment: str = 'st123: multi-SCI JHAT sky',
) -> int:
    """Add (DeltaRA, DeltaDec) in degrees to ``CRVAL`` of selected SCI HDUs. Returns count."""
    idxs = sci_indices if sci_indices is not None else _sci_hdu_indices(hdul)
    n = 0
    for i in idxs:
        hdr = hdul[i].header
        if 'CRVAL1' not in hdr or 'CRVAL2' not in hdr:
            continue
        hdr['CRVAL1'] = (
            float(hdr['CRVAL1']) + float(dra_deg),
            f'{comment} dRA',
        )
        hdr['CRVAL2'] = (
            float(hdr['CRVAL2']) + float(ddec_deg),
            f'{comment} dDec',
        )
        n += 1
    return n


# Pre-drizzle internal alignment: frames in one coadd must agree better than this.
# Defaults track the shared JWST/HST 50 mas gate (sparse soft-fallback 80 mas).
HST_INTERNAL_ALIGN_MAX_ARCSEC = FRAME_INTERNAL_TOL_ARCSEC
# Level-3 coadds (different filters / instruments) should agree to this.
HST_L3_ALIGN_MAX_ARCSEC = FRAME_ABS_TOL_ARCSEC
# Soft L3 ceiling used for mosaic warnings (not a hard drizzle gate).
HST_L3_SOFT_ALIGN_MAX_ARCSEC = FRAME_L3_SOFT_TOL_ARCSEC
# Absolute tie search radius for 2-D offset histograms (handles ~arcsec pipeline errors).
HST_ABS_OFFSET_MAX_ARCSEC = 5.0
# Sparse-match soft internal/abs tolerance (legacy 80 mas).
HST_SPARSE_ALIGN_MAX_ARCSEC = FRAME_SPARSE_TOL_ARCSEC


def measure_hst_sky_offset_2dhist(
    path_img: str | Path,
    path_ref: str | Path,
    *,
    sci_order: int = 0,
    max_offset_arcsec: float = HST_ABS_OFFSET_MAX_ARCSEC,
    bin_arcsec: float = 0.1,
    nbright: int = 400,
    min_peak: int = 4,
    exclude_zero_arcsec: float = 0.35,
) -> dict[str, Any]:
    """
    Robust sky offset of *path_img* relative to *path_ref* via a 2-D histogram.

    Nearest-neighbour matching inside a large radius is contaminated by chance
    pairs near 0 when the true offset is ~1-5". The histogram peak (optionally
    excluding a small core around zero) recovers the coherent shift. Returns
    ``img - ref``; add ``-dra/-ddec`` to *path_img* CRVAL to place it on *path_ref*.
    """
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy.stats import sigma_clipped_stats
    from astropy.wcs import WCS
    from photutils.detection import DAOStarFinder
    import astropy.units as u

    stats: dict[str, Any] = {
        'ok': False,
        'method': '2dhist',
        'n_pairs': 0,
        'peak_count': 0,
        'dra_deg': 0.0,
        'ddec_deg': 0.0,
        'dra_arcsec': 0.0,
        'ddec_arcsec': 0.0,
        'abs_arcsec': 0.0,
    }

    def _det(path: Path):
        with as_datamodel(path).open(memmap=True) as hdul:
            idxs = _sci_hdu_indices(hdul)
            if not idxs or sci_order >= len(idxs):
                return None
            ext = hdul[idxs[sci_order]]
            data = np.asarray(ext.data, dtype=float)
            wcs = WCS(ext.header, hdul, naxis=2)
        mask = ~np.isfinite(data)
        if data.ndim > 2:
            data = data[0]
            mask = ~np.isfinite(data)
        # Coadds often have empty edges.
        mask = mask | (data == 0)
        _, med, std = sigma_clipped_stats(data, mask=mask, sigma=3.0, maxiters=5)
        tbl = DAOStarFinder(fwhm=2.5, threshold=4.0 * std)(data - med, mask=mask)
        if tbl is None or len(tbl) < 5:
            return None
        xcol = 'x_centroid' if 'x_centroid' in tbl.colnames else 'xcentroid'
        ycol = 'y_centroid' if 'y_centroid' in tbl.colnames else 'ycentroid'
        tbl.sort('flux')
        tbl.reverse()
        tbl = tbl[: int(nbright)]
        ra, dec = wcs.pixel_to_world_values(tbl[xcol], tbl[ycol])
        return SkyCoord(np.asarray(ra, dtype=float) * u.deg, np.asarray(dec, dtype=float) * u.deg)

    c_i = _det(Path(path_img))
    c_r = _det(Path(path_ref))
    if c_i is None or c_r is None:
        return stats
    idx, sep, _ = c_i.match_to_catalog_sky(c_r)
    good = sep.arcsec < float(max_offset_arcsec)
    n = int(np.count_nonzero(good))
    stats['n_pairs'] = n
    if n < int(min_peak):
        return stats
    dra = (c_i.ra - c_r.ra[idx]).to(u.arcsec).value * np.cos(np.radians(c_i.dec.value))
    ddec = (c_i.dec - c_r.dec[idx]).to(u.arcsec).value
    dra = np.asarray(dra)[good]
    ddec = np.asarray(ddec)[good]
    rmax = float(max_offset_arcsec)
    bins = np.arange(-rmax, rmax + float(bin_arcsec), float(bin_arcsec))
    if len(bins) < 4:
        return stats
    H, xe, ye = np.histogram2d(dra, ddec, bins=[bins, bins])
    # Prefer a peak outside the false-match core near zero only when that
    # outer peak *dominates* the core. Crowded fields always produce weak
    # secondary peaks (>= min_peak) outside ~0.35"; blindly preferring them
    # false-fails well-aligned coadds at ~0.3-0.5".
    cx = 0.5 * (xe[:-1] + xe[1:])
    cy = 0.5 * (ye[:-1] + ye[1:])
    XX, YY = np.meshgrid(cx, cy, indexing='ij')
    H_use = H.copy()
    core = np.hypot(XX, YY) < float(exclude_zero_arcsec)
    peak_core = int(H[core].max(initial=0)) if bool(np.any(core)) else 0
    peak_outer = int(H[~core].max(initial=0)) if bool(np.any(~core)) else 0
    # Require outer to beat the core (strictly) and clear min_peak; a small
    # margin avoids ties when chance and true signal are both weak.
    if peak_outer >= int(min_peak) and peak_outer > peak_core:
        H_use[core] = 0
        stats['peak_mode'] = 'outer'
    else:
        stats['peak_mode'] = 'core' if peak_core >= peak_outer else 'global'
    iy, ix = np.unravel_index(int(np.argmax(H_use)), H_use.shape)
    peak = int(H_use[iy, ix])
    if peak < int(min_peak):
        iy, ix = np.unravel_index(int(np.argmax(H)), H.shape)
        peak = int(H[iy, ix])
        stats['peak_mode'] = 'global_fallback'
        if peak < int(min_peak):
            return stats
    dra_as = float(cx[iy])
    ddec_as = float(cy[ix])
    # Refine with matches near the peak.
    near = np.hypot(dra - dra_as, ddec - ddec_as) < max(2.5 * float(bin_arcsec), 0.2)
    if int(np.count_nonzero(near)) >= int(min_peak):
        dra_as = float(np.median(dra[near]))
        ddec_as = float(np.median(ddec[near]))
        peak = int(np.count_nonzero(near))
    dec0 = float(np.median(c_i.dec.degree))
    dra_deg = dra_as / (3600.0 * float(np.cos(np.radians(dec0))))
    ddec_deg = ddec_as / 3600.0
    stats.update(
        {
            'ok': True,
            'peak_count': peak,
            'peak_core': peak_core,
            'peak_outer': peak_outer,
            'dra_deg': dra_deg,
            'ddec_deg': ddec_deg,
            'dra_arcsec': dra_as,
            'ddec_arcsec': ddec_as,
            'abs_arcsec': float(np.hypot(dra_as, ddec_as)),
        }
    )
    return stats


def find_hst_abs_ref_image(jhat_dir: str | Path) -> Path | None:
    """
    Prefer a deep aligned frame/coadd for absolute ties (F625 / ACS*WFC3 F814).

    Prefers boxed ``reference/group_*/ref_*/`` products, then legacy flat
    coadds. Modern ACS/WFC3 coadds beat WFPC2 when both exist - WFPC2
    multi-visit coadds can be self-ghosted and make a poor absolute reference.
    Candidates with healthy ``JWDISPM``/``JWNCAL`` or dense phot catalogs are
    boosted (JWST-style hub quality gate).
    """

    jhat = Path(jhat_dir).expanduser().resolve()
    refdir = jhat.parent / 'reference'
    candidates: list[Path] = []
    if refdir.is_dir():
        for pat in (
            'group_*/ref_*/coadd_*_wfc3_f625w_drc.fits',
            'group_*/ref_*/coadd_*_wfc3_f814w_drc.fits',
            'group_*/ref_*/coadd_*_acs_f814w_drc.fits',
            'group_*/ref_*/coadd_*_wfpc2_f814w_drz.fits',
            'group_*/ref_*/coadd_*_f625w_*.fits',
            'group_*/ref_*/coadd_*_f814w_*.fits',
            'group_*/ref_*/coadd_*_f555w_*.fits',
            'coadd_wfc3_f625w_drc.fits',
            'coadd_wfc3_f814w_drc.fits',
            'coadd_acs_f814w_drc.fits',
            'coadd_wfpc2_f814w_drz.fits',
            'coadd_*_f625w_*.fits',
            'coadd_*_f814w_*.fits',
            'coadd_*_f555w_*.fits',
        ):
            candidates.extend(sorted(refdir.glob(pat)))
    candidates.extend(
        [
            jhat / 'iey902seq_jhat.fits',
            jhat / 'iey902shq_jhat.fits',
        ]
    )
    # Prefer Gaia-anchored L3 under jhat/l3_ref when present.
    l3 = jhat / 'l3_ref'
    if l3.is_dir():
        candidates.extend(sorted(l3.glob('coadd_*_drc.fits')))
        candidates.extend(sorted(l3.glob('coadd_*_drz.fits')))
        candidates.extend(sorted(l3.glob('coadd_*_jhat.fits')))

    def _score(p: Path) -> tuple[int, int, int]:
        name = p.name.lower()
        pref = 0
        if 'wfc3' in name and 'f625' in name:
            pref = 400
        elif 'wfc3' in name and 'f814' in name:
            pref = 350
        elif 'acs' in name and 'f814' in name:
            pref = 320
        elif 'wfpc2' in name and 'f814' in name:
            pref = 120  # demote ghost-prone WFPC2 hubs
        elif 'f625' in name:
            pref = 150
        elif 'f814' in name:
            pref = 100
        layout = 1 if 'group_' in p.as_posix() else 0
        # Prefer hubs that already carry a healthy JWDISPM stamp / dense phot.
        quality = 0
        try:
            with as_datamodel(p).open(memmap=True) as hdul:
                disp = hdul[0].header.get('JWDISPM')
                ncal = hdul[0].header.get('JWNCAL')
                if disp is not None and ncal is not None:
                    try:
                        d = float(disp)
                        n = int(ncal)
                        if n >= FRAME_MIN_MATCH_HEALTHY and d <= FRAME_ABS_TOL_ARCSEC:
                            quality = 50
                        elif n >= FRAME_MIN_MATCH_HEALTHY and d <= FRAME_SPARSE_TOL_ARCSEC:
                            quality = 25
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass
        phot = p.with_name(
            p.name.replace('_drc.fits', '_drc.phot.txt')
            .replace('_drz.fits', '_drz.phot.txt')
            .replace('_jhat.fits', '.phot.txt')
        )
        # Also check l3_ref sibling naming.
        if not phot.is_file() and 'drc' in p.name:
            alt = p.parent / 'l3_ref' / (p.stem.replace('_drc', '_drc') + '.phot.txt')
            # common: coadd_acs_f814w_drc.phot.txt next to jhat/l3_ref
            for cand in (
                p.parent / f'{p.name.replace(".fits", ".phot.txt")}',
                jhat / 'l3_ref' / f'{p.stem}.phot.txt',
                jhat / 'l3_ref' / p.name.replace('_drc.fits', '_drc.phot.txt').replace(
                    '_drz.fits', '_drz.phot.txt'
                ),
            ):
                if cand.is_file():
                    phot = cand
                    break
        if phot.is_file():
            try:
                n_lines = sum(1 for _ in phot.open())
                quality += min(40, n_lines // 50)
            except Exception:
                pass
        try:
            size = int(p.stat().st_size)
        except OSError:
            size = 0
        return (pref + 10 * layout + quality, size, quality)

    usable: list[Path] = []
    for cand in candidates:
        try:
            if cand.is_file() and cand.stat().st_size > 500_000:
                usable.append(cand.resolve())
        except OSError:
            continue
    if not usable:
        return None
    return max(usable, key=_score)


def _frame_exptime(path: str | Path) -> float:
    """EXPTIME from primary/SCI, else EXPSTART/EXPEND; 0.0 if missing."""

    p = Path(path)
    try:
        with as_datamodel(p).open(memmap=True) as hdul:
            for hdu in hdul:
                val = hdu.header.get('EXPTIME')
                if val is not None:
                    try:
                        exp = float(val)
                    except (TypeError, ValueError):
                        exp = 0.0
                    if exp > 0:
                        return exp
            prim = hdul[0].header
            start, end = prim.get('EXPSTART'), prim.get('EXPEND')
            if start is not None and end is not None:
                derived = (float(end) - float(start)) * 86400.0
                if derived > 0:
                    return float(derived)
    except Exception:
        return 0.0
    return 0.0


def _detect_sci_sky_sources(
    path: str | Path,
    *,
    sci_order: int = 0,
    nbright: int = 250,
    fwhm: float = 2.5,
    nsig: float = 5.0,
):
    """
    Detect bright sources on one SCI and return (ra, dec, wcs) arrays.

    Returns ``(None, None, None)`` when detection fails.
    """
    import numpy as np
    from astropy.stats import sigma_clipped_stats
    from astropy.wcs import WCS
    from photutils.detection import DAOStarFinder

    p = Path(path)
    with as_datamodel(p).open(memmap=True) as hdul:
        idxs = _sci_hdu_indices(hdul)
        if not idxs or sci_order < 0 or sci_order >= len(idxs):
            return None, None, None
        ext = hdul[idxs[sci_order]]
        data = np.asarray(ext.data, dtype=float)
        wcs = WCS(ext.header, hdul, naxis=2)
    mask = ~np.isfinite(data)
    try:
        _, med, std = sigma_clipped_stats(data, mask=mask, sigma=3.0, maxiters=5)
        tbl = DAOStarFinder(fwhm=float(fwhm), threshold=float(nsig) * std)(
            data - med, mask=mask
        )
    except Exception:
        return None, None, None
    if tbl is None or len(tbl) < 5:
        return None, None, None
    xcol = 'x_centroid' if 'x_centroid' in tbl.colnames else 'xcentroid'
    ycol = 'y_centroid' if 'y_centroid' in tbl.colnames else 'ycentroid'
    tbl.sort('flux')
    tbl.reverse()
    tbl = tbl[: int(nbright)]
    ra, dec = wcs.pixel_to_world_values(tbl[xcol], tbl[ycol])
    return np.asarray(ra, dtype=float), np.asarray(dec, dtype=float), wcs


def measure_hst_frame_sky_offset(
    path_img: str | Path,
    path_ref: str | Path,
    *,
    sci_order: int = 0,
    match_radius_arcsec: float = 1.0,
    min_matches: int = 8,
    nbright: int = 250,
) -> dict[str, Any]:
    """
    Median sky offset of *path_img* relative to *path_ref* (same SCI order).

    Sources are detected independently; matched in sky. Positive
    ``dra_arcsec`` / ``ddec_arcsec`` mean img catalogs sit east/north of ref -
    subtract those CRVAL shifts from *path_img* to place it on *path_ref*.
    """
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    stats: dict[str, Any] = {
        'path_img': str(Path(path_img)),
        'path_ref': str(Path(path_ref)),
        'sci_order': int(sci_order),
        'n_match': 0,
        'ok': False,
        'dra_deg': 0.0,
        'ddec_deg': 0.0,
        'dra_arcsec': 0.0,
        'ddec_arcsec': 0.0,
        'abs_arcsec': 0.0,
        'med_sep_arcsec': 0.0,
    }
    ra_i, dec_i, _ = _detect_sci_sky_sources(
        path_img, sci_order=sci_order, nbright=nbright
    )
    ra_r, dec_r, _ = _detect_sci_sky_sources(
        path_ref, sci_order=sci_order, nbright=nbright
    )
    if ra_i is None or ra_r is None:
        return stats
    c_i = SkyCoord(ra_i * u.deg, dec_i * u.deg)
    c_r = SkyCoord(ra_r * u.deg, dec_r * u.deg)
    idx, sep, _ = c_i.match_to_catalog_sky(c_r)
    good = sep.arcsec < float(match_radius_arcsec)
    n = int(np.count_nonzero(good))
    stats['n_match'] = n
    if n < int(min_matches):
        return stats
    # img - ref in degrees (RA wrapped).
    dra = (c_i.ra - c_r.ra[idx]).to(u.deg).value
    dra = (dra + 180.0) % 360.0 - 180.0
    ddec = (c_i.dec - c_r.dec[idx]).to(u.deg).value
    dra_m = float(np.median(dra[good]))
    ddec_m = float(np.median(ddec[good]))
    dec0 = float(np.median(dec_i[good]))
    dra_as = dra_m * 3600.0 * float(np.cos(np.radians(dec0)))
    ddec_as = ddec_m * 3600.0
    stats.update(
        {
            'ok': True,
            'dra_deg': dra_m,
            'ddec_deg': ddec_m,
            'dra_arcsec': dra_as,
            'ddec_arcsec': ddec_as,
            'abs_arcsec': float(np.hypot(dra_as, ddec_as)),
            'med_sep_arcsec': float(np.median(sep.arcsec[good])),
        }
    )
    return stats


def measure_hst_frame_sky_offset_via_refcat(
    path_img: str | Path,
    path_ref: str | Path,
    refcat: str | Path,
    *,
    sci_order: int = 0,
    search_radius_arcsec: float = 0.8,
    min_matches: int = 8,
    max_sources: int = 2000,
) -> dict[str, Any]:
    """
    Sky offset of *path_img* vs *path_ref* using the same refcat sources.

    Centroids each refcat star on both detectors and differences the WCS sky
    positions. This avoids false matches between independent detections.
    """
    import numpy as np
    import pandas as pd
    from astropy.stats import sigma_clip
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales

    from st123.stages.alignment.gaia_simple import _centroid

    stats: dict[str, Any] = {
        'path_img': str(Path(path_img)),
        'path_ref': str(Path(path_ref)),
        'sci_order': int(sci_order),
        'method': 'refcat',
        'n_match': 0,
        'ok': False,
        'dra_deg': 0.0,
        'ddec_deg': 0.0,
        'dra_arcsec': 0.0,
        'ddec_arcsec': 0.0,
        'abs_arcsec': 0.0,
        'med_sep_arcsec': 0.0,
    }
    ref_path = Path(refcat).expanduser().resolve()
    if not ref_path.is_file():
        return stats
    ref_df = pd.read_csv(ref_path, sep=r'\s+', engine='python')
    if 'ra' not in ref_df.columns or 'dec' not in ref_df.columns:
        return stats
    if 'mag' in ref_df.columns:
        ref_df = ref_df.sort_values('mag')
    ras = np.asarray(ref_df['ra'], dtype=float)[: int(max_sources)]
    decs = np.asarray(ref_df['dec'], dtype=float)[: int(max_sources)]

    def _load(path: Path):
        with as_datamodel(path).open(memmap=True) as hdul:
            idxs = _sci_hdu_indices(hdul)
            if not idxs or sci_order >= len(idxs):
                return None
            ext = hdul[idxs[sci_order]]
            data = np.asarray(ext.data, dtype=float)
            wcs = WCS(ext.header, hdul, naxis=2)
            return data, wcs

    loaded_i = _load(Path(path_img))
    loaded_r = _load(Path(path_ref))
    if loaded_i is None or loaded_r is None:
        return stats
    data_i, w_i = loaded_i
    data_r, w_r = loaded_r
    bad_i = ~np.isfinite(data_i)
    bad_r = ~np.isfinite(data_r)
    ny_i, nx_i = data_i.shape[-2:]
    ny_r, nx_r = data_r.shape[-2:]
    scale_i = float(np.nanmedian(proj_plane_pixel_scales(w_i)) * 3600.0)
    scale_r = float(np.nanmedian(proj_plane_pixel_scales(w_r)) * 3600.0)
    r_pix_i = float(search_radius_arcsec) / max(scale_i, 1e-6)
    r_pix_r = float(search_radius_arcsec) / max(scale_r, 1e-6)

    dra_list: list[float] = []
    ddec_list: list[float] = []
    sep_list: list[float] = []
    # One quiet vectorized projection -- avoids per-star all_world2pix warnings.
    x_i_all, y_i_all = world_to_pixel_quiet(w_i, ras, decs)
    x_r_all, y_r_all = world_to_pixel_quiet(w_r, ras, decs)
    for x_i0, y_i0, x_r0, y_r0, ra, dec in zip(
        x_i_all, y_i_all, x_r_all, y_r_all, ras, decs
    ):
        if not (
            np.isfinite(x_i0)
            and np.isfinite(y_i0)
            and np.isfinite(x_r0)
            and np.isfinite(y_r0)
        ):
            continue
        if (
            x_i0 < -r_pix_i
            or x_i0 > (nx_i - 1) + r_pix_i
            or y_i0 < -r_pix_i
            or y_i0 > (ny_i - 1) + r_pix_i
            or x_r0 < -r_pix_r
            or x_r0 > (nx_r - 1) + r_pix_r
            or y_r0 < -r_pix_r
            or y_r0 > (ny_r - 1) + r_pix_r
        ):
            continue
        try:
            x_i, y_i, _ = _centroid(
                data_i, bad_i, x0=float(x_i0), y0=float(y_i0), r_pix=r_pix_i
            )
            x_r, y_r, _ = _centroid(
                data_r, bad_r, x0=float(x_r0), y0=float(y_r0), r_pix=r_pix_r
            )
            ra_i, dec_i = w_i.pixel_to_world_values(x_i, y_i)
            ra_r, dec_r = w_r.pixel_to_world_values(x_r, y_r)
        except Exception:
            continue
        dra = (float(ra_i) - float(ra_r) + 180.0) % 360.0 - 180.0
        ddec = float(dec_i) - float(dec_r)
        dra_list.append(dra)
        ddec_list.append(ddec)
        sep_list.append(
            float(
                np.hypot(
                    dra * 3600.0 * np.cos(np.radians(0.5 * (dec_i + dec_r))),
                    ddec * 3600.0,
                )
            )
        )
    if len(dra_list) < int(min_matches):
        stats['n_match'] = len(dra_list)
        return stats
    dra_a = np.asarray(dra_list, dtype=float)
    ddec_a = np.asarray(ddec_list, dtype=float)
    sep_a = np.asarray(sep_list, dtype=float)
    clipped = sigma_clip(sep_a, sigma=3.0, maxiters=5, masked=True)
    keep = ~np.asarray(getattr(clipped, 'mask', np.zeros_like(sep_a, dtype=bool)))
    if int(np.count_nonzero(keep)) < int(min_matches):
        stats['n_match'] = int(np.count_nonzero(keep))
        return stats
    dra_m = float(np.median(dra_a[keep]))
    ddec_m = float(np.median(ddec_a[keep]))
    dec0 = float(np.median(decs[: len(dra_list)]))
    dra_as = dra_m * 3600.0 * float(np.cos(np.radians(dec0)))
    ddec_as = ddec_m * 3600.0
    stats.update(
        {
            'ok': True,
            'n_match': int(np.count_nonzero(keep)),
            'dra_deg': dra_m,
            'ddec_deg': ddec_m,
            'dra_arcsec': dra_as,
            'ddec_arcsec': ddec_as,
            'abs_arcsec': float(np.hypot(dra_as, ddec_as)),
            'med_sep_arcsec': float(np.median(sep_a[keep])),
        }
    )
    return stats


def _measure_offset_one_sci(
    path_img: str | Path,
    path_ref: str | Path,
    *,
    sci_order: int,
    refcat: str | Path | None,
    match_radius_arcsec: float,
    min_matches: int,
) -> dict[str, Any]:
    """Prefer refcat centroids; fall back to independent source matching."""
    if refcat is not None and Path(refcat).is_file():
        st = measure_hst_frame_sky_offset_via_refcat(
            path_img,
            path_ref,
            refcat,
            sci_order=sci_order,
            search_radius_arcsec=min(float(match_radius_arcsec), 0.8),
            min_matches=min_matches,
        )
        if st.get('ok'):
            return st
    return measure_hst_frame_sky_offset(
        path_img,
        path_ref,
        sci_order=sci_order,
        match_radius_arcsec=match_radius_arcsec,
        min_matches=min_matches,
    )


def measure_hst_frame_sky_offset_multichip(
    path_img: str | Path,
    path_ref: str | Path,
    *,
    match_radius_arcsec: float = 1.0,
    min_matches: int = 8,
    max_sci: int = 4,
    refcat: str | Path | None = None,
) -> dict[str, Any]:
    """
    Per-SCI sky offsets of *path_img* vs *path_ref*.

    Top-level ``dra_*`` / ``abs_arcsec`` are from SCI1 (the primary drizzle
    alignment chip). ``max_abs_arcsec`` is the worst chip - used for QA.
    """

    chip_stats: list[dict[str, Any]] = []
    with as_datamodel(path_img).open(memmap=True) as hdul:
        n_sci = min(len(_sci_hdu_indices(hdul)), int(max_sci))
    for k in range(n_sci):
        st = _measure_offset_one_sci(
            path_img,
            path_ref,
            sci_order=k,
            refcat=refcat,
            match_radius_arcsec=match_radius_arcsec,
            min_matches=min_matches,
        )
        if st.get('ok'):
            chip_stats.append(st)
    out: dict[str, Any] = {
        'path_img': str(Path(path_img)),
        'path_ref': str(Path(path_ref)),
        'ok': False,
        'n_chips': len(chip_stats),
        'chips': chip_stats,
        'dra_deg': 0.0,
        'ddec_deg': 0.0,
        'dra_arcsec': 0.0,
        'ddec_arcsec': 0.0,
        'abs_arcsec': 0.0,
        'max_abs_arcsec': 0.0,
    }
    if not chip_stats:
        return out
    # Prefer SCI1 (sci_order==0) for the global translation; else first chip.
    primary = next((c for c in chip_stats if c.get('sci_order') == 0), chip_stats[0])
    max_abs = max(float(c['abs_arcsec']) for c in chip_stats)
    out.update(
        {
            'ok': True,
            'dra_deg': float(primary['dra_deg']),
            'ddec_deg': float(primary['ddec_deg']),
            'dra_arcsec': float(primary['dra_arcsec']),
            'ddec_arcsec': float(primary['ddec_arcsec']),
            'abs_arcsec': float(primary['abs_arcsec']),
            'max_abs_arcsec': float(max_abs),
        }
    )
    return out


def _raw_sibling_for_jhat(path: Path) -> Path | None:
    """Calibrated ``raw/`` sibling for a JHAT product, if present."""
    stem = path.name
    for suffix in ('_jhat.fits', '.fits'):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for raw_dir in (path.parent.parent / 'raw', path.parent / 'raw'):
        if not raw_dir.is_dir():
            continue
        for cand in (
            raw_dir / f'{stem}_flc.fits',
            raw_dir / f'{stem}_flt.fits',
            raw_dir / f'{stem}_c0m.fits',
        ):
            if cand.is_file():
                return cand.resolve()
    return None


_WCS_COPY_KEYS = (
    'WCSAXES', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
    'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
    'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
    'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
    'CDELT1', 'CDELT2', 'CROTA2',
    'LONPOLE', 'LATPOLE', 'RADESYS', 'EQUINOX',
    'WCSNAME', 'ORIENTAT',
)


def world_to_pixel_quiet(wcs, ra, dec, *, origin: int = 0):
    """
    Sky -> pixel without per-star ``all_world2pix`` warning spam.

    Uses :meth:`astropy.wcs.WCS.all_world2pix` with ``quiet=True`` so
    divergent SIP / WFPC2 solutions return NaN instead of flooding stderr.
    Accepts scalars or arrays; returns ``(x, y)`` with the same shape.
    """
    import numpy as np

    ra_a = np.asarray(ra, dtype=float)
    dec_a = np.asarray(dec, dtype=float)
    scalar = ra_a.ndim == 0 and dec_a.ndim == 0
    if scalar:
        ra_a = ra_a.reshape(1)
        dec_a = dec_a.reshape(1)
    try:
        raw = wcs.all_world2pix(ra_a, dec_a, int(origin), quiet=True)
        # Astropy returns [x_arr, y_arr] (list) or an Nx2 / 2xN array.
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            x = np.asarray(raw[0], dtype=float)
            y = np.asarray(raw[1], dtype=float)
        else:
            xy = np.asarray(raw, dtype=float)
            if xy.ndim == 2 and xy.shape[0] == 2:
                x, y = xy[0], xy[1]
            elif xy.ndim == 2 and xy.shape[-1] == 2:
                x, y = xy[:, 0], xy[:, 1]
            else:
                raise TypeError(f'unexpected all_world2pix shape {xy.shape}')
    except Exception:
        x = np.full(np.broadcast(ra_a, dec_a).shape, np.nan, dtype=float)
        y = np.full(np.broadcast(ra_a, dec_a).shape, np.nan, dtype=float)
    if scalar:
        return float(x.reshape(-1)[0]), float(y.reshape(-1)[0])
    return x, y


def measure_hst_frame_relative_offset_pixel(
    path_img: str | Path,
    path_ref: str | Path,
    *,
    sci_order: int = 0,
    max_match_pix: float = 10.0,
    min_matches: int = 12,
    nbright: int = 250,
) -> dict[str, Any]:
    """
    Relative WCS offset via detections matched on the sky.

    Both frames are projected with ``pixel_to_world`` only (stable for HST),
    then matched with :meth:`~astropy.coordinates.SkyCoord.match_to_catalog_sky`.
    This avoids ``all_world2pix`` divergence that WFPC2 / mistied SIP solutions
    trigger when projecting one frame into the other's pixel grid.
    """
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy.stats import sigma_clipped_stats
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    from photutils.detection import DAOStarFinder
    import astropy.units as u

    stats: dict[str, Any] = {
        'path_img': str(Path(path_img)),
        'path_ref': str(Path(path_ref)),
        'sci_order': int(sci_order),
        'method': 'sky_match',
        'n_match': 0,
        'ok': False,
        'dra_deg': 0.0,
        'ddec_deg': 0.0,
        'dra_arcsec': 0.0,
        'ddec_arcsec': 0.0,
        'abs_arcsec': 0.0,
        'med_sep_arcsec': 0.0,
        'med_dpix': 0.0,
    }

    def _det(path: Path):
        with as_datamodel(path).open(memmap=True) as hdul:
            idxs = _sci_hdu_indices(hdul)
            if not idxs or sci_order >= len(idxs):
                return None
            ext = hdul[idxs[sci_order]]
            data = np.asarray(ext.data, dtype=float)
            wcs = WCS(ext.header, hdul, naxis=2)
        mask = ~np.isfinite(data)
        _, med, std = sigma_clipped_stats(data, mask=mask, sigma=3.0, maxiters=5)
        if not np.isfinite(std) or std <= 0:
            return None
        tbl = DAOStarFinder(fwhm=2.5, threshold=5.0 * std)(data - med, mask=mask)
        if tbl is None or len(tbl) < 5:
            return None
        xcol = 'x_centroid' if 'x_centroid' in tbl.colnames else 'xcentroid'
        ycol = 'y_centroid' if 'y_centroid' in tbl.colnames else 'ycentroid'
        tbl.sort('flux')
        tbl.reverse()
        tbl = tbl[: int(nbright)]
        return (
            np.asarray(tbl[xcol], dtype=float),
            np.asarray(tbl[ycol], dtype=float),
            wcs,
        )

    det_i = _det(Path(path_img))
    det_r = _det(Path(path_ref))
    if det_i is None or det_r is None:
        return stats
    xi, yi, wi = det_i
    xr, yr, wr = det_r
    try:
        ra_i, dec_i = wi.pixel_to_world_values(xi, yi)
        ra_r, dec_r = wr.pixel_to_world_values(xr, yr)
    except Exception:
        return stats
    c_i = SkyCoord(
        np.asarray(ra_i, dtype=float) * u.deg,
        np.asarray(dec_i, dtype=float) * u.deg,
    )
    c_r = SkyCoord(
        np.asarray(ra_r, dtype=float) * u.deg,
        np.asarray(dec_r, dtype=float) * u.deg,
    )
    try:
        scale = float(np.nanmedian(proj_plane_pixel_scales(wr)) * 3600.0)
    except Exception:
        scale = 0.04
    match_as = float(max_match_pix) * max(scale, 1e-6)
    idx, sep, _ = c_i.match_to_catalog_sky(c_r)
    good = np.isfinite(sep.arcsec) & (sep.arcsec < match_as)
    n = int(np.count_nonzero(good))
    stats['n_match'] = n
    if n < int(min_matches):
        return stats
    # img - ref on matched stars (same convention as other offset helpers).
    dra = (c_i.ra[good] - c_r.ra[idx[good]]).to(u.deg).value
    dra = (dra + 180.0) % 360.0 - 180.0
    ddec = (c_i.dec[good] - c_r.dec[idx[good]]).to(u.deg).value
    dra_m = float(np.median(dra))
    ddec_m = float(np.median(ddec))
    dec0 = float(np.median(c_i.dec[good].degree))
    dra_as = dra_m * 3600.0 * float(np.cos(np.radians(dec0)))
    ddec_as = ddec_m * 3600.0
    stats.update(
        {
            'ok': True,
            'dra_deg': dra_m,
            'ddec_deg': ddec_m,
            'dra_arcsec': dra_as,
            'ddec_arcsec': ddec_as,
            'abs_arcsec': float(np.hypot(dra_as, ddec_as)),
            'med_sep_arcsec': float(np.median(sep.arcsec[good])),
            'med_dpix': float(np.median(sep.arcsec[good]) / max(scale, 1e-6)),
        }
    )
    return stats


def validate_hst_group_internal_alignment(
    frames: list[str | Path],
    *,
    max_coherent_arcsec: float | None = None,
    match_radius_arcsec: float = 1.0,
    min_matches: int = 12,
    refcat: str | Path | None = None,
) -> dict[str, Any]:
    """
    Pairwise internal-alignment QA for level-2 frames that will share a coadd.

    Uses **sky-space** source matching (``pixel_to_world`` on both frames; not
    ``all_world2pix`` into a peer pixel grid). ``ok`` is True when every
    measurable SCI1 pair has coherent |Delta| <= *max_coherent_arcsec*
    (default: 50 mas when matches are healthy, else 80 mas soft fallback).
    Single-frame groups always pass.

    *refcat* is accepted for API compatibility but ignored for the pass/fail
    decision (refcat centroid QA can false-pass after per-frame CRPIX refine).
    """
    del match_radius_arcsec, refcat  # unused; kept for call-site compatibility
    paths = [Path(p).expanduser().resolve() for p in frames]
    # Adaptive default: start at the tight gate; raise to sparse if needed.
    limit0 = (
        float(max_coherent_arcsec)
        if max_coherent_arcsec is not None
        else float(HST_INTERNAL_ALIGN_MAX_ARCSEC)
    )
    report: dict[str, Any] = {
        'ok': True,
        'n_frames': len(paths),
        'max_coherent_arcsec': limit0,
        'max_abs_arcsec': 0.0,
        'pairs': [],
        'failed_pairs': [],
        'method': 'sky_match',
    }
    if len(paths) < 2:
        return report
    worst = 0.0
    match_counts: list[int] = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            st = measure_hst_frame_relative_offset_pixel(
                paths[i],
                paths[j],
                sci_order=0,
                min_matches=min_matches,
            )
            abs_as = float(st.get('abs_arcsec') or 0.0)
            n_match = int(st.get('n_match') or 0)
            pair = {
                'a': paths[i].name,
                'b': paths[j].name,
                'ok': bool(st.get('ok')),
                'abs_arcsec': abs_as,
                'max_abs_arcsec': abs_as,
                'dra_arcsec': float(st.get('dra_arcsec') or 0.0),
                'ddec_arcsec': float(st.get('ddec_arcsec') or 0.0),
                'n_match': n_match,
                'med_dpix': float(st.get('med_dpix') or 0.0),
                'method': 'sky_match',
            }
            report['pairs'].append(pair)
            # Too few matches => pair is unmeasurable, not a failed alignment.
            if not st.get('ok') and n_match < int(min_matches):
                continue
            if st.get('ok'):
                match_counts.append(n_match)
            if not st.get('ok'):
                report['failed_pairs'].append(pair)
                report['ok'] = False
                continue
            worst = max(worst, abs_as)
    # Adaptive limit from median healthy match count when caller left default.
    if max_coherent_arcsec is None and match_counts:
        limit0 = coherent_tol_arcsec(int(sorted(match_counts)[len(match_counts) // 2]))
        report['max_coherent_arcsec'] = limit0
    for pair in report['pairs']:
        if not pair.get('ok'):
            continue
        if int(pair.get('n_match') or 0) < int(min_matches):
            continue
        if float(pair['abs_arcsec']) > float(limit0):
            if pair not in report['failed_pairs']:
                report['failed_pairs'].append(pair)
            report['ok'] = False
    report['max_abs_arcsec'] = float(worst)
    return report


def validate_hst_coadds_alignment(
    coadds: list[str | Path],
    *,
    max_coherent_arcsec: float = HST_L3_ALIGN_MAX_ARCSEC,
    match_radius_arcsec: float = 1.0,
    min_matches: int = 8,
    refcat: str | Path | None = None,
) -> dict[str, Any]:
    """
    Pairwise relative-alignment QA among level-3 coadds (SCI,1).

    Prefers a shared *refcat* (L3 phot) when provided.
    """
    paths = [Path(p).expanduser().resolve() for p in coadds if Path(p).is_file()]
    report: dict[str, Any] = {
        'ok': True,
        'n_coadds': len(paths),
        'max_coherent_arcsec': float(max_coherent_arcsec),
        'max_abs_arcsec': 0.0,
        'pairs': [],
        'failed_pairs': [],
        'refcat': str(refcat) if refcat else None,
    }
    if len(paths) < 2:
        return report
    worst = 0.0
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            # 2-D histogram recovers arcsecond-scale systematics that small-radius
            # NN matching false-locks near zero.
            st = measure_hst_sky_offset_2dhist(
                paths[i],
                paths[j],
                max_offset_arcsec=HST_ABS_OFFSET_MAX_ARCSEC,
            )
            if not st.get('ok'):
                st = measure_hst_frame_relative_offset_pixel(
                    paths[i],
                    paths[j],
                    sci_order=0,
                    max_match_pix=12.0,
                    min_matches=min_matches,
                    nbright=400,
                )
            if not st.get('ok'):
                st = _measure_offset_one_sci(
                    paths[i],
                    paths[j],
                    sci_order=0,
                    refcat=refcat,
                    match_radius_arcsec=match_radius_arcsec,
                    min_matches=min_matches,
                )
            pair = {
                'a': paths[i].name,
                'b': paths[j].name,
                'ok': bool(st.get('ok')),
                'abs_arcsec': float(st.get('abs_arcsec') or 0.0),
                'dra_arcsec': float(st.get('dra_arcsec') or 0.0),
                'ddec_arcsec': float(st.get('ddec_arcsec') or 0.0),
                'n_match': int(st.get('n_match') or st.get('peak_count') or 0),
                'method': st.get('method', 'dao'),
            }
            report['pairs'].append(pair)
            if not st.get('ok'):
                log.warning(
                    'L3 QA: could not measure %s vs %s (n=%d)',
                    paths[i].name,
                    paths[j].name,
                    pair['n_match'],
                )
                continue
            worst = max(worst, float(st['abs_arcsec']))
            if float(st['abs_arcsec']) > float(max_coherent_arcsec):
                report['failed_pairs'].append(pair)
                report['ok'] = False
    report['max_abs_arcsec'] = float(worst)
    return report


def find_hst_l3_refcat(jhat_dir: str | Path) -> Path | None:
    """Locate ``l3_ref/*.phot.txt`` under a JHAT directory when present."""
    jhat = Path(jhat_dir).expanduser().resolve()
    l3 = jhat / 'l3_ref'
    preferred = (
        l3 / 'coadd_acs_f814w_drc.phot.txt',
        l3 / 'coadd_acs_f814w_jhat.phot.txt',
        l3 / 'coadd_wfc3_f814w_drc.phot.txt',
        l3 / 'coadd_wfc3_f625w.phot.txt',
        l3 / 'coadd_wfc3_f625w_jhat.phot.txt',
        l3 / 'coadd_wfc3_f625w_drc.phot.txt',
    )
    for cand in preferred:
        if cand.is_file() and cand.stat().st_size > 0:
            return cand.resolve()
    if l3.is_dir():
        for cand in sorted(l3.glob('*.phot.txt')):
            if cand.is_file() and cand.stat().st_size > 0:
                return cand.resolve()
    return None


def _jhat_minus_raw_sky_shift(jhat_path: Path, raw_path: Path) -> tuple[float, float]:
    """SCI1 sky translation (degrees) from *raw_path* -> *jhat_path* WCS."""
    import numpy as np
    from astropy.wcs import WCS

    with as_datamodel(raw_path).open(memmap=True) as raw_hdul, as_datamodel(jhat_path).open(memmap=True) as jhat_hdul:
        ri = _sci_hdu_indices(raw_hdul)
        ji = _sci_hdu_indices(jhat_hdul)
        if not ri or not ji:
            return 0.0, 0.0
        shape = np.asarray(raw_hdul[ri[0]].data).shape[-2:]
        w0 = WCS(raw_hdul[ri[0]].header, raw_hdul, naxis=2)
        w1 = WCS(jhat_hdul[ji[0]].header, jhat_hdul, naxis=2)
        return measure_sci_sky_translation(w0, w1, shape)


def measure_hst_abs_offset_vs_refcat(
    path: str | Path,
    refcat: str | Path,
    *,
    sci_order: int = 0,
    match_radius_arcsec: float = 1.5,
    min_matches: int = 15,
    nbright: int = 300,
) -> dict[str, Any]:
    """
    Absolute sky offset of *path* vs *refcat* (ref - image), SCI *sci_order*.

    Returns degrees/arcsec suitable to **add** to CRVAL to place the frame on
    the refcat frame.
    """
    import numpy as np
    import pandas as pd
    from astropy.coordinates import SkyCoord
    from astropy.stats import sigma_clipped_stats
    from astropy.wcs import WCS
    from photutils.detection import DAOStarFinder
    import astropy.units as u

    stats: dict[str, Any] = {
        'ok': False,
        'n_match': 0,
        'dra_deg': 0.0,
        'ddec_deg': 0.0,
        'dra_arcsec': 0.0,
        'ddec_arcsec': 0.0,
        'abs_arcsec': 0.0,
    }
    ref_path = Path(refcat).expanduser().resolve()
    if not ref_path.is_file():
        return stats
    ref_df = pd.read_csv(ref_path, sep=r'\s+', engine='python')
    if 'ra' not in ref_df.columns or 'dec' not in ref_df.columns:
        return stats
    c_ref = SkyCoord(
        np.asarray(ref_df['ra'], dtype=float) * u.deg,
        np.asarray(ref_df['dec'], dtype=float) * u.deg,
    )

    with as_datamodel(path).open(memmap=True) as hdul:
        idxs = _sci_hdu_indices(hdul)
        if not idxs or sci_order >= len(idxs):
            return stats
        ext = hdul[idxs[sci_order]]
        data = np.asarray(ext.data, dtype=float)
        wcs = WCS(ext.header, hdul, naxis=2)
    mask = ~np.isfinite(data)
    _, med, std = sigma_clipped_stats(data, mask=mask, sigma=3.0, maxiters=5)
    tbl = DAOStarFinder(fwhm=2.5, threshold=5.0 * std)(data - med, mask=mask)
    if tbl is None or len(tbl) < 5:
        return stats
    xcol = 'x_centroid' if 'x_centroid' in tbl.colnames else 'xcentroid'
    ycol = 'y_centroid' if 'y_centroid' in tbl.colnames else 'ycentroid'
    tbl.sort('flux')
    tbl.reverse()
    tbl = tbl[: int(nbright)]
    ra, dec = wcs.pixel_to_world_values(tbl[xcol], tbl[ycol])
    c_img = SkyCoord(np.asarray(ra, dtype=float) * u.deg, np.asarray(dec, dtype=float) * u.deg)
    idx, sep, _ = c_img.match_to_catalog_sky(c_ref)
    good = sep.arcsec < float(match_radius_arcsec)
    n = int(np.count_nonzero(good))
    stats['n_match'] = n
    if n < int(min_matches):
        return stats
    # ref - img ; add to CRVAL to move image onto ref.
    dra = (c_ref.ra[idx] - c_img.ra).to(u.deg).value
    dra = (dra + 180.0) % 360.0 - 180.0
    ddec = (c_ref.dec[idx] - c_img.dec).to(u.deg).value
    dra_m = float(np.median(dra[good]))
    ddec_m = float(np.median(ddec[good]))
    dec0 = float(np.median(c_img.dec.degree[good]))
    dra_as = dra_m * 3600.0 * float(np.cos(np.radians(dec0)))
    ddec_as = ddec_m * 3600.0
    stats.update(
        {
            'ok': True,
            'dra_deg': dra_m,
            'ddec_deg': ddec_m,
            'dra_arcsec': dra_as,
            'ddec_arcsec': ddec_as,
            'abs_arcsec': float(np.hypot(dra_as, ddec_as)),
        }
    )
    return stats


def _snapshot_sci_wcs(path: Path) -> dict[int, dict[str, Any]]:
    """Capture SCI WCS keywords (+ a few ST123 flags) for later restore."""

    snap: dict[int, dict[str, Any]] = {}
    with as_datamodel(path).open(memmap=True) as hdul:
        for idx in _sci_hdu_indices(hdul):
            hdr = hdul[idx].header
            snap[idx] = {k: hdr[k] for k in _WCS_COPY_KEYS if k in hdr}
        # Primary flags written by prior harmonize / chip refine.
        pflags = {}
        for key in ('ST123HAR', 'ST123HRA', 'ST123HDE', 'ST123HNC', 'ST123CHP'):
            if key in hdul[0].header:
                pflags[key] = hdul[0].header[key]
        snap[-1] = pflags
    return snap


def _restore_sci_wcs_snapshot(path: Path, snap: dict[int, dict[str, Any]]) -> None:
    """Rewrite SCI WCS (+ primary ST123 flags) from :func:`_snapshot_sci_wcs`."""

    with as_datamodel(path).open(mode='update', memmap=False) as hdul:
        for idx, values in snap.items():
            if idx < 0:
                continue
            if idx >= len(hdul):
                continue
            hdr = hdul[idx].header
            for key, val in values.items():
                hdr[key] = val
        pflags = snap.get(-1) or {}
        for key in ('ST123HAR', 'ST123HRA', 'ST123HDE', 'ST123HNC', 'ST123CHP'):
            if key in pflags:
                hdul[0].header[key] = pflags[key]
            elif key in hdul[0].header:
                del hdul[0].header[key]
        hdul.flush()


def _hst_visit_key(path: Path) -> str:
    """
    Coarse HST visit / obset key for splitting mixed archives.

    Prefer ``ROOTNAME`` (ipppssoot) chars [:6]; fall back to the JHAT stem.
    Distinct programs/visits (e.g. ``ie9801`` vs ``iejn02``) must not share a
    single pipeline-relative restore - their calibrated relative WCS differ.
    """

    try:
        with as_datamodel(path).open(memmap=True) as hdul:
            for hdu in hdul:
                root = hdu.header.get('ROOTNAME') or hdu.header.get('ROOT')
                if root:
                    root = str(root).strip().lower()
                    if len(root) >= 6:
                        return root[:6]
    except Exception:
        pass
    stem = path.name.lower()
    for suf in ('_jhat.fits', '_flc.fits', '_flt.fits', '_c0m.fits', '.fits'):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem[:6] if len(stem) >= 6 else stem


def measure_hst_sky_offset_vs_gaia(
    path: str | Path,
    *,
    max_offset_arcsec: float = HST_ABS_OFFSET_MAX_ARCSEC,
    nbright: int = 400,
    min_matches: int = 12,
) -> dict[str, Any]:
    """
    Absolute sky offset of one frame vs Gaia (Vizier), via detection matching.

    Matches detections to Gaia on sky (RA/Dec) only - avoids
    ``all_world2pix`` failures on pathological WFPC2 WCS solutions that break
    refcat x/y projection.

    Returns the same key shape as :func:`measure_hst_sky_offset_2dhist`
    (``ok``, ``dra_deg``, ``ddec_deg``, ``abs_arcsec``, ...).
    """
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy.stats import sigma_clipped_stats
    from astropy.wcs import WCS
    from photutils.detection import DAOStarFinder
    import astropy.units as u

    from st123.stages.alignment.gaia_catalog import (
        default_gaia_cache_dir,
        fetch_gaia_cone,
        load_gaia_cache,
    )

    stats: dict[str, Any] = {
        'path': str(Path(path)),
        'method': 'gaia_match',
        'ok': False,
        'n_match': 0,
        'dra_deg': 0.0,
        'ddec_deg': 0.0,
        'dra_arcsec': 0.0,
        'ddec_arcsec': 0.0,
        'abs_arcsec': 0.0,
        'peak_count': 0,
    }
    path = Path(path).expanduser().resolve()
    try:
        # Prefer the field cache (or a CRVAL cone) and skip cut_gaia_sources -
        # WFPC2 WCS often fails all_world2pix during the image footprint cut.
        gaia = None
        cache_dir = default_gaia_cache_dir(path)
        if cache_dir is not None:
            cached, _meta = load_gaia_cache(cache_dir)
            if cached is not None and len(cached) >= int(min_matches):
                gaia = cached
        if gaia is None:
            with as_datamodel(path).open(memmap=True) as hdul:
                idxs = _sci_hdu_indices(hdul)
                if not idxs:
                    return stats
                hdr = hdul[idxs[0]].header
                ra0 = float(hdr.get('CRVAL1') or hdul[0].header.get('CRVAL1'))
                dec0 = float(hdr.get('CRVAL2') or hdul[0].header.get('CRVAL2'))
            gaia = fetch_gaia_cone(
                SkyCoord(ra0 * u.deg, dec0 * u.deg),
                0.08 * u.deg,
            )
        if gaia is None or len(gaia) < int(min_matches):
            stats['error'] = 'insufficient_gaia'
            return stats
        c_ref = SkyCoord(
            np.asarray(gaia['ra'], dtype=float) * u.deg,
            np.asarray(gaia['dec'], dtype=float) * u.deg,
        )

        best: dict[str, Any] | None = None
        with as_datamodel(path).open(memmap=True) as hdul:
            idxs = _sci_hdu_indices(hdul)
            if not idxs:
                return stats
            for sci_i in idxs:
                ext = hdul[sci_i]
                data = np.asarray(ext.data, dtype=float)
                try:
                    wcs = WCS(ext.header, hdul, naxis=2)
                except Exception:
                    continue
                mask = ~np.isfinite(data)
                _, med, std = sigma_clipped_stats(
                    data, mask=mask, sigma=3.0, maxiters=5
                )
                if not np.isfinite(std) or std <= 0:
                    continue
                tbl = DAOStarFinder(fwhm=2.5, threshold=5.0 * std)(
                    data - med, mask=mask
                )
                if tbl is None or len(tbl) < int(min_matches):
                    continue
                xcol = 'x_centroid' if 'x_centroid' in tbl.colnames else 'xcentroid'
                ycol = 'y_centroid' if 'y_centroid' in tbl.colnames else 'ycentroid'
                tbl.sort('flux')
                tbl.reverse()
                tbl = tbl[: int(nbright)]
                try:
                    ra_img, dec_img = wcs.pixel_to_world_values(
                        np.asarray(tbl[xcol], dtype=float),
                        np.asarray(tbl[ycol], dtype=float),
                    )
                except Exception:
                    continue
                c_img = SkyCoord(
                    np.asarray(ra_img, dtype=float) * u.deg,
                    np.asarray(dec_img, dtype=float) * u.deg,
                )
                idx, sep, _ = c_img.match_to_catalog_sky(c_ref)
                good = sep.arcsec < float(max_offset_arcsec)
                n = int(np.count_nonzero(good))
                if n < int(min_matches):
                    continue
                # Image - Gaia; apply -Delta to CRVAL to move image onto Gaia.
                dra = (c_img.ra[good] - c_ref.ra[idx[good]]).to(u.deg).value
                ddec = (c_img.dec[good] - c_ref.dec[idx[good]]).to(u.deg).value
                dra_m = float(np.median(dra))
                ddec_m = float(np.median(ddec))
                dec0 = float(np.median(c_img.dec[good].deg))
                dra_as = dra_m * 3600.0 * float(np.cos(np.radians(dec0)))
                ddec_as = ddec_m * 3600.0
                cand = {
                    'ok': True,
                    'n_match': n,
                    'peak_count': n,
                    'dra_deg': dra_m,
                    'ddec_deg': ddec_m,
                    'dra_arcsec': dra_as,
                    'ddec_arcsec': ddec_as,
                    'abs_arcsec': float(np.hypot(dra_as, ddec_as)),
                    'sci_ext': int(sci_i),
                }
                if best is None or cand['n_match'] > best['n_match']:
                    best = cand
        if best is not None:
            stats.update(best)
    except Exception as exc:
        stats['error'] = f'{type(exc).__name__}: {exc}'
    return stats


def apply_common_abs_shift_vs_ref(
    frames: list[Path],
    abs_ref: Path | None,
    *,
    max_abs_offset_arcsec: float = HST_ABS_OFFSET_MAX_ARCSEC,
    min_apply_arcsec: float = 0.005,
    allow_gaia_fallback: bool = True,
) -> dict[str, Any]:
    """
    Apply one common CRVAL shift so the visit's deepest frame matches *abs_ref*.

    Used after within-visit relative harmonize so multi-visit filter groups share
    an absolute frame before AstroDrizzle. When the coadd/image 2-D hist tie
    fails, fall back to a Gaia match on the visit anchor (same common shift).
    """

    paths = [Path(p).resolve() for p in frames]
    report: dict[str, Any] = {
        'ok': False,
        'n_frames': len(paths),
        'abs_ref': str(abs_ref) if abs_ref else None,
        'dra_arcsec': 0.0,
        'ddec_arcsec': 0.0,
        'abs_arcsec': 0.0,
        'applied': False,
        'method': None,
    }
    if not paths:
        return report
    anchor = max(paths, key=_frame_exptime)
    report['anchor'] = str(anchor)

    off: dict[str, Any] = {'ok': False}
    if abs_ref is not None and Path(abs_ref).is_file():
        off = measure_hst_sky_offset_2dhist(
            anchor,
            abs_ref,
            max_offset_arcsec=max_abs_offset_arcsec,
        )
        report['probe'] = {
            'ok': bool(off.get('ok')),
            'abs_arcsec': off.get('abs_arcsec'),
            'dra_arcsec': off.get('dra_arcsec'),
            'ddec_arcsec': off.get('ddec_arcsec'),
            'peak_count': off.get('peak_count'),
        }
        if off.get('ok'):
            report['method'] = 'abs_ref_2dhist'

    if not off.get('ok') and allow_gaia_fallback:
        log.info(
            'Visit abs retie: coadd/image probe failed for %s; trying Gaia',
            anchor.name,
        )
        off = measure_hst_sky_offset_vs_gaia(
            anchor, max_offset_arcsec=max_abs_offset_arcsec
        )
        report['gaia_probe'] = {
            'ok': bool(off.get('ok')),
            'abs_arcsec': off.get('abs_arcsec'),
            'dra_arcsec': off.get('dra_arcsec'),
            'ddec_arcsec': off.get('ddec_arcsec'),
            'n_match': off.get('n_match'),
        }
        if off.get('ok'):
            report['method'] = 'gaia_match'
            report['abs_ref'] = 'Gaia'

    if not off.get('ok'):
        return report
    abs_as = float(off['abs_arcsec'])
    report['ok'] = True
    report['dra_arcsec'] = float(off['dra_arcsec'])
    report['ddec_arcsec'] = float(off['ddec_arcsec'])
    report['abs_arcsec'] = abs_as
    if abs_as < float(min_apply_arcsec):
        return report
    # Cap large multimodal / false-Gaia peaks (JWST hub retie semantics).
    if abs_as > float(FRAME_ABS_RETIE_MAX_APPLY_ARCSEC):
        report['skipped_large'] = True
        log.warning(
            'Visit abs retie skip %s: |Delta|=%.3f" > max apply %.3f"',
            anchor.name,
            abs_as,
            FRAME_ABS_RETIE_MAX_APPLY_ARCSEC,
        )
        return report
    peak = int(off.get('peak_count') or off.get('n_match') or 0)
    if peak and peak < int(FRAME_ABS_RETIE_MIN_PEAK) and report.get('method') == 'abs_ref_2dhist':
        report['skipped_weak_peak'] = True
        log.warning(
            'Visit abs retie skip %s: weak 2dhist peak=%d < %d',
            anchor.name,
            peak,
            FRAME_ABS_RETIE_MIN_PEAK,
        )
        return report
    dra_deg = -float(off['dra_deg'])
    ddec_deg = -float(off['ddec_deg'])
    for path in paths:
        with as_datamodel(path).open(mode='update', memmap=False) as hdul:
            apply_sky_translation_to_sci(
                hdul,
                dra_deg,
                ddec_deg,
                comment=f'st123: visit common abs ({report["method"]})',
            )
            hdul[0].header['ST123VAB'] = (
                True,
                'st123: visit-level abs retie to shared ref',
            )
            hdul[0].header['ST123VAM'] = (
                str(report['method'] or ''),
                'st123: visit abs retie method',
            )
            hdul.flush()
    report['applied'] = True
    log.info(
        'Visit abs retie %s -> %s (%s): applied opposite of dRA=%+.3f" dDec=%+.3f" '
        '(|Delta|=%.3f") on %d frame(s)',
        anchor.name,
        report.get('abs_ref'),
        report.get('method'),
        off['dra_arcsec'],
        off['ddec_arcsec'],
        abs_as,
        len(paths),
    )
    return report


def validate_hst_visits_internal_alignment(
    frames: list[str | Path],
    *,
    max_coherent_arcsec: float = HST_INTERNAL_ALIGN_MAX_ARCSEC,
    min_matches: int = 12,
) -> dict[str, Any]:
    """
    Within-visit internal QA for mixed archives (cross-visit pairs ignored).
    """
    from collections import defaultdict

    paths = [Path(p).expanduser().resolve() for p in frames]
    by_visit: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        by_visit[_hst_visit_key(p)].append(p)
    visit_rows: list[dict[str, Any]] = []
    all_ok = True
    max_abs = 0.0
    for vid, vpaths in sorted(by_visit.items()):
        if len(vpaths) < 2:
            visit_rows.append({'visit': vid, 'n_frames': 1, 'ok': True, 'max_abs_arcsec': 0.0})
            continue
        qa = validate_hst_group_internal_alignment(
            vpaths,
            max_coherent_arcsec=max_coherent_arcsec,
            min_matches=min_matches,
        )
        row = {
            'visit': vid,
            'n_frames': len(vpaths),
            'ok': bool(qa.get('ok')),
            'max_abs_arcsec': qa.get('max_abs_arcsec'),
            'failed_pairs': qa.get('failed_pairs'),
        }
        visit_rows.append(row)
        all_ok = all_ok and bool(qa.get('ok'))
        if qa.get('max_abs_arcsec') is not None:
            max_abs = max(max_abs, float(qa['max_abs_arcsec']))
    return {
        'ok': all_ok,
        'max_abs_arcsec': max_abs,
        'n_visits': len(by_visit),
        'visits': visit_rows,
        'scope': 'within_visit',
    }


def restore_pipeline_relative_wcs_with_common_shift(
    frames: list[Path],
    *,
    dra_deg: float,
    ddec_deg: float,
) -> list[dict[str, Any]]:
    """
    Reset each JHAT MEF SCI WCS to its calibrated raw sibling, then apply one
    common sky shift. Preserves pipeline dither / chip geometry.
    """

    rows: list[dict[str, Any]] = []
    for path in frames:
        raw = _raw_sibling_for_jhat(path)
        row: dict[str, Any] = {
            'path': str(path),
            'raw': str(raw) if raw else None,
            'applied': False,
            'n_sci': 0,
        }
        if raw is None:
            log.warning('No raw sibling for %s; cannot restore relative WCS', path.name)
            rows.append(row)
            continue
        with as_datamodel(raw).open(memmap=True) as raw_hdul, as_datamodel(path).open(mode='update', memmap=False) as jhat_hdul:
            raw_idxs = _sci_hdu_indices(raw_hdul)
            jhat_idxs = _sci_hdu_indices(jhat_hdul)
            n = min(len(raw_idxs), len(jhat_idxs))
            row['n_sci'] = n
            for k in range(n):
                src = raw_hdul[raw_idxs[k]].header
                dst = jhat_hdul[jhat_idxs[k]].header
                for key in _WCS_COPY_KEYS:
                    if key in src:
                        dst[key] = src[key]
            apply_sky_translation_to_sci(
                jhat_hdul,
                float(dra_deg),
                float(ddec_deg),
                sci_indices=jhat_idxs[:n],
                comment='st123: common group abs shift on pipeline WCS',
            )
            dec0 = float(jhat_hdul[jhat_idxs[0]].header.get('CRVAL2', 0.0))
            import numpy as np

            dra_as = float(dra_deg) * 3600.0 * float(np.cos(np.radians(dec0)))
            ddec_as = float(ddec_deg) * 3600.0
            jhat_hdul[0].header['ST123HAR'] = (
                True,
                'st123: pipeline-relative WCS + common abs shift',
            )
            jhat_hdul[0].header['ST123HRA'] = (dra_as, '[arcsec] common dRA cos(Dec)')
            jhat_hdul[0].header['ST123HDE'] = (ddec_as, '[arcsec] common dDec')
            jhat_hdul[0].header['ST123CHP'] = (
                False,
                'st123: per-SCI CRPIX refine cleared by relative restore',
            )
            jhat_hdul.flush()
            row['applied'] = True
            row['dra_arcsec'] = dra_as
            row['ddec_arcsec'] = ddec_as
        rows.append(row)
        log.info(
            'Restored pipeline WCS + common shift on %s (dRA=%+.3f" dDec=%+.3f")',
            path.name,
            row.get('dra_arcsec', 0.0),
            row.get('ddec_arcsec', 0.0),
        )
    return rows


def harmonize_hst_visits_across_filters(
    frames: Sequence[str | Path],
    *,
    abs_ref: str | Path | None = None,
    max_internal_arcsec: float = 0.25,
    max_abs_offset_arcsec: float = HST_ABS_OFFSET_MAX_ARCSEC,
    min_apply_arcsec: float = 0.005,
) -> dict[str, Any]:
    """
    Within each HST visit, put **all filters** on one relative + absolute frame.

    Per-filter harmonize can Gaia-retie only the F555W members of a visit while
    skipping an "already aligned" F814W pair, leaving a ~0.05" inter-filter
    split. This pass:

    1. Groups JHAT frames by visit key (ignoring filter).
    2. Shifts every frame onto the deepest visit anchor (2-D hist, else pixel).
    3. Applies one common absolute retie (abs_ref / Gaia) to the whole visit.
    """
    from collections import defaultdict


    paths = [Path(p).expanduser().resolve() for p in frames if Path(p).is_file()]
    abs_ref_path = Path(abs_ref).expanduser().resolve() if abs_ref else None
    if abs_ref_path is not None and not abs_ref_path.is_file():
        abs_ref_path = None

    by_visit: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        by_visit[_hst_visit_key(p)].append(p)

    report: dict[str, Any] = {
        'ok': True,
        'n_visits': len(by_visit),
        'abs_ref': str(abs_ref_path) if abs_ref_path else None,
        'visits': [],
    }
    for vid, vpaths in sorted(by_visit.items()):
        uniq = sorted({p.resolve() for p in vpaths})
        row: dict[str, Any] = {
            'visit': vid,
            'n_frames': len(uniq),
            'ok': True,
            'relative': [],
            'abs_retie': None,
        }
        if not uniq:
            report['visits'].append(row)
            continue

        anchor = max(uniq, key=_frame_exptime)
        row['anchor'] = str(anchor)
        for path in uniq:
            if path == anchor:
                continue
            off = measure_hst_sky_offset_2dhist(
                path, anchor, max_offset_arcsec=max_abs_offset_arcsec
            )
            if not off.get('ok'):
                off = measure_hst_frame_relative_offset_pixel(
                    path, anchor, min_matches=8
                )
            if not off.get('ok'):
                row['ok'] = False
                row['relative'].append(
                    {
                        'path': str(path),
                        'ok': False,
                        'error': 'relative_match_failed',
                    }
                )
                continue
            abs_as = float(off.get('abs_arcsec') or 0.0)
            entry = {
                'path': str(path),
                'ok': True,
                'abs_arcsec': abs_as,
                'dra_arcsec': float(off.get('dra_arcsec') or 0.0),
                'ddec_arcsec': float(off.get('ddec_arcsec') or 0.0),
                'applied': False,
            }
            if abs_as > float(max_internal_arcsec):
                row['ok'] = False
                entry['ok'] = False
                entry['error'] = (
                    f'relative |Delta|={abs_as:.3f}" > {max_internal_arcsec:.3f}"'
                )
                row['relative'].append(entry)
                continue
            if abs_as >= float(min_apply_arcsec):
                with as_datamodel(path).open(mode='update', memmap=False) as hdul:
                    apply_sky_translation_to_sci(
                        hdul,
                        -float(off['dra_deg']),
                        -float(off['ddec_deg']),
                        comment='st123: visit cross-filter relative',
                    )
                    hdul[0].header['ST123VXF'] = (
                        True,
                        'st123: visit cross-filter relative to deepest',
                    )
                    hdul.flush()
                entry['applied'] = True
                entry['dra_arcsec'] = -float(off['dra_arcsec'])
                entry['ddec_arcsec'] = -float(off['ddec_arcsec'])
            row['relative'].append(entry)

        row['abs_retie'] = apply_common_abs_shift_vs_ref(
            uniq,
            abs_ref_path,
            max_abs_offset_arcsec=max_abs_offset_arcsec,
            min_apply_arcsec=min_apply_arcsec,
            allow_gaia_fallback=True,
        )
        if row['relative'] and any(not r.get('ok') for r in row['relative']):
            report['ok'] = False
        n_rel = sum(1 for r in row['relative'] if r.get('applied'))
        if n_rel or (row.get('abs_retie') or {}).get('applied'):
            log.info(
                'Visit cross-filter harmonize %s: %d frame(s), '
                'relative-shifted %d, abs_retie=%s',
                vid,
                len(uniq),
                n_rel,
                (row.get('abs_retie') or {}).get('method'),
            )
        report['visits'].append(row)
    return report


def harmonize_hst_group_wcs(
    frames: list[str | Path],
    *,
    max_internal_arcsec: float = HST_INTERNAL_ALIGN_MAX_ARCSEC,
    match_radius_arcsec: float = 1.0,
    min_matches: int = 12,
    min_correct_arcsec: float = 0.008,
    max_iters: int = 4,
    anchor: str | Path | None = None,
    refcat: str | Path | None = None,
    abs_ref: str | Path | None = None,
    max_abs_offset_arcsec: float = HST_ABS_OFFSET_MAX_ARCSEC,
    force: bool = False,
    allow_visit_split: bool = True,
) -> dict[str, Any]:
    """
    Force relative WCS agreement among L2 frames before AstroDrizzle.

    Independent JHAT / per-chip CRPIX solutions often break the pipeline's
    intra-visit dither WCS (classic drizzle ghosting). This restores each
    frame's calibrated-raw SCI WCS (preserving dithers) and applies **one
    common** absolute sky shift measured with a 2-D offset histogram vs
    *abs_ref* (default: F814 coadd or F625), searching up to
    *max_abs_offset_arcsec* (default 5"). Small-radius catalog matches are
    not used for the absolute tie - they false-lock near 0".

    Mixed multi-visit / multi-program archives are split by visit key before
    pipeline-relative restore (cross-visit residuals are left for mosaic).
    When a restore would leave the group worse than *max_internal_arcsec*,
    the pre-restore JHAT WCS is reinstated and the report returns
    ``ok=False`` with ``method='jhat_kept_unharmonized'`` (no raise).
    """
    import numpy as np
    from collections import defaultdict

    del match_radius_arcsec, min_correct_arcsec, max_iters, refcat  # API compat
    paths = [Path(p).expanduser().resolve() for p in frames]
    abs_ref_path = Path(abs_ref).expanduser().resolve() if abs_ref else None
    if abs_ref_path is None:
        abs_ref_path = find_hst_abs_ref_image(paths[0].parent)
    report: dict[str, Any] = {
        'n_frames': len(paths),
        'anchor': None,
        'method': 'pipeline_relative_common_shift',
        'abs_ref': str(abs_ref_path) if abs_ref_path else None,
        'corrections': [],
        'pre': None,
        'post': None,
        'ok': True,
        'iterations': 0,
        'common_dra_arcsec': 0.0,
        'common_ddec_arcsec': 0.0,
    }
    qa_kw = dict(
        max_coherent_arcsec=max_internal_arcsec,
        min_matches=min_matches,
    )
    if len(paths) < 2:
        report['pre'] = validate_hst_group_internal_alignment(paths, **qa_kw)
        report['post'] = report['pre']
        return report

    if anchor is not None:
        anchor_path = Path(anchor).expanduser().resolve()
    else:
        anchor_path = max(paths, key=_frame_exptime)
    if anchor_path not in paths:
        raise FileNotFoundError(f'anchor {anchor_path} not in group')
    report['anchor'] = str(anchor_path)

    # Multi-visit filter groups: pipeline-relative WCS is only coherent within
    # a visit. Split *before* expensive full-group QA; leave cross-visit for mosaic.
    by_visit: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        by_visit[_hst_visit_key(p)].append(p)
    if allow_visit_split and len(by_visit) > 1:
        log.info(
            'Splitting %d-frame group into %d visit(s) for harmonize: %s',
            len(paths),
            len(by_visit),
            ', '.join(f'{k}:{len(v)}' for k, v in sorted(by_visit.items())),
        )
        sub_reports: list[dict[str, Any]] = []
        all_ok = True
        for vid, vpaths in sorted(by_visit.items()):
            if len(vpaths) < 2:
                sub = {
                    'visit': vid,
                    'n_frames': len(vpaths),
                    'ok': True,
                    'method': 'singleton_visit',
                }
            else:
                sub = harmonize_hst_group_wcs(
                    vpaths,
                    max_internal_arcsec=max_internal_arcsec,
                    min_matches=min_matches,
                    abs_ref=abs_ref_path,
                    max_abs_offset_arcsec=max_abs_offset_arcsec,
                    force=force,
                    allow_visit_split=False,
                )
                sub['visit'] = vid
            # Tie every visit (incl. singletons) to the shared abs_ref so a
            # multi-visit filter coadd does not ghost at the ~0.1-0.2" level.
            # Gaia fallback covers visits where coadd 2-D hist matching fails
            # (e.g. sparse WFPC2 epochs vs a modern ACS/WFC3 abs_ref).
            if bool(sub.get('ok')):
                ref_for_retie = (
                    abs_ref_path
                    if abs_ref_path is not None and abs_ref_path.is_file()
                    else None
                )
                sub['abs_retie'] = apply_common_abs_shift_vs_ref(
                    vpaths,
                    ref_for_retie,
                    max_abs_offset_arcsec=max_abs_offset_arcsec,
                    allow_gaia_fallback=True,
                )
            sub_reports.append(sub)
            all_ok = all_ok and bool(sub.get('ok'))
        report['method'] = 'visit_split_harmonize'
        report['subgroups'] = sub_reports
        report['ok'] = all_ok
        # Avoid O(n^2) full-group QA across visits; summarize from subgroups.
        sub_max = [
            float((s.get('post') or s.get('pre') or {}).get('max_abs_arcsec') or 0.0)
            for s in sub_reports
            if (s.get('post') or s.get('pre')) is not None
        ]
        report['pre'] = {
            'ok': False,
            'max_abs_arcsec': max(sub_max) if sub_max else None,
            'note': 'multi-visit; per-visit QA only',
        }
        report['post'] = {
            'ok': all_ok,
            'max_abs_arcsec': max(sub_max) if sub_max else None,
            'n_visits': len(by_visit),
            'n_visit_ok': sum(1 for s in sub_reports if s.get('ok')),
        }
        if all_ok:
            log.info(
                'Visit-split harmonize OK for %d visit(s) (within-visit max |Delta|=%.3f")',
                len(by_visit),
                report['post']['max_abs_arcsec'] or 0.0,
            )
        else:
            log.warning(
                'Visit-split harmonize incomplete for group around %s '
                '(%d/%d visit subgroup(s) kept prior WCS / failed)',
                anchor_path.name,
                sum(1 for s in sub_reports if not s.get('ok')),
                len(sub_reports),
            )
        return report

    report['pre'] = validate_hst_group_internal_alignment(paths, **qa_kw)

    # Probe absolute offset before deciding to rewrite WCS.
    need_abs = bool(force)
    if abs_ref_path is not None and abs_ref_path.is_file() and not need_abs:
        probe = measure_hst_sky_offset_2dhist(
            anchor_path,
            abs_ref_path,
            max_offset_arcsec=max_abs_offset_arcsec,
        )
        report['abs_probe'] = {
            'ok': bool(probe.get('ok')),
            'abs_arcsec': probe.get('abs_arcsec'),
            'dra_arcsec': probe.get('dra_arcsec'),
            'ddec_arcsec': probe.get('ddec_arcsec'),
            'peak_count': probe.get('peak_count'),
        }
        if probe.get('ok') and float(probe['abs_arcsec']) > FRAME_ABS_TOL_ARCSEC:
            need_abs = True
            log.info(
                'Absolute probe %s vs %s: |Delta|=%.3f" (dRA=%+.3f dDec=%+.3f) -> retie',
                anchor_path.name,
                abs_ref_path.name,
                probe['abs_arcsec'],
                probe['dra_arcsec'],
                probe['ddec_arcsec'],
            )

    if report['pre']['ok'] and not need_abs:
        report['post'] = report['pre']
        log.info(
            'Group already aligned (internal max |Delta|=%.3f"; abs OK) - skip',
            report['pre']['max_abs_arcsec'],
        )
        return report

    raw_siblings = {p: _raw_sibling_for_jhat(p) for p in paths}
    # Snapshot JHAT WCS so a failed pipeline restore can be undone.
    wcs_backup = {p: _snapshot_sci_wcs(p) for p in paths}

    def _revert_jhat_wcs(reason: str) -> None:
        for p in paths:
            try:
                _restore_sci_wcs_snapshot(p, wcs_backup[p])
            except Exception as exc:
                log.error('Failed to revert JHAT WCS on %s: %s', p.name, exc)
        log.warning(
            'Reverted JHAT WCS for %d frame(s) after failed harmonize (%s)',
            len(paths),
            reason,
        )

    if not any(raw_siblings.values()):
        log.warning(
            'No raw siblings for group around %s; falling back to sibling CRVAL tweak',
            anchor_path.name,
        )
        for path in paths:
            if path == anchor_path:
                continue
            off = measure_hst_frame_relative_offset_pixel(
                path, anchor_path, min_matches=min_matches
            )
            if not off.get('ok'):
                continue
            with as_datamodel(path).open(mode='update', memmap=False) as hdul:
                apply_sky_translation_to_sci(
                    hdul,
                    -float(off['dra_deg']),
                    -float(off['ddec_deg']),
                    comment='st123: sibling relative harmonize',
                )
                hdul[0].header['ST123HAR'] = True
                hdul.flush()
            report['corrections'].append(
                {
                    'path': str(path),
                    'dra_arcsec': -float(off['dra_arcsec']),
                    'ddec_arcsec': -float(off['ddec_arcsec']),
                }
            )
    else:
        # Skip restoring frames that are themselves the abs reference.
        if abs_ref_path is not None and any(
            p.resolve() == abs_ref_path for p in paths
        ):
            report['post'] = report['pre']
            report['method'] = 'abs_ref_group_skip'
            log.info('Skipping harmonize for abs-ref group (%s)', abs_ref_path.name)
            return report

        # 1) Restore pipeline WCS (relative OK, absolute = pipeline).
        restore_pipeline_relative_wcs_with_common_shift(
            paths, dra_deg=0.0, ddec_deg=0.0
        )
        # 2) Absolute via 2-D histogram vs abs_ref (handles up to ~5").
        dra_deg = ddec_deg = 0.0
        if abs_ref_path is not None and abs_ref_path.is_file():
            abs_off = measure_hst_sky_offset_2dhist(
                anchor_path,
                abs_ref_path,
                max_offset_arcsec=max_abs_offset_arcsec,
            )
            if abs_off.get('ok'):
                # img-ref -> apply -Delta to CRVAL.
                dra_deg = -float(abs_off['dra_deg'])
                ddec_deg = -float(abs_off['ddec_deg'])
                report['method'] = 'pipeline_relative_2dhist_abs'
                log.info(
                    '2D-hist abs %s -> %s: img-ref dRA=%+.3f" dDec=%+.3f" '
                    '(peak n=%d); applying opposite to group',
                    anchor_path.name,
                    abs_ref_path.name,
                    abs_off['dra_arcsec'],
                    abs_off['ddec_arcsec'],
                    abs_off['peak_count'],
                )
            else:
                log.warning(
                    '2D-hist abs failed for %s vs %s (pairs=%d peak=%d)',
                    anchor_path.name,
                    abs_ref_path.name,
                    abs_off.get('n_pairs', 0),
                    abs_off.get('peak_count', 0),
                )
                report['method'] = 'pipeline_relative_only'
        else:
            report['method'] = 'pipeline_relative_only'
            log.warning('No abs_ref for %s; leaving pipeline absolute WCS', anchor_path.name)

        if abs(dra_deg) > 0 or abs(ddec_deg) > 0:
            for path in paths:
                with as_datamodel(path).open(mode='update', memmap=False) as hdul:
                    apply_sky_translation_to_sci(
                        hdul,
                        dra_deg,
                        ddec_deg,
                        comment='st123: common 2dhist abs shift',
                    )
                    dec0 = float(
                        hdul[_sci_hdu_indices(hdul)[0]].header.get('CRVAL2', 0.0)
                    )
                    dra_as = dra_deg * 3600.0 * float(np.cos(np.radians(dec0)))
                    ddec_as = ddec_deg * 3600.0
                    hdul[0].header['ST123HRA'] = (
                        dra_as,
                        '[arcsec] common dRA cos(Dec)',
                    )
                    hdul[0].header['ST123HDE'] = (
                        ddec_as,
                        '[arcsec] common dDec',
                    )
                    if abs_ref_path is not None:
                        hdul[0].header['ST123HNC'] = (
                            abs_ref_path.name,
                            'absolute reference for common shift',
                        )
                    hdul.flush()
            report['common_dra_arcsec'] = float(
                dra_deg * 3600.0 * np.cos(np.radians(55.35))
            )
            report['common_ddec_arcsec'] = float(ddec_deg * 3600.0)
        report['iterations'] = 1
        report['corrections'] = [{'path': str(p), 'restored': True} for p in paths]
        log.info(
            'Pipeline-relative restore + common abs on %d frames: dRA=%+.3f" dDec=%+.3f"',
            len(paths),
            report['common_dra_arcsec'],
            report['common_ddec_arcsec'],
        )

    report['post'] = validate_hst_group_internal_alignment(paths, **qa_kw)
    report['ok'] = bool(report['post']['ok'])
    if not report['ok']:
        reason = (
            f'max |Delta|={report["post"]["max_abs_arcsec"]:.3f}" > '
            f'{max_internal_arcsec:.3f}"'
        )
        _revert_jhat_wcs(reason)
        # Pipeline relative can itself be incoherent (e.g. some WFC3/IR visits).
        # Fall back to pairwise CRVAL tweaks onto the anchor.
        log.info(
            'Trying sibling CRVAL harmonize for %d frame(s) after pipeline restore failed',
            len(paths),
        )
        sib_corrections: list[dict[str, Any]] = []
        for path in paths:
            if path == anchor_path:
                continue
            off = measure_hst_frame_relative_offset_pixel(
                path, anchor_path, min_matches=min_matches
            )
            if not off.get('ok'):
                continue
            with as_datamodel(path).open(mode='update', memmap=False) as hdul:
                apply_sky_translation_to_sci(
                    hdul,
                    -float(off['dra_deg']),
                    -float(off['ddec_deg']),
                    comment='st123: sibling relative after pipeline fail',
                )
                hdul[0].header['ST123HAR'] = (
                    True,
                    'st123: sibling CRVAL harmonize (pipeline relative failed)',
                )
                hdul.flush()
            sib_corrections.append(
                {
                    'path': str(path),
                    'dra_arcsec': -float(off['dra_arcsec']),
                    'ddec_arcsec': -float(off['ddec_arcsec']),
                }
            )
        report['post'] = validate_hst_group_internal_alignment(paths, **qa_kw)
        if report['post']['ok'] and sib_corrections:
            report['method'] = 'sibling_relative_after_pipeline_fail'
            report['ok'] = True
            report['corrections'] = sib_corrections
            log.info(
                'Sibling CRVAL harmonize OK: max |Delta| -> %.3f" (limit %.3f")',
                report['post']['max_abs_arcsec'],
                max_internal_arcsec,
            )
            return report
        # Sibling path also failed - reinstate pre-harmonize WCS and soft-fail.
        _revert_jhat_wcs('sibling CRVAL harmonize also failed')
        report['method'] = 'jhat_kept_unharmonized'
        report['ok'] = False
        report['post_failed'] = report['post']
        report['corrections'] = sib_corrections
        report['post'] = validate_hst_group_internal_alignment(paths, **qa_kw)
        log.warning(
            'HST group harmonize aborted; kept prior WCS (%s; failed_pairs=%s)',
            reason,
            report.get('post_failed', {}).get('failed_pairs'),
        )
        return report
    log.info(
        'Group harmonize OK: max |Delta| %.3f" -> %.3f" (limit %.3f")',
        report['pre']['max_abs_arcsec'],
        report['post']['max_abs_arcsec'],
        max_internal_arcsec,
    )
    return report


def harmonize_hst_jhat_dir(
    jhat_dir: str | Path,
    *,
    pattern: str = '*_jhat.fits',
    max_internal_arcsec: float = HST_INTERNAL_ALIGN_MAX_ARCSEC,
    refcat: str | Path | None = None,
    abs_ref: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Harmonize every (instrument, filter) group under a JHAT directory.
    """
    from collections import defaultdict

    from st123.utils.helpers import get_filter, get_instrument

    jhat = Path(jhat_dir).expanduser().resolve()
    del refcat  # absolute ties use abs_ref / 2dhist, not sparse L3 catalog matches
    abs_ref_path = (
        Path(abs_ref).expanduser().resolve()
        if abs_ref
        else find_hst_abs_ref_image(jhat)
    )
    # Mixed JWST/HST projects often share reduction/jhat; never harmonize JWST
    # products with the HST relative-WCS path.
    _hst_insts = frozenset({'acs', 'wfc3', 'wfpc2'})
    frames = sorted(
        p
        for p in jhat.glob(pattern)
        if not p.name.lower().startswith('coadd_')
        and not p.name.lower().startswith('jw')
        and 'l3_ref' not in p.parts
    )
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for p in frames:
        try:
            inst = get_instrument(p).split('_')[0].lower()
            filt = get_filter(p).lower()
        except Exception as exc:
            log.warning('Skipping ungroupable frame %s (%s)', p, exc)
            continue
        if inst not in _hst_insts:
            log.warning('Skipping non-HST instrument %s for %s', inst, p.name)
            continue
        groups[(inst, filt)].append(p.resolve())
    results: list[dict[str, Any]] = []
    all_imgs: list[Path] = []
    for (inst, filt), imgs in sorted(groups.items()):
        try:
            harm = harmonize_hst_group_wcs(
                sorted(imgs),
                max_internal_arcsec=max_internal_arcsec,
                abs_ref=abs_ref_path,
            )
            method = harm.get('method')
            if harm.get('ok'):
                status = 'ok'
            elif method in (
                'jhat_kept_unharmonized',
                'visit_split_harmonize',
            ):
                # Soft: prior WCS retained for incoherent visit(s); mosaic continues.
                status = 'skipped'
                log.warning(
                    'Harmonize skipped for %s/%s (kept prior WCS; method=%s)',
                    inst,
                    filt,
                    method,
                )
            else:
                status = 'failed'
                log.error(
                    'Harmonize failed for %s/%s: method=%s',
                    inst,
                    filt,
                    method,
                )
            results.append(
                {
                    'instrument': inst,
                    'filter': filt,
                    'status': status,
                    'harmonize': harm,
                }
            )
            all_imgs.extend(Path(p).resolve() for p in imgs)
        except Exception as exc:
            log.error('Harmonize failed for %s/%s: %s', inst, filt, exc)
            results.append(
                {
                    'instrument': inst,
                    'filter': filt,
                    'status': 'failed',
                    'error': f'{type(exc).__name__}: {exc}',
                }
            )
            all_imgs.extend(Path(p).resolve() for p in imgs)

    # Cross-filter visit retie: one common absolute + relative frame per visit.
    if all_imgs:
        try:
            visit_x = harmonize_hst_visits_across_filters(
                all_imgs,
                abs_ref=abs_ref_path,
            )
            results.append(
                {
                    'instrument': 'all',
                    'filter': 'visit_cross_filter',
                    'status': 'ok' if visit_x.get('ok') else 'failed',
                    'harmonize': visit_x,
                }
            )
        except Exception as exc:
            log.error('Visit cross-filter harmonize failed: %s', exc)
            results.append(
                {
                    'instrument': 'all',
                    'filter': 'visit_cross_filter',
                    'status': 'failed',
                    'error': f'{type(exc).__name__}: {exc}',
                }
            )
    return results

def propagate_jhat_wcs_to_all_sci(
    aligned: str | Path,
    source: str | Path,
    *,
    min_shift_arcsec: float = 1e-4,
) -> dict[str, Any]:
    """
    Propagate JHAT's first-SCI sky tweak to every SCI extension.

    Upstream JHAT / TweakReg often updates only ``SCI,1`` on multi-chip HST
    MEFs (WFPC2 4-chip, WFC3/UVIS 2-chip, ACS/WFC 2-chip). AstroDrizzle then
    stacks mostly uncorrected chips, leaving cross-visit residuals of several
    tenths of an arcsecond.

    This measures the median sky translation on the first SCI that changed
    relative to *source*, then applies the same DeltaCRVAL to every SCI that still
    matches *source* (typically SCI2+). Already-updated chips are left as JHAT
    wrote them.

    Returns a stats dict (``dra_deg``, ``ddec_deg``, ``n_updated``, ...).
    """
    import numpy as np
    from astropy.wcs import WCS

    aligned_path = Path(aligned).expanduser().resolve()
    source_path = Path(source).expanduser().resolve()
    stats: dict[str, Any] = {
        'path': str(aligned_path),
        'source': str(source_path),
        'dra_deg': 0.0,
        'ddec_deg': 0.0,
        'dra_arcsec': 0.0,
        'ddec_arcsec': 0.0,
        'n_sci': 0,
        'n_updated': 0,
        'ref_sci_index': None,
    }
    if not aligned_path.is_file() or not source_path.is_file():
        raise FileNotFoundError(
            f'propagate_jhat_wcs_to_all_sci needs aligned and source FITS '
            f'(got {aligned_path}, {source_path})'
        )

    with as_datamodel(source_path).open(memmap=True) as src_hdul:
        src_idxs = _sci_hdu_indices(src_hdul)
        if len(src_idxs) < 2:
            stats['n_sci'] = len(src_idxs)
            return stats  # single-SCI: nothing to propagate

        with as_datamodel(aligned_path).open(mode='update', memmap=False) as aln_hdul:
            aln_idxs = _sci_hdu_indices(aln_hdul)
            stats['n_sci'] = len(aln_idxs)
            if len(aln_idxs) < 2:
                return stats

            # Pair by order among SCI HDUs.
            n_pair = min(len(src_idxs), len(aln_idxs))
            ref_i = None
            dra = ddec = 0.0
            unchanged: list[int] = []
            for k in range(n_pair):
                si, ai = src_idxs[k], aln_idxs[k]
                w0 = WCS(src_hdul[si].header, src_hdul, naxis=2)
                w1 = WCS(aln_hdul[ai].header, aln_hdul, naxis=2)
                shape = np.asarray(src_hdul[si].data).shape[-2:]
                d_ra, d_dec = measure_sci_sky_translation(w0, w1, shape)
                shift_as = float(
                    np.hypot(
                        d_ra * 3600.0 * np.cos(np.radians(float(src_hdul[si].header['CRVAL2']))),
                        d_dec * 3600.0,
                    )
                )
                if shift_as >= float(min_shift_arcsec) and ref_i is None:
                    ref_i = ai
                    dra, ddec = d_ra, d_dec
                elif shift_as < float(min_shift_arcsec):
                    unchanged.append(ai)

            if ref_i is None or not unchanged:
                # Either no JHAT tweak, or all chips already updated.
                stats['ref_sci_index'] = ref_i
                return stats

            n_upd = apply_sky_translation_to_sci(
                aln_hdul, dra, ddec, sci_indices=unchanged
            )
            aln_hdul[0].header['ST123WPR'] = (
                True,
                'st123: JHAT sky tweak propagated to all SCI',
            )
            dec0 = float(aln_hdul[aln_idxs[0]].header.get('CRVAL2', 0.0))
            dra_as = float(dra * 3600.0 * np.cos(np.radians(dec0)))
            ddec_as = float(ddec * 3600.0)
            aln_hdul[0].header['ST123DRA'] = (dra_as, '[arcsec] multi-SCI dRA cos(Dec)')
            aln_hdul[0].header['ST123DDE'] = (ddec_as, '[arcsec] multi-SCI dDec')
            aln_hdul[0].header['ST123NUP'] = (int(n_upd), 'SCI extensions updated')
            aln_hdul.flush()

            stats.update(
                {
                    'dra_deg': float(dra),
                    'ddec_deg': float(ddec),
                    'dra_arcsec': dra_as,
                    'ddec_arcsec': ddec_as,
                    'n_updated': int(n_upd),
                    'ref_sci_index': int(ref_i),
                }
            )
    log.info(
        'Propagated JHAT WCS to %d/%d SCI on %s (dRA=%+.3f" dDec=%+.3f")',
        stats['n_updated'],
        stats['n_sci'],
        aligned_path.name,
        stats['dra_arcsec'],
        stats['ddec_arcsec'],
    )
    return stats


def refine_hst_wcs_per_chip_from_refcat(
    aligned: str | Path,
    refcat: str | Path,
    *,
    search_radius_arcsec: float = 1.0,
    min_matches: int = 5,
    max_shift_arcsec: float = 2.0,
) -> dict[str, Any]:
    """
    Per-SCI CRPIX refine against a sky refcat (L3 phot / Gaia).

    For each SCI chip, centroid refcat sources on the detector and apply a
    sigma-clipped median (dx, dy) as a CRPIX shift. This places WFPC2 WF chips
    (and WFC3/ACS chip 2) on the same absolute frame as the reference, not just
    SCI1.
    """
    import numpy as np
    import pandas as pd
    from astropy.stats import sigma_clip
    from astropy.table import Table
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales

    from st123.stages.alignment.gaia_simple import _centroid

    aligned_path = Path(aligned).expanduser().resolve()
    ref_path = Path(refcat).expanduser().resolve()
    stats: dict[str, Any] = {
        'path': str(aligned_path),
        'n_sci': 0,
        'n_updated': 0,
        'n_copied': 0,
        'complete': False,
        'chips': [],
    }
    if not aligned_path.is_file() or not ref_path.is_file():
        return stats

    ref_df = pd.read_csv(ref_path, sep=r'\s+', engine='python')
    if 'ra' not in ref_df.columns or 'dec' not in ref_df.columns:
        return stats
    ref = Table.from_pandas(ref_df)
    if 'mag' in ref.colnames:
        ref.sort('mag')

    with as_datamodel(aligned_path).open(mode='update', memmap=False) as hdul:
        idxs = _sci_hdu_indices(hdul)
        stats['n_sci'] = len(idxs)
        for ai in idxs:
            data = np.asarray(hdul[ai].data, dtype=float)
            ny, nx = data.shape[-2:]
            bad = ~np.isfinite(data)
            w = WCS(hdul[ai].header, hdul, naxis=2)
            scale = float(np.nanmedian(proj_plane_pixel_scales(w)) * 3600.0)
            r_pix = float(search_radius_arcsec) / max(scale, 1e-6)
            dxs: list[float] = []
            dys: list[float] = []
            # Quiet vectorized sky->pixel; skip off-detector / divergent rows.
            n_ref = min(4000, len(ref))
            ras = np.asarray(ref['ra'][:n_ref], dtype=float)
            decs = np.asarray(ref['dec'][:n_ref], dtype=float)
            x_pred_all, y_pred_all = world_to_pixel_quiet(w, ras, decs)
            for x_pred, y_pred in zip(x_pred_all, y_pred_all):
                if not (np.isfinite(x_pred) and np.isfinite(y_pred)):
                    continue
                if (
                    x_pred < -r_pix
                    or x_pred > (nx - 1) + r_pix
                    or y_pred < -r_pix
                    or y_pred > (ny - 1) + r_pix
                ):
                    continue
                try:
                    x_meas, y_meas, _flux = _centroid(
                        data, bad, x0=float(x_pred), y0=float(y_pred), r_pix=r_pix
                    )
                except Exception:
                    continue
                dxs.append(float(x_meas - x_pred))
                dys.append(float(y_meas - y_pred))
            chip_stat = {
                'sci_index': int(ai),
                'n_match': len(dxs),
                'dx_pix': 0.0,
                'dy_pix': 0.0,
                'applied': False,
            }
            if len(dxs) < int(min_matches):
                stats['chips'].append(chip_stat)
                continue
            dx_a = np.asarray(dxs, dtype=float)
            dy_a = np.asarray(dys, dtype=float)
            rr = np.hypot(dx_a, dy_a)
            clipped = sigma_clip(rr, sigma=3.0, maxiters=5, masked=True)
            keep = ~np.asarray(getattr(clipped, 'mask', np.zeros_like(rr, dtype=bool)))
            if int(np.count_nonzero(keep)) < int(min_matches):
                stats['chips'].append(chip_stat)
                continue
            dx_m = float(np.median(dx_a[keep]))
            dy_m = float(np.median(dy_a[keep]))
            shift_as = float(np.hypot(dx_m, dy_m) * scale)
            chip_stat.update(
                {
                    'n_match': int(np.count_nonzero(keep)),
                    'dx_pix': dx_m,
                    'dy_pix': dy_m,
                    'shift_arcsec': shift_as,
                }
            )
            if shift_as > float(max_shift_arcsec):
                stats['chips'].append(chip_stat)
                continue
            if shift_as < 0.005:
                stats['chips'].append(chip_stat)
                continue
            hdr = hdul[ai].header
            hdr['CRPIX1'] = (
                float(hdr['CRPIX1']) + dx_m,
                'st123: per-SCI refine dx',
            )
            hdr['CRPIX2'] = (
                float(hdr['CRPIX2']) + dy_m,
                'st123: per-SCI refine dy',
            )
            chip_stat['applied'] = True
            stats['n_updated'] += 1
            stats['chips'].append(chip_stat)

        # Partial refine (e.g. 1/2 SCI) leaves chip-chip WCS inconsistent and
        # doubles PSFs in the drizzle. Copy the median CRPIX shift from chips
        # that matched onto chips that did not.
        applied = [c for c in stats['chips'] if c.get('applied')]
        pending = [c for c in stats['chips'] if not c.get('applied')]
        stats['n_copied'] = 0
        if applied and pending:
            dx_m = float(np.median([float(c['dx_pix']) for c in applied]))
            dy_m = float(np.median([float(c['dy_pix']) for c in applied]))
            ref_ai = int(applied[0]['sci_index'])
            scale0 = float(
                np.nanmedian(
                    proj_plane_pixel_scales(
                        WCS(hdul[ref_ai].header, hdul, naxis=2)
                    )
                )
                * 3600.0
            )
            shift_as = float(np.hypot(dx_m, dy_m) * max(scale0, 1e-6))
            if shift_as <= float(max_shift_arcsec):
                for chip_stat in pending:
                    ai = int(chip_stat['sci_index'])
                    hdr = hdul[ai].header
                    if shift_as >= 0.005:
                        hdr['CRPIX1'] = (
                            float(hdr['CRPIX1']) + dx_m,
                            'st123: per-SCI refine dx (sibling copy)',
                        )
                        hdr['CRPIX2'] = (
                            float(hdr['CRPIX2']) + dy_m,
                            'st123: per-SCI refine dy (sibling copy)',
                        )
                    chip_stat.update(
                        {
                            'dx_pix': dx_m,
                            'dy_pix': dy_m,
                            'shift_arcsec': shift_as,
                            'applied': True,
                            'copied_from_sibling': True,
                        }
                    )
                    stats['n_copied'] += 1
                    stats['n_updated'] += 1
                log.info(
                    'Per-chip refine %s: copied CRPIX shift to %d/%d SCI '
                    '(sibling median dx=%.3f dy=%.3f px)',
                    aligned_path.name,
                    stats['n_copied'],
                    stats['n_sci'],
                    dx_m,
                    dy_m,
                )

        stats['complete'] = (
            int(stats['n_sci']) > 0
            and int(stats['n_updated']) >= int(stats['n_sci'])
        )
        if stats['n_updated']:
            hdul[0].header['ST123CHP'] = (
                True,
                'st123: per-SCI CRPIX refine vs refcat',
            )
            hdul[0].header['ST123CNU'] = (
                int(stats['n_updated']),
                'SCI chips CRPIX-refined',
            )
            if stats['complete']:
                hdul[0].header['ST123CAF'] = (
                    True,
                    'st123: all SCI chips CRPIX-refined',
                )
            hdul.flush()
    if stats['n_updated']:
        log.info(
            'Per-chip refine %s: updated %d/%d SCI%s',
            aligned_path.name,
            stats['n_updated'],
            stats['n_sci'],
            '' if stats.get('complete') else ' (INCOMPLETE)',
        )
        if stats['n_sci'] >= 2 and not stats.get('complete'):
            log.warning(
                'Per-chip refine incomplete for %s (%d/%d SCI); '
                'mosaic L2 QA will reject this frame',
                aligned_path.name,
                stats['n_updated'],
                stats['n_sci'],
            )
    return stats


def heal_hst_partial_chip_refine(
    frames: Sequence[str | Path],
) -> dict[str, Any]:
    """
    Finish incomplete multi-SCI CRPIX refines left on disk.

    Older / interrupted per-chip refine can leave ``ST123CHP`` with
    ``ST123CNU < n_SCI`` (one UVIS chip shifted, sibling untouched). That
    doubles PSFs in AstroDrizzle and trips :func:`validate_hst_multi_sci_chip_refine`.

    Recover the applied (dx, dy) from each refined SCI vs its ERR CRPIX (ERR
    keeps the pre-refine reference pixel), copy onto unreined SCI, and mark
    ``ST123CAF``.
    """

    report: dict[str, Any] = {
        'ok': True,
        'n_frames': 0,
        'n_healed': 0,
        'n_failed': 0,
        'frames': [],
    }
    for raw in frames:
        path = Path(raw).expanduser().resolve()
        row: dict[str, Any] = {'path': str(path), 'healed': False}
        report['n_frames'] += 1
        if not path.is_file():
            row['error'] = 'missing'
            report['n_failed'] += 1
            report['ok'] = False
            report['frames'].append(row)
            continue
        try:
            with as_datamodel(path).open(mode='update', memmap=False) as hdul:
                idxs = _sci_hdu_indices(hdul)
                n_sci = len(idxs)
                row['n_sci'] = n_sci
                if n_sci < 2:
                    report['frames'].append(row)
                    continue
                chp = bool(hdul[0].header.get('ST123CHP'))
                n_ref = hdul[0].header.get('ST123CNU')
                n_ref_i = int(n_ref) if n_ref is not None else None
                row['n_refined_before'] = n_ref_i
                if not chp or n_ref_i is None or n_ref_i >= n_sci:
                    report['frames'].append(row)
                    continue

                refined: list[tuple[int, float, float]] = []
                pending: list[int] = []
                for ai in idxs:
                    hdr = hdul[ai].header
                    comment = str(hdr.comments['CRPIX1'] if 'CRPIX1' in hdr else '')
                    is_refined = 'per-SCI refine' in comment
                    if is_refined:
                        # Prefer ERR companion (same EXTVER) as pre-refine CRPIX.
                        dx = dy = None
                        ver = int(getattr(hdul[ai], 'ver', 1) or 1)
                        for j, hdu in enumerate(hdul):
                            if (
                                getattr(hdu, 'name', '') == 'ERR'
                                and int(getattr(hdu, 'ver', 1) or 1) == ver
                                and hdu.header.get('CRPIX1') is not None
                            ):
                                dx = float(hdr['CRPIX1']) - float(hdu.header['CRPIX1'])
                                dy = float(hdr['CRPIX2']) - float(hdu.header['CRPIX2'])
                                break
                        if dx is None:
                            # Fallback: treat pipeline-like integers as baseline.
                            dx = float(hdr['CRPIX1']) - round(float(hdr['CRPIX1']))
                            dy = float(hdr['CRPIX2']) - round(float(hdr['CRPIX2']))
                            # If already on integer CRPIX, use delta from 2048/1026-ish
                            # is unreliable; skip this chip as a shift donor.
                            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                                pending.append(ai)
                                continue
                        refined.append((ai, float(dx), float(dy)))
                    else:
                        pending.append(ai)

                if not refined or not pending:
                    row['error'] = (
                        f'cannot heal partial refine '
                        f'(refined={len(refined)} pending={len(pending)})'
                    )
                    report['n_failed'] += 1
                    report['ok'] = False
                    report['frames'].append(row)
                    continue

                dx_m = float(sum(t[1] for t in refined) / len(refined))
                dy_m = float(sum(t[2] for t in refined) / len(refined))
                for ai in pending:
                    hdr = hdul[ai].header
                    hdr['CRPIX1'] = (
                        float(hdr['CRPIX1']) + dx_m,
                        'st123: per-SCI refine dx (sibling heal)',
                    )
                    hdr['CRPIX2'] = (
                        float(hdr['CRPIX2']) + dy_m,
                        'st123: per-SCI refine dy (sibling heal)',
                    )
                hdul[0].header['ST123CHP'] = (
                    True,
                    'st123: per-SCI CRPIX refine vs refcat',
                )
                hdul[0].header['ST123CNU'] = (
                    n_sci,
                    'SCI chips CRPIX-refined',
                )
                hdul[0].header['ST123CAF'] = (
                    True,
                    'st123: all SCI chips CRPIX-refined',
                )
                hdul.flush()
                row['healed'] = True
                row['dx_pix'] = dx_m
                row['dy_pix'] = dy_m
                row['n_copied'] = len(pending)
                report['n_healed'] += 1
                log.info(
                    'Healed partial chip refine %s: copied dx=%.3f dy=%.3f px '
                    'to %d SCI (%d -> %d)',
                    path.name,
                    dx_m,
                    dy_m,
                    len(pending),
                    int(n_ref_i),
                    n_sci,
                )
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
            report['n_failed'] += 1
            report['ok'] = False
        report['frames'].append(row)
    return report


def validate_hst_multi_sci_chip_refine(
    frames: Sequence[str | Path],
) -> dict[str, Any]:
    """
    Pre-drizzle QA: multi-SCI frames must not retain a partial chip refine.

    Frames with ``ST123CHP`` set and ``ST123CNU < n_SCI`` fail (classic 1/2
    UVIS refine -> elongated PSFs). Frames with no chip-refine flag are allowed
    (global-only / single-SCI paths).
    """

    rows: list[dict[str, Any]] = []
    all_ok = True
    for raw in frames:
        path = Path(raw).expanduser().resolve()
        row: dict[str, Any] = {
            'path': str(path),
            'ok': True,
            'n_sci': 0,
            'n_refined': None,
        }
        if not path.is_file():
            row['ok'] = False
            row['error'] = 'missing'
            all_ok = False
            rows.append(row)
            continue
        try:
            with as_datamodel(path).open(memmap=True) as hdul:
                n_sci = len(_sci_hdu_indices(hdul))
                row['n_sci'] = int(n_sci)
                if n_sci < 2:
                    rows.append(row)
                    continue
                chp = bool(hdul[0].header.get('ST123CHP'))
                n_ref = hdul[0].header.get('ST123CNU')
                row['chip_refine'] = chp
                row['n_refined'] = int(n_ref) if n_ref is not None else None
                row['all_chips'] = bool(hdul[0].header.get('ST123CAF'))
                if chp and row['n_refined'] is not None and row['n_refined'] < n_sci:
                    row['ok'] = False
                    row['error'] = (
                        f'partial chip refine {row["n_refined"]}/{n_sci}'
                    )
                    all_ok = False
        except Exception as exc:
            row['ok'] = False
            row['error'] = f'{type(exc).__name__}: {exc}'
            all_ok = False
        rows.append(row)
    return {
        'ok': all_ok,
        'n_frames': len(rows),
        'n_failed': sum(1 for r in rows if not r.get('ok')),
        'frames': rows,
    }


def refine_hst_wcs_from_refcat(
    aligned: str | Path,
    refcat: str | Path,
    *,
    match_radius_arcsec: float = 1.0,
    max_residual_arcsec: float = 0.5,
) -> dict[str, Any]:
    """
    Optional residual CRVAL tweak from aligned-frame phot vs *refcat*.

    Uses ``{stem}.phot.txt`` beside *aligned* when present. Applies a median
    sky residual to **all** SCI extensions when |Delta| is between a noise floor
    and *max_residual_arcsec* (rejects gross mismatches).
    """
    import numpy as np
    import pandas as pd
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    import astropy.units as u

    aligned_path = Path(aligned).expanduser().resolve()
    ref_path = Path(refcat).expanduser().resolve()
    stats: dict[str, Any] = {
        'path': str(aligned_path),
        'n_match': 0,
        'applied': False,
        'dra_arcsec': 0.0,
        'ddec_arcsec': 0.0,
    }
    # JHAT phot next to product: iey902seq_jhat.fits -> iey902seq.phot.txt
    stem = aligned_path.name.replace('_jhat.fits', '').replace('.fits', '')
    phot_path = aligned_path.parent / f'{stem}.phot.txt'
    if not phot_path.is_file():
        # Also try stripping _flc/_c0m style roots already handled by stem
        return stats
    if not ref_path.is_file():
        return stats

    phot = pd.read_csv(phot_path, sep=r'\s+', engine='python')
    ref = pd.read_csv(ref_path, sep=r'\s+', engine='python')
    if 'x' not in phot.columns or 'y' not in phot.columns:
        return stats
    if 'ra' not in ref.columns or 'dec' not in ref.columns:
        return stats

    with as_datamodel(aligned_path).open(mode='update', memmap=False) as hdul:
        idxs = _sci_hdu_indices(hdul)
        if not idxs:
            return stats
        # JHAT phot x,y are in the primary/first-SCI (or do_driz) frame - not
        # per-chip. Project only through SCI1, then apply residual to all SCI.
        ai0 = idxs[0]
        w = WCS(hdul[ai0].header, hdul, naxis=2)
        ny, nx = np.asarray(hdul[ai0].data).shape[-2:]
        x = np.asarray(phot['x'], dtype=float)
        y = np.asarray(phot['y'], dtype=float)
        on = (x >= 0) & (x < nx) & (y >= 0) & (y < ny) & np.isfinite(x) & np.isfinite(y)
        if int(np.count_nonzero(on)) < 5:
            return stats
        ra_img, dec_img = w.pixel_to_world_values(x[on], y[on])
        ra_img = np.asarray(ra_img, dtype=float)
        dec_img = np.asarray(dec_img, dtype=float)
        c_img = SkyCoord(ra_img * u.deg, dec_img * u.deg)
        c_ref = SkyCoord(
            np.asarray(ref['ra'], dtype=float) * u.deg,
            np.asarray(ref['dec'], dtype=float) * u.deg,
        )
        idx, sep, _ = c_img.match_to_catalog_sky(c_ref)
        good = sep.arcsec < float(match_radius_arcsec)
        stats['n_match'] = int(good.sum())
        if stats['n_match'] < 5:
            return stats
        dra = (
            (c_ref.ra[idx] - c_img.ra).to(u.deg).value
        )
        ddec = (c_ref.dec[idx] - c_img.dec).to(u.deg).value
        # Image -> ref residual; apply to CRVAL so image moves onto ref.
        dra_m = float(np.median(dra[good]))
        ddec_m = float(np.median(ddec[good]))
        dec0 = float(np.median(dec_img[good]))
        dra_as = dra_m * 3600.0 * float(np.cos(np.radians(dec0)))
        ddec_as = ddec_m * 3600.0
        stats['dra_arcsec'] = dra_as
        stats['ddec_arcsec'] = ddec_as
        shift = float(np.hypot(dra_as, ddec_as))
        if shift < 0.01 or shift > float(max_residual_arcsec):
            return stats
        apply_sky_translation_to_sci(hdul, dra_m, ddec_m, sci_indices=idxs)
        hdul[0].header['ST123REF'] = (
            True,
            'st123: residual CRVAL refine vs refcat',
        )
        hdul[0].header['ST123RAS'] = (dra_as, '[arcsec] refine dRA cos(Dec)')
        hdul[0].header['ST123RDE'] = (ddec_as, '[arcsec] refine dDec')
        hdul.flush()
        stats['applied'] = True
    if stats['applied']:
        log.info(
            'Refined %s vs refcat: dRA=%+.3f" dDec=%+.3f" (n=%d)',
            aligned_path.name,
            stats['dra_arcsec'],
            stats['ddec_arcsec'],
            stats['n_match'],
        )
    return stats


def _post_jhat_refcat_refine(
    recovered: Path,
    photfilename: str | Path | None,
    *,
    refine_refcat: bool = True,
    refine_per_chip: bool = True,
) -> dict[str, Any]:
    """
    Post-JHAT absolute refine vs *photfilename*.

    Prefers per-chip CRPIX refine; runs global CRVAL refine only when per-chip
    is disabled or updates no SCI (fallback for sparse chips / missing phot).
    """
    stats: dict[str, Any] = {
        'per_chip_updated': 0,
        'global_ran': False,
        'global_applied': False,
    }
    if photfilename is None:
        return stats

    n_chip = 0
    if refine_per_chip:
        try:
            chip_stats = refine_hst_wcs_per_chip_from_refcat(
                recovered, photfilename
            )
            n_chip = int(chip_stats.get('n_updated') or 0)
        except Exception as exc:
            log.warning(
                'per-chip WCS refine failed for %s: %s', recovered.name, exc
            )
            n_chip = 0
        stats['per_chip_updated'] = n_chip

    run_global = refine_refcat and (
        not refine_per_chip or n_chip == 0
    )
    if run_global:
        try:
            gstats = refine_hst_wcs_from_refcat(recovered, photfilename)
            stats['global_ran'] = True
            stats['global_applied'] = bool(gstats.get('applied'))
        except Exception as exc:
            log.warning(
                'refcat WCS refine failed for %s: %s', recovered.name, exc
            )
    return stats


def align_hst_image(
    image: str | Path,
    outdir: str | Path,
    *,
    gaia: bool = True,
    photfilename: str | None = None,
    verbose: bool = False,
    jhat_params: dict | None = None,
    propagate_multi_sci: bool = True,
    refine_refcat: bool = True,
    refine_per_chip: bool = True,
) -> Path:
    """
    Align one HST frame with JHAT (``telescope='hst'``).

    Applies the WFPC2 patch and pandas ``delim_whitespace`` compat, then runs
    ``st_wcs_align().run_all`` under :func:`capture_output`.

    After JHAT:
    1. Propagate the first-SCI sky tweak to all SCI extensions (WFPC2 / WFC3 /
       ACS multi-chip MEFs).
    2. Optional per-SCI CRPIX refine by centroiding *photfilename* sources on
       each chip (puts WF / ACS chips on the reference frame).
    3. Global CRVAL refine vs *photfilename* only as a fallback when per-chip
       refine is disabled or updates no SCI (avoids a redundant second absolute
       tie to the same catalog).

    Returns
    -------
    pathlib.Path
        Path to the aligned ``*_jhat.fits`` product.
    """
    try:
        import warnings

        # JHAT still imports deprecated tweakwcs.tpwcs; harmless at runtime.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r".*tweakwcs\.tpwcs.*",
            )
            from jhat import st_wcs_align
    except ImportError as exc:
        raise ImportError(
            'align_hst_image requires the jhat package '
            '(install vendored extdeps/jhat or set PYTHONPATH).'
        ) from exc

    install_jhat_pandas_read_table_compat()
    ensure_wfpc2_jhat_patch()

    image_path = Path(image).expanduser().resolve()
    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    params = dict(jhat_params or {})
    outroot = str(out)
    wcs_align = st_wcs_align()
    # Instance-level histocut caps (not formal run_all kwargs) - set before run.
    for key in (
        'rough_cut_px_min',
        'rough_cut_px_max',
        'd_rotated_Nsigma',
        'gaussian_sigma_px',
        'binsize_px',
    ):
        if key in params:
            setattr(wcs_align, key, params.pop(key))
    if gaia:
        # Prefer the shared Vizier field cache over JHAT's ESA TAP Gaia query.
        refcat = 'Gaia'
        try:
            from st123.stages.alignment.gaia_catalog import (
                default_gaia_cache_dir,
                write_gaia_refcat,
            )

            cache_dir = default_gaia_cache_dir(image_path)
            if cache_dir is not None:
                local_ref = out / f'{image_path.stem}_gaia_refcat.txt'
                write_gaia_refcat(
                    image_path,
                    local_ref,
                    telescope='hst',
                    cache_dir=cache_dir,
                )
                if local_ref.is_file() and local_ref.stat().st_size > 20:
                    refcat = str(local_ref)
                    params.setdefault('refcat_racol', 'ra')
                    params.setdefault('refcat_deccol', 'dec')
                    params.setdefault('refcat_magcol', 'mag')
                    log.info(
                        'JHAT using local Gaia refcat %s (Vizier cache)',
                        local_ref.name,
                    )
        except Exception as exc:
            log.warning('Local Gaia refcat unavailable (%s); JHAT will use TAP', exc)
        run_kwargs = dict(
            outrootdir=outroot,
            telescope='hst',
            refcatname=refcat,
            pmflag=True,
            use_dq=False,
            verbose=verbose,
            **params,
        )
    else:
        if photfilename is None:
            raise ValueError('photfilename is required when gaia=False')
        # File-based refcats need real column names (not JHAT's 'auto' default).
        params.setdefault('refcat_racol', 'ra')
        params.setdefault('refcat_deccol', 'dec')
        params.setdefault('refcat_magcol', 'mag')
        run_kwargs = dict(
            outrootdir=outroot,
            telescope='hst',
            refcatname=str(photfilename),
            use_dq=False,
            verbose=verbose,
            **params,
        )

    mode = 'Gaia' if gaia else f'refcat={Path(photfilename).name}'
    t0 = time.perf_counter()
    log.info('JHAT run_all starting for %s (%s)', image_path.name, mode)
    run_exc: Exception | None = None
    try:
        with capture_output():
            wcs_align.run_all(str(image_path), **run_kwargs)
    except Exception as exc:
        run_exc = exc

    recovered = _recover_jhat_product(image_path, out)
    log.info(
        'JHAT run_all finished for %s in %.1fs (product=%s%s)',
        image_path.name,
        time.perf_counter() - t0,
        recovered.name if recovered is not None else 'none',
        f'; error={run_exc}' if run_exc is not None else '',
    )
    if recovered is not None:
        if propagate_multi_sci:
            try:
                propagate_jhat_wcs_to_all_sci(recovered, image_path)
            except Exception as exc:
                log.warning(
                    'multi-SCI WCS propagation failed for %s: %s',
                    recovered.name,
                    exc,
                )
        _post_jhat_refcat_refine(
            recovered,
            photfilename,
            refine_refcat=refine_refcat,
            refine_per_chip=refine_per_chip,
        )
        return recovered

    if run_exc is not None:
        raise run_exc
    raise FileNotFoundError(
        f'JHAT finished but aligned product not found under {out} '
        f'(expected {_jhat_hst_output_path(image_path, out).name})'
    )


def find_jhat_phot(
    image: Path,
    jhat_outdir: Path,
    *,
    search_dirs: list[Path] | None = None,
) -> Path | None:
    """Locate JHAT ``*.phot.txt`` written for *image*.

    Searches *jhat_outdir* and any extra *search_dirs* (e.g. parent ``jhat/``
    when L3 products were written beside science frames).
    """
    name = image.name
    stems: list[str] = []
    for suf in (
        '_jhat.fits',
        '_flc.fits',
        '_flt.fits',
        '_c0m.fits',
        '_drc.fits',
        '_drz.fits',
        '_drw.fits',
        '.fits',
    ):
        if name.endswith(suf):
            stems.append(name[: -len(suf)])
    stems.append(image.stem)
    # coadd_wfc3_f625w_drc -> also try coadd_wfc3_f625w
    extra: list[str] = []
    for stem in stems:
        for tok in ('_drc', '_drz', '_drw', '_jhat'):
            if stem.endswith(tok):
                extra.append(stem[: -len(tok)])
    stems.extend(extra)

    dirs: list[Path] = [Path(jhat_outdir)]
    if search_dirs:
        dirs.extend(Path(d) for d in search_dirs)
    # Also check the parent of outdir (common when L3 phot landed in jhat/).
    parent = Path(jhat_outdir).parent
    if parent not in dirs:
        dirs.append(parent)

    uniq_stems = []
    seen_stem: set[str] = set()
    for stem in stems:
        if not stem or stem in seen_stem:
            continue
        seen_stem.add(stem)
        uniq_stems.append(stem)

    seen_dir: set[str] = set()
    for directory in dirs:
        try:
            dkey = str(directory.resolve())
        except Exception:
            dkey = str(directory)
        if dkey in seen_dir:
            continue
        seen_dir.add(dkey)
        if not directory.is_dir():
            continue
        for stem in uniq_stems:
            for cand in (
                directory / f'{stem}.phot.txt',
                directory / f'{stem}_jhat.phot.txt',
            ):
                if cand.is_file() and cand.stat().st_size > 0:
                    return cand.resolve()
            for m in sorted(directory.glob(f'{stem}*.phot.txt')):
                if m.is_file() and m.stat().st_size > 0:
                    return m.resolve()
    return None


def _parallel_worker_budget(n_jobs: int, workers: int) -> int:
    """Clamp parallel worker count to ``[1, n_jobs]``."""
    return max(1, min(int(n_jobs), max(1, int(workers))))


def _hst_jhat_frame_worker(job: dict[str, Any]) -> dict[str, Any]:
    """
    Align one HST frame (module-level for :class:`ProcessPoolExecutor`).

    Uses ``spawn``-safe imports inside the worker. Returns the same status
    dict shape as the serial path in :func:`align_hst_raw_dir`.
    """
    # Spawn workers have no StreamHandler; attach the shared log file so
    # per-frame progress is visible (parent also logs on future completion).
    configure_worker_logging()
    t0 = time.perf_counter()
    frame = Path(job['frame'])
    out = Path(job['outdir'])
    idx = int(job['index'])
    n_frames = int(job['n_frames'])
    ref_desc = str(job.get('ref_desc') or '')
    entry: dict[str, Any] = {
        'path': str(frame),
        'status': 'pending',
        'error': None,
        'outpath': None,
    }
    try:
        from st123.utils.helpers import get_filter as _get_filter_for_nb

        frame_filt = ''
        try:
            frame_filt = _get_filter_for_nb(frame)
        except Exception:
            frame_filt = ''
        narrow = is_hst_narrowband_filter(frame_filt)
        frame_params = dict(job.get('jhat_params') or {})
        if narrow:
            merged = dict(HST_JHAT_NARROWBAND_PARAMS)
            merged.update(frame_params)
            frame_params = merged
            log.info(
                'HST JHAT [%d/%d] starting %s (%s; narrowband %s, relaxed params)',
                idx,
                n_frames,
                frame.name,
                ref_desc,
                frame_filt or '?',
            )
        else:
            log.info(
                'HST JHAT [%d/%d] starting %s (%s)',
                idx,
                n_frames,
                frame.name,
                ref_desc,
            )
        outpath = align_hst_image(
            frame,
            out,
            gaia=bool(job.get('use_gaia')),
            photfilename=job.get('photfilename'),
            verbose=bool(job.get('verbose')),
            jhat_params=frame_params or None,
        )
        entry['status'] = 'ok'
        entry['outpath'] = str(outpath)
        if narrow:
            entry['align_mode'] = 'JHAT'
        log.info(
            'HST JHAT [%d/%d] ok %s -> %s (%.1fs)',
            idx,
            n_frames,
            frame.name,
            Path(outpath).name,
            time.perf_counter() - t0,
        )
    except Exception as exc:
        entry['status'] = 'failed'
        entry['error'] = f'{type(exc).__name__}: {exc}'
        err = str(exc)
        soft = (
            'initial cut' in err.lower()
            or 'objects pass' in err.lower()
            or type(exc).__name__ == 'KeyError'
        )
        log_fn = log.warning if soft else log.error
        log_fn(
            'HST JHAT [%d/%d] failed %s after %.1fs: %s',
            idx,
            n_frames,
            frame.name,
            time.perf_counter() - t0,
            exc,
        )
    return entry


def align_hst_raw_dir(
    raw_dir: str | Path,
    jhat_outdir: str | Path,
    *,
    patterns: tuple[str, ...] = ('*flc.fits', '*flt.fits', '*c0m.fits'),
    soft_fail: bool = True,
    gaia: bool = True,
    photfilename: str | Path | None = None,
    verbose: bool = False,
    jhat_params: dict | None = None,
    instruments: Sequence[str] | None = None,
    workers: int = 1,
) -> list[dict]:
    """
    Align all matching science frames under *raw_dir*.

    Skips ``*c1m.fits`` (DQ companions). When *photfilename* is set, frames are
    aligned to that catalog (``gaia`` is ignored). Returns per-file status dicts
    with keys ``path``, ``status``, ``error``, ``outpath``.

    Parameters
    ----------
    instruments : sequence of str or None, optional
        If set, only align frames whose ``INSTRUME`` (via
        :func:`st123.utils.helpers.get_instrument`) matches one of these
        names (case-insensitive; e.g. ``ACS``, ``WFC3`` to skip WFPC2).
    workers : int, optional
        Parallel JHAT worker processes (default 1 = serial). Values ``>1`` use
        a ``spawn`` :class:`~concurrent.futures.ProcessPoolExecutor`.
    """
    from st123.utils.helpers import get_filter as _get_filter_for_nb
    from st123.utils.helpers import get_instrument

    raw = Path(raw_dir).expanduser().resolve()
    out = Path(jhat_outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref_phot = Path(photfilename).expanduser().resolve() if photfilename else None
    use_gaia = bool(gaia) and ref_phot is None
    allow = None
    if instruments:
        allow = {str(i).split('_')[0].strip().lower() for i in instruments if str(i).strip()}

    seen: set[str] = set()
    frames: list[Path] = []
    for pat in patterns:
        for path in sorted(raw.glob(pat)):
            name = path.name.lower()
            if name.endswith('c1m.fits'):
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            frames.append(path)

    # Also catch any stray absolute globs if raw_dir was passed oddly.
    if not frames:
        for pat in patterns:
            for path in sorted(Path(p) for p in glob.glob(str(raw / pat))):
                if path.name.lower().endswith('c1m.fits'):
                    continue
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                frames.append(path)

    if allow:
        kept: list[Path] = []
        for path in frames:
            try:
                inst = get_instrument(path).split('_')[0].lower()
            except Exception:
                continue
            if inst in allow:
                kept.append(path)
        log.info(
            'HST instrument filter %s: %d -> %d frame(s)',
            sorted(allow),
            len(frames),
            len(kept),
        )
        frames = kept

    from st123.datamodels.hst import filter_paths_for_stage

    frames = filter_paths_for_stage(frames, stage='align')

    n_frames = len(frames)
    ref_desc = (
        f'refcat={ref_phot.name}'
        if ref_phot is not None
        else ('Gaia' if use_gaia else 'no-ref')
    )
    n_workers = _parallel_worker_budget(n_frames, workers) if n_frames else 1
    log.info(
        'HST JHAT batch: %d frame(s) under %s -> %s (%s; workers=%d)',
        n_frames,
        raw,
        out,
        ref_desc,
        n_workers,
    )

    jobs: list[dict[str, Any]] = []
    for i, frame in enumerate(frames, start=1):
        jobs.append(
            {
                'frame': str(frame.resolve()),
                'outdir': str(out),
                'index': i,
                'n_frames': n_frames,
                'ref_desc': ref_desc,
                'use_gaia': use_gaia,
                'photfilename': str(ref_phot) if ref_phot is not None else None,
                'verbose': bool(verbose),
                'jhat_params': dict(jhat_params or {}),
            }
        )

    results: list[dict] = []
    if n_workers <= 1 or n_frames <= 1:
        for job in jobs:
            entry = _hst_jhat_frame_worker(job)
            results.append(entry)
            if entry.get('status') == 'failed' and not soft_fail:
                raise RuntimeError(entry.get('error') or 'JHAT failed')
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # spawn avoids fork+OpenMP/BLAS deadlocks after heavy scientific imports.
        ctx = mp.get_context('spawn')
        ordered: list[dict | None] = [None] * len(jobs)
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
        ) as pool:
            future_map = {
                pool.submit(_hst_jhat_frame_worker, job): i
                for i, job in enumerate(jobs)
            }
            for fut in as_completed(future_map):
                idx = future_map[fut]
                try:
                    ordered[idx] = fut.result()
                except Exception as exc:
                    frame = Path(jobs[idx]['frame'])
                    ordered[idx] = {
                        'path': str(frame),
                        'status': 'failed',
                        'error': f'{type(exc).__name__}: {exc}',
                        'outpath': None,
                    }
                    log.error(
                        'HST JHAT [%d/%d] worker crashed %s: %s',
                        jobs[idx]['index'],
                        n_frames,
                        frame.name,
                        exc,
                    )
                    continue
                entry = ordered[idx] or {}
                done = sum(1 for r in ordered if r is not None)
                if entry.get('status') == 'ok':
                    log.info(
                        'HST JHAT [%d/%d] ok %s (%d/%d done)',
                        jobs[idx]['index'],
                        n_frames,
                        Path(jobs[idx]['frame']).name,
                        done,
                        n_frames,
                    )
                else:
                    log.error(
                        'HST JHAT [%d/%d] failed %s (%d/%d done): %s',
                        jobs[idx]['index'],
                        n_frames,
                        Path(jobs[idx]['frame']).name,
                        done,
                        n_frames,
                        entry.get('error') or 'unknown',
                    )
        results = [r for r in ordered if r is not None]
        if not soft_fail:
            for entry in results:
                if entry.get('status') == 'failed':
                    raise RuntimeError(entry.get('error') or 'JHAT failed')

    # Narrowband mitigation: HST_REL -> L3 iterative refine -> group retie,
    # else PIPELINE copy. Prefer the batch photfilename, then l3_ref/.
    failed_nb = 0
    for r in results:
        if r.get('status') == 'ok' and r.get('outpath'):
            continue
        raw_p = Path(r.get('path') or '')
        if not raw_p.is_file():
            continue
        try:
            if is_hst_narrowband_filter(_get_filter_for_nb(raw_p)):
                failed_nb += 1
        except Exception:
            continue
    if failed_nb:
        nb_refcat = ref_phot if ref_phot is not None and ref_phot.is_file() else None
        if nb_refcat is None:
            nb_refcat = find_hst_l3_refcat(out)
        nb_abs = find_hst_abs_ref_image(out)
        log.info(
            'HST narrowband recovery: %d failed narrowband frame(s) -> '
            'HST_REL / L3 refine / PIPELINE (refcat=%s, abs_ref=%s)',
            failed_nb,
            nb_refcat.name if nb_refcat else 'none',
            nb_abs.name if nb_abs else 'none',
        )
        recover_failed_hst_narrowbands(
            results,
            out,
            pipeline_fallback=True,
            refcat=nb_refcat,
            abs_ref=nb_abs,
        )

    # Relative harmonize within each filter group so coadds are not ghosted
    # by inconsistent per-exposure JHAT solutions.
    n_ok = sum(1 for r in results if r.get('status') == 'ok' and r.get('outpath'))
    if n_ok >= 2:
        log.info(
            '[6/6] within-filter WCS harmonize on %d JHAT product(s) under %s',
            n_ok,
            out,
        )
        try:
            harm_rows = harmonize_hst_jhat_dir(out)
            for row in harm_rows:
                status = row.get('status')
                if status == 'ok':
                    continue
                msg = row.get('error') or (
                    (row.get('harmonize') or {}).get('method') or 'harmonize failed'
                )
                if status == 'skipped':
                    log.warning(
                        'Post-align group harmonize skipped for %s/%s: %s',
                        row.get('instrument'),
                        row.get('filter'),
                        msg,
                    )
                    continue
                log.error(
                    'Post-align group harmonize failed for %s/%s: %s',
                    row.get('instrument'),
                    row.get('filter'),
                    msg,
                )
                if not soft_fail:
                    raise RuntimeError(msg)
        except Exception as exc:
            log.error('Post-align group harmonize failed: %s', exc)
            if not soft_fail:
                raise

    # Absolute residual polish + JWST-style quality stamps / frame_qa.json.
    try:
        finalize_hst_jhat_dir_quality(
            out,
            results,
            refcat=ref_phot if ref_phot is not None else find_hst_l3_refcat(out),
            abs_ref=find_hst_abs_ref_image(out),
        )
    except Exception as exc:
        log.warning('HST frame quality finalize failed under %s: %s', out, exc)
    return results


def write_hst_alignment_summary(
    results: list[dict],
    summary_path: str | Path,
) -> Path:
    """Write ``alignment_summary.json`` (frame_qa written by finalize)."""
    path = Path(summary_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = sum(1 for r in results if r.get('status') == 'ok')
    n_fail = sum(1 for r in results if r.get('status') == 'failed')
    payload = {
        'n_total': len(results),
        'n_ok': n_ok,
        'n_failed': n_fail,
        'results': results,
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + '\n')
    # Ensure ASCII summary exists even if finalize already ran.
    outdir = path.parent
    if not (outdir / 'frame_qa.json').is_file():
        try:
            finalize_hst_jhat_dir_quality(outdir, results)
        except Exception as exc:
            log.warning('Could not finalize HST frame_qa under %s: %s', outdir, exc)
    elif not (outdir / 'alignment_summary.txt').is_file():
        rows = []
        for r in results:
            q = r.get('quality') or {}
            rows.append(
                {
                    'path': Path(r.get('outpath') or r.get('path') or '').name,
                    'filter': 'NA',
                    'status': r.get('status'),
                    'n_calibrators': q.get('n_calibrators'),
                    'dispersion_mas': q.get('dispersion_mas'),
                    'internal_max_delta_mas': q.get('internal_max_delta_mas'),
                    'align_mode': q.get('align_mode') or r.get('align_mode') or 'JHAT',
                    'algnref': q.get('algnref'),
                    'aligned_to': q.get('aligned_to'),
                }
            )
        try:
            write_alignment_summary_table(rows, outdir / 'alignment_summary.txt')
        except Exception:
            pass
    return path


def stamp_hst_jhat_quality(
    jhat_path: str | Path,
    *,
    refcat: str | Path | None = None,
    abs_ref: str | Path | None = None,
    align_mode: str = 'JHAT',
    internal_max_arcsec: float | None = None,
    match_radius_arcsec: float = 0.5,
) -> dict[str, Any]:
    """
    Measure abs catalog residual and stamp JWST-style quality headers.

    Prefers *refcat* (L3 phot); falls back to abs_ref image phot when needed.
    """
    jhat = Path(jhat_path).expanduser().resolve()
    report: dict[str, Any] = {
        'path': str(jhat),
        'ok': False,
        'align_mode': align_mode,
        'n_calibrators': None,
        'dispersion_mas': None,
        'internal_max_delta_mas': (
            float(internal_max_arcsec) * 1000.0
            if internal_max_arcsec is not None
            else None
        ),
    }
    if not jhat.is_file():
        report['error'] = 'missing jhat'
        return report

    ref_path = Path(refcat).expanduser().resolve() if refcat else None
    if ref_path is None or not ref_path.is_file():
        # Try sibling phot of abs_ref / l3_ref.
        for cand in (
            jhat.parent / 'l3_ref',
            jhat.parent,
        ):
            if not cand.is_dir():
                continue
            phots = sorted(cand.glob('coadd_*_drc.phot.txt')) + sorted(
                cand.glob('coadd_*_drz.phot.txt')
            )
            if phots:
                ref_path = phots[0]
                break
    aligned_to = str(ref_path) if ref_path and ref_path.is_file() else (
        str(abs_ref) if abs_ref else 'NA'
    )
    original_ref = str(abs_ref) if abs_ref else aligned_to

    residual = {
        'ok': False,
        'n_match': 0,
        'residual_arcsec': None,
        'residual_mean_arcsec': None,
        'residual_std_arcsec': None,
    }
    if ref_path is not None and ref_path.is_file():
        residual = measure_hst_narrowband_residual_vs_refcat(
            jhat,
            ref_path,
            match_radius_arcsec=float(match_radius_arcsec),
        )

    mean_as = residual.get('residual_mean_arcsec')
    med_as = residual.get('residual_arcsec')
    std_as = residual.get('residual_std_arcsec')
    n_cal = int(residual.get('n_match') or 0) if residual.get('ok') else None
    if med_as is not None:
        report['dispersion_mas'] = float(med_as) * 1000.0
    report['n_calibrators'] = n_cal
    report['ok'] = bool(residual.get('ok')) and abs_ok_local(
        med_as, n_calibrators=n_cal
    )
    report['algnref'] = original_ref
    report['aligned_to'] = aligned_to
    report['residual'] = residual

    stamp_quality_headers(
        jhat,
        align_mode=align_mode,
        original_ref=original_ref,
        aligned_to=aligned_to,
        abs_mean_arcsec=float(mean_as) if mean_as is not None else None,
        abs_median_arcsec=float(med_as) if med_as is not None else None,
        abs_std_arcsec=float(std_as) if std_as is not None else None,
        n_calibrators=n_cal,
        internal_max_arcsec=internal_max_arcsec,
        catalog_basename=ref_path.name if ref_path else None,
    )
    if med_as is not None:
        log.info(
            'Final dispersion %s: mean=%.1f mas median=%.1f mas n_cal=%s '
            '(internal max=%.1f mas)',
            jhat.name,
            1000.0 * float(mean_as or med_as),
            1000.0 * float(med_as),
            n_cal if n_cal is not None else 'NA',
            1000.0 * float(internal_max_arcsec or 0.0),
        )
    return report


def abs_ok_local(
    residual_arcsec: float | None, *, n_calibrators: int | None = None
) -> bool:
    """Local wrapper so stamp helpers need not import frame_qa.abs_ok by name."""
    from st123.stages.alignment.frame_qa import abs_ok as _abs_ok

    return _abs_ok(residual_arcsec, n_calibrators=n_calibrators)


def polish_hst_jhat_abs_residual(
    jhat_path: str | Path,
    refcat: str | Path,
    *,
    tol_arcsec: float = FRAME_ABS_RETIE_TOL_ARCSEC,
    max_apply_arcsec: float = FRAME_ABS_RETIE_MAX_APPLY_ARCSEC,
    min_matches: int = FRAME_MIN_MATCH_HEALTHY,
) -> dict[str, Any]:
    """
    Catalog residual polish onto *refcat* when residual is in (tol, max_apply].

    Mirrors JWST hub abs retie: small coherent CRVAL nudge then re-measure.
    """

    jhat = Path(jhat_path).expanduser().resolve()
    ref = Path(refcat).expanduser().resolve()
    report: dict[str, Any] = {
        'path': str(jhat),
        'applied': False,
        'ok': False,
    }
    if not jhat.is_file() or not ref.is_file():
        report['error'] = 'missing jhat/refcat'
        return report
    pre = measure_hst_narrowband_residual_vs_refcat(jhat, ref, match_radius_arcsec=0.5)
    report['pre'] = pre
    if not pre.get('ok'):
        return report
    n = int(pre.get('n_match') or 0)
    shift = float(pre.get('abs_arcsec') or 0.0)
    report['ok'] = shift <= coherent_tol_arcsec(n)
    if n < int(min_matches):
        report['skipped_sparse'] = True
        return report
    if shift <= float(tol_arcsec) or shift > float(max_apply_arcsec):
        return report
    dra_deg = float(pre['dra_deg'])
    ddec_deg = float(pre['ddec_deg'])
    with as_datamodel(jhat).open(mode='update', memmap=False) as hdul:
        apply_sky_translation_to_sci(
            hdul,
            dra_deg,
            ddec_deg,
            comment='st123: abs residual polish vs refcat',
        )
        hdul[0].header['ST123VAB'] = (True, 'st123: abs residual polish')
        hdul[0].header['ALGNMODE'] = ('VISIT_ABS', 'abs residual polish')
        hdul.flush()
    report['applied'] = True
    post = measure_hst_narrowband_residual_vs_refcat(jhat, ref, match_radius_arcsec=0.5)
    report['post'] = post
    if post.get('ok'):
        report['ok'] = float(post['residual_arcsec']) <= coherent_tol_arcsec(
            int(post.get('n_match') or 0)
        )
    log.info(
        'Abs residual polish %s: %.1f -> %.1f mas (n=%s)',
        jhat.name,
        1000.0 * shift,
        1000.0 * float((post.get('residual_arcsec') or shift)),
        post.get('n_match') or n,
    )
    return report


def finalize_hst_jhat_dir_quality(
    jhat_dir: str | Path,
    results: list[dict],
    *,
    refcat: str | Path | None = None,
    abs_ref: str | Path | None = None,
) -> dict[str, Any]:
    """
    Post-harmonize abs polish + stamp quality headers + write frame_qa.

    Returns the aggregated ``frame_qa`` report.
    """
    from collections import defaultdict

    from st123.utils.helpers import get_filter, get_instrument

    out = Path(jhat_dir).expanduser().resolve()
    ref_phot = Path(refcat).expanduser().resolve() if refcat else find_hst_l3_refcat(out)
    hub = Path(abs_ref).expanduser().resolve() if abs_ref else find_hst_abs_ref_image(out)

    ok_paths: list[Path] = []
    for r in results:
        if r.get('status') == 'ok' and r.get('outpath'):
            p = Path(r['outpath'])
            if p.is_file():
                ok_paths.append(p.resolve())

    # Internal max |Delta| per instrument+filter group.
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for p in ok_paths:
        try:
            inst = get_instrument(p).split('_')[0].lower()
            filt = str(get_filter(p)).lower()
        except Exception:
            inst, filt = 'hst', 'unknown'
        groups[(inst, filt)].append(p)

    internal_by_path: dict[Path, float] = {}
    for (_inst, _filt), paths in groups.items():
        if len(paths) < 2:
            for p in paths:
                internal_by_path[p] = 0.0
            continue
        qa = validate_hst_group_internal_alignment(paths)
        max_as = float(qa.get('max_abs_arcsec') or 0.0)
        for p in paths:
            internal_by_path[p] = max_as

    # Optional catalog polish then stamp.
    frame_rows: list[dict[str, Any]] = []
    if ref_phot is not None and ref_phot.is_file():
        for p in ok_paths:
            polish_hst_jhat_abs_residual(p, ref_phot)

    for r in results:
        p = Path(r.get('outpath') or '')
        if r.get('status') != 'ok' or not p.is_file():
            frame_rows.append(
                {
                    'path': str(r.get('path') or p),
                    'filter': 'NA',
                    'status': str(r.get('status') or 'failed'),
                    'n_calibrators': 'NA',
                    'dispersion_mas': 'NA',
                    'internal_max_delta_mas': 'NA',
                    'align_mode': 'FAILED',
                    'algnref': 'NA',
                    'aligned_to': 'NA',
                    'mission': 'hst',
                }
            )
            continue
        mode = str(r.get('align_mode') or 'JHAT')
        stamped = stamp_hst_jhat_quality(
            p,
            refcat=ref_phot,
            abs_ref=hub,
            align_mode=mode,
            internal_max_arcsec=internal_by_path.get(p.resolve()),
        )
        r['quality'] = stamped
        try:
            filt = str(get_filter(p)).lower()
        except Exception:
            filt = 'NA'
        frame_rows.append(
            {
                'path': p.name,
                'filter': filt,
                'status': 'ok',
                'n_calibrators': stamped.get('n_calibrators'),
                'dispersion_mas': stamped.get('dispersion_mas'),
                'internal_max_delta_mas': stamped.get('internal_max_delta_mas'),
                'align_mode': stamped.get('align_mode') or mode,
                'algnref': stamped.get('algnref'),
                'aligned_to': stamped.get('aligned_to'),
                'mission': 'hst',
            }
        )

    abs_vals = [
        float(r['dispersion_mas'])
        for r in frame_rows
        if isinstance(r.get('dispersion_mas'), (int, float))
    ]
    int_vals = [
        float(r['internal_max_delta_mas'])
        for r in frame_rows
        if isinstance(r.get('internal_max_delta_mas'), (int, float))
    ]
    n_cals = [
        int(r['n_calibrators'])
        for r in frame_rows
        if isinstance(r.get('n_calibrators'), int)
    ]
    qa = build_frame_qa(
        mission='hst',
        hub_id=hub.name if hub else None,
        align_mode='JHAT',
        abs_ref=str(hub) if hub else (str(ref_phot) if ref_phot else None),
        residual_mas=max(abs_vals) if abs_vals else None,
        n_calibrators=min(n_cals) if n_cals else None,
        abs_method='catalog_residual',
        max_delta_mas=max(int_vals) if int_vals else None,
        frames=frame_rows,
    )
    write_frame_qa(out, qa)
    write_alignment_summary_table(frame_rows, out / 'alignment_summary.txt')
    warn_if_frame_qa_soft(qa, log=log, context=out.name)
    return qa
