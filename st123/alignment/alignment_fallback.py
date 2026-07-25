"""
Fallback MIRI→MIRI relative alignment helpers.

When direct NIRCam alignment fails, align the failed frame to a successfully
aligned MIRI image that is closest in wavelength and has the largest footprint
overlap. Absolute dispersion is the quadrature sum of the parent absolute
dispersion and the new relative dispersion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from st123.mosaic.region import SRegionPolygon


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
    """Return central wavelength (µm) for a MIRI filter name."""
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


def sort_frames_blue_to_red(
    frames: list,
    *,
    filter_from_path,
) -> list:
    """Sort overlap frames by increasing filter wavelength, then path."""

    def key(frame) -> tuple:
        path = frame['miri_path'] if isinstance(frame, dict) else frame.miri_path
        filt = filter_from_path(path) or 'UNKNOWN'
        return (filter_wavelength_um(filt), str(path))

    return sorted(frames, key=key)


def load_s_region(fits_path: str) -> SRegionPolygon:
    """Parse ``S_REGION`` from the first HDU that defines it."""
    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            if 'S_REGION' in hdu.header:
                return SRegionPolygon.parse(hdu.header['S_REGION'])
    raise KeyError(f'No S_REGION in {fits_path}')


def sky_overlap_fraction(miri_a: str, miri_b: str) -> float:
    """
    Fraction of ``miri_a``'s sky footprint overlapped by ``miri_b``.

    Uses header ``S_REGION`` polygons in a local tangent plane (arcsec).
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

    Preference order (among parents with sky overlap ≥ ``min_overlap_fraction``):
      1. Lowest score
         ``sqrt(parent_abs² + assume_relative_mas²)
          + wavelength_penalty_mas_per_um * |Δλ|``
         — prefers high-quality parents, but not arbitrarily blue ones that
         often fail relative matching across large wavelength gaps
      2. Closest filter wavelength
      3. Largest sky footprint overlap with ``miri_path``
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
    for score, dlam, neg_frac, parent in ranked[: max(1, int(max_parents))]:
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
    """Choose the single best fallback parent (see ``rank_fallback_parents``)."""
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


def combine_dispersion_mas(parent_mas: float, relative_mas: float) -> float:
    """Absolute dispersion = quadrature sum of parent and relative terms."""
    return float(math.sqrt(parent_mas**2 + relative_mas**2))


def find_aligned_photfile(jhat_path: str) -> str | None:
    """Locate post-alignment photometry next to a JHAT product, if present."""
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

    Keywords
    --------
    ALGNMODE : ``REFERENCE`` (absolute align to a reference image) or
               ``MIRI_REL`` (relative align to another MIRI frame)
    ALGNREF  : original (root) reference image used for the absolute frame
    ALGNTO   : image / catalog this frame was aligned to in this step
    JWDISPR  : relative step dispersion (arcsec)
    JWDISPM  : absolute dispersion (arcsec); for ``MIRI_REL`` this is the
               quadrature combination of parent absolute + relative
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
