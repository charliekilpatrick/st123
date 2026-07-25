"""
Filter-dependent astrometric calibrator selection for MIRI alignment.

F770W fields often yield hundreds of detections (PAH / arm structure). Across
frames, higher ``n_calibrators`` correlates with worse dispersion because busy
fields are harder — but *within* a frame the brightest JHAT matches are
typically the most coherent calibrators. Severely trimming ``Nbright`` and
hard-clipping refine residuals therefore improves F770W solutions more than
magnitude / morphology cuts that reject bright sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CalibratorSettings:
    """Per-filter knobs for JHAT + iterative refine."""

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
    # JHAT native source cuts (applied before Nbright). None = keep defaults
    # from ``nircam_settings.strict_jwst_params``.
    sharpness_lim: tuple[float, float] | None = None
    roundness1_lim: tuple[float, float] | None = None
    objmag_lim: tuple[float, float] | None = None
    dmag_max: float | None = None

    def jhat_param_overrides(self) -> dict[str, Any]:
        """Extra kwargs for ``st123.align_jwst_image`` / JHAT ``run_all``."""
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
        """Keyword arguments consumed by ``alignment_dispersion.run_alignment``."""
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


def calibrator_settings_for_filter(filter_name: str | None) -> CalibratorSettings:
    """Return calibrator settings for a MIRI filter name."""
    key = str(filter_name or '').upper().split('_', 1)[0]
    if key == 'F770W':
        return F770W_CALIBRATOR_SETTINGS
    return DEFAULT_CALIBRATOR_SETTINGS


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


def max_reference_dispersion_mas(filter_name: str | None) -> float | None:
    """
    Return the REFERENCE quality-hold threshold (mas) for ``filter_name``.

    ``None`` means do not quality-hold / try MIRI_REL for dispersion alone
    (used for F560W).
    """
    key = str(filter_name or '').upper().split('_', 1)[0]
    if key in FILTER_MAX_REFERENCE_DISPERSION_MAS:
        return FILTER_MAX_REFERENCE_DISPERSION_MAS[key]
    return DEFAULT_MAX_REFERENCE_DISPERSION_MAS


def describe_calibrator_settings(settings: CalibratorSettings) -> str:
    """One-line summary for logs."""
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
