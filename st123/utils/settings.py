"""
Project-wide defaults for filters, MAST queries, alignment, JHAT, and DOLPHOT.

Edit values here to tune pipeline defaults. Call sites import these symbols
rather than hardcoding instrument/filter lists or parameter dictionaries.

Sections
--------
Imaging filters
    Header-style filter names (``FILTER`` / ``FILTER1`` / ``FILTER2``) grouped
    by instrument, with telescope documented per group. Membership checks use
    the flat :data:`acceptable_filters` tuple (uppercase); :func:`st123.utils.helpers.get_filter`
    returns lowercase names from FITS headers.
Reference-image preference
    :data:`BEST_REFERENCE_FILTERS` / :data:`BEST_FILTER_TYPES` for
    :func:`~st123.utils.helpers.pick_deepest_images`.
MAST defaults
    Product rules, default instruments/filters, download layout.
Alignment quality holds
    Per-filter MIRI REFERENCE→MIRI_REL dispersion thresholds.
JHAT / DOLPHOT parameter dictionaries
    Merged into JHAT runs and DOLPHOT paramfiles by
    :mod:`st123.alignment.align` and :mod:`st123.photometry.dolphot`.
"""

from __future__ import annotations

# =============================================================================
# Imaging filters (FITS header style, uppercase)
# =============================================================================
#
# Names match typical ``FILTER`` / ``FILTER1`` / ``FILTER2`` values after
# uppercasing. Dual-wheel JWST frames often have ``FILTER1=CLEAR`` and the
# science band in ``FILTER2``; :func:`~st123.utils.helpers.get_filter` resolves
# that and returns a lowercase band name for comparisons.
#
# Telescope is documented on each instrument group. Some bandpasses are shared
# across instruments (e.g. HST ``F606W`` on WFPC2, ACS, and WFC3).

# Hubble Space Telescope — WFPC2 imaging (legacy; ``FILTER`` / ``FILTNAM1``)
WFPC2_FILTERS: tuple[str, ...] = (
    'F122M', 'F160BW', 'F185W', 'F218W', 'F255W', 'F300W', 'F336W', 'F375N',
    'F380W', 'F390N', 'F437N', 'F439W', 'F450W', 'F467M', 'F469N', 'F487N',
    'F502N', 'F547M', 'F555W', 'F569W', 'F588N', 'F606W', 'F622W', 'F631N',
    'F656N', 'F658N', 'F673N', 'F675W', 'F702W', 'F785LP', 'F791W', 'F814W',
    'F850LP', 'F953N', 'F1042M',
)

# Hubble Space Telescope — ACS (WFC / HRC / SBC imaging)
ACS_FILTERS: tuple[str, ...] = (
    # ACS/WFC + ACS/HRC broadband / medium / narrow
    'F220W', 'F250W', 'F330W', 'F344N', 'F435W', 'F475W', 'F502N', 'F550M',
    'F555W', 'F606W', 'F625W', 'F658N', 'F660N', 'F775W', 'F814W', 'F850LP',
    'F892N',
    # ACS/SBC long-pass / medium
    'F115LP', 'F122M', 'F125LP', 'F140LP', 'F150LP', 'F165LP',
)

# Hubble Space Telescope — WFC3 (UVIS + IR imaging)
WFC3_FILTERS: tuple[str, ...] = (
    # WFC3/UVIS
    'F200LP', 'F218W', 'F225W', 'F275W', 'F280N', 'F300X', 'F336W', 'F343N',
    'F350LP', 'F373N', 'F390M', 'F390W', 'F395N', 'F410M', 'F438W', 'F467M',
    'F469N', 'F475W', 'F475X', 'F487N', 'F502N', 'F547M', 'F555W', 'F600LP',
    'F606W', 'F621M', 'F625W', 'F631N', 'F645N', 'F656N', 'F657N', 'F658N',
    'F665N', 'F673N', 'F680N', 'F689M', 'F763M', 'F775W', 'F814W', 'F845M',
    'F850LP', 'F953N',
    # WFC3/IR
    'F098M', 'F105W', 'F110W', 'F125W', 'F126N', 'F127M', 'F128N', 'F130N',
    'F132N', 'F139M', 'F140W', 'F153M', 'F160W', 'F164N', 'F167N',
)

# James Webb Space Telescope — NIRCam imaging (FILTER / FILTER2)
NIRCAM_FILTERS: tuple[str, ...] = (
    # Short-wavelength channel
    'F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F150W2', 'F162M', 'F164N',
    'F182M', 'F187N', 'F200W', 'F210M', 'F212N',
    # Long-wavelength channel
    'F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 'F335M', 'F356W', 'F360M',
    'F405N', 'F410M', 'F430M', 'F444W', 'F460M', 'F466N', 'F470N', 'F480M',
    # Dual-wheel clear positions sometimes appear as the active FILTER value
    'CLEAR',
)

# James Webb Space Telescope — MIRI imaging / coronagraph
MIRI_FILTERS: tuple[str, ...] = (
    # Imager (MIRIM) broadband
    'F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W',
    'F2100W', 'F2550W',
    # Coronagraphic
    'F1065C', 'F1140C', 'F1550C', 'F2300C',
)

# Nancy Grace Roman Space Telescope — WFI imaging elements (``FILTER``)
ROMAN_WFI_FILTERS: tuple[str, ...] = (
    'F062', 'F087', 'F106', 'F129', 'F146', 'F158', 'F184', 'F213',
)

# Euclid — VIS + NISP photometric bands
# Product headers may use instrument names (YE/JE/HE) or mosaic tags (NIR_Y/…).
EUCLID_FILTERS: tuple[str, ...] = (
    'VIS',
    'YE', 'JE', 'HE',
    'NIR_Y', 'NIR_J', 'NIR_H',
    'Y', 'J', 'H',
)

# Instrument → telescope metadata and filter tuple (for docs / introspection).
FILTERS_BY_INSTRUMENT: dict[str, dict[str, object]] = {
    'WFPC2': {
        'telescope': 'HST',
        'filters': WFPC2_FILTERS,
        'header_keys': ('FILTER', 'FILTNAM1', 'FILTNAM2'),
    },
    'ACS': {
        'telescope': 'HST',
        'filters': ACS_FILTERS,
        'header_keys': ('FILTER1', 'FILTER2', 'FILTER'),
    },
    'WFC3': {
        'telescope': 'HST',
        'filters': WFC3_FILTERS,
        'header_keys': ('FILTER', 'FILTER1', 'FILTER2'),
    },
    'NIRCAM': {
        'telescope': 'JWST',
        'filters': NIRCAM_FILTERS,
        'header_keys': ('FILTER', 'FILTER1', 'FILTER2'),
    },
    'MIRI': {
        'telescope': 'JWST',
        'filters': MIRI_FILTERS,
        'header_keys': ('FILTER',),
    },
    'WFI': {
        'telescope': 'Roman',
        'filters': ROMAN_WFI_FILTERS,
        'header_keys': ('FILTER',),
    },
    'VIS': {
        'telescope': 'Euclid',
        'filters': ('VIS',),
        'header_keys': ('FILTER',),
    },
    'NISP': {
        'telescope': 'Euclid',
        'filters': ('YE', 'JE', 'HE', 'NIR_Y', 'NIR_J', 'NIR_H', 'Y', 'J', 'H'),
        'header_keys': ('FILTER',),
    },
}

# Flat membership list used by :func:`~st123.utils.helpers.pick_deepest_images`
# (compare with ``name.upper() in acceptable_filters``).
acceptable_filters: tuple[str, ...] = tuple(
    sorted(
        {
            *WFPC2_FILTERS,
            *ACS_FILTERS,
            *WFC3_FILTERS,
            *NIRCAM_FILTERS,
            *MIRI_FILTERS,
            *ROMAN_WFI_FILTERS,
            *EUCLID_FILTERS,
        }
    )
)

# =============================================================================
# Reference-image preference (pick_deepest_images)
# =============================================================================

# Preferred bands for a DOLPHOT / drizzle reference, in roughly decreasing
# preference. Stored lowercase to match :func:`~st123.utils.helpers.get_filter`.
BEST_REFERENCE_FILTERS: tuple[str, ...] = (
    'f625w',
    'f606w',
    'f555w',
    'f814w',
    'f350lp',
    'f110w',
    'f105w',
    'f336w',
)

# Preferred filter-name suffixes when no ``BEST_REFERENCE_FILTERS`` band is
# available (long-pass → wide → … → narrow).
BEST_FILTER_TYPES: tuple[str, ...] = ('lp', 'w', 'x', 'm', 'n')

# =============================================================================
# MAST query / download defaults
# =============================================================================

# Default HST science products used by download helpers / notebooks.
# Each entry is ``(filename_suffix, instrument_tag_substring)``.
HST_PRODUCT_RULES: tuple[tuple[str, str], ...] = (
    ('c0m.fits', 'WFPC2'),
    ('c1m.fits', 'WFPC2'),
    ('c0m.fits', 'PC/WFC'),
    ('c1m.fits', 'PC/WFC'),
    ('flc.fits', 'ACS/WFC'),
    ('flt.fits', 'ACS/HRC'),
    ('flc.fits', 'WFC3/UVIS'),
    ('flt.fits', 'WFC3/IR'),
)

# Default HST imaging filters for MAST queries. ``None`` at call sites means
# no filter restriction (recommended for SN fields with mixed legacy bands).
DEFAULT_HST_FILTERS: tuple[str, ...] | None = None
DEFAULT_HST_INSTRUMENTS: tuple[str, ...] = ('ACS', 'WFC3', 'WFPC2')
DEFAULT_JWST_INSTRUMENTS: tuple[str, ...] = ('NIRCAM', 'MIRI')

# Relative subdirectory pattern under the download root.
DEFAULT_DOWNLOAD_LAYOUT: str = 'telescope/instrument/filter/obsid'

# =============================================================================
# Alignment quality-hold defaults (REFERENCE → MIRI_REL)
# =============================================================================

# Per-filter REFERENCE absolute-dispersion ceilings (mas). Empirically, MIRI_REL
# absolute dispersion is typically worse than REFERENCE below these cuts and
# better above them when a good overlapping parent exists. ``None`` disables
# the quality-hold fallback (F560W stays on REFERENCE).
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

# Default when a filter is absent from :data:`FILTER_MAX_REFERENCE_DISPERSION_MAS`.
DEFAULT_MAX_REFERENCE_DISPERSION_MAS: float = 70.0

# Default output directory name for ``align --mode pair`` when ``--base-dir``
# is omitted.
DEFAULT_PAIR_OUTDIR: str = 'alignment_output'

# =============================================================================
# JHAT parameter dictionaries
# =============================================================================

strict_gaia_params = {
    'telescope': 'jwst',
    'overwrite': True,
    'd2d_max': 0.5,
    'showplots': 0,
    'find_stars_threshold': 5,
    'iterate_with_xyshifts': True,
    'histocut_order': 'dxdy',
    'sharpness_lim': (0.3, 0.95),
    'roundness1_lim': (-0.7, 0.7),
    'SNR_min': 5,
    'dmag_max': 0.1,
    'objmag_lim': (15, 25),
    'slope_min': -20 / 2048,
    'binsize_px': 1.0,
    'savephottable': 0,
}

relaxed_gaia_params = {
    'telescope': 'jwst',
    'overwrite': True,
    'd2d_max': 2.0,
    'showplots': 0,
    'find_stars_threshold': 3,
    'iterate_with_xyshifts': False,
    'histocut_order': 'dxdy',
    'sharpness_lim': (0.3, 0.95),
    'roundness1_lim': (-0.7, 0.7),
    'SNR_min': 3,
    'dmag_max': 0.1,
    'slope_min': -20 / 2048,
    'binsize_px': 1.0,
    'savephottable': 0,
}

strict_jwst_params = {
    'telescope': 'jwst',
    'refcat_racol': 'ra',
    'refcat_deccol': 'dec',
    'refcat_magcol': 'mag',
    'refcat_magerrcol': 'dmag',
    'overwrite': True,
    'd2d_max': 0.5,
    'showplots': 0,
    'find_stars_threshold': 5,
    'iterate_with_xyshifts': True,
    'histocut_order': 'dxdy',
    'sharpness_lim': (0.3, 0.95),
    'roundness1_lim': (-0.7, 0.7),
    'SNR_min': 5,
    'dmag_max': 0.1,
    'objmag_lim': (15, 25),
    'slope_min': -20 / 2048,
    'binsize_px': 1.0,
    'savephottable': 0,
}

relaxed_jwst_params = {
    'telescope': 'jwst',
    'refcat_racol': 'ra',
    'refcat_deccol': 'dec',
    'refcat_magcol': 'mag',
    'refcat_magerrcol': 'dmag',
    'overwrite': True,
    'd2d_max': 2.0,
    'showplots': 0,
    'find_stars_threshold': 3,
    'iterate_with_xyshifts': False,
    'histocut_order': 'dxdy',
    'sharpness_lim': (0.3, 0.95),
    'roundness1_lim': (-0.7, 0.7),
    'SNR_min': 3,
    'dmag_max': 0.1,
    'slope_min': -20 / 2048,
    'binsize_px': 1.0,
    'savephottable': 0,
}

# =============================================================================
# DOLPHOT parameter dictionaries
# =============================================================================

# Global DOLPHOT defaults for NIRCam mosaic runs.
base_params = {
    'FitSky': '2',
    'SigPSF': '5.0',
    'FlagMask': '4',
    'SecondPass': '5',
    'PSFPhotIt': '2',
    'ApCor': '1',
    'FSat': '0.999',
    'NoiseMult': '0.1',
    'RCombine': '1.5',
    'CombineChi': '0',
    'MaxIT': '25',
    'InterpPSFlib': '1',
    'SigFindMult': '0.85',
    'PSFPhot': '1',
    'Force1': '0',
    'SkySig': '2.25',
    'SkipSky': '1',
    'UseWCS': '2',
    'PSFres': '1',
    'PosStep': '0.25',
    'NIRCAMvega': '0',
    'Align': '4',
    'aligntol': '0',
    'Rotate': '1',
}

# Per-image NIRCam DOLPHOT geometry (short- vs long-wavelength channels).
short_params = {
    'shift': '0 0',
    'xform': '1 0 0',
    'raper': '2',
    'rchi': '1.5',
    'rsky0': '15',
    'rsky1': '35',
    'rsky2': '3 10',
    'rpsf': '15',
    'apsky': '20 35',
}

long_params = {
    'shift': '0 0',
    'xform': '1 0 0',
    'raper': '3',
    'rchi': '2.0',
    'rsky0': '15',
    'rsky1': '35',
    'rsky2': '4 10',
    'rpsf': '15',
    'apsky': '20 35',
}

# MIRI per-image params for FitSky=2 (dolphotMIRI.pdf §4.1).
# RAper/RPSF cannot exceed 24 for MIRI.
miri_params = {
    'shift': '0 0',
    'xform': '1 0 0',
    'raper': '3',
    'rchi': '2.0',
    'rsky0': '15',
    'rsky1': '35',
    'rsky2': '4 10',
    'rpsf': '15',
    'apsky': '20 35',
}

# Global params when MIRI frames are present (UseWCS=2 required).
# MIRIvega=0 matches NIRCAMvega=0 (AB mag / Jy) used in NIRCam runs.
miri_base_params = {
    **base_params,
    'MIRIvega': '0',
    'RCentroid': '1',
}

# calcsky: NIRCam (mosaic defaults) vs MIRI (dolphotMIRI.pdf §3.4).
nircam_calcsky_params = {
    'rin': 15,
    'rout': 25,
    'step': -64,
    'sigma_low': 2.25,
    'sigma_high': 2.00,
}

miri_calcsky_params = {
    'rin': 10,
    'rout': 25,
    'step': -64,  # quick sky; sufficient with FitSky != 0
    'sigma_low': 2.25,
    'sigma_high': 2.00,
}

# -----------------------------------------------------------------------------
# HST DOLPHOT (ACS / WFC3 / WFPC2) — ported from hst123 detector_defaults
# -----------------------------------------------------------------------------

# Global DOLPHOT knobs for HST runs. UseWCS=1 trusts the image WCS for
# frame→reference registration (JHAT / TweakReg already aligned the stack).
hst_base_params = {
    **base_params,
    'UseWCS': '1',
    'Align': '2',
    'ACSuseCTE': '0',
    'WFC3useCTE': '0',
    'WFPC2useCTE': '1',
    'FlagMask': '7',
    'RCentroid': '2',
    'Force1': '1',
}

# Per-image geometry (img_*_raper / img_*_rpsf, …). Keys match write_paramfile.
acs_params = {
    'shift': '0 0',
    'xform': '1 0 0',
    'raper': '2',
    'rchi': '1.5',
    'rsky0': '15',
    'rsky1': '35',
    'rsky2': '3 6',
    'rpsf': '10',
    'apsky': '15 25',
}

wfc3_uvis_params = {
    'shift': '0 0',
    'xform': '1 0 0',
    'raper': '3',
    'rchi': '2.0',
    'rsky0': '15',
    'rsky1': '35',
    'rsky2': '4 10',
    'rpsf': '13',
    'apsky': '15 25',
}

wfc3_ir_params = {
    'shift': '0 0',
    'xform': '1 0 0',
    'raper': '2',
    'rchi': '1.5',
    'rsky0': '8',
    'rsky1': '20',
    'rsky2': '3 10',
    'rpsf': '15',
    'apsky': '8 20',
}

# Default WFC3 per-image params (UVIS); IR frames override via classify_image_kind.
wfc3_params = dict(wfc3_uvis_params)

wfpc2_params = {
    'shift': '0 0',
    'xform': '1 0 0',
    'raper': '3',
    'rchi': '2.0',
    'rsky0': '15',
    'rsky1': '35',
    'rsky2': '4 10',
    'rpsf': '13',
    'apsky': '15 25',
}

# calcsky annulus defaults (hst123 detector_defaults dolphot_sky).
acs_calcsky_params = {
    'rin': 15,
    'rout': 35,
    'step': 4,
    'sigma_low': 2.25,
    'sigma_high': 2.00,
}

wfc3_calcsky_params = {
    'rin': 15,
    'rout': 35,
    'step': 4,
    'sigma_low': 2.25,
    'sigma_high': 2.00,
}

wfpc2_calcsky_params = {
    'rin': 10,
    'rout': 25,
    'step': 2,
    'sigma_low': 2.25,
    'sigma_high': 2.00,
}

# AstroDrizzle defaults (subset of hst123.drizzle_defaults).
hst_drizzle_defaults = {
    'final_pixfrac': 0.8,
    'driz_sep_pixfrac': 0.8,
    'combine_maskpt': 0.2,
    'combine_nsigma': '4 3',
    'driz_cr_snr': '3.5 3.0',
    'driz_cr_grow': 1,
    'driz_cr_scale': '1.2 0.7',
    'num_cores': 4,
}

# DQ bits treated as good by AstroDrizzle (driz_sep_bits / final_bits).
# Matches hst123.detector_defaults.
hst_driz_bits = {
    'acs': 96,
    'wfc3': 96,  # UVIS
    'wfc3_uvis': 96,
    'wfc3_ir': 576,
    'wfpc2': 1032,
}

# LAcosmic / astroscrappy defaults (hst123.instrument_defaults crpars).
hst_crpars = {
    'wfc3': {
        'rdnoise': 6.5,
        'gain': 1.0,
        'saturate': 70000.0,
        'sig_clip': 4.0,
        'sig_frac': 0.2,
        'obj_lim': 6.0,
    },
    'acs': {
        'rdnoise': 6.5,
        'gain': 1.0,
        'saturate': 70000.0,
        'sig_clip': 3.0,
        'sig_frac': 0.1,
        'obj_lim': 5.0,
    },
    'wfpc2': {
        'rdnoise': 10.0,
        'gain': 7.0,
        'saturate': 27000.0,
        'sig_clip': 4.0,
        'sig_frac': 0.3,
        'obj_lim': 6.0,
    },
}

# Bit value written into DQ / WFPC2 c1m for astroscrappy CR pixels.
HST_CR_DQ_BIT = 4096

# WFPC2 calibrated c0m chips retain a bad left-edge / overscan strip. Mask this
# many pixels on each side in c1m before AstroDrizzle (bit must NOT be in
# hst_driz_bits['wfpc2'] = 1032). Left edge is worse (A/D overscan bleed).
WFPC2_OVERSCAN_EDGE_PIX = 24
WFPC2_OVERSCAN_LEFT_EXTRA = 16  # total left mask width = EDGE + LEFT_EXTRA
WFPC2_OVERSCAN_DQ_BIT = 256
# Blank extreme negative SCI before drizzle (overscan bleed / fill values).
WFPC2_SCI_FLOOR = -20.0
# Grow negative / edge mask by this many pixels (binary dilation).
WFPC2_BAD_GROW_PIX = 3
# Kill entire columns in the left half when this fraction of pixels are < floor.
WFPC2_BAD_COL_FRAC = 0.50
