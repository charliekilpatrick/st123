"""
Forced aperture photometry on JWST Level-3 / coadd images.

Uses photutils circular apertures sized from CRDS APCORR at the highest
tabulated encircled-energy fraction per instrument (NIRCam EE=0.90, MIRI
EE=0.80), applies the matching aperture correction and background annulus,
and converts ``MJy/sr`` → μJy/arcsec² → μJy → AB mag with uncertainty
propagation. Centers are supplied by the caller (forced photometry only).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry

from st123.utils.compatibility import (
    ensure_local_crds_context,
    patch_jwst_for_photutils3,
)

logger = logging.getLogger(__name__)

# AB magnitude zeropoint for flux in μJy:
#   m_AB = -2.5 log10(f_Jy) + 8.90
#   f_uJy = f_Jy * 1e6  →  ZP = 8.90 + 2.5 * 6 = 23.9
AB_ZEROPOINT_UJY = 23.9
_LN10 = math.log(10.0)

# Highest tabulated CRDS APCORR EE fractions (see local apcorr tables).
DEFAULT_EE_FRACTION = {
    'NIRCAM': 0.90,
    'MIRI': 0.80,
}


@dataclass(frozen=True)
class ApertureParams:
    """CRDS APCORR aperture parameters for one image / EE fraction."""

    radius_px: float
    apcorr: float
    sky_in_px: float
    sky_out_px: float
    ee_fraction: float
    filter: str
    instrument: str


def default_ee_fraction(instrument: str | None) -> float:
    """
    Return the default encircled-energy fraction for an instrument.

    Parameters
    ----------
    instrument : str or None
        Instrument name (e.g. ``NIRCAM``, ``MIRI``).

    Returns
    -------
    float
        EE fraction in ``[0, 1]``. Defaults to NIRCam EE=0.90 when unknown.
    """
    key = str(instrument or 'NIRCAM').upper().split('_', 1)[0]
    return float(DEFAULT_EE_FRACTION.get(key, DEFAULT_EE_FRACTION['NIRCAM']))


def _header_instrument_filter(image: str) -> tuple[str, str]:
    """Return ``(INSTRUME, FILTER)`` from a FITS primary/SCI header."""
    with fits.open(image) as hdul:
        hdr = hdul[0].header
        sci = hdul['SCI'].header if 'SCI' in hdul else hdr
    instrument = str(
        hdr.get('INSTRUME') or sci.get('INSTRUME') or 'NIRCAM'
    ).upper()
    filt = str(hdr.get('FILTER') or sci.get('FILTER') or 'UNKNOWN').upper()
    return instrument, filt


def get_aperture_params(
    image: str | Path,
    ee_fraction: float | None = None,
) -> ApertureParams:
    """
    Load CRDS APCORR radius, aperture correction, and sky annulus.

    Parameters
    ----------
    image : str or path
        JWST Level-2/3 FITS image used to select the APCORR reference.
    ee_fraction : float, optional
        Encircled-energy fraction (e.g. ``0.9``). Default follows
        :func:`default_ee_fraction` for the image instrument.

    Returns
    -------
    ApertureParams
        Aperture and background radii in pixels plus the multiplicative
        aperture correction.
    """
    image = str(Path(image).expanduser().resolve())
    instrument, filt = _header_instrument_filter(image)
    if ee_fraction is None:
        ee_fraction = default_ee_fraction(instrument)
    ee_fraction = float(ee_fraction)
    if ee_fraction > 1.0:
        # Allow CLI-style percentages (90 → 0.90).
        ee_fraction /= 100.0
    if not (0.0 < ee_fraction <= 1.0):
        raise ValueError(f'ee_fraction must be in (0, 1], got {ee_fraction}')

    ensure_local_crds_context()
    patch_jwst_for_photutils3()

    from jwst import datamodels
    from jwst.source_catalog import reference_data
    from jwst.source_catalog.source_catalog_step import SourceCatalogStep

    # SourceCatalogStep expects EE percentages as ints (30, 40, 70, …).
    ee_pct = int(round(ee_fraction * 100.0))
    # Provide two lower anchors so ReferenceData can build its EE ladder;
    # the last entry is the science aperture.
    lower = sorted({max(10, ee_pct - 20), max(20, ee_pct - 10), ee_pct})
    aperture_ee = tuple(lower)

    sc = SourceCatalogStep()
    with datamodels.open(image) as model:
        reffile_paths = sc._get_reffile_paths(model)
        refdata = reference_data.ReferenceData(
            model, reffile_paths, aperture_ee
        )
        params = refdata.aperture_params

    radii = list(params['aperture_radii'])
    apcorrs = list(params['aperture_corrections'])
    if not radii or not apcorrs:
        raise RuntimeError(f'No APCORR aperture parameters for {image}')

    return ApertureParams(
        radius_px=float(radii[-1]),
        apcorr=float(apcorrs[-1]),
        sky_in_px=float(params['bkg_aperture_inner_radius']),
        sky_out_px=float(params['bkg_aperture_outer_radius']),
        ee_fraction=ee_fraction,
        filter=filt,
        instrument=instrument,
    )


def pixel_scale_arcsec(wcs: WCS) -> float:
    """
    Return the mean pixel scale in arcseconds from a celestial WCS.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        Science WCS.

    Returns
    -------
    float
        Pixel scale in arcsec / pixel.
    """
    scales = wcs.proj_plane_pixel_scales()
    # Celestial axes are the first two for standard JWST SCI WCS.
    return float(np.mean([s.to_value(u.arcsec) for s in scales[:2]]))


def mjy_sr_to_ujy_arcsec2(data: np.ndarray | float) -> np.ndarray | float:
    """
    Convert surface brightness from ``MJy/sr`` to ``μJy/arcsec²``.

    Parameters
    ----------
    data : array or float
        Values in megajansky per steradian.

    Returns
    -------
    array or float
        Values in microjansky per square arcsecond.
    """
    arr = np.asarray(data, dtype=float)
    quantity = arr * (u.MJy / u.sr)
    out = quantity.to(u.uJy / u.arcsec**2).value
    if np.isscalar(data):
        return float(out)
    return out


def surface_brightness_to_ujy(
    sb_ujy_arcsec2: np.ndarray | float,
    area_arcsec2: np.ndarray | float,
) -> np.ndarray | float:
    """
    Convert surface brightness × solid angle to integrated μJy.

    Parameters
    ----------
    sb_ujy_arcsec2 : array or float
        Surface brightness in μJy/arcsec².
    area_arcsec2 : array or float
        Solid angle in arcsec².

    Returns
    -------
    array or float
        Flux density in μJy.
    """
    return np.asarray(sb_ujy_arcsec2, dtype=float) * np.asarray(
        area_arcsec2, dtype=float
    )


def ujy_to_abmag(
    flux_ujy: np.ndarray | float,
    flux_err_ujy: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
    """
    Convert μJy flux (and optional error) to AB magnitude.

    Parameters
    ----------
    flux_ujy : array or float
        Flux density in μJy.
    flux_err_ujy : array or float, optional
        1σ flux uncertainty in μJy.

    Returns
    -------
    mag, mag_err
        AB magnitudes and uncertainties. Non-positive fluxes yield ``nan``.
    """
    flux = np.asarray(flux_ujy, dtype=float)
    mag = np.full(flux.shape, np.nan, dtype=float)
    mag_err = np.full(flux.shape, np.nan, dtype=float)
    ok = np.isfinite(flux) & (flux > 0.0)
    mag[ok] = -2.5 * np.log10(flux[ok]) + AB_ZEROPOINT_UJY
    if flux_err_ujy is not None:
        err = np.asarray(flux_err_ujy, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            mag_err[ok] = (2.5 / _LN10) * (err[ok] / flux[ok])
    if np.ndim(flux_ujy) == 0:
        return float(mag), float(mag_err)
    return mag, mag_err


def load_sci_extensions(
    image: str | Path,
) -> tuple[np.ndarray, np.ndarray | None, WCS, fits.Header]:
    """
    Load SCI data, optional ERR, WCS, and SCI header from a FITS image.

    Parameters
    ----------
    image : str or path
        FITS path.

    Returns
    -------
    data, err, wcs, header
        Science array (float), error array or ``None``, WCS, and SCI header.
    """
    image = str(Path(image).expanduser().resolve())
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
        data = np.asarray(hdu.data, dtype=float)
        header = hdu.header.copy()
        wcs = WCS(header)
        err = None
        if 'ERR' in hdul and hdul['ERR'].data is not None:
            err = np.asarray(hdul['ERR'].data, dtype=float)
        bunit = str(header.get('BUNIT') or hdul[0].header.get('BUNIT') or '')
        if bunit and 'mjy' in bunit.lower() and '/sr' not in bunit.lower().replace(
            ' ', ''
        ):
            logger.warning(
                'Unexpected BUNIT=%s for %s; assuming MJy/sr surface brightness',
                bunit,
                image,
            )
        elif bunit and 'mjy/sr' not in bunit.lower().replace(' ', ''):
            logger.warning(
                'BUNIT=%s for %s; photometry assumes MJy/sr',
                bunit,
                image,
            )
    return data, err, wcs, header


def resolve_positions(
    *,
    x: Sequence[float] | np.ndarray | float | None = None,
    y: Sequence[float] | np.ndarray | float | None = None,
    ra: Sequence[float] | np.ndarray | float | None = None,
    dec: Sequence[float] | np.ndarray | float | None = None,
    wcs: WCS | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve forced-photometry centers to both pixel and sky coordinates.

    Provide either ``(x, y)`` or ``(ra, dec)`` (or both). Missing sky/pixel
    values are filled with ``wcs``.

    Parameters
    ----------
    x, y : array-like, optional
        Pixel coordinates (0-based, origin=0 for ``wcs`` conversions).
    ra, dec : array-like, optional
        ICRS degrees.
    wcs : astropy.wcs.WCS, optional
        Required when converting between pixel and sky.

    Returns
    -------
    x, y, ra, dec
        1-D float arrays of equal length.
    """
    has_xy = x is not None and y is not None
    has_sky = ra is not None and dec is not None
    if not has_xy and not has_sky:
        raise ValueError('Provide (x, y) and/or (ra, dec) positions')

    if has_xy:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        if x_arr.shape != y_arr.shape:
            raise ValueError('x and y must have the same shape')
    else:
        x_arr = y_arr = None

    if has_sky:
        ra_arr = np.atleast_1d(np.asarray(ra, dtype=float))
        dec_arr = np.atleast_1d(np.asarray(dec, dtype=float))
        if ra_arr.shape != dec_arr.shape:
            raise ValueError('ra and dec must have the same shape')
    else:
        ra_arr = dec_arr = None

    if has_xy and has_sky and x_arr.shape != ra_arr.shape:
        raise ValueError('Pixel and sky position arrays must match in length')

    if has_xy and not has_sky:
        if wcs is None:
            raise ValueError('wcs is required to convert (x, y) → (ra, dec)')
        sky = wcs.pixel_to_world(x_arr, y_arr)
        ra_arr = np.atleast_1d(np.asarray(sky.ra.deg, dtype=float))
        dec_arr = np.atleast_1d(np.asarray(sky.dec.deg, dtype=float))
    elif has_sky and not has_xy:
        if wcs is None:
            raise ValueError('wcs is required to convert (ra, dec) → (x, y)')
        world = wcs.pixel_to_world_values  # noqa: F841 — clarity
        xy = wcs.world_to_pixel_values(ra_arr, dec_arr)
        x_arr = np.atleast_1d(np.asarray(xy[0], dtype=float))
        y_arr = np.atleast_1d(np.asarray(xy[1], dtype=float))

    return x_arr, y_arr, ra_arr, dec_arr


def _annulus_stats(
    data: np.ndarray,
    annulus: CircularAnnulus,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-position sigma-clipped median and std of annulus pixels."""
    masks = annulus.to_mask(method='center')
    if not isinstance(masks, list):
        masks = [masks]
    medians = np.empty(len(masks), dtype=float)
    stds = np.empty(len(masks), dtype=float)
    for i, mask in enumerate(masks):
        values = mask.multiply(data)
        if values is None:
            medians[i] = np.nan
            stds[i] = np.nan
            continue
        flat = values[mask.data > 0]
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            medians[i] = np.nan
            stds[i] = np.nan
            continue
        _, median, std = sigma_clipped_stats(flat, sigma=3.0)
        medians[i] = float(median)
        stds[i] = float(std) if std > 0 else 0.0
    return medians, stds


def forced_aperture_photometry(
    image: str | Path,
    *,
    x: Sequence[float] | np.ndarray | float | None = None,
    y: Sequence[float] | np.ndarray | float | None = None,
    ra: Sequence[float] | np.ndarray | float | None = None,
    dec: Sequence[float] | np.ndarray | float | None = None,
    ee_fraction: float | None = None,
    aperture_params: ApertureParams | None = None,
) -> Table:
    """
    Perform forced aperture photometry at supplied positions.

    Parameters
    ----------
    image : str or path
        Level-3 / coadd FITS image (``MJy/sr`` SCI).
    x, y : array-like, optional
        Pixel centers.
    ra, dec : array-like, optional
        ICRS degrees.
    ee_fraction : float, optional
        Override default EE fraction for :func:`get_aperture_params`.
    aperture_params : ApertureParams, optional
        Precomputed APCORR parameters (skips CRDS lookup; useful in tests).

    Returns
    -------
    astropy.table.Table
        Forced-photometry catalog with fluxes in μJy and AB magnitudes.
    """
    image = str(Path(image).expanduser().resolve())
    data, err, wcs, _header = load_sci_extensions(image)
    x_arr, y_arr, ra_arr, dec_arr = resolve_positions(
        x=x, y=y, ra=ra, dec=dec, wcs=wcs
    )
    npos = len(x_arr)
    if npos == 0:
        raise ValueError('No positions provided for forced photometry')

    params = aperture_params or get_aperture_params(image, ee_fraction=ee_fraction)
    if params.sky_out_px <= params.sky_in_px:
        raise ValueError(
            f'Invalid sky annulus: sky_in={params.sky_in_px}, '
            f'sky_out={params.sky_out_px}'
        )

    pixscale = pixel_scale_arcsec(wcs)
    positions = np.column_stack([x_arr, y_arr])
    aperture = CircularAperture(positions, r=params.radius_px)
    annulus = CircularAnnulus(
        positions, r_in=params.sky_in_px, r_out=params.sky_out_px
    )

    phot = aperture_photometry(data, aperture, error=err, method='exact')
    bkg_median, bkg_std = _annulus_stats(data, annulus)
    aper_area_pix = float(aperture.area)
    aper_bkg = bkg_median * aper_area_pix
    sum_bkgsub = np.asarray(phot['aperture_sum'], dtype=float) - aper_bkg

    if err is not None and 'aperture_sum_err' in phot.colnames:
        flux_err_native = np.asarray(phot['aperture_sum_err'], dtype=float)
        # Background uncertainty on the aperture sum.
        bkg_err = bkg_std * math.sqrt(aper_area_pix)
        flux_err_native = np.hypot(flux_err_native, bkg_err)
    else:
        flux_err_native = bkg_std * math.sqrt(aper_area_pix)

    # Native aperture_sum is Σ(MJy/sr) over pixels. Convert via mean SB ×
    # solid angle (equivalent to sum × pixel_area_sr × 1e12 → μJy).
    area_arcsec2 = aper_area_pix * (pixscale**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_sb = np.where(
            aper_area_pix > 0,
            sum_bkgsub / aper_area_pix,
            np.nan,
        )
        mean_sb_err = np.where(
            aper_area_pix > 0,
            flux_err_native / aper_area_pix,
            np.nan,
        )
        sb_ujy_arcsec2 = mjy_sr_to_ujy_arcsec2(mean_sb)
        sb_err_ujy_arcsec2 = mjy_sr_to_ujy_arcsec2(mean_sb_err)
    flux_ujy = (
        surface_brightness_to_ujy(sb_ujy_arcsec2, area_arcsec2) * params.apcorr
    )
    flux_ujy_err = (
        surface_brightness_to_ujy(sb_err_ujy_arcsec2, area_arcsec2)
        * params.apcorr
    )

    abmag, abmag_err = ujy_to_abmag(flux_ujy, flux_ujy_err)

    return Table(
        {
            'x': x_arr,
            'y': y_arr,
            'ra': ra_arr,
            'dec': dec_arr,
            'ee_fraction': np.full(npos, params.ee_fraction),
            'radius_px': np.full(npos, params.radius_px),
            'apcorr': np.full(npos, params.apcorr),
            'sky_in_px': np.full(npos, params.sky_in_px),
            'sky_out_px': np.full(npos, params.sky_out_px),
            'filter': np.full(npos, params.filter),
            'instrument': np.full(npos, params.instrument),
            'sb_ujy_arcsec2': np.asarray(sb_ujy_arcsec2, dtype=float),
            'flux_ujy': np.asarray(flux_ujy, dtype=float),
            'flux_ujy_err': np.asarray(flux_ujy_err, dtype=float),
            'abmag': np.asarray(abmag, dtype=float),
            'abmag_err': np.asarray(abmag_err, dtype=float),
        }
    )


def read_coords_table(path: str | Path) -> tuple[dict[str, np.ndarray], str]:
    """
    Load a coordinate table with ``x,y`` or ``ra,dec`` columns.

    Parameters
    ----------
    path : str or path
        ASCII / ECSV catalog.

    Returns
    -------
    kwargs, mode
        ``kwargs`` for :func:`forced_aperture_photometry` and ``'xy'`` or
        ``'sky'`` describing which columns were used (sky preferred when both
        are present).
    """
    path = Path(path).expanduser().resolve()
    try:
        table = Table.read(path)
    except Exception:
        table = Table.read(path, format='ascii')
    lower = {c.lower(): c for c in table.colnames}
    has_xy = 'x' in lower and 'y' in lower
    has_sky = 'ra' in lower and 'dec' in lower
    if not has_xy and not has_sky:
        raise ValueError(
            f'{path} must contain columns x,y and/or ra,dec; '
            f'found {list(table.colnames)}'
        )
    if has_sky:
        return (
            {
                'ra': np.asarray(table[lower['ra']], dtype=float),
                'dec': np.asarray(table[lower['dec']], dtype=float),
            },
            'sky',
        )
    return (
        {
            'x': np.asarray(table[lower['x']], dtype=float),
            'y': np.asarray(table[lower['y']], dtype=float),
        },
        'xy',
    )
