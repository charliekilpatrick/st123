"""JWST science-product datamodel (array sanitize + WCS cards)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from st123.datamodels.instrument import InstrumentDataModel, PathLike

# JWST DQ bit: do not use pixel in science.
_JWST_DO_NOT_USE = 1

__all__ = ['JWSTDataModel', 'sanitize_jwst_l2']


def _sky_fill_value(sci: np.ndarray, good: np.ndarray) -> float:
    """Sigma-clipped median of illuminated finite SCI, else 0."""
    if not np.any(good):
        return 0.0
    try:
        from astropy.stats import sigma_clipped_stats

        _, med, _ = sigma_clipped_stats(sci[good], sigma=3.0, maxiters=5)
        if med is not None and np.isfinite(med):
            return float(med)
    except Exception:
        pass
    return float(np.nanmedian(sci[good]))


class JWSTDataModel(InstrumentDataModel):
    """JWST L2/L3 product: finite SCI/ERR, DQ ``DO_NOT_USE``, WHT consistency."""

    telescope = 'JWST'
    dq_do_not_use = _JWST_DO_NOT_USE
    INSTRUMENTS: tuple[str, ...] = ('NIRCAM', 'MIRI')
    DOLPHOT_BASE_PARAMS: dict[str, str] = {
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
    JHAT_STRICT: dict = {
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
    JHAT_RELAXED: dict = {
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
    JHAT_GAIA_STRICT: dict = {
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
    JHAT_GAIA_RELAXED: dict = {
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
    JHAT_CROWDED: dict = {
        'telescope': 'jwst',
        'refcat_racol': 'ra',
        'refcat_deccol': 'dec',
        'refcat_magcol': 'mag',
        'refcat_magerrcol': 'dmag',
        'overwrite': True,
        'd2d_max': 0.35,
        'showplots': 0,
        'find_stars_threshold': 8,
        'iterate_with_xyshifts': True,
        'histocut_order': 'dxdy',
        'sharpness_lim': (0.35, 0.90),
        'roundness1_lim': (-0.55, 0.55),
        'SNR_min': 8,
        'dmag_max': 0.08,
        'objmag_lim': (12, 20),
        'slope_min': -20 / 2048,
        'binsize_px': 1.0,
        'savephottable': 0,
    }
    CROWDED_JHAT_NBRIGHT: int = 100

    @classmethod
    def matches(cls, value: object | None) -> bool:
        """True when a MAST name, ``INSTRUME`` card, or path is this instrument."""
        inst = (cls.instrument or '').upper()
        if not inst or value is None:
            return False
        text = str(value).strip().upper()
        return bool(text) and inst in text

    @classmethod
    def coverage_under(cls, *roots: PathLike) -> tuple[int, int]:
        """Return ``(n_nircam, n_miri)`` science FITS counts under *roots*."""
        from st123.datamodels.jwst.miri import MIRIDataModel
        from st123.datamodels.jwst.nircam import NIRCamDataModel

        n_nrc = 0
        n_miri = 0
        seen: set[Path] = set()
        for root in roots:
            base = Path(root)
            if not base.is_dir():
                continue
            for path in base.rglob('*.fits'):
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                if NIRCamDataModel.matches(resolved):
                    n_nrc += 1
                elif MIRIDataModel.matches(resolved):
                    n_miri += 1
        return n_nrc, n_miri

    def sanitize(
        self,
        *,
        materialize_headers: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Enforce finite SCI/ERR and consistent DQ/WHT on a JWST L2/L3 product.

        Non-finite SCI (or ERR) pixels are filled with a robust sky estimate and
        flagged ``DO_NOT_USE`` in DQ. When a ``WHT`` extension exists (i2d), those
        pixels are set to weight 0 - same spirit as HST drizzle SCI fill.

        Optionally materializes DATE-* / OBSGEO-L/B/H on headers that carry MJD /
        OBSGEO-XYZ so later WCS opens stay quiet.

        Idempotent for already-clean products.
        """
        p = self.path
        report: dict[str, Any] = {
            'path': str(p),
            'ok': True,
            'skipped': False,
            'n_sci_filled': 0,
            'n_err_fixed': 0,
            'n_wht_zeroed': 0,
            'headers': {},
            'telescope': 'JWST',
            'instrument': self.instrument,
        }
        if not force and self._sanitized:
            report['skipped'] = True
            report['reason'] = 'already sanitized'
            return report
        hdul = self._open_update(report)
        if hdul is None:
            return report

        with hdul:
            pri = hdul[0].header
            report['instrument'] = str(
                pri.get('INSTRUME') or pri.get('INSTRUMENT') or self.instrument or ''
            ).strip().upper()
            if not force and pri.get('ST123SAN'):
                report['skipped'] = True
                report['reason'] = 'already sanitized'
                self._sanitized = True
                return report
            try:
                sci_hdu = hdul['SCI']
            except KeyError:
                report['skipped'] = True
                report['reason'] = 'no SCI'
                self._sanitized = True
                return report

            sci = np.asarray(sci_hdu.data, dtype=np.float32)
            err_hdu = hdul['ERR'] if 'ERR' in hdul else None
            dq_hdu = hdul['DQ'] if 'DQ' in hdul else None
            wht_hdu = hdul['WHT'] if 'WHT' in hdul else None

            err = (
                np.asarray(err_hdu.data, dtype=np.float32)
                if err_hdu is not None and err_hdu.data is not None
                else None
            )
            bad = ~np.isfinite(sci)
            if err is not None and err.shape == sci.shape:
                bad |= ~np.isfinite(err)

            if np.any(bad):
                good = np.isfinite(sci) & (~bad)
                if err is not None and err.shape == sci.shape:
                    good &= np.isfinite(err)
                sky = _sky_fill_value(sci, good)
                n_fill = int(np.count_nonzero(bad))
                sci[bad] = sky
                sci_hdu.data = sci
                report['n_sci_filled'] = n_fill

                if err is not None and err.shape == sci.shape:
                    fill_err = np.nanmedian(err[np.isfinite(err)]) if np.any(
                        np.isfinite(err)
                    ) else 1.0
                    if not np.isfinite(fill_err) or fill_err <= 0:
                        fill_err = 1.0
                    n_err = int(np.count_nonzero(~np.isfinite(err) | bad))
                    err[~np.isfinite(err) | bad] = float(fill_err)
                    err_hdu.data = err
                    report['n_err_fixed'] = n_err

                if dq_hdu is not None and dq_hdu.data is not None:
                    dq = np.asarray(dq_hdu.data)
                    if dq.shape == sci.shape:
                        dq = dq.copy()
                        dq[bad] = np.bitwise_or(
                            dq[bad].astype(np.uint32),
                            int(self.dq_do_not_use),
                        )
                        dq_hdu.data = dq

                if wht_hdu is not None and wht_hdu.data is not None:
                    wht = np.asarray(wht_hdu.data, dtype=np.float32)
                    if wht.shape == sci.shape:
                        n_w = int(np.count_nonzero(bad & (wht != 0)))
                        wht[bad] = 0.0
                        wht_hdu.data = wht
                        report['n_wht_zeroed'] = n_w

            if materialize_headers:
                report['headers'] = self._materialize_open_headers(hdul)

            pri['ST123SAN'] = (
                True,
                'st123 datamodel sanitize applied (finite SCI/ERR + WCS cards)',
            )
            self._sanitized = True

        return report


def sanitize_jwst_l2(
    path: PathLike,
    *,
    materialize_headers: bool = True,
) -> dict[str, Any]:
    """Sanitize *path* with the JWST L2/L3 contract (array fill + WCS cards)."""
    from st123.datamodels.instrument import as_datamodel

    model = as_datamodel(path)
    if not isinstance(model, JWSTDataModel):
        model = JWSTDataModel.from_path(path)
    return model.sanitize(materialize_headers=materialize_headers)
