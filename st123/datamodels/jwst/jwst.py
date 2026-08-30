"""JWST science-product datamodel (array sanitize + WCS cards)."""

from __future__ import annotations

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
