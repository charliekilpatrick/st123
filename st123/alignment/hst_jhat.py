"""
HST JHAT alignment helpers (Gaia by default) with a WFPC2 ``*_c0m.fits`` patch.

Upstream ``jhat.simple_jwst_phot.hst_photclass`` assumes ACS/WFC3-style headers
(``FILTER1`` is a string; filenames are flt/flc/drz/drc only). WFPC2 calibrated
products use ``FILTNAM1``/``FILTNAM2`` and ``*_c0m.fits``, and often store a
numeric ``FILTER1`` that crashes ``'CLEAR' not in FILTER1``.

Also patches JHAT's encircled-energy helper: SciPy ≥1.14 removed ``interp2d``,
which upstream ``hst_get_ee_corr`` still calls.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from st123.utils.logging import capture_output

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

    Also installs SciPy ``interp2d`` / EE-correction shims needed on SciPy ≥1.14
    and a tolerant HST ``match_refcat`` retry for poor initial WCS.
    """
    install_scipy_interp2d_compat()
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

    Upstream JHAT renames ``{stem}_tweakregstep.fits`` → ``{stem}_jhat.fits``, but
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
    Median sky translation (ΔRA, ΔDec) in degrees from *wcs_before* → *wcs_after*.

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
    """Add (ΔRA, ΔDec) in degrees to ``CRVAL`` of selected SCI HDUs. Returns count."""
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
HST_INTERNAL_ALIGN_MAX_ARCSEC = 0.08
# Level-3 coadds (different filters / instruments) should agree to this.
HST_L3_ALIGN_MAX_ARCSEC = 0.12
# Absolute tie search radius for 2-D offset histograms (handles ~arcsec pipeline errors).
HST_ABS_OFFSET_MAX_ARCSEC = 5.0


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
    pairs near 0 when the true offset is ~1–5″. The histogram peak (optionally
    excluding a small core around zero) recovers the coherent shift. Returns
    ``img − ref``; add ``−dra/−ddec`` to *path_img* CRVAL to place it on *path_ref*.
    """
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
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
        with fits.open(path, memmap=True) as hdul:
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
    # false-fails well-aligned coadds at ~0.3–0.5".
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
    Prefer a deep aligned frame/coadd for absolute ties (F625 L2, else F814).
    """
    jhat = Path(jhat_dir).expanduser().resolve()
    refdir = jhat.parent / 'reference'
    candidates = [
        jhat / 'iey902seq_jhat.fits',
        jhat / 'iey902shq_jhat.fits',
        refdir / 'coadd_wfc3_f625w_drc.fits',
        refdir / 'coadd_wfpc2_f814w_drz.fits',
    ]
    # Prefer F814 coadd when present — same optical band family as WFPC2 and
    # typically already locked to WFC3 for this field.
    preferred = [
        refdir / 'coadd_wfpc2_f814w_drz.fits',
        jhat / 'iey902seq_jhat.fits',
        refdir / 'coadd_wfc3_f625w_drc.fits',
    ]
    for cand in preferred + candidates:
        if cand.is_file() and cand.stat().st_size > 0:
            return cand.resolve()
    return None


def _frame_exptime(path: str | Path) -> float:
    """EXPTIME from primary or first SCI; 0.0 if missing."""
    from astropy.io import fits

    p = Path(path)
    try:
        with fits.open(p, memmap=True) as hdul:
            for hdu in hdul:
                val = hdu.header.get('EXPTIME')
                if val is not None:
                    return float(val)
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
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from astropy.wcs import WCS
    from photutils.detection import DAOStarFinder

    p = Path(path)
    with fits.open(p, memmap=True) as hdul:
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
    ``dra_arcsec`` / ``ddec_arcsec`` mean img catalogs sit east/north of ref —
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
    # img − ref in degrees (RA wrapped).
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
    from astropy.io import fits
    from astropy.stats import sigma_clip
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales

    from st123.alignment.gaia_simple import _centroid

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
        with fits.open(path, memmap=True) as hdul:
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
    for ra, dec in zip(ras, decs):
        try:
            x_i0, y_i0 = w_i.world_to_pixel_values(float(ra), float(dec))
            x_r0, y_r0 = w_r.world_to_pixel_values(float(ra), float(dec))
        except Exception:
            continue
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
    alignment chip). ``max_abs_arcsec`` is the worst chip — used for QA.
    """
    from astropy.io import fits

    chip_stats: list[dict[str, Any]] = []
    with fits.open(path_img, memmap=True) as hdul:
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
    Relative WCS offset via detections matched in the reference pixel frame.

    Projects *path_img* sources through its WCS into *path_ref* pixels and
    matches to *path_ref* detections. This is the metric that predicts
    drizzle ghosting; shared-refcat centroid QA can false-pass after
    per-frame CRPIX refine.
    """
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from astropy.wcs import WCS
    from photutils.detection import DAOStarFinder
    from scipy.spatial import cKDTree
    import astropy.units as u

    stats: dict[str, Any] = {
        'path_img': str(Path(path_img)),
        'path_ref': str(Path(path_ref)),
        'sci_order': int(sci_order),
        'method': 'pixel_match',
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
        with fits.open(path, memmap=True) as hdul:
            idxs = _sci_hdu_indices(hdul)
            if not idxs or sci_order >= len(idxs):
                return None
            ext = hdul[idxs[sci_order]]
            data = np.asarray(ext.data, dtype=float)
            wcs = WCS(ext.header, hdul, naxis=2)
        mask = ~np.isfinite(data)
        _, med, std = sigma_clipped_stats(data, mask=mask, sigma=3.0, maxiters=5)
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
    ra_i, dec_i = wi.pixel_to_world_values(xi, yi)
    xp, yp = wr.world_to_pixel_values(ra_i, dec_i)
    tree = cKDTree(np.column_stack([xr, yr]))
    dist, idx = tree.query(
        np.column_stack([np.asarray(xp), np.asarray(yp)]),
        distance_upper_bound=float(max_match_pix),
    )
    good = np.isfinite(dist) & (np.asarray(idx) < len(xr))
    n = int(np.count_nonzero(good))
    stats['n_match'] = n
    if n < int(min_matches):
        return stats
    ra_r, dec_r = wr.pixel_to_world_values(xr[idx[good]], yr[idx[good]])
    c_i = SkyCoord(np.asarray(ra_i)[good] * u.deg, np.asarray(dec_i)[good] * u.deg)
    c_r = SkyCoord(np.asarray(ra_r, dtype=float) * u.deg, np.asarray(dec_r, dtype=float) * u.deg)
    # img − ref on matched stars (same convention as other offset helpers).
    dra = (c_i.ra - c_r.ra).to(u.deg).value
    dra = (dra + 180.0) % 360.0 - 180.0
    ddec = (c_i.dec - c_r.dec).to(u.deg).value
    dra_m = float(np.median(dra))
    ddec_m = float(np.median(ddec))
    dec0 = float(np.median(c_i.dec.degree))
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
            'med_sep_arcsec': float(np.median(c_i.separation(c_r).arcsec)),
            'med_dpix': float(np.median(dist[good])),
        }
    )
    return stats


def validate_hst_group_internal_alignment(
    frames: list[str | Path],
    *,
    max_coherent_arcsec: float = HST_INTERNAL_ALIGN_MAX_ARCSEC,
    match_radius_arcsec: float = 1.0,
    min_matches: int = 12,
    refcat: str | Path | None = None,
) -> dict[str, Any]:
    """
    Pairwise internal-alignment QA for level-2 frames that will share a coadd.

    Uses **pixel-space** source matching (not shared-refcat centroids). ``ok``
    is True when every measurable SCI1 pair has coherent |Δ| ≤
    *max_coherent_arcsec*. Single-frame groups always pass.

    *refcat* is accepted for API compatibility but ignored for the pass/fail
    decision (refcat centroid QA can false-pass after per-frame CRPIX refine).
    """
    del match_radius_arcsec, refcat  # unused; kept for call-site compatibility
    paths = [Path(p).expanduser().resolve() for p in frames]
    report: dict[str, Any] = {
        'ok': True,
        'n_frames': len(paths),
        'max_coherent_arcsec': float(max_coherent_arcsec),
        'max_abs_arcsec': 0.0,
        'pairs': [],
        'failed_pairs': [],
        'method': 'pixel_match',
    }
    if len(paths) < 2:
        return report
    worst = 0.0
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            st = measure_hst_frame_relative_offset_pixel(
                paths[i],
                paths[j],
                sci_order=0,
                min_matches=min_matches,
            )
            abs_as = float(st.get('abs_arcsec') or 0.0)
            pair = {
                'a': paths[i].name,
                'b': paths[j].name,
                'ok': bool(st.get('ok')),
                'abs_arcsec': abs_as,
                'max_abs_arcsec': abs_as,
                'dra_arcsec': float(st.get('dra_arcsec') or 0.0),
                'ddec_arcsec': float(st.get('ddec_arcsec') or 0.0),
                'n_match': int(st.get('n_match') or 0),
                'med_dpix': float(st.get('med_dpix') or 0.0),
                'method': 'pixel_match',
            }
            report['pairs'].append(pair)
            if not st.get('ok'):
                report['failed_pairs'].append(pair)
                report['ok'] = False
                continue
            worst = max(worst, abs_as)
            if abs_as > float(max_coherent_arcsec):
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
        l3 / 'coadd_wfc3_f625w.phot.txt',
        l3 / 'coadd_wfc3_f625w_jhat.phot.txt',
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
    """SCI1 sky translation (degrees) from *raw_path* → *jhat_path* WCS."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(raw_path, memmap=True) as raw_hdul, fits.open(
        jhat_path, memmap=True
    ) as jhat_hdul:
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
    Absolute sky offset of *path* vs *refcat* (ref − image), SCI *sci_order*.

    Returns degrees/arcsec suitable to **add** to CRVAL to place the frame on
    the refcat frame.
    """
    import numpy as np
    import pandas as pd
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
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

    with fits.open(path, memmap=True) as hdul:
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
    # ref − img ; add to CRVAL to move image onto ref.
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
    from astropy.io import fits

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
        with fits.open(raw, memmap=True) as raw_hdul, fits.open(
            path, mode='update', memmap=False
        ) as jhat_hdul:
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
            jhat_hdul[0].header['ST123WHAR'] = (
                True,
                'st123: pipeline-relative WCS + common abs shift',
            )
            jhat_hdul[0].header['ST123HARA'] = (dra_as, '[arcsec] common dRA cos(Dec)')
            jhat_hdul[0].header['ST123HADE'] = (ddec_as, '[arcsec] common dDec')
            jhat_hdul[0].header['ST123WCHP'] = (
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
) -> dict[str, Any]:
    """
    Force relative WCS agreement among L2 frames before AstroDrizzle.

    Independent JHAT / per-chip CRPIX solutions often break the pipeline's
    intra-visit dither WCS (classic drizzle ghosting). This restores each
    frame's calibrated-raw SCI WCS (preserving dithers) and applies **one
    common** absolute sky shift measured with a 2-D offset histogram vs
    *abs_ref* (default: F814 coadd or F625), searching up to
    *max_abs_offset_arcsec* (default 5″). Small-radius catalog matches are
    not used for the absolute tie — they false-lock near 0″.

    Raises ``RuntimeError`` if pixel-match pairwise |Δ| remains above
    *max_internal_arcsec*.
    """
    import numpy as np
    from astropy.io import fits

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

    report['pre'] = validate_hst_group_internal_alignment(paths, **qa_kw)

    if anchor is not None:
        anchor_path = Path(anchor).expanduser().resolve()
    else:
        anchor_path = max(paths, key=_frame_exptime)
    if anchor_path not in paths:
        raise FileNotFoundError(f'anchor {anchor_path} not in group')
    report['anchor'] = str(anchor_path)

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
        if probe.get('ok') and float(probe['abs_arcsec']) > 0.12:
            need_abs = True
            log.info(
                'Absolute probe %s vs %s: |Δ|=%.3f" (dRA=%+.3f dDec=%+.3f) → retie',
                anchor_path.name,
                abs_ref_path.name,
                probe['abs_arcsec'],
                probe['dra_arcsec'],
                probe['ddec_arcsec'],
            )

    if report['pre']['ok'] and not need_abs:
        report['post'] = report['pre']
        log.info(
            'Group already aligned (internal max |Δ|=%.3f"; abs OK) — skip',
            report['pre']['max_abs_arcsec'],
        )
        return report

    raw_siblings = {p: _raw_sibling_for_jhat(p) for p in paths}
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
            with fits.open(path, mode='update', memmap=False) as hdul:
                apply_sky_translation_to_sci(
                    hdul,
                    -float(off['dra_deg']),
                    -float(off['ddec_deg']),
                    comment='st123: sibling relative harmonize',
                )
                hdul[0].header['ST123WHAR'] = True
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
        # 2) Absolute via 2-D histogram vs abs_ref (handles up to ~5″).
        dra_deg = ddec_deg = 0.0
        if abs_ref_path is not None and abs_ref_path.is_file():
            abs_off = measure_hst_sky_offset_2dhist(
                anchor_path,
                abs_ref_path,
                max_offset_arcsec=max_abs_offset_arcsec,
            )
            if abs_off.get('ok'):
                # img−ref → apply −Δ to CRVAL.
                dra_deg = -float(abs_off['dra_deg'])
                ddec_deg = -float(abs_off['ddec_deg'])
                report['method'] = 'pipeline_relative_2dhist_abs'
                log.info(
                    '2D-hist abs %s → %s: img-ref dRA=%+.3f" dDec=%+.3f" '
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
                with fits.open(path, mode='update', memmap=False) as hdul:
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
                    hdul[0].header['ST123HARA'] = (
                        dra_as,
                        '[arcsec] common dRA cos(Dec)',
                    )
                    hdul[0].header['ST123HADE'] = (
                        ddec_as,
                        '[arcsec] common dDec',
                    )
                    if abs_ref_path is not None:
                        hdul[0].header['ST123HANC'] = (
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
        raise RuntimeError(
            'HST group internal alignment failed after pipeline-relative restore '
            f'(max |Δ|={report["post"]["max_abs_arcsec"]:.3f}" > '
            f'{max_internal_arcsec:.3f}"; '
            f'failed={report["post"]["failed_pairs"]})'
        )
    log.info(
        'Group harmonize OK: max |Δ| %.3f" → %.3f" (limit %.3f")',
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
    frames = sorted(
        p
        for p in jhat.glob(pattern)
        if not p.name.lower().startswith('coadd_') and 'l3_ref' not in p.parts
    )
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for p in frames:
        try:
            inst = get_instrument(p).split('_')[0].lower()
            filt = get_filter(p).lower()
        except Exception as exc:
            log.warning('Skipping ungroupable frame %s (%s)', p, exc)
            continue
        groups[(inst, filt)].append(p.resolve())
    results: list[dict[str, Any]] = []
    for (inst, filt), imgs in sorted(groups.items()):
        try:
            harm = harmonize_hst_group_wcs(
                sorted(imgs),
                max_internal_arcsec=max_internal_arcsec,
                abs_ref=abs_ref_path,
            )
            results.append(
                {
                    'instrument': inst,
                    'filter': filt,
                    'status': 'ok' if harm.get('ok') else 'failed',
                    'harmonize': harm,
                }
            )
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
    relative to *source*, then applies the same ΔCRVAL to every SCI that still
    matches *source* (typically SCI2+). Already-updated chips are left as JHAT
    wrote them.

    Returns a stats dict (``dra_deg``, ``ddec_deg``, ``n_updated``, …).
    """
    import numpy as np
    from astropy.io import fits
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

    with fits.open(source_path, memmap=True) as src_hdul:
        src_idxs = _sci_hdu_indices(src_hdul)
        if len(src_idxs) < 2:
            stats['n_sci'] = len(src_idxs)
            return stats  # single-SCI: nothing to propagate

        with fits.open(aligned_path, mode='update', memmap=False) as aln_hdul:
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
            aln_hdul[0].header['ST123WPROP'] = (
                True,
                'st123: JHAT sky tweak propagated to all SCI',
            )
            dec0 = float(aln_hdul[aln_idxs[0]].header.get('CRVAL2', 0.0))
            dra_as = float(dra * 3600.0 * np.cos(np.radians(dec0)))
            ddec_as = float(ddec * 3600.0)
            aln_hdul[0].header['ST123WDRA'] = (dra_as, '[arcsec] multi-SCI dRA cos(Dec)')
            aln_hdul[0].header['ST123WDDE'] = (ddec_as, '[arcsec] multi-SCI dDec')
            aln_hdul[0].header['ST123WNUP'] = (int(n_upd), 'SCI extensions updated')
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
    from astropy.io import fits
    from astropy.stats import sigma_clip
    from astropy.table import Table
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales

    from st123.alignment.gaia_simple import _centroid

    aligned_path = Path(aligned).expanduser().resolve()
    ref_path = Path(refcat).expanduser().resolve()
    stats: dict[str, Any] = {
        'path': str(aligned_path),
        'n_sci': 0,
        'n_updated': 0,
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

    with fits.open(aligned_path, mode='update', memmap=False) as hdul:
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
            for row in ref[:4000]:
                ra = float(row['ra'])
                dec = float(row['dec'])
                try:
                    x_pred, y_pred = w.world_to_pixel_values(ra, dec)
                except Exception:
                    continue
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
        if stats['n_updated']:
            hdul[0].header['ST123WCHP'] = (
                True,
                'st123: per-SCI CRPIX refine vs refcat',
            )
            hdul[0].header['ST123WCNU'] = (
                int(stats['n_updated']),
                'SCI chips CRPIX-refined',
            )
            hdul.flush()
    if stats['n_updated']:
        log.info(
            'Per-chip refine %s: updated %d/%d SCI',
            aligned_path.name,
            stats['n_updated'],
            stats['n_sci'],
        )
    return stats


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
    sky residual to **all** SCI extensions when |Δ| is between a noise floor
    and *max_residual_arcsec* (rejects gross mismatches).
    """
    import numpy as np
    import pandas as pd
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
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
    # JHAT phot next to product: iey902seq_jhat.fits → iey902seq.phot.txt
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

    with fits.open(aligned_path, mode='update', memmap=False) as hdul:
        idxs = _sci_hdu_indices(hdul)
        if not idxs:
            return stats
        # JHAT phot x,y are in the primary/first-SCI (or do_driz) frame — not
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
        # Image → ref residual; apply to CRVAL so image moves onto ref.
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
        hdul[0].header['ST123WREF'] = (
            True,
            'st123: residual CRVAL refine vs refcat',
        )
        hdul[0].header['ST123WRAS'] = (dra_as, '[arcsec] refine dRA cos(Dec)')
        hdul[0].header['ST123WRDE'] = (ddec_as, '[arcsec] refine dDec')
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
    2. Optional global CRVAL refine from the JHAT phot table vs *photfilename*.
    3. Optional per-SCI CRPIX refine by centroiding *photfilename* sources on
       each chip (puts WF chips on the reference frame).

    Returns
    -------
    pathlib.Path
        Path to the aligned ``*_jhat.fits`` product.
    """
    try:
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
    # Instance-level histocut caps (not formal run_all kwargs) — set before run.
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
        run_kwargs = dict(
            outrootdir=outroot,
            telescope='hst',
            refcatname='Gaia',
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

    run_exc: Exception | None = None
    try:
        with capture_output():
            wcs_align.run_all(str(image_path), **run_kwargs)
    except Exception as exc:
        run_exc = exc

    recovered = _recover_jhat_product(image_path, out)
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
        if refine_refcat and photfilename is not None:
            try:
                refine_hst_wcs_from_refcat(recovered, photfilename)
            except Exception as exc:
                log.warning(
                    'refcat WCS refine failed for %s: %s', recovered.name, exc
                )
        if refine_per_chip and photfilename is not None:
            try:
                refine_hst_wcs_per_chip_from_refcat(recovered, photfilename)
            except Exception as exc:
                log.warning(
                    'per-chip WCS refine failed for %s: %s', recovered.name, exc
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
    # coadd_wfc3_f625w_drc → also try coadd_wfc3_f625w
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
) -> list[dict]:
    """
    Align all matching science frames under *raw_dir*.

    Skips ``*c1m.fits`` (DQ companions). When *photfilename* is set, frames are
    aligned to that catalog (``gaia`` is ignored). Returns per-file status dicts
    with keys ``path``, ``status``, ``error``, ``outpath``.
    """
    raw = Path(raw_dir).expanduser().resolve()
    out = Path(jhat_outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref_phot = Path(photfilename).expanduser().resolve() if photfilename else None
    use_gaia = bool(gaia) and ref_phot is None

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

    results: list[dict] = []
    for frame in frames:
        entry: dict = {
            'path': str(frame),
            'status': 'pending',
            'error': None,
            'outpath': None,
        }
        try:
            outpath = align_hst_image(
                frame,
                out,
                gaia=use_gaia,
                photfilename=str(ref_phot) if ref_phot is not None else None,
                verbose=verbose,
                jhat_params=jhat_params,
            )
            entry['status'] = 'ok'
            entry['outpath'] = str(outpath)
        except Exception as exc:
            entry['status'] = 'failed'
            entry['error'] = f'{type(exc).__name__}: {exc}'
            log.error('HST JHAT failed for %s: %s', frame.name, exc)
            if not soft_fail:
                results.append(entry)
                raise
        results.append(entry)

    # Relative harmonize within each filter group so coadds are not ghosted
    # by inconsistent per-exposure JHAT solutions.
    n_ok = sum(1 for r in results if r.get('status') == 'ok' and r.get('outpath'))
    if n_ok >= 2:
        try:
            harm_rows = harmonize_hst_jhat_dir(out)
            for row in harm_rows:
                if row.get('status') != 'ok':
                    log.error(
                        'Post-align group harmonize failed for %s/%s: %s',
                        row.get('instrument'),
                        row.get('filter'),
                        row.get('error'),
                    )
                    if not soft_fail:
                        raise RuntimeError(row.get('error') or 'harmonize failed')
        except Exception as exc:
            log.error('Post-align group harmonize failed: %s', exc)
            if not soft_fail:
                raise
    return results


def write_hst_alignment_summary(
    results: list[dict],
    summary_path: str | Path,
) -> Path:
    """Write ``alignment_summary.json`` for an HST JHAT batch."""
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
    path.write_text(json.dumps(payload, indent=2) + '\n')
    return path
