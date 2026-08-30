"""Overarching on-disk science-product datamodel.

:class:`InstrumentDataModel` is the common handle for a FITS path. Telescope
and instrument subclasses override only what differs (JWST array sanitize,
HST ``EXPFLAG``). Factory :func:`open_datamodel` returns the most specific
class that matches the file.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from astropy.io import fits
from astropy.time import Time

logger = logging.getLogger(__name__)

PathLike = str | Path

__all__ = [
    'DataModelLike',
    'InstrumentDataModel',
    'as_datamodel',
    'as_datamodels',
    'classify_image_kind',
    'filter_good_frames',
    'filter_paths_for_stage',
    'materialize_fits_wcs_keywords',
    'open_datamodel',
    'path_of',
]


def materialize_fits_wcs_keywords(header: fits.Header) -> dict[str, Any]:
    """
    Write derived DATE-* / OBSGEO-L/B/H cards that Astropy otherwise invents.

    JWST SCI headers often store ``MJD-*`` and ``OBSGEO-[XYZ]``. Opening a WCS
    then emits ``FITSFixedWarning`` ``datfix`` / ``obsfix`` on every read.
    Materializing the derived keywords once silences those warnings without
    changing the sky WCS.

    Returns a small report of cards written.
    """
    written: dict[str, Any] = {'date': [], 'obsgeo': []}

    for mjd_key, date_key in (
        ('MJD-BEG', 'DATE-BEG'),
        ('MJD-AVG', 'DATE-AVG'),
        ('MJD-END', 'DATE-END'),
    ):
        if mjd_key not in header:
            continue
        if date_key in header and header.get(date_key):
            continue
        try:
            mjd = float(header[mjd_key])
            if not np.isfinite(mjd):
                continue
            header[date_key] = (
                Time(mjd, format='mjd', scale='utc').isot,
                f'converted from {mjd_key} by st123',
            )
            written['date'].append(date_key)
        except Exception:
            continue

    need_lbh = any(
        k not in header or header.get(k) is None
        for k in ('OBSGEO-L', 'OBSGEO-B', 'OBSGEO-H')
    )
    if need_lbh and all(k in header for k in ('OBSGEO-X', 'OBSGEO-Y', 'OBSGEO-Z')):
        try:
            import warnings

            from astropy.wcs import FITSFixedWarning, WCS

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FITSFixedWarning)
                wcs_obj = WCS(header, relax=True, naxis=2)
            og = getattr(wcs_obj.wcs, 'obsgeo', None)
            if og is not None and len(og) >= 6 and all(
                np.isfinite(float(og[i])) for i in range(6)
            ):
                header['OBSGEO-L'] = (
                    float(og[3]),
                    '[deg] lon from OBSGEO-XYZ (st123)',
                )
                header['OBSGEO-B'] = (
                    float(og[4]),
                    '[deg] lat from OBSGEO-XYZ (st123)',
                )
                header['OBSGEO-H'] = (
                    float(og[5]),
                    '[m] height from OBSGEO-XYZ (st123)',
                )
                written['obsgeo'] = ['OBSGEO-L', 'OBSGEO-B', 'OBSGEO-H']
        except Exception:
            pass

    return written


def _filename_chip(path: PathLike) -> str | None:
    """Detector token from a JWST-style filename (``nrcb1``, ``mirimage``)."""
    tokens = Path(path).name.split('_')
    for token in tokens:
        if 'nrc' in token.lower():
            return token
    for token in tokens:
        if 'mirimage' in token.lower():
            return token
    return None


@dataclass(frozen=True)
class FileIdentity:
    """Header / filename cues used to pick a datamodel class."""

    telescope: str = ''
    instrument: str = ''
    detector: str = ''
    aperture: str = ''
    photmode: str = ''


def read_file_identity(path: PathLike) -> FileIdentity:
    """Read telescope / instrument / detector cues without constructing a model."""
    p = Path(path)
    telescope = ''
    instrument = ''
    detector = ''
    aperture = ''
    photmode = ''
    if p.is_file():
        try:
            with fits.open(p, memmap=True) as hdul:
                for hdu in hdul:
                    hdr = hdu.header
                    if not telescope:
                        telescope = str(hdr.get('TELESCOP') or '').strip().upper()
                    if not instrument:
                        inst = hdr.get('INSTRUME') or hdr.get('INSTRUMENT')
                        if inst is not None and str(inst).strip():
                            instrument = str(inst).strip().upper()
                    if not detector:
                        det = hdr.get('DETECTOR')
                        if det is not None and str(det).strip():
                            detector = str(det).strip().upper()
                    if not photmode:
                        photmode = str(hdr.get('PHOTMODE') or '').strip().upper()
                if hdul:
                    aperture = str(hdul[0].header.get('APERTURE') or '').strip().upper()
                    if not photmode:
                        photmode = str(
                            hdul[0].header.get('PHOTMODE') or ''
                        ).strip().upper()
                if not instrument and photmode:
                    token = photmode.replace(',', ' ').split()[0]
                    if token:
                        instrument = token.upper()
                if not instrument:
                    if aperture.startswith('UVIS') or aperture.startswith('IR'):
                        instrument = 'WFC3'
                    elif aperture.startswith('WFC') or aperture.startswith('HRC'):
                        instrument = 'ACS'
        except Exception:
            pass
    return FileIdentity(
        telescope=telescope,
        instrument=instrument,
        detector=detector,
        aperture=aperture,
        photmode=photmode,
    )


def _select_class(path: PathLike, ident: FileIdentity) -> type[InstrumentDataModel]:
    """Return the most specific datamodel class for *ident* / filename."""
    from st123.datamodels.hst.acs_hrc import ACSHRCDataModel
    from st123.datamodels.hst.acs_wfc import ACSWFCDataModel
    from st123.datamodels.hst.hst import HSTDataModel
    from st123.datamodels.hst.wfc3_ir import WFC3IRDataModel
    from st123.datamodels.hst.wfc3_uvis import WFC3UVISDataModel
    from st123.datamodels.hst.wfpc2 import WFPC2DataModel
    from st123.datamodels.jwst.jwst import JWSTDataModel
    from st123.datamodels.jwst.miri import MIRIDataModel
    from st123.datamodels.jwst.nircam import NIRCamDataModel

    name = Path(path).name.lower()
    inst = ident.instrument
    tele = ident.telescope
    det = ident.detector
    aper = ident.aperture
    phot = ident.photmode

    chip = _filename_chip(path)
    chip_l = chip.lower() if chip else ''

    if inst == 'NIRCAM' or 'nrc' in chip_l:
        return NIRCamDataModel
    if inst == 'MIRI' or 'mir' in chip_l or 'mirimage' in name or '/miri/' in str(path).lower():
        return MIRIDataModel
    if inst in ('NIRISS', 'NIRSPEC'):
        return JWSTDataModel

    if inst == 'WFPC2' or name.endswith(('c0m.fits', 'c1m.fits')):
        return WFPC2DataModel
    if inst == 'ACS':
        if (
            det == 'HRC'
            or det.startswith('HRC')
            or aper.startswith('HRC')
            or 'HRC' in phot
        ):
            return ACSHRCDataModel
        if det == 'SBC' or aper.startswith('SBC') or 'SBC' in phot:
            return HSTDataModel
        if (
            det.startswith('WFC')
            or aper.startswith('WFC')
            or 'ACS/WFC' in phot
            or name.endswith('_flc.fits')
        ):
            return ACSWFCDataModel
        if name.endswith('_flt.fits'):
            return ACSHRCDataModel
        return HSTDataModel
    if inst == 'WFC3':
        if WFC3IRDataModel.identity_is_ir(ident, name):
            return WFC3IRDataModel
        return WFC3UVISDataModel
    if inst not in ('ACS', 'WFPC2', 'NIRCAM', 'MIRI') and WFC3IRDataModel.identity_is_ir(
        ident, name
    ):
        return WFC3IRDataModel

    if tele == 'JWST':
        return JWSTDataModel
    if tele == 'HST' or name.endswith(('_flc.fits', '_flt.fits', '_c0m.fits', '_jhat.fits')):
        return HSTDataModel
    return InstrumentDataModel


def open_datamodel(path: PathLike) -> InstrumentDataModel:
    """Construct the most specific datamodel for *path*."""
    ident = read_file_identity(path)
    cls = _select_class(path, ident)
    return cls.from_path(path, identity=ident)


def as_datamodel(image: InstrumentDataModel | PathLike) -> InstrumentDataModel:
    """Return *image* if it is already a datamodel, otherwise :func:`open_datamodel`."""
    if isinstance(image, InstrumentDataModel):
        return image
    return open_datamodel(image)


def as_datamodels(
    images: Iterable[InstrumentDataModel | PathLike],
) -> list[InstrumentDataModel]:
    """Coerce each entry of *images* with :func:`as_datamodel`."""
    return [as_datamodel(image) for image in images]


def path_of(image: InstrumentDataModel | PathLike) -> Path:
    """On-disk science path for a datamodel or path-like value."""
    if isinstance(image, InstrumentDataModel):
        return Path(image.path)
    return Path(image)


def classify_image_kind(image: InstrumentDataModel | PathLike) -> str:
    """
    Classify a FITS frame for per-image DOLPHOT parameters.

    Filename chip tokens (``nrcb1``, ``nrcblong``, ``mirimage``) win over
    headers, matching historic DOLPHOT prep behavior.

    Returns
    -------
    str
        One of ``'short'`` / ``'long'`` (NIRCam), ``'miri'``, ``'acs'``,
        ``'wfc3'``, ``'wfc3_ir'``, or ``'wfpc2'``.

    Raises
    ------
    ValueError
        If the frame cannot be classified from the filename or FITS headers.
    """
    path = path_of(image)
    name = path.name.lower()
    chip = _filename_chip(path)
    if chip:
        chip_l = chip.lower()
        if 'mir' in chip_l:
            return 'miri'
        if 'long' in chip_l:
            return 'long'
        if 'nrc' in chip_l:
            return 'short'
    if 'mirimage' in name or '/miri/' in str(path).lower():
        return 'miri'
    model = image if isinstance(image, InstrumentDataModel) else open_datamodel(path)
    kind = model.image_kind
    if kind:
        return kind
    raise ValueError(f'Cannot classify DOLPHOT image kind for {path}')


class InstrumentDataModel:
    """
    Path-centric science FITS handle.

    Mosaic, align, and dolphot-prep work on files. Vendor JWST/HST datamodels
    remain available when a step needs them. :meth:`sanitize` enforces the
    st123 array/header contract in place.
    """

    telescope: str = ''
    instrument: str = ''
    detector: str = ''

    def __init__(
        self,
        path: PathLike,
        *,
        telescope: str = '',
        instrument: str = '',
        detector: str = '',
        aperture: str = '',
        photmode: str = '',
    ) -> None:
        self.path = Path(path)
        self.telescope = (telescope or type(self).telescope).upper()
        self.instrument = (instrument or type(self).instrument).upper()
        self.detector = (detector or type(self).detector).upper()
        self.aperture = (aperture or '').upper()
        self.photmode = (photmode or '').upper()
        self._sanitized = False

    @classmethod
    def from_path(
        cls,
        path: PathLike,
        identity: FileIdentity | None = None,
    ) -> InstrumentDataModel:
        """Build a handle from a FITS path.

        Calling this on :class:`InstrumentDataModel` dispatches through
        :func:`open_datamodel`. Concrete subclasses construct themselves.
        """
        if cls is InstrumentDataModel and identity is None:
            return open_datamodel(path)
        ident = identity if identity is not None else read_file_identity(path)
        return cls(
            path,
            telescope=ident.telescope or cls.telescope,
            instrument=ident.instrument or cls.instrument,
            detector=ident.detector or cls.detector,
            aperture=ident.aperture,
            photmode=ident.photmode,
        )

    @property
    def is_jwst(self) -> bool:
        return self.telescope == 'JWST' or type(self).telescope == 'JWST'

    @property
    def is_hst(self) -> bool:
        return self.telescope == 'HST' or type(self).telescope == 'HST'

    @property
    def image_kind(self) -> str:
        """DOLPHOT per-image kind, or ``''`` when unknown at this class."""
        return ''

    def is_science_product(self) -> bool:
        """True when *path* looks like a science MEF for this instrument."""
        return self.path.suffix.lower() in {'.fits', '.fit'} and not self.path.name.lower().endswith(
            '.sky.fits'
        )

    def is_good_exposure(self, *, missing_ok: bool = True) -> bool:
        """Exposure-quality gate. Default: accept (JWST has no ``EXPFLAG``)."""
        return True

    def is_good_alignment(
        self,
        *,
        max_internal_arcsec: float | None = None,
        max_abs_arcsec: float | None = None,
        missing_ok: bool = True,
    ) -> bool:
        """Optional residual-quality gate. Default: accept."""
        return True

    def is_good(
        self,
        *,
        missing_ok: bool = True,
        require_expflag: bool = True,
        max_internal_arcsec: float | None = None,
        max_abs_arcsec: float | None = None,
    ) -> bool:
        """Combined exposure + optional alignment gate used by stages."""
        if require_expflag and not self.is_good_exposure(missing_ok=missing_ok):
            return False
        if (
            max_internal_arcsec is not None or max_abs_arcsec is not None
        ) and not self.is_good_alignment(
            max_internal_arcsec=max_internal_arcsec,
            max_abs_arcsec=max_abs_arcsec,
            missing_ok=missing_ok,
        ):
            return False
        return True

    def reject_detail(self) -> str:
        """Short quality-keyword summary for log lines, or ``''``."""
        return ''

    def open(
        self,
        *,
        mode: str = 'readonly',
        memmap: bool = False,
        sanitize: bool = True,
    ) -> fits.HDUList:
        """
        Open this product's science FITS (same contract as ``fits.open``).

        When *sanitize* is True (default), :meth:`sanitize` runs first so the
        on-disk SCI/ERR/DQ/WCS contract is in place. Use as a context manager
        or close the returned handle.
        """
        if sanitize:
            self.sanitize()
        return fits.open(self.path, mode=mode, memmap=memmap)

    def header(self, ext: int | str = 0) -> fits.Header:
        """Copy of the header at *ext* (sanitize runs on open)."""
        with self.open() as hdul:
            return hdul[ext].header.copy()

    def _sci_hdu(self, hdul: fits.HDUList) -> fits.ImageHDU | fits.PrimaryHDU:
        if 'SCI' in hdul:
            return hdul['SCI']
        for hdu in hdul:
            data = getattr(hdu, 'data', None)
            if data is not None and getattr(data, 'ndim', 0) >= 2:
                return hdu
        if hdul:
            return hdul[0]
        raise KeyError(f'No image HDU in {self.path}')

    def keyword(self, key: str, default: Any = None, *, ext: int | str | None = None) -> Any:
        """
        Read a header keyword from this datamodel.

        When *ext* is None, search the primary then every extension (same
        spirit as ``fits.getval``). Missing keys return *default*.
        """
        try:
            with self.open() as hdul:
                if not hdul:
                    return default
                if ext is not None:
                    hdr = hdul[ext].header
                    return hdr[key] if key in hdr else default
                for hdu in hdul:
                    if key in hdu.header:
                        return hdu.header[key]
        except Exception:
            return default
        return default

    def _looks_like_filter_name(self, value: object) -> bool:
        text = str(value).strip()
        if not text or text.lower() in {'none', 'n/a', 'clear', 'clear1', 'clear2'}:
            return False
        try:
            float(text)
            return False
        except ValueError:
            return True

    def _filter_from_photmode(self, photmode: object) -> str | None:
        text = str(photmode or '')
        match = re.search(r'\b(F\d{3,}[A-Z0-9]*)\b', text, flags=re.IGNORECASE)
        return match.group(1).lower() if match else None

    @property
    def filter_name(self) -> str:
        """Lowercase bandpass (``FILTNAM1`` / ``FILTER`` / ``FILTER1``+``FILTER2`` / ``PHOTMODE``)."""
        with self.open() as hdul:
            if not hdul:
                raise KeyError(f'No filter keyword found in {self.path}')
            for key in ('FILTNAM1', 'FILTER'):
                for hdu in hdul:
                    if key not in hdu.header:
                        continue
                    val = hdu.header[key]
                    if self._looks_like_filter_name(val):
                        return str(val).strip().lower()
            f1 = None
            for hdu in hdul:
                if 'FILTER1' in hdu.header:
                    f1 = str(hdu.header['FILTER1'])
                    break
            if f1 is not None:
                filt = f1
                if 'clear' in filt.lower():
                    for hdu in hdul:
                        if 'FILTER2' in hdu.header:
                            filt = str(hdu.header['FILTER2'])
                            break
                if self._looks_like_filter_name(filt):
                    return filt.strip().lower()
            for hdu in hdul:
                filt = self._filter_from_photmode(hdu.header.get('PHOTMODE'))
                if filt:
                    return filt
        raise KeyError(f'No filter keyword found in {self.path}')

    @property
    def instrument_name(self) -> str:
        """Lowercase instrument id from headers, else class identity."""
        val = self.keyword('INSTRUME')
        if val is not None and str(val).strip():
            return str(val).strip().lower()
        val = self.keyword('INSTRUMENT')
        if val is not None and str(val).strip():
            return str(val).strip().lower()
        if self.photmode:
            token = self.photmode.replace(',', ' ').split()[0]
            if token:
                return token.lower()
        aper = (self.aperture or '').upper()
        if aper.startswith('UVIS') or aper.startswith('IR'):
            return 'wfc3'
        if aper.startswith('WFC') or aper.startswith('HRC'):
            return 'acs'
        if self.instrument:
            return self.instrument.lower()
        raise KeyError(f'No instrument keyword found in {self.path}')

    @property
    def module_name(self) -> str:
        """JWST module (``a`` / ``b`` / ``miri``) or ``unknown``."""
        module = self.keyword('MODULE')
        if module is not None and str(module).strip():
            return str(module).strip().lower()
        detector = str(self.detector or self.keyword('DETECTOR') or '').strip().upper()
        if detector.startswith('NRC') and len(detector) > 3:
            return detector[3].lower()
        inst = (self.instrument or '').lower()
        if not inst:
            raw = self.keyword('INSTRUME')
            inst = str(raw).strip().lower() if raw else ''
        if 'miri' in inst or detector.startswith('MIR'):
            return 'miri'
        chip = _filename_chip(self.path)
        if chip:
            chip_l = chip.lower()
            if chip_l.startswith('nrc') and len(chip_l) > 3:
                return chip_l[3]
            if 'mir' in chip_l:
                return 'miri'
        return 'unknown'

    @property
    def chip(self) -> int | str:
        """``CCDCHIP`` or ``DETECTOR``; ``1`` when neither is present."""
        found = None
        try:
            with self.open() as hdul:
                for hdu in hdul:
                    if 'CCDCHIP' in hdu.header:
                        found = 1 if found is not None else hdu.header['CCDCHIP']
                    elif 'DETECTOR' in hdu.header:
                        found = 1 if found is not None else hdu.header['DETECTOR']
        except Exception:
            found = None
        return found if found is not None else 1

    @property
    def filename_chip(self) -> str | None:
        """Detector token from the file name (``nrcb1``, ``mirimage``)."""
        return _filename_chip(self.path)

    def sci_data(self) -> np.ndarray | None:
        """Science array, or ``None`` when missing."""
        try:
            with self.open() as hdul:
                return np.asarray(self._sci_hdu(hdul).data)
        except Exception:
            return None

    def sci_wcs(self, *, naxis: int = 2):
        """Astropy WCS for the science extension (SIP / TAN / header distortion)."""
        import warnings

        from astropy.wcs import FITSFixedWarning, WCS

        with self.open() as hdul:
            hdu = self._sci_hdu(hdul)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FITSFixedWarning)
                try:
                    return WCS(hdu.header, fobj=hdul, naxis=naxis, relax=True)
                except Exception:
                    return WCS(hdu.header, naxis=naxis, relax=True)

    def zeropoint(
        self,
        *,
        chip: int | str | None = None,
        zptype: str = 'abmag',
    ) -> float | None:
        """Photometric zero point from ``PHOTFLAM`` / ``PHOTPLAM``."""
        ccdchip = self.chip if chip is None else chip
        inst = (self.instrument or '').lower()
        with self.open() as hdul:
            sci = [
                h for h in hdul
                if 'PHOTPLAM' in h.header and 'PHOTFLAM' in h.header
            ]
            use_hdu = None
            if len(sci) == 1:
                use_hdu = sci[0]
            elif len(sci) > 1:
                for h in sci:
                    if 'acs' in inst or 'wfc3' in inst:
                        if h.header.get('CCDCHIP') == ccdchip:
                            use_hdu = h
                            break
                    elif h.header.get('DETECTOR') == ccdchip:
                        use_hdu = h
                        break
            if use_hdu is None:
                return None
            photplam = float(use_hdu.header['PHOTPLAM'])
            photflam = float(use_hdu.header['PHOTFLAM'])
        if 'ab' in zptype:
            return -2.5 * np.log10(photflam) - 5 * np.log10(photplam) - 2.408
        if 'st' in zptype:
            return -2.5 * np.log10(photflam) - 21.1
        return None

    @staticmethod
    def _wavelength_um_from_header(header: fits.Header) -> float | None:
        """Pivot / effective wavelength in microns from one header."""
        for key in ('PHOTPLAM', 'WAVELEN', 'RESTWAV', 'WAVECENT'):
            if key not in header:
                continue
            try:
                wave = float(header[key])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(wave) or wave <= 0.0:
                continue
            # PHOTPLAM is Angstroms on HST and JWST science frames. Other
            # cards use Angstroms when the value is large, else microns.
            if key == 'PHOTPLAM' or wave > 100.0:
                return wave / 1.0e4
            return wave
        return None

    @property
    def wavelength_um(self) -> float:
        """
        Pivot / effective wavelength in microns from photometric headers.

        Prefers ``PHOTPLAM`` on the science extension (Angstroms on HST and
        JWST imaging). Falls back to ``WAVELEN`` / ``RESTWAV`` / ``WAVECENT``
        when those cards are present. Missing or unusable values return
        ``inf`` so callers can rank unknown frames last.
        """
        try:
            with self.open() as hdul:
                ordered: list[Any] = []
                try:
                    ordered.append(self._sci_hdu(hdul))
                except Exception:
                    pass
                for hdu in hdul:
                    if hdu not in ordered:
                        ordered.append(hdu)
                for hdu in ordered:
                    hdr = getattr(hdu, 'header', None)
                    if hdr is None:
                        continue
                    wave = self._wavelength_um_from_header(hdr)
                    if wave is not None:
                        return float(wave)
        except Exception:
            pass
        return float('inf')

    def is_full_frame(self) -> bool:
        """Instrument-specific full-frame check; base is False."""
        return False

    @property
    def exptime(self) -> float | None:
        """Effective exposure time (``EFFEXPTM`` / ``EXPTIME`` / ``TEXPTIME``)."""
        for key in ('EFFEXPTM', 'EXPTIME', 'TEXPTIME'):
            val = self.keyword(key)
            if val is None:
                continue
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
        return None

    @property
    def visit_id(self) -> str:
        """JWST ``VISIT_ID``, HST ``ROOTNAME`` visit stem, else ``ASN_ID`` / filename."""
        for key in ('VISIT_ID', 'ROOTNAME', 'ASN_ID'):
            val = self.keyword(key)
            if val is None or str(val).strip() == '':
                continue
            token = str(val).strip()
            if key == 'ROOTNAME' and len(token) >= 6:
                return token[:6].lower()
            return token
        return self.path.name[:6].lower()

    @property
    def obs_datetime(self) -> str:
        """Observation time as ``DATE-OBS`` T ``TIME-OBS``, else ``EXPSTART`` UTC."""
        date_obs = self.keyword('DATE-OBS')
        time_obs = self.keyword('TIME-OBS')
        if date_obs is not None and time_obs is not None:
            return f'{date_obs}T{time_obs}'
        expstart = self.keyword('EXPSTART')
        if expstart is not None:
            return Time(expstart, format='mjd').datetime.strftime(
                '%Y-%m-%dT%H:%M:%S'
            )
        raise ValueError(
            f'Cannot determine observation time from headers of {self.path}'
        )

    @property
    def pupil(self) -> str:
        """``PUPIL`` wheel, or ``CLEAR`` when the keyword is absent."""
        val = self.keyword('PUPIL')
        if val is None or str(val).strip() == '':
            return 'CLEAR'
        return str(val)

    @property
    def s_region(self) -> str | None:
        """``S_REGION`` polygon string from the science (or any) header."""
        try:
            with self.open() as hdul:
                if 'SCI' in hdul:
                    val = hdul['SCI'].header.get('S_REGION')
                    if val:
                        return str(val)
                for hdu in hdul:
                    hdr = getattr(hdu, 'header', None)
                    if hdr is None:
                        continue
                    val = hdr.get('S_REGION')
                    if val:
                        return str(val)
        except Exception:
            return None
        return None

    def sci_header(self) -> fits.Header:
        """Copy of the science-extension header (WCS + distortion cards)."""
        with self.open() as hdul:
            return self._sci_hdu(hdul).header.copy()

    def sky_polygon(self):
        """Shapely sky polygon from ``S_REGION``, else the SCI WCS footprint."""
        from shapely.geometry import Polygon
        from st123.stages.download.mast import parse_s_region

        region = self.s_region
        if region:
            return parse_s_region(region)
        try:
            wcs = self.sci_wcs()
            corners = np.asarray(wcs.calc_footprint(center=False), dtype=float)
        except Exception as exc:
            raise KeyError(f'No SCI / S_REGION in {self.path}') from exc
        return Polygon(corners)

    def distortion_keywords(self) -> dict[str, Any]:
        """SIP orders and HST lookup-table filenames from science / primary headers."""
        keys = (
            'CTYPE1',
            'CTYPE2',
            'A_ORDER',
            'B_ORDER',
            'AP_ORDER',
            'BP_ORDER',
            'IDCTAB',
            'D2IMFILE',
            'NPOLFILE',
            'DGEOFILE',
        )
        out: dict[str, Any] = {}
        with self.open() as hdul:
            headers = []
            if hdul:
                headers.append(hdul[0].header)
            try:
                headers.append(self._sci_hdu(hdul).header)
            except Exception:
                pass
            for hdr in headers:
                for key in keys:
                    if key in hdr and key not in out:
                        out[key] = hdr[key]
        return out

    def matching_dq_array(
        self,
        hdul: fits.HDUList,
        sci_hdu: fits.ImageHDU | fits.PrimaryHDU,
        sci_index: int,
    ) -> np.ndarray:
        """DQ array with the same 2D shape as *sci_hdu* (same MEF by default)."""
        sci_data = getattr(sci_hdu, 'data', None)
        if sci_data is None:
            raise ValueError(f'No science array in {self.path}')
        shape = tuple(sci_data.shape)
        if 'DQ' in hdul:
            dq_hdu = hdul['DQ']
            if dq_hdu.data is not None and tuple(dq_hdu.data.shape) == shape:
                return np.asarray(dq_hdu.data)
        for hdu in hdul:
            if (
                getattr(hdu, 'name', '') == 'DQ'
                and hdu.data is not None
                and getattr(hdu.data, 'ndim', 0) == 2
                and tuple(hdu.data.shape) == shape
            ):
                return np.asarray(hdu.data)
        for offset in (2, 1, -1, 3):
            candidate_index = sci_index + offset
            if 0 <= candidate_index < len(hdul):
                candidate = hdul[candidate_index]
                if (
                    candidate.data is not None
                    and getattr(candidate.data, 'ndim', 0) == 2
                    and tuple(candidate.data.shape) == shape
                ):
                    return np.asarray(candidate.data)
        raise ValueError(
            f'No DQ extension with matching shape found in {self.path}'
        )

    def _materialize_open_headers(self, hdul: fits.HDUList) -> dict[str, Any]:
        hdr_report: dict[str, Any] = {}
        for hdu in hdul:
            if hdu.header is None:
                continue
            hdr = hdu.header
            if not any(
                k in hdr
                for k in (
                    'MJD-BEG',
                    'MJD-AVG',
                    'MJD-END',
                    'OBSGEO-X',
                    'OBSGEO-Y',
                    'OBSGEO-Z',
                )
            ):
                continue
            wrote = materialize_fits_wcs_keywords(hdr)
            if wrote.get('date') or wrote.get('obsgeo'):
                hdr_report[str(hdu.name or 'PRIMARY')] = wrote
        return hdr_report

    def _open_update(self, report: dict[str, Any]) -> fits.HDUList | None:
        """Open *path* for in-place update, or fill *report* and return None."""
        p = self.path
        if not p.is_file():
            report['ok'] = False
            report['skipped'] = True
            report['reason'] = 'missing'
            return None
        try:
            hdul = fits.open(p, mode='update', memmap=False)
        except Exception as exc:
            report['ok'] = False
            report['skipped'] = True
            report['reason'] = f'unreadable: {exc}'
            return None
        if not len(hdul):
            hdul.close()
            report['skipped'] = True
            report['reason'] = 'empty FITS'
            return None
        return hdul

    def sanitize(self, *, materialize_headers: bool = True, force: bool = False) -> dict[str, Any]:
        """
        Enforce the st123 on-disk contract.

        The base implementation only materializes DATE-* / OBSGEO-L/B/H.
        JWST subclasses also fill non-finite SCI/ERR and flag DQ/WHT.
        ``ST123SAN`` in the primary header makes later opens cheap.
        """
        report: dict[str, Any] = {
            'path': str(self.path),
            'ok': True,
            'skipped': False,
            'telescope': self.telescope,
            'instrument': self.instrument,
            'headers': {},
        }
        if not force and self._sanitized:
            report['skipped'] = True
            report['reason'] = 'already sanitized'
            return report
        hdul = self._open_update(report)
        if hdul is None:
            return report
        try:
            if not force and hdul[0].header.get('ST123SAN'):
                report['skipped'] = True
                report['reason'] = 'already sanitized'
                self._sanitized = True
                return report
            if not materialize_headers:
                report['skipped'] = True
                report['reason'] = 'header materialize disabled'
                return report
            report['headers'] = self._materialize_open_headers(hdul)
            hdul[0].header['ST123SAN'] = (
                True,
                'st123 datamodel sanitize applied (WCS cards)',
            )
            self._sanitized = True
        finally:
            hdul.close()
        return report


DataModelLike = InstrumentDataModel | PathLike


def filter_good_frames(
    paths: Sequence[PathLike],
    *,
    missing_ok: bool = True,
    require_expflag: bool = True,
    max_internal_arcsec: float | None = None,
    max_abs_arcsec: float | None = None,
) -> tuple[list[Path], list[Path]]:
    """Split *paths* into ``(kept, rejected)`` under each file's quality gate."""
    kept: list[Path] = []
    rejected: list[Path] = []
    for raw in paths:
        p = path_of(raw)
        model = as_datamodel(raw)
        if model.is_good(
            missing_ok=missing_ok,
            require_expflag=require_expflag,
            max_internal_arcsec=max_internal_arcsec,
            max_abs_arcsec=max_abs_arcsec,
        ):
            kept.append(p)
        else:
            rejected.append(p)
    return kept, rejected


def log_rejected_frames(
    rejected: Sequence[PathLike],
    *,
    stage: str,
) -> None:
    """Log each rejected path with instrument quality keywords when present."""
    for raw in rejected:
        p = path_of(raw)
        detail = as_datamodel(raw).reject_detail()
        extra = f' ({detail})' if detail else ''
        logger.warning(
            'Rejecting bad HST frame at %s: %s%s',
            stage,
            p.name,
            extra,
        )


def filter_paths_for_stage(
    paths: Iterable[PathLike],
    *,
    stage: str,
    require_expflag: bool = True,
    max_internal_arcsec: float | None = None,
    max_abs_arcsec: float | None = None,
) -> list[Path]:
    """Filter HST EXPFLAG / residual rejects; JWST and non-science names stay kept."""
    from st123.datamodels.hst.hst import filter_good_hst_frames, log_rejected_hst_frames

    kept, rejected = filter_good_hst_frames(
        list(paths),
        require_expflag=require_expflag,
        max_internal_arcsec=max_internal_arcsec,
        max_abs_arcsec=max_abs_arcsec,
    )
    if rejected:
        log_rejected_hst_frames(rejected, stage=stage)
        logger.warning(
            '%s: dropped %d bad HST frame(s); %d remain',
            stage,
            len(rejected),
            len(kept),
        )
    return kept
