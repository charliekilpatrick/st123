"""JWST NIRCam science-product datamodel."""

from __future__ import annotations

from pathlib import Path

from st123.datamodels.jwst.jwst import JWSTDataModel

__all__ = ['NIRCamDataModel']

# Pupil-wheel tokens that are not a photometric bandpass.
_NON_SCIENCE_PUPILS = frozenset({
    '',
    'CLEAR',
    'CLEARP',
    'NONE',
    'N/A',
    'FND',
    'WLP4',
    'WLP8',
    'WLM8',
    'MASKRND',
    'MASKBAR',
    'MASKIPR',
    'GRISMR',
    'GRISMC',
})


class NIRCamDataModel(JWSTDataModel):
    """NIRCam calibrated or i2d / JHAT product."""

    instrument = 'NIRCAM'
    FILTERS: tuple[str, ...] = (
        'F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F150W2', 'F162M', 'F164N',
        'F182M', 'F187N', 'F200W', 'F210M', 'F212N',
        'F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 'F335M', 'F356W', 'F360M',
        'F405N', 'F410M', 'F430M', 'F444W', 'F460M', 'F466N', 'F470N', 'F480M',
        'CLEAR',
    )
    FILTER_HEADER_KEYS: tuple[str, ...] = ('FILTER', 'FILTER1', 'FILTER2', 'PUPIL')
    MOSAIC_FILTERS: tuple[str, ...] = (
        'F150W', 'F150W2', 'F164N', 'F187N', 'F200W', 'F300M', 'F335M', 'F360M',
        'F405N', 'F430M', 'F444W',
    )
    DOLPHOT_SHORT_PARAMS: dict[str, str] = {
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
    DOLPHOT_LONG_PARAMS: dict[str, str] = {
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
    CALCSKY_PARAMS: dict = {
        'rin': 15,
        'rout': 25,
        'step': -64,
        'sigma_low': 2.25,
        'sigma_high': 2.00,
    }

    @classmethod
    def science_filter(
        cls,
        filter_value: object | None,
        pupil_value: object | None = None,
    ) -> str:
        """
        Photometric bandpass from ``FILTER`` + ``PUPIL``.

        Dual-filter NIRCam visits put a blocking wide filter in ``FILTER``
        (``F444W``, ``F150W2``, ``F322W2``) and the science bandpass in
        ``PUPIL`` (``F405N``, ``F164N``, ...). Mosaic / coadd names and
        ``filter_name`` use that pupil-wheel bandpass. ``CLEAR`` (and other
        non-photometric pupil elements) fall back to ``FILTER``.
        """
        pupil = str(pupil_value or '').strip().upper()
        if pupil and pupil not in _NON_SCIENCE_PUPILS and pupil in cls.FILTERS:
            return pupil.lower()
        filt = str(filter_value or '').strip()
        if filt and filt.lower() not in {'none', 'n/a', 'clear', 'clear1', 'clear2'}:
            return filt.lower()
        raise KeyError('No NIRCam science filter in FILTER/PUPIL')

    @property
    def filter_name(self) -> str:
        """Science bandpass: pupil-wheel filter when set, else ``FILTER``."""
        try:
            return self.science_filter(self.keyword('FILTER'), self.pupil)
        except KeyError:
            return super().filter_name

    @classmethod
    def matches(cls, value: object | None) -> bool:
        """True for MAST ``instrument_name``, ``INSTRUME``, or NIRCam paths."""
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        upper = text.upper()
        if 'NIRCAM' in upper:
            return True
        if upper.startswith('NRC'):
            return True
        name = Path(text).name.upper()
        return 'NRC' in name

    @property
    def image_kind(self) -> str:
        det = (self.detector or '').upper()
        if 'LONG' in det:
            return 'long'
        return 'short'

    @property
    def module_name(self) -> str:
        module = self.keyword('MODULE')
        if module is not None and str(module).strip():
            return str(module).strip().lower()
        detector = str(self.detector or self.keyword('DETECTOR') or '').strip().upper()
        if detector.startswith('NRC') and len(detector) > 3:
            return detector[3].lower()
        chip = self.filename_chip
        if chip and chip.lower().startswith('nrc') and len(chip) > 3:
            return chip.lower()[3]
        return super().module_name

    def dolphot_image_params(self) -> dict[str, str]:
        if self.image_kind == 'long':
            return dict(self.DOLPHOT_LONG_PARAMS)
        return dict(self.DOLPHOT_SHORT_PARAMS)
