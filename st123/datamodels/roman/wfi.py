"""Roman WFI science-product datamodel."""

from __future__ import annotations

import re
from pathlib import Path

from st123.datamodels.roman.roman import RomanDataModel

__all__ = ['RomanWFIDataModel']

_FILENAME_FILTER = re.compile(r'_(f\d{3}|g150|p127)_', re.IGNORECASE)


class RomanWFIDataModel(RomanDataModel):
    """
    Roman Wide Field Instrument (WFI) calibrated detector image.

    18 H4RG-10 SCAs (``WFI01``-``WFI18``), 0.11 arcsec/pix, ~0.281 deg2
    active FOV. Imaging filters F062..F213 plus the ultrawide F146.
    Element-wheel spectroscopy (G150 grism, P127 prism) is listed for
    identity only -- mosaic / DOLPHOT use imaging filters.

    L2 filenames look like
    ``r0012301008002013005_0005_wfi06_f184_cal.asdf``. Metadata lives
    under ``meta.instrument.optical_element`` / ``detector`` in
    ``roman_datamodels``, not FITS ``FILTER`` / ``INSTRUME`` cards.
    """

    instrument = 'WFI'
    FILTERS: tuple[str, ...] = (
        'F062', 'F087', 'F106', 'F129', 'F146', 'F158', 'F184', 'F213',
    )
    FILTER_HEADER_KEYS: tuple[str, ...] = ('FILTER',)
    MOSAIC_FILTERS: tuple[str, ...] = (
        'F062', 'F087', 'F106', 'F129', 'F146', 'F158', 'F184', 'F213',
    )
    # NASA GSFC WFI Technical: element center wavelength (microns).
    FILTER_WAVELENGTH_UM: dict[str, float] = {
        'F062': 0.620,
        'F087': 0.869,
        'F106': 1.060,
        'F129': 1.293,
        'F146': 1.464,
        'F158': 1.577,
        'F184': 1.842,
        'F213': 2.125,
    }
    SPECTROSCOPIC_ELEMENTS: tuple[str, ...] = ('G150', 'P127', 'GRISM', 'PRISM')
    PIXEL_SCALE_ARCSEC: float = 0.11
    DETECTOR_COUNT: int = 18
    DETECTOR_SHAPE: tuple[int, int] = (4096, 4096)
    FIELD_OF_VIEW_DEG2: float = 0.281
    science_suffixes: tuple[str, ...] = ('_cal.asdf', '_i2d.asdf')

    @classmethod
    def matches(cls, value: object | None) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        upper = text.upper()
        if 'ROMAN' in upper or upper == 'RST':
            return True
        if upper == 'WFI' or '/WFI/' in upper or '_WFI' in upper:
            return True
        name = Path(text).name.upper()
        if name.startswith('WFI') and (len(name) == 3 or name[3:5].isdigit()):
            return True
        if name.endswith('.ASDF') and name.startswith('R'):
            return True
        return False

    @classmethod
    def detector_ids(cls) -> tuple[str, ...]:
        """SCA ids ``WFI01`` .. ``WFI18``."""
        return tuple(f'WFI{i:02d}' for i in range(1, 19))

    @classmethod
    def filter_from_filename(cls, path: Path | str) -> str | None:
        """Parse ``_f184_`` / ``_g150_`` from a Roman L2 basename."""
        match = _FILENAME_FILTER.search(Path(path).name)
        if not match:
            return None
        return match.group(1).upper()

    @property
    def image_kind(self) -> str:
        return 'wfi'

    @property
    def module_name(self) -> str:
        chip = self.filename_chip
        if chip:
            return chip.lower()
        det = str(self.detector or self.keyword('DETECTOR') or '').strip()
        if det:
            return det.lower()
        return super().module_name

    @property
    def filter_name(self) -> str:
        from_name = self.filter_from_filename(self.path)
        if from_name and from_name in self.FILTERS:
            return from_name.lower()
        try:
            return super().filter_name
        except Exception:
            if from_name:
                return from_name.lower()
            raise

    def is_imaging(self) -> bool:
        """False for grism / prism element-wheel positions."""
        try:
            name = str(self.filter_name or '').upper()
        except Exception:
            name = ''
        if name in self.SPECTROSCOPIC_ELEMENTS:
            return False
        token = self.filter_from_filename(self.path) or ''
        return token not in self.SPECTROSCOPIC_ELEMENTS

    def is_full_frame(self) -> bool:
        """L2 WFI files are one full SCA (no imaging subarrays)."""
        return True
