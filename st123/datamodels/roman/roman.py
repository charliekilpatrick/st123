"""Roman science-product datamodel (WFI shared identity)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from st123.datamodels.instrument import InstrumentDataModel

__all__ = ['RomanDataModel']


class RomanDataModel(InstrumentDataModel):
    """
    Nancy Grace Roman Space Telescope science product.

    Official L1/L2 WFI products are ASDF (``roman_datamodels``), one detector
    per file, not multi-extension FITS. st123 still accepts FITS stubs for
    tests and any exported FITS. DOLPHOT / JHAT knobs are not filled yet --
    there is no Roman PSF library in DOLPHOT, and JHAT has no Roman driver.
    """

    telescope = 'ROMAN'
    INSTRUMENTS: tuple[str, ...] = ('WFI',)
    FILTER_HEADER_KEYS: tuple[str, ...] = ('FILTER',)
    SCIENCE_FORMAT: str = 'asdf'
    science_suffixes: tuple[str, ...] = ('_cal.asdf', '_i2d.asdf', '.asdf')

    @classmethod
    def matches(cls, value: object | None) -> bool:
        """True for ``TELESCOP=Roman``, MAST ``Roman``, or WFI-like paths."""
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        upper = text.upper()
        if 'ROMAN' in upper or upper == 'RST':
            return True
        name = Path(text).name.upper()
        if upper == 'WFI' or '/WFI/' in upper or '_WFI' in upper:
            return True
        if name.startswith('WFI') and (len(name) == 3 or name[3:5].isdigit()):
            return True
        return name.endswith('.ASDF') and name.startswith('R')

    @property
    def image_kind(self) -> str:
        return 'wfi'

    def is_asdf(self) -> bool:
        return self.path.suffix.lower() == '.asdf'

    def is_science_product(self) -> bool:
        name = self.path.name.lower()
        if name.endswith('.sky.fits'):
            return False
        if self.is_asdf():
            return any(
                name.endswith(suf) for suf in ('_cal.asdf', '_i2d.asdf', '.asdf')
            )
        return self.path.suffix.lower() in {'.fits', '.fit'}

    def sanitize(
        self,
        *,
        materialize_headers: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        """ASDF products skip the FITS header-card sanitize."""
        if self.is_asdf():
            report: dict[str, Any] = {
                'path': str(self.path),
                'ok': True,
                'skipped': True,
                'reason': 'asdf product; FITS sanitize not applied',
                'telescope': self.telescope,
                'instrument': self.instrument,
            }
            self._sanitized = True
            return report
        return super().sanitize(materialize_headers=materialize_headers, force=force)

    def asdf_model(self):
        """
        Open an official Roman ASDF product.

        Uses ``roman_datamodels.open`` when that package is installed.
        """
        if not self.is_asdf():
            raise TypeError(f'{self.path} is not an ASDF product')
        try:
            import roman_datamodels as rdm
        except ImportError as exc:
            raise ImportError(
                'roman_datamodels is required to open Roman ASDF products'
            ) from exc
        return rdm.open(self.path)

    def is_imaging(self) -> bool:
        return True
