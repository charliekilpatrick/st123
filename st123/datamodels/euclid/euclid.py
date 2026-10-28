"""Euclid science-product datamodel (VIS + NISP/NIR shared identity)."""

from __future__ import annotations

from st123.datamodels.instrument import InstrumentDataModel

__all__ = ['EuclidDataModel']


class EuclidDataModel(InstrumentDataModel):
    """
    Euclid calibrated imaging product.

    Official science products are multi-extension FITS (VIS quadrants or
    NISP ``DETxy.SCI`` / ``.RMS`` / ``.DQ`` layers). DOLPHOT / JHAT knobs
    are not filled yet -- there is no Euclid PSF library in the DOLPHOT
    tree, and JHAT has no Euclid WCS driver.
    """

    telescope = 'EUCLID'
    INSTRUMENTS: tuple[str, ...] = ('VIS', 'NISP')
    FILTER_HEADER_KEYS: tuple[str, ...] = ('FILTER',)
    SCIENCE_FORMAT: str = 'fits'
    science_suffixes: tuple[str, ...] = ('.fits',)

    @classmethod
    def matches(cls, value: object | None) -> bool:
        """True for ``TELESCOP=Euclid``, ``INSTRUME``, or Euclid-like paths."""
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        upper = text.upper()
        if 'EUCLID' in upper:
            return True
        inst = (cls.instrument or '').upper()
        return bool(inst) and inst in upper

    @property
    def image_kind(self) -> str:
        inst = (self.instrument or '').upper()
        if inst in ('NISP', 'NIR'):
            return 'nisp'
        if inst == 'VIS':
            return 'vis'
        return 'euclid'

    def is_science_product(self) -> bool:
        name = self.path.name.lower()
        if name.endswith('.sky.fits'):
            return False
        return self.path.suffix.lower() in {'.fits', '.fit'}

    def is_imaging(self) -> bool:
        """True for photometric frames (not NISP slitless spectroscopy)."""
        return True
