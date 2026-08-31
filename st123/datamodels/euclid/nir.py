"""Euclid NIR (NISP photometer) science-product datamodel."""

from __future__ import annotations

from pathlib import Path

from st123.datamodels.euclid.euclid import EuclidDataModel

__all__ = ['EuclidNIRDataModel']

# FWA_POS wheel tokens -> photometric FILTER aliases (Euclid DPDD).
_FWA_TO_FILTER: dict[str, str] = {
    'Y': 'YE',
    'J': 'JE',
    'H': 'HE',
    'NIR_Y': 'YE',
    'NIR_J': 'JE',
    'NIR_H': 'HE',
}


class EuclidNIRDataModel(EuclidDataModel):
    """
    Euclid NISP photometric (NIR imaging) calibrated frame.

    The flight instrument is NISP; Euclid DPDD calls the imaging product
    ``nir.calibratedScienceFrame``. ``INSTRUME`` is ``NISP``. Photometry
    uses the filter wheel (FWA: Y/J/H). Grism-wheel spectroscopy
    (``GWA_POS`` not OPEN) is not an imaging product for mosaic / DOLPHOT.

    NISP-P bands (Schirmer+2022 50% cut-on/off): YE 950-1212 nm, JE
    1168-1567 nm, HE 1522-2021 nm. Headers use ``FILTER=NIR_J`` and/or
    ``FWA_POS=J``.
    """

    instrument = 'NISP'
    FILTERS: tuple[str, ...] = (
        'YE', 'JE', 'HE',
        'NIR_Y', 'NIR_J', 'NIR_H',
        'Y', 'J', 'H',
    )
    FILTER_HEADER_KEYS: tuple[str, ...] = ('FILTER', 'FWA_POS')
    MOSAIC_FILTERS: tuple[str, ...] = ('YE', 'JE', 'HE')
    FILTER_WAVELENGTH_UM: dict[str, float] = {
        'YE': 1.081,
        'JE': 1.367,
        'HE': 1.771,
        'NIR_Y': 1.081,
        'NIR_J': 1.367,
        'NIR_H': 1.771,
        'Y': 1.081,
        'J': 1.367,
        'H': 1.771,
    }
    PIXEL_SCALE_ARCSEC: float = 0.30
    DETECTOR_COUNT: int = 16
    DETECTOR_SHAPE: tuple[int, int] = (2048, 2048)
    FIELD_OF_VIEW_DEG2: float = 0.55
    # Grism-wheel tokens that mean slitless spectroscopy, not imaging.
    SPECTROSCOPIC_GWA: tuple[str, ...] = (
        'BGS000', 'RGS000', 'RGS180', 'RGS270',
    )
    PRODUCT_FITS_DEF: tuple[str, ...] = (
        'nir.calibratedScienceFrame',
    )

    @classmethod
    def matches(cls, value: object | None) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        upper = text.upper()
        if 'NISP' in upper:
            return True
        if 'NIR_' in upper or '/NIR/' in upper:
            return True
        name = Path(text).name.upper()
        if name.startswith('NIR') and 'NIRCAM' not in name:
            return True
        return any(tok in name for tok in ('_YE', '_JE', '_HE', 'NIR_Y', 'NIR_J', 'NIR_H'))

    @classmethod
    def detector_ids(cls) -> tuple[str, ...]:
        """NISP detector ids ``DET11`` .. ``DET44`` (4 x 4 H2RG array)."""
        return tuple(f'DET{row}{col}' for row in range(1, 5) for col in range(1, 5))

    @classmethod
    def canonical_filter(cls, name: str | None) -> str:
        """Map header / FWA tokens onto YE / JE / HE."""
        key = str(name or '').strip().upper()
        if key in _FWA_TO_FILTER:
            return _FWA_TO_FILTER[key]
        if key in ('YE', 'JE', 'HE'):
            return key
        return key

    @property
    def image_kind(self) -> str:
        return 'nisp'

    @property
    def filter_name(self) -> str:
        try:
            raw = super().filter_name
        except Exception:
            raw = ''
        if raw:
            return self.canonical_filter(raw).lower()
        fwa = self.keyword('FWA_POS')
        if fwa is not None and str(fwa).strip():
            return self.canonical_filter(str(fwa)).lower()
        raise KeyError(f'No NISP filter keyword found in {self.path}')

    def is_imaging(self) -> bool:
        """False when the grism wheel is in a spectroscopic position."""
        gwa = str(self.keyword('GWA_POS') or 'OPEN').strip().upper()
        if not gwa or gwa in ('OPEN', 'NONE', 'N/A'):
            return True
        return gwa not in self.SPECTROSCOPIC_GWA

    def is_full_frame(self) -> bool:
        """NISP calibrated frames are the 16-detector MEF (no subarrays)."""
        return True
