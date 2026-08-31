"""Euclid VIS science-product datamodel."""

from __future__ import annotations

from pathlib import Path

from st123.datamodels.euclid.euclid import EuclidDataModel

__all__ = ['EuclidVISDataModel']


class EuclidVISDataModel(EuclidDataModel):
    """
    Euclid VIS calibrated exposure (single broad IE band).

    VIS has no filter wheel. The bandpass (~530-920 nm) is set by the
    dichroic plus coatings. ESA / Consortium products use ``INSTRUME=VIS``
    and ``TELESCOP=Euclid``. Calibrated frames are MEFs with science, RMS,
    and flag extensions per CCD quadrant (36 CCDs x 4 quadrants).
    """

    instrument = 'VIS'
    FILTERS: tuple[str, ...] = ('VIS', 'IE', 'I_E')
    FILTER_HEADER_KEYS: tuple[str, ...] = ('FILTER',)
    MOSAIC_FILTERS: tuple[str, ...] = ('VIS',)
    # Cropper+2024 / Euclid Consortium: IE ~530-920 nm; use mid-band pivot.
    FILTER_WAVELENGTH_UM: dict[str, float] = {
        'VIS': 0.725,
        'IE': 0.725,
        'I_E': 0.725,
    }
    PIXEL_SCALE_ARCSEC: float = 0.101
    DETECTOR_COUNT: int = 36
    DETECTOR_SHAPE: tuple[int, int] = (4096, 4132)
    QUADRANTS_PER_DETECTOR: int = 4
    FIELD_OF_VIEW_DEG2: float = 0.57
    # Product card ``FITS_DEF`` values seen in the Euclid DPDD.
    PRODUCT_FITS_DEF: tuple[str, ...] = (
        'vis.calibratedQuadFrame',
        'vis.calibratedScienceFrame',
    )

    @classmethod
    def matches(cls, value: object | None) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        upper = text.upper()
        if 'NISP' in upper or 'NIR_' in upper:
            return False
        if 'VIS' in upper:
            return True
        name = Path(text).name.upper()
        return '_VIS' in name or name.startswith('VIS')

    @classmethod
    def detector_ids(cls) -> tuple[str, ...]:
        """CCD ids ``1-1`` .. ``6-6`` (6 x 6 focal plane)."""
        return tuple(f'{row}-{col}' for row in range(1, 7) for col in range(1, 7))

    @property
    def image_kind(self) -> str:
        return 'vis'

    @property
    def filter_name(self) -> str:
        try:
            return super().filter_name
        except Exception:
            return 'vis'

    def is_full_frame(self) -> bool:
        """VIS science is always the full focal-plane MEF (no subarrays)."""
        return True
