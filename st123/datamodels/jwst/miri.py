"""JWST MIRI science-product datamodel."""

from __future__ import annotations

from pathlib import Path

from st123.datamodels.jwst.jwst import JWSTDataModel

__all__ = ['MIRIDataModel']


class MIRIDataModel(JWSTDataModel):
    """MIRI imager calibrated or i2d / JHAT product."""

    instrument = 'MIRI'
    FILTERS: tuple[str, ...] = (
        'F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W',
        'F2100W', 'F2550W',
        'F1065C', 'F1140C', 'F1550C', 'F2300C',
    )
    FILTER_HEADER_KEYS: tuple[str, ...] = ('FILTER',)
    MOSAIC_FILTERS: tuple[str, ...] = (
        'F770W', 'F1000W', 'F1130W', 'F2100W',
    )
    DOLPHOT_IMAGE_PARAMS: dict[str, str] = {
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
    DOLPHOT_BASE_PARAMS: dict[str, str] = {
        **JWSTDataModel.DOLPHOT_BASE_PARAMS,
        'MIRIvega': '0',
        'RCentroid': '1',
    }
    CALCSKY_PARAMS: dict = {
        'rin': 10,
        'rout': 25,
        'step': -64,
        'sigma_low': 2.25,
        'sigma_high': 2.00,
    }
    MAX_REFERENCE_DISPERSION_MAS: dict[str, float | None] = {
        'F560W': None,
        'F770W': 50.0,
        'F1000W': 35.0,
        'F1130W': 55.0,
        'F1280W': 50.0,
        'F1500W': 50.0,
        'F1800W': 50.0,
        'F2100W': 65.0,
    }
    DEFAULT_MAX_REFERENCE_DISPERSION_MAS: float = 70.0

    @classmethod
    def max_reference_dispersion_mas(cls, filter_name: str | None) -> float | None:
        """REFERENCE quality-hold ceiling in mas, or None to never hold."""
        key = str(filter_name or '').upper().split('_', 1)[0]
        if key in cls.MAX_REFERENCE_DISPERSION_MAS:
            return cls.MAX_REFERENCE_DISPERSION_MAS[key]
        return cls.DEFAULT_MAX_REFERENCE_DISPERSION_MAS

    @classmethod
    def matches(cls, value: object | None) -> bool:
        """True for MAST ``instrument_name``, ``INSTRUME``, or MIRI imager paths."""
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        upper = text.upper()
        if 'MIRI' in upper:
            return True
        return 'MIRIMAGE' in Path(text).name.upper()

    @property
    def image_kind(self) -> str:
        return 'miri'

    @property
    def module_name(self) -> str:
        return 'miri'

    def is_full_frame(self) -> bool:
        """True for full-frame MIRI imager SCI (1024 x 1032)."""
        subarray = str(self.keyword('SUBARRAY') or '').strip().upper()
        if subarray not in ('', 'FULL', 'N/A', 'NONE'):
            return False
        data = self.sci_data()
        if data is None:
            naxis1 = int(self.keyword('NAXIS1') or 0)
            naxis2 = int(self.keyword('NAXIS2') or 0)
            shape = (naxis2, naxis1) if naxis1 and naxis2 else ()
        else:
            shape = tuple(int(x) for x in data.shape)
        if len(shape) == 3 and shape[0] == 1:
            shape = shape[1:]
        return shape == (1024, 1032)
