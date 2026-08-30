"""JWST MIRI science-product datamodel."""

from __future__ import annotations

from st123.datamodels.jwst.jwst import JWSTDataModel

__all__ = ['MIRIDataModel']


class MIRIDataModel(JWSTDataModel):
    """MIRI imager calibrated or i2d / JHAT product."""

    instrument = 'MIRI'

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
