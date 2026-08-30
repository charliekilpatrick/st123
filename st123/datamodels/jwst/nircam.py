"""JWST NIRCam science-product datamodel."""

from __future__ import annotations

from st123.datamodels.jwst.jwst import JWSTDataModel

__all__ = ['NIRCamDataModel']


class NIRCamDataModel(JWSTDataModel):
    """NIRCam calibrated or i2d / JHAT product."""

    instrument = 'NIRCAM'

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
