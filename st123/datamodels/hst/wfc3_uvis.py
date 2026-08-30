"""HST WFC3/UVIS science-product datamodel."""

from __future__ import annotations

from st123.datamodels.hst.hst import HSTDataModel

__all__ = ['WFC3UVISDataModel']


class WFC3UVISDataModel(HSTDataModel):
    """WFC3/UVIS ``flc`` / JHAT science product."""

    instrument = 'WFC3'
    detector = 'UVIS'
    science_suffixes = ('_flc.fits', '_jhat.fits')

    @property
    def image_kind(self) -> str:
        return 'wfc3'
