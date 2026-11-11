"""HST ACS/WFC science-product datamodel."""

from __future__ import annotations

from st123.datamodels.hst.acs import ACSDataModel

__all__ = ['ACSWFCDataModel']


class ACSWFCDataModel(ACSDataModel):
    """ACS/WFC ``flc`` / JHAT science product."""

    instrument = 'ACS'
    detector = 'WFC'
    science_suffixes = ('_flc.fits', '_jhat.fits')

    @property
    def image_kind(self) -> str:
        return 'acs'
