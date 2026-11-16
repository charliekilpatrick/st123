"""HST ACS/HRC science-product datamodel."""

from __future__ import annotations

from st123.datamodels.hst.acs import ACSDataModel

__all__ = ['ACSHRCDataModel']


class ACSHRCDataModel(ACSDataModel):
    """ACS/HRC ``flt`` / JHAT science product."""

    instrument = 'ACS'
    detector = 'HRC'
    science_suffixes = ('_flt.fits', '_jhat.fits')

    @property
    def image_kind(self) -> str:
        return 'acs'
