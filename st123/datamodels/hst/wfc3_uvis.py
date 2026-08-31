"""HST WFC3/UVIS science-product datamodel."""

from __future__ import annotations

from st123.datamodels.hst.hst import HSTDataModel

__all__ = ['WFC3UVISDataModel']


class WFC3UVISDataModel(HSTDataModel):
    """WFC3/UVIS ``flc`` / JHAT science product."""

    instrument = 'WFC3'
    detector = 'UVIS'
    science_suffixes = ('_flc.fits', '_jhat.fits')
    FILTERS: tuple[str, ...] = (
        'F200LP', 'F218W', 'F225W', 'F275W', 'F280N', 'F300X', 'F336W', 'F343N',
        'F350LP', 'F373N', 'F390M', 'F390W', 'F395N', 'F410M', 'F438W', 'F467M',
        'F469N', 'F475W', 'F475X', 'F487N', 'F502N', 'F547M', 'F555W', 'F600LP',
        'F606W', 'F621M', 'F625W', 'F631N', 'F645N', 'F656N', 'F657N', 'F658N',
        'F665N', 'F673N', 'F680N', 'F689M', 'F763M', 'F775W', 'F814W', 'F845M',
        'F850LP', 'F953N',
    )
    FILTER_HEADER_KEYS: tuple[str, ...] = ('FILTER', 'FILTER1', 'FILTER2')
    DOLPHOT_IMAGE_PARAMS: dict[str, str] = {
        'shift': '0 0',
        'xform': '1 0 0',
        'raper': '3',
        'rchi': '2.0',
        'rsky0': '15',
        'rsky1': '35',
        'rsky2': '4 10',
        'rpsf': '13',
        'apsky': '15 25',
    }
    CALCSKY_PARAMS: dict = {
        'rin': 15,
        'rout': 35,
        'step': 4,
        'sigma_low': 2.25,
        'sigma_high': 2.00,
    }
    DRIZ_BITS: int = 96
    CRPARS: dict[str, float] = {
        'rdnoise': 6.5,
        'gain': 1.0,
        'saturate': 70000.0,
        'sig_clip': 4.0,
        'sig_frac': 0.2,
        'obj_lim': 6.0,
    }

    @property
    def image_kind(self) -> str:
        return 'wfc3'
