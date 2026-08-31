"""HST ACS science-product datamodel (WFC / HRC shared filters and DOLPHOT)."""

from __future__ import annotations

from st123.datamodels.hst.hst import HSTDataModel

__all__ = ['ACSDataModel']


class ACSDataModel(HSTDataModel):
    """ACS imaging: WFC, HRC, and SBC share filter / DOLPHOT / drizzle knobs."""

    instrument = 'ACS'
    FILTERS: tuple[str, ...] = (
        'F220W', 'F250W', 'F330W', 'F344N', 'F435W', 'F475W', 'F502N', 'F550M',
        'F555W', 'F606W', 'F625W', 'F658N', 'F660N', 'F775W', 'F814W', 'F850LP',
        'F892N',
        'F115LP', 'F122M', 'F125LP', 'F140LP', 'F150LP', 'F165LP',
    )
    FILTER_HEADER_KEYS: tuple[str, ...] = ('FILTER1', 'FILTER2', 'FILTER')
    DOLPHOT_IMAGE_PARAMS: dict[str, str] = {
        'shift': '0 0',
        'xform': '1 0 0',
        'raper': '2',
        'rchi': '1.5',
        'rsky0': '15',
        'rsky1': '35',
        'rsky2': '3 6',
        'rpsf': '10',
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
        'sig_clip': 3.0,
        'sig_frac': 0.1,
        'obj_lim': 5.0,
    }

    @property
    def image_kind(self) -> str:
        return 'acs'
