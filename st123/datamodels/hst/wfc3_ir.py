"""HST WFC3/IR science-product datamodel."""

from __future__ import annotations

from st123.datamodels.hst.hst import HSTDataModel
from st123.datamodels.instrument import FileIdentity

__all__ = ['WFC3IRDataModel']

_IR_FILTER_TOKS = ('_f110w', '_f160w', '_f125w', '_f140w')


class WFC3IRDataModel(HSTDataModel):
    """WFC3/IR ``flt`` / JHAT science product (no lacosmic)."""

    instrument = 'WFC3'
    detector = 'IR'
    science_suffixes = ('_flt.fits', '_jhat.fits')
    FILTERS: tuple[str, ...] = (
        'F098M', 'F105W', 'F110W', 'F125W', 'F126N', 'F127M', 'F128N', 'F130N',
        'F132N', 'F139M', 'F140W', 'F153M', 'F160W', 'F164N', 'F167N',
    )
    FILTER_HEADER_KEYS: tuple[str, ...] = ('FILTER', 'FILTER1', 'FILTER2')
    DOLPHOT_IMAGE_PARAMS: dict[str, str] = {
        'shift': '0 0',
        'xform': '1 0 0',
        'raper': '2',
        'rchi': '1.5',
        'rsky0': '8',
        'rsky1': '20',
        'rsky2': '3 10',
        'rpsf': '15',
        'apsky': '8 20',
    }
    CALCSKY_PARAMS: dict = {
        'rin': 15,
        'rout': 35,
        'step': 4,
        'sigma_low': 2.25,
        'sigma_high': 2.00,
    }
    DRIZ_BITS: int = 576
    DRIZ_CR: bool = False
    SCI_FLOOR: float = -50.0
    BAD_GROW_PIX: int = 2
    BAD_DQ_BIT: int = 4096

    @property
    def image_kind(self) -> str:
        return 'wfc3_ir'

    @classmethod
    def identity_is_ir(cls, ident: FileIdentity, name: str) -> bool:
        """True when header / filename cues indicate WFC3/IR."""
        det = (ident.detector or '').upper()
        aper = (ident.aperture or '').upper()
        phot = (ident.photmode or '').upper()
        if aper.startswith('IR'):
            return True
        if det == 'IR' or det.startswith('IR'):
            return True
        if 'WFC3' in phot and ' IR' in f' {phot}':
            return True
        if ident.instrument and ident.instrument.upper() not in ('', 'WFC3'):
            return False
        name_l = name.lower()
        if any(tok in name_l for tok in _IR_FILTER_TOKS):
            if 'wfc3' in name_l or name_l.endswith('_flt.fits') or '_jhat' in name_l:
                return True
        return False
