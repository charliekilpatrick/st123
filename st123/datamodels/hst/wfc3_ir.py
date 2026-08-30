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
