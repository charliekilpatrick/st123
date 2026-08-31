"""HST WFPC2 science-product datamodel (``c0m`` science + ``c1m`` DQ)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from st123.datamodels.hst.hst import HSTDataModel
from st123.datamodels.instrument import FileIdentity, PathLike, read_file_identity

__all__ = ['WFPC2DataModel']


class WFPC2DataModel(HSTDataModel):
    """
    One WFPC2 exposure: calibrated ``c0m`` (science) plus sibling ``c1m`` (DQ).

    Opening either file yields this class. :attr:`path` is always the science
    MEF when a ``c0m`` exists; :attr:`dq_path` is the DQ companion.
    """

    instrument = 'WFPC2'
    science_suffixes = ('_c0m.fits', '_jhat.fits')
    FILTERS: tuple[str, ...] = (
        'F122M', 'F160BW', 'F185W', 'F218W', 'F255W', 'F300W', 'F336W', 'F375N',
        'F380W', 'F390N', 'F437N', 'F439W', 'F450W', 'F467M', 'F469N', 'F487N',
        'F502N', 'F547M', 'F555W', 'F569W', 'F588N', 'F606W', 'F622W', 'F631N',
        'F656N', 'F658N', 'F673N', 'F675W', 'F702W', 'F785LP', 'F791W', 'F814W',
        'F850LP', 'F953N', 'F1042M',
    )
    FILTER_HEADER_KEYS: tuple[str, ...] = ('FILTER', 'FILTNAM1', 'FILTNAM2')
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
        'rin': 10,
        'rout': 25,
        'step': 2,
        'sigma_low': 2.25,
        'sigma_high': 2.00,
    }
    DRIZ_BITS: int = 1032
    CRPARS: dict[str, float] = {
        'rdnoise': 10.0,
        'gain': 7.0,
        'saturate': 27000.0,
        'sig_clip': 4.0,
        'sig_frac': 0.3,
        'obj_lim': 6.0,
    }
    OVERSCAN_EDGE_PIX: int = 32
    OVERSCAN_LEFT_EXTRA: int = 20
    OVERSCAN_DQ_BIT: int = 256
    SCI_FLOOR: float = -20.0
    BAD_GROW_PIX: int = 2
    BAD_COL_FRAC: float = 0.50
    VAR_EDGE_PIX: int = 48
    VAR_SIGMA: float = 5.0
    DROP_SINGLE_CTX_EDGE_PIX: int = 16

    def __init__(
        self,
        path: PathLike,
        *,
        dq_path: PathLike | None = None,
        telescope: str = '',
        instrument: str = '',
        detector: str = '',
        aperture: str = '',
        photmode: str = '',
    ) -> None:
        super().__init__(
            path,
            telescope=telescope,
            instrument=instrument,
            detector=detector,
            aperture=aperture,
            photmode=photmode,
        )
        self._dq_path = Path(dq_path) if dq_path is not None else None

    @classmethod
    def c1m_path_for(cls, science: PathLike) -> Path:
        """Conventional sibling ``*_c1m.fits`` path for a ``c0m`` / JHAT file."""
        path = Path(science)
        name = path.name
        if name.endswith('_c0m.fits'):
            return path.with_name(name.replace('_c0m.fits', '_c1m.fits'))
        if name.endswith('c0m.fits'):
            return path.with_name(name[:-8] + 'c1m.fits')
        if name.endswith('_jhat.fits'):
            stem = name[: -len('_jhat.fits')]
            return path.with_name(f'{stem}_c1m.fits')
        stem = path.stem
        if '_c0m' in stem:
            return path.with_name(stem.replace('_c0m', '_c1m') + path.suffix)
        return path.with_name(stem + '_c1m.fits')

    @classmethod
    def c0m_path_for(cls, dq: PathLike) -> Path:
        """Conventional sibling ``*_c0m.fits`` path for a ``c1m`` DQ file."""
        path = Path(dq)
        name = path.name
        if name.endswith('_c1m.fits'):
            return path.with_name(name.replace('_c1m.fits', '_c0m.fits'))
        if name.endswith('c1m.fits'):
            return path.with_name(name[:-8] + 'c0m.fits')
        stem = path.stem
        if '_c1m' in stem:
            return path.with_name(stem.replace('_c1m', '_c0m') + path.suffix)
        return path.with_name(stem + '_c0m.fits')

    @classmethod
    def find_dq_beside(cls, science: PathLike) -> Path | None:
        """Locate an on-disk ``c1m`` next to *science* (c0m, jhat, or MEF)."""
        science_path = Path(science)
        name = science_path.name
        candidates: list[Path] = [cls.c1m_path_for(science_path)]
        if name.endswith('_jhat.fits'):
            stem = name[: -len('_jhat.fits')]
            candidates.append(
                science_path.with_name(f'{stem}_c0m'.replace('_c0m', '') + '_c1m.fits')
            )
            candidates.append(science_path.parent.parent / 'raw' / f'{stem}_c1m.fits')
            candidates.append(science_path.parent / f'{stem}_c1m.fits')
        seen: set[Path] = set()
        for cand in candidates:
            key = cand.resolve() if cand.exists() else cand
            if key in seen:
                continue
            seen.add(key)
            if cand.is_file():
                return cand
        return None

    @classmethod
    def from_path(
        cls,
        path: PathLike,
        identity: FileIdentity | None = None,
    ) -> WFPC2DataModel:
        p = Path(path)
        name = p.name.lower()
        dq: Path | None = None
        sci = p
        if name.endswith('c1m.fits'):
            dq = p
            cand = cls.c0m_path_for(p)
            if cand.is_file():
                sci = cand
        else:
            dq = cls.find_dq_beside(sci)
        ident = identity if identity is not None else read_file_identity(sci)
        if sci != p:
            ident = read_file_identity(sci)
        return cls(
            sci,
            dq_path=dq,
            telescope=ident.telescope or cls.telescope,
            instrument=ident.instrument or cls.instrument,
            detector=ident.detector or cls.detector,
            aperture=ident.aperture,
            photmode=ident.photmode,
        )

    @property
    def image_kind(self) -> str:
        return 'wfpc2'

    @property
    def science_path(self) -> Path:
        """Calibrated science MEF (``c0m`` or JHAT)."""
        return self.path

    @property
    def dq_path(self) -> Path:
        """DQ companion path (``c1m``), even if the file is not on disk yet."""
        if self._dq_path is not None:
            return self._dq_path
        found = self.find_dq_beside(self.path)
        if found is not None:
            return found
        return self.c1m_path_for(self.path)

    def has_dq(self) -> bool:
        """True when the ``c1m`` companion exists on disk."""
        return self.dq_path.is_file()

    def open_dq(
        self,
        *,
        mode: str = 'readonly',
        memmap: bool = False,
    ) -> fits.HDUList:
        """Open the ``c1m`` DQ companion (same contract as ``fits.open``)."""
        dq = self.dq_path
        if not dq.is_file():
            raise FileNotFoundError(dq)
        return fits.open(dq, mode=mode, memmap=memmap)

    def matching_dq_array(
        self,
        hdul,
        sci_hdu,
        sci_index: int,
    ):
        """DQ from the science MEF when present, else the sibling ``c1m``."""
        try:
            return super().matching_dq_array(hdul, sci_hdu, sci_index)
        except ValueError:
            pass
        if not self.has_dq():
            raise ValueError(
                f'No DQ extension with matching shape found in {self.path} '
                f'(no sibling {self.dq_path.name})'
            )
        with self.open_dq() as dq_hdul:
            shape = tuple(sci_hdu.data.shape)
            if (
                0 <= sci_index < len(dq_hdul)
                and dq_hdul[sci_index].data is not None
            ):
                data = np.asarray(dq_hdul[sci_index].data)
                if getattr(data, 'ndim', 0) >= 2 and tuple(data.shape) == shape:
                    return data
            return super().matching_dq_array(dq_hdul, sci_hdu, sci_index)
