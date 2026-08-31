"""HST science-product datamodel (EXPFLAG / residual gates, header sanitize)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from st123.datamodels.instrument import (
    InstrumentDataModel,
    PathLike,
    as_datamodel,
    log_rejected_frames,
    path_of,
)

logger = logging.getLogger(__name__)

# MAST / OPUS exposure quality. Blank / missing treated as good (legacy products).
GOOD_HST_EXPFLAGS = frozenset({None, '', 'NORMAL'})

# HST science product suffixes that carry EXPFLAG (and JHAT copies of them).
HST_SCIENCE_SUFFIXES = (
    '_flc.fits',
    '_flt.fits',
    '_c0m.fits',
    '_jhat.fits',
)

# Catastrophic absolute residual ceiling (arcsec). ``ST123INT`` is a *group*
# pairwise stamp and must not be used to reject individual frames -- a single
# bad visit can pollute every frame's ST123INT (e.g. F160W after iejn02).
DEFAULT_MAX_ABS_ARCSEC = 0.50
# Prefer leaving internal gating off.
DEFAULT_MAX_INTERNAL_ARCSEC = None

__all__ = [
    'DEFAULT_MAX_ABS_ARCSEC',
    'DEFAULT_MAX_INTERNAL_ARCSEC',
    'GOOD_HST_EXPFLAGS',
    'HSTDataModel',
    'HST_SCIENCE_SUFFIXES',
    'filter_good_hst_frames',
    'is_good_hst_alignment',
    'is_good_hst_expflag',
    'is_hst_science_path',
    'log_rejected_hst_frames',
    'prune_bad_hst_expflag',
    'read_expflag',
]


def _float_keyword_arcsec(model: InstrumentDataModel, key: str) -> float | None:
    val = model.keyword(key)
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def read_expflag(image) -> str | None:
    """
    Return primary ``EXPFLAG`` as an uppercased string, or ``None`` if absent.

    Non-FITS / unreadable paths return ``None`` (treated as good when
    ``missing_ok`` is True).
    """
    raw = as_datamodel(image).keyword('EXPFLAG')
    if raw is None:
        return None
    text = str(raw).strip()
    return text.upper() if text else None


class HSTDataModel(InstrumentDataModel):
    """HST calibrated or JHAT product: ``EXPFLAG`` gates and WCS-card sanitize."""

    telescope = 'HST'
    science_suffixes = HST_SCIENCE_SUFFIXES
    INSTRUMENTS: tuple[str, ...] = ('ACS', 'WFC3', 'WFPC2')
    DEFAULT_MAST_FILTERS: tuple[str, ...] | None = None
    BEST_REFERENCE_FILTERS: tuple[str, ...] = (
        'f625w',
        'f606w',
        'f555w',
        'f814w',
        'f350lp',
        'f110w',
        'f105w',
        'f336w',
    )
    MAST_PRODUCT_RULES: tuple[tuple[str, str], ...] = (
        ('c0m.fits', 'WFPC2'),
        ('c1m.fits', 'WFPC2'),
        ('c0m.fits', 'PC/WFC'),
        ('c1m.fits', 'PC/WFC'),
        ('flc.fits', 'ACS/WFC'),
        ('flt.fits', 'ACS/HRC'),
        ('flt.fits', 'ACS/SBC'),
        ('flc.fits', 'WFC3/UVIS'),
        ('flt.fits', 'WFC3/IR'),
    )
    CR_DQ_BIT: int = 4096
    DRIZZLE_DEFAULTS: dict = {
        'final_pixfrac': 0.8,
        'driz_sep_pixfrac': 0.8,
        'combine_maskpt': 0.2,
        'combine_nsigma': '4 3',
        'driz_cr_snr': '3.5 3.0',
        'driz_cr_grow': 1,
        'driz_cr_scale': '1.2 0.7',
        'num_cores': 4,
    }
    DOLPHOT_BASE_PARAMS: dict[str, str] = {
        'FitSky': '2',
        'SigPSF': '5.0',
        'FlagMask': '7',
        'SecondPass': '5',
        'PSFPhotIt': '2',
        'ApCor': '1',
        'FSat': '0.999',
        'NoiseMult': '0.1',
        'RCombine': '1.5',
        'CombineChi': '0',
        'MaxIT': '25',
        'InterpPSFlib': '1',
        'SigFindMult': '0.85',
        'PSFPhot': '1',
        'Force1': '1',
        'SkySig': '2.25',
        'SkipSky': '1',
        'UseWCS': '2',
        'PSFres': '0',
        'PosStep': '0.25',
        'NIRCAMvega': '0',
        'Align': '0',
        'aligntol': '0',
        'Rotate': '0',
        'AlignStep': '1',
        'AlignIter': '2',
        'ACSuseCTE': '0',
        'WFC3useCTE': '0',
        'WFPC2useCTE': '1',
        'RCentroid': '2',
    }

    @classmethod
    def drizzle_bits(cls, kind: str) -> int:
        """AstroDrizzle ``final_bits`` / ``driz_sep_bits`` for an image kind."""
        from st123.datamodels.hst.acs import ACSDataModel
        from st123.datamodels.hst.wfc3_ir import WFC3IRDataModel
        from st123.datamodels.hst.wfc3_uvis import WFC3UVISDataModel
        from st123.datamodels.hst.wfpc2 import WFPC2DataModel

        table = {
            'acs': ACSDataModel.DRIZ_BITS,
            'wfc3': WFC3UVISDataModel.DRIZ_BITS,
            'wfc3_uvis': WFC3UVISDataModel.DRIZ_BITS,
            'wfc3_ir': WFC3IRDataModel.DRIZ_BITS,
            'wfpc2': WFPC2DataModel.DRIZ_BITS,
        }
        return int(table.get(str(kind or '').lower(), 0))

    @classmethod
    def crpars_for(cls, instrument: str) -> dict[str, float]:
        """astroscrappy CR parameters for ``acs`` / ``wfc3`` / ``wfpc2``."""
        from st123.datamodels.hst.acs import ACSDataModel
        from st123.datamodels.hst.wfc3_uvis import WFC3UVISDataModel
        from st123.datamodels.hst.wfpc2 import WFPC2DataModel

        key = instrument.split('_')[0].lower()
        table = {
            'acs': ACSDataModel.CRPARS,
            'wfc3': WFC3UVISDataModel.CRPARS,
            'wfpc2': WFPC2DataModel.CRPARS,
        }
        if key not in table:
            raise ValueError(
                f'No CRPARS for instrument={instrument!r}; '
                f'expected one of {sorted(table)}'
            )
        return dict(table[key])

    @classmethod
    def matches_science_filename(cls, path: PathLike) -> bool:
        """True when *path* looks like an HST science MEF (not DQ-only ``c1m``)."""
        name = Path(path).name.lower()
        if name.endswith('c1m.fits') or name.endswith('.sky.fits'):
            return False
        return any(name.endswith(suf) for suf in HST_SCIENCE_SUFFIXES)

    def is_science_product(self) -> bool:
        return self.matches_science_filename(self.path)

    @property
    def image_kind(self) -> str:
        inst = (self.instrument or '').upper()
        phot = f' {self.photmode} '
        if inst == 'ACS' or self.photmode.startswith('ACS') or ',ACS' in self.photmode:
            return 'acs'
        if inst == 'WFPC2' or self.photmode.startswith('WFPC2') or 'WFPC2,' in self.photmode:
            return 'wfpc2'
        if inst == 'WFC3' or self.photmode.startswith('WFC3'):
            if self._looks_ir():
                return 'wfc3_ir'
            return 'wfc3'
        aper = self.aperture
        if aper.startswith('UVIS'):
            return 'wfc3'
        if aper.startswith('IR'):
            return 'wfc3_ir'
        if aper.startswith('WFC') or aper.startswith('HRC'):
            return 'acs'
        if ' ACS' in phot or phot.strip().startswith('ACS'):
            return 'acs'
        return ''

    def _looks_ir(self) -> bool:
        det = (self.detector or '').upper()
        aper = (self.aperture or '').upper()
        phot = (self.photmode or '').upper()
        if det == 'IR' or det.startswith('IR'):
            return True
        if aper.startswith('IR'):
            return True
        if 'WFC3' in phot and ' IR' in f' {phot}':
            return True
        return False

    def read_expflag(self) -> str | None:
        raw = self.keyword('EXPFLAG')
        if raw is None:
            return None
        text = str(raw).strip()
        return text.upper() if text else None

    def is_good_exposure(self, *, missing_ok: bool = True) -> bool:
        if not self.matches_science_filename(self.path):
            return True
        flag = self.read_expflag()
        if flag is None:
            return bool(missing_ok)
        return flag in GOOD_HST_EXPFLAGS

    def is_good_alignment(
        self,
        *,
        max_internal_arcsec: float | None = DEFAULT_MAX_INTERNAL_ARCSEC,
        max_abs_arcsec: float | None = DEFAULT_MAX_ABS_ARCSEC,
        missing_ok: bool = True,
    ) -> bool:
        if not self.matches_science_filename(self.path):
            return True
        if max_internal_arcsec is not None:
            st_int = _float_keyword_arcsec(self, 'ST123INT')
            if st_int is None:
                if not missing_ok:
                    return False
            elif float(st_int) > float(max_internal_arcsec):
                return False
        if max_abs_arcsec is not None:
            disp = _float_keyword_arcsec(self, 'JWDISPM')
            if disp is None:
                if not missing_ok:
                    return False
            elif float(disp) > float(max_abs_arcsec):
                return False
        return True

    def reject_detail(self) -> str:
        extra = []
        flag = self.read_expflag()
        st_int = _float_keyword_arcsec(self, 'ST123INT')
        disp = _float_keyword_arcsec(self, 'JWDISPM')
        if flag is not None:
            extra.append(f'EXPFLAG={flag}')
        if st_int is not None:
            extra.append(f'ST123INT={st_int:.4f}"')
        if disp is not None:
            extra.append(f'JWDISPM={disp:.4f}"')
        return ', '.join(extra)


def is_hst_science_path(path: PathLike) -> bool:
    """True when *path* looks like an HST science MEF (not DQ-only ``c1m``)."""
    return HSTDataModel.matches_science_filename(path)


def is_good_hst_expflag(
    image,
    *,
    missing_ok: bool = True,
) -> bool:
    """
    True when *image* is acceptable under the EXPFLAG gate.

    Non-HST science products always return True. Missing ``EXPFLAG`` is
    accepted when *missing_ok* is True (default).
    """
    return as_datamodel(image).is_good_exposure(missing_ok=missing_ok)


def is_good_hst_alignment(
    image,
    *,
    max_internal_arcsec: float | None = DEFAULT_MAX_INTERNAL_ARCSEC,
    max_abs_arcsec: float | None = DEFAULT_MAX_ABS_ARCSEC,
    missing_ok: bool = True,
) -> bool:
    """True when stamped JHAT residuals are within optional ceilings."""
    return as_datamodel(image).is_good_alignment(
        max_internal_arcsec=max_internal_arcsec,
        max_abs_arcsec=max_abs_arcsec,
        missing_ok=missing_ok,
    )


def filter_good_hst_frames(
    paths: Sequence[PathLike],
    *,
    missing_ok: bool = True,
    require_expflag: bool = True,
    max_internal_arcsec: float | None = None,
    max_abs_arcsec: float | None = None,
) -> tuple[list[Path], list[Path]]:
    """
    Split *paths* into ``(kept, rejected)`` under EXPFLAG / optional residual gates.

    Non-HST filenames always stay in *kept*. Alignment ceilings default to off
    (``None``).
    """
    kept: list[Path] = []
    rejected: list[Path] = []
    for raw in paths:
        p = path_of(raw)
        model = as_datamodel(raw)
        if require_expflag and not model.is_good_exposure(missing_ok=missing_ok):
            rejected.append(p)
            continue
        if (
            max_internal_arcsec is not None or max_abs_arcsec is not None
        ) and not model.is_good_alignment(
            max_internal_arcsec=max_internal_arcsec,
            max_abs_arcsec=max_abs_arcsec,
            missing_ok=missing_ok,
        ):
            rejected.append(p)
            continue
        kept.append(p)
    return kept, rejected


def log_rejected_hst_frames(
    rejected: Sequence[PathLike],
    *,
    stage: str,
) -> None:
    """Log each rejected path with EXPFLAG / residual keywords when present."""
    log_rejected_frames(rejected, stage=stage)


def prune_bad_hst_expflag(
    root: PathLike,
    *,
    remove: bool = True,
    patterns: tuple[str, ...] = (
        '*_flc.fits',
        '*_flt.fits',
        '*_c0m.fits',
        '*_jhat.fits',
    ),
) -> list[str]:
    """
    Scan *root* for HST science products with bad ``EXPFLAG``.

    When *remove* is True, unlink matching files (and dangling symlinks).
    Returns the rejected path strings.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    rejected: list[str] = []
    seen: set[Path] = set()
    for pat in patterns:
        for path in root_path.rglob(pat):
            key = path.resolve() if path.exists() else path
            if key in seen:
                continue
            seen.add(key)
            if not is_hst_science_path(path):
                continue
            if is_good_hst_expflag(path, missing_ok=True):
                continue
            rejected.append(str(path))
            log_rejected_hst_frames([path], stage='prune')
            if remove:
                try:
                    path.unlink()
                except OSError as exc:
                    logger.warning('could not remove %s: %s', path, exc)
            return rejected
