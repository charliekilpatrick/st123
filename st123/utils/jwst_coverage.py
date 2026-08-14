"""JWST NIRCam / MIRI coverage helpers and MIRI-only skip policy."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

_NIRCAM_TOKENS = frozenset({'NIRCAM', 'NRC'})
_MIRI_TOKENS = frozenset({'MIRI'})


def _norm_instruments(instruments: Sequence[str] | None) -> set[str]:
    if not instruments:
        from st123.utils.settings import DEFAULT_JWST_INSTRUMENTS

        return {str(i).upper() for i in DEFAULT_JWST_INSTRUMENTS}
    out: set[str] = set()
    for raw in instruments:
        for part in str(raw).replace(',', ' ').split():
            token = part.strip().upper()
            if token:
                out.add(token)
    return out


def requested_nircam(instruments: Sequence[str] | None) -> bool:
    """True when *instruments* (or JWST defaults) include NIRCam."""
    return bool(_norm_instruments(instruments) & _NIRCAM_TOKENS)


def requested_miri(instruments: Sequence[str] | None) -> bool:
    """True when *instruments* (or JWST defaults) include MIRI."""
    return bool(_norm_instruments(instruments) & _MIRI_TOKENS)


def explicit_miri_only(instruments: Sequence[str] | None) -> bool:
    """True when the caller requested MIRI and not NIRCam."""
    inst = _norm_instruments(instruments)
    # Defaults are NIRCAM+MIRI — that is not "explicit MIRI-only".
    if instruments is None:
        return False
    return bool(inst & _MIRI_TOKENS) and not bool(inst & _NIRCAM_TOKENS)


def force_miri_effective(
    force_miri: bool,
    instruments: Sequence[str] | None,
) -> bool:
    """``--force-miri`` or an explicit MIRI-only ``--instruments`` list."""
    return bool(force_miri) or explicit_miri_only(instruments)


def instrument_name_is_nircam(instrument_name: object | None) -> bool:
    text = str(instrument_name or '').upper()
    return 'NIRCAM' in text or text.startswith('NRC')


def instrument_name_is_miri(instrument_name: object | None) -> bool:
    return 'MIRI' in str(instrument_name or '').upper()


def mast_table_jwst_coverage(obs_table) -> tuple[bool, bool]:
    """
    Return ``(has_nircam, has_miri)`` from a JWST MAST observations table.

    Uses the ``instrument_name`` column when present.
    """
    if obs_table is None or len(obs_table) == 0:
        return False, False
    if 'instrument_name' not in getattr(obs_table, 'colnames', []):
        return False, False
    has_nircam = False
    has_miri = False
    for name in obs_table['instrument_name']:
        if instrument_name_is_nircam(name):
            has_nircam = True
        if instrument_name_is_miri(name):
            has_miri = True
        if has_nircam and has_miri:
            break
    return has_nircam, has_miri


def should_skip_miri_only_jwst(
    has_nircam: bool,
    has_miri: bool,
    instruments: Sequence[str] | None = None,
    *,
    force_miri: bool = False,
) -> bool:
    """
    Return True when JWST should be skipped (MIRI-only, no NIRCam).

    Skip only when both NIRCam and MIRI were requested (default JWST set),
    MIRI coverage exists, NIRCam does not, and the caller did not force MIRI
    (via ``force_miri`` or explicit ``--instruments MIRI``).
    """
    if force_miri_effective(force_miri, instruments):
        return False
    if not (requested_nircam(instruments) and requested_miri(instruments)):
        return False
    return bool(has_miri) and not bool(has_nircam)


def count_jwst_frames_on_disk(base_dir: str | Path) -> tuple[int, int]:
    """
    Count NIRCam and MIRI science frames under a dataset root.

    Searches ``download/JWST/`` and ``reduction/raw/`` (and ``raw/`` when
    *base_dir* is already a reduction workdir).
    """
    root = Path(base_dir).expanduser().resolve()
    nircam: set[Path] = set()
    miri: set[Path] = set()

    search_roots = [
        root / 'download' / 'JWST',
        root / 'reduction' / 'raw',
        root / 'raw',
    ]
    for base in search_roots:
        if not base.is_dir():
            continue
        for path in base.rglob('*.fits'):
            if not path.is_file():
                continue
            name = path.name.lower()
            resolved = path.resolve()
            if 'mirimage' in name:
                miri.add(resolved)
            elif 'nrc' in name:
                nircam.add(resolved)
            else:
                # Fall back to download-tree directory names.
                parts = {p.lower() for p in path.parts}
                if 'miri' in parts:
                    miri.add(resolved)
                elif 'nircam' in parts:
                    nircam.add(resolved)

    return len(nircam), len(miri)


def disk_has_nircam(base_dir: str | Path) -> bool:
    n_nrc, _ = count_jwst_frames_on_disk(base_dir)
    return n_nrc > 0


def disk_has_miri(base_dir: str | Path) -> bool:
    _, n_miri = count_jwst_frames_on_disk(base_dir)
    return n_miri > 0


def should_skip_miri_only_jwst_on_disk(
    base_dir: str | Path,
    instruments: Sequence[str] | None = None,
    *,
    force_miri: bool = False,
) -> bool:
    """On-disk variant of :func:`should_skip_miri_only_jwst`."""
    n_nrc, n_miri = count_jwst_frames_on_disk(base_dir)
    return should_skip_miri_only_jwst(
        n_nrc > 0,
        n_miri > 0,
        instruments,
        force_miri=force_miri,
    )
