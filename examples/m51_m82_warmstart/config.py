"""
Dataset-specific warm-start architecture for M51 and M82.

Only reductions with a significant MIRI footprint are eligible. References
with fewer than :data:`MIN_MIRI_IMAGES` SUCCESS JHAT frames are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Minimum number of unique MIRI *_jhat.fits frames aligned to a reference
# for that reference to be included in warm-start staging / DOLPHOT.
MIN_MIRI_IMAGES = 10

# Prefer JHAT with finite alignment dispersion below this (mas). Frames at
# the 99990 sentinel (or otherwise pathological) are dropped when building
# usable lists (M82).
MAX_USABLE_DISPERSION_MAS = 200.0

DEFAULT_NCORES = 16
MAX_PARALLEL_DOLPHOT = 3

# Sidecar directory for per-ref JHAT path lists (under the repo examples/).
LIST_DIR = Path(__file__).resolve().parents[1] / 'outdir_m51_m82_warmstart'


@dataclass(frozen=True)
class DatasetSpec:
    """One galaxy project root under /data/rwisenbaker/jwst_data."""

    name: str
    base_dir: Path
    # Object name used in DOLPHOT phot catalog stems (M51 → ngc5194).
    phot_stem: str
    # If True, require an existing dolphot/nircam_{g}_{b} before staging.
    require_nircam_dolphot: bool = True


DATASETS: dict[str, DatasetSpec] = {
    'M51': DatasetSpec(
        name='M51',
        base_dir=Path('/data/rwisenbaker/jwst_data/M51'),
        phot_stem='ngc5194',
    ),
    'M82': DatasetSpec(
        name='M82',
        base_dir=Path('/data/rwisenbaker/jwst_data/M82'),
        # NIRCam DOLPHOT catalogs are named ngc3034_{g}_{b}.phot
        phot_stem='ngc3034',
    ),
}


def alignment_summary_path(spec: DatasetSpec) -> Path:
    return spec.base_dir / f'{spec.name}_alignment_summary.txt'


def nircam_dir(spec: DatasetSpec, group: int, box: int) -> Path:
    return spec.base_dir / 'dolphot' / f'nircam_{group}_{box}'


def warmstart_outdir(spec: DatasetSpec, group: int, box: int) -> Path:
    return spec.base_dir / 'dolphot' / f'nircam_miri_{group}_{box}'


def jhat_list_path(spec: DatasetSpec, group: int, box: int) -> Path:
    return LIST_DIR / f'{spec.name}_ref_{group}_{box}_miri_jhat.txt'


def phot_out_name(spec: DatasetSpec, group: int, box: int) -> str:
    return f'{spec.phot_stem}_{group}_{box}_nircam_miri.phot'
