"""
Discover per-reference MIRI JHAT lists for M51/M82 and apply the min-image cut.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import (
    DATASETS,
    LIST_DIR,
    MAX_USABLE_DISPERSION_MAS,
    MIN_MIRI_IMAGES,
    DatasetSpec,
    alignment_summary_path,
    jhat_list_path,
    nircam_dir,
)

logger = logging.getLogger('m51_m82_discover')

_FILTER_RE = re.compile(r'^F\d{3,4}W$')
# Mosaic box id from coadd filename (preferred; M82 stores many boxes under ref_0).
_COADD_RE = re.compile(r'coadd_(\d+)_(\d+)_', re.IGNORECASE)
# Fallback: group_G/ref_B directory (matches M51 layout).
_REF_RE = re.compile(r'group_(\d+)/ref_(\d+)')


@dataclass(frozen=True)
class RefWarmstartPlan:
    """One eligible (or skipped) reference warm-start target."""

    dataset: str
    group: int
    box: int
    n_miri: int
    jhat_paths: tuple[Path, ...]
    nircam_ready: bool
    eligible: bool
    skip_reason: str = ''

    @property
    def ref_key(self) -> str:
        return f'{self.group}_{self.box}'


def parse_alignment_summary(path: Path) -> list[dict]:
    """Parse ``*_alignment_summary.txt`` into row dicts."""
    rows: list[dict] = []
    if not path.is_file():
        raise FileNotFoundError(path)
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('miri_path') or line.startswith('-'):
            continue
        parts = line.split()
        filt_i = next(
            (i for i, t in enumerate(parts) if _FILTER_RE.fullmatch(t)), None
        )
        if filt_i is None or len(parts) <= filt_i + 6:
            continue
        status = parts[filt_i + 1]
        disp_tok = parts[filt_i + 4]
        try:
            disp = float(disp_tok) if disp_tok != 'NA' else np.nan
        except ValueError:
            disp = np.nan
        aligned = parts[filt_i + 6]
        original_ref = parts[filt_i + 7] if len(parts) > filt_i + 7 else 'NA'
        aligned_to = parts[filt_i + 8] if len(parts) > filt_i + 8 else 'NA'
        ref = None
        # Prefer coadd_{group}_{box}_* so M82 boxes under a single ref_* directory
        # are split correctly; fall back to group_*/ref_* for other layouts.
        for cand in (aligned_to, original_ref, aligned):
            if not cand or cand == 'NA':
                continue
            m = _COADD_RE.search(cand)
            if m:
                ref = (int(m.group(1)), int(m.group(2)))
                break
            m = _REF_RE.search(cand)
            if m:
                ref = (int(m.group(1)), int(m.group(2)))
                break
        rows.append(
            dict(
                status=status,
                disp=disp,
                aligned=aligned,
                ref=ref,
                filter=parts[filt_i],
            )
        )
    return rows


def jhats_for_ref(
    rows: list[dict],
    *,
    group: int,
    box: int,
    max_dispersion_mas: float = MAX_USABLE_DISPERSION_MAS,
) -> list[Path]:
    """Unique existing SUCCESS JHAT paths for one reference, quality-filtered."""
    out: list[Path] = []
    seen: set[str] = set()
    for r in rows:
        if r['ref'] != (group, box):
            continue
        if r['status'] != 'SUCCESS':
            continue
        if not np.isfinite(r['disp']) or r['disp'] >= max_dispersion_mas:
            continue
        aligned = r['aligned']
        if aligned == 'NA':
            continue
        path = Path(aligned)
        if not path.is_file() and Path(str(aligned) + '.fits').is_file():
            path = Path(str(aligned) + '.fits')
        if not path.is_file():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        # Drop cutouts / non-full-frame MIRI products (rejected upstream too).
        try:
            from st123.utils.helpers import is_full_frame_miri

            if not is_full_frame_miri(path):
                continue
        except Exception:
            pass
        seen.add(key)
        out.append(path)
    return sorted(out)


def discover_dataset(
    spec: DatasetSpec,
    *,
    min_miri: int = MIN_MIRI_IMAGES,
    max_dispersion_mas: float = MAX_USABLE_DISPERSION_MAS,
    write_lists: bool = True,
) -> list[RefWarmstartPlan]:
    """
    Build warm-start plans for every reference in *spec*'s alignment summary.

    References with ``n_miri < min_miri`` are returned with ``eligible=False``.
    """
    rows = parse_alignment_summary(alignment_summary_path(spec))
    by_ref: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in rows:
        if r['ref'] is not None:
            by_ref[r['ref']].append(r)

    plans: list[RefWarmstartPlan] = []
    LIST_DIR.mkdir(parents=True, exist_ok=True)

    for group, box in sorted(by_ref):
        jhats = jhats_for_ref(
            by_ref[(group, box)],
            group=group,
            box=box,
            max_dispersion_mas=max_dispersion_mas,
        )
        nircam = nircam_dir(spec, group, box)
        nircam_ready = (nircam / 'dolphot.param').is_file() and any(
            p.is_file() and p.stat().st_size > 0 and p.name.count('.') == 1
            for p in nircam.glob('*.phot')
        )
        n_miri = len(jhats)
        skip = ''
        eligible = True
        if n_miri < min_miri:
            eligible = False
            skip = f'n_miri={n_miri} < min_miri={min_miri}'
        elif spec.require_nircam_dolphot and not nircam_ready:
            eligible = False
            skip = f'NIRCam dolphot not ready under {nircam}'

        if write_lists and jhats:
            jhat_list_path(spec, group, box).write_text(
                '\n'.join(str(p) for p in jhats) + '\n'
            )

        plans.append(
            RefWarmstartPlan(
                dataset=spec.name,
                group=group,
                box=box,
                n_miri=n_miri,
                jhat_paths=tuple(jhats),
                nircam_ready=nircam_ready,
                eligible=eligible,
                skip_reason=skip,
            )
        )
    return plans


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            f'Discover M51/M82 warm-start refs with ≥{MIN_MIRI_IMAGES} MIRI JHAT frames.'
        )
    )
    p.add_argument(
        '--dataset',
        choices=sorted(DATASETS),
        action='append',
        default=None,
        help='Dataset(s) to scan (default: both).',
    )
    p.add_argument(
        '--min-miri',
        type=int,
        default=MIN_MIRI_IMAGES,
        help='Minimum unique MIRI JHAT frames per reference.',
    )
    p.add_argument(
        '--max-dispersion-mas',
        type=float,
        default=MAX_USABLE_DISPERSION_MAS,
        help='Drop SUCCESS rows with dispersion at/above this (sentinel filter).',
    )
    p.add_argument(
        '--no-write',
        action='store_true',
        help='Do not rewrite per-ref JHAT list files.',
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    names = args.dataset or sorted(DATASETS)
    for name in names:
        spec = DATASETS[name]
        plans = discover_dataset(
            spec,
            min_miri=args.min_miri,
            max_dispersion_mas=args.max_dispersion_mas,
            write_lists=not args.no_write,
        )
        elig = [p for p in plans if p.eligible]
        skip = [p for p in plans if not p.eligible]
        print(f'\n=== {name} (min_miri={args.min_miri}) ===')
        print(f'eligible: {len(elig)}  skipped: {len(skip)}')
        for p in elig:
            print(
                f'  KEEP  {p.ref_key:8s}  n_miri={p.n_miri:3d}  '
                f'nircam_ready={p.nircam_ready}'
            )
        for p in skip:
            print(f'  SKIP  {p.ref_key:8s}  {p.skip_reason}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
