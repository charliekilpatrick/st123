#!/usr/bin/env python3
"""
End-to-end MIRI–reference overlap + alignment pipeline.

Expected dataset layout under ``--data-dir``::

    <data-dir>/JWST/MIRI/<FILTER>/<obsid>/mastDownload/JWST/*_mirimage/*_cal.fits
    <data-dir>/reference/group_*/ref_*/coadd*i2d.fits

    Legacy layouts ``<FILTER>/<obsid>/...`` and ``<FILTER>_<obsid>/...`` are
    still discovered for older trees.

Steps
-----
1. Discover MIRI ``*_cal.fits`` and reference ``coadd*i2d.fits`` under
   ``--data-dir`` (default: ``/data/rwisenbaker/jwst_data/M51``).
2. For each MIRI frame, record:
   - the reference with maximum footprint overlap, and
   - every reference with any nonzero overlap.
3. Align MIRI frames filter-by-filter from blue→red (F560W, then F770W,
   then F1000W, …). Within each filter, frames are aligned in parallel
   (``--workers``).
4. If reference-image alignment fails, or succeeds but exceeds the per-filter
   REFERENCE quality-hold threshold (or uniform ``--max-nircam-dispersion-mas``),
   try relative MIRI→MIRI alignment against the best overlapping parent
   (quality-aware: low parent dispersion, modest wavelength gap, large
   overlap). Absolute dispersion is the quadrature sum of the parent absolute
   dispersion and the new relative dispersion. MIRI_REL is kept only when it
   improves on REFERENCE; otherwise the REFERENCE WCS is retained as SUCCESS.
   Provenance (``ALGNMODE``, ``ALGNREF``, ``ALGNTO``) is written to the JHAT
   header. F560W never quality-holds (must stay on REFERENCE).
5. Reject MIRI frames whose cumulative (union) reference footprint coverage
   of the MIRI ROI is below ``--min-ref-overlap-frac`` (default 0.02) before
   alignment (logged to console / overlap summaries only; they never enter
   ``alignment_dispersion`` / JHAT, including MIRI→MIRI fallback, and are
   omitted from ``alignment_summary.txt``).
6. Write ``<data-dir>/<name>_alignment_summary.txt`` only for frames that
   proceed to alignment, with per-frame ``SUCCESS``/``FAILURE`` status,
   ``ref_overlap_frac`` (unique MIRI-ROI fraction covered by the union of all
   overlapping reference footprints), ``align_mode`` (``REFERENCE`` or
   ``MIRI_REL``), filter, calibrator count, dispersion, aligned JHAT path, and
   provenance (updated live after each finished frame).

Example::

    Use ``python -m st123.scripts.alignment_wrap`` (CLI in scripts/).

Legacy sequential pair mode (original ``alignment_wrap`` behavior)::

    python alignment_wrap.py --legacy-overlap-file overlap_summary.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import traceback
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


def _resolve_repo_root(explicit: Path | None = None) -> Path:
    """Locate the installable st123 package / repo root for worker bootstrap."""
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not (root / 'st123' / 'alignment' / 'alignment_wrap.py').is_file() and not (
            root / 'alignment_wrap.py'
        ).is_file():
            # Allow pointing at the package directory itself.
            if not (root / 'alignment_wrap.py').is_file():
                raise FileNotFoundError(f'--repo does not look like st123: {root}')
        return root

    here = Path(__file__).resolve().parent  # st123/
    candidates = [
        here.parent,  # repo root containing st123/
        Path.cwd(),
        Path('/data/rwisenbaker/st123'),
    ]
    for cand in candidates:
        if (cand / 'st123' / 'alignment' / 'alignment_wrap.py').is_file():
            return cand.resolve()
        if (cand / 'alignment_wrap.py').is_file():
            return cand.resolve()
    # Fall back to the package directory so spawn workers can still import.
    return here.parent.resolve()


def _bootstrap_imports(repo: Path) -> None:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


@dataclass(frozen=True)
class FrameOverlaps:
    """Best reference plus every reference with nonzero overlap for one MIRI frame."""

    miri_path: str
    best: object  # image_overlap.BestOverlap
    overlapping: list  # list[image_overlap.OverlapResult]
    # Unique MIRI-ROI fraction covered by the union of all overlapping refs.
    union_overlap_fraction: float = 0.0

    def to_dict(self) -> dict:
        return {
            'miri_path': self.miri_path,
            'best': {
                'ref_path': self.best.ref_path,
                'overlap_area': asdict(self.best.overlap_area),
            },
            'overlapping': [
                {
                    'ref_path': r.ref_path,
                    'overlap_area': asdict(r.overlap_area),
                    'ref_area': asdict(r.ref_area),
                }
                for r in self.overlapping
            ],
            'union_overlap_fraction': float(self.union_overlap_fraction),
        }


@dataclass
class AlignmentSummaryRow:
    """One row of the galaxy alignment summary table."""

    miri_path: str
    filter: str
    status: str
    n_calibrators: int | str
    dispersion_mas: float | str
    aligned_path: str = 'NA'
    align_mode: str = 'NA'
    original_ref: str = 'NA'
    aligned_to: str = 'NA'
    # Cumulative unique fraction of MIRI ROI covered by all overlapping refs.
    ref_overlap_frac: float | str = 'NA'

    def format_line(self, widths: dict[str, int]) -> str:
        disp = (
            f'{self.dispersion_mas:.3f}'
            if isinstance(self.dispersion_mas, float)
            else str(self.dispersion_mas)
        )
        ov = (
            f'{self.ref_overlap_frac:.4f}'
            if isinstance(self.ref_overlap_frac, float)
            else str(self.ref_overlap_frac)
        )
        ncal = str(self.n_calibrators)
        return (
            f'{self.miri_path:<{widths["miri_path"]}}  '
            f'{self.filter:<{widths["filter"]}}  '
            f'{self.status:<{widths["status"]}}  '
            f'{ov:>{widths["ref_overlap_frac"]}}  '
            f'{ncal:>{widths["n_calibrators"]}}  '
            f'{disp:>{widths["dispersion_mas"]}}  '
            f'{self.align_mode:<{widths["align_mode"]}}  '
            f'{self.aligned_path:<{widths["aligned_path"]}}  '
            f'{self.original_ref:<{widths["original_ref"]}}  '
            f'{self.aligned_to:<{widths["aligned_to"]}}'
        )


def read_miri_filter(miri_path: str) -> str:
    """Return FILTER from the MIRI FITS primary (or SCI) header."""
    from astropy.io import fits

    with fits.open(miri_path) as hdul:
        filt = hdul[0].header.get('FILTER')
        if not filt and 'SCI' in hdul:
            filt = hdul['SCI'].header.get('FILTER')
    return str(filt) if filt else 'UNKNOWN'


def find_jhat_product(outdir: Path, miri_path: str) -> Path | None:
    """Locate the JHAT FITS product for a MIRI frame."""
    stem = Path(miri_path).name.replace('_cal.fits', '_jhat.fits').replace(
        '_i2d.fits', '_jhat.fits'
    )
    candidate = outdir / stem
    if candidate.is_file():
        return candidate
    matches = sorted(outdir.glob('*jhat*.fits'))
    return matches[0] if matches else None


def _normalize_align_mode(align_mode: str | None) -> str:
    """Map legacy ``NIRCAM`` labels to ``REFERENCE``; otherwise uppercase."""
    mode = str(align_mode or 'NA').upper()
    if mode == 'NIRCAM':
        return 'REFERENCE'
    return mode


def _is_reference_quality_hold(row: AlignmentSummaryRow) -> bool:
    """
    True for a REFERENCE solution held as PENDING after the dispersion cut.

    These rows keep finite metrics / JHAT paths so MIRI_REL can be tried.
    They are omitted from the live alignment summary until MIRI_REL finishes
    (kept if improved) or the REFERENCE solution is restored as SUCCESS.
    """
    return (
        row.status == 'PENDING'
        and _normalize_align_mode(row.align_mode) == 'REFERENCE'
        and isinstance(row.dispersion_mas, float)
        and bool(row.aligned_path)
        and str(row.aligned_path) != 'NA'
    )


def harvest_alignment_metrics(
    miri_path: str,
    outdir: Path,
    *,
    ran_ok: bool,
    default_align_mode: str = 'NA',
    default_original_ref: str = 'NA',
    default_aligned_to: str = 'NA',
    ref_overlap_frac: float | str = 'NA',
) -> AlignmentSummaryRow:
    """
    Build a summary row from alignment products / headers.

    SUCCESS requires a JHAT product with a finite final mean dispersion
    (``JWDISPM`` / ``GADISPM``, stored in arcsec, reported in mas). Method is
    recorded separately in ``align_mode`` (``REFERENCE`` or ``MIRI_REL``).
    ``ref_overlap_frac`` is the unique MIRI-ROI fraction covered by the union
    of all overlapping reference footprints (geometry; independent of JHAT).
    """
    from astropy.io import fits

    filt = read_miri_filter(miri_path)
    empty = dict(
        miri_path=miri_path,
        filter=filt,
        status='FAILURE',
        n_calibrators='NA',
        dispersion_mas='NA',
        aligned_path='NA',
        align_mode='NA',
        original_ref='NA',
        aligned_to='NA',
        ref_overlap_frac=ref_overlap_frac,
    )
    if not ran_ok:
        return AlignmentSummaryRow(**empty)

    jhat = find_jhat_product(outdir, miri_path)
    if jhat is None:
        return AlignmentSummaryRow(**empty)
    aligned_path = str(jhat.resolve())

    with fits.open(jhat) as hdul:
        hdr = hdul[0].header
        if hdr.get('FILTER'):
            filt = str(hdr['FILTER'])
        disp_std = hdr.get('JWDISPS', hdr.get('GADISPS'))
        disp_mean = hdr.get('JWDISPM', hdr.get('GADISPM'))
        n_cal = hdr.get('JWNCAL', hdr.get('GANCAL'))
        align_mode = _normalize_align_mode(
            hdr.get('ALGNMODE', default_align_mode) or default_align_mode
        )
        original_ref = str(hdr.get('ALGNREF', default_original_ref) or default_original_ref)
        aligned_to = str(hdr.get('ALGNTO', default_aligned_to) or default_aligned_to)

    # Soft-failure path in align_jwst_image writes JWDISPS as the string 'NaN'.
    rejected = isinstance(disp_std, str) and disp_std.upper() == 'NAN'

    dispersion_mas: float | str = 'NA'
    try:
        if disp_mean is not None and not (
            isinstance(disp_mean, str) and str(disp_mean).upper() == 'NAN'
        ):
            disp_val = float(disp_mean)
            if math.isfinite(disp_val):
                dispersion_mas = disp_val * 1000.0
    except (TypeError, ValueError):
        dispersion_mas = 'NA'

    n_calibrators: int | str = 'NA'
    try:
        if n_cal is not None and not (
            isinstance(n_cal, str) and str(n_cal).upper() == 'NAN'
        ):
            n_calibrators = int(n_cal)
    except (TypeError, ValueError):
        n_calibrators = 'NA'

    if rejected or dispersion_mas == 'NA':
        return AlignmentSummaryRow(**empty)

    return AlignmentSummaryRow(
        miri_path=miri_path,
        filter=filt,
        status='SUCCESS',
        n_calibrators=n_calibrators,
        dispersion_mas=dispersion_mas,
        aligned_path=aligned_path,
        align_mode=align_mode,
        original_ref=original_ref,
        aligned_to=aligned_to,
        ref_overlap_frac=ref_overlap_frac,
    )


def write_alignment_summary(
    rows: list[AlignmentSummaryRow],
    outfile: Path,
) -> Path:
    """Write a plain ASCII alignment summary table and return its path."""
    outfile = Path(outfile).expanduser().resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        'miri_path': 'miri_path',
        'filter': 'filter',
        'status': 'status',
        'ref_overlap_frac': 'ref_overlap_frac',
        'n_calibrators': 'n_calibrators',
        'dispersion_mas': 'dispersion_mas',
        'align_mode': 'align_mode',
        'aligned_path': 'aligned_path',
        'original_ref': 'original_ref',
        'aligned_to': 'aligned_to',
    }

    def _disp_len(r: AlignmentSummaryRow) -> int:
        if isinstance(r.dispersion_mas, float):
            return len(f'{r.dispersion_mas:.3f}')
        return len(str(r.dispersion_mas))

    def _ov_len(r: AlignmentSummaryRow) -> int:
        if isinstance(r.ref_overlap_frac, float):
            return len(f'{r.ref_overlap_frac:.4f}')
        return len(str(r.ref_overlap_frac))

    widths = {
        'miri_path': max(
            [len(headers['miri_path'])] + [len(r.miri_path) for r in rows] + [1]
        ),
        'filter': max(
            [len(headers['filter'])] + [len(r.filter) for r in rows] + [1]
        ),
        'status': max(
            [len(headers['status'])] + [len(r.status) for r in rows] + [1]
        ),
        'ref_overlap_frac': max(
            [len(headers['ref_overlap_frac'])]
            + [_ov_len(r) for r in rows]
            + [1]
        ),
        'n_calibrators': max(
            [len(headers['n_calibrators'])]
            + [len(str(r.n_calibrators)) for r in rows]
            + [1]
        ),
        'dispersion_mas': max(
            [len(headers['dispersion_mas'])]
            + [_disp_len(r) for r in rows]
            + [1]
        ),
        'align_mode': max(
            [len(headers['align_mode'])] + [len(r.align_mode) for r in rows] + [1]
        ),
        'aligned_path': max(
            [len(headers['aligned_path'])]
            + [len(r.aligned_path) for r in rows]
            + [1]
        ),
        'original_ref': max(
            [len(headers['original_ref'])]
            + [len(r.original_ref) for r in rows]
            + [1]
        ),
        'aligned_to': max(
            [len(headers['aligned_to'])] + [len(r.aligned_to) for r in rows] + [1]
        ),
    }

    header = (
        f'{headers["miri_path"]:<{widths["miri_path"]}}  '
        f'{headers["filter"]:<{widths["filter"]}}  '
        f'{headers["status"]:<{widths["status"]}}  '
        f'{headers["ref_overlap_frac"]:>{widths["ref_overlap_frac"]}}  '
        f'{headers["n_calibrators"]:>{widths["n_calibrators"]}}  '
        f'{headers["dispersion_mas"]:>{widths["dispersion_mas"]}}  '
        f'{headers["align_mode"]:<{widths["align_mode"]}}  '
        f'{headers["aligned_path"]:<{widths["aligned_path"]}}  '
        f'{headers["original_ref"]:<{widths["original_ref"]}}  '
        f'{headers["aligned_to"]:<{widths["aligned_to"]}}'
    )
    lines = [header, '-' * len(header)]
    lines.extend(r.format_line(widths) for r in rows)
    lines.append('')
    # Atomic replace so a live reader never sees a partially written table.
    tmp = outfile.with_name(outfile.name + '.tmp')
    tmp.write_text('\n'.join(lines))
    tmp.replace(outfile)
    return outfile


def _looks_like_filter(token: str) -> bool:
    """True for names like F560W / F1000W / F150W2."""
    return bool(re.fullmatch(r'F\d+[WMN]\d*', str(token).upper()))


# Path segments that are telescope/instrument roots, not filters.
_PATH_SKIP_TOKENS = frozenset(
    {
        'JWST',
        'HST',
        'ROMAN',
        'EUCLID',
        'MIRI',
        'NIRCAM',
        'NIRISS',
        'ACS',
        'WFC3',
        'WFPC2',
        'WFI',
        'VIS',
        'NISP',
        'MASTDOWNLOAD',
    }
)


def discover_miri_images(data_dir: Path) -> list[str]:
    """
    Sorted MIRI cal images under ``data_dir``.

    Preferred layout::

        <data-dir>/JWST/MIRI/<FILTER>/<obsid>/mastDownload/JWST/*_mirimage/*_cal.fits

    Also accepts older layouts::

        <data-dir>/<FILTER>/<obsid>/mastDownload/JWST/*_mirimage/*_cal.fits
        <data-dir>/<FILTER>_<obsid>/mastDownload/JWST/*_mirimage/*_cal.fits
    """
    data_dir = Path(data_dir)
    found: set[str] = set()
    for path in data_dir.glob('**/mastDownload/JWST/*_mirimage/*_cal.fits'):
        found.add(str(path.resolve()))
    return sorted(found)


def discover_ref_images(data_dir: Path) -> list[str]:
    """Sorted reference coadds under ``data_dir/reference/group_*/ref_*/``."""
    return sorted(
        str(p.resolve())
        for p in Path(data_dir).glob('reference/group_*/ref_*/coadd*i2d.fits')
    )


def parse_filters_arg(filters: str | None) -> list[str] | None:
    """
    Parse a comma-separated filter list (e.g. ``F560W`` or ``F560W,F770W``).

    Returns ``None`` when no filter restriction is requested.
    """
    if filters is None or not str(filters).strip():
        return None
    parsed = []
    for part in str(filters).split(','):
        name = part.strip().upper()
        if not name:
            continue
        parsed.append(name)
    return parsed or None


def filter_name_from_miri_path(miri_path: str) -> str | None:
    """
    Infer the MIRI filter from the MAST download path.

    Supports::

        .../JWST/MIRI/<FILTER>/<obsid>/mastDownload/JWST/...
        .../<FILTER>/<obsid>/mastDownload/JWST/...
        .../<FILTER>_<obsid>/mastDownload/JWST/...
    """
    parts = Path(miri_path).parts
    for i, part in enumerate(parts):
        if part != 'mastDownload' or i < 1:
            continue
        # Walk upward from the directory containing mastDownload.
        for j in range(i - 1, -1, -1):
            tok = str(parts[j])
            if tok.isdigit():
                continue
            if tok.upper() in _PATH_SKIP_TOKENS:
                continue
            if _looks_like_filter(tok):
                return tok.upper()
            # Legacy: <FILTER>_<obsid>
            head = tok.split('_', 1)[0]
            if _looks_like_filter(head):
                return head.upper()
        break
    return None


def filter_miri_images(
    miri_images: list[str],
    filters: list[str] | None,
) -> list[str]:
    """Keep MIRI images whose path filter is in ``filters`` (case-insensitive)."""
    if not filters:
        return list(miri_images)
    wanted = {f.upper() for f in filters}
    selected = []
    for path in miri_images:
        name = filter_name_from_miri_path(path)
        if name is not None and name in wanted:
            selected.append(path)
    return selected


def find_frame_overlaps(
    miri_images: list[str],
    refs: list[str],
    *,
    MirIFootprint,
    BestOverlap,
    compute_overlap,
) -> list[FrameOverlaps]:
    """Compute best-overlap and all-nonzero-overlap refs for each MIRI frame."""
    results: list[FrameOverlaps] = []

    for image in miri_images:
        print(f'MIRI: {image}')
        miri = MirIFootprint.from_fits(image)
        print(f'  illuminated S_REGION: {miri.s_region.to_string()}')
        print(
            f'  WCS pixel solid angle: {miri.pixel_area_arcmin2:.8e} '
            f'arcmin^2 / pixel'
        )
        print(f'  illuminated area: {miri.area.format()}')

        overlapping = []
        best = None

        for ref in refs:
            try:
                result = compute_overlap(miri, ref)
            except Exception as exc:
                print(f'  FAILED for ref {ref}: {exc}')
                continue

            print(f'  ref: {ref}')
            print(f'    S_REGION: {result.ref_s_region.to_string()}')
            print(f'    ref area: {result.ref_area.format()}')
            print(f'    overlap area: {result.overlap_area.format()}')

            if result.overlap_area.pixels2 > 0.0:
                overlapping.append(result)

            if best is None or result.overlap_area.pixels2 > best.overlap_area.pixels2:
                best = BestOverlap(
                    science_path=image,
                    ref_path=result.ref_path,
                    overlap_area=result.overlap_area,
                )

        if best is None or best.overlap_area.pixels2 <= 0.0:
            best = BestOverlap(
                science_path=image,
                ref_path=None,
                overlap_area=miri.metrics(0.0),
            )

        overlapping.sort(key=lambda r: r.overlap_area.pixels2, reverse=True)

        from st123.mosaic.image_overlap import compute_cumulative_overlap_fraction

        union_frac = compute_cumulative_overlap_fraction(
            miri, [r.ref_path for r in overlapping]
        )

        print(
            f'Overlap maximized: MIRI image: {best.miri_path}, '
            f'Reference image: {best.ref_path}, '
            f'Max overlap area: {best.overlap_area.pixels2:.3f} pixels^2 '
            f'({best.overlap_area.arcmin2:.6f} arcmin^2, '
            f'{best.overlap_area.fraction_of_miri_roi:.4f} of MIRI illuminated ROI); '
            f'{len(overlapping)} reference(s) with any overlap; '
            f'union coverage {union_frac:.4f} of MIRI ROI'
        )
        if overlapping:
            print('  References with any overlap (largest first):')
            for result in overlapping:
                print(
                    f'    {result.ref_path}: '
                    f'{result.overlap_area.pixels2:.3f} pixels^2 '
                    f'({result.overlap_area.fraction_of_miri_roi:.4f} of MIRI ROI)'
                )
        print()

        results.append(
            FrameOverlaps(
                miri_path=image,
                best=best,
                overlapping=overlapping,
                union_overlap_fraction=union_frac,
            )
        )

    return results


def write_overlap_summaries(
    frames: list[FrameOverlaps],
    outdir: Path,
) -> tuple[Path, Path]:
    """Write text + JSON summaries of best and any-overlap references."""
    outdir.mkdir(parents=True, exist_ok=True)
    txt_path = outdir / 'overlap_summary.txt'
    json_path = outdir / 'overlap_summary.json'

    lines: list[str] = []
    for frame in frames:
        best = frame.best
        lines.append(
            f'Overlap maximized: MIRI image: {best.miri_path}, '
            f'Reference image: {best.ref_path}, '
            f'Max overlap area: {best.overlap_area.pixels2:.3f} pixels^2 '
            f'({best.overlap_area.arcmin2:.6f} arcmin^2, '
            f'{best.overlap_area.fraction_of_miri_roi:.4f} of MIRI illuminated ROI); '
            f'{len(frame.overlapping)} reference(s) with any overlap; '
            f'union coverage {frame.union_overlap_fraction:.4f} of MIRI ROI\n'
        )
        for result in frame.overlapping:
            lines.append(
                f'  any-overlap: {result.ref_path} '
                f'{result.overlap_area.pixels2:.3f} pixels^2 '
                f'({result.overlap_area.fraction_of_miri_roi:.4f} of MIRI ROI)\n'
            )
        lines.append('\n')

    txt_path.write_text(''.join(lines))
    payload = {
        'n_miri': len(frames),
        'n_with_overlap': sum(1 for f in frames if f.best.ref_path is not None),
        'frames': [f.to_dict() for f in frames],
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f'Wrote overlap summary: {txt_path}')
    print(f'Wrote overlap JSON:    {json_path}')
    return txt_path, json_path


def alignment_outdir_for(miri_path: str) -> Path:
    """``<directory of MIRI cal file>/alignment_output``."""
    return Path(miri_path).resolve().parent / 'alignment_output'


def run_legacy_overlap_file_pipeline(
    overlap_file: Path,
    *,
    outdir: Path = Path('alignment_output_auto'),
    success_file: Path = Path('successful_alignments.txt'),
    fail_file: Path = Path('failed_alignments.txt'),
    filters: tuple[str, ...] = ('F560W', 'F770W'),
    alignment_script: str = 'alignment_dispersion.py',
    plot: bool = True,
    verbose: bool = True,
) -> int:
    """
    Original ``alignment_wrap`` behavior: align each MIRI/reference pair listed
    in an overlap text file via ``alignment_dispersion.py``.
    """
    overlap_file = Path(overlap_file).expanduser().resolve()
    if not overlap_file.is_file():
        print(f'ERROR: legacy overlap file not found: {overlap_file}', file=sys.stderr)
        return 1

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    success_file = Path(success_file)
    fail_file = Path(fail_file)
    filt_tokens = tuple(f.upper() for f in filters)

    n_ok = 0
    n_fail = 0
    with open(success_file, 'w') as success, open(fail_file, 'w') as failed:
        with open(overlap_file) as file:
            for line in file:
                if 'Overlap maximized' not in line:
                    continue
                if not any(tok in line for tok in filt_tokens):
                    continue

                align_image = line.split('MIRI image: ')[1].split(
                    ', Reference image: '
                )[0]
                ref_image = line.split(', Reference image: ')[1].split(
                    ', Max overlap area'
                )[0]

                align_name = Path(align_image).stem
                ref_name = Path(ref_image).stem
                pair_outdir = os.path.join(
                    str(outdir), f'{align_name}_aligned_to_{ref_name}'
                )

                command = [
                    sys.executable,
                    alignment_script,
                    '--ref',
                    ref_image,
                    '--align',
                    align_image,
                    '--outdir',
                    pair_outdir,
                ]
                if plot:
                    command.append('--plot')
                if verbose:
                    command.append('--verbose')

                result = subprocess.run(command)
                if result.returncode == 0:
                    success.write(
                        f'MIRI image: {align_image}, \n'
                        f'Reference image: {ref_image}\n\n'
                    )
                    success.flush()
                    n_ok += 1
                else:
                    failed.write(
                        f'MIRI image: {align_image}, \n'
                        f'Reference image: {ref_image}\n\n'
                    )
                    failed.flush()
                    n_fail += 1

    print('Done')
    print(f'Successful pairs written to {success_file} ({n_ok})')
    print(f'Failed pairs written to {fail_file} ({n_fail})')
    return 1 if n_fail else 0


def resolve_data_dir(
    *,
    data_dir: Path | None,
    data_root: Path | None,
    galaxy: str | None,
    default_data_dir: Path,
) -> tuple[Path, str]:
    """
    Resolve ``(data_dir, dataset_label)``.

    Preference order for the dataset root:
      1. ``--data-dir``
      2. ``--data-root`` / ``--galaxy`` (legacy)
      3. ``default_data_dir``
    """
    if data_dir is not None:
        root = Path(data_dir).expanduser().resolve()
    elif data_root is not None:
        label = galaxy or 'M51'
        root = (Path(data_root).expanduser().resolve() / label).resolve()
    else:
        root = Path(default_data_dir).expanduser().resolve()

    label = galaxy or root.name
    return root, label


def run_overlaps(
    args: argparse.Namespace,
    *,
    MirIFootprint,
    BestOverlap,
    compute_overlap,
) -> list[FrameOverlaps]:
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Data directory not found: {data_dir}')

    filters = parse_filters_arg(getattr(args, 'filters', None))
    miri_images = filter_miri_images(discover_miri_images(data_dir), filters)
    refs = discover_ref_images(data_dir)
    if args.limit is not None:
        miri_images = miri_images[: args.limit]

    if not miri_images:
        msg = (
            f'No MIRI *_cal.fits found under '
            f'{data_dir}/JWST/MIRI/<FILTER>/<obsid>/mastDownload/JWST/'
        )
        if filters:
            msg += f' for filters {",".join(filters)}'
        raise FileNotFoundError(msg)
    if not refs:
        raise FileNotFoundError(
            f'No reference coadd*i2d.fits found under {data_dir}/reference/'
        )

    print(f'Repo:              {args.repo}')
    print(f'Data dir:          {data_dir}')
    print(f'Dataset label:     {args.galaxy}')
    print(f'Filters:           {", ".join(filters) if filters else "ALL"}')
    print(f'MIRI images:       {len(miri_images)}')
    print(f'Reference images:  {len(refs)}')
    print()

    frames = find_frame_overlaps(
        miri_images,
        refs,
        MirIFootprint=MirIFootprint,
        BestOverlap=BestOverlap,
        compute_overlap=compute_overlap,
    )
    overlap_outdir = Path(
        args.overlap_outdir or (data_dir / 'overlap')
    ).expanduser().resolve()
    write_overlap_summaries(frames, overlap_outdir)
    return frames


def _frame_ref_images(frame: FrameOverlaps | dict) -> tuple[str, list[str], str | None]:
    """Return ``(miri_path, ordered_ref_images, best_ref)`` for one overlap frame."""
    if isinstance(frame, dict):
        miri_path = frame['miri_path']
        best_ref = frame['best']['ref_path']
        overlapping = [r['ref_path'] for r in frame.get('overlapping', [])]
    else:
        miri_path = frame.miri_path
        best_ref = frame.best.ref_path
        overlapping = [r.ref_path for r in frame.overlapping]

    ref_images = overlapping or ([best_ref] if best_ref else [])
    ordered: list[str] = []
    for path in ([best_ref] if best_ref else []) + list(ref_images):
        if path and path not in ordered:
            ordered.append(path)
    return miri_path, ordered, best_ref


def _frame_ref_overlap_frac(frame: FrameOverlaps | dict) -> float:
    """
    Unique MIRI-ROI fraction covered by the union of overlapping references.

    Uses a cached ``union_overlap_fraction`` when present; otherwise recomputes
    from the overlapping reference paths.
    """
    if isinstance(frame, dict):
        cached = frame.get('union_overlap_fraction')
        if cached is not None:
            try:
                return float(cached)
            except (TypeError, ValueError):
                pass
        miri_path, ref_images, _best = _frame_ref_images(frame)
    else:
        try:
            return float(frame.union_overlap_fraction)
        except (TypeError, ValueError, AttributeError):
            pass
        miri_path, ref_images, _best = _frame_ref_images(frame)

    if not ref_images:
        return 0.0
    from st123.mosaic.image_overlap import MirIFootprint, compute_cumulative_overlap_fraction

    return compute_cumulative_overlap_fraction(
        MirIFootprint.from_fits(miri_path), ref_images
    )


def frame_has_nircam_overlap(
    frame: FrameOverlaps | dict,
    *,
    min_ref_overlap_frac: float = 0.02,
) -> bool:
    """
    True if cumulative reference coverage of the MIRI ROI meets the threshold.

    ``min_ref_overlap_frac`` is the unique (union) fraction of the MIRI
    illuminated footprint covered by all overlapping reference images.
    """
    _miri_path, ref_images, best_ref = _frame_ref_images(frame)
    if not ref_images or best_ref is None:
        return False
    return _frame_ref_overlap_frac(frame) >= float(min_ref_overlap_frac)


def reject_zero_nircam_overlap_frames(
    frames: list[FrameOverlaps] | list[dict],
    *,
    min_ref_overlap_frac: float = 0.02,
) -> tuple[list[FrameOverlaps] | list[dict], int]:
    """
    Drop MIRI frames with insufficient cumulative reference footprint overlap.

    Frames whose unique (union) reference coverage of the MIRI ROI is below
    ``min_ref_overlap_frac`` are rejected. Returns ``(kept_frames, n_rejected)``.
    Rejected frames remain in overlap summaries only — they are not written to
    ``alignment_summary.txt`` and are not passed to alignment (including
    MIRI→MIRI fallback).
    """
    min_frac = float(min_ref_overlap_frac)
    kept: list[FrameOverlaps | dict] = []
    n_rejected = 0

    for frame in frames:
        miri_path, _ref_images, best_ref = _frame_ref_images(frame)
        filt = filter_name_from_miri_path(miri_path) or read_miri_filter(miri_path)
        ov_frac = _frame_ref_overlap_frac(frame)
        if best_ref and ov_frac >= min_frac:
            kept.append(frame)
            continue

        n_rejected += 1
        print(
            f'REJECT {Path(miri_path).name}  {filt}  '
            f'ref_overlap_frac={ov_frac:.4f} < {min_frac:.4f} '
            f'(excluded from alignment)',
            flush=True,
        )

    if n_rejected:
        print(
            f'Rejected {n_rejected} MIRI frame(s) with '
            f'ref_overlap_frac < {min_frac:.4f}; '
            f'{len(kept)} frame(s) remain for alignment',
            flush=True,
        )
    return kept, n_rejected


def _group_frames_by_filter(
    frames: list[FrameOverlaps] | list[dict],
) -> OrderedDict[str, list[FrameOverlaps | dict]]:
    """Group frames by filter, preserving blue→red order of first appearance."""
    from st123.alignment.alignment_fallback import sort_frames_blue_to_red

    ordered = sort_frames_blue_to_red(
        frames, filter_from_path=filter_name_from_miri_path
    )
    groups: OrderedDict[str, list[FrameOverlaps | dict]] = OrderedDict()
    for frame in ordered:
        miri_path, _, _ = _frame_ref_images(frame)
        filt = filter_name_from_miri_path(miri_path) or read_miri_filter(miri_path)
        groups.setdefault(filt, []).append(frame)
    return groups


def _format_worker_done(result) -> str:
    """One-line DONE status for an alignment worker result."""
    base = Path(result.miri_path).name
    filt = (
        result.row.get('filter')
        or getattr(result, 'filter', None)
        or 'UNKNOWN'
    )
    status = str(result.row.get('status', 'FAILURE'))
    mode = _normalize_align_mode(result.row.get('align_mode'))
    disp = result.row.get('dispersion_mas', 'NA')
    disp_s = f'{disp:.3f}' if isinstance(disp, float) else str(disp)
    if status == 'SUCCESS':
        return (
            f'DONE  {base}  {filt}  SUCCESS  align_mode={mode}  '
            f'dispersion_mas={disp_s}'
        )
    if status == 'PENDING':
        return (
            f'DONE  {base}  {filt}  PENDING  align_mode={mode}  '
            f'dispersion_mas={disp_s} (over threshold; try MIRI_REL)'
        )
    if status in ('SKIP', 'REJECTED'):
        return f'DONE  {base}  {filt}  {status}'
    return f'DONE  {base}  {filt}  FAILURE'


def _needs_miri_fallback(row: AlignmentSummaryRow) -> bool:
    """True if a frame should enter MIRI_REL after the reference-align pass."""
    return row.status not in ('SUCCESS', 'REJECTED', 'SKIP')


def _run_jobs_parallel(
    jobs: list[dict],
    worker,
    *,
    workers: int,
    label: str,
    on_result=None,
) -> list:
    """
    Run picklable worker jobs with up to ``workers`` processes (or serially).

    Emits a brief ``START`` line when each job begins and invokes ``on_result``
    (if given) as each job finishes so the caller can log ``DONE`` / update
    the summary without interleaving JHAT chatter.
    """
    import multiprocessing as mp

    from st123.alignment.alignment_parallel import AlignWorkerResult

    if not jobs:
        return []

    n_workers = max(1, int(workers))
    print(
        f'{label}: {len(jobs)} job(s), workers={min(n_workers, len(jobs))}',
        flush=True,
    )

    def _start_line(job: dict) -> None:
        print(f'START {Path(job["miri_path"]).name}', flush=True)

    def _handle(result) -> None:
        if on_result is not None:
            on_result(result)

    if n_workers == 1 or len(jobs) == 1:
        results = []
        for job in jobs:
            _start_line(job)
            result = worker(job)
            results.append(result)
            _handle(result)
        return results

    results: list[AlignWorkerResult | None] = [None] * len(jobs)
    # spawn avoids fork+OpenMP/BLAS deadlocks after heavy scientific imports
    from st123.alignment.alignment_parallel import worker_initializer

    repo = str(jobs[0].get('repo') or '')
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(
        max_workers=min(n_workers, len(jobs)),
        mp_context=ctx,
        initializer=worker_initializer,
        initargs=(repo,),
    ) as pool:
        future_map = {}
        for i, job in enumerate(jobs):
            _start_line(job)
            future_map[pool.submit(worker, job)] = i
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                job = jobs[idx]
                results[idx] = AlignWorkerResult(
                    miri_path=job['miri_path'],
                    filter=job['filter'],
                    mode=job.get('mode', label),
                    ok=False,
                    row=asdict(
                        AlignmentSummaryRow(
                            miri_path=job['miri_path'],
                            filter=job['filter'],
                            status='FAILURE',
                            n_calibrators='NA',
                            dispersion_mas='NA',
                            aligned_path='NA',
                            ref_overlap_frac=job.get('ref_overlap_frac', 'NA'),
                        )
                    ),
                    error=str(exc),
                    message=f'Worker crashed: {exc}',
                )
            _handle(results[idx])
    return [r for r in results if r is not None]


def align_from_frames(
    frames: list[FrameOverlaps] | list[dict],
    *,
    run_alignment,
    nbright: int,
    plot: bool,
    verbose: bool,
    continue_on_error: bool,
    cache_dir: Path | None = None,
    match_radius_arcsec: float = 0.1,
    clip_to_align_footprint: bool = True,
    refine: bool = True,
    refine_sigma: float = 2.0,
    refine_max_iter: int = 5,
    use_filter_calibrators: bool = True,
    fallback: bool = True,
    max_nircam_dispersion_mas: float | None = None,
    min_ref_overlap_frac: float = 0.02,
    summary_outfile: Path | None = None,
    workers: int = 1,
    repo: Path | None = None,
) -> tuple[int, list[AlignmentSummaryRow]]:
    """
    Align MIRI frames filter-by-filter (blue→red), parallel within each filter.

    For each filter wave:
      1. Align all frames to overlapping reference images in parallel
         (``align_mode=REFERENCE`` on success)
      2. Run MIRI→MIRI fallback in parallel for hard failures and for REFERENCE
         solutions whose dispersion exceeds the per-filter (or uniform CLI)
         quality-hold threshold. MIRI_REL is kept only when its absolute
         dispersion improves on REFERENCE. Parents prefer high-quality
         overlapping successes (usually bluer), not merely closest wavelength.
      3. Optionally repeat fallback once so same-filter MIRI_REL successes can
         parent remaining hard failures

    Summary ``status`` is binary ``SUCCESS``/``FAILURE``; method is in
    ``align_mode``. ``run_alignment`` is accepted for API compatibility;
    workers import it themselves. If ``summary_outfile`` is set, the summary
    is rewritten after each finished frame.
    """
    del run_alignment  # workers import alignment_dispersion.run_alignment

    from st123.alignment.calibrators import FILTER_MAX_REFERENCE_DISPERSION_MAS
    from st123.alignment.alignment_fallback import (
        SuccessfulAlignment,
        filter_wavelength_um,
        rank_fallback_parents,
    )
    from st123.alignment.alignment_parallel import run_fallback_align_job, run_nircam_align_job

    repo_str = str((repo or _resolve_repo_root(None)).resolve())
    workers = max(1, int(workers))
    # CLI: None → per-filter map; <=0 → disable; >0 → uniform override.
    if max_nircam_dispersion_mas is not None and max_nircam_dispersion_mas <= 0:
        max_nircam_dispersion_mas = 0.0  # sentinel: disabled for all filters

    # Drop low-overlap frames before any alignment work. These remain in
    # overlap_summary* only and are omitted from alignment_summary.txt.
    frames, n_rejected = reject_zero_nircam_overlap_frames(
        frames, min_ref_overlap_frac=min_ref_overlap_frac
    )
    groups = _group_frames_by_filter(frames)

    failures = 0
    n_ok = 0
    n_fallback = 0
    rows: list[AlignmentSummaryRow] = []
    row_by_miri: dict[str, AlignmentSummaryRow] = {}
    successes: list[SuccessfulAlignment] = []

    def flush_summary() -> None:
        if summary_outfile is None:
            return
        # Omit PENDING quality-holds until MIRI_REL finishes or is exhausted.
        public = [r for r in rows if r.status != 'PENDING']
        write_alignment_summary(public, summary_outfile)

    def _finalize_quality_hold_keep_reference(
        prev: AlignmentSummaryRow, *, reason: str
    ) -> None:
        """
        Keep the REFERENCE solution after MIRI_REL does not improve it.

        The per-filter dispersion cut is a *try MIRI_REL* trigger, not a hard
        reject: a usable REFERENCE WCS remains SUCCESS when fallback cannot
        beat it.
        """
        nonlocal n_ok
        final = AlignmentSummaryRow(
            miri_path=prev.miri_path,
            filter=prev.filter,
            status='SUCCESS',
            n_calibrators=prev.n_calibrators,
            dispersion_mas=prev.dispersion_mas,
            aligned_path=prev.aligned_path,
            align_mode=_normalize_align_mode(prev.align_mode),
            original_ref=prev.original_ref,
            aligned_to=prev.aligned_to,
            ref_overlap_frac=prev.ref_overlap_frac,
        )
        idx = rows.index(prev)
        rows[idx] = final
        row_by_miri[prev.miri_path] = final
        if isinstance(final.dispersion_mas, float) and final.aligned_path not in (
            None,
            'NA',
        ):
            successes.append(
                SuccessfulAlignment(
                    miri_path=final.miri_path,
                    jhat_path=str(final.aligned_path),
                    filter=final.filter,
                    wavelength_um=filter_wavelength_um(final.filter),
                    dispersion_mas=float(final.dispersion_mas),
                    relative_dispersion_mas=float(final.dispersion_mas),
                    align_mode='REFERENCE',
                    original_ref=str(final.original_ref),
                    aligned_to=str(final.aligned_to),
                    photfile=None,
                )
            )
            n_ok += 1
        print(
            f'DONE  {Path(prev.miri_path).name}  {prev.filter}  SUCCESS  '
            f'align_mode=REFERENCE  dispersion_mas={final.dispersion_mas:.3f} '
            f'({reason})',
            flush=True,
        )
        flush_summary()

    def record_result(result, *, count_fallback: bool = False) -> None:
        nonlocal n_ok, n_fallback, failures
        row = AlignmentSummaryRow(**result.row)
        prev = row_by_miri.get(result.miri_path)

        # MIRI_REL did not improve a REFERENCE quality hold: keep REFERENCE.
        if (
            not result.ok
            and result.mode == 'fallback'
            and prev is not None
            and _is_reference_quality_hold(prev)
        ):
            why = 'MIRI_REL did not improve REFERENCE'
            if result.error:
                why = f'{why}: {result.error}'
            _finalize_quality_hold_keep_reference(prev, reason=why)
            if verbose and result.error:
                print(f'  detail: {result.error}', file=sys.stderr, flush=True)
            return

        if prev is None:
            rows.append(row)
        else:
            idx = rows.index(prev)
            rows[idx] = row
        row_by_miri[result.miri_path] = row

        if result.ok and result.success is not None:
            successes.append(SuccessfulAlignment(**result.success))
            if prev is None or prev.status != 'SUCCESS':
                n_ok += 1
                if count_fallback or result.mode == 'fallback':
                    n_fallback += 1

        print(_format_worker_done(result), flush=True)
        if verbose and result.error:
            print(f'  detail: {result.error}', file=sys.stderr, flush=True)
        flush_summary()

    if summary_outfile is not None:
        summary_outfile = Path(summary_outfile).expanduser().resolve()
        write_alignment_summary(rows, summary_outfile)
        print(f'Live alignment summary → {summary_outfile}')

    print(
        f'Alignment plan: {len(groups)} filter wave(s), '
        f'{sum(len(v) for v in groups.values())} frame(s) after rejecting '
        f'{n_rejected} low-overlap '
        f'(ref_overlap_frac < {min_ref_overlap_frac:.4f}), workers={workers}'
    )
    for filt, group in groups.items():
        print(f'  {filt}: {len(group)} frame(s)')

    if not groups:
        print('No MIRI frames with NIRCam overlap remain to align.')
        return 0, rows

    common_job = dict(
        repo=repo_str,
        nbright=nbright,
        plot=plot,
        verbose=verbose,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        match_radius_arcsec=match_radius_arcsec,
        clip_to_align_footprint=clip_to_align_footprint,
        refine=refine,
        refine_sigma=refine_sigma,
        refine_max_iter=refine_max_iter,
        use_filter_calibrators=use_filter_calibrators,
        max_nircam_dispersion_mas=max_nircam_dispersion_mas,
    )

    if use_filter_calibrators:
        from st123.alignment.calibrators import (
            F770W_CALIBRATOR_SETTINGS,
            describe_calibrator_settings,
        )

        print(
            'Filter calibrators: F770W → '
            f'{describe_calibrator_settings(F770W_CALIBRATOR_SETTINGS)}'
        )
    else:
        print('Filter calibrators: disabled (--no-filter-calibrators)')

    if max_nircam_dispersion_mas == 0.0:
        print('REFERENCE quality hold: disabled')
    elif max_nircam_dispersion_mas is not None:
        print(
            f'REFERENCE quality hold: uniform > '
            f'{max_nircam_dispersion_mas:.1f} mas → try MIRI_REL '
            f'(keep only if improved)'
        )
    else:
        parts = []
        for name, thr in FILTER_MAX_REFERENCE_DISPERSION_MAS.items():
            parts.append(f'{name}:{"off" if thr is None else f"{thr:.0f}"}')
        print(
            'REFERENCE quality hold (per filter, mas → try MIRI_REL; '
            f'keep only if improved): {", ".join(parts)}'
        )

    for filt, group in groups.items():
        print()
        print('=' * 72)
        print(f'Filter wave {filt}: {len(group)} frame(s)')
        print('=' * 72)

        pending: dict[str, dict] = {}
        for frame in group:
            miri_path, ref_images, best_ref = _frame_ref_images(frame)
            # Safety: low-overlap frames are rejected above and must not reach
            # alignment_dispersion / JHAT.
            if not ref_images or not best_ref:
                continue
            outdir = alignment_outdir_for(miri_path)
            pending[miri_path] = {
                **common_job,
                'miri_path': miri_path,
                'filter': filt,
                'ref_images': ref_images,
                'best_ref': best_ref,
                'outdir': str(outdir),
                'ref_overlap_frac': _frame_ref_overlap_frac(frame),
            }

        # --- Pass 1: parallel REFERENCE alignment ---
        nircam_jobs = [
            {**job, 'mode': 'nircam'}
            for job in pending.values()
        ]

        _run_jobs_parallel(
            nircam_jobs,
            run_nircam_align_job,
            workers=workers,
            label=f'{filt} REFERENCE',
            on_result=record_result,
        )

        # --- Passes 2+: parallel MIRI fallback ---
        if fallback:
            for pass_idx in (1, 2):
                need_fallback = [
                    miri
                    for miri, row in row_by_miri.items()
                    if miri in pending and _needs_miri_fallback(row)
                ]
                for miri in pending:
                    if miri not in row_by_miri:
                        need_fallback.append(miri)
                seen: set[str] = set()
                ordered_need: list[str] = []
                for miri in need_fallback:
                    if miri not in seen:
                        seen.add(miri)
                        ordered_need.append(miri)

                fb_jobs = []
                for miri in ordered_need:
                    ranked = rank_fallback_parents(
                        miri, filt, successes, max_parents=5
                    )
                    if not ranked:
                        continue
                    prev_row = row_by_miri.get(miri)
                    ref_disp = (
                        float(prev_row.dispersion_mas)
                        if prev_row is not None
                        and _is_reference_quality_hold(prev_row)
                        and isinstance(prev_row.dispersion_mas, float)
                        else None
                    )
                    fb_jobs.append(
                        {
                            **pending[miri],
                            'mode': 'fallback',
                            'parent': asdict(ranked[0][0]),
                            'parents': [asdict(p) for p, _ov in ranked],
                            'overlap_fraction': ranked[0][1],
                            'reference_dispersion_mas': ref_disp,
                        }
                    )

                if not fb_jobs:
                    break

                any_new = False

                def _on_fallback(result, _pass=pass_idx) -> None:
                    nonlocal any_new
                    before = row_by_miri.get(result.miri_path)
                    record_result(result, count_fallback=True)
                    if result.ok and (
                        before is None or before.status != 'SUCCESS'
                    ):
                        any_new = True

                _run_jobs_parallel(
                    fb_jobs,
                    run_fallback_align_job,
                    workers=workers,
                    label=f'{filt} fallback pass {pass_idx}',
                    on_result=_on_fallback,
                )
                if not any_new:
                    break

        # Finalize remaining REFERENCE quality-holds when no MIRI_REL parent
        # was available (or fallback was disabled): keep the REFERENCE WCS.
        for miri in list(pending):
            row = row_by_miri.get(miri)
            if row is None or not _is_reference_quality_hold(row):
                continue
            _finalize_quality_hold_keep_reference(
                row,
                reason='no MIRI_REL parent; keeping REFERENCE solution',
            )

        # Final failure tally for this filter wave.
        wave_failures = [
            miri
            for miri in pending
            if row_by_miri.get(miri) is not None
            and row_by_miri[miri].status not in ('SUCCESS', 'REJECTED')
        ]
        # Ensure every pending frame has a row.
        for miri, job in pending.items():
            if miri in row_by_miri:
                continue
            row = harvest_alignment_metrics(
                miri,
                Path(job['outdir']),
                ran_ok=False,
                ref_overlap_frac=job.get('ref_overlap_frac', 'NA'),
            )
            row.filter = filt
            rows.append(row)
            row_by_miri[miri] = row
            wave_failures.append(miri)
            flush_summary()

        failures += len(wave_failures)
        if wave_failures and not continue_on_error:
            raise RuntimeError(
                f'Alignment failed for {len(wave_failures)} {filt} frame(s); '
                f'first={wave_failures[0]}'
            )

        print(
            f'Filter wave {filt} done: '
            f'{sum(1 for m in pending if row_by_miri[m].status == "SUCCESS")} ok, '
            f'{len(wave_failures)} failed'
        )

    print()
    print(
        f'Alignment finished: {n_ok} ok ({n_fallback} via MIRI fallback), '
        f'{n_rejected} rejected (ref_overlap_frac < {min_ref_overlap_frac:.4f}), '
        f'{failures} failed'
    )
    return failures, rows
