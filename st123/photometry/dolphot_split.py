"""
Split oversized DOLPHOT image lists and merge the resulting catalogs.

The installed DOLPHOT binary is compiled with ``MAXNIMG=501`` (``Nimg`` up to
500). Staging still caps each run at :data:`DOLPHOT_MAX_NIMG` (400) so we stay
comfortably below that hard limit. Oversized lists are partitioned into roughly
equal chunks that share the same reference (and optional ``xytfile``); after
all parts finish, :func:`merge_dolphot_phot_catalogs` rebuilds a single
``.phot`` / ``.phot.columns`` pair in the usual DOLPHOT layout.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

from astropy.io import fits

PathLike = Union[str, os.PathLike]

logger = logging.getLogger(__name__)

# Compiled hard limit (MAXNIMG-1 with MAXNIMG=501).
DOLPHOT_COMPILE_MAX_NIMG = 500
# Soft staging limit — appreciably below the compile ceiling.
DOLPHOT_MAX_NIMG = 400

SPLIT_MANIFEST_NAME = 'dolphot_split.json'
OBJECT_NCOLS = 12
PHOT_BLOCK_NCOLS = 13

_COMBINED_FILT_RE = re.compile(
    r',?\s*((?:NIRCAM|MIRI|ACS|WFC3|WFPC2|ROMAN|EUCLID)_[A-Z0-9]+)\s*$'
)
_PERIMG_FILT_RE = re.compile(
    r'\((?:NIRCAM|MIRI|ACS|WFC3|WFPC2|ROMAN|EUCLID)_([A-Z0-9]+),'
)


@dataclass(frozen=True)
class DolphotRunPart:
    """One DOLPHOT invocation within a (possibly split) staging directory."""

    param_file: str
    phot_out: str
    nimg: int
    images: list[str] = field(default_factory=list)


@dataclass
class DolphotRunPlan:
    """Staging plan for a DOLPHOT reduction (one or more parts)."""

    outdir: Path
    ref_base: str
    phot_out: str
    parts: list[DolphotRunPart]
    max_nimg: int = DOLPHOT_MAX_NIMG
    xytfile: Optional[str] = None
    split_manifest: Optional[Path] = None

    @property
    def needs_merge(self) -> bool:
        return len(self.parts) > 1

    @property
    def param_file(self) -> Path:
        """Primary parameter file (single-run ``dolphot.param`` or part 0)."""
        return self.outdir / self.parts[0].param_file

    def to_manifest_dict(self) -> dict:
        return {
            'outdir': str(self.outdir),
            'ref_base': self.ref_base,
            'phot_out': self.phot_out,
            'max_nimg': self.max_nimg,
            'compile_max_nimg': DOLPHOT_COMPILE_MAX_NIMG,
            'xytfile': self.xytfile,
            'needs_merge': self.needs_merge,
            'parts': [asdict(p) for p in self.parts],
        }


def chunk_images(
    images: Sequence[PathLike],
    *,
    max_nimg: int = DOLPHOT_MAX_NIMG,
    group_keys: Optional[Sequence[str]] = None,
) -> list[list[Path]]:
    """
    Partition science images into chunks with ``len(chunk) <= max_nimg``.

    When *group_keys* is provided (e.g. FILTER names), images that share a key
    stay in the same chunk when possible (first-fit decreasing by group size).
    Otherwise images are split into roughly equal contiguous blocks.
    """
    paths = [Path(p) for p in images]
    n = len(paths)
    if n == 0:
        return [[]]
    if max_nimg < 1:
        raise ValueError('max_nimg must be >= 1')
    if max_nimg > DOLPHOT_COMPILE_MAX_NIMG:
        raise ValueError(
            f'max_nimg={max_nimg} exceeds compile limit '
            f'{DOLPHOT_COMPILE_MAX_NIMG}'
        )
    if n <= max_nimg:
        return [paths]

    if group_keys is not None:
        if len(group_keys) != n:
            raise ValueError('group_keys length must match images')
        groups: dict[str, list[Path]] = {}
        order: list[str] = []
        for path, key in zip(paths, group_keys):
            k = key or path.name
            if k not in groups:
                groups[k] = []
                order.append(k)
            groups[k].append(path)

        # Expand oversized filter groups into contiguous sub-chunks first.
        units: list[list[Path]] = []
        for key in order:
            grp = groups[key]
            if len(grp) > max_nimg:
                units.extend(_equal_chunks(grp, max_nimg=max_nimg))
            else:
                units.append(grp)

        # First-fit decreasing by unit size.
        units.sort(key=lambda u: (-len(u), u[0].name))
        packed: list[list[Path]] = []
        for unit in units:
            placed = False
            for chunk in packed:
                if len(chunk) + len(unit) <= max_nimg:
                    chunk.extend(unit)
                    placed = True
                    break
            if not placed:
                packed.append(list(unit))
        return packed

    n_parts = int(math.ceil(n / float(max_nimg)))
    return _equal_chunks(paths, n_parts=n_parts)


def _equal_chunks(
    items: Sequence[Path],
    max_nimg: Optional[int] = None,
    *,
    n_parts: Optional[int] = None,
) -> list[list[Path]]:
    items = list(items)
    n = len(items)
    if n == 0:
        return []
    if n_parts is None:
        if max_nimg is None:
            raise ValueError('max_nimg or n_parts required')
        n_parts = int(math.ceil(n / float(max_nimg)))
    n_parts = max(1, int(n_parts))
    base, rem = divmod(n, n_parts)
    chunks: list[list[Path]] = []
    idx = 0
    for i in range(n_parts):
        size = base + (1 if i < rem else 0)
        chunks.append(items[idx : idx + size])
        idx += size
    return [c for c in chunks if c]


def image_filter_key(path: PathLike) -> str:
    """Return ``INSTRUMENT_FILTER`` from a FITS header, or the basename."""
    p = Path(path)
    try:
        with fits.open(p, memmap=True) as hdul:
            filt = None
            det = None
            for hdu in hdul:
                if filt is None:
                    v = hdu.header.get('FILTER')
                    if v and str(v).strip() not in ('N/A', 'CLEAR', 'NONE', ''):
                        filt = str(v).strip().upper()
                if det is None:
                    v = hdu.header.get('DETECTOR')
                    if v:
                        det = str(v).strip().upper()
            if filt:
                if det and 'MIR' in det:
                    return f'MIRI_{filt}'
                if det and 'NRC' in det:
                    return f'NIRCAM_{filt}'
                name = p.name.lower()
                if 'mirimage' in name:
                    return f'MIRI_{filt}'
                if 'nrc' in name:
                    return f'NIRCAM_{filt}'
                return filt
    except Exception:
        pass
    return p.name


def write_split_paramfiles(
    outdir: PathLike,
    *,
    refimage: PathLike,
    images: Sequence[PathLike],
    phot_out: str,
    global_params: Optional[Mapping[str, str]] = None,
    xytfile: Optional[PathLike] = None,
    image_kinds: Optional[Sequence[str]] = None,
    max_nimg: int = DOLPHOT_MAX_NIMG,
    group_keys: Optional[Sequence[str]] = None,
    read_filters: bool = True,
) -> DolphotRunPlan:
    """
    Write one or more DOLPHOT parameter files under *outdir*.

    When ``len(images) <= max_nimg``, writes ``dolphot.param`` (unchanged
    layout). Otherwise writes ``dolphot_partXX.param`` plus
    :data:`SPLIT_MANIFEST_NAME`, keeping the same reference / ``xytfile``.
    """
    from st123.photometry.dolphot import write_paramfile

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ref = Path(refimage)
    imgs = [Path(p) for p in images]
    kinds = list(image_kinds) if image_kinds is not None else None
    if kinds is not None and len(kinds) != len(imgs):
        raise ValueError('image_kinds length must match images')

    keys = list(group_keys) if group_keys is not None else None
    if keys is None and read_filters and len(imgs) > max_nimg:
        keys = [image_filter_key(p) for p in imgs]

    chunks = chunk_images(imgs, max_nimg=max_nimg, group_keys=keys)
    ref_base = ref.name.replace('.fits', '')
    xyt_name = Path(xytfile).name if xytfile is not None else None

    # Index kinds by path name for chunk subsetting.
    kind_by_name = {}
    if kinds is not None:
        kind_by_name = {p.name: k for p, k in zip(imgs, kinds)}

    parts: list[DolphotRunPart] = []
    if len(chunks) == 1:
        param_name = 'dolphot.param'
        chunk_kinds = (
            [kind_by_name[p.name] for p in chunks[0]] if kind_by_name else None
        )
        write_paramfile(
            out / param_name,
            refimage=ref,
            images=chunks[0],
            global_params=global_params,
            xytfile=xytfile,
            image_kinds=chunk_kinds,
        )
        parts.append(
            DolphotRunPart(
                param_file=param_name,
                phot_out=phot_out,
                nimg=len(chunks[0]),
                images=[p.name.replace('.fits', '') for p in chunks[0]],
            )
        )
        # Remove stale split manifest from a prior oversized staging.
        stale = out / SPLIT_MANIFEST_NAME
        if stale.is_file():
            stale.unlink()
        plan = DolphotRunPlan(
            outdir=out,
            ref_base=ref_base,
            phot_out=phot_out,
            parts=parts,
            max_nimg=max_nimg,
            xytfile=xyt_name,
            split_manifest=None,
        )
        return plan

    stem = phot_out[:-5] if phot_out.endswith('.phot') else phot_out
    for i, chunk in enumerate(chunks):
        param_name = f'dolphot_part{i:02d}.param'
        part_phot = f'{stem}_part{i:02d}.phot'
        chunk_kinds = (
            [kind_by_name[p.name] for p in chunk] if kind_by_name else None
        )
        write_paramfile(
            out / param_name,
            refimage=ref,
            images=chunk,
            global_params=global_params,
            xytfile=xytfile,
            image_kinds=chunk_kinds,
        )
        parts.append(
            DolphotRunPart(
                param_file=param_name,
                phot_out=part_phot,
                nimg=len(chunk),
                images=[p.name.replace('.fits', '') for p in chunk],
            )
        )

    # Full combined param for provenance / merge column ordering (not launched).
    write_paramfile(
        out / 'dolphot_full.param',
        refimage=ref,
        images=imgs,
        global_params=global_params,
        xytfile=xytfile,
        image_kinds=kinds,
    )

    plan = DolphotRunPlan(
        outdir=out,
        ref_base=ref_base,
        phot_out=phot_out,
        parts=parts,
        max_nimg=max_nimg,
        xytfile=xyt_name,
        split_manifest=out / SPLIT_MANIFEST_NAME,
    )
    plan.split_manifest.write_text(json.dumps(plan.to_manifest_dict(), indent=2) + '\n')
    logger.info(
        'Split %d images into %d DOLPHOT parts (max_nimg=%d) under %s',
        len(imgs),
        len(parts),
        max_nimg,
        out,
    )
    return plan


def load_split_manifest(outdir: PathLike) -> Optional[DolphotRunPlan]:
    """Load :data:`SPLIT_MANIFEST_NAME` from *outdir*, or ``None``."""
    path = Path(outdir) / SPLIT_MANIFEST_NAME
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    parts = [DolphotRunPart(**p) for p in data['parts']]
    return DolphotRunPlan(
        outdir=Path(data.get('outdir', outdir)),
        ref_base=data['ref_base'],
        phot_out=data['phot_out'],
        parts=parts,
        max_nimg=int(data.get('max_nimg', DOLPHOT_MAX_NIMG)),
        xytfile=data.get('xytfile'),
        split_manifest=path,
    )


def _parse_columns_file(columns_path: Path) -> list[str]:
    names: list[str] = []
    for line in columns_path.read_text(errors='replace').splitlines():
        m = re.match(r'\s*(\d+)\s*\.\s*(.+)$', line)
        if m:
            names.append(m.group(2).strip())
    return names


def _classify_columns(names: Sequence[str]) -> tuple[list[int], dict[str, list[int]], dict[str, list[int]]]:
    """Return object indices, combined filter→indices, per-image base→indices."""
    object_idx = list(range(min(OBJECT_NCOLS, len(names))))
    combined: dict[str, list[int]] = {}
    per_image: dict[str, list[int]] = {}

    i = OBJECT_NCOLS
    while i < len(names):
        name = names[i]
        if 'jhat' in name or re.search(r'jw\d{5,}', name):
            # Per-image block starts at Measured counts / similar.
            base_m = re.search(r',\s*([A-Za-z0-9_]+_jhat)\b', name)
            if not base_m:
                base_m = re.search(r'\b([A-Za-z0-9_]+_jhat)\b', name)
            key = base_m.group(1) if base_m else f'img_{i}'
            block = list(range(i, min(i + PHOT_BLOCK_NCOLS, len(names))))
            per_image[key] = block
            i += len(block)
            continue
        m = _COMBINED_FILT_RE.search(name)
        if m and name.startswith(('Total counts', 'Total sky', 'Normalized',
                                  'Instrumental', 'Transformed', 'Magnitude',
                                  'Chi', 'Signal-to-noise', 'Sharpness',
                                  'Roundness', 'Crowding', 'Photometry')):
            filt = m.group(1)
            block = list(range(i, min(i + PHOT_BLOCK_NCOLS, len(names))))
            combined[filt] = block
            i += len(block)
            continue
        # Unknown trailing column — attach to object section.
        object_idx.append(i)
        i += 1
    return object_idx, combined, per_image


def merge_dolphot_phot_catalogs(
    phot_files: Sequence[PathLike],
    outfile: PathLike,
    *,
    match_tol_pix: float = 0.05,
) -> Path:
    """
    Merge split DOLPHOT ``.phot`` catalogs into one catalog + ``.columns``.

    Stars are matched by reference ``(X, Y)``. Object-level columns come from
    the first catalog; combined-filter and per-image blocks are concatenated
    (filters present in multiple parts keep the block from the part with more
    finite detections for that filter).
    """
    paths = [Path(p) for p in phot_files]
    if not paths:
        raise ValueError('No phot files to merge')
    if len(paths) == 1:
        out = Path(outfile)
        if paths[0].resolve() != out.resolve():
            out.write_bytes(paths[0].read_bytes())
            cols = Path(str(paths[0]) + '.columns')
            if cols.is_file():
                Path(str(out) + '.columns').write_text(cols.read_text())
        return out

    parsed = []
    for phot in paths:
        cols_path = Path(str(phot) + '.columns')
        if not cols_path.is_file():
            raise FileNotFoundError(f'Missing columns file for {phot}')
        names = _parse_columns_file(cols_path)
        obj_idx, combined, per_image = _classify_columns(names)
        rows = [line.split() for line in phot.read_text().splitlines() if line.strip()]
        parsed.append(
            dict(
                phot=phot,
                names=names,
                obj_idx=obj_idx,
                combined=combined,
                per_image=per_image,
                rows=rows,
            )
        )

    primary = parsed[0]
    # Build XY index for primary.
    def _xy(row: list[str]) -> tuple[float, float]:
        return float(row[2]), float(row[3])

    primary_xy = [_xy(r) for r in primary['rows']]

    def _match_index(x: float, y: float, xy_list: list[tuple[float, float]]) -> int:
        best_i = -1
        best_d = match_tol_pix**2 + 1.0
        for i, (xx, yy) in enumerate(xy_list):
            d = (x - xx) ** 2 + (y - yy) ** 2
            if d <= match_tol_pix**2 and d < best_d:
                best_d = d
                best_i = i
        return best_i

    # Prefer row-aligned merge when all catalogs share length + XY.
    row_aligned = all(len(p['rows']) == len(primary['rows']) for p in parsed[1:])
    if row_aligned:
        for p in parsed[1:]:
            for a, b in zip(primary_xy, (_xy(r) for r in p['rows'])):
                if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 > match_tol_pix**2:
                    row_aligned = False
                    break
            if not row_aligned:
                break

    # Choose combined filter blocks (prefer more finite mags).
    # Values: (n_det, part_index, column_indices)
    chosen_combined: dict[str, tuple[int, int, list[int]]] = {}
    for pi, p in enumerate(parsed):
        for filt, idxs in p['combined'].items():
            mag_i = idxs[4] if len(idxs) > 4 else idxs[0]
            n_det = sum(
                1
                for row in p['rows']
                if mag_i < len(row) and _finite_mag(row[mag_i])
            )
            prev = chosen_combined.get(filt)
            if prev is None or n_det > prev[0]:
                chosen_combined[filt] = (n_det, pi, idxs)

    filt_order: list[str] = []
    for p in parsed:
        for filt in p['combined']:
            if filt not in filt_order:
                filt_order.append(filt)

    # Per-image: keep first occurrence (parts are disjoint by construction).
    per_order: list[tuple[int, str, list[int]]] = []
    seen_img: set[str] = set()
    for pi, p in enumerate(parsed):
        for base, idxs in p['per_image'].items():
            if base in seen_img:
                continue
            seen_img.add(base)
            per_order.append((pi, base, idxs))

    out_names: list[str] = [primary['names'][i] for i in primary['obj_idx']]
    for filt in filt_order:
        _n, pi, idxs = chosen_combined[filt]
        out_names.extend(parsed[pi]['names'][j] for j in idxs)
    for pi, _base, idxs in per_order:
        out_names.extend(parsed[pi]['names'][j] for j in idxs)

    out_rows: list[list[str]] = []
    secondary_xy = [[_xy(r) for r in p['rows']] for p in parsed]

    for ri, prow in enumerate(primary['rows']):
        out = [prow[i] for i in primary['obj_idx']]
        xy = primary_xy[ri]
        part_row_idx = [ri]
        for pi in range(1, len(parsed)):
            if row_aligned:
                part_row_idx.append(ri)
            else:
                part_row_idx.append(
                    _match_index(xy[0], xy[1], secondary_xy[pi])
                )

        for filt in filt_order:
            _n, pi, idxs = chosen_combined[filt]
            rj = part_row_idx[pi]
            if rj < 0:
                out.extend(['99.999'] * len(idxs))
            else:
                row = parsed[pi]['rows'][rj]
                out.extend(row[j] if j < len(row) else '99.999' for j in idxs)
        for pi, _base, idxs in per_order:
            rj = part_row_idx[pi]
            if rj < 0:
                out.extend(['99.999'] * len(idxs))
            else:
                row = parsed[pi]['rows'][rj]
                out.extend(row[j] if j < len(row) else '99.999' for j in idxs)
        out_rows.append(out)

    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as fh:
        for row in out_rows:
            fh.write(' '.join(row) + '\n')
    cols_out = Path(str(out_path) + '.columns')
    with cols_out.open('w') as fh:
        for i, name in enumerate(out_names, start=1):
            fh.write(f'{i}. {name}\n')
    logger.info(
        'Merged %d catalogs → %s (%d stars, %d columns)',
        len(paths),
        out_path,
        len(out_rows),
        len(out_names),
    )
    return out_path


def _finite_mag(token: str) -> bool:
    try:
        return float(token) < 90
    except ValueError:
        return False


def finalize_split_outdir(outdir: PathLike) -> Optional[Path]:
    """
    If *outdir* has a finished split plan, merge part catalogs into ``phot_out``.

    Returns the merged catalog path, or ``None`` if no split / not ready.
    """
    plan = load_split_manifest(outdir)
    if plan is None or not plan.needs_merge:
        return None
    out = Path(outdir)
    part_phots = []
    for part in plan.parts:
        phot = out / part.phot_out
        if not phot.is_file() or phot.stat().st_size == 0:
            logger.info('Split part not ready: %s', phot)
            return None
        part_phots.append(phot)
    merged = out / plan.phot_out
    return merge_dolphot_phot_catalogs(part_phots, merged)


def iter_launchable_parts(
    outdir: PathLike,
    *,
    phot_out: Optional[str] = None,
) -> list[tuple[str, str]]:
    """
    Return ``(param_basename, phot_basename)`` pairs still needing DOLPHOT.

    Handles both single-param runs and split manifests. Skips parts whose
    phot catalog already exists and is non-empty. When all parts exist but the
    merged catalog does not, returns an empty list (caller should merge).
    """
    out = Path(outdir)
    plan = load_split_manifest(out)
    if plan is None:
        param = out / 'dolphot.param'
        if not param.is_file():
            return []
        phot = phot_out or _guess_phot_name(out)
        if phot and (out / phot).is_file() and (out / phot).stat().st_size > 0:
            return []
        return [('dolphot.param', phot or f'{out.name}.phot')]

    pending = []
    for part in plan.parts:
        phot = out / part.phot_out
        if phot.is_file() and phot.stat().st_size > 0:
            continue
        pending.append((part.param_file, part.phot_out))
    return pending


def _guess_phot_name(outdir: Path) -> Optional[str]:
    for p in sorted(outdir.glob('*.phot')):
        if p.name.count('.') == 1 and p.stat().st_size > 0:
            return p.name
    return None
