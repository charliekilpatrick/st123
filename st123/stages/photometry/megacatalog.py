"""
Build a multi-instrument DOLPHOT \"mega catalog\" from free NIRCam + warmstarts.

Free NIRCam is the master star list. MIRI / HST warmstart photometry is attached
by matching reference ``(X, Y)``. Stars missing from a warmstart get ``99.999``
in those filter columns (same convention as split-catalog merges).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Sequence, Union

from st123.stages.photometry.dolphot_split import merge_dolphot_phot_catalogs

PathLike = Union[str, os.PathLike]

logger = logging.getLogger(__name__)

_PHOT_DIR_RE = re.compile(
    r'^(?:phot|nircam|miri|acs|wfc3|wfpc2|hst|nircam_miri|nircam_hst)_(\d+)_(.+)$',
    re.IGNORECASE,
)


def _project_and_reduction(base_dir: PathLike) -> tuple[Path, Path]:
    base = Path(base_dir).expanduser().resolve()
    if base.name == 'reduction':
        return base.parent, base
    return base, base / 'reduction'


def _parse_group_box(dirname: str) -> tuple[int, str] | None:
    match = _PHOT_DIR_RE.match(dirname)
    if match is None:
        return None
    return int(match.group(1)), str(match.group(2))


def _is_free_nircam_dir(name: str) -> bool:
    key = name.lower()
    if key.startswith('nircam_miri') or key.startswith('nircam_hst'):
        return False
    return key.startswith('phot_') or key.startswith('nircam_')


def _is_miri_warmstart_dir(name: str) -> bool:
    return name.lower().startswith('nircam_miri')


def _is_hst_warmstart_dir(name: str) -> bool:
    return name.lower().startswith('nircam_hst')


def _finished_phot(outdir: Path) -> Path | None:
    phot = outdir / f'{outdir.name}.phot'
    columns = Path(str(phot) + '.columns')
    if phot.is_file() and columns.is_file():
        return phot
    return None


def _iter_run_dirs(base_dir: PathLike) -> list[Path]:
    project, reduction = _project_and_reduction(base_dir)
    roots = [reduction, project / 'dolphot']
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(p for p in root.iterdir() if p.is_dir()):
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
    return out


def discover_megacatalog_phot_files(
    base_dir: PathLike,
    *,
    group: int | None = None,
    box: int | str | None = None,
) -> list[Path]:
    """
    Discover free-NIRCam + MIRI/HST warmstart ``.phot`` files for a mega catalog.

    Returns an ordered list ``[nircam, miri?, hst?]``. Raises ``FileNotFoundError``
    when no finished free-NIRCam catalog is found.
    """
    nircam: list[tuple[tuple[int, str] | None, Path]] = []
    miri: list[tuple[tuple[int, str] | None, Path]] = []
    hst: list[tuple[tuple[int, str] | None, Path]] = []

    for outdir in _iter_run_dirs(base_dir):
        phot = _finished_phot(outdir)
        if phot is None:
            continue
        gb = _parse_group_box(outdir.name)
        if group is not None or box is not None:
            if gb is None:
                continue
            g, b = gb
            if group is not None and g != int(group):
                continue
            if box is not None and str(b) != str(box):
                continue
        if _is_free_nircam_dir(outdir.name):
            nircam.append((gb, phot))
        elif _is_miri_warmstart_dir(outdir.name):
            miri.append((gb, phot))
        elif _is_hst_warmstart_dir(outdir.name):
            hst.append((gb, phot))

    if not nircam:
        raise FileNotFoundError(
            f'No finished free-NIRCam DOLPHOT catalog under {base_dir} '
            '(need phot_* or nircam_* with .phot/.columns; excluding '
            'nircam_miri_* / nircam_hst_*)'
        )

    def _rank(item: tuple[tuple[int, str] | None, Path]) -> tuple[int, str]:
        gb, phot = item
        name = phot.parent.name.lower()
        kind = 0 if name.startswith('phot_') else 1
        label = gb[1] if gb else name
        return (kind, label)

    nircam_sorted = sorted(nircam, key=_rank)
    primary_gb, primary = nircam_sorted[0]

    def _matching(items: list[tuple[tuple[int, str] | None, Path]]) -> list[Path]:
        if primary_gb is None:
            return [p for _, p in items]
        matched = [p for gb, p in items if gb == primary_gb]
        return matched if matched else [p for _, p in items]

    ordered = [primary]
    ordered.extend(_matching(miri))
    ordered.extend(_matching(hst))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in ordered:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def build_megacatalog(
    phot_files: Sequence[PathLike],
    outfile_h5: PathLike,
    *,
    match_tol_pix: float = 0.05,
    force: bool = False,
    compression: bool = True,
    dry_run: bool = False,
) -> Path:
    """
    Merge *phot_files* (primary first) into one mega ``.phot`` + ``.h5``.

    The HDF5 path is *outfile_h5*. Sibling ``.phot`` / ``.columns`` use the same
    stem in the same directory. Copies the primary run's ``dolphot.param`` into
    the staging directory when present for HDF5 metadata.
    """
    from st123.stages.photometry.dolphot_catalog_hdf5 import ensure_dolphot_catalog_hdf5

    paths = [Path(p).expanduser().resolve() for p in phot_files]
    if not paths:
        raise ValueError('No phot files to merge into a mega catalog')
    for phot in paths:
        cols = Path(str(phot) + '.columns')
        if not phot.is_file():
            raise FileNotFoundError(f'missing phot catalog: {phot}')
        if not cols.is_file():
            raise FileNotFoundError(f'missing columns file: {cols}')

    out_h5 = Path(outfile_h5).expanduser().resolve()
    if out_h5.suffix.lower() != '.h5':
        out_h5 = out_h5.with_suffix('.h5')
    outdir = out_h5.parent
    stem = out_h5.stem
    phot_out = f'{stem}.phot'
    phot_path = outdir / phot_out

    if out_h5.is_file() and not force and not dry_run:
        logger.info(
            'Mega catalog HDF5 already exists: %s (use --force to overwrite)',
            out_h5,
        )
        return out_h5

    if dry_run:
        logger.info(
            'Dry run: would merge %d catalog(s) -> %s (+ %s)',
            len(paths),
            phot_path,
            out_h5,
        )
        for i, p in enumerate(paths):
            role = 'primary' if i == 0 else 'secondary'
            logger.info('  [%s] %s', role, p)
        return out_h5

    outdir.mkdir(parents=True, exist_ok=True)
    merge_dolphot_phot_catalogs(
        paths,
        phot_path,
        match_tol_pix=float(match_tol_pix),
    )

    primary_param = paths[0].parent / 'dolphot.param'
    dest_param = outdir / 'dolphot.param'
    if primary_param.is_file():
        if dest_param.resolve() != primary_param.resolve():
            shutil.copy2(primary_param, dest_param)
    elif not dest_param.is_file():
        dest_param.write_text(
            '# mega catalog staging (merged from multiple DOLPHOT runs)\n'
            f'# primary: {paths[0]}\n',
            encoding='utf-8',
        )

    if out_h5.is_file() and force:
        out_h5.unlink()
    written = ensure_dolphot_catalog_hdf5(
        outdir,
        phot_out=phot_out,
        out_path=out_h5,
        force=True,
        compression=compression,
        include_raw_sidecars=True,
        include_directory_manifest=False,
    )
    if written is None:
        raise RuntimeError(f'failed to write mega catalog HDF5 under {outdir}')
    logger.info('Wrote mega catalog %s (%d input catalogs)', written, len(paths))
    return Path(written)
