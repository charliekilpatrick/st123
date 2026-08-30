"""
Shared alignment quality datamodel for JWST and HST JHAT products.

Absolute quality uses catalog residual RMS (``JWDISPM`` / ``JWNCAL``).
Internal quality is pairwise max sky ``|Delta|`` among frames that share a
coadd. Tolerances default to the JWST hub retie class (~50 mas) when match
counts are healthy; sparse fields soft-fall back to 80 mas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from st123.datamodels import as_datamodel

logger = logging.getLogger(__name__)

# Primary gates (arcsec) -- same meaning for JWST abs retie and HST QA.
FRAME_ABS_TOL_ARCSEC = 0.050
FRAME_INTERNAL_TOL_ARCSEC = 0.050
# Soft fallback when n_match / JWNCAL is below the healthy threshold.
FRAME_SPARSE_TOL_ARCSEC = 0.080
# Legacy L3 pairwise soft ceiling (warn, do not hard-fail mosaic).
FRAME_L3_SOFT_TOL_ARCSEC = 0.120
FRAME_MIN_MATCH_HEALTHY = 12

# Abs retie polish (mirrors JWST hub retie).
FRAME_ABS_RETIE_TOL_ARCSEC = FRAME_ABS_TOL_ARCSEC
FRAME_ABS_RETIE_MAX_APPLY_ARCSEC = 0.40
FRAME_ABS_RETIE_MIN_PEAK = 8

PathLike = str | Path


def coherent_tol_arcsec(n_match: int | None) -> float:
    """
    Return the internal/abs tolerance for a given match count.

    Healthy solutions use :data:`FRAME_INTERNAL_TOL_ARCSEC` (50 mas); sparse
    solutions soft-fall back to :data:`FRAME_SPARSE_TOL_ARCSEC` (80 mas).
    """
    n = int(n_match or 0)
    if n >= int(FRAME_MIN_MATCH_HEALTHY):
        return float(FRAME_INTERNAL_TOL_ARCSEC)
    return float(FRAME_SPARSE_TOL_ARCSEC)


def abs_ok(
    residual_arcsec: float | None,
    *,
    n_calibrators: int | None = None,
) -> bool:
    """True when absolute residual is within the adaptive tolerance."""
    if residual_arcsec is None:
        return False
    return float(residual_arcsec) <= coherent_tol_arcsec(n_calibrators)


def internal_ok(
    max_delta_arcsec: float | None,
    *,
    n_match: int | None = None,
) -> bool:
    """True when internal max pairwise |Delta| is within tolerance."""
    if max_delta_arcsec is None:
        return True  # singleton / unmeasurable
    return float(max_delta_arcsec) <= coherent_tol_arcsec(n_match)


def stamp_quality_headers(
    jhat_path: PathLike,
    *,
    align_mode: str,
    original_ref: str | None = None,
    aligned_to: str | None = None,
    abs_mean_arcsec: float | None = None,
    abs_median_arcsec: float | None = None,
    abs_std_arcsec: float | None = None,
    n_calibrators: int | None = None,
    internal_max_arcsec: float | None = None,
    catalog_basename: str | None = None,
) -> None:
    """
    Write unified JWST-style quality keywords onto a JHAT primary header.

    Parameters
    ----------
    jhat_path : path-like
        JHAT product updated in place.
    align_mode : str
        ``JHAT``, ``VISIT``, ``REFERENCE``, ``HST_REL``, ``PIPELINE``,
        ``VISIT_ABS``, etc.
    original_ref, aligned_to : str, optional
        Absolute hub / this-step reference paths.
    abs_mean_arcsec, abs_median_arcsec, abs_std_arcsec : float, optional
        Absolute catalog residual in arcsec (``JWDISPM/D/S``).
    n_calibrators : int, optional
        Clipped match count (``JWNCAL``).
    internal_max_arcsec : float, optional
        Pairwise internal max |Delta| (``ST123INT``); NaN card omitted when
        None.
    catalog_basename : str, optional
        Stored in ``JWCAT`` when provided.
    """
    path = Path(jhat_path).expanduser().resolve()
    if not path.is_file():
        return
    with as_datamodel(path).open(mode='update', memmap=False) as hdul:
        hdr = hdul[0].header
        mode = str(align_mode or 'JHAT').upper()
        if mode == 'NIRCAM':
            mode = 'REFERENCE'
        hdr['ALGNMODE'] = (mode, 'VISIT/REFERENCE/JHAT/HST_REL/PIPELINE/...')
        if original_ref:
            hdr['ALGNREF'] = (str(original_ref), 'Original abs reference')
        if aligned_to:
            hdr['ALGNTO'] = (str(aligned_to), 'Aligned-to image/catalog')
        if abs_mean_arcsec is not None:
            hdr['JWDISPM'] = (
                float(abs_mean_arcsec),
                '[arcsec] absolute dispersion mean',
            )
        if abs_median_arcsec is not None:
            hdr['JWDISPD'] = (
                float(abs_median_arcsec),
                '[arcsec] absolute dispersion median',
            )
        if abs_std_arcsec is not None:
            hdr['JWDISPS'] = (
                float(abs_std_arcsec),
                '[arcsec] absolute dispersion std',
            )
        if n_calibrators is not None:
            hdr['JWNCAL'] = (
                int(n_calibrators),
                'N calibrators for JWDISPM / align solution',
            )
        if catalog_basename:
            hdr['JWCAT'] = (str(catalog_basename)[:68], 'Abs refcat basename')
        if internal_max_arcsec is not None:
            hdr['ST123INT'] = (
                float(internal_max_arcsec),
                '[arcsec] internal max pairwise |Delta|',
            )
        hdul.flush()


def build_frame_qa(
    *,
    mission: str,
    hub_id: str | None = None,
    align_mode: str | None = None,
    abs_ref: str | None = None,
    residual_mas: float | None = None,
    n_calibrators: int | None = None,
    abs_method: str | None = None,
    max_delta_mas: float | None = None,
    pairs: Sequence[Mapping[str, Any]] | None = None,
    frames: Sequence[Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a unified ``frame_qa`` payload (abs + internal sections)."""
    n_cal = int(n_calibrators) if n_calibrators is not None else None
    residual_as = (
        float(residual_mas) / 1000.0 if residual_mas is not None else None
    )
    max_delta_as = (
        float(max_delta_mas) / 1000.0 if max_delta_mas is not None else None
    )
    report: dict[str, Any] = {
        'mission': str(mission).lower(),
        'hub_id': hub_id,
        'align_mode': align_mode,
        'abs': {
            'ref': abs_ref,
            'residual_mas': (
                float(residual_mas) if residual_mas is not None else None
            ),
            'n_calibrators': n_cal,
            'method': abs_method,
            # Missing residual is unknown, not a soft-fail.
            'ok': (
                True
                if residual_as is None
                else abs_ok(residual_as, n_calibrators=n_cal)
            ),
            'tol_arcsec': coherent_tol_arcsec(n_cal),
        },
        'internal': {
            'max_delta_mas': (
                float(max_delta_mas) if max_delta_mas is not None else None
            ),
            'pairs': list(pairs or []),
            'ok': internal_ok(max_delta_as, n_match=n_cal),
            'tol_arcsec': coherent_tol_arcsec(n_cal),
        },
        'frames': list(frames or []),
        'tol': {
            'abs_arcsec': FRAME_ABS_TOL_ARCSEC,
            'internal_arcsec': FRAME_INTERNAL_TOL_ARCSEC,
            'sparse_arcsec': FRAME_SPARSE_TOL_ARCSEC,
            'min_match_healthy': FRAME_MIN_MATCH_HEALTHY,
        },
    }
    if extra:
        report.update(dict(extra))
    report['ok'] = bool(report['abs']['ok']) and bool(report['internal']['ok'])
    return report


def write_frame_qa(outdir: PathLike, report: Mapping[str, Any]) -> Path:
    """Write ``frame_qa.json`` under *outdir* (creates parent dirs)."""
    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'frame_qa.json'
    path.write_text(json.dumps(dict(report), indent=2, default=str) + '\n')
    return path


def read_frame_qa(path: PathLike) -> dict[str, Any] | None:
    """Load a ``frame_qa.json`` file, or None if missing/invalid."""
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        p = p / 'frame_qa.json'
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        logger.warning('Could not read frame_qa %s: %s', p, exc)
        return None


def write_alignment_summary_table(
    rows: Sequence[Mapping[str, Any]],
    outfile: PathLike,
) -> Path:
    """
    Write a mission-agnostic ASCII alignment summary.

    Columns: path, filter, status, n_calibrators, dispersion_mas,
    internal_max_delta_mas, align_mode, algnref, aligned_to.
    """
    out = Path(outfile).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = (
        'path',
        'filter',
        'status',
        'n_calibrators',
        'dispersion_mas',
        'internal_max_delta_mas',
        'align_mode',
        'algnref',
        'aligned_to',
    )

    def _cell(row: Mapping[str, Any], key: str) -> str:
        val = row.get(key, 'NA')
        if val is None:
            return 'NA'
        if key in {'dispersion_mas', 'internal_max_delta_mas'} and isinstance(
            val, (int, float)
        ):
            return f'{float(val):.3f}'
        return str(val)

    widths = {c: len(c) for c in cols}
    rendered = [{c: _cell(r, c) for c in cols} for r in rows]
    for row in rendered:
        for c in cols:
            widths[c] = max(widths[c], len(row[c]))

    lines = [
        '  '.join(f'{c:<{widths[c]}}' for c in cols),
        '  '.join('-' * widths[c] for c in cols),
    ]
    for row in rendered:
        lines.append('  '.join(f'{row[c]:<{widths[c]}}' for c in cols))
    tmp = out.with_suffix(out.suffix + '.tmp')
    tmp.write_text('\n'.join(lines) + '\n')
    tmp.replace(out)
    return out


def warn_if_frame_qa_soft(
    report: Mapping[str, Any] | None,
    *,
    log: logging.Logger | None = None,
    context: str = '',
) -> bool:
    """
    Log a warning when abs/internal exceed 50 mas (mosaic assume-aligned path).

    Returns True when the report is OK (or missing); False when soft-failing.
    """
    lg = log or logger
    if not report:
        return True
    abs_ok_flag = bool((report.get('abs') or {}).get('ok', True))
    int_ok_flag = bool((report.get('internal') or {}).get('ok', True))
    if abs_ok_flag and int_ok_flag and report.get('ok', True):
        return True
    abs_mas = (report.get('abs') or {}).get('residual_mas')
    int_mas = (report.get('internal') or {}).get('max_delta_mas')
    lg.warning(
        'Alignment QA soft-fail%s: abs=%.1f mas (ok=%s) internal=%.1f mas '
        '(ok=%s); coadds proceed (align owns frame) -- see frame_qa.json',
        f' [{context}]' if context else '',
        float(abs_mas) if abs_mas is not None else float('nan'),
        abs_ok_flag,
        float(int_mas) if int_mas is not None else float('nan'),
        int_ok_flag,
    )
    return False


def merge_frame_rows(
    frames: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate per-frame QA rows into box/dir-level abs/internal summaries.
    """
    rows = list(frames)
    abs_vals: list[float] = []
    int_vals: list[float] = []
    n_cals: list[int] = []
    for r in rows:
        if r.get('dispersion_mas') is not None:
            try:
                abs_vals.append(float(r['dispersion_mas']))
            except (TypeError, ValueError):
                pass
        if r.get('internal_max_delta_mas') is not None:
            try:
                int_vals.append(float(r['internal_max_delta_mas']))
            except (TypeError, ValueError):
                pass
        if r.get('n_calibrators') is not None:
            try:
                n_cals.append(int(r['n_calibrators']))
            except (TypeError, ValueError):
                pass
    residual_mas = max(abs_vals) if abs_vals else None
    max_delta_mas = max(int_vals) if int_vals else None
    n_cal = min(n_cals) if n_cals else None
    return build_frame_qa(
        mission=str((rows[0].get('mission') if rows else None) or 'unknown'),
        residual_mas=residual_mas,
        n_calibrators=n_cal,
        max_delta_mas=max_delta_mas,
        frames=rows,
        abs_method='aggregate',
    )
