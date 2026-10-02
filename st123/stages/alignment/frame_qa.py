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
# Hard fail: residual this large is a wrong solution, not a 50-80 mas miss.
# Frames at or above this must not enter a coadd (SN 2014cx F606W was 245-316 mas).
FRAME_CATASTROPHIC_TOL_ARCSEC = 0.200
# Wide recovery search / apply cap when a frame is already catastrophic.
# Shared-stamp galaxy peaks can sit tens of arcsec apart (SN 2014cx F606W is
# 25" west and 6" north of F555W after a false ~69" FFT lock; pipeline WCS
# placed the nucleus ~65" off). Catalog 2-D histograms (5") and a 15"
# cutout cannot see that. 90" covers an ACS/WFC chip-scale translation.
FRAME_CATASTROPHIC_RECOVER_MAX_ARCSEC = 90.0
# L3 coadds of the same box: smoothed galaxy-peak offset above this is a
# wrong WCS, not a 50 mas catalog residual (F555W vs F814W nucleus color
# is ~0.16"; F606W vs F555W was 25").
FRAME_L3_IMAGE_OFFSET_TOL_ARCSEC = 1.0
# L3 coadds of the same box that share <= this many sources are treated as
# a failed WCS, not "unmeasurable".
FRAME_L3_NO_MATCH_MAX = 1
FRAME_MIN_MATCH_HEALTHY = 12
# Peer-sparse: n_cal this far below healthy peers plus a residual above the
# sparse gate is a false JHAT lock, even when the residual is < 200 mas.
FRAME_PEER_CAL_FRACTION = 0.25

# Abs retie polish (mirrors JWST hub retie). HST visit retie vs a coadd
# 2-D histogram uses ``HST_ABS_OFFSET_MAX_ARCSEC`` instead -- WFPC2 pipeline
# WCS is often ~1" off, which this 0.40" cap would discard.
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


def is_catastrophic_alignment(
    residual_arcsec: float | None,
    n_calibrators: int | None = None,
    *,
    peer_n_calibrators: int | None = None,
    gaia_residual_arcsec: float | None = None,
    n_gaia: int | None = None,
) -> bool:
    """
    True when the solution is a wrong lock that must not be mosaicked.

    Catastrophic if the residual exceeds
    :data:`FRAME_CATASTROPHIC_TOL_ARCSEC` (200 mas), if the match count
    is a small fraction of healthy peers **and** the residual still exceeds
    the sparse 80 mas gate, or if a healthy Gaia 2-D histogram residual
    exceeds 200 mas (pipeline WCS ~arcseconds off).
    """
    res = None if residual_arcsec is None else float(residual_arcsec)
    if res is not None and res > float(FRAME_CATASTROPHIC_TOL_ARCSEC):
        return True
    gaia_res = (
        None if gaia_residual_arcsec is None else float(gaia_residual_arcsec)
    )
    n_g = int(n_gaia or 0)
    if (
        gaia_res is not None
        and gaia_res > float(FRAME_CATASTROPHIC_TOL_ARCSEC)
        and n_g >= int(FRAME_MIN_MATCH_HEALTHY)
    ):
        return True
    peer = int(peer_n_calibrators or 0)
    n_cal = int(n_calibrators or 0)
    if (
        res is not None
        and peer >= 4 * int(FRAME_MIN_MATCH_HEALTHY)
        and n_cal > 0
        and n_cal < float(FRAME_PEER_CAL_FRACTION) * peer
        and res > float(FRAME_SPARSE_TOL_ARCSEC)
    ):
        return True
    return False


def median_healthy_n_calibrators(n_cals: Iterable[int | None]) -> int | None:
    """Median n_cal among healthy (>= 12) solutions, else None."""
    healthy = [
        int(n)
        for n in n_cals
        if n is not None and int(n) >= int(FRAME_MIN_MATCH_HEALTHY)
    ]
    if not healthy:
        return None
    healthy.sort()
    return int(healthy[len(healthy) // 2])


def read_jhat_quality_headers(path: PathLike) -> dict[str, Any]:
    """Read ``JWDISPD`` / ``JWDISPM`` / ``JWNCAL`` / ``ST123CAT`` from a JHAT."""
    report: dict[str, Any] = {
        'path': str(path),
        'residual_arcsec': None,
        'n_calibrators': None,
        'catastrophic': None,
    }
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return report
    try:
        with as_datamodel(p).open(memmap=True) as hdul:
            hdr = hdul[0].header
            med = hdr.get('JWDISPD')
            mean = hdr.get('JWDISPM')
            val = med if med is not None else mean
            if val is not None:
                report['residual_arcsec'] = float(val)
            ncal = hdr.get('JWNCAL')
            if ncal is not None:
                report['n_calibrators'] = int(ncal)
            if 'ST123CAT' in hdr:
                report['catastrophic'] = bool(hdr['ST123CAT'])
    except Exception as exc:
        logger.warning('Could not read quality headers %s: %s', p.name, exc)
    return report


def filter_catastrophic_jhat_frames(
    paths: Sequence[PathLike],
    *,
    peer_n_calibrators: int | None = None,
) -> tuple[list[Path], list[tuple[Path, dict[str, Any]]]]:
    """
    Split JHAT paths into mosaic-safe vs catastrophically misaligned.

    *peer_n_calibrators* defaults to the median healthy ``JWNCAL`` among
    *paths*. Frames without quality headers are kept (L3 QA is the backup).
    """
    resolved = [Path(p).expanduser().resolve() for p in paths]
    qualities = [read_jhat_quality_headers(p) for p in resolved]
    peer = peer_n_calibrators
    if peer is None:
        peer = median_healthy_n_calibrators(
            q.get('n_calibrators') for q in qualities
        )
    kept: list[Path] = []
    dropped: list[tuple[Path, dict[str, Any]]] = []
    for path, qual in zip(resolved, qualities):
        flagged = qual.get('catastrophic')
        if flagged is False:
            kept.append(path)
            continue
        if flagged is not True:
            flagged = is_catastrophic_alignment(
                qual.get('residual_arcsec'),
                qual.get('n_calibrators'),
                peer_n_calibrators=peer,
            )
        if flagged:
            qual['catastrophic'] = True
            dropped.append((path, qual))
        else:
            kept.append(path)
    return kept, dropped


def isolated_unmatched_coadds(
    report: Mapping[str, Any],
    *,
    min_matches: int = FRAME_MIN_MATCH_HEALTHY,
    abs_ref_name: str | None = None,
) -> list[str]:
    """
    Return coadd basenames that share no sources with any healthy peer.

    A pair with ``n_match <= FRAME_L3_NO_MATCH_MAX`` is a failed WCS, not
    an unmeasurable pair. When every coadd is unmatched, keep *abs_ref_name*
    if given so the hub is not discarded.
    """
    pairs = list(report.get('pairs') or [])
    involved: set[str] = set()
    matched: set[str] = set()
    unmatched: set[str] = set()
    for pair in pairs:
        a = str(pair.get('a') or '')
        b = str(pair.get('b') or '')
        if a:
            involved.add(a)
        if b:
            involved.add(b)
        n_match = int(pair.get('n_match') or 0)
        if pair.get('ok') and n_match >= int(min_matches):
            if a:
                matched.add(a)
            if b:
                matched.add(b)
        elif n_match <= int(FRAME_L3_NO_MATCH_MAX):
            if a:
                unmatched.add(a)
            if b:
                unmatched.add(b)
    if not involved:
        return []
    isolated = sorted(involved - matched)
    if not isolated:
        return []
    if matched:
        return isolated
    # All pairs unmatched: drop everyone except the abs hub.
    if abs_ref_name and abs_ref_name in involved:
        return sorted(n for n in involved if n != abs_ref_name)
    return sorted(unmatched)


def frame_qa_is_catastrophic(report: Mapping[str, Any] | None) -> bool:
    """True when a ``frame_qa`` payload must not proceed to mosaic."""
    if not report:
        return False
    abs_sec = report.get('abs') or {}
    residual_as = None
    if abs_sec.get('residual_mas') is not None:
        try:
            residual_as = float(abs_sec['residual_mas']) / 1000.0
        except (TypeError, ValueError):
            residual_as = None
    n_cal = abs_sec.get('n_calibrators')
    try:
        n_cal_i = int(n_cal) if n_cal is not None else None
    except (TypeError, ValueError):
        n_cal_i = None
    peer = median_healthy_n_calibrators(
        (
            int(fr['n_calibrators'])
            if fr.get('n_calibrators') is not None
            else None
        )
        for fr in (report.get('frames') or [])
        if isinstance(fr, Mapping)
    )
    if is_catastrophic_alignment(
        residual_as, n_cal_i, peer_n_calibrators=peer
    ):
        return True
    extra = report.get('mosaic_coadd_qa') or {}
    if isolated_unmatched_coadds(extra):
        return True
    for pair in extra.get('failed_pairs') or []:
        if int(pair.get('n_match') or 0) <= int(FRAME_L3_NO_MATCH_MAX):
            return True
    return False


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
    catastrophic: bool | None = None,
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
    catastrophic : bool, optional
        When set, writes ``ST123CAT`` so mosaic can refuse the frame.
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
        if catastrophic is not None:
            hdr['ST123CAT'] = (
                bool(catastrophic),
                'st123: catastrophic alignment; do not mosaic',
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
            'catastrophic_arcsec': FRAME_CATASTROPHIC_TOL_ARCSEC,
            'min_match_healthy': FRAME_MIN_MATCH_HEALTHY,
        },
    }
    if extra:
        report.update(dict(extra))
    report['ok'] = bool(report['abs']['ok']) and bool(report['internal']['ok'])
    peer = median_healthy_n_calibrators(
        (
            int(fr['n_calibrators'])
            if isinstance(fr, Mapping) and fr.get('n_calibrators') is not None
            else None
        )
        for fr in (frames or [])
    )
    cat_ok = not is_catastrophic_alignment(
        residual_as, n_cal, peer_n_calibrators=peer
    )
    extra_qa = extra or {}
    mosaic_qa = extra_qa.get('mosaic_coadd_qa') if isinstance(extra_qa, Mapping) else None
    if mosaic_qa and isolated_unmatched_coadds(mosaic_qa):
        cat_ok = False
    report['catastrophic'] = {
        'ok': cat_ok,
        'tol_arcsec': FRAME_CATASTROPHIC_TOL_ARCSEC,
    }
    report['mosaic_ok'] = bool(report['ok']) and bool(cat_ok)
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
    Log when abs/internal exceed the 50 mas gate.

    Catastrophic residuals log an error and must not mosaic. Smaller misses
    remain a warning (mosaic assume-aligned path).

    Returns True when the report is OK (or missing); False when failing.
    """
    lg = log or logger
    if not report:
        return True
    abs_ok_flag = bool((report.get('abs') or {}).get('ok', True))
    int_ok_flag = bool((report.get('internal') or {}).get('ok', True))
    if (
        abs_ok_flag
        and int_ok_flag
        and report.get('ok', True)
        and not frame_qa_is_catastrophic(report)
    ):
        return True
    abs_mas = (report.get('abs') or {}).get('residual_mas')
    int_mas = (report.get('internal') or {}).get('max_delta_mas')
    ctx = f' [{context}]' if context else ''
    if frame_qa_is_catastrophic(report):
        lg.error(
            'Alignment QA catastrophic-fail%s: abs=%.1f mas (ok=%s) '
            'internal=%.1f mas (ok=%s); do not mosaic these frames -- '
            'see frame_qa.json',
            ctx,
            float(abs_mas) if abs_mas is not None else float('nan'),
            abs_ok_flag,
            float(int_mas) if int_mas is not None else float('nan'),
            int_ok_flag,
        )
        return False
    lg.warning(
        'Alignment QA soft-fail%s: abs=%.1f mas (ok=%s) internal=%.1f mas '
        '(ok=%s); coadds proceed (align owns frame) -- see frame_qa.json',
        ctx,
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
