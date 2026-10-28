"""
Local HST alignment clinic: diagnose, flag, and salvage failed reductions.

This is heavier than the unit suite. It reads on-disk JHAT / mosaic QA for
specific fields (GitHub issues plus go18338 ic5267 / ngc1350), classifies
failure modes, optionally recovers salvageable visits, and writes
``alignment_clinic.json``. Unsalvageable frames (bad EXPFLAG, loss of
guide lock, empty UV with no sibling WCS) are tagged rather than retried.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from st123.datamodels.hst.hst import (
    GOOD_HST_EXPFLAGS,
    is_good_hst_expflag,
    read_expflag,
)
from st123.stages.alignment.frame_qa import (
    FRAME_CATASTROPHIC_TOL_ARCSEC,
    FRAME_IMAGE_XCORR_MIN_INNER,
    FRAME_L3_IMAGE_OFFSET_TOL_ARCSEC,
    L3_PAIRWISE_SOFT_LIMIT,
    QF_CATASTROPHIC,
    QF_EXPFLAG,
    QF_JHAT_NO_SOURCES,
    QF_NO_GUIDE,
    QF_OK,
    QF_RECOVERED_GAIA,
    QF_UNSALVAGEABLE,
    QF_UV_SPARSE,
    UNSALVAGEABLE_EXPFLAGS,
    filter_catastrophic_jhat_frames,
    isolated_unmatched_coadds,
    read_jhat_quality_headers,
    stamp_quality_headers,
    trusted_abs_residual_arcsec,
)

logger = logging.getLogger(__name__)

PathLike = str | Path

# Fields from GitHub issues plus the go18338 overnight mosaic failures.
# Paths that do not exist on this host are skipped.
KNOWN_TARGETS: tuple[dict[str, Any], ...] = (
    {
        'name': 'ic5267',
        'issue': 'go18338',
        'bases': ('/data/ckilpatrick/go18338/ic5267',),
        'notes': 'WFC3 F300X/F475X L3 n_match=1 image lock rejected',
    },
    {
        'name': 'ngc1350',
        'issue': 'go18338',
        'bases': ('/data/ckilpatrick/go18338/ngc1350',),
        'notes': 'WFPC2 F606W catalog 350 mas / Gaia 17" vs 0" xcorr leftover',
    },
    {
        'name': 'sn2020mmz',
        'issue': 15,
        'bases': (
            '/data/ckilpatrick/new/sn2020MMZ',
            '/data/maridaf/new/sn2020MMZ',
        ),
        'notes': 'WFC3 F275W L3 no common sources vs ACS',
    },
    {
        'name': 'sn2004g',
        'issue': 14,
        'bases': ('/data/maridaf/new/sn2004g',),
        'notes': 'WFPC2 catastrophic 310-319 mas, Gaia 6-8"',
    },
    {
        'name': 'sn2017bzb',
        'issue': 13,
        'bases': ('/data/maridaf/new/sn2017bzb',),
        'notes': 'JHAT 0 objects pass initial cut (20/29 frames)',
    },
    {
        'name': 'sn2020zbv',
        'issue': 12,
        'bases': ('/data/maridaf/new/sn2020zbv',),
        'notes': '40/175 dropped JHAT frames',
    },
    {
        'name': 'sn2016bkv',
        'issue': 11,
        'bases': ('/data/maridaf/new/sn2016bkv',),
        'notes': 'JHAT 0 objects + visit_cross_filter harmonize failed',
    },
    {
        'name': 'sn2021abkm',
        'issue': 10,
        'bases': ('/data/maridaf/new/sn2021abkm',),
        'notes': 'one frame 167 mas catalog / 83" Gaia',
    },
    {
        'name': 'sn2021dbg',
        'issue': 9,
        'bases': ('/data/maridaf/new/sn2021dbg',),
        'notes': 'one frame 348 mas catalog / 6" Gaia',
    },
    {
        'name': 'sn2020pni',
        'issue': 7,
        'bases': ('/data/maridaf/new/sn2020pni',),
        'notes': 'ACS vs WFC3 F814W 0.4" L3 residual, harmonize failed',
    },
)

# Successful lock paths, ranked. Clinic records which one actually held.
SUCCESS_PATHS: tuple[str, ...] = (
    'jhat_stars',
    'image_xcorr',
    'galaxy_peak',
    'visit_relative',
    'peer_xcorr',
    'gaia_visit_shift',
    'pipeline_wcs',
)

_JHAT_NO_SOURCE_NEEDLE = 'Only 0 objects pass the initial cut'


def _as_path(value: PathLike | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning('Could not read %s: %s', path, exc)
        return None
    return payload


def _jhat_dirs(base: Path) -> list[Path]:
    red = base / 'reduction'
    out: list[Path] = []
    seen: set[Path] = set()
    for name in ('jhat_hst', 'jhat', 'jhat_jwst'):
        cand = red / name
        if not cand.is_dir():
            continue
        key = cand.resolve()
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _jhat_frames(base: Path) -> list[Path]:
    frames: list[Path] = []
    seen: set[Path] = set()
    for folder in _jhat_dirs(base):
        for path in sorted(
            p
            for p in folder.glob('*_jhat.fits')
            if not p.name.lower().startswith('coadd_')
            and 'l3_ref' not in p.parts
        ):
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            frames.append(path)
    return frames


def _jhat_name_from_record(rec: Mapping[str, Any]) -> str | None:
    """Map an alignment-summary row to a ``*_jhat.fits`` basename."""
    path = str(rec.get('outpath') or rec.get('path') or '')
    if not path:
        return None
    name = Path(path).name
    lower = name.lower()
    if lower.endswith('_jhat.fits'):
        return name
    for suf in ('_flc.fits', '_flt.fits', '_c0m.fits'):
        if lower.endswith(suf):
            return name[: -len(suf)] + '_jhat.fits'
    return name


def _merge_summary_row(
    existing: dict[str, Any], rec: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep nested ``quality`` when a later row is a thin frame_qa stub."""
    q = rec.get('quality')
    have = existing.get('quality')
    if q and not have:
        existing['quality'] = dict(q)
    elif q and have:
        merged = dict(have)
        for key, val in dict(q).items():
            if merged.get(key) in (None, '', []):
                merged[key] = val
        existing['quality'] = merged
    for key, val in rec.items():
        if key == 'quality':
            continue
        if existing.get(key) in (None, '', []):
            existing[key] = val
    return existing


def classify_expflag(path: PathLike) -> str | None:
    """Return a quality flag for a bad EXPFLAG, else None."""
    flag = read_expflag(path)
    if is_good_hst_expflag(path):
        return None
    raw = str(flag or '').strip().upper()
    if raw in UNSALVAGEABLE_EXPFLAGS or 'LOCK' in raw or 'GUIDE' in raw:
        return QF_NO_GUIDE if ('LOCK' in raw or 'GUIDE' in raw) else QF_EXPFLAG
    if raw not in {str(v or '').upper() for v in GOOD_HST_EXPFLAGS}:
        return QF_EXPFLAG
    return None


def classify_frame(
    quality: Mapping[str, Any] | None,
    *,
    expflag: str | None = None,
    error: str | None = None,
    filter_name: str | None = None,
) -> dict[str, Any]:
    """
    Classify one JHAT / mosaic frame into a salvage plan.

    Returns ``flag``, ``salvageable``, ``success_path``, and ``action``.
    """
    q = dict(quality or {})
    err = str(error or '')
    filt = str(filter_name or q.get('filter') or '').upper()
    uv = filt in {
        'F225W',
        'F275W',
        'F300X',
        'F336W',
        'F343N',
        'F373N',
        'F395N',
        'F300W',
        'F255W',
    }

    if expflag == QF_NO_GUIDE or (
        expflag and str(expflag).upper() in UNSALVAGEABLE_EXPFLAGS
    ):
        return {
            'flag': QF_NO_GUIDE if expflag == QF_NO_GUIDE else QF_EXPFLAG,
            'salvageable': False,
            'success_path': None,
            'action': 'skip',
            'notes': 'Loss of lock / failed guide; do not retry JHAT',
        }
    if expflag == QF_EXPFLAG:
        return {
            'flag': QF_EXPFLAG,
            'salvageable': False,
            'success_path': None,
            'action': 'skip',
            'notes': (
                'Non-NORMAL EXPFLAG (e.g. TDF-DOWN); exclude from align '
                'and all downstream stages'
            ),
        }

    if _JHAT_NO_SOURCE_NEEDLE in err:
        return {
            'flag': QF_JHAT_NO_SOURCES,
            'salvageable': True,
            'success_path': 'visit_relative',
            'action': 'visit_relative',
            'notes': 'JHAT found no stars; copy sibling / pipeline WCS',
        }

    trusted = trusted_abs_residual_arcsec(q)
    method = str(trusted.get('method') or '')
    residual = trusted.get('residual_arcsec')
    x_as = q.get('image_residual_arcsec')
    n_inner = 0
    xc = q.get('image_xcorr')
    if isinstance(xc, Mapping):
        try:
            n_inner = int(xc.get('n_inner') or 0)
        except (TypeError, ValueError):
            n_inner = 0
    if n_inner <= 0:
        try:
            n_inner = int(q.get('image_xcorr_n_inner') or 0)
        except (TypeError, ValueError):
            n_inner = 0
    xcorr_ok = (
        x_as is not None
        and float(x_as) <= float(FRAME_CATASTROPHIC_TOL_ARCSEC)
        and n_inner >= int(FRAME_IMAGE_XCORR_MIN_INNER)
    )
    peak = q.get('image_peak_arcsec')
    peak_ok = (
        peak is not None
        and float(peak) <= float(FRAME_L3_IMAGE_OFFSET_TOL_ARCSEC)
    )
    gaia = q.get('gaia_residual_arcsec')
    try:
        gaia_as = None if gaia is None else float(gaia)
    except (TypeError, ValueError):
        gaia_as = None
    internal = q.get('internal_max_delta_mas')
    try:
        internal_as = (
            None if internal is None else float(internal) / 1000.0
        )
    except (TypeError, ValueError):
        internal_as = None

    if xcorr_ok:
        if gaia_as is not None and gaia_as > 1.0:
            return {
                'flag': QF_RECOVERED_GAIA,
                'salvageable': True,
                'success_path': 'gaia_visit_shift',
                'action': 'apply_gaia',
                'notes': (
                    f'Internal xcorr leftover {float(x_as):.3f}"; '
                    f'Gaia {gaia_as:.2f}" still to apply as a visit shift'
                ),
            }
        return {
            'flag': QF_OK,
            'salvageable': True,
            'success_path': 'image_xcorr',
            'action': 'keep',
            'notes': f'Image xcorr leftover {float(x_as):.3f}"',
        }
    if peak_ok:
        return {
            'flag': QF_OK,
            'salvageable': True,
            'success_path': 'galaxy_peak',
            'action': 'keep',
            'notes': f'Galaxy-peak leftover {float(peak):.3f}"',
        }
    if uv and (
        residual is None
        or q.get('n_calibrators') in (None, 0, 1)
    ):
        return {
            'flag': QF_UV_SPARSE,
            'salvageable': True,
            'success_path': 'visit_relative',
            'action': 'keep_forced',
            'notes': 'UV/blue sparse; keep for forced photometry',
        }
    if q.get('catastrophic') or (
        residual is not None
        and float(residual) > float(FRAME_CATASTROPHIC_TOL_ARCSEC)
    ):
        if internal_as is not None and internal_as <= 0.08:
            return {
                'flag': QF_RECOVERED_GAIA,
                'salvageable': True,
                'success_path': 'gaia_visit_shift',
                'action': 'apply_gaia',
                'notes': (
                    'Internally aligned; absolute residual is a visit shift'
                ),
            }
        return {
            'flag': QF_CATASTROPHIC,
            'salvageable': True,
            'success_path': 'gaia_visit_shift',
            'action': 'recover',
            'notes': 'Wide recover vs hub / Gaia (up to 90")',
        }
    if method == 'catalog_residual' and residual is not None:
        if float(residual) <= float(FRAME_CATASTROPHIC_TOL_ARCSEC):
            return {
                'flag': QF_OK,
                'salvageable': True,
                'success_path': 'jhat_stars',
                'action': 'keep',
                'notes': f'Catalog residual {float(residual):.3f}"',
            }
    return {
        'flag': QF_OK,
        'salvageable': True,
        'success_path': method or 'pipeline_wcs',
        'action': 'keep',
        'notes': '',
    }


def _summary_quality_rows(base: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for folder in _jhat_dirs(base):
        payload = _load_json(folder / 'alignment_summary.json')
        if isinstance(payload, Mapping):
            for rec in payload.get('results') or []:
                if isinstance(rec, Mapping):
                    rows.append(dict(rec))
        payload = _load_json(folder / 'frame_qa.json')
        if isinstance(payload, Mapping):
            for rec in payload.get('frames') or []:
                if isinstance(rec, Mapping):
                    rows.append(dict(rec))
    return rows


def diagnose_base_dir(base_dir: PathLike) -> dict[str, Any]:
    """Build a clinic report for one reduction root."""
    base = Path(base_dir).expanduser().resolve()
    report: dict[str, Any] = {
        'base_dir': str(base),
        'ok': True,
        'n_frames': 0,
        'n_salvageable': 0,
        'n_unsalvageable': 0,
        'n_keep': 0,
        'success_paths': {},
        'flags': {},
        'frames': [],
        'mosaic': [],
        'exists': base.is_dir(),
    }
    if not base.is_dir():
        report['ok'] = False
        report['error'] = 'base_dir does not exist'
        return report

    frames = _jhat_frames(base)
    report['n_frames'] = len(frames)
    summary_rows = _summary_quality_rows(base)
    by_name: dict[str, dict[str, Any]] = {}
    for rec in summary_rows:
        name = _jhat_name_from_record(rec)
        if not name:
            continue
        existing = by_name.get(name)
        if existing is None:
            by_name[name] = dict(rec)
        else:
            _merge_summary_row(existing, rec)

    path_counts: dict[str, int] = {}
    flag_counts: dict[str, int] = {}
    seen_names: set[str] = set()
    for path in frames:
        hdr_q = read_jhat_quality_headers(path)
        rec = by_name.get(path.name) or {}
        quality = dict(rec.get('quality') or {})
        for key, val in hdr_q.items():
            if key not in quality or quality.get(key) in (None, ''):
                quality[key] = val
        exp_flag = classify_expflag(path)
        classed = classify_frame(
            quality,
            expflag=exp_flag,
            error=str(rec.get('error') or ''),
            filter_name=str(rec.get('filter') or quality.get('filter') or ''),
        )
        row = {
            'path': str(path),
            'expflag': read_expflag(path),
            **classed,
            'dispersion_mas': quality.get('dispersion_mas'),
            'gaia_residual_arcsec': quality.get('gaia_residual_arcsec'),
            'image_peak_arcsec': quality.get('image_peak_arcsec'),
            'image_residual_arcsec': quality.get('image_residual_arcsec'),
            'image_xcorr_n_inner': (
                (quality.get('image_xcorr') or {}).get('n_inner')
                if isinstance(quality.get('image_xcorr'), Mapping)
                else quality.get('image_xcorr_n_inner')
            ),
            'n_calibrators': quality.get('n_calibrators'),
            'status': rec.get('status'),
        }
        report['frames'].append(row)
        flag = str(classed['flag'])
        flag_counts[flag] = flag_counts.get(flag, 0) + 1
        sp = classed.get('success_path')
        if sp:
            path_counts[str(sp)] = path_counts.get(str(sp), 0) + 1
        if classed.get('salvageable'):
            report['n_salvageable'] += 1
        else:
            report['n_unsalvageable'] += 1
        if classed.get('action') in ('keep', 'keep_forced'):
            report['n_keep'] += 1
        seen_names.add(path.name)

    for name, rec in by_name.items():
        if name in seen_names:
            continue
        status = str(rec.get('status') or '').lower()
        err = str(rec.get('error') or '')
        if status not in {'fail', 'failed', 'error'} and not err:
            continue
        classed = classify_frame(
            rec.get('quality'),
            error=err,
            filter_name=str(rec.get('filter') or ''),
        )
        row = {
            'path': str(rec.get('outpath') or rec.get('path') or name),
            'missing_jhat': True,
            **classed,
            'status': rec.get('status'),
            'error': err,
        }
        report['frames'].append(row)
        report['n_frames'] += 1
        flag = str(classed['flag'])
        flag_counts[flag] = flag_counts.get(flag, 0) + 1
        sp = classed.get('success_path')
        if sp:
            path_counts[str(sp)] = path_counts.get(str(sp), 0) + 1
        if classed.get('salvageable'):
            report['n_salvageable'] += 1
        else:
            report['n_unsalvageable'] += 1

    report['success_paths'] = path_counts
    report['flags'] = flag_counts

    ref_root = base / 'reduction' / 'reference'
    if ref_root.is_dir():
        for qa_path in sorted(ref_root.glob('group_*/ref_*/frame_qa.json')):
            payload = _load_json(qa_path)
            if not isinstance(payload, Mapping):
                continue
            mosaic_qa = payload.get('mosaic_coadd_qa') or {}
            isolated = isolated_unmatched_coadds(mosaic_qa)
            report['mosaic'].append(
                {
                    'qa_path': str(qa_path),
                    'ok': bool(payload.get('ok')),
                    'mosaic_ok': payload.get('mosaic_ok'),
                    'isolated': isolated,
                    'assume_aligned': payload.get('assume_aligned'),
                }
            )
            if isolated:
                report['ok'] = False
        leftover: list[str] = []
        for rejected in sorted(ref_root.glob('group_*/ref_*/coadd_*.rejected.fits')):
            live = Path(str(rejected).replace('.rejected.fits', '.fits'))
            if not live.is_file():
                leftover.append(str(rejected))
        report['rejected_only'] = leftover
        if leftover:
            report['ok'] = False

    kept, dropped = filter_catastrophic_jhat_frames(frames)
    report['mosaic_keep'] = [str(p) for p in kept]
    report['mosaic_drop'] = [str(p) for p, _ in dropped]
    if dropped and not kept:
        report['ok'] = False
    raw = base / 'reduction' / 'raw'
    if report['n_frames'] == 0 and raw.is_dir() and any(raw.iterdir()):
        report['ok'] = False
        report['error'] = 'no JHAT products; raw frames are present'
    return report


def stamp_salvage_headers(frame_row: Mapping[str, Any]) -> None:
    """Write ``ST123QF`` / xcorr leftover onto a JHAT product in place."""
    path = Path(str(frame_row.get('path') or ''))
    if not path.is_file() or not path.name.lower().endswith('_jhat.fits'):
        return
    if frame_row.get('missing_jhat'):
        return
    try:
        if path.stat().st_size <= 0:
            logger.warning('Skip stamp on empty JHAT %s', path.name)
            return
    except OSError:
        return
    x_as = frame_row.get('image_residual_arcsec')
    try:
        x_as_f = None if x_as is None else float(x_as)
    except (TypeError, ValueError):
        x_as_f = None
    n_inner = frame_row.get('image_xcorr_n_inner')
    try:
        n_inner_i = None if n_inner is None else int(n_inner)
    except (TypeError, ValueError):
        n_inner_i = None
    catastrophic = str(frame_row.get('flag')) in {
        QF_CATASTROPHIC,
        QF_UNSALVAGEABLE,
        QF_EXPFLAG,
        QF_NO_GUIDE,
    }
    keep = str(frame_row.get('action')) in (
        'keep',
        'keep_forced',
        'apply_gaia',
        'recover',
    )
    try:
        stamp_quality_headers(
            path,
            align_mode='CLINIC',
            catastrophic=bool(catastrophic and not keep),
            image_xcorr_arcsec=x_as_f,
            image_xcorr_n_inner=n_inner_i,
            quality_flag=str(frame_row.get('flag') or ''),
        )
    except Exception as exc:
        logger.warning('Could not stamp clinic headers on %s: %s', path.name, exc)


def _resolve_raw_for_clinic(path: Path, base: Path) -> Path | None:
    """Locate calibrated raw for a JHAT product or raw path under *base*."""
    from st123.stages.alignment.hst_jhat import _raw_sibling_for_jhat

    p = path.expanduser()
    if not p.is_file():
        # Basename-only search under reduction/raw.
        raw_dir = base / 'reduction' / 'raw'
        if raw_dir.is_dir():
            for cand in raw_dir.rglob(p.name):
                if cand.is_file():
                    return cand.resolve()
        return None
    lower = p.name.lower()
    if lower.endswith(('_flc.fits', '_flt.fits', '_c0m.fits')):
        return p.resolve()
    if lower.endswith('_jhat.fits'):
        sib = _raw_sibling_for_jhat(p)
        if sib is not None:
            return sib
        stem = p.name[: -len('_jhat.fits')]
        raw_dir = base / 'reduction' / 'raw'
        if raw_dir.is_dir():
            for suf in ('_flc.fits', '_flt.fits', '_c0m.fits'):
                for cand in raw_dir.rglob(stem + suf):
                    if cand.is_file():
                        return cand.resolve()
    return None


def write_visit_relative_jhat(
    target: PathLike,
    *,
    base_dir: PathLike,
    donor: PathLike | None = None,
) -> dict[str, Any]:
    """
    Write a JHAT product with pipeline geometry + donor absolute shift.

    Used for ``JHAT_NO_SOURCES`` frames: copy raw -> JHAT, restore pipeline
    WCS, then apply the same sky shift the *donor* JHAT carries vs its raw.
    """
    from st123.stages.alignment.hst_jhat import (
        _jhat_minus_raw_sky_shift,
        _raw_sibling_for_jhat,
        restore_pipeline_relative_wcs_with_common_shift,
        write_hst_jhat_from_raw,
        write_hst_narrowband_provenance,
    )

    base = Path(base_dir).expanduser().resolve()
    target_p = Path(target).expanduser()
    out: dict[str, Any] = {
        'target': str(target_p),
        'ok': False,
        'jhat': None,
        'donor': str(donor) if donor else None,
        'method': None,
        'error': None,
    }
    raw = _resolve_raw_for_clinic(target_p, base)
    if raw is None:
        out['error'] = 'no raw sibling'
        return out
    jhat_dir = base / 'reduction' / 'jhat_hst'
    try:
        dest = write_hst_jhat_from_raw(raw, jhat_dir)
    except Exception as exc:
        out['error'] = f'write_jhat_failed:{exc}'
        return out
    out['jhat'] = str(dest)
    dra_deg = 0.0
    ddec_deg = 0.0
    method = 'pipeline_wcs'
    donor_p = Path(donor).expanduser() if donor else None
    if donor_p is not None and donor_p.is_file():
        donor_raw = _raw_sibling_for_jhat(donor_p)
        if donor_raw is not None:
            try:
                dra_deg, ddec_deg = _jhat_minus_raw_sky_shift(donor_p, donor_raw)
                method = 'copy_sibling_wcs'
            except Exception as exc:
                logger.warning(
                    'Could not read donor shift from %s: %s; using pipeline WCS',
                    donor_p.name,
                    exc,
                )
    restore_pipeline_relative_wcs_with_common_shift(
        [dest], dra_deg=float(dra_deg), ddec_deg=float(ddec_deg)
    )
    try:
        write_hst_narrowband_provenance(
            dest,
            align_mode='HST_REL' if method == 'copy_sibling_wcs' else 'PIPELINE',
            aligned_to=str(donor_p) if donor_p else 'NA',
        )
        stamp_quality_headers(
            dest,
            align_mode='CLINIC',
            catastrophic=False,
            quality_flag=QF_JHAT_NO_SOURCES,
        )
    except Exception as exc:
        logger.warning('Could not stamp visit_relative headers on %s: %s', dest.name, exc)
    out['ok'] = True
    out['method'] = method
    logger.info(
        'Clinic visit_relative wrote %s method=%s donor=%s',
        dest.name,
        method,
        None if donor_p is None else donor_p.name,
    )
    return out


def apply_salvage(report: Mapping[str, Any], *, mosaic_after: bool = False) -> dict[str, Any]:
    """Recover salvageable visits in *report* and optionally remosaic."""
    from st123.stages.alignment.hst_jhat import (
        _hst_visit_key,
        recover_catastrophic_hst_jhat,
    )

    base = Path(str(report.get('base_dir') or ''))
    out: dict[str, Any] = {
        'base_dir': str(base),
        'recovered': [],
        'visit_relative': [],
        'stamped': 0,
        'mosaic_rc': None,
        'pending_visit_relative': [],
        'visit_relative_implemented': True,
    }
    # Annotate visit keys for donor planning.
    frame_rows: list[dict[str, Any]] = []
    for row in report.get('frames') or []:
        entry = dict(row)
        path = Path(str(entry.get('path') or ''))
        if path.is_file():
            try:
                entry['visit'] = _hst_visit_key(path)
            except Exception:
                entry['visit'] = entry.get('visit') or ''
        frame_rows.append(entry)

    vr_plan = plan_visit_relative_salvage(frame_rows)
    for plan in vr_plan.get('plans') or []:
        written = write_visit_relative_jhat(
            plan.get('target') or '',
            base_dir=base,
            donor=plan.get('donor'),
        )
        out['visit_relative'].append(written)
        if not written.get('ok'):
            out['pending_visit_relative'].append(str(plan.get('target') or ''))

    recover_paths: list[Path] = []
    gaia_paths: list[Path] = []
    for row in frame_rows:
        stamp_salvage_headers(row)
        out['stamped'] += 1
        action = str(row.get('action') or '')
        path = Path(str(row.get('path') or ''))
        if not path.is_file():
            continue
        if action == 'recover':
            recover_paths.append(path)
        elif action == 'apply_gaia':
            gaia_paths.append(path)
    visits: dict[str, list[Path]] = {}
    for path in recover_paths + gaia_paths:
        visits.setdefault(_hst_visit_key(path), []).append(path)
    for visit, paths in visits.items():
        recov = recover_catastrophic_hst_jhat(paths)
        out['recovered'].append(
            {
                'visit': visit,
                'applied': recov.get('applied'),
                'ok': recov.get('ok'),
                'method': recov.get('method'),
                'abs_arcsec': recov.get('abs_arcsec'),
                'error': recov.get('error'),
                'n_frames': recov.get('n_frames'),
            }
        )
        logger.info(
            'Clinic recover visit=%s applied=%s method=%s |Delta|=%s',
            visit,
            recov.get('applied'),
            recov.get('method'),
            recov.get('abs_arcsec'),
        )
    if mosaic_after and base.is_dir():
        from st123.scripts import mosaic as mosaic_script

        rc = mosaic_script.main(
            [
                '--base-dir',
                str(base),
                '--telescope',
                'hst',
                '--ncores',
                '8',
            ]
        )
        out['mosaic_rc'] = int(rc)
    return out


def resolve_known_targets() -> list[dict[str, Any]]:
    """Return known targets whose first existing base path is filled in."""
    found: list[dict[str, Any]] = []
    for spec in KNOWN_TARGETS:
        live = None
        for cand in spec.get('bases') or []:
            path = Path(str(cand)).expanduser()
            if path.is_dir():
                live = path.resolve()
                break
        row = dict(spec)
        row['base_dir'] = str(live) if live is not None else None
        row['available'] = live is not None
        found.append(row)
    return found


def load_targets_file(path: PathLike) -> list[Path]:
    """Read extra reduction roots, one path per line."""
    text = Path(path).expanduser().read_text(encoding='utf-8')
    out: list[Path] = []
    for line in text.splitlines():
        raw = line.split('#', 1)[0].strip()
        if not raw:
            continue
        out.append(Path(raw).expanduser().resolve())
    return out


def run_clinic(
    bases: Sequence[PathLike],
    *,
    apply: bool = False,
    mosaic_after: bool = False,
    outfile: PathLike | None = None,
) -> dict[str, Any]:
    """Diagnose each base dir; optionally salvage. Returns the combined report."""
    reports: list[dict[str, Any]] = []
    for base in bases:
        logger.info('Clinic diagnose %s', base)
        report = diagnose_base_dir(base)
        if apply:
            report['salvage'] = apply_salvage(
                report, mosaic_after=mosaic_after
            )
            report['after'] = diagnose_base_dir(base)
        reports.append(report)
        dest = Path(base) / 'logs'
        try:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / 'alignment_clinic.json').write_text(
                json.dumps(report, indent=2, default=str) + '\n',
                encoding='utf-8',
            )
        except OSError as exc:
            logger.warning('Could not write clinic JSON under %s: %s', dest, exc)
    combined = {
        'n_targets': len(reports),
        'targets': reports,
    }
    if outfile is not None:
        Path(outfile).expanduser().write_text(
            json.dumps(combined, indent=2, default=str) + '\n',
            encoding='utf-8',
        )
    return combined


# Optical filters preferred as L3 hubs when UV peaks sit tens of arcsec off.
_OPTICAL_HUB_FILTERS: tuple[str, ...] = (
    'F814W',
    'F606W',
    'F625W',
    'F555W',
    'F775W',
    'F850LP',
)
_UV_FILTER_TOKENS: frozenset[str] = frozenset(
    {
        'F225W',
        'F275W',
        'F300X',
        'F336W',
        'F343N',
        'F373N',
        'F395N',
        'F300W',
        'F255W',
        'F218W',
        'F280N',
        'F450W',
        'F439W',
    }
)
# 1-D bin width when clustering L3 peak offsets into absolute WCS families.
L3_OFFSET_CLUSTER_WIDTH_ARCSEC = 5.0
# UV-vs-optical peak above this means do not use UV as the L3 hub.
L3_UV_OPTICAL_SPLIT_ARCSEC = 10.0


def filter_token_from_name(name: str) -> str:
    """Best-effort filter token from a coadd / JHAT basename."""
    upper = str(name or '').upper()
    for tok in (
        'F850LP',
        'F814W',
        'F775W',
        'F658N',
        'F656N',
        'F625W',
        'F606W',
        'F555W',
        'F547M',
        'F475X',
        'F475W',
        'F438W',
        'F435W',
        'F450W',
        'F439W',
        'F395N',
        'F373N',
        'F350LP',
        'F343N',
        'F336W',
        'F300X',
        'F300W',
        'F280N',
        'F275W',
        'F255W',
        'F225W',
        'F218W',
        'F160W',
        'F140W',
        'F125W',
        'F110W',
        'F105W',
        'F098M',
    ):
        if tok in upper:
            return tok
    return ''


def is_uv_filter_token(token: str) -> bool:
    return str(token or '').upper() in _UV_FILTER_TOKENS


def prefer_optical_l3_hub(
    pairs: Sequence[Mapping[str, Any]],
    *,
    split_arcsec: float = L3_UV_OPTICAL_SPLIT_ARCSEC,
) -> dict[str, Any]:
    """
    Assess L3 recovery when UV peaks sit tens of arcsec from optical (abkm).

    Prefer an optical hub (F814W first) and sideline UV coadds that drive
    the split, so optical filters like F555W can retie to F814W alone.
    """
    plan: dict[str, Any] = {
        'mode': 'prefer_optical_hub',
        'hub': None,
        'sideline_uv': [],
        'retie_optical': [],
        'split_pairs': [],
        'needs_recovery': False,
    }
    names: set[str] = set()
    splits: list[dict[str, Any]] = []
    for pair in pairs:
        a = str(pair.get('a') or '')
        b = str(pair.get('b') or '')
        if a:
            names.add(a)
        if b:
            names.add(b)
        try:
            abs_as = float(pair.get('abs_arcsec'))
        except (TypeError, ValueError):
            continue
        ta = filter_token_from_name(a)
        tb = filter_token_from_name(b)
        uv_a = is_uv_filter_token(ta)
        uv_b = is_uv_filter_token(tb)
        if abs_as < float(split_arcsec):
            continue
        if uv_a == uv_b:
            continue
        splits.append(
            {
                'a': a,
                'b': b,
                'abs_arcsec': abs_as,
                'uv': a if uv_a else b,
                'optical': b if uv_a else a,
            }
        )
    plan['split_pairs'] = splits
    if not splits:
        return plan
    plan['needs_recovery'] = True
    optical_names = [
        n
        for n in sorted(names)
        if not is_uv_filter_token(filter_token_from_name(n))
    ]
    hub = None
    for pref in _OPTICAL_HUB_FILTERS:
        for name in optical_names:
            if pref in name.upper():
                hub = name
                break
        if hub is not None:
            break
    if hub is None and optical_names:
        hub = optical_names[0]
    plan['hub'] = hub
    plan['sideline_uv'] = sorted({str(s['uv']) for s in splits})
    plan['retie_optical'] = [
        n for n in optical_names if n != hub
    ]
    return plan


def plan_visit_relative_salvage(
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Plan sibling / pipeline-WCS JHAT products for ``JHAT_NO_SOURCES`` frames.

    Clinic classify already sets ``action=visit_relative``. This plan names
    donors (same-visit OK frames) for ``write_visit_relative_jhat``.
    """
    no_src: list[dict[str, Any]] = []
    donors: list[dict[str, Any]] = []
    for row in frames:
        flag = str(row.get('flag') or '')
        action = str(row.get('action') or '')
        path = str(row.get('path') or '')
        visit = str(row.get('visit') or row.get('visit_key') or '')
        entry = {
            'path': path,
            'visit': visit,
            'flag': flag,
            'action': action,
        }
        if flag == QF_JHAT_NO_SOURCES or action == 'visit_relative':
            no_src.append(entry)
        elif flag in {QF_OK, QF_RECOVERED_GAIA} and action in {
            'keep',
            'apply_gaia',
            'recover',
        }:
            donors.append(entry)
    planned: list[dict[str, Any]] = []
    for target in no_src:
        donor = None
        for cand in donors:
            if target['visit'] and cand['visit'] == target['visit']:
                donor = cand
                break
        if donor is None and donors:
            donor = donors[0]
        planned.append(
            {
                'target': target['path'],
                'visit': target['visit'],
                'donor': None if donor is None else donor['path'],
                'method': 'copy_sibling_wcs'
                if donor is not None
                else 'pipeline_wcs',
                'action': 'visit_relative',
            }
        )
    return {
        'n_targets': len(no_src),
        'n_donors': len(donors),
        'n_planned': len(planned),
        'n_with_donor': sum(1 for p in planned if p.get('donor')),
        'plans': planned,
        'apply_implemented': True,
    }


def l3_qa_pair_budget(
    n_coadds: int,
    *,
    pairwise_soft_limit: int = L3_PAIRWISE_SOFT_LIMIT,
) -> dict[str, Any]:
    """
    Assess L3 QA cost: full pairwise vs hub-vs-each (bkv / zbv hangs).

    Full pairwise is ``n*(n-1)/2`` image-to-image measures. Large multi-filter
    mosaics should use hub-vs-each (``n-1``) or a downsampled gate.
    """
    n = max(0, int(n_coadds))
    pairwise = n * (n - 1) // 2 if n >= 2 else 0
    hub = max(0, n - 1)
    mode = 'pairwise'
    if n > int(pairwise_soft_limit):
        mode = 'hub_vs_each'
    return {
        'n_coadds': n,
        'n_pairs_pairwise': pairwise,
        'n_pairs_hub_vs_each': hub,
        'pairwise_soft_limit': int(pairwise_soft_limit),
        'recommended_mode': mode,
        'needs_fast_path': mode != 'pairwise',
    }


def cluster_l3_peak_offsets(
    offsets_arcsec: Sequence[float],
    *,
    width_arcsec: float = L3_OFFSET_CLUSTER_WIDTH_ARCSEC,
) -> dict[str, Any]:
    """
    Cluster absolute L3 peak offsets into WCS families (zbv multi-cluster).

    Offsets near 0" are the locked family; additional clusters near 47",
    65", 87" mean visit Gaia recover alone did not unify filters.
    """
    vals: list[float] = []
    for raw in offsets_arcsec:
        try:
            vals.append(float(raw))
        except (TypeError, ValueError):
            continue
    vals = sorted(v for v in vals if v >= 0.0)
    clusters: list[dict[str, Any]] = []
    width = max(0.5, float(width_arcsec))
    for val in vals:
        placed = False
        for cl in clusters:
            if abs(val - float(cl['center'])) <= width:
                cl['members'].append(val)
                cl['center'] = sum(cl['members']) / len(cl['members'])
                placed = True
                break
        if not placed:
            clusters.append({'center': val, 'members': [val]})
    clusters.sort(key=lambda c: float(c['center']))
    locked = [c for c in clusters if float(c['center']) <= 1.0]
    wide = [c for c in clusters if float(c['center']) > 1.0]
    return {
        'n_offsets': len(vals),
        'n_clusters': len(clusters),
        'clusters': [
            {
                'center_arcsec': round(float(c['center']), 3),
                'n': len(c['members']),
                'min_arcsec': round(min(c['members']), 3),
                'max_arcsec': round(max(c['members']), 3),
            }
            for c in clusters
        ],
        'n_locked_clusters': len(locked),
        'n_wide_clusters': len(wide),
        'needs_cross_filter_unify': len(wide) >= 1 and len(clusters) >= 2,
    }


def assess_clinic_recovery(
    *,
    frames: Sequence[Mapping[str, Any]] | None = None,
    l3_pairs: Sequence[Mapping[str, Any]] | None = None,
    n_coadds: int | None = None,
    l3_peak_offsets_arcsec: Sequence[float] | None = None,
) -> dict[str, Any]:
    """
    Bundle recovery assessments for the remaining clinic failure modes.

    Pure / synthetic-friendly: unit tests pass clinic-like payloads without
    reading GB coadds.
    """
    frames = list(frames or [])
    pairs = list(l3_pairs or [])
    offsets = list(l3_peak_offsets_arcsec or [])
    if not offsets:
        for pair in pairs:
            try:
                offsets.append(float(pair.get('abs_arcsec')))
            except (TypeError, ValueError):
                continue
    n_coadd = n_coadds
    if n_coadd is None and pairs:
        names: set[str] = set()
        for pair in pairs:
            for key in ('a', 'b'):
                name = str(pair.get(key) or '')
                if name:
                    names.add(name)
        n_coadd = len(names)
    if n_coadd is None:
        n_coadd = 0
    optical = prefer_optical_l3_hub(pairs)
    visit_rel = plan_visit_relative_salvage(frames)
    qa_budget = l3_qa_pair_budget(int(n_coadd))
    clusters = cluster_l3_peak_offsets(offsets)
    return {
        'optical_hub': optical,
        'visit_relative': visit_rel,
        'l3_qa': qa_budget,
        'offset_clusters': clusters,
        'open_actions': [
            key
            for key, needed in (
                ('prefer_optical_hub', optical.get('needs_recovery')),
                (
                    'write_visit_relative_jhat',
                    int(visit_rel.get('n_planned') or 0) > 0,
                ),
                ('fast_l3_qa', qa_budget.get('needs_fast_path')),
                (
                    'cross_filter_crval_unify',
                    clusters.get('needs_cross_filter_unify'),
                ),
            )
            if needed
        ],
    }


__all__ = [
    'KNOWN_TARGETS',
    'L3_OFFSET_CLUSTER_WIDTH_ARCSEC',
    'L3_PAIRWISE_SOFT_LIMIT',
    'L3_UV_OPTICAL_SPLIT_ARCSEC',
    'SUCCESS_PATHS',
    'apply_salvage',
    'assess_clinic_recovery',
    'classify_expflag',
    'classify_frame',
    'cluster_l3_peak_offsets',
    'diagnose_base_dir',
    'l3_qa_pair_budget',
    'load_targets_file',
    'filter_token_from_name',
    'is_uv_filter_token',
    'plan_visit_relative_salvage',
    'prefer_optical_l3_hub',
    'resolve_known_targets',
    'run_clinic',
    'write_visit_relative_jhat',
]
