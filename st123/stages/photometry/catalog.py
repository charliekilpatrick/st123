"""DOLPHOT column mapping, per-image photometry, and visit-binned light curves."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import glob
import itertools
import math
import os
import re

import numpy as np
import pandas as pd
from astropy.table import Table
from tqdm import tqdm

from st123.stages.photometry.aperture import AB_ZEROPOINT_UJY, ujy_to_abmag

_LN10 = math.log(10.0)
_MAG_INVALID = 90.0
_MAGERR_INVALID = 9.99

_INST_PREFIXES = (
    'NIRCAM_',
    'MIRI_',
    'WFPC2_',
    'WFC3_',
    'ACS_',
    'ROMAN_',
    'EUCLID_',
)

# Per-image field: "<field>, <basename> (<INSTR>_<FILT>, <exptime> sec)"
_PER_IMAGE_FIELD = re.compile(
    r'^(?P<field>.+),\s+(?P<image>\S+)\s+\((?P<instfilt>[^,)]+)'
)


def get_filters(columns: str) -> tuple[np.ndarray, list[str]]:
    """
    Extract unique NIRCam filters and raw lines from a DOLPHOT column file.

    Parameters
    ----------
    columns : str
        Path to a DOLPHOT ``*.columns`` file.

    Returns
    -------
    lines : numpy.ndarray
        Raw lines from the column file.
    filters : list of str
        Unique NIRCam filter names.
    """
    with open(columns, 'r') as f:
        lines = f.readlines()
    lines = np.array(lines)

    filters = []
    for line in lines:
        if 'Normalized count rate, NIRCAM' in line:
            filters.append(line.split('NIRCAM_')[-1].split('\n')[0])

    return lines, filters


def map_columns(
    columns: str,
) -> tuple[dict[str, int | None], list[str], dict[str, dict[str, list[str]]]]:
    """
    Map DOLPHOT column names to zero-based indices.

    Parameters
    ----------
    columns : str
        Path to a DOLPHOT ``*.columns`` file.

    Returns
    -------
    col_dict : dict
        Combined-photometry column name mapped to index (or ``None`` if absent).
    filters : list of str
        Unique NIRCam filter names in the column file.
    filter_cols : dict
        Filter name mapped to per-image ``Counts``, ``Err``, and ``Flag`` index
        lists.
    """
    #combined photometry columns
    column_strings = ['Object X position', 'Object Y position', 'Signal-to-noise',
                      'Object sharpness', 'Crowding', 'Object type']
    column_key = ['X', 'Y', 'SNR', 'Sharpness', 'Crowding', 'Type']

    #get unique filters and lines from column file
    lines, filters = get_filters(columns)

    #define keys and strings for instrumental magnitudes and uncertainties
    mag_strings = [f'Instrumental VEGAMAG magnitude, NIRCAM_{filt}' for filt in filters]
    magerr_strings = [f'Magnitude uncertainty, NIRCAM_{filt}' for filt in filters]
    mag_key, magerr_key = [i + '_mag' for i in filters], [i + '_err' for i in filters]

    keys = column_key+mag_key+magerr_key
    strings = column_strings+mag_strings+magerr_strings

    col_dict = {key: [] for key in keys}

    #get column indices for combined photometry
    for key, string in zip(keys, strings):
        col = lines[np.char.find(lines, string) > 0]
        if len(col) > 0:
            col_dict[key] = int(col[0].split('.')[0]) - 1
        else:
            col_dict[key] = None

    #get column indices for individual images
    filter_cols = {key: [] for key in filters}
    for filter_name in filters:
        phot_keys = ['Counts', 'Err', 'Flag']
        per_filter_cols = {key: [] for key in phot_keys}
        idx_cts, idx_err, idx_flag = [], [], []
        for line in lines:
            #get column indices for counts, errors and flags
            #count uncertainty is used to get the index since 'Normalized count rate' is not unique
            if (filter_name in line) & ('Normalized count rate uncertainty' in line):
                idx = int(line.split(' ')[0].split('.')[0]) - 1
                idx_cts.append(str(idx - 1))
                idx_err.append(str(idx))
                idx_flag.append(str(idx + 9))
        #first index corresponds to combined photometry
        per_filter_cols['Counts'] = idx_cts[1:]
        per_filter_cols['Err'] = idx_err[1:]
        per_filter_cols['Flag'] = idx_flag[1:]
        filter_cols[filter_name] = per_filter_cols

    return col_dict, filters, filter_cols


def _split_instrument_filter(token: str) -> tuple[str, str]:
    """Return ``(instrument, filter)`` from a DOLPHOT ``NIRCAM_F200W`` token."""
    text = str(token or '').strip().upper().replace(' ', '')
    for prefix in _INST_PREFIXES:
        if text.startswith(prefix):
            return prefix[:-1], text[len(prefix):]
    if '_' in text:
        inst, filt = text.split('_', 1)
        return inst, filt
    return '', text


@dataclass(frozen=True)
class PerImageColumn:
    """Column indices for one DOLPHOT per-image photometry block."""

    image: str
    instrument: str
    filter: str
    mag_idx: int
    magerr_idx: int
    flag_idx: int
    snr_idx: int
    counts_idx: int
    counts_err_idx: int


def parse_per_image_phot_columns(columns: str | Path) -> list[PerImageColumn]:
    """
    Map DOLPHOT ``*.columns`` per-image blocks to magnitude / flag indices.

    Combined-filter stacks (``Total counts, NIRCAM_F200W``) are skipped;
    only frame-level ``Measured counts, <image> (...)`` blocks are returned.
    """
    from st123.stages.photometry.dolphot_catalog_hdf5 import (
        parse_dolphot_columns_file,
    )

    cols = parse_dolphot_columns_file(columns)
    by_image: dict[str, dict[str, int]] = {}
    instfilt_by_image: dict[str, str] = {}
    for col in cols:
        m = _PER_IMAGE_FIELD.match(col.description)
        if not m:
            continue
        image = m.group('image')
        field = m.group('field').strip()
        instfilt_by_image[image] = m.group('instfilt').strip()
        by_image.setdefault(image, {})[field] = col.index_0based

    out: list[PerImageColumn] = []
    for image, fields in by_image.items():
        mag_idx = fields.get('Instrumental ABMAG magnitude')
        if mag_idx is None:
            mag_idx = fields.get('Instrumental VEGAMAG magnitude')
        if mag_idx is None:
            mag_idx = fields.get('Instrumental magnitude')
        magerr_idx = fields.get('Magnitude uncertainty')
        flag_idx = fields.get('Photometry quality flag')
        snr_idx = fields.get('Signal-to-noise')
        counts_idx = fields.get('Normalized count rate')
        counts_err_idx = fields.get('Normalized count rate uncertainty')
        if mag_idx is None or magerr_idx is None:
            continue
        instrument, filt = _split_instrument_filter(
            instfilt_by_image.get(image, '')
        )
        out.append(
            PerImageColumn(
                image=image,
                instrument=instrument,
                filter=filt,
                mag_idx=int(mag_idx),
                magerr_idx=int(magerr_idx),
                flag_idx=-1 if flag_idx is None else int(flag_idx),
                snr_idx=-1 if snr_idx is None else int(snr_idx),
                counts_idx=-1 if counts_idx is None else int(counts_idx),
                counts_err_idx=(
                    -1 if counts_err_idx is None else int(counts_err_idx)
                ),
            )
        )
    return out


def _phot_search_dirs(phot_file: Path, param_file: Path | None = None) -> list[Path]:
    dirs: list[Path] = [phot_file.parent]
    if param_file is not None:
        dirs.append(param_file.parent)
    seen: set[Path] = set()
    out: list[Path] = []
    for d in dirs:
        key = d.resolve() if d.exists() else d
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _resolve_param_file(phot_file: Path, param_file: Path | None = None) -> Path | None:
    if param_file is not None and Path(param_file).is_file():
        return Path(param_file)
    parent = phot_file.parent
    candidate = parent / 'dolphot.param'
    if candidate.is_file():
        return candidate
    return None


def _fits_for_image(image_base: str, search_dirs: Sequence[Path]) -> Path | None:
    name = image_base if image_base.endswith('.fits') else f'{image_base}.fits'
    chip = f'{image_base}.fits'
    for directory in search_dirs:
        for cand in (directory / name, directory / chip):
            if cand.is_file():
                return cand
        matches = sorted(directory.glob(f'{image_base}.chip*.fits'))
        if matches:
            return matches[0]
    return None


def image_mjd(
    image_base: str,
    search_dirs: Sequence[Path],
    *,
    info_mjd: float | None = None,
) -> float | None:
    """MJD for a DOLPHOT science frame (FITS header, else ``.info`` if > 0)."""
    from st123.datamodels import as_datamodel

    path = _fits_for_image(image_base, search_dirs)
    if path is not None:
        try:
            mjd = as_datamodel(path).mjd
        except Exception:
            mjd = None
        if mjd is not None:
            return float(mjd)
    if info_mjd is not None:
        try:
            val = float(info_mjd)
        except (TypeError, ValueError):
            val = float('nan')
        if np.isfinite(val) and val > 0.0:
            return val
    return None


def _info_mjd_by_image(phot_file: Path) -> dict[str, float]:
    from st123.stages.photometry.dolphot_catalog_hdf5 import parse_dolphot_info_file

    info_path = Path(str(phot_file) + '.info')
    if not info_path.is_file():
        alt = phot_file.with_suffix(phot_file.suffix + '.info')
        info_path = alt if alt.is_file() else info_path
    if not info_path.is_file():
        return {}
    parsed = parse_dolphot_info_file(info_path)
    out: dict[str, float] = {}
    for entry in parsed.get('paths_mjd') or []:
        raw = str(entry.get('path') or '')
        mjd = entry.get('mjd')
        if not raw or mjd is None:
            continue
        base = Path(raw).name
        if base.endswith('.fits'):
            base = base[:-5]
        try:
            out[base] = float(mjd)
        except (TypeError, ValueError):
            continue
    return out


def _load_phot_row(phot_file: Path, index: int) -> np.ndarray:
    n = -1
    with phot_file.open(encoding='utf-8', errors='replace') as handle:
        for line in handle:
            if not line.strip():
                continue
            n += 1
            if n == index:
                row = np.fromstring(line, sep=' ', dtype=np.float64)
                if row.size == 0:
                    raise ValueError(f'Empty photometry row {index} in {phot_file}')
                return row
    raise IndexError(f'Source index {index} not in {phot_file} ({n + 1} rows)')


def _cell(row: np.ndarray, idx: int, default: float = float('nan')) -> float:
    if idx < 0 or idx >= row.size:
        return default
    try:
        val = float(row[idx])
    except (TypeError, ValueError):
        return default
    return val


def combined_snr(flux_ujy: float, flux_err_ujy: float) -> float:
    """Flux / flux-error; NaN when the error is not positive."""
    if not np.isfinite(flux_ujy) or not np.isfinite(flux_err_ujy):
        return float('nan')
    if flux_err_ujy <= 0.0:
        return float('nan')
    return float(flux_ujy / flux_err_ujy)


def three_sigma_limit_abmag(flux_err_ujy: float, nsigma: float = 3.0) -> float:
    """AB magnitude of ``nsigma * flux_err`` (3-sigma limit)."""
    if not np.isfinite(flux_err_ujy) or flux_err_ujy <= 0.0:
        return float('nan')
    mag, _ = ujy_to_abmag(float(nsigma) * float(flux_err_ujy), None)
    return float(mag)


def _build_mjd_cache(
    blocks: Sequence[PerImageColumn],
    phot: Path,
    param_file: Path | None,
) -> dict[str, float | None]:
    search = _phot_search_dirs(phot, param_file)
    info_mjd = _info_mjd_by_image(phot)
    cache: dict[str, float | None] = {}
    for block in blocks:
        if block.image in cache:
            continue
        cache[block.image] = image_mjd(
            block.image, search, info_mjd=info_mjd.get(block.image)
        )
    return cache


def _row_to_measurements(
    row: np.ndarray,
    blocks: Sequence[PerImageColumn],
    mjd_cache: dict[str, float | None],
    source_index: int,
    max_flag: int,
) -> Table:
    records: list[dict] = []
    for block in blocks:
        mag = _cell(row, block.mag_idx)
        magerr = _cell(row, block.magerr_idx)
        flag = _cell(row, block.flag_idx, default=99.0)
        snr = _cell(row, block.snr_idx)
        flag_i = int(flag) if np.isfinite(flag) else 99
        good = (
            np.isfinite(mag)
            and np.isfinite(magerr)
            and mag < _MAG_INVALID
            and magerr > 0.0
            and magerr < _MAGERR_INVALID
            and flag_i <= int(max_flag)
        )
        records.append(
            {
                'image': block.image,
                'instrument': block.instrument,
                'filter': block.filter,
                'mjd': mjd_cache.get(block.image),
                'mag': mag,
                'magerr': magerr,
                'flag': flag_i,
                'snr': snr,
                'good': good,
                'source_index': int(source_index),
            }
        )
    return Table(rows=records)


def source_image_photometry(
    phot_file: str | Path,
    source_index: int,
    *,
    columns: str | Path | None = None,
    param_file: str | Path | None = None,
    max_flag: int = 7,
    row: np.ndarray | None = None,
    blocks: Sequence[PerImageColumn] | None = None,
    mjd_cache: dict[str, float | None] | None = None,
) -> Table:
    """
    Per-image DOLPHOT photometry for one catalog row.

    Parameters
    ----------
    phot_file : path-like
        DOLPHOT ``.phot`` catalog.
    source_index : int
        Zero-based row in *phot_file*.
    columns : path-like, optional
        ``*.columns`` file; defaults to ``<phot_file>.columns``.
    param_file : path-like, optional
        ``dolphot.param`` used to locate science FITS for MJD.
    max_flag : int, optional
        Rows with quality flag above this value are still returned but
        marked ``good=False``.
    row, blocks, mjd_cache : optional
        Preloaded catalog row / column map / MJD cache for neighbor loops.

    Returns
    -------
    astropy.table.Table
        One row per science image with ``mjd``, ``filter``, ``mag``,
        ``magerr``, ``flag``, ``snr``, and ``good``.
    """
    phot = Path(phot_file)
    col_path = Path(columns) if columns is not None else Path(str(phot) + '.columns')
    if not col_path.is_file():
        raise FileNotFoundError(f'Missing columns file {col_path}')
    if blocks is None:
        blocks = parse_per_image_phot_columns(col_path)
    if not blocks:
        raise ValueError(f'No per-image photometry columns in {col_path}')
    if row is None:
        row = _load_phot_row(phot, int(source_index))
    param = _resolve_param_file(
        phot, None if param_file is None else Path(param_file)
    )
    if mjd_cache is None:
        mjd_cache = _build_mjd_cache(blocks, phot, param)
    return _row_to_measurements(
        row, blocks, mjd_cache, int(source_index), int(max_flag)
    )


def inverse_variance_abmag(
    mag: np.ndarray | Sequence[float],
    magerr: np.ndarray | Sequence[float],
    *,
    mjd: np.ndarray | Sequence[float] | None = None,
) -> dict[str, float]:
    """
    Inverse-variance mean in flux space, returned as AB mag.

    Parameters
    ----------
    mag, magerr : array-like
        AB magnitudes and 1-sigma uncertainties.
    mjd : array-like, optional
        Observation times; the combined MJD is the same-weight mean.

    Returns
    -------
    dict
        ``mjd``, ``mag``, ``magerr``, ``flux_ujy``, ``flux_err_ujy``,
        ``n``. Empty input yields NaNs and ``n=0``.
    """
    mag_a = np.asarray(mag, dtype=float)
    err_a = np.asarray(magerr, dtype=float)
    ok = (
        np.isfinite(mag_a)
        & np.isfinite(err_a)
        & (mag_a < _MAG_INVALID)
        & (err_a > 0.0)
        & (err_a < _MAGERR_INVALID)
    )
    n_ok = int(np.count_nonzero(ok))
    empty = {
        'mjd': float('nan'),
        'mag': float('nan'),
        'magerr': float('nan'),
        'flux_ujy': float('nan'),
        'flux_err_ujy': float('nan'),
        'n': 0,
    }
    if n_ok == 0:
        return empty
    flux = np.power(10.0, -0.4 * (mag_a - AB_ZEROPOINT_UJY))
    flux_err = flux * err_a * (_LN10 / 2.5)
    weight = np.zeros(mag_a.shape, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        weight[ok] = 1.0 / np.square(flux_err[ok])
    wsum = float(np.sum(weight))
    if not np.isfinite(wsum) or wsum <= 0.0:
        return empty
    flux_c = float(np.sum(weight * flux) / wsum)
    flux_e = float(1.0 / math.sqrt(wsum))
    mag_c, mag_e = ujy_to_abmag(flux_c, flux_e)
    mjd_c = float('nan')
    if mjd is not None:
        mjd_a = np.asarray(mjd, dtype=float)
        mjd_ok = ok & np.isfinite(mjd_a)
        w_mjd = float(np.sum(weight[mjd_ok]))
        if w_mjd > 0.0:
            mjd_c = float(np.sum(weight[mjd_ok] * mjd_a[mjd_ok]) / w_mjd)
    return {
        'mjd': mjd_c,
        'mag': float(mag_c),
        'magerr': float(mag_e),
        'flux_ujy': flux_c,
        'flux_err_ujy': flux_e,
        'n': n_ok,
    }


def _group_filter_visits(
    table: Table,
    *,
    max_gap_days: float,
) -> list[np.ndarray]:
    """Row-index groups: same filter, MJD span ``<= max_gap_days``."""
    groups: list[np.ndarray] = []
    filters = np.unique(np.asarray(table['filter'], dtype=str))
    for filt in filters:
        idx = np.flatnonzero(np.asarray(table['filter'], dtype=str) == filt)
        if idx.size == 0:
            continue
        mjds = np.asarray(table['mjd'][idx], dtype=float)
        order = np.argsort(np.where(np.isfinite(mjds), mjds, np.inf))
        idx = idx[order]
        mjds = np.asarray(table['mjd'][idx], dtype=float)
        start = 0
        for i in range(idx.size):
            mjd_i = mjds[i]
            mjd_s = mjds[start]
            new_group = False
            if i == start:
                continue
            if not np.isfinite(mjd_i) or not np.isfinite(mjd_s):
                new_group = True
            elif float(mjd_i) - float(mjd_s) > float(max_gap_days):
                new_group = True
            if new_group:
                groups.append(idx[start:i])
                start = i
        groups.append(idx[start:])
    return groups


def combine_visit_photometry(
    measurements: Table,
    *,
    max_gap_days: float = 1.0,
    good_only: bool = True,
) -> Table:
    """
    Inverse-variance combine per-image rows by filter and 1-day visits.

    Frames of the same filter whose MJD span is at most *max_gap_days*
    are averaged in flux, then converted back to AB magnitude.

    Parameters
    ----------
    measurements : Table
        Output of :func:`source_image_photometry`.
    max_gap_days : float, optional
        Maximum MJD span of a combined visit (default 1).
    good_only : bool, optional
        If True (default), drop rows with ``good=False``.

    Returns
    -------
    astropy.table.Table
        Columns ``mjd``, ``filter``, ``mag``, ``magerr``, plus ``n``,
        ``instrument``, ``flux_ujy``, ``flux_err_ujy``, ``snr``, and ``images``.
    """
    empty_names = (
        'mjd', 'filter', 'mag', 'magerr', 'n', 'instrument',
        'flux_ujy', 'flux_err_ujy', 'snr', 'images', 'source_index',
    )
    if len(measurements) == 0:
        return Table(names=empty_names)
    work = measurements
    if good_only and 'good' in work.colnames:
        work = work[np.asarray(work['good'], dtype=bool)]
    if len(work) == 0:
        return Table(names=empty_names)
    rows: list[dict] = []
    for grp in _group_filter_visits(work, max_gap_days=max_gap_days):
        sub = work[grp]
        combined = inverse_variance_abmag(
            sub['mag'], sub['magerr'], mjd=sub['mjd']
        )
        inst = str(sub['instrument'][0]) if 'instrument' in sub.colnames else ''
        src = (
            int(sub['source_index'][0])
            if 'source_index' in sub.colnames
            else -1
        )
        images = ','.join(str(x) for x in sub['image'])
        snr = combined_snr(combined['flux_ujy'], combined['flux_err_ujy'])
        rows.append(
            {
                'mjd': combined['mjd'],
                'filter': str(sub['filter'][0]),
                'mag': combined['mag'],
                'magerr': combined['magerr'],
                'n': combined['n'],
                'instrument': inst,
                'flux_ujy': combined['flux_ujy'],
                'flux_err_ujy': combined['flux_err_ujy'],
                'snr': snr,
                'images': images,
                'source_index': src,
            }
        )
    out = Table(rows=rows)
    if len(out) == 0:
        return out
    order = np.lexsort(
        (np.asarray(out['filter'], dtype=str), np.asarray(out['mjd'], dtype=float))
    )
    return out[order]


def _load_catalog_xy_type(phot_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference X, Y, and object type (column 11 / 0-based 10) for every row."""
    data = np.loadtxt(phot_file, usecols=(2, 3, 10), ndmin=2, dtype=np.float64)
    return data[:, 0], data[:, 1], data[:, 2]


def _load_phot_rows(phot_file: Path, indices: Sequence[int]) -> dict[int, np.ndarray]:
    wanted = {int(i) for i in indices}
    found: dict[int, np.ndarray] = {}
    if not wanted:
        return found
    n = -1
    with phot_file.open(encoding='utf-8', errors='replace') as handle:
        for line in handle:
            if not line.strip():
                continue
            n += 1
            if n not in wanted:
                continue
            found[n] = np.fromstring(line, sep=' ', dtype=np.float64)
            if len(found) == len(wanted):
                break
    return found


def _neighbor_indices(
    x: np.ndarray,
    y: np.ndarray,
    objtype: np.ndarray,
    source_index: int,
    radius_pix: float,
    *,
    min_neighbors: int = 5,
    max_radius_pix: float = 200.0,
) -> np.ndarray:
    """Nearby star indices, growing *radius_pix* until *min_neighbors* exist."""
    dx = x - float(x[int(source_index)])
    dy = y - float(y[int(source_index)])
    dist = np.hypot(dx, dy)
    star = (objtype <= 2.0) | ~np.isfinite(objtype)
    not_self = np.arange(x.size) != int(source_index)
    radius = float(radius_pix)
    chosen = np.array([], dtype=int)
    while radius <= float(max_radius_pix) + 1e-6:
        mask = not_self & star & np.isfinite(dist) & (dist > 1.0) & (
            dist <= radius
        )
        chosen = np.flatnonzero(mask)
        if chosen.size >= int(min_neighbors) or radius >= float(max_radius_pix):
            break
        radius = min(float(max_radius_pix), radius * 1.5)
    return chosen


def _visit_flux_err(
    meas: Table,
    image_names: Sequence[str],
) -> float:
    names = {str(n) for n in image_names if str(n)}
    if not names or len(meas) == 0:
        return float('nan')
    keep = np.array([str(im) in names for im in meas['image']], dtype=bool)
    if 'good' in meas.colnames:
        keep &= np.asarray(meas['good'], dtype=bool)
    sub = meas[keep]
    if len(sub) == 0:
        return float('nan')
    combined = inverse_variance_abmag(sub['mag'], sub['magerr'])
    return float(combined['flux_err_ujy'])


def local_visit_limit_abmag(
    neighbor_rows: dict[int, np.ndarray],
    blocks: Sequence[PerImageColumn],
    mjd_cache: dict[str, float | None],
    image_names: Sequence[str],
    *,
    max_flag: int = 7,
    nsigma: float = 3.0,
) -> tuple[float, int]:
    """
    3-sigma AB limit from combined flux errors of nearby sources.

    Uses the same visit image set as the target. Sky-limited depth is the
    median of the smaller half of neighbor flux errors (bright stars have
    larger Poisson flux errors and would make a too-shallow limit).
    """
    ferrs: list[float] = []
    for idx, row in neighbor_rows.items():
        meas = _row_to_measurements(
            row, blocks, mjd_cache, int(idx), int(max_flag)
        )
        ferr = _visit_flux_err(meas, image_names)
        if np.isfinite(ferr) and ferr > 0.0:
            ferrs.append(ferr)
    n_used = len(ferrs)
    if n_used == 0:
        return float('nan'), 0
    arr = np.sort(np.asarray(ferrs, dtype=float))
    sky = arr[: max(1, n_used // 2)]
    mag_lim = three_sigma_limit_abmag(float(np.median(sky)), nsigma=nsigma)
    return mag_lim, n_used


def apply_neighbor_limits(
    lightcurve: Table,
    phot_file: str | Path,
    source_index: int,
    *,
    columns: str | Path | None = None,
    param_file: str | Path | None = None,
    max_flag: int = 7,
    nsigma: float = 3.0,
    limit_radius_pix: float = 50.0,
    min_neighbors: int = 5,
    blocks: Sequence[PerImageColumn] | None = None,
    mjd_cache: dict[str, float | None] | None = None,
) -> Table:
    """
    Replace sub-*nsigma* epochs with a local 3-sigma AB limit.

    Detections (combined SNR >= *nsigma*) keep their measured mag/magerr.
    Other epochs set ``mag`` to the limit from nearby sources on the same
    visit frames, ``magerr`` to NaN, and ``detection`` to False.
    """
    lc = lightcurve.copy()
    n_row = len(lc)
    if n_row == 0:
        return lc
    if 'snr' not in lc.colnames:
        snr = np.array(
            [
                combined_snr(float(f), float(e))
                for f, e in zip(lc['flux_ujy'], lc['flux_err_ujy'])
            ],
            dtype=float,
        )
        lc['snr'] = snr
    detection = np.array(
        [np.isfinite(s) and float(s) >= float(nsigma) for s in lc['snr']],
        dtype=bool,
    )
    lc['detection'] = detection
    lc['mag_meas'] = np.array(lc['mag'], dtype=float, copy=True)
    lc['magerr_meas'] = np.array(lc['magerr'], dtype=float, copy=True)
    lc['n_neighbors'] = np.zeros(n_row, dtype=int)
    need = np.flatnonzero(~detection)
    if need.size == 0:
        return lc

    phot = Path(phot_file)
    col_path = Path(columns) if columns is not None else Path(str(phot) + '.columns')
    if blocks is None:
        blocks = parse_per_image_phot_columns(col_path)
    param = _resolve_param_file(
        phot, None if param_file is None else Path(param_file)
    )
    if mjd_cache is None:
        mjd_cache = _build_mjd_cache(blocks, phot, param)

    x, y, objtype = _load_catalog_xy_type(phot)
    neigh = _neighbor_indices(
        x,
        y,
        objtype,
        int(source_index),
        float(limit_radius_pix),
        min_neighbors=int(min_neighbors),
    )
    neighbor_rows = _load_phot_rows(phot, neigh.tolist())

    mag = np.array(lc['mag'], dtype=float, copy=True)
    magerr = np.array(lc['magerr'], dtype=float, copy=True)
    n_neighbors = np.array(lc['n_neighbors'], dtype=int, copy=True)
    for i in need:
        images = [p for p in str(lc['images'][i]).split(',') if p]
        lim, nnb = local_visit_limit_abmag(
            neighbor_rows,
            blocks,
            mjd_cache,
            images,
            max_flag=int(max_flag),
            nsigma=float(nsigma),
        )
        if not np.isfinite(lim):
            lim = three_sigma_limit_abmag(
                float(lc['flux_err_ujy'][i]), nsigma=float(nsigma)
            )
        mag[i] = lim
        magerr[i] = float('nan')
        n_neighbors[i] = int(nnb)
    lc['mag'] = mag
    lc['magerr'] = magerr
    lc['n_neighbors'] = n_neighbors
    return lc


def source_lightcurve(
    phot_file: str | Path,
    source_index: int,
    *,
    columns: str | Path | None = None,
    param_file: str | Path | None = None,
    max_gap_days: float = 1.0,
    max_flag: int = 7,
    nsigma: float = 3.0,
    limit_radius_pix: float = 50.0,
) -> Table:
    """
    Visit-binned AB light curve for one DOLPHOT catalog source.

    Per-image magnitudes are grouped by filter and by MJD (frames within
    *max_gap_days*), combined with inverse-variance weights in flux, and
    converted back to AB mag. Epochs with combined SNR below *nsigma* are
    reported as 3-sigma limits from nearby sources on the same frames.
    """
    phot = Path(phot_file)
    col_path = Path(columns) if columns is not None else Path(str(phot) + '.columns')
    blocks = parse_per_image_phot_columns(col_path)
    param = _resolve_param_file(
        phot, None if param_file is None else Path(param_file)
    )
    mjd_cache = _build_mjd_cache(blocks, phot, param)
    per_image = source_image_photometry(
        phot,
        source_index,
        columns=col_path,
        param_file=param,
        max_flag=max_flag,
        blocks=blocks,
        mjd_cache=mjd_cache,
    )
    lc = combine_visit_photometry(per_image, max_gap_days=max_gap_days)
    return apply_neighbor_limits(
        lc,
        phot,
        source_index,
        columns=col_path,
        param_file=param,
        max_flag=max_flag,
        nsigma=nsigma,
        limit_radius_pix=limit_radius_pix,
        blocks=blocks,
        mjd_cache=mjd_cache,
    )


def save_photfiles(
    photfile_path: str,
    outdir: str,
    obj: str,
    chunksize: int = 100000,
) -> None:
    """
    Write quality-cut DOLPHOT photometry chunks to CSV files.

    Parameters
    ----------
    photfile_path : str
        Directory containing DOLPHOT ``*.phot`` files.
    outdir : str
        Directory for output CSV files.
    obj : str
        Object name prefix for output filenames.
    chunksize : int, optional
        Rows read per chunk from the photometry file.

    Returns
    -------
    None
    """
    photfiles = sorted(glob.glob(os.path.join(photfile_path, '*phot')))[:1]
    for i, photfile in enumerate(photfiles):
        column_file = photfile + '.columns'
        #map columns to indices
        col_idx, filters, _ = map_columns(column_file)

        #read photometry file in chunks
        photdf = pd.read_csv(photfile, sep=r'\s+', memory_map=True,
                             header = None, iterator = True, chunksize = chunksize)

        for j, chunk in tqdm(enumerate(photdf)):
            #apply cuts
            cuts = (chunk[col_idx['SNR']] >= 10) & \
                    ((chunk[col_idx['Sharpness']])**2 <= 0.01) & \
                    (chunk[col_idx['Crowding']] <= 0.5) & \
                    (chunk[col_idx['Type']] <=2)

            #generate index by combining x and y positions
            #the x, y positions are rounded to the floor of the value since dolphot output
            #does not use the same star center from different runs
            df_idx = ["{:.2f}_{:.2f}".format(x, y) for x, y in zip(np.floor(np.array(chunk[2])), np.floor(np.array(chunk[3])))]
            chunk['idx'] = df_idx
            chunk.set_index('idx', inplace = True)
            chunk[cuts].to_csv(f'{outdir}/{obj}_{i}_{j}.csv', mode = 'a', header = False)


def create_common_catalog(
    common_ids: Sequence[str],
    dfs: Sequence[pd.DataFrame],
    columns: Sequence[str],
    outfile: str,
) -> None:
    """
    Combine photometry for sources shared across catalogs into one CSV.

    Parameters
    ----------
    common_ids : sequence of str
        Shared source index keys (``x_y`` strings).
    dfs : sequence of pandas.DataFrame
        Per-run photometry tables indexed by source key.
    columns : sequence of str
        Paths to DOLPHOT ``*.columns`` files matching ``dfs``.
    outfile : str
        Output combined-catalog CSV path.

    Returns
    -------
    None
    """
    combined_data = []
    all_filters = ['F115W', 'F150W', 'F200W', 'F277W', 'F360M']

    for source_id in tqdm(common_ids):
        phot_row = []
        positions = []
        for filt in all_filters:
            invvar_sum, weight_sum = 0, 0
            for df, col in zip(dfs, columns):
                if source_id not in df.index:
                    continue
                col_dict, filters, filter_cols = map_columns(col)
                positions.append([df[str(col_dict['X'])].loc[source_id], df[str(col_dict['Y'])].loc[source_id]])
                if filt not in filters:
                    continue
                #weighted average of magnitudes
                counts = np.array(df[filter_cols[filt]['Counts']].loc[source_id])
                err = np.array(df[filter_cols[filt]['Err']].loc[source_id])
                flag = np.array(df[filter_cols[filt]['Flag']].loc[source_id])
                good = (flag < 8) & (counts > 0)
                good_counts, good_err = counts[good], err[good]
                invvar_sum += np.sum(good_counts / good_err**2)
                weight_sum += np.sum(1 / good_err**2)

            if invvar_sum > 0:
                mag = -2.5*np.log10(invvar_sum/weight_sum)
                #dolphot calculates magnitude uncertainties as below
                magerr = 1.0857362*(1/np.sqrt(weight_sum))/(invvar_sum/weight_sum)
            else:
                mag, magerr = 99.99, 99.99
            phot_row.extend([mag, magerr])
        phot_row = [np.mean(np.array(positions)[:, 0]), np.mean(np.array(positions)[:, 1])] + phot_row
        combined_data.append(phot_row)

    combined_data = np.array(combined_data)
    #save combined photometry to csv
    df_col = ['X', 'Y'] + [[f'{filt}_mag', f'{filt}_err'] for filt in all_filters]
    df_col = list(itertools.chain(*df_col))
    df = pd.DataFrame(combined_data, columns = df_col)
    df.to_csv(outfile, index = False)
