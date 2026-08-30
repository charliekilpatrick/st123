"""
Parse DOLPHOT multi-extension output (catalog, ``.columns``, ``.param``, ``.data``,
``.info``, ``.warnings``) and export a single HDF5 file with a labeled catalog plus metadata.

Adapted from ``hst123.utils.dolphot_catalog_hdf5`` for st123 photometry directories
(``<run>/<run>.phot`` + ``dolphot.param``). Layout attrs keep the ``hst123_*``
names for read compatibility with hst123 tooling.

Designed against DOLPHOT 3.x text products: whitespace-separated numeric catalog,
one descriptive line per column in ``*.columns``, and sidecar metadata files.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

PathLike = Union[str, Path]

# HDF5 column names: alphanumeric + underscore; must not start with digit in some tools
_MAX_HDF5_NAME_LEN = 63

# Astropy writes tables as a single compound HDF5 datatype. HDF5 object headers
# reject compound types with too many fields (~1092 with short names; fewer with
# long DOLPHOT-derived names). Above this count we write a 2-D float matrix.
_MAX_COMPOUND_HDF5_COLUMNS = 800

# Prefer datasets over attributes for large JSON (HDF5 attr / object-header limits).
_HDF5_ATTR_JSON_SOFT_LIMIT = 16_384

_LAYOUT_COMPOUND = "compound"
_LAYOUT_MATRIX = "matrix"


@dataclass(frozen=True)
class DolphotColumn:
    """One row from a DOLPHOT ``*.columns`` file."""

    index_1based: int
    index_0based: int
    description: str
    raw_line: str


def parse_column_index_and_description(line: str) -> Optional[tuple[int, str]]:
    """
    Parse a single ``*.columns`` line of the form ``N. description``.

    Returns
    -------
    (1-based index, description) or None if the line does not match.
    """
    m = re.match(r"^\s*(\d+)\.\s*(.*)$", line.rstrip("\n"))
    if not m:
        return None
    n = int(m.group(1))
    desc = m.group(2).strip()
    return (n, desc)


def parse_dolphot_columns_file(path: PathLike) -> list[DolphotColumn]:
    """
    Read a DOLPHOT ``*.columns`` file into structured column definitions.

    Parameters
    ----------
    path : path-like
        Path to ``dpXXXX.columns``.

    Returns
    -------
    list of DolphotColumn
        One entry per line, in file order (column 1 .. N).
    """
    path = Path(path)
    out: list[DolphotColumn] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_column_index_and_description(line)
            if not parsed:
                continue
            n1, desc = parsed
            out.append(
                DolphotColumn(
                    index_1based=n1,
                    index_0based=n1 - 1,
                    description=desc,
                    raw_line=line.rstrip("\n"),
                )
            )
    return out


def _sanitize_hdf5_column_name(description: str, index_1based: int) -> str:
    """Build a unique, HDF5-safe ASCII name from a full column description."""
    # Drop path noise for readability but keep filter/instrument hints
    s = description.strip()
    s = re.sub(r"/[^\s]+", "", s)  # remove absolute paths
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = f"col_{index_1based}"
    if s[0].isdigit():
        s = "c_" + s
    if len(s) > _MAX_HDF5_NAME_LEN:
        s = s[: _MAX_HDF5_NAME_LEN]
    return s


def unique_hdf5_column_names(columns: list[DolphotColumn]) -> list[str]:
    """Return safe column names, de-duplicated with numeric suffixes if needed."""
    names: list[str] = []
    used: set[str] = set()
    for c in columns:
        base = _sanitize_hdf5_column_name(c.description, c.index_1based)
        cand = base
        n_dup = 0
        while cand in used:
            n_dup += 1
            suffix = f"_{n_dup}"
            cand = (base[: _MAX_HDF5_NAME_LEN - len(suffix)] + suffix).rstrip("_")
        used.add(cand)
        names.append(cand)
    return names


def find_column_index_0based(
    columns: list[DolphotColumn],
    key: str,
    image: str = "",
) -> Optional[int]:
    """
    Match DOLPHOT ``*.columns`` semantics used by
    :meth:`hst123.primitives.scrape_dolphot.ScrapeDolphotPrimitive.get_dolphot_column`.

    *key* must appear in the column description. If *image* is non-empty, it must
    appear in the description (after stripping a ``.fits`` suffix). If *image* is
    empty, any column whose description contains *key* may match—the first such
    column is returned (same as substring ``"" in line`` being true for every line).
    """
    img = image.replace(".fits", "")
    for col in columns:
        if key not in col.description:
            continue
        if not img:
            return col.index_0based
        if img in col.description:
            return col.index_0based
    return None


def load_dolphot_catalog_array(path: PathLike) -> np.ndarray:
    """
    Load the main DOLPHOT catalog (numeric rows, whitespace-separated).

    Parameters
    ----------
    path : path-like
        Path to ``dpXXXX`` (no extension), same as ``dolphot['base']`` in the pipeline.

    Returns
    -------
    numpy.ndarray
        2-D float array of shape (n_sources, n_columns).

    Notes
    -----
    Uses :func:`numpy.loadtxt`. Scraping uses a single load plus vectorized
    filtering (see :mod:`hst123.primitives.scrape_dolphot`) so the catalog is
    read once per scrape.
    """
    path = Path(path)
    return np.loadtxt(path, dtype=np.float64)


def dolphot_columns_to_astropy_table(
    catalog: np.ndarray,
    columns: list[DolphotColumn],
    names: Optional[list[str]] = None,
):
    """
    Build an :class:`astropy.table.Table` with descriptive column names.

    Parameters
    ----------
    catalog : ndarray
        Output of :func:`load_dolphot_catalog_array`.
    columns : list of DolphotColumn
        From :func:`parse_dolphot_columns_file`.
    names : list of str, optional
        HDF5-safe names; defaults to :func:`unique_hdf5_column_names`.
    """
    from astropy.table import Table

    if catalog.ndim != 2:
        raise ValueError("catalog must be a 2-D array")
    ncol = catalog.shape[1]
    if len(columns) != ncol:
        raise ValueError(
            f"column definition count ({len(columns)}) != catalog columns ({ncol})"
        )
    if names is None:
        names = unique_hdf5_column_names(columns)
    if len(names) != ncol:
        raise ValueError("names length must match catalog columns")
    return Table(data=[catalog[:, i] for i in range(ncol)], names=names)


def parse_dolphot_param_file(path: PathLike) -> dict[str, str]:
    """
    Parse ``dpXXXX.param`` (``key = value`` lines, DOLPHOT style).
    """
    path = Path(path)
    out: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def group_param_by_image(param: Mapping[str, str]) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """
    Split flat ``imgNNNN_*`` keys into per-image dicts (``img0001``, ...).
    """
    global_params: dict[str, str] = {}
    by_image: dict[str, dict[str, str]] = defaultdict(dict)
    for k, v in param.items():
        m = re.match(r"^img(\d+)_(.+)$", k)
        if m:
            img_key = f"img{m.group(1)}"
            sub_key = m.group(2)
            by_image[img_key][sub_key] = v
        else:
            global_params[k] = v
    return global_params, dict(by_image)


def parse_dolphot_info_file(path: PathLike) -> dict[str, Any]:
    """
    Parse ``dpXXXX.info``: MJD per input image, limits, filter/exptime lines, alignment, apcor.

    Structure varies slightly by instrument; unknown lines are kept in ``extra_lines``.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: dict[str, Any] = {"raw_text": text, "paths_mjd": []}

    i = 0
    if lines:
        msets = re.match(r"^(\d+)\s+sets\s+of\s+output", lines[0], re.I)
        if msets:
            out["n_output_sets"] = int(msets.group(1))
            i = 1

    # Alternating: absolute path, then indented MJD
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue
        if ln.startswith("EXTENSION") or ln.startswith("Limits"):
            break
        if ln.startswith("/") or ln.startswith("~"):
            path_line = ln
            mjd: Optional[float] = None
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                try:
                    mjd = float(nxt)
                    i += 2
                except ValueError:
                    i += 1
            else:
                i += 1
            out["paths_mjd"].append({"path": path_line, "mjd": mjd})
            continue
        i += 1

    # Limits line after "Limits"
    for j in range(len(lines)):
        if lines[j].strip() == "Limits" and j + 1 < len(lines):
            parts = lines[j + 1].split()
            if len(parts) >= 4:
                try:
                    out["limits_xy"] = [int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])]
                except ValueError:
                    out["limits_xy_raw"] = lines[j + 1]
            break

    image_meta: list[dict[str, Any]] = []
    for ln in lines:
        m = re.match(r"^\*\s+image\s+(\d+):\s+(\S+)\s+(\S+)\s+(\S+)", ln)
        if m:
            image_meta.append(
                {
                    "image_index": int(m.group(1)),
                    "filter": m.group(2),
                    "chip": m.group(3),
                    "exptime": float(m.group(4)),
                }
            )
    if image_meta:
        out["image_filter_exptime"] = image_meta

    # Alignment block: after "Alignment" keyword, next non-empty lines until blank or keyword
    align_rows: list[list[float]] = []
    for j, ln in enumerate(lines):
        if ln.strip() == "Alignment":
            k = j + 1
            while k < len(lines):
                row = lines[k].strip()
                if not row or row.startswith("*") or row.startswith("Aperture"):
                    break
                nums = [float(x) for x in row.split()]
                if len(nums) >= 5:
                    align_rows.append(nums)
                k += 1
            break
    if align_rows:
        out["alignment"] = align_rows

    apcor: list[float] = []
    for j, ln in enumerate(lines):
        if ln.strip() == "Aperture corrections":
            k = j + 1
            while k < len(lines):
                row = lines[k].strip()
                if not row:
                    break
                try:
                    apcor.append(float(row.split()[0]))
                except ValueError:
                    pass
                k += 1
            break
    if apcor:
        out["aperture_corrections"] = apcor

    return out


def parse_dolphot_data_file(path: PathLike) -> dict[str, Any]:
    """
    Parse ``dpXXXX.data``: WCS, alignment iteration counts, PSF quality, aperture-correction stats.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: dict[str, Any] = {"raw_text": text}

    wcs_lines: list[dict[str, Any]] = []
    for ln in lines:
        m = re.match(r"^WCS image\s+(\d+):\s+(.*)$", ln.strip())
        if m:
            parts = m.group(2).split()
            nums = [float(x) for x in parts]
            wcs_lines.append({"image_index": int(m.group(1)), "values": nums})
    if wcs_lines:
        out["wcs"] = wcs_lines

    align_header = None
    for ln in lines:
        m = re.match(r"^Align:\s+(\d+)", ln.strip())
        if m:
            align_header = int(m.group(1))
            break
    if align_header is not None:
        out["align_n_stars"] = align_header

    align_img: list[dict[str, Any]] = []
    for ln in lines:
        m = re.match(r"^Align image\s+(\d+):\s+(.*)$", ln.strip())
        if m:
            parts = m.group(2).split()
            nums = [float(x) for x in parts[2:]] if len(parts) > 2 else []
            align_img.append(
                {
                    "image_index": int(m.group(1)),
                    "n1": int(parts[0]) if len(parts) > 0 else None,
                    "n2": int(parts[1]) if len(parts) > 1 else None,
                    "values": nums,
                }
            )
    if align_img:
        out["align_images"] = align_img

    psf_lines: list[dict[str, Any]] = []
    for ln in lines:
        m = re.match(r"^PSF image\s+(\d+):\s+(.*)$", ln.strip())
        if m:
            parts = m.group(2).split()
            nums = [float(x) for x in parts]
            psf_lines.append({"image_index": int(m.group(1)), "values": nums})
    if psf_lines:
        out["psf"] = psf_lines

    apcor_lines: list[dict[str, Any]] = []
    for ln in lines:
        m = re.match(r"^Apcor image\s+(\d+):\s+(.*)$", ln.strip())
        if m:
            parts = m.group(2).split()
            nums = [float(x) for x in parts]
            apcor_lines.append({"image_index": int(m.group(1)), "values": nums})
    if apcor_lines:
        out["apcor"] = apcor_lines

    return out


def parse_dolphot_warnings_file(path: PathLike) -> dict[str, Any]:
    """
    Parse ``dpXXXX.warnings`` into all lines and optional buckets by image path.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    by_path: dict[str, list[str]] = defaultdict(list)
    global_lines: list[str] = []
    # Match full paths ending in .chipN or .fits
    path_re = re.compile(r"(/[^\s,]+(?:\.fits|\.chip\d+))")
    for ln in lines:
        found = path_re.findall(ln)
        if found:
            for pth in found:
                by_path[pth].append(ln)
        else:
            global_lines.append(ln)
    return {
        "lines": lines,
        "by_image_path": dict(by_path),
        "global_lines": global_lines,
    }


def merge_image_metadata(
    param: Mapping[str, str],
    info: Mapping[str, Any],
    data: Mapping[str, Any],
    warnings: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Combine parsed sidecars into one JSON-serializable tree keyed by ``img0000`` .. ``imgNNNN``.
    """
    _, by_img = group_param_by_image(param)
    merged: dict[str, Any] = {"images": {}}

    path_to_img: dict[str, str] = {}
    for ik, kv in by_img.items():
        fp = kv.get("file")
        if fp:
            path_to_img[str(Path(fp).resolve())] = ik
            path_to_img[fp] = ik

    pmjd = info.get("paths_mjd") or []
    for entry in pmjd:
        p = entry.get("path")
        if not p:
            continue
        key = None
        try:
            rp = str(Path(p).resolve())
            key = path_to_img.get(rp) or path_to_img.get(p)
        except OSError:
            key = path_to_img.get(p)
        if key and key not in merged["images"]:
            merged["images"][key] = {"param": by_img.get(key, {})}
        if key:
            merged["images"][key].setdefault("mjd", entry.get("mjd"))

    for ik in by_img:
        merged["images"].setdefault(ik, {})
        merged["images"][ik]["param"] = by_img[ik]

    ife = info.get("image_filter_exptime") or []
    for row in ife:
        idx = row.get("image_index")
        if idx is None:
            continue
        ik = f"img{idx:04d}"
        merged["images"].setdefault(ik, {})
        merged["images"][ik]["info_filter_exptime"] = row

    wcs = data.get("wcs") or []
    for w in wcs:
        idx = w.get("image_index")
        if idx is None:
            continue
        ik = f"img{idx:04d}"
        merged["images"].setdefault(ik, {})
        merged["images"][ik]["data_wcs"] = w

    for w in data.get("align_images") or []:
        idx = w.get("image_index")
        if idx is None:
            continue
        ik = f"img{idx:04d}"
        merged["images"].setdefault(ik, {})
        merged["images"][ik]["data_align"] = w

    for w in data.get("psf") or []:
        idx = w.get("image_index")
        if idx is None:
            continue
        ik = f"img{idx:04d}"
        merged["images"].setdefault(ik, {})
        merged["images"][ik]["data_psf"] = w

    for w in data.get("apcor") or []:
        idx = w.get("image_index")
        if idx is None:
            continue
        ik = f"img{idx:04d}"
        merged["images"].setdefault(ik, {})
        merged["images"][ik]["data_apcor"] = w

    warn_by = warnings.get("by_image_path") or {}
    for pth, wlines in warn_by.items():
        key = path_to_img.get(pth)
        if not key:
            try:
                key = path_to_img.get(str(Path(pth).resolve()))
            except OSError:
                key = None
        if key:
            merged["images"].setdefault(key, {})
            merged["images"][key]["warnings"] = wlines

    merged["global_param"], _ = group_param_by_image(param)
    merged["info_parsed"] = {k: v for k, v in info.items() if k != "raw_text"}
    merged["warnings_global_lines"] = warnings.get("global_lines", [])
    return merged


def _embed_dolphot_directory_metadata(
    hf: Any,
    dolphot_dir: Path,
    *,
    embed_text: bool = False,
    max_text_bytes: int = 8 * 1024 * 1024,
) -> None:
    """
    Under ``metadata/dolphot_directory/``, store a JSON manifest of every file
    under *dolphot_dir* (path, size, mtime). Optionally embed UTF-8 text for
    non-FITS files up to *max_text_bytes* (skips binary-looking content).
    """
    import h5py

    root = dolphot_dir.resolve()
    manifest: list[dict[str, Any]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        rel = p.relative_to(root).as_posix()
        manifest.append(
            {"relpath": rel, "size": st.st_size, "mtime": st.st_mtime}
        )

    meta = hf.require_group("metadata")
    gpath = "metadata/dolphot_directory"
    if gpath in hf:
        del hf[gpath]
    dg = meta.create_group("dolphot_directory")
    dt = h5py.string_dtype(encoding="utf-8")
    mj = dg.create_dataset("manifest_json", (1,), dtype=dt)
    mj[0] = json.dumps(manifest, indent=2)
    dg.attrs["root"] = str(root)
    dg.attrs["n_files"] = len(manifest)
    dg.attrs["text_embed"] = bool(embed_text)

    if not embed_text:
        return

    tg = dg.create_group("text_embed")
    i = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if ".fits" in p.name.lower():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_size == 0 or st.st_size > max_text_bytes:
            continue
        try:
            raw = p.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:8192]:
            continue
        try:
            txt = raw.decode("utf-8")
        except UnicodeDecodeError:
            txt = raw.decode("utf-8", errors="replace")
        ds = tg.create_dataset(f"file_{i:06d}", (1,), dtype=dt)
        ds[0] = txt
        ds.attrs["relpath"] = rel
        ds.attrs["size_bytes"] = st.st_size
        i += 1


def _write_utf8_dataset(g: Any, key: str, text: str) -> None:
    """Store *text* as a length-1 UTF-8 string dataset under group *g*."""
    import h5py

    dt = h5py.string_dtype(encoding="utf-8")
    if key in g:
        del g[key]
    ds = g.create_dataset(key, (1,), dtype=dt)
    ds[0] = text


def _store_json_payload(container: Any, key: str, payload: Any) -> None:
    """
    Store JSON on *container*.

    Small payloads stay as attributes (legacy-compatible). Large payloads are
    written as UTF-8 datasets on the same group (datasets cannot live on an
    HDF5 Dataset, so callers must pass a Group when the value may be large).
    """
    text = json.dumps(payload, indent=2)
    if len(text.encode("utf-8")) <= _HDF5_ATTR_JSON_SOFT_LIMIT:
        container.attrs[key] = text
        return
    # h5py Dataset has no create_dataset; require a Group for oversized JSON
    if not hasattr(container, "create_dataset"):
        raise TypeError(
            f"JSON payload for {key!r} exceeds attr soft limit "
            f"({_HDF5_ATTR_JSON_SOFT_LIMIT} bytes); pass a Group to store a dataset"
        )
    _write_utf8_dataset(container, key, text)


def _load_json_payload(container: Any, key: str, fallback_group: Any = None) -> Any:
    """Read JSON from an attribute or a same-named UTF-8 dataset."""
    if key in getattr(container, "attrs", {}):
        return json.loads(container.attrs[key])
    for g in (container, fallback_group):
        if g is None or key not in g:
            continue
        val = g[key][0]
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        return json.loads(val)
    raise KeyError(key)


def _write_photometry_matrix_hdf5(
    out_path: Path,
    catalog: np.ndarray,
    names: list[str],
    *,
    photometry_path: str,
    compression: bool,
) -> None:
    """Write catalog as ``photometry/data`` (n×m float64) + ``column_names``."""
    import h5py

    out_path = Path(out_path)
    if out_path.is_file():
        out_path.unlink()
    with h5py.File(out_path, "w") as hf:
        g = hf.create_group(photometry_path)
        ds_kw: dict[str, Any] = {}
        if compression:
            ds_kw["compression"] = "gzip"
            ds_kw["compression_opts"] = 4
        g.create_dataset("data", data=np.asarray(catalog, dtype=np.float64), **ds_kw)
        dt = h5py.string_dtype(encoding="utf-8")
        g.create_dataset(
            "column_names",
            data=np.asarray(names, dtype=object),
            dtype=dt,
        )
        g.attrs["hst123_dolphot_hdf5_layout"] = _LAYOUT_MATRIX
        g.attrs["hst123_dolphot_hdf5_format"] = 1
        g.attrs["st123_dolphot_hdf5_layout"] = _LAYOUT_MATRIX
        g.attrs["st123_dolphot_hdf5_format"] = 1
        g.attrs["n_sources"] = int(catalog.shape[0])
        g.attrs["n_columns"] = int(catalog.shape[1])


def write_dolphot_catalog_hdf5(
    out_path: PathLike,
    base: PathLike,
    *,
    photometry_path: str = "photometry",
    include_raw_sidecars: bool = True,
    compression: bool = True,
    dolphot_dir: Optional[PathLike] = None,
    embed_dolphot_directory_text: bool = False,
    catalog_array: Optional[np.ndarray] = None,
    include_directory_manifest: bool = True,
    serialize_meta: bool = True,
) -> Path:
    """
    Write one HDF5 file with the full DOLPHOT catalog and metadata.

    Parameters
    ----------
    out_path : path-like
        Output ``.h5`` file.
    base : path-like
        DOLPHOT output base path **without extension** (same as pipeline ``dolphot['base']``).

    photometry_path : str
        HDF5 path for the catalog. Narrow catalogs use Astropy's compound table
        dataset; wide catalogs (many columns) use a group with a 2-D ``data``
        matrix plus ``column_names`` (HDF5 compound dtypes hit object-header limits).

    include_raw_sidecars : bool
        If True, store UTF-8 text of ``.param``, ``.info``, ``.data``, ``.warnings``,
        and ``.columns`` as datasets under ``metadata/raw/``.

    compression : bool
        If True, use gzip compression on the table (requires Astropy + h5py).

    dolphot_dir : path-like, optional
        Directory scanned for a full file manifest (default: ``base.parent``, i.e.
        the DOLPHOT output folder such as ``<work>/dolphot/``). Every file path,
        size, and mtime are stored. Optional text embedding under
        ``metadata/dolphot_directory/text_embed/`` when *embed_dolphot_directory_text*
        is True.
    embed_dolphot_directory_text : bool, optional
        If True, embed small UTF-8 text files from *dolphot_dir* (can be slow for
        large trees). Default False (manifest only).
    catalog_array : ndarray, optional
        If provided, skips re-reading the DOLPHOT ``base`` catalog from disk
        (avoids a second full ``numpy.loadtxt`` when the pipeline already holds
        the array in memory after scraping).
    include_directory_manifest : bool, optional
        If True, walk *dolphot_dir* and store a JSON file manifest (can be slow
        for trees with very many files). Set False for faster HDF5 when the
        manifest is not needed.
    serialize_meta : bool, optional
        If False, skip rich Astropy table metadata (faster writes, smaller files).
        Ignored for matrix-layout (wide) catalogs; column descriptions remain in
        ``metadata/dolphot_column_json``.

    Returns
    -------
    pathlib.Path
        Path to the written file.

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - env-specific
        raise ImportError(
            "write_dolphot_catalog_hdf5 requires h5py; pip install h5py"
        ) from exc

    base = Path(base)
    out_path = Path(out_path)
    if dolphot_dir is None:
        dolphot_dir = base.parent
    else:
        dolphot_dir = Path(dolphot_dir)

    col_path = Path(str(base) + ".columns")
    if not col_path.is_file():
        raise FileNotFoundError(f"missing columns file: {col_path}")

    columns = parse_dolphot_columns_file(col_path)
    names = unique_hdf5_column_names(columns)
    if catalog_array is not None:
        catalog = np.asarray(catalog_array, dtype=np.float64)
        if catalog.ndim != 2:
            raise ValueError("catalog_array must be 2-D")
    else:
        catalog = load_dolphot_catalog_array(base)

    use_matrix = len(names) > _MAX_COMPOUND_HDF5_COLUMNS
    if use_matrix:
        _write_photometry_matrix_hdf5(
            out_path,
            catalog,
            names,
            photometry_path=photometry_path,
            compression=compression,
        )
    else:
        table = dolphot_columns_to_astropy_table(catalog, columns, names=names)
        if serialize_meta:
            for i, col in enumerate(columns):
                table.meta[f"dolphot_desc_{names[i]}"] = col.description
                table.meta[f"dolphot_index_{names[i]}"] = col.index_1based

        write_kw: dict[str, Any] = {
            "format": "hdf5",
            "path": photometry_path,
            "overwrite": True,
            "serialize_meta": serialize_meta,
        }
        if compression:
            write_kw["compression"] = "gzip"
        try:
            table.write(str(out_path), **write_kw)
        except ValueError as exc:
            if "object header" not in str(exc).lower():
                raise
            # Compound dtype still too large (long column names); fall back.
            _write_photometry_matrix_hdf5(
                out_path,
                catalog,
                names,
                photometry_path=photometry_path,
                compression=compression,
            )
            use_matrix = True

    param_p = Path(str(base) + ".param")
    if not param_p.is_file():
        # st123 run dirs use a shared dolphot.param input file.
        alt_param = base.parent / "dolphot.param"
        if alt_param.is_file():
            param_p = alt_param
    info_p = Path(str(base) + ".info")
    data_p = Path(str(base) + ".data")
    warn_p = Path(str(base) + ".warnings")

    param_d: dict[str, str] = parse_dolphot_param_file(param_p) if param_p.is_file() else {}
    info_d = parse_dolphot_info_file(info_p) if info_p.is_file() else {}
    data_d = parse_dolphot_data_file(data_p) if data_p.is_file() else {}
    warn_d = parse_dolphot_warnings_file(warn_p) if warn_p.is_file() else {}

    merged = merge_image_metadata(param_d, info_d, data_d, warn_d)
    column_payload = [
        {
            "index_1based": c.index_1based,
            "description": c.description,
            "hdf5_name": names[i],
        }
        for i, c in enumerate(columns)
    ]

    with h5py.File(out_path, "a") as hf:
        root = hf[photometry_path]
        root.attrs["hst123_dolphot_hdf5_format"] = 1
        root.attrs["st123_dolphot_hdf5_format"] = 1
        root.attrs["dolphot_base"] = str(base)
        if "hst123_dolphot_hdf5_layout" not in root.attrs:
            layout = _LAYOUT_MATRIX if use_matrix else _LAYOUT_COMPOUND
            root.attrs["hst123_dolphot_hdf5_layout"] = layout
            root.attrs["st123_dolphot_hdf5_layout"] = layout
        elif "st123_dolphot_hdf5_layout" not in root.attrs:
            root.attrs["st123_dolphot_hdf5_layout"] = root.attrs[
                "hst123_dolphot_hdf5_layout"
            ]

        meta = hf.require_group("metadata")
        # Large JSON always under metadata/ (Group) so oversized blobs are datasets.
        _store_json_payload(meta, "dolphot_column_json", column_payload)
        _store_json_payload(meta, "dolphot_merged_metadata_json", merged)
        _store_json_payload(meta, "global_param_json", merged.get("global_param", {}))
        _store_json_payload(meta, "merged_images_json", merged.get("images", {}))

        # Keep small pointers on photometry for tools that only open that path;
        # skip when payloads are large (attrs would fail / bloat object headers).
        for attr_key, payload in (
            ("dolphot_column_json", column_payload),
            ("dolphot_merged_metadata_json", merged),
        ):
            text = json.dumps(payload, indent=2)
            if len(text.encode("utf-8")) <= _HDF5_ATTR_JSON_SOFT_LIMIT:
                root.attrs[attr_key] = text

        if include_raw_sidecars:
            raw = meta.require_group("raw")
            if col_path.is_file():
                _write_utf8_dataset(
                    raw, "columns", col_path.read_text(encoding="utf-8", errors="replace")
                )
            if param_p.is_file():
                _write_utf8_dataset(
                    raw, "param", param_p.read_text(encoding="utf-8", errors="replace")
                )
            if info_p.is_file():
                _write_utf8_dataset(
                    raw, "info", info_p.read_text(encoding="utf-8", errors="replace")
                )
            if data_p.is_file():
                _write_utf8_dataset(
                    raw, "data", data_p.read_text(encoding="utf-8", errors="replace")
                )
            if warn_p.is_file():
                _write_utf8_dataset(
                    raw, "warnings", warn_p.read_text(encoding="utf-8", errors="replace")
                )

        if include_directory_manifest and dolphot_dir.is_dir():
            _embed_dolphot_directory_metadata(
                hf,
                dolphot_dir,
                embed_text=embed_dolphot_directory_text,
            )

    return out_path


def append_scraped_final_phot_hdf5(
    out_path: PathLike,
    final_phot: list,
    *,
    group_path: str = "scraped_photometry",
    compression: bool = False,
    serialize_meta: bool = False,
) -> None:
    """
    Append all scraped-source photometry tables as one stacked dataset in an existing HDF5 file.

    Adds column ``scrape_source_index`` (0-based) so rows from different sources
    can be split after readback. Uses a single HDF5 write (no per-source files).

    Parameters
    ----------
    out_path : path-like
        Existing file produced by :func:`write_dolphot_catalog_hdf5`.
    final_phot : list of astropy.table.Table
        One table per source from the scrape pipeline (e.g. output of ``scrapedolphot``).
    group_path : str, optional
        HDF5 path for the stacked table. Default ``scraped_photometry``.
    compression : bool, optional
        If True, gzip the appended table.
    serialize_meta : bool, optional
        If False, smaller/faster writes (recommended for large ``--scrape-all`` runs).
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - env-specific
        raise ImportError(
            "append_scraped_final_phot_hdf5 requires h5py; pip install h5py"
        ) from exc

    from astropy.table import vstack

    if not final_phot:
        return
    out_path = Path(out_path)
    if not out_path.is_file():
        raise FileNotFoundError(f"expected existing HDF5: {out_path}")

    pieces = []
    for i, t in enumerate(final_phot):
        if t is None or len(t) == 0:
            continue
        if "scrape_source_index" in t.colnames:
            raise ValueError("table already has column scrape_source_index")
        ti = t.copy()
        ti["scrape_source_index"] = np.full(len(ti), i, dtype=np.int32)
        pieces.append(ti)
    if not pieces:
        return

    stacked = vstack(pieces, metadata_conflicts="silent")
    write_kw: dict[str, Any] = {
        "format": "hdf5",
        "path": group_path,
        "append": True,
        "overwrite": False,
        "serialize_meta": serialize_meta,
    }
    if compression:
        write_kw["compression"] = "gzip"
    stacked.write(str(out_path), **write_kw)

    with h5py.File(out_path, "a") as hf:
        if group_path in hf:
            grp = hf[group_path]
            grp.attrs["hst123_scraped_photometry_format"] = 1
            grp.attrs["n_scrape_sources"] = int(len(final_phot))


def read_dolphot_catalog_hdf5(path: PathLike, photometry_path: str = "photometry"):
    """
    Read back a table written by :func:`write_dolphot_catalog_hdf5`.

    Supports both Astropy compound datasets and the matrix layout used for
    wide DOLPHOT catalogs (``photometry/data`` + ``column_names``).

    Returns
    -------
    astropy.table.Table
    """
    from astropy.table import Table

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - env-specific
        raise ImportError(
            "read_dolphot_catalog_hdf5 requires h5py; pip install h5py"
        ) from exc

    path = Path(path)
    with h5py.File(path, "r") as hf:
        node = hf[photometry_path]
        layout = node.attrs.get("hst123_dolphot_hdf5_layout", _LAYOUT_COMPOUND)
        if layout == _LAYOUT_MATRIX or (
            hasattr(node, "keys") and "data" in node and "column_names" in node
        ):
            data = np.asarray(node["data"], dtype=np.float64)
            raw_names = node["column_names"][:]
            names = [
                n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
                for n in raw_names
            ]
            return Table(data=[data[:, i] for i in range(data.shape[1])], names=names)

    return Table.read(str(path), format="hdf5", path=photometry_path)


def phot_catalog_base_for_run(
    outdir: PathLike,
    phot_out: str | None = None,
) -> Path:
    """
    Resolve the DOLPHOT catalog path for a prepared run directory.

    Parameters
    ----------
    outdir : path-like
        DOLPHOT working directory (contains ``dolphot.param`` / ``*.phot``).
    phot_out : str, optional
        Catalog basename (default ``<outdir.name>.phot``).

    Returns
    -------
    pathlib.Path
        Path to the numeric ``*.phot`` catalog (also used as the HDF5 "base").
    """
    outdir = Path(outdir)
    name = phot_out or f"{outdir.name}.phot"
    return (outdir / name).resolve()


def hdf5_path_for_phot_base(base: PathLike) -> Path:
    """
    Map a DOLPHOT catalog base to its default ``.h5`` sidecar.

    ``phot_0_0.phot`` → ``phot_0_0.h5`` (hst123-style ``<base>.h5`` without the
    ``.phot`` suffix). Any other basename keeps ``.h5`` via :meth:`Path.with_suffix`.
    """
    base = Path(base)
    name = base.name
    if name.endswith(".phot"):
        return base.with_name(f"{name[: -len('.phot')]}.h5")
    if name.endswith(".h5"):
        return base
    return base.with_suffix(".h5")


def ensure_dolphot_catalog_hdf5(
    outdir: PathLike,
    *,
    phot_out: str | None = None,
    out_path: PathLike | None = None,
    force: bool = False,
    compression: bool = True,
    include_raw_sidecars: bool = True,
    include_directory_manifest: bool = False,
    serialize_meta: bool = True,
) -> Path | None:
    """
    Write a compressed DOLPHOT catalog HDF5 for one run directory if needed.

    Skips when the target ``.h5`` already exists unless *force* is True. Returns
    ``None`` when the catalog / columns products are missing (DOLPHOT not finished).

    Parameters
    ----------
    outdir : path-like
        Prepared DOLPHOT directory.
    phot_out : str, optional
        Catalog basename under *outdir*.
    out_path : path-like, optional
        Explicit HDF5 destination (default beside the catalog).
    force : bool, optional
        Overwrite an existing HDF5 file.
    compression : bool, optional
        Gzip-compress photometry datasets (default True).
    include_raw_sidecars : bool, optional
        Embed ``.columns`` / param / info / data / warnings text.
    include_directory_manifest : bool, optional
        Walk the run directory for a file manifest (off by default; can be slow).
    serialize_meta : bool, optional
        Rich Astropy table metadata for compound layouts.

    Returns
    -------
    pathlib.Path or None
        Path written or skipped-as-existing, or ``None`` if inputs are incomplete.
    """
    base = phot_catalog_base_for_run(outdir, phot_out=phot_out)
    columns = Path(str(base) + ".columns")
    if not base.is_file() or not columns.is_file():
        return None
    dest = Path(out_path) if out_path is not None else hdf5_path_for_phot_base(base)
    if dest.is_file() and not force:
        return dest
    return write_dolphot_catalog_hdf5(
        dest,
        base,
        compression=compression,
        include_raw_sidecars=include_raw_sidecars,
        include_directory_manifest=include_directory_manifest,
        serialize_meta=serialize_meta,
        dolphot_dir=base.parent,
    )
