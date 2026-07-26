# st123

`st123` is a **space-telescope** imaging toolkit intended for **HST, JWST, and
Roman** (and related MAST-hosted missions), not a JWST-only package. It
supports downloading imaging from MAST, aligning frames with a
**custom, repository-local build of [JHAT](https://jhat.readthedocs.io/)**,
building mosaics / coadds, and preparing DOLPHOT runs.

Current workflows are strongest for JWST (and HST reference / MAST helpers);
Roman support is part of the same generalized design as facility-specific
pipelines are added.

## Provenance

`st123` is a new repository whose code descends from earlier HST and JWST
reduction toolchains and is being generalized across space telescopes. It is
**not** a GitHub fork of either project; history starts with this repository.
The main lineage is:

- [charliekilpatrick/hst123](https://github.com/charliekilpatrick/hst123) —
  original HST download / registration / drizzle / DOLPHOT tooling
- [aswinsuresh24/jwst123](https://github.com/aswinsuresh24/jwst123) —
  JWST-focused evolution of that work (including the `dev-cdk` branch used as
  the starting snapshot for `st123`)

Package, module, and CLI names that previously used `jwst123` are renamed to
`st123` in this tree to reflect the broader HST / JWST / Roman scope.

## Repository layout

| Path | Role |
| --- | --- |
| `st123/` | Installable package (library, scripts, and notebooks) |
| `st123/scripts/` | Console-script entry points (`__main__` CLIs); helpers in `scripts/utils/` |
| `st123/notebooks/` | Generic notebooks for download, alignment, and mosaics |
| `extdeps/` | Vendored / customized external packages (see below) |
| `extdeps/jhat/` | **Custom JHAT build for this repository** (not PyPI) |
| `pyproject.toml` | Package metadata; dependencies loaded from `requirements.txt` |
| `requirements.txt` | Pinned Python dependencies (includes local `jhat @ file:./extdeps/jhat`) |

### Package modules

| Module | Role |
| --- | --- |
| `st123.alignment` | Alignment package; re-exports the public surface of `align` |
| `st123.alignment.align` | Single unified module: per-filter calibrators, MIRI→MIRI fallback ranking and provenance, JHAT / Gaia alignment and visit grouping, relative align + iterative refine, MIRI↔reference overlap discovery, spawn-safe REFERENCE / MIRI_REL workers, and filter-wave orchestration |
| `st123.mosaic` | Footprints, overlap, Level-3 mosaics / coadds / GWCS |
| `st123.mosaic.region` | Illuminated footprint / `S_REGION` from science+DQ |
| `st123.mosaic.image_overlap` | Science vs reference footprint overlap |
| `st123.mosaic.mosaic` | Overlap splitting, PSF matching, coadds, GWCS |
| `st123.photometry.dolphot_prep` | DOLPHOT staging (paramfile, nircammask/mirimask, calcsky) |
| `st123.mast` | MAST query and download helpers (`mast`, `download` submodules) |
| `st123.photometry` | Per-image / combined photometry catalog helpers |
| `st123.photometry.catalog` | DOLPHOT column mapping and combined catalogs |
| `st123.utils` | Shared utilities package (helpers, settings, link, logging) |
| `st123.utils.helpers` | Coordinates, FITS bookkeeping, visits, xmatch |
| `st123.utils.settings` | JHAT and DOLPHOT parameter sets |
| `st123.utils.link` | Symlink helpers for reduction ``raw/`` trees |
| `st123.utils.logging` | POTPyRI-style console/file logging (`<base-dir>/logs/`) |
| `st123.utils.compatibility` | Cross-package compatibility adapters |

### Scripts

| Script | Role |
| --- | --- |
| `st123/scripts/download.py` | Download JWST products from MAST |
| `st123/scripts/align.py` | Unified alignment CLI (`--mode visit|reference|pair`) |
| `st123/scripts/mosaic.py` | Mosaic / coadd (writes `dolphot_frames.txt` for prep) |
| `st123/scripts/dolphot_prep.py` | DOLPHOT prep (`--from-mosaic` or explicit files) |
| `st123/scripts/link_raw.py` | Symlink FITS into a reduction `raw/` directory |
| `st123/scripts/image_overlap.py` | Maximum-overlap reference selection |
| `st123/scripts/region.py` | Illuminated `S_REGION` CLI |
| `st123/scripts/catalog.py` | Combined photometry catalog CLI |

All CLI entry points with a `__main__` block live under `st123/scripts/`
(package root only) and are registered in `pyproject.toml`
`[project.scripts]`. Shared CLI helpers (e.g. `scripts/utils/options.py`)
are not entry points. See `tests/test_entry_point_convention.py` and the
local (gitignored) `.cursor/rules/`.

### Notebooks

| Notebook | Role |
| --- | --- |
| `st123/notebooks/download.ipynb` | Interactive MAST queries (HST or JWST) |
| `st123/notebooks/align.ipynb` | Relative / Gaia JHAT alignment |
| `st123/notebooks/mosaic.ipynb` | Level-3 mosaics and PSF-matched coadds |

## Custom JHAT (`extdeps/jhat`)

This repository vendors a **custom JHAT build** under [`extdeps/jhat`](extdeps/jhat).
It is based on upstream
[arminrest/jhat](https://github.com/arminrest/jhat) but is **not** the
unmodified PyPI package (`jhat` on PyPI).

Use this tree for all JHAT-backed alignment code in st123, including:

- `align_jwst_image` (`jhat_params`, soft-fail behavior)
- `run_alignment` (master catalogs, iterative refine, F560W/F770W knobs)
- `align_from_frames` (REFERENCE → MIRI_REL pipeline)

all in `st123.alignment.align`.

`pip install -e .` (or `pip install -e ".[dev]"`) installs this tree
automatically via the `jhat @ file:./extdeps/jhat` entry in
`requirements.txt`. Do **not** `pip install jhat` from PyPI for this project
unless you intentionally want upstream instead of the custom build. Details and
version marking (`0.3.7+st123`) are in
[`extdeps/jhat/README.md`](extdeps/jhat/README.md).

## Requirements

- **Python 3.12** (3.11 also supported)
- External **DOLPHOT** binaries if you run PSF photometry
  ([DOLPHOT](http://americano.dolphinsim.com/dolphot/)).
  Put the DOLPHOT `bin/` directory on your `PATH` (st123 resolves it via
  `which dolphot`), or pass `--dolphot-bin`. There is no machine-specific
  default install path.
- The custom JHAT package under `extdeps/jhat` (pulled in by `pip install -e .`)

## Installation

The same conda + pip flow works on macOS and Linux/Ubuntu. Pinned dependencies
are declared in `requirements.txt` (via `pyproject.toml`), including the local
custom JHAT path dependency.

### macOS and Linux / Ubuntu

```bash
conda create -n st123 python=3.12 pip
conda activate st123
```

`drizzlepac` / `tables` need HDF5. Install the libraries with conda (recommended
on both platforms) before the editable install:

```bash
conda install -c conda-forge hdf5 blosc pytables -y
```

Then, from the repository root:

```bash
pip install -e .
# or, with test/notebook extras:
pip install -e ".[dev]"
```

If `tables` still cannot find HDF5 on macOS Homebrew:

```bash
brew install hdf5 c-blosc
export HDF5_DIR="$(brew --prefix hdf5)"
export BLOSC_DIR="$(brew --prefix c-blosc)"
pip install -e .
```

On Ubuntu, if you prefer system packages instead of conda HDF5:

```bash
sudo apt-get install -y libhdf5-dev libblosc-dev
pip install -e .
```

### Verify

```bash
python -c "import jhat, st123; print(jhat.__version__, jhat.__file__); print(st123.__version__)"
# jhat.__version__ should be 0.3.7+st123
# st123.__version__ comes from git tags via setuptools-scm (see Versioning)
download --help
align --help
```

This install path was validated with Python 3.12
(`conda create -n st123 python=3.12 pip` then `pip install -e .`). The same
steps apply on macOS and Linux/Ubuntu.

## Versioning

Package version is **dynamic** and derived from git tags with
[setuptools-scm](https://setuptools-scm.readthedocs.io/):

| State | Example `st123.__version__` |
| --- | --- |
| On tag `v0.1.0` | `0.1.0` |
| N commits after `v0.1.0` | `0.1.1.devN+g<hash>` |
| No git metadata (fallback) | `0.1.0` |

Release a version by tagging (annotated tags preferred):

```bash
git tag -a v0.2.0 -m "st123 v0.2.0"
git push origin v0.2.0
```

Then reinstall (`pip install -e .`) so the generated `st123/_version.py` and
installed metadata pick up the new tag. Do not set `version` manually in
`pyproject.toml`.

## Quick start

### Download MAST data

```bash
download \
  --ra "10:38:47.961" --dec "+53:30:34.10" \
  --base-dir /data/ckilpatrick/JWST/NGC3310
```

The object name is the basename of `--base-dir` (here `NGC3310`). MIRI-only
download into the canonical
`<telescope>/<instrument>/<filter>/<obsid>/mastDownload/...` layout used by
`align --mode reference` (e.g. `JWST/MIRI/F560W/<obsid>/...`). Missing output
directories are created automatically:

```bash
download \
  --ra 159.694014 --dec 53.502851 \
  --base-dir /data/ckilpatrick/JWST/NGC3310 \
  --radius 3 --stage 2 --instruments MIRI
```

### MIRI ↔ NIRCam alignment pipeline

With MIRI cals under
`--base-dir/JWST/MIRI/<FILTER>/<obsid>/mastDownload/...` and NIRCam
coadds under `--base-dir/reference/`:

```bash
align --base-dir /data/ckilpatrick/JWST/NGC3310 \
  --mode reference --instrument MIRI --plot --ncores 8
```

Per-frame failures are recorded as `FAILURE` rows in the alignment summary
and do not abort the run.

For proprietary data, pass a MAST API token
([create one here](https://auth.mast.stsci.edu/info)), the same way hst123
used `--token` with `Observations.login`:

```bash
download \
  --ra 189.9976 --dec -11.623 \
  --base-dir /path/to/NGC4536 \
  --token YOUR_MAST_TOKEN
```

Or export the token and omit `--token`:

```bash
export MAST_API_TOKEN=YOUR_MAST_TOKEN
download \
  --ra 189.9976 --dec -11.623 \
  --base-dir /path/to/NGC4536
```

Useful options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--base-dir` | (required) | Dataset root; object name = directory basename |
| `--radius` | `3.0` | Search radius in arcminutes |
| `--stage` | `2` | `2` = CAL, `3` = I2D |
| `--instruments` | `NIRCAM MIRI` | Instrument name filters |
| `--token` | `MAST_API_TOKEN` / `MAST_TOKEN` | MAST API token for proprietary data |

After install, console scripts from `pyproject.toml` are available (`download`, `align`, `mosaic`, `link-raw`, `image-overlap`, `region`, `catalog`, `dolphot-prep`, `dolphot-warmstart`).

Each CLI writes a UTC log under `<base-dir>/logs/{script}_{YYYYMMDD_HHMMSS}_{uid}.log` (or `./logs/` when `--base-dir` is omitted). Package code uses stdlib `logging.getLogger(__name__)`; scripts call `setup_script_logging` once in `main()`. External tools (JHAT, JWST Image3, DOLPHOT binaries, MAST clients) have their stdout/stderr captured into that log file at DEBUG via `capture_output` / `run_logged_subprocess` (console stays INFO unless `--verbose`).

### Stage files for a reduction

```bash
link-raw --base-dir /path/to/NGC4536 --instrument NIRCAM
```

### Pair alignment (one image → reference)

```bash
align \
  --ref /path/to/coadd_i2d.fits \
  --image /path/to/cal_or_i2d.fits \
  --base-dir /path/to/alignment_output
# optional: --photfile existing.phot.txt
```

### Visit-level alignment pipeline

Self-align NIRCam (or another visit-mode instrument) under `reduction/`:

```bash
align --base-dir /path/to/NGC4536 --ncores 8
# equivalent:
align --base-dir /path/to/NGC4536 --mode visit --instrument NIRCAM --ncores 8
```

### Mosaics / DOLPHOT prep

```bash
mosaic --base-dir /path/to/NGC4536 --ncores 8
dolphot-prep --from-mosaic --base-dir /path/to/NGC4536 --instrument nircam
```

`mosaic` builds a shared `FITSImagingWCSTransform` output WCS
(`create_gwcs` → `mosaic_gwcs.asdf`), drizzles each filter, writes the
coadd, and leaves a `dolphot_frames.txt` manifest per box. DOLPHOT staging
(`dolphot.param`, `nircammask` / `mirimask`, `calcsky`) is a separate
`dolphot-prep` step.

## Notes

- This repository targets **space telescopes in general** (HST, JWST, Roman).
  Some CLIs and modules are still JWST-oriented (e.g. `align --mode reference`)
  while shared MAST / alignment / mosaic / DOLPHOT paths (`download`, etc.)
  are meant to grow across facilities.
- CRDS reference files are required for `jwst` pipeline steps; set `CRDS_PATH` /
  `CRDS_SERVER_URL` as recommended by STScI.
- JHAT alignment parameters live in `st123/utils/settings.py` (`strict_*` /
  `relaxed_*` Gaia and JWST sets).
- HST MAST helpers remain in `st123.mast` for reference-image queries; the
  legacy standalone `hst123` reduction pipeline is not vendored here.

### Docstrings and naming

Library APIs use **NumPy-style** docstrings: `Parameters` / `Returns` (and
`Raises` when needed) with `----------` underlines, explicit types on each
parameter/return line, and matching type hints on signatures.

Canonical names:

| Concept | Prefer | Notes |
| --- | --- | --- |
| Project / dataset root | `base_dir` / `--base-dir` | Legacy `--workdir`, `--outdir`, … alias to this |
| Parallelism | `ncores` / `--ncores` | Alias `--workers`; DOLPHOT `MaxThreads` |
| Science FITS path(s) | `--image` | Avoid inventing `--miri` / `--align` for that role |
| Photometry catalog path | `photfile` / `--photfile` | JHAT APIs keep `photfilename` (upstream name) |
| Reduction staging dir | `proc_dir` | Used by symlink helpers |
| Output directory | `outdir` | Local job/output trees (not the CLI project root) |
