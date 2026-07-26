# st123

[Build and Test](https://github.com/charliekilpatrick/st123/actions/workflows/ci.yml)
[Documentation](https://github.com/charliekilpatrick/st123/actions/workflows/documentation.yml)
[Docs site](https://charliekilpatrick.github.io/st123/)

Toolkit for space-telescope imaging: download from MAST, align with a
**custom, repository-local build of [JHAT](https://jhat.readthedocs.io/)**,
build mosaics / coadds, prepare DOLPHOT runs, and scrape photometry catalogs.
Designed for **HST, JWST, and Roman** (and related MAST-hosted missions), with
current workflows strongest for JWST (and HST reference / MAST helpers).

**Issues:** Report bugs and feature requests via the
[project issue tracker](https://github.com/charliekilpatrick/st123/issues).
Other questions or publications using st123: see
[Citing and contact](#citing-and-contact).

---



## Repository status

- **Python:** **3.11–3.12** (`requires-python` in `pyproject.toml`: `>=3.11,<3.13`); CI targets **3.12**
- **Versioning:** From git tags (setuptools-scm); `python -c "import st123; print(st123.__version__)"`
- **Tests:** `pytest` in `tests/` (see `pyproject.toml`)
- **Provenance:** 
[charliekilpatrick/hst123](https://github.com/charliekilpatrick/hst123) and
[aswinsuresh24/jwst123](https://github.com/aswinsuresh24/jwst123)

---



## Installation

**1. Environment (recommended: Conda)**  
From the repo root:

```bash
conda create -n st123 python=3.12 pip
conda activate st123
```

`drizzlepac` / `tables` need HDF5. Install the libraries with conda before the
editable install:

```bash
conda install -c conda-forge hdf5 blosc pytables -y
```

**2. Package and heavy deps**

```bash
pip install -e .
# or, with test / notebook extras:
pip install -e ".[dev]"
```

Pinned dependencies live in `requirements.txt` (via `pyproject.toml`), including
the local custom JHAT path dependency
(`jhat @ file:./extdeps/jhat`). Do **not** `pip install jhat` from PyPI for
this project unless you intentionally want upstream instead of the custom
build. Details are in `[extdeps/jhat/README.md](extdeps/jhat/README.md)`.

If `tables` still cannot find HDF5 on macOS Homebrew:

```bash
brew install hdf5 c-blosc
export HDF5_DIR="$(brew --prefix hdf5)"
export BLOSC_DIR="$(brew --prefix c-blosc)"
pip install -e .
```

On Ubuntu, system packages instead of conda HDF5:

```bash
sudo apt-get install -y libhdf5-dev libblosc-dev
pip install -e .
```

Verify:

```bash
python -c "import jhat, st123; print(jhat.__version__, jhat.__file__); print(st123.__version__)"
# jhat.__version__ should be 0.3.7+st123
download --help
align --help
```

**3. DOLPHOT (only for photometry runs)**  
DOLPHOT is external; not bundled. Put the DOLPHOT `bin/` directory on your
`PATH` (st123 resolves it via `which dolphot`), or pass `--dolphot-bin`.
There is no machine-specific default install path. See
[DOLPHOT](http://americano.dolphinsim.com/dolphot/).

Console scripts from `pyproject.toml` after install: `download`, `align`,  
`mosaic`, `link-raw`, `image-overlap`, `region`, `catalog`, `dolphot-prep`,  
`dolphot-warmstart`.

---



## Usage

```bash
download --ra <ra> --dec <dec> --base-dir /path/to/<object>
# Example:
download \
  --ra "10:38:47.961" --dec "+53:30:34.10" \
  --base-dir /path/to/NGC3310
```

- `--base-dir` — Dataset / project root. The object name is the directory
basename (e.g. `NGC3310`). Prefer an absolute path. Products are organized
under layouts such as
`<base-dir>/JWST/<Instrument>/<FILTER>/<obsid>/...` for downloads and
`<base-dir>/reduction/` for visit alignment / mosaics.
- `--token` — MAST API token for proprietary data
([create one here](https://auth.mast.stsci.edu/info)). Also read from
`MAST_API_TOKEN` / `MAST_TOKEN`.
- **Alignment** — `align --mode visit|reference|pair` (JHAT-backed).
- **Mosaics / DOLPHOT** — `mosaic` writes coadds and `dolphot_frames.txt`;
`dolphot-prep` stages masks, calcsky, and `dolphot.param`.

Without `--download`-style flags, CLIs operate on files already under
`--base-dir`. Full option lists: `<command> --help`.

### Download

```bash
download \
  --ra 159.694014 --dec 53.502851 \
  --base-dir /path/to/NGC3310 \
  --radius 3 --stage 2 --instruments MIRI
```



### Alignment

```bash
# Visit-level (NIRCam under reduction/)
align --base-dir /path/to/NGC4536 --ncores 8

# MIRI → NIRCam reference coadds
align --base-dir /path/to/NGC3310 \
  --mode reference --instrument MIRI --plot --ncores 8

# Single frame → reference
align \
  --ref /path/to/coadd_i2d.fits \
  --image /path/to/cal_or_i2d.fits \
  --base-dir /path/to/alignment_output
```



### Stage files, mosaics, DOLPHOT prep

```bash
link-raw --base-dir /path/to/NGC4536 --instrument NIRCAM
mosaic --base-dir /path/to/NGC4536 --ncores 8
dolphot-prep --from-mosaic --base-dir /path/to/NGC4536 --instrument nircam
```

`mosaic` builds a shared `FITSImagingWCSTransform` output WCS, drizzles each
filter, writes the coadd, and leaves a `dolphot_frames.txt` manifest per box.
DOLPHOT staging (`dolphot.param`, `nircammask` / `mirimask`, `calcsky`) is a
separate `dolphot-prep` step.

Interactive notebooks: `st123/notebooks/download.ipynb`, `align.ipynb`,
`mosaic.ipynb`.

---



## Supported instruments


| Facility | Instruments / products (typical)                                                                                                 |
| -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| HST      | WFPC2 (`c0m`/`c1m`), ACS/WFC (`flc`), ACS/HRC (`flt`), WFC3/UVIS (`flc`), WFC3/IR (`flt`) — MAST helpers and reference selection |
| JWST     | NIRCam, MIRI (download, JHAT align, mosaic, DOLPHOT prep / warm-start)                                                           |
| Roman    | WFI filter catalog and path helpers (pipeline growth)                                                                            |
| Euclid   | VIS / NISP filter catalog helpers (pipeline growth)                                                                              |


Filter names accepted for reference selection and downloads are listed in
`st123/utils/settings.py` (`acceptable_filters`, `FILTERS_BY_INSTRUMENT`),
in the style of FITS `FILTER` / `FILTER1` / `FILTER2` header values.

Alignment uses the **custom JHAT** tree under `extdeps/jhat` (not unmodified
PyPI `jhat`). CRDS reference files are required for `jwst` pipeline steps;
set `CRDS_PATH` / `CRDS_SERVER_URL` as recommended by STScI.

---



## Options (summary)

- **Paths / runtime:** `--base-dir` (aliases `--workdir`, `--basedir`, …),
`--ncores` / `--workers`, `--plot`, `--verbose`, `--dry-run`, `--version`
- **Download:** `--ra`, `--dec`, `--radius`, `--stage`, `--instruments`,
`--filters`, `--token`
- **Alignment:** `--mode visit|reference|pair`, `--instrument`, `--ref`,
`--image`, `--photfile`
- **Mosaic / DOLPHOT:** `--from-mosaic`, `--files`, `--refimage`,
`--dolphot-bin`, `--instrument`, `--skip-sky` / related prep flags
- **Link:** `--instrument` (with `--base-dir`) for `link-raw`

Canonical naming conventions used across CLIs and library APIs:


| Concept                 | Prefer                    | Notes                                                    |
| ----------------------- | ------------------------- | -------------------------------------------------------- |
| Project / dataset root  | `base_dir` / `--base-dir` | Legacy `--workdir`, `--outdir`, … alias here             |
| Parallelism             | `ncores` / `--ncores`     | Alias `--workers`; DOLPHOT `MaxThreads`                  |
| Science FITS path(s)    | `--image`                 | Prefer over inventing `--miri` / `--align` for that role |
| Photometry catalog path | `photfile` / `--photfile` | JHAT APIs keep upstream `photfilename`                   |


---



## Documentation

- **Docs site:** [charliekilpatrick.github.io/st123](https://charliekilpatrick.github.io/st123/)
(Sphinx + Read the Docs theme; published from `main` via
`[.github/workflows/documentation.yml](.github/workflows/documentation.yml)`)
- **This README** — install, usage, instruments, and citation
- **Module docstrings** — NumPy-style `Parameters` / `Returns` with explicit
types on public library APIs
- **Lineage docs:** [hst123 documentation](https://charliekilpatrick.github.io/hst123/)
for the HST-focused predecessor pipeline
- **JHAT:** `[extdeps/jhat/README.md](extdeps/jhat/README.md)` and
[JHAT docs](https://jhat.readthedocs.io/)

Build locally:

```bash
pip install -e ".[docs]"
cd docs && make html
# open docs/build/html/index.html
```

---



## Citing and contact

**Citation:** C. D. Kilpatrick & A. Suresh, *st123: space-telescope download,
alignment, mosaic, and DOLPHOT helpers*, GitHub
([https://github.com/charliekilpatrick/st123](https://github.com/charliekilpatrick/st123)). If a DOI (e.g. Zenodo) is
assigned to a release, cite that. We welcome notice of papers that use st123.

**Suggested references:** If you cite this software in a paper or proposal, you
may also reference peer-reviewed works that use related JWST / HST reduction
paths and list both authors:

- Kilpatrick, C. D., Suresh, A., et al., “The Type II SN 2025pht in NGC 1637: A
Red Supergiant with Carbon-rich Circumstellar Dust as the First JWST
Detection of a Supernova Progenitor Star,” *ApJL* **992**, L10 (2025).
doi:[10.3847/2041-8213/ae04de](https://doi.org/10.3847/2041-8213/ae04de) ·
[arXiv:2508.10994](https://arxiv.org/abs/2508.10994)
- Blanchard, P. K., Berger, E., Andrew, S. E., Suresh, A., Uno, K.,
Kilpatrick, C. D., et al., “James Webb Space Telescope Observations of the
Nearby and Precisely Localized FRB 20250316A: A Potential Near-IR Counterpart
and Implications for the Progenitors of Fast Radio Bursts,” *ApJL* **989**,
L49 (2025).
doi:[10.3847/2041-8213/adf29f](https://doi.org/10.3847/2041-8213/adf29f) ·
[arXiv:2506.19007](https://arxiv.org/abs/2506.19007)

Related HST-pipeline lineage (hst123): see
[charliekilpatrick/hst123](https://github.com/charliekilpatrick/hst123)
citing section for additional suggested references.

**Contact:**

- Charlie Kilpatrick — [ckilpatrick@northwestern.edu](mailto:ckilpatrick@northwestern.edu)
- Aswin Suresh — [aswin.suresh@northwestern.edu](mailto:aswin.suresh@northwestern.edu)

**Bugs and feature requests:** please open an issue on
[charliekilpatrick/st123](https://github.com/charliekilpatrick/st123/issues).