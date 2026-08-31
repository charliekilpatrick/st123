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

- **Python:** **3.11-3.12** (`requires-python` in `pyproject.toml`: `>=3.11,<3.13`); CI targets **3.12**
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
`run-dolphot`, `dolphot-hdf5`, `dolphot-warmstart-prep`, `dolphot-warmstart`,
`coadd-phot`.

---



## Usage

```bash
download --ra <ra> --dec <dec> --base-dir /path/to/<object>
# Example:
download \
  --ra "10:38:47.961" --dec "+53:30:34.10" \
  --base-dir /path/to/NGC3310
```

- `--base-dir` - Dataset / project root. The object name is the directory
basename (e.g. `NGC3310`). Prefer an absolute path. MAST products are
organized as
`<base-dir>/download/<telescope>/<instrument>/<filter>/<obsid>/<filename>`
(e.g. `.../download/HST/ACS/F814W/102617486/jey335ehq_flc.fits`). Visit
alignment / mosaics use `<base-dir>/reduction/`.
- `--token` - MAST API token for proprietary data
([create one here](https://auth.mast.stsci.edu/info)). Also read from
`MAST_API_TOKEN` / `MAST_TOKEN`.
- **Alignment** - `align --mode visit|reference|pair` (JHAT-backed).
- **Mosaics / DOLPHOT** - `mosaic` writes coadds and `dolphot_frames.txt`;
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

# MIRI -> NIRCam reference coadds
align --base-dir /path/to/NGC3310 \
  --mode reference --instrument MIRI --plot --ncores 8

# Single frame -> reference
align \
  --ref /path/to/coadd_i2d.fits \
  --image /path/to/cal_or_i2d.fits \
  --base-dir /path/to/alignment_output
```



### Stage files, mosaics, DOLPHOT prep

```bash
link-raw --base-dir /path/to/NGC4536 --instrument NIRCAM
mosaic --base-dir /path/to/NGC4536 --ncores 8
dolphot-prep --from-mosaic --base-dir /path/to/NGC4536 --instruments nircam
```

`mosaic` builds a shared sky stamp per `reference/group_*/ref_*` box
(`stamp_wcs.fits`), mosaics every JWST filter onto that footprint (with
pre-mosaic NIRCam harmonize + coadd unify), and AstroDrizzles HST onto the
same stamp WCS - **one combined coadd per instrument+filter** (all visits
stacked; optional `--visit-coadds` / `--drop-outlier-visits`). Custom stamps
(`--center-ra/--center-dec`, optional `--stamp-size` / `--stamp-ref` /
`--stamp-id`) and remosaics (`--existing-box`) always drop frames outside the
stamp FoV (and frames that miss the stamp center). For auto-split boxes, pass
`--require-coverage` (or `--require-coverage-ra/dec`) to apply the same rule.
Box ids are matched to existing stamps by sky center so replans do not
renumber regions. Each box gets a `dolphot_frames.txt` manifest. DOLPHOT
staging (`dolphot.param`, `nircammask` / `mirimask`, `calcsky`) is a separate
`dolphot-prep` step.

Interactive notebooks: `st123/notebooks/download.ipynb`, `align.ipynb`,
`mosaic.ipynb`.

### HST one-target end-to-end (mixed DOLPHOT)

One project directory per target. Flow: **download -> link-raw -> align (JHAT) ->
mosaic (drizzle) -> dolphot-prep -> dolphot -> HDF5 catalog (``.h5``)**.

After mosaic, `dolphot-prep --instruments hst` stages **all** ACS/WFC3/WFPC2
JHAT frames against the best coadd reference (prefer **WFC3 -> ACS -> WFPC2**,
then `HSTDataModel.BEST_REFERENCE_FILTERS` with **F625W** first). That ranking matches the
bands that usually give the cleanest HST PSF photometry (e.g. WFC3/F625W and
WFPC2/F606W). Shallower or awkward coverage (e.g. WFPC2 F450W / F814W on sparse
PC or short stacks) often yields only ~1-1.5sigma **forced** photometry at an
F625W/F606W position - treat those as limits, not independent detections.

Globals written into `dolphot.param` (`HSTDataModel.DOLPHOT_BASE_PARAMS`) are intentional
departures from NIRCam defaults for JHAT-aligned multi-instrument HST:

| Knob | Value | Why |
|------|-------|-----|
| `UseWCS` | `2` | JHAT L2 is TAN-SIP; drizzle L3 is TAN. `UseWCS=1` seeds tens of px off. |
| `Align` | `0` | Trust the JHAT SIP seed. `Align=1` can refine ~0.2 px but has hit DOLPHOT NaN asserts on mixed full phot. |
| `Rotate` | `0` | Do not fight the WCS seed. |
| `Force1` | `1` | Point-source shape (SN / compact counterparts). |
| `FlagMask` | `7` | HST DQ bits. |
| CTE | ACS/WFC3 off, WFPC2 on | Matches typical pipeline products. |

```bash
# Make sure you have up to date dolphot in your path
conda activate st123
BASE=/path/to/YourTarget
NCORES=8
RA=177.66
DEC=55.36
# optional: export MAST_API_TOKEN=...

# download -> align -> mosaic -> DOLPHOT
# --instruments hst == ACS WFC3 WFPC2 (same flag on every stage)
download     --base-dir "$BASE" --instruments hst --ra="$RA" --dec="$DEC" -v
align        --base-dir "$BASE" --instruments hst --ncores "$NCORES" -v
mosaic       --base-dir "$BASE" --instruments hst --ncores "$NCORES" -v
dolphot-prep --base-dir "$BASE" --instruments hst --ncores "$NCORES" -v
run-dolphot  --base-dir "$BASE" --instruments hst --ncores "$NCORES" -v
# Optional: (re)build compressed <run>.h5 catalogs if missing
dolphot-hdf5 --base-dir "$BASE" -v
```

**QA checklist** (before trusting photometry):

1. In `dolphot.param`: `UseWCS = 2`, `Align = 0`, `Force1 = 1`.
2. Reference is the expected deep coadd under the shared mosaic box layout
   (typically `reduction/reference/group_*/ref_*/coadd_*_wfc3_f625w_drc.fits`).
3. Inspect `*.phot.info` / per-image quality flags; chips with `flag=9/10` are off-detector or bad.
4. For a target RA/Dec, compare per-band SNR: secure detections (F625W/F606W) vs forced limits.
5. Confirm each finished photometry directory has a compressed ``<run>.h5`` sidecar
   (written by ``run-dolphot`` wait mode by default, or via ``dolphot-hdf5``).

**Scrape nearest source** after `dolphot` finishes:

```bash
python - <<'PY'
from st123.stages.photometry.dolphot import nearest_phot_source
rows = nearest_phot_source(
    '/path/to/hst_0_5/hst_0_5.phot',
    '/path/to/hst_0_5/coadd_0_5_wfc3_f625w_drc.fits',
    ra=177.65590, dec=55.35357, n=3,
)
for r in rows:
    print(r)
PY
```

Layout: `$PROJ/download/HST/.../<obsid>/<filename>`,
`$PROJ/reduction/{raw,jhat_hst,reference/group_*/ref_*}/`, `$PROJ/dolphot/hst_G_B/`.
HST and JWST mosaics share the same `group_*/ref_*` boxes; each box may hold
both `coadd_*_i2d.fits` (JWST) and `coadd_*_{acs|wfc3}_*_{drc|drz}.fits` (HST).

### HST warmstart from NIRCam

When the same field has a finished NIRCam DOLPHOT run, seed HST photometry from
that catalog (`xytfile`) with the **NIRCam coadd as `img0`** and the same
`HSTDataModel.DOLPHOT_BASE_PARAMS` as the free HST path. Align HST with JHAT to Gaia (or the
NIRCam WCS) first so chip WCS matches the NIRCam reference pixel grid.

```bash
# NIRCam mosaic + dolphot already done (e.g. $PROJ/reduction/phot_0_0)
# HST download -> align -> mosaic so reduction/jhat_hst + reference/group_*/ref_* exist

dolphot-warmstart \
  --instruments HST \
  --base-dir "$PROJ" \
  --ncores "$NCORES"
# optional: --ref-dir ... --hst-jhat path1_jhat.fits ...
# optional seed cuts: --prune-xyt-for-hst (type=1, SNR>=5, ...)

cd "$PROJ/dolphot/nircam_hst_0_0" && \
  dolphot nircam_hst_0_0.phot -pdolphot.param MaxThreads="$NCORES"
# (use the exact phot name printed by dolphot-warmstart)
```

`--instruments MIRI` (default target) remains NIRCam->MIRI warmstart into
`dolphot/nircam_miri_{group}_{box}` (group/box inherited from `--ref-dir`).

### Mega catalog (NIRCam + MIRI + HST)

After free NIRCam DOLPHOT and MIRI/HST warmstarts finish, merge into one
coherent catalog. Free NIRCam is the master star list; warmstart photometry is
matched by reference XY (`99.999` where a star was pruned/missing):

```bash
dolphot-hdf5 --base-dir "$PROJ" --merge \
  --outfile "$PROJ/dolphot/megacatalog_0_sn/megacatalog_0_sn.h5" \
  --group 0 --box sn -v

# Or pass catalogs explicitly (primary = first):
dolphot-hdf5 --base-dir "$PROJ" --merge \
  --phot "$PROJ/reduction/phot_0_sn/phot_0_sn.phot" \
         "$PROJ/dolphot/nircam_miri_0_sn/nircam_miri_0_sn.phot" \
         "$PROJ/dolphot/nircam_hst_0_sn/nircam_hst_0_sn.phot" \
  --outfile "$PROJ/dolphot/megacatalog_0_sn/megacatalog_0_sn.h5" -v
```

Without `--merge`, `dolphot-hdf5` still writes one `.h5` sidecar per finished run.

---



## Supported instruments


| Facility | Instruments / products (typical)                                                                                                 |
| -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| HST      | WFPC2 (`c0m`/`c1m`), ACS/WFC (`flc`), WFC3/UVIS (`flc`), WFC3/IR (`flt`) - MAST helpers download `project=HST` pipeline products once |
| JWST     | NIRCam, MIRI (download, JHAT align, mosaic, DOLPHOT prep / warm-start)                                                           |
| Roman    | WFI (`_cal.asdf` L2; imaging filters F062-F213) -- datamodel identity / catalogs; stages not wired yet |
| Euclid   | VIS (IE band) and NISP/NIR photometer (YE/JE/HE) -- datamodel identity / catalogs; stages not wired yet |


Filter names accepted for reference selection and downloads live on the
instrument datamodel classes (`NIRCamDataModel.FILTERS`, `ACSDataModel.FILTERS`,
`EuclidVISDataModel.FILTERS`, `RomanWFIDataModel.FILTERS`, ...).

Alignment uses the **custom JHAT** tree under `extdeps/jhat` (not unmodified
PyPI `jhat`). CRDS reference files are required for `jwst` pipeline steps;
set `CRDS_PATH` / `CRDS_SERVER_URL` as recommended by STScI.

---



## Options (summary)

- **Paths / runtime:** `--base-dir` (aliases `--workdir`, `--basedir`, ...),
`--ncores` / `--workers`, `--plot`, `--verbose`, `--dry-run`, `--version`
- **Shared pipeline flags:** `--instruments` (alias `--instrument`; mission
  aliases `hst`->ACS WFC3 WFPC2, `jwst`->NIRCAM MIRI, `all`), plus optional
  `--ra` / `--dec` / `--radius` on download/align/mosaic/dolphot-prep/run-dolphot
- **Download:** `--ra`, `--dec`, `--radius`, `--stage`, `--instruments`,
`--filters`, `--token`
- **Alignment:** `--mode visit|reference|pair`, `--instruments`, `--ref`,
`--image`, `--photfile`
- **Mosaic / DOLPHOT:** `--from-mosaic`, `--files`, `--refimage`,
`--dolphot-bin`, `--instruments`, `--skip-sky` / related prep flags
- **Link:** `--instrument` (with `--base-dir`) for `link-raw`

Canonical naming conventions used across CLIs and library APIs:


| Concept                 | Prefer                    | Notes                                                    |
| ----------------------- | ------------------------- | -------------------------------------------------------- |
| Project / dataset root  | `base_dir` / `--base-dir` | Legacy `--workdir`, `--outdir`, ... alias here             |
| Parallelism             | `ncores` / `--ncores`     | Alias `--workers`; mosaic JWST = concurrent filter Image3; HST AstroDrizzle; dolphot-prep pool; DOLPHOT `MaxThreads` |
| Science FITS path(s)    | `--image`                 | Prefer over inventing `--miri` / `--align` for that role |
| Photometry catalog path | `photfile` / `--photfile` | JHAT APIs keep upstream `photfilename`                   |


---



## Documentation

- **Docs site:** [charliekilpatrick.github.io/st123](https://charliekilpatrick.github.io/st123/)
(Sphinx + Read the Docs theme; published from `main` via
`[.github/workflows/documentation.yml](.github/workflows/documentation.yml)`)
- **This README** - install, usage, instruments, and citation
- **Module docstrings** - NumPy-style `Parameters` / `Returns` with explicit
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

- Kilpatrick, C. D., Suresh, A., et al., "The Type II SN 2025pht in NGC 1637: A
Red Supergiant with Carbon-rich Circumstellar Dust as the First JWST
Detection of a Supernova Progenitor Star," *ApJL* **992**, L10 (2025).
doi:[10.3847/2041-8213/ae04de](https://doi.org/10.3847/2041-8213/ae04de) *
[arXiv:2508.10994](https://arxiv.org/abs/2508.10994)
- Blanchard, P. K., Berger, E., Andrew, S. E., Suresh, A., Uno, K.,
Kilpatrick, C. D., et al., "James Webb Space Telescope Observations of the
Nearby and Precisely Localized FRB 20250316A: A Potential Near-IR Counterpart
and Implications for the Progenitors of Fast Radio Bursts," *ApJL* **989**,
L49 (2025).
doi:[10.3847/2041-8213/adf29f](https://doi.org/10.3847/2041-8213/adf29f) *
[arXiv:2506.19007](https://arxiv.org/abs/2506.19007)

Related HST-pipeline lineage (hst123): see
[charliekilpatrick/hst123](https://github.com/charliekilpatrick/hst123)
citing section for additional suggested references.

**Contact:**

- Charlie Kilpatrick - [ckilpatrick@northwestern.edu](mailto:ckilpatrick@northwestern.edu)
- Aswin Suresh - [aswin.suresh@northwestern.edu](mailto:aswin.suresh@northwestern.edu)

**Bugs and feature requests:** please open an issue on
[charliekilpatrick/st123](https://github.com/charliekilpatrick/st123/issues).