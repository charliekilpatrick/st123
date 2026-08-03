# M51 / M82 warm-start architecture

Dataset-specific driver for NIRCam→MIRI DOLPHOT warm-start under
`/data/rwisenbaker/jwst_data/{M51,M82}`.

## Layout expectations

| Galaxy | Project root | NIRCam DOLPHOT | Phot stem |
|--------|--------------|----------------|-----------|
| M51 | `.../M51` | `dolphot/nircam_{g}_{b}/` | `ngc5194` |
| M82 | `.../M82` | `dolphot/nircam_{g}_{b}/` | `ngc3034` |

M82 NIRCam products were downloaded as `ngc3034/`; that directory is renamed
to `dolphot/` so both galaxies share the same path convention. Mosaic box
IDs for MIRI→reference matching come from `coadd_{g}_{b}_*` in the alignment
summary (required for M82, where many boxes live under `group_0/ref_0`).

## Eligibility rule

A reference is staged / run **only if** it has at least
**`MIN_MIRI_IMAGES = 10`** unique usable MIRI `*_jhat.fits` frames
(SUCCESS, dispersion &lt; 200 mas) **and** a finished NIRCam DOLPHOT catalog
at `dolphot/nircam_{g}_{b}`.

Sparse MIRI footprints (`n_miri < 10`) are ignored.

## Image-count limits

The DOLPHOT binary is compiled with `MAXNIMG=501` (up to **500** science
frames). Staging still caps each invocation at **`DOLPHOT_MAX_NIMG = 400`**.
If a warm-start would exceed that, setup writes `dolphot_partXX.param` plus
`dolphot_split.json`; `launch_dolphot.py` runs each part and merges catalogs
into the usual `*_nircam_miri.phot`.

## Commands

```bash
conda activate st123
cd /data/ckilpatrick/st123/examples/m51_m82_warmstart

# List KEEP / SKIP for both galaxies
python discover.py

# Stage eligible warm-start dirs (does not run dolphot)
python setup_warmstarts.py -v

# Print dolphot launch lines (16 threads); add --go to start ≤3 at a time
python launch_dolphot.py
```

Legacy wrappers in `examples/outdir_m51_m82_warmstart/*.sh` call these tools.

## Recovering MIRI warm-start photometry (M51 / M82)

After the DOLPHOT rebuild and this split-aware staging code:

1. **Re-stage incomplete refs** (skips boxes that already have a non-empty
   `*_nircam_miri.phot`):
   ```bash
   python setup_warmstarts.py -v
   ```
   For a broken partial dir (e.g. M82 `nircam_miri_0_5` missing param/sky),
   remove that directory first, then re-run setup.

2. **Dry-run the launcher** (sanity-check commands; do not start jobs yet):
   ```bash
   python launch_dolphot.py
   ```

3. **Launch** when ready:
   ```bash
   python launch_dolphot.py --go
   ```

Finished M51 boxes (`1_4`, `1_6`, `1_8`, `1_10`, `1_11`) are left alone.
Previously failed M51 `1_0` / `1_9` (`Nimg` ~208–211) now fit in a single
run under the new binary. All eligible M82 refs need staging + launch.
