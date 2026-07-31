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
