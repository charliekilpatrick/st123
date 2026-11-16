"""Layout constants for download / alignment output directories.

HST / JWST / Euclid / Roman filter lists, DOLPHOT / JHAT / drizzle knobs,
and MAST instrument defaults live on :mod:`st123.datamodels` classes.
"""

from __future__ import annotations

# Relative subdirectory pattern under the download root.
DEFAULT_DOWNLOAD_LAYOUT: str = 'telescope/instrument/filter/obsid'
# MAST products land under ``<base-dir>/<DOWNLOAD_DIR_NAME>/...``.
DOWNLOAD_DIR_NAME: str = 'download'

# Default output directory name for ``align --mode pair`` when ``--base-dir``
# is omitted.
DEFAULT_PAIR_OUTDIR: str = 'alignment_output'
