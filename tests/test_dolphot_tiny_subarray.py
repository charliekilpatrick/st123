"""Drop tiny WFC3/IR subarrays before DOLPHOT prep."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from st123.stages.photometry.dolphot import (
    _DOLPHOT_MIN_SCI_AXIS,
    _drop_tiny_hst_subarrays,
)


def _write_sci(path: Path, shape: tuple[int, int]) -> Path:
    data = np.ones(shape, dtype=np.float32)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=data, name='SCI'),
        ]
    ).writeto(path, overwrite=True)
    return path


def test_drop_tiny_hst_subarrays(tmp_path: Path):
    tiny = _write_sci(tmp_path / 'tiny_flt.fits', (128, 128))
    full = _write_sci(tmp_path / 'full_flt.fits', (1014, 1014))
    kept = _drop_tiny_hst_subarrays([tiny, full])
    assert kept == [full]
    assert min((128, 128)) < _DOLPHOT_MIN_SCI_AXIS
