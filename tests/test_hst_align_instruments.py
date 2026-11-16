"""HST visit align instrument filtering (ACS/WFC3 vs WFPC2)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from st123.stages.alignment.hst_jhat import align_hst_raw_dir


def _write_hst_raw(path: Path, instrument: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = instrument
    primary.header['TELESCOP'] = 'HST'
    if instrument.upper() == 'WFPC2':
        primary.header['FILTNAM1'] = 'F814W'
    else:
        primary.header['FILTER'] = 'F814W'
    sci = fits.ImageHDU(np.ones((8, 8), dtype=np.float32), name='SCI')
    fits.HDUList([primary, sci]).writeto(path, overwrite=True)
    return path


def test_align_hst_raw_dir_instruments_skips_wfpc2(tmp_path: Path):
    raw = tmp_path / 'raw'
    jhat = tmp_path / 'jhat'
    _write_hst_raw(raw / 'j9acs_flc.fits', 'ACS')
    _write_hst_raw(raw / 'iewfc3_flc.fits', 'WFC3')
    _write_hst_raw(raw / 'u2wfpc2_c0m.fits', 'WFPC2')

    aligned: list[str] = []

    def _fake_align(frame, outdir, **kwargs):
        aligned.append(Path(frame).name)
        out = Path(outdir) / f'{Path(frame).stem}_jhat.fits'
        out.write_bytes(b'x')
        return out

    with (
        patch(
            'st123.stages.alignment.hst_jhat.align_hst_image', side_effect=_fake_align
        ),
        patch(
            'st123.stages.alignment.hst_jhat.harmonize_hst_jhat_dir', return_value=[]
        ),
    ):
        results = align_hst_raw_dir(
            raw,
            jhat,
            instruments=['ACS', 'WFC3'],
            soft_fail=True,
            gaia=True,
        )

    names = {Path(r['path']).name for r in results}
    assert names == {'j9acs_flc.fits', 'iewfc3_flc.fits'}
    assert set(aligned) == {'j9acs_flc.fits', 'iewfc3_flc.fits'}
    assert all(r.get('status') == 'ok' for r in results)
