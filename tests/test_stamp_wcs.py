"""Shared stamp WCS persistence and stable ref_* identity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from st123.mosaic.mosaic import (
    STAMP_WCS_BASENAME,
    assign_stable_box_ids,
    ensure_box_stamp_wcs,
    load_stamp_wcs,
    local_bbox_for_wcs,
    stamp_sky_center,
    write_stamp_wcs,
    MosaicBox,
)


def _tan_wcs(
    *,
    crval=(185.74, 15.83),
    crpix=(50.0, 40.0),
    cdelt=(-0.001, 0.001),
    naxis=(100, 80),
) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = list(crval)
    w.wcs.crpix = list(crpix)
    w.wcs.cdelt = list(cdelt)
    w.wcs.cunit = ['deg', 'deg']
    w.pixel_shape = (int(naxis[0]), int(naxis[1]))
    w._naxis = [int(naxis[0]), int(naxis[1])]
    return w


def test_write_load_stamp_wcs_roundtrip(tmp_path: Path):
    w = _tan_wcs()
    path = write_stamp_wcs(tmp_path, w)
    assert path.name == STAMP_WCS_BASENAME
    loaded = load_stamp_wcs(tmp_path)
    assert loaded is not None
    assert loaded.pixel_shape == (100, 80)
    ra0, dec0 = stamp_sky_center(w)
    ra1, dec1 = stamp_sky_center(loaded)
    assert abs(ra0 - ra1) < 1e-8
    assert abs(dec0 - dec1) < 1e-8


def test_load_stamp_wcs_falls_back_to_i2d(tmp_path: Path):
    w = _tan_wcs(naxis=(32, 24))
    data = np.ones((24, 32), dtype=np.float32)
    hdr = w.to_header(relax=True)
    hdu = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=data, header=hdr, name='SCI'),
        ]
    )
    hdu.writeto(tmp_path / 'coadd_0_0_f200w_i2d.fits')
    loaded = load_stamp_wcs(tmp_path)
    assert loaded is not None
    assert loaded.pixel_shape == (32, 24)


def test_assign_stable_box_ids_reuses_existing_stamp(tmp_path: Path):
    group = tmp_path / 'group_0'
    old = group / 'ref_5'
    old.mkdir(parents=True)
    write_stamp_wcs(old, _tan_wcs(crval=(185.74, 15.83)))

    # New planner order would have been ref_0, but sky matches ref_5.
    ids = assign_stable_box_ids(
        group,
        [_tan_wcs(crval=(185.7401, 15.8301), naxis=(90, 70))],
    )
    assert ids == [5]


def test_assign_stable_box_ids_avoids_colliding_with_unmatched(tmp_path: Path):
    group = tmp_path / 'group_0'
    (group / 'ref_0').mkdir(parents=True)
    write_stamp_wcs(group / 'ref_0', _tan_wcs(crval=(180.0, 10.0)))

    # Far from existing stamp → new id, not overwriting ref_0.
    ids = assign_stable_box_ids(
        group,
        [_tan_wcs(crval=(185.74, 15.83))],
    )
    assert ids == [1]


def test_ensure_box_stamp_wcs_persists_and_localizes(tmp_path: Path):
    out = tmp_path / 'ref_0'
    out.mkdir()
    box = MosaicBox(
        group_id=0,
        box_id=0,
        outdir=out,
        bbox=None,
        frames=[],
        wcs=_tan_wcs(),
    )
    stamp = ensure_box_stamp_wcs(box)
    assert (out / STAMP_WCS_BASENAME).is_file()
    assert box.wcs is stamp
    assert local_bbox_for_wcs(stamp).bounds == (0.0, 0.0, 100.0, 80.0)


def test_build_centered_stamp_wcs_puts_target_at_center():
    from st123.mosaic.mosaic import build_centered_stamp_wcs

    ra, dec = 185.733958, 15.826119
    stamp = build_centered_stamp_wcs(
        ra,
        dec,
        size_x_arcsec=68.0,
        size_y_arcsec=51.0,
        pixel_scale_arcsec=0.031,
        orientat_deg=21.33,
    )
    nx, ny = stamp.pixel_shape
    cra, cdec = stamp.pixel_to_world_values(0.5 * (nx - 1), 0.5 * (ny - 1))
    assert abs(float(cra) - ra) < 1e-8
    assert abs(float(cdec) - dec) < 1e-8
