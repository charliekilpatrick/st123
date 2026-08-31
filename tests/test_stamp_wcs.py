"""Shared stamp WCS persistence and stable ref_* identity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from st123.stages.mosaic.mosaic import (
    STAMP_WCS_BASENAME,
    _ensure_pc_cdelt_header,
    assign_stable_box_ids,
    create_gwcs,
    ensure_box_stamp_wcs,
    load_stamp_wcs,
    local_bbox_for_wcs,
    rescale_wcs_to_pixel_scale,
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
    import warnings

    from astropy.wcs import FITSFixedWarning

    w = _tan_wcs()
    path = write_stamp_wcs(tmp_path, w)
    assert path.name == STAMP_WCS_BASENAME
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        loaded = load_stamp_wcs(tmp_path)
    assert loaded is not None
    assert loaded.pixel_shape == (100, 80)
    axis_msgs = [
        str(w.message)
        for w in caught
        if issubclass(w.category, FITSFixedWarning)
        and 'more axes' in str(w.message).lower()
    ]
    assert not axis_msgs
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

    # Far from existing stamp -> new id, not overwriting ref_0.
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


def test_box_coadd_i2d_paths_lists_stamp_coadds(tmp_path: Path):
    from st123.stages.mosaic.mosaic import box_coadd_i2d_paths, resolve_existing_box_dir

    box = tmp_path / 'reduction' / 'reference' / 'group_0' / 'ref_sn'
    box.mkdir(parents=True)
    (box / 'coadd_0_sn_f150w_i2d.fits').write_bytes(b'x')
    (box / 'stamp_wcs.fits').write_bytes(b'y')
    resolved = resolve_existing_box_dir(tmp_path, 'group_0/ref_sn')
    assert resolved == box.resolve()
    paths = box_coadd_i2d_paths(resolved)
    assert [Path(p).name for p in paths] == ['coadd_0_sn_f150w_i2d.fits']


def test_build_centered_stamp_wcs_puts_target_at_center():
    from st123.stages.mosaic.mosaic import build_centered_stamp_wcs

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


def _north_up_pc_header_omitting_offdiag() -> fits.Header:
    """2025pht-like stamp: astropy omits zero PC1_2/PC2_1."""
    return fits.Header(
        {
            'CRPIX1': 968.5,
            'CRPIX2': 968.5,
            'CRVAL1': 70.3703,
            'CRVAL2': -2.865511,
            'PC1_1': -8.6111111111111e-06,
            'PC2_2': 8.6111111111111e-06,
            'CDELT1': 1.0,
            'CDELT2': 1.0,
            'CTYPE1': 'RA---TAN',
            'CTYPE2': 'DEC--TAN',
            'CUNIT1': 'deg',
            'CUNIT2': 'deg',
            'NAXIS1': 1936,
            'NAXIS2': 1936,
        }
    )


def test_ensure_pc_cdelt_fills_omitted_off_diagonal_pc():
    hdr = _north_up_pc_header_omitting_offdiag()
    assert 'PC1_2' not in hdr and 'PC2_1' not in hdr
    out = _ensure_pc_cdelt_header(hdr)
    assert out['PC1_2'] == 0.0
    assert out['PC2_1'] == 0.0
    assert out['PC1_1'] == hdr['PC1_1']
    assert out['PC2_2'] == hdr['PC2_2']
    assert out['CDELT1'] == 1.0
    assert out['CDELT2'] == 1.0


def test_create_gwcs_accepts_stamp_header_missing_pc12(tmp_path: Path):
    """Regression: Image3 failed all 2025pht filters with KeyError PC1_2."""
    hdr = _north_up_pc_header_omitting_offdiag()
    path = create_gwcs(outdir=str(tmp_path), sci_header=hdr, filename='stamp.asdf')
    assert Path(path).is_file()


def test_centered_stamp_write_has_full_pc_and_image3_gwcs(tmp_path: Path):
    from st123.stages.mosaic.mosaic import build_centered_stamp_wcs

    stamp = build_centered_stamp_wcs(
        70.370300,
        -2.865511,
        size_x_arcsec=60.0,
        size_y_arcsec=60.0,
        pixel_scale_arcsec=0.031,
        orientat_deg=0.0,
    )
    write_stamp_wcs(tmp_path, stamp)
    hdr = fits.getheader(tmp_path / STAMP_WCS_BASENAME)
    for key in ('PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'CDELT1', 'CDELT2'):
        assert key in hdr, key
    loaded = load_stamp_wcs(tmp_path)
    assert loaded is not None
    filt_hdr = rescale_wcs_to_pixel_scale(loaded, 0.031)
    for key in ('PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'CDELT1', 'CDELT2'):
        assert key in filt_hdr, key
    create_gwcs(outdir=str(tmp_path), sci_header=filt_hdr, filename='filt.asdf')
