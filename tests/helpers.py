"""Synthetic FITS helpers shared across tests (not imported by the package)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def make_wcs_header(
    shape: tuple[int, int] = (80, 100),
    crval: tuple[float, float] = (150.0, 2.0),
    cdelt: float = 0.001,
) -> fits.Header:
    ny, nx = shape
    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    header['CRPIX1'] = nx / 2.0
    header['CRPIX2'] = ny / 2.0
    header['CRVAL1'] = crval[0]
    header['CRVAL2'] = crval[1]
    header['CDELT1'] = -cdelt
    header['CDELT2'] = cdelt
    return header


def write_illuminated_fits(
    path: Path,
    *,
    shape: tuple[int, int] = (80, 100),
    crval: tuple[float, float] = (150.0, 2.0),
    include_s_region: bool = False,
) -> Path:
    ny, nx = shape
    sci = np.ones(shape, dtype=np.float32)
    # Leave a dark border so skimage contours close (edge-touching masks
    # yield zero-area polygons).
    dq = np.full(shape, 1024, dtype=np.uint16)
    dq[2:-2, max(nx // 2, 2) : nx - 2] = 0

    header = make_wcs_header(shape=shape, crval=crval)
    if include_s_region:
        wcs = WCS(header)
        corners = np.array(
            [
                [nx // 2, 0],
                [nx - 1, 0],
                [nx - 1, ny - 1],
                [nx // 2, ny - 1],
            ],
            dtype=float,
        )
        ra, dec = wcs.pixel_to_world_values(corners[:, 0], corners[:, 1])
        verts = ' '.join(f'{r:.9f} {d:.9f}' for r, d in zip(ra, dec))
        header['S_REGION'] = f'POLYGON ICRS {verts}'

    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=sci, header=header, name='SCI'),
            fits.ImageHDU(data=dq, name='DQ'),
        ]
    ).writeto(path, overwrite=True)
    return path


def write_ref_with_s_region(
    path: Path,
    *,
    shape: tuple[int, int] = (80, 100),
    crval: tuple[float, float] = (150.0, 2.0),
) -> Path:
    ny, nx = shape
    header = make_wcs_header(shape=shape, crval=crval)
    wcs = WCS(header)
    corners = np.array(
        [[0, 0], [nx - 1, 0], [nx - 1, ny - 1], [0, ny - 1]],
        dtype=float,
    )
    ra, dec = wcs.pixel_to_world_values(corners[:, 0], corners[:, 1])
    verts = ' '.join(f'{r:.9f} {d:.9f}' for r, d in zip(ra, dec))
    header['S_REGION'] = f'POLYGON ICRS {verts}'
    data = np.ones(shape, dtype=np.float32)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=data, header=header, name='SCI'),
        ]
    ).writeto(path, overwrite=True)
    return path
