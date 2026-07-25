#!/usr/bin/env python3
"""
Relative JHAT alignment of one JWST image to a reference, with dispersion metrics.

Builds a photometry catalog from ``--ref`` (or a master catalog from multiple
overlapping references), then runs ``st123.alignment.align_jwst_image`` (which
photometers ``--align`` internally) and reports initial/final dispersion.

Example:

    python -m st123.scripts.relative_align \\
        --ref /path/to/coadd_i2d.fits \\
        --align /path/to/mirimage_cal.fits \\
        --outdir alignment_output

Library module: ``st123.alignment.relative_align``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder

warnings.filterwarnings('ignore')

import st123  # noqa: E402


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Align --align to a photometry catalog from --ref with JHAT, '
            'and report alignment dispersion.'
        )
    )
    parser.add_argument(
        '--ref',
        required=True,
        help='Reference image used to build the photometry catalog (e.g. coadd i2d).',
    )
    parser.add_argument(
        '--align',
        required=True,
        help='Image to align (e.g. MIRI *_cal.fits or NIRCam *_i2d.fits).',
    )
    parser.add_argument(
        '--outdir',
        default='alignment_output',
        help='Output directory for JHAT products (default: alignment_output).',
    )
    parser.add_argument(
        '--photfile',
        default=None,
        help='Reuse an existing reference .phot.txt catalog instead of building one.',
    )
    parser.add_argument(
        '--nbright',
        type=int,
        default=800,
        help='Number of bright sources for JHAT (default: 800).',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Enable JHAT diagnostic plots (saved via Agg; non-interactive).',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose JHAT / alignment output.',
    )
    return parser


def has_jwst_gwcs(image: str) -> bool:
    """Return True if the FITS file has a JWST ASDF / GWCS extension."""
    with fits.open(image) as hdul:
        return any(hdu.name == 'ASDF' for hdu in hdul)


def phot_catalog_path(image: str, outdir: str) -> str:
    """Return ``outdir/<image-stem>.phot.txt``."""
    return str(Path(outdir) / f'{Path(image).stem}.phot.txt')


def write_jhat_phot_table(table: Table, photfilename: str) -> str:
    """
    Write a catalog JHAT can load.

    JHAT's pdastro loader does not treat '#' as a comment header, so write a
    plain space-separated table via pandas.
    """
    Path(photfilename).parent.mkdir(parents=True, exist_ok=True)
    table.to_pandas().to_string(photfilename, index=False)
    return photfilename


def load_sci_data_wcs(image: str) -> tuple[np.ndarray, WCS]:
    """Load science array + WCS from SCI if present, else the first 2-D HDU."""
    with fits.open(image) as hdul:
        if 'SCI' in hdul and hdul['SCI'].data is not None:
            hdu = hdul['SCI']
        else:
            hdu = next(
                (h for h in hdul if h.data is not None and getattr(h.data, 'ndim', 0) == 2),
                None,
            )
            if hdu is None:
                raise ValueError(f'No 2-D image HDU found in {image}')
        return hdu.data.astype(float), WCS(hdu.header)


def photutils_phot(image: str, photfilename: str, nsigma: float = 5.0, fwhm: float = 3.0) -> str:
    """Fallback DAOStarFinder photometry using the science WCS."""
    data, wcs = load_sci_data_wcs(image)
    _, median, std = sigma_clipped_stats(data, sigma=3.0)
    sources = DAOStarFinder(fwhm=fwhm, threshold=nsigma * std)(data - median)
    if sources is None or len(sources) == 0:
        raise RuntimeError(f'No sources found in {image}')

    ra, dec = wcs.all_pix2world(sources['xcentroid'], sources['ycentroid'], 0)
    flux = np.asarray(sources['flux'], dtype=float)
    flux = np.where(flux > 0, flux, np.nan)
    mag = -2.5 * np.log10(flux)
    dmag = np.full(len(mag), 0.05)

    catalog = Table(
        {
            'x': sources['xcentroid'],
            'y': sources['ycentroid'],
            'ra': ra,
            'dec': dec,
            'mag': mag,
            'dmag': dmag,
        }
    )
    return write_jhat_phot_table(catalog, photfilename)


def read_jhat_phot_table(photfilename: str) -> Table:
    """Load a JHAT-style space-separated photometry catalog."""
    return Table.read(photfilename, format='ascii')


def build_ref_catalog(
    image: str,
    outdir: str,
    photfile: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """
    Build or stage a JHAT-compatible reference photometry catalog in ``outdir``.

    Preference order for new catalogs:
      1. ``st123.jwst_phot`` when JWST GWCS / ASDF is present
      2. ``st123.fix_phot`` for i2d mosaics (rewrites RA/Dec from SCI WCS)
      3. photutils DAOStarFinder for custom coadds without pipeline WCS

    If ``cache_dir`` is set, per-image catalogs are reused across MIRI frames.
    """
    if photfile is not None:
        photfile = str(Path(photfile).expanduser().resolve())
        if not os.path.exists(photfile):
            raise FileNotFoundError(f'Reference photometry catalog not found: {photfile}')
        print(f'Using existing reference catalog: {photfile}')
        return stage_photfile(photfile, outdir)

    dest_name = Path(phot_catalog_path(image, outdir)).name
    if cache_dir is not None:
        cache_path = Path(cache_dir).expanduser().resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        cached = cache_path / dest_name
        if cached.is_file():
            print(f'Reusing cached reference catalog: {cached}')
            return stage_photfile(str(cached), outdir, dest_name=dest_name)

    dest = phot_catalog_path(image, outdir)
    print(f'Running photometry on reference: {image}')

    if has_jwst_gwcs(image):
        print('  detected JWST ASDF/GWCS → jwst_phot')
        _, src = st123.jwst_phot(image)
        staged = stage_photfile(src, outdir, dest_name=Path(dest).name)
    elif image.endswith(('i2d.fits', 'i2d.fits.gz')):
        try:
            print('  no JWST ASDF/GWCS; trying fix_phot')
            src = st123.fix_phot(image)
            staged = stage_photfile(src, outdir, dest_name=Path(dest).name)
        except Exception as exc:
            print(f'  fix_phot failed ({exc}); falling back to photutils')
            staged = photutils_phot(image, dest)
    else:
        print('  falling back to photutils DAOStarFinder')
        staged = photutils_phot(image, dest)

    if cache_dir is not None:
        cache_dest = Path(cache_dir).expanduser().resolve() / Path(staged).name
        if not cache_dest.exists():
            shutil.copy2(staged, cache_dest)
            print(f'Cached reference catalog → {cache_dest}')

    return staged


def _normalize_phot_columns(table: Table) -> Table:
    """Keep the JHAT columns used for sky matching; synthesize missing ones."""
    required = ('ra', 'dec', 'mag')
    lower = {c.lower(): c for c in table.colnames}
    for name in required:
        if name not in lower:
            raise ValueError(f'Photometry table missing required column {name!r}')

    ra = np.asarray(table[lower['ra']], dtype=float)
    dec = np.asarray(table[lower['dec']], dtype=float)
    mag = np.asarray(table[lower['mag']], dtype=float)
    if 'dmag' in lower:
        dmag = np.asarray(table[lower['dmag']], dtype=float)
    else:
        dmag = np.full(len(mag), 0.05)

    # x/y are image-specific; JHAT sky matching uses ra/dec. Fill placeholders.
    if 'x' in lower and 'y' in lower:
        x = np.asarray(table[lower['x']], dtype=float)
        y = np.asarray(table[lower['y']], dtype=float)
    else:
        x = np.zeros(len(mag))
        y = np.zeros(len(mag))

    return Table({'x': x, 'y': y, 'ra': ra, 'dec': dec, 'mag': mag, 'dmag': dmag})


def merge_phot_catalogs(
    tables: list[Table],
    match_radius_arcsec: float = 0.1,
) -> Table:
    """
    Merge photometry tables on sky, keeping the brightest source on conflicts.

    Assumes all catalogs share the same astrometric frame.
    """
    from astropy.coordinates import SkyCoord
    from astropy.table import vstack
    import astropy.units as u

    if not tables:
        raise ValueError('No photometry tables to merge')

    merged = _normalize_phot_columns(tables[0])
    for table in tables[1:]:
        incoming = _normalize_phot_columns(table)
        if len(merged) == 0:
            merged = incoming
            continue
        if len(incoming) == 0:
            continue

        ref_coord = SkyCoord(ra=merged['ra'] * u.deg, dec=merged['dec'] * u.deg)
        new_coord = SkyCoord(ra=incoming['ra'] * u.deg, dec=incoming['dec'] * u.deg)
        idx, sep, _ = new_coord.match_to_catalog_sky(ref_coord)
        matched = sep < (match_radius_arcsec * u.arcsec)

        # Replace existing entries with brighter matches.
        for i in np.where(matched)[0]:
            j = int(idx[i])
            if incoming['mag'][i] < merged['mag'][j]:
                for col in ('x', 'y', 'ra', 'dec', 'mag', 'dmag'):
                    merged[col][j] = incoming[col][i]

        # Append unmatched sources.
        if np.any(~matched):
            merged = vstack([merged, incoming[~matched]], join_type='exact')

    # Brightest first so JHAT's Nbright cut prefers useful stars.
    merged.sort('mag')
    return merged


def clip_catalog_to_miri_footprint(table: Table, miri_image: str) -> Table:
    """
    Keep sources whose sky positions fall in the MIRI illuminated ROI.

    Clipping is done in sky coordinates using the illuminated ``S_REGION``
    polygon.  This avoids ``WCS.all_world2pix`` ``NoConvergence`` errors that
    occur when the merged master catalog contains stars far outside the MIRI
    WCS validity domain (common once several large reference coadds are merged).
    """
    from st123.mosaic.image_overlap import MirIFootprint
    from matplotlib.path import Path as MplPath

    miri = MirIFootprint.from_fits(miri_image)
    ra = np.asarray(table['ra'], dtype=float)
    dec = np.asarray(table['dec'], dtype=float)

    # Illuminated S_REGION vertices are (lon, lat) in degrees.
    verts = np.asarray(miri.s_region.vertices, dtype=float)
    if len(verts) < 3:
        raise RuntimeError(f'Illuminated S_REGION has too few vertices for {miri_image}')

    # Handle simple RA wrap near 0/360 if the footprint crosses the branch cut.
    ra_v = verts[:, 0].copy()
    if ra_v.max() - ra_v.min() > 180.0:
        ra_v = np.where(ra_v < 180.0, ra_v + 360.0, ra_v)
        ra_use = np.where(ra < 180.0, ra + 360.0, ra)
    else:
        ra_use = ra

    sky_path = MplPath(np.column_stack([ra_v, verts[:, 1]]))
    # Slightly pad the path test; contains_points is inclusive of edges with radius.
    keep = sky_path.contains_points(np.column_stack([ra_use, dec]))

    # Fallback / cross-check with pixel-plane illuminated polygon when sky clip
    # yields nothing (should be rare; quiet=True avoids hard WCS failures).
    if not np.any(keep):
        print(
            'WARNING: sky-footprint clip kept 0 sources; '
            'retrying with quiet WCS pixel clip'
        )
        from shapely.geometry import Point

        x, y = miri.wcs.all_world2pix(ra, dec, 0, quiet=True)
        keep = np.array(
            [
                np.isfinite(xi)
                and np.isfinite(yi)
                and miri.polygon.contains(Point(float(xi), float(yi)))
                for xi, yi in zip(x, y)
            ],
            dtype=bool,
        )

    clipped = table[keep]
    print(
        f'Clipped master catalog to MIRI footprint: '
        f'{len(table)} → {len(clipped)} sources'
    )
    return clipped


def build_master_ref_catalog(
    ref_images: list[str],
    outdir: str,
    *,
    align_image: str | None = None,
    cache_dir: str | None = None,
    match_radius_arcsec: float = 0.1,
    clip_to_align_footprint: bool = True,
    dest_name: str = 'master_ref.phot.txt',
) -> str:
    """
    Build a master JHAT catalog from every overlapping reference image.

    Photometry is run (or loaded from ``cache_dir``) for each reference, merged
    on sky (brightest kept within ``match_radius_arcsec``), optionally clipped
    to the MIRI illuminated footprint, and written to ``outdir/dest_name``.
    """
    if not ref_images:
        raise ValueError('ref_images is empty')

    outdir = resolve_outdir(outdir)
    tables: list[Table] = []
    for ref in ref_images:
        ref = str(Path(ref).expanduser().resolve())
        try:
            phot = build_ref_catalog(ref, outdir, cache_dir=cache_dir)
            tables.append(read_jhat_phot_table(phot))
            print(f'  + {len(tables[-1])} sources from {ref}')
        except Exception as exc:
            print(f'  WARNING: photometry failed for {ref}: {exc}')

    if not tables:
        raise RuntimeError('No reference photometry catalogs could be built')

    master = merge_phot_catalogs(tables, match_radius_arcsec=match_radius_arcsec)
    print(
        f'Merged {len(tables)} reference catalog(s) → {len(master)} unique sources '
        f'(match radius {match_radius_arcsec}\" )'
    )

    if clip_to_align_footprint and align_image is not None:
        master = clip_catalog_to_miri_footprint(master, align_image)
        if len(master) == 0:
            raise RuntimeError(
                'Master catalog has no sources inside the MIRI illuminated footprint'
            )

    dest = str(Path(outdir) / dest_name)
    write_jhat_phot_table(master, dest)
    print(f'Wrote master reference catalog → {dest} ({len(master)} sources)')
    return dest


def stage_photfile(
    photfile: str,
    outdir: str,
    dest_name: str | None = None,
) -> str:
    """Copy a catalog into ``outdir`` (no-op if already there) and return that path."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    dest = os.path.join(outdir, dest_name or os.path.basename(photfile))
    if os.path.exists(dest) and os.path.samefile(photfile, dest):
        return dest
    shutil.copy2(photfile, dest)
    print(f'Copied reference catalog → {dest}')
    return dest


@contextmanager
def working_directory(path: str):
    """Temporarily change the process working directory."""
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


@contextmanager
def plot_saver(outdir: str, prefix: str):
    """Save JHAT figures that only call ``plt.show()`` into ``outdir``."""
    counter = {'n': 0}
    already_saved: set[int] = set()
    original_show = plt.show
    original_savefig = plt.Figure.savefig

    def savefig_track(self, *args, **kwargs):
        already_saved.add(self.number)
        return original_savefig(self, *args, **kwargs)

    def show_and_save(*args, **kwargs):
        for num in plt.get_fignums():
            if num in already_saved:
                continue
            fig = plt.figure(num)
            counter['n'] += 1
            out = os.path.join(outdir, f'{prefix}.diag_{counter["n"]:02d}.png')
            # Use the original savefig so we do not mark this as a JHAT savefig.
            original_savefig(fig, out, dpi=150, bbox_inches='tight')
            print(f'Saved diagnostic plot: {out}')
        already_saved.clear()
        plt.close('all')

    plt.Figure.savefig = savefig_track
    plt.show = show_and_save
    try:
        yield
    finally:
        plt.show = original_show
        plt.Figure.savefig = original_savefig


def resolve_outdir(outdir: str) -> str:
    path = Path(outdir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _jhat_paths(align_image: str, outdir: str) -> tuple[str, str, str]:
    """Return ``(jhat_fits, align_phot_guess, stem)`` paths under ``outdir``."""
    base = Path(align_image).name
    if base.endswith('_cal.fits'):
        stem = base.replace('_cal.fits', '')
        jhat = str(Path(outdir) / base.replace('_cal.fits', '_jhat.fits'))
    elif base.endswith('_i2d.fits'):
        stem = base.replace('_i2d.fits', '')
        jhat = str(Path(outdir) / base.replace('_i2d.fits', '_jhat.fits'))
    else:
        stem = Path(base).stem
        jhat = str(Path(outdir) / f'{stem}_jhat.fits')
    # Prefer post-alignment photometry written by jwst_dispersion.
    phot_candidates = [
        str(Path(outdir) / f'{stem}_jhat_cal.phot.txt'),
        str(Path(outdir) / f'{stem}_jhat_i2d.phot.txt'),
        str(Path(outdir) / f'{stem}.phot.txt'),
    ]
    align_phot = next((p for p in phot_candidates if os.path.exists(p)), phot_candidates[0])
    return jhat, align_phot, stem


def read_dispersion_mas(jhat_image: str) -> tuple[float | None, int | None]:
    """Read final mean dispersion (mas) and calibrator count from a JHAT product."""
    with fits.open(jhat_image) as hdul:
        hdr = hdul[0].header
        disp = hdr.get('JWDISPM', hdr.get('GADISPM'))
        ncal = hdr.get('JWNCAL', hdr.get('GANCAL'))
    disp_mas = float(disp) * 1000.0 if disp is not None else None
    n_cal = int(ncal) if ncal is not None else None
    return disp_mas, n_cal


def read_jhat_pixel_offset(jhat_image: str) -> tuple[float, float]:
    """Return ``(xshift, yshift)`` pixel offsets from a JHAT product header."""
    with fits.open(jhat_image) as hdul:
        hdr = hdul[0].header
        xoff = hdr.get('XOFFSET', 0.0)
        yoff = hdr.get('YOFFSET', 0.0)
    return float(xoff or 0.0), float(yoff or 0.0)


def _apply_miri_calibrator_mask(
    jhat_df,
    *,
    miri_mag_min: float | None = None,
    miri_mag_max: float | None = None,
    miri_round_max: float | None = None,
    miri_sharp_min: float | None = None,
    miri_sharp_max: float | None = None,
):
    """Return a boolean mask of star-like MIRI detections suitable as calibrators."""
    import pandas as pd

    if not isinstance(jhat_df, pd.DataFrame):
        jhat_df = pd.DataFrame(jhat_df)
    keep = np.ones(len(jhat_df), dtype=bool)
    if miri_mag_min is not None and 'mag' in jhat_df.columns:
        keep &= np.asarray(jhat_df['mag'], dtype=float) >= float(miri_mag_min)
    if miri_mag_max is not None and 'mag' in jhat_df.columns:
        keep &= np.asarray(jhat_df['mag'], dtype=float) <= float(miri_mag_max)
    if miri_round_max is not None and 'roundness1' in jhat_df.columns:
        keep &= np.abs(np.asarray(jhat_df['roundness1'], dtype=float)) <= float(
            miri_round_max
        )
    if miri_sharp_min is not None and 'sharpness' in jhat_df.columns:
        keep &= np.asarray(jhat_df['sharpness'], dtype=float) >= float(miri_sharp_min)
    if miri_sharp_max is not None and 'sharpness' in jhat_df.columns:
        keep &= np.asarray(jhat_df['sharpness'], dtype=float) <= float(miri_sharp_max)
    return keep


def iterative_sigma_clip_matches(
    align_phot: str,
    ref_table: Table,
    *,
    dist_limit_arcsec: float = 0.5,
    sigma: float = 2.0,
    max_clip_iter: int = 10,
    max_residual_arcsec: float | None = None,
    miri_mag_min: float | None = None,
    miri_mag_max: float | None = None,
    miri_round_max: float | None = None,
    miri_sharp_min: float | None = None,
    miri_sharp_max: float | None = None,
) -> tuple[Table, dict]:
    """
    Crossmatch aligned photometry to a reference table and iteratively clip outliers.

    Optional MIRI morphology / magnitude cuts are applied before matching when
    those columns exist. An optional hard residual ceiling is applied after
    sigma-clipping.

    Returns a cleaned reference table (matched ref stars that survive clipping)
    and a stats dict describing the clip.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.stats import sigma_clipped_stats

    import pandas as pd

    jhat_df = pd.read_csv(align_phot, sep=r'\s+')
    morph_keep = _apply_miri_calibrator_mask(
        jhat_df,
        miri_mag_min=miri_mag_min,
        miri_mag_max=miri_mag_max,
        miri_round_max=miri_round_max,
        miri_sharp_min=miri_sharp_min,
        miri_sharp_max=miri_sharp_max,
    )
    n_morph = int(morph_keep.sum())
    if n_morph == 0:
        raise RuntimeError('No MIRI sources survive calibrator morphology/mag cuts')
    if n_morph < len(jhat_df):
        print(
            f'  calibrator morph/mag cut: kept {n_morph}/{len(jhat_df)} MIRI sources'
        )
    jhat_df = jhat_df.loc[morph_keep].reset_index(drop=True)

    jh = SkyCoord(
        ra=np.asarray(jhat_df['ra'], dtype=float) * u.deg,
        dec=np.asarray(jhat_df['dec'], dtype=float) * u.deg,
    )
    rf = SkyCoord(
        ra=np.asarray(ref_table['ra'], dtype=float) * u.deg,
        dec=np.asarray(ref_table['dec'], dtype=float) * u.deg,
    )
    matched = st123.xmatch_common(jh, rf, dist_limit=dist_limit_arcsec)
    if len(matched) == 0:
        raise RuntimeError('No matches for iterative outlier clipping')

    d2d = np.asarray(matched['d2d'], dtype=float)
    keep = np.ones(len(matched), dtype=bool)
    for i in range(max_clip_iter):
        di = d2d[keep]
        _mn, med, std = sigma_clipped_stats(di, sigma=sigma)
        thr = float(med + sigma * std)
        if max_residual_arcsec is not None:
            thr = min(thr, float(max_residual_arcsec))
        new_keep = keep & (d2d <= thr)
        print(
            f'  clip iter {i}: n={int(new_keep.sum())}/{len(matched)} '
            f'med={np.median(d2d[new_keep])*1000:.2f} mas '
            f'thr={thr*1000:.2f} mas'
        )
        if new_keep.sum() == keep.sum():
            keep = new_keep
            break
        keep = new_keep
        if keep.sum() < 10:
            break

    if max_residual_arcsec is not None:
        hard = d2d <= float(max_residual_arcsec)
        if hard.sum() < keep.sum():
            print(
                f'  hard residual cut ({max_residual_arcsec*1000:.1f} mas): '
                f'{int(keep.sum())} → {int((keep & hard).sum())}'
            )
        keep = keep & hard

    good_ref_idx = np.unique(np.asarray(matched['idx_2'], dtype=int)[keep])
    cleaned = ref_table[good_ref_idx]
    stats = {
        'n_match_initial': int(len(matched)),
        'n_match_clipped': int(keep.sum()),
        'n_ref_kept': int(len(cleaned)),
        'median_d2d_mas': float(np.median(d2d[keep]) * 1000.0) if keep.any() else float('nan'),
        'mean_d2d_mas': float(np.mean(d2d[keep]) * 1000.0) if keep.any() else float('nan'),
        'n_miri_morph': n_morph,
    }
    return cleaned, stats


def refine_alignment_iteratively(
    align_image: str,
    outdir: str,
    ref_phot: str,
    *,
    nbright: int = 800,
    plot: bool = False,
    verbose: bool = False,
    sigma: float = 2.0,
    max_iter: int = 5,
    tol_mas: float = 1.0,
    min_calibrators: int = 20,
    dist_limit_arcsec: float = 0.5,
    max_residual_arcsec: float | None = None,
    miri_mag_min: float | None = None,
    miri_mag_max: float | None = None,
    miri_round_max: float | None = None,
    miri_sharp_min: float | None = None,
    miri_sharp_max: float | None = None,
    jhat_params: dict | None = None,
) -> tuple[object, float | None, int | None]:
    """
    Iteratively prune outlier reference stars and re-run JHAT until dispersion converges.

    Each iteration:
      1. Crossmatch the current aligned photometry to the reference catalog
      2. Sigma-clip significant residual outliers
      3. Rewrite a cleaned reference catalog from surviving matches
      4. Re-run JHAT against the cleaned catalog (from the original science image)
      5. Stop when |Δdispersion| < ``tol_mas`` or no further stars are clipped
    """
    outdir = resolve_outdir(outdir)
    align_image = str(Path(align_image).expanduser().resolve())
    ref_phot = str(Path(ref_phot).expanduser().resolve())
    stem = Path(align_image).stem.replace('_cal', '').replace('_i2d', '')

    jhat, align_phot, _ = _jhat_paths(align_image, outdir)
    if not os.path.exists(jhat):
        raise FileNotFoundError(f'JHAT product not found for refinement: {jhat}')

    disp_mas, n_cal = read_dispersion_mas(jhat)
    xshift, yshift = read_jhat_pixel_offset(jhat)
    guess_offset: object = (xshift, yshift)
    print(
        f'Iterative refinement starting from dispersion={disp_mas} mas, '
        f'n_calibrators={n_cal}, xshift={xshift:.3f}, yshift={yshift:.3f}'
    )

    current_ref = ref_phot
    for it in range(1, max_iter + 1):
        if not os.path.exists(align_phot):
            print(f'  refine iter {it}: missing align photometry {align_phot}; stopping')
            break

        ref_table = read_jhat_phot_table(current_ref)
        try:
            cleaned, stats = iterative_sigma_clip_matches(
                align_phot,
                ref_table,
                dist_limit_arcsec=dist_limit_arcsec,
                sigma=sigma,
                max_residual_arcsec=max_residual_arcsec,
                miri_mag_min=miri_mag_min,
                miri_mag_max=miri_mag_max,
                miri_round_max=miri_round_max,
                miri_sharp_min=miri_sharp_min,
                miri_sharp_max=miri_sharp_max,
            )
        except Exception as exc:
            print(f'  refine iter {it}: clipping failed ({exc}); stopping')
            break

        if stats['n_ref_kept'] < min_calibrators:
            print(
                f'  refine iter {it}: only {stats["n_ref_kept"]} ref stars left '
                f'(<{min_calibrators}); stopping'
            )
            break

        # Only re-run JHAT when the matched set actually shrank (sigma-clip /
        # hard residual). Do NOT treat "matched subset << full master_ref" as
        # progress — that is always true on the first pass and was forcing a
        # destructive catalog rewrite that hurt F560W.
        #
        # Optional MIRI morph/mag cuts are applied *before* matching, so when
        # they are active force one cleaned-catalog re-run on iter 1 even if
        # sigma-clip finds no further outliers.
        morph_active = any(
            v is not None
            for v in (
                miri_mag_min,
                miri_mag_max,
                miri_round_max,
                miri_sharp_min,
                miri_sharp_max,
            )
        )
        no_clip_progress = stats['n_match_clipped'] >= stats['n_match_initial']
        force_morph_pass = (
            morph_active
            and it == 1
            and int(stats.get('n_miri_morph', 0)) > 0
            and current_ref == ref_phot
        )
        if no_clip_progress and not force_morph_pass:
            print(f'  refine iter {it}: no outliers clipped; converged')
            break

        cleaned_path = str(Path(outdir) / f'master_ref_refined_iter{it:02d}.phot.txt')
        write_jhat_phot_table(cleaned, cleaned_path)
        print(
            f'  refine iter {it}: wrote cleaned catalog {cleaned_path} '
            f'({len(cleaned)} stars; match median {stats["median_d2d_mas"]:.2f} mas)'
        )

        # Snapshot current JHAT products so a worse iteration can be rolled back.
        jhat, align_phot, _ = _jhat_paths(align_image, outdir)
        backup_jhat = jhat + '.refine_bak'
        backup_phot = align_phot + '.refine_bak'
        if os.path.exists(jhat):
            shutil.copy2(jhat, backup_jhat)
        if os.path.exists(align_phot):
            shutil.copy2(align_phot, backup_phot)

        # Only seed pixel offsets for F770W-style tight refine (hard residual
        # ceiling). Seeding large F560W XOFFSET/YOFFSET (~50–130 px) into JHAT
        # with a cleaned catalog routinely fails matching and destroys the
        # ~10 mas solutions refine would otherwise recover.
        if max_residual_arcsec is not None:
            xshift, yshift = read_jhat_pixel_offset(jhat)
        else:
            xshift, yshift = 0.0, 0.0

        with working_directory(outdir):
            # soft_fail=False: never replace a refine JHAT product with an
            # unaligned copy; rollback handles worsened iterations instead.
            align_kw = dict(
                align_image=align_image,
                outdir='.',
                gaia=False,
                photfilename=cleaned_path,
                Nbright=min(nbright, len(cleaned)),
                verbose=verbose,
                soft_fail=False,
                jhat_params=jhat_params,
                xshift=xshift,
                yshift=yshift,
            )
            if plot:
                with plot_saver(outdir, prefix=f'{stem}.refine{it:02d}'):
                    guess_offset = st123.align_jwst_image(plot=True, **align_kw)
            else:
                guess_offset = st123.align_jwst_image(plot=False, **align_kw)

        jhat, align_phot, _ = _jhat_paths(align_image, outdir)
        new_disp, new_ncal = read_dispersion_mas(jhat)
        print(
            f'  refine iter {it}: dispersion {disp_mas} → {new_disp} mas '
            f'(n_calibrators={new_ncal})'
        )

        def _restore_backup_and_stop(reason: str) -> None:
            print(f'  refine iter {it}: {reason}; restoring previous JHAT products and stopping')
            if os.path.exists(backup_jhat):
                shutil.copy2(backup_jhat, jhat)
            if os.path.exists(backup_phot):
                shutil.copy2(backup_phot, align_phot)
            for bak in (backup_jhat, backup_phot):
                if os.path.exists(bak):
                    os.remove(bak)

        # Failed / soft-failed refine attempts can leave absurd dispersions.
        if new_disp is not None and new_disp > 500.0:
            _restore_backup_and_stop(
                f'refine product unusable (dispersion={new_disp:.1f} mas)'
            )
            break

        if disp_mas is not None and new_disp is not None:
            if abs(new_disp - disp_mas) < tol_mas:
                print(
                    f'  refine iter {it}: |Δdispersion|='
                    f'{abs(new_disp - disp_mas):.3f} mas < {tol_mas} mas; converged'
                )
                disp_mas, n_cal = new_disp, new_ncal
                current_ref = cleaned_path
                for bak in (backup_jhat, backup_phot):
                    if os.path.exists(bak):
                        os.remove(bak)
                break
            if new_disp > disp_mas + tol_mas:
                _restore_backup_and_stop(
                    f'dispersion worsened ({disp_mas:.3f} → {new_disp:.3f} mas)'
                )
                break

        disp_mas, n_cal = new_disp, new_ncal
        current_ref = cleaned_path
        for bak in (backup_jhat, backup_phot):
            if os.path.exists(bak):
                os.remove(bak)

    # Stage final cleaned catalog under a stable name when available.
    if current_ref != ref_phot and os.path.exists(current_ref):
        final_clean = str(Path(outdir) / 'master_ref_refined.phot.txt')
        if os.path.abspath(current_ref) != os.path.abspath(final_clean):
            shutil.copy2(current_ref, final_clean)
            print(f'Final refined reference catalog → {final_clean}')

    print(f'Iterative refinement finished: dispersion_mas={disp_mas}, n_calibrators={n_cal}')
    return guess_offset, disp_mas, n_cal


def run_alignment(
    ref_image: str | None = None,
    align_image: str | None = None,
    outdir: str = 'alignment_output',
    photfile: str | None = None,
    ref_images: list[str] | None = None,
    nbright: int = 800,
    plot: bool = False,
    verbose: bool = False,
    cache_dir: str | None = None,
    match_radius_arcsec: float = 0.1,
    clip_to_align_footprint: bool = True,
    refine: bool = True,
    refine_sigma: float = 2.0,
    refine_max_iter: int = 5,
    refine_tol_mas: float = 1.0,
    refine_dist_limit_arcsec: float = 0.5,
    max_residual_arcsec: float | None = None,
    miri_mag_min: float | None = None,
    miri_mag_max: float | None = None,
    miri_round_max: float | None = None,
    miri_sharp_min: float | None = None,
    miri_sharp_max: float | None = None,
    min_calibrators: int = 20,
    jhat_params: dict | None = None,
) -> tuple[object, str]:
    """
    Build a reference catalog and align ``align_image`` to it.

    Catalog source (first match wins):
      1. ``photfile`` — reuse an existing catalog
      2. ``ref_images`` — merge photometry from all listed references into a
         master catalog (optionally clipped to the MIRI footprint)
      3. ``ref_image`` — single-reference photometry (legacy behavior)

    If ``refine`` is True, iteratively sigma-clip outlier matches and re-run
    JHAT until the dispersion converges.

    Filter-specific calibrator knobs (``nbright``, JHAT ``objmag_lim`` /
    morphology cuts, refine residual ceilings) are typically supplied by
    ``alignment_calibrators`` via the parallel workers.
    """
    if align_image is None:
        raise ValueError('align_image is required')

    align_image = str(Path(align_image).expanduser().resolve())
    outdir = resolve_outdir(outdir)

    refs = list(ref_images) if ref_images else []
    if not refs and ref_image is not None:
        refs = [ref_image]
    refs = [str(Path(r).expanduser().resolve()) for r in refs]

    print(f'Output directory: {outdir}')
    print(f'Align image:      {align_image}')
    if jhat_params:
        print(f'JHAT param overrides: {jhat_params}')
    print(
        f'Calibrator knobs: nbright={nbright}, refine_sigma={refine_sigma}, '
        f'dist_limit={refine_dist_limit_arcsec}", '
        f'max_resid={max_residual_arcsec}"'
    )

    if photfile is not None:
        ref_phot = build_ref_catalog(refs[0] if refs else align_image, outdir, photfile=photfile)
        print(f'Reference catalog: {ref_phot}')
    elif len(refs) > 1:
        print(f'Reference images ({len(refs)}): building master catalog')
        for path in refs:
            print(f'  {path}')
        ref_phot = build_master_ref_catalog(
            refs,
            outdir,
            align_image=align_image,
            cache_dir=cache_dir,
            match_radius_arcsec=match_radius_arcsec,
            clip_to_align_footprint=clip_to_align_footprint,
        )
    elif len(refs) == 1:
        print(f'Reference image:  {refs[0]}')
        ref_phot = build_ref_catalog(refs[0], outdir, cache_dir=cache_dir)
    else:
        raise ValueError('Provide photfile, ref_image, or ref_images')

    # JHAT's outsubdir is relative to cwd; run from outdir so products land there.
    # Paths above are absolute so chdir is safe.
    stem = Path(align_image).stem.replace('_cal', '').replace('_i2d', '')
    align_kw = dict(
        align_image=align_image,
        outdir='.',
        gaia=False,
        photfilename=ref_phot,
        Nbright=nbright,
        verbose=verbose,
        jhat_params=jhat_params,
    )
    with working_directory(outdir):
        if plot:
            with plot_saver(outdir, prefix=stem):
                guess_offset = st123.align_jwst_image(plot=True, **align_kw)
        else:
            guess_offset = st123.align_jwst_image(plot=False, **align_kw)

    if refine:
        guess_offset, disp_mas, n_cal = refine_alignment_iteratively(
            align_image=align_image,
            outdir=outdir,
            ref_phot=ref_phot,
            nbright=nbright,
            plot=plot,
            verbose=verbose,
            sigma=refine_sigma,
            max_iter=refine_max_iter,
            tol_mas=refine_tol_mas,
            min_calibrators=min_calibrators,
            dist_limit_arcsec=refine_dist_limit_arcsec,
            max_residual_arcsec=max_residual_arcsec,
            miri_mag_min=miri_mag_min,
            miri_mag_max=miri_mag_max,
            miri_round_max=miri_round_max,
            miri_sharp_min=miri_sharp_min,
            miri_sharp_max=miri_sharp_max,
            jhat_params=jhat_params,
        )
        print(f'Refined dispersion_mas={disp_mas}, n_calibrators={n_cal}')

    return guess_offset, outdir


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)

    for path, label in ((args.ref, '--ref'), (args.align, '--align')):
        if not os.path.exists(path):
            print(f'ERROR: {label} file not found: {path}', file=sys.stderr)
            return 1

    try:
        guess_offset, outdir = run_alignment(
            ref_image=args.ref,
            align_image=args.align,
            outdir=args.outdir,
            photfile=args.photfile,
            nbright=args.nbright,
            plot=args.plot,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f'ERROR: alignment failed: {exc}', file=sys.stderr)
        return 1

    print(f'Guess offset (x, y): {guess_offset}')
    print(f'Done. Products in {outdir}')
    return 0

