"""Shared helpers: coordinates, FITS bookkeeping, visits, and cross-matching."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Column, Table
from astropy.time import Time

from st123.mast import parse_s_region
from st123.utils.settings import (
    BEST_FILTER_TYPES,
    BEST_REFERENCE_FILTERS,
    acceptable_filters,
)

logger = logging.getLogger(__name__)


def is_number(num: Any) -> bool:
    """
    Return whether ``num`` can be converted to a float.

    Parameters
    ----------
    num : object
        Value to test.

    Returns
    -------
    bool
        True when ``float(num)`` succeeds.
    """
    try:
        float(num)
    except (TypeError, ValueError):
        return False
    return True


def parse_coord(ra: str | float, dec: str | float) -> SkyCoord | None:
    """
    Parse RA/Dec into an ICRS :class:`~astropy.coordinates.SkyCoord`.

    Accepts decimal degrees or sexagesimal strings (``':'`` in both values).

    Parameters
    ----------
    ra : str or float
        Right ascension.
    dec : str or float
        Declination.

    Returns
    -------
    SkyCoord or None
        Parsed coordinate, or ``None`` when parsing fails.
    """
    if (not (is_number(ra) and is_number(dec)) and
            (':' not in str(ra) and ':' not in str(dec))):
        logger.error('cannot interpret: %s %s', ra, dec)
        return None

    if ':' in str(ra) and ':' in str(dec):
        unit = (u.hourangle, u.deg)
    else:
        unit = (u.deg, u.deg)

    try:
        return SkyCoord(ra, dec, frame='icrs', unit=unit)
    except ValueError:
        logger.error('Cannot parse coordinates: %s %s', ra, dec)
        return None


def get_filter(image: str | Path) -> str:
    """
    Read the filter name from a FITS header.

    Parameters
    ----------
    image : str or Path
        Path to a calibrated science FITS file.

    Returns
    -------
    str
        Lowercase filter name (``FILTER``, or ``FILTER1``/``FILTER2``).
    """
    try:
        f = str(fits.getval(image, 'FILTER'))
    except Exception:
        f = str(fits.getval(image, 'FILTER1'))
        if 'clear' in f.lower():
            f = str(fits.getval(image, 'FILTER2'))
    return f.lower()


def get_module(image: str | Path) -> str:
    """
    Read the JWST module or NIRCam letter from a FITS header.

    Parameters
    ----------
    image : str or Path
        Path to a science FITS file.

    Returns
    -------
    str
        Lowercase module identifier (``MODULE`` or first NRC detector letter).
    """
    try:
        f = str(fits.getval(image, 'MODULE'))
    except Exception:
        f = str(fits.getval(image, 'DETECTOR'))
        f = f.split('NRC')[1][0]
    return f.lower()


def get_instrument(image: str | Path) -> str:
    """
    Read the instrument name from a FITS primary header.

    Parameters
    ----------
    image : str or Path
        Path to a science FITS file.

    Returns
    -------
    str
        Lowercase ``INSTRUME`` value.
    """
    return str(fits.getval(image, 'INSTRUME')).lower()


def get_chip(image: str | Path) -> int | str:
    """
    Return the detector chip identifier from a FITS file.

    For multi-extension HST images, ambiguous cases default to chip ``1``.
    JWST products typically expose ``DETECTOR`` instead of ``CCDCHIP``.

    Parameters
    ----------
    image : str or Path
        Path to a science FITS file.

    Returns
    -------
    int or str
        Chip number or detector name; ``1`` when no chip keyword is found.
    """
    with fits.open(image) as hdu:
        chip = None
        for h in hdu:
            if 'CCDCHIP' in h.header:
                chip = 1 if chip is not None else h.header['CCDCHIP']
            elif 'DETECTOR' in h.header:
                chip = 1 if chip is not None else h.header['DETECTOR']
    return chip if chip is not None else 1


def get_detector_chip(filename: str) -> str | None:
    """
    Get the detector chip token from a file name.

    Parameters
    ----------
    filename : str
        File name or path.

    Returns
    -------
    str or None
        Detector chip (e.g. ``nrcb1``, ``nrcblong``, ``mirimage``), or ``None``.
    """
    tokens = os.path.basename(filename).split('_')
    for token in tokens:
        if 'nrc' in token:
            return token
    for token in tokens:
        if 'mirimage' in token.lower():
            return token
    return None


def get_zpt(
    image: str | Path,
    ccdchip: int | str = 1,
    zptype: str = 'abmag',
) -> float | None:
    """
    Compute the photometric zero point from ``PHOTFLAM`` and ``PHOTPLAM``.

    Parameters
    ----------
    image : str or Path
        Path to a science FITS file.
    ccdchip : int or str, optional
        CCD chip or detector name when multiple science HDUs are present.
    zptype : str, optional
        ``'abmag'`` for AB magnitude or ``'st'`` for ST magnitude.

    Returns
    -------
    float or None
        Zero point in magnitudes, or ``None`` when header keywords are missing.
    """
    with fits.open(image, mode='readonly') as hdu:
        inst = get_instrument(image).lower()
        sci = [
            h for h in hdu
            if 'PHOTPLAM' in h.header and 'PHOTFLAM' in h.header
        ]

        use_hdu = None
        if len(sci) == 1:
            use_hdu = sci[0]
        elif len(sci) > 1:
            for h in sci:
                if 'acs' in inst or 'wfc3' in inst:
                    if h.header.get('CCDCHIP') == ccdchip:
                        use_hdu = h
                        break
                elif h.header.get('DETECTOR') == ccdchip:
                    use_hdu = h
                    break

        if use_hdu is None:
            return None

        photplam = float(use_hdu.header['PHOTPLAM'])
        photflam = float(use_hdu.header['PHOTFLAM'])

    if 'ab' in zptype:
        return -2.5 * np.log10(photflam) - 5 * np.log10(photplam) - 2.408
    if 'st' in zptype:
        return -2.5 * np.log10(photflam) - 21.1
    return None


def organize_visit_tables(obstable: Table, byvisit: bool = False) -> list[Table]:
    """
    Split an observation table into per-visit subtables.

    Parameters
    ----------
    obstable : astropy.table.Table
        Table with a ``visit`` column.
    byvisit : bool, optional
        If True, return one table per unique visit; otherwise return a
        single-element list containing ``obstable``.

    Returns
    -------
    list of astropy.table.Table
        Visit-grouped subtables.
    """
    if not byvisit:
        return [obstable]
    return [
        obstable[obstable['visit'] == visit]
        for visit in sorted(set(obstable['visit'].data))
    ]


def organize_reduction_tables(
    obstable: Table,
    byvisit: bool = False,
    bymodule: bool = False,
) -> list[list[Table]]:
    """
    Group an observation table by module and optionally by visit.

    Parameters
    ----------
    obstable : astropy.table.Table
        Table with ``module`` and ``visit`` columns.
    byvisit : bool, optional
        Split each module group by visit (passed to
        :func:`organize_visit_tables`).
    bymodule : bool, optional
        If True, split by unique ``module`` values before visit grouping.

    Returns
    -------
    list of list of astropy.table.Table
        Nested lists of subtables (one outer entry per module when
        ``bymodule`` is True, otherwise a single outer entry).
    """
    if not bymodule:
        return [organize_visit_tables(obstable, byvisit=byvisit)]
    return [
        organize_visit_tables(obstable[obstable['module'] == mod], byvisit=byvisit)
        for mod in sorted(set(obstable['module'].data))
    ]


def pick_deepest_images(
    images: list[str | Path],
    reffilter: str | None = None,
    avoid_wfpc2: bool = False,
    refinst: str | None = None,
) -> list[str]:
    """
    Select reference images with the deepest total exposure.

    Chooses instrument/filter combinations preferring common wide filters,
    then suffix types (``lp``, ``w``, ``x``, ``m``, ``n``).

    Parameters
    ----------
    images : list of str or Path
        Science FITS paths to consider.
    reffilter : str or None, optional
        Force a specific filter (must appear in
        :data:`st123.utils.settings.acceptable_filters`).
    avoid_wfpc2 : bool, optional
        Exclude WFPC2 instrument/filter pairs when alternatives exist.
    refinst : str or None, optional
        Restrict candidates to this instrument substring.

    Returns
    -------
    list of str
        Paths of images in the best instrument/filter group.
    """
    best_filters = list(BEST_REFERENCE_FILTERS)
    if reffilter and reffilter.upper() in acceptable_filters:
        best_filters = [reffilter.lower()]
    best_types = list(BEST_FILTER_TYPES)

    filts = [get_filter(im) for im in images]
    insts = [
        get_instrument(im).replace('_full', '').replace('_sub', '')
        for im in images
    ]

    if refinst:
        mask = [refinst.lower() in i for i in insts]
        if any(mask):
            filts = list(np.array(filts)[mask])
            insts = list(np.array(insts)[mask])

    unique_filter_inst = list({
        f'{filt}_{inst}' for filt, inst in zip(filts, insts)
    })

    # Prefer not to build a reference from ACS/HRC when other options exist.
    if any('hrc' not in val for val in unique_filter_inst):
        unique_filter_inst = [val for val in unique_filter_inst if 'hrc' not in val]

    if avoid_wfpc2 and any('wfpc2' not in val for val in unique_filter_inst):
        unique_filter_inst = [
            val for val in unique_filter_inst if 'wfpc2' not in val
        ]

    total_exposure = []
    for val in unique_filter_inst:
        exposure = 0.0
        for im in images:
            if (
                get_filter(im) in val
                and get_instrument(im).split('_')[0] in val
            ):
                exposure += float(fits.getval(im, 'EFFEXPTM'))
        total_exposure.append(exposure)

    best_filt_inst = ''
    best_exposure = 0.0

    for filt in best_filters:
        for v in (s for s in unique_filter_inst if filt in s):
            exposure = total_exposure[unique_filter_inst.index(v)]
            if exposure > best_exposure:
                best_filt_inst = v
                best_exposure = exposure

    if not best_filt_inst:
        for filt_type in best_types:
            for v in (s for s in unique_filter_inst if filt_type in s):
                exposure = total_exposure[unique_filter_inst.index(v)]
                if exposure > best_exposure:
                    best_filt_inst = v
                    best_exposure = exposure

    reference_images = []
    for im in images:
        filt = get_filter(im)
        inst = get_instrument(im).replace('_full', '').replace('_sub', '')
        if f'{filt}_{inst}' == best_filt_inst:
            reference_images.append(im)
    return reference_images


def get_sky_pgons(table: Table) -> np.ndarray:
    """
    Parse ``S_REGION`` sky polygons from images listed in a table.

    Parameters
    ----------
    table : astropy.table.Table
        Table with an ``image`` column of FITS paths.

    Returns
    -------
    numpy.ndarray
        Object-dtype array of Shapely polygons (one per row).
    """
    pgons = []
    for im in table['image']:
        with fits.open(im) as hdul:
            region = hdul['SCI'].header['S_REGION']
        pgons.append(parse_s_region(region))
    return np.array(pgons, dtype=object)


def edit_visits_groups(table: Table) -> Table:
    """
    Renumber visits and assign spatial ``group`` IDs from footprint overlap.

    Groups are derived from the union of ``S_REGION`` polygons; visits spanning
    multiple groups are collapsed to the minimum group index.

    Parameters
    ----------
    table : astropy.table.Table
        Observation table with ``visit`` and ``image`` columns.

    Returns
    -------
    astropy.table.Table
        Copy of ``table`` with reindexed ``visit`` and new ``group`` column.
    """
    unique_visits = np.unique(table['visit'])
    visit_mapping = {old: new for new, old in enumerate(unique_visits)}
    table['visit'] = [visit_mapping[v] for v in table['visit']]

    table.add_column(Column(name='group', data=[None] * len(table)))
    pgons = get_sky_pgons(table)

    net_field = shapely.unary_union(pgons)
    if isinstance(net_field, shapely.geometry.polygon.Polygon):
        net_field = shapely.MultiPolygon([net_field])

    for i, component in enumerate(net_field.geoms):
        int_area = np.array([
            poly.intersection(component).area / poly.area for poly in pgons
        ])
        table['group'][int_area > 0] = i

    for visit in np.unique(table['visit']):
        groups = np.unique(table[table['visit'] == visit]['group'])
        if len(groups) > 1:
            table['group'][table['visit'] == visit] = min(groups)

    unique_groups = np.unique(table['group'])
    group_mapping = {old: new for new, old in enumerate(unique_groups)}
    table['group'] = [group_mapping[g] for g in table['group']]
    table.sort(['visit'])
    return table


def input_list(input_images: list[str | Path]) -> Table:
    """
    Build an observation metadata table from existing FITS paths.

    Reads exposure time, datetime, filter, instrument, module, chip, zero
    point, visit, and pupil from headers, then assigns visits/groups via
    :func:`edit_visits_groups`.

    Parameters
    ----------
    input_images : list of str or Path
        Science FITS paths; missing files are skipped.

    Returns
    -------
    astropy.table.Table
        Sorted observation table with columns ``image``, ``exptime``,
        ``datetime``, ``filter``, ``instrument``, ``module``, ``zeropoint``,
        ``chip``, ``imagenumber``, ``visit``, and ``pupil``.

    Raises
    ------
    ValueError
        When no input paths exist on disk.
    """
    img = [str(image) for image in input_images if os.path.exists(image)]
    if not img:
        raise ValueError(
            'No existing input images found. Check --base-dir points at the '
            'dataset root (…/<object>) or reduction workdir containing jhat/.'
        )

    with fits.open(img[0]) as hdu:
        primary = hdu[0].header

    exp = [fits.getval(image, 'EFFEXPTM') for image in img]
    if 'DATE-OBS' in primary and 'TIME-OBS' in primary:
        dat = [
            f"{fits.getval(image, 'DATE-OBS')}T{fits.getval(image, 'TIME-OBS')}"
            for image in img
        ]
    elif 'EXPSTART' in primary:
        dat = [
            Time(fits.getval(image, 'EXPSTART'), format='mjd').datetime.strftime(
                '%Y-%m-%dT%H:%M:%S'
            )
            for image in img
        ]
    else:
        raise ValueError(
            f'Cannot determine observation time from headers of {img[0]}'
        )

    fil = [get_filter(image) for image in img]
    ins = [get_instrument(image) for image in img]
    module = [get_module(image) for image in img]
    chip = [get_chip(image) for image in img]
    zpt = [get_zpt(i, ccdchip=c, zptype='abmag') for i, c in zip(img, chip)]
    visit = [fits.getval(i, 'VISIT_ID', ext=0) for i in img]
    pupil = [fits.getval(i, 'PUPIL', ext=0) for i in img]
    image_number = [0] * len(img)

    obstable = Table(
        [img, exp, dat, fil, ins, module, zpt, chip, image_number, visit, pupil],
        names=[
            'image', 'exptime', 'datetime', 'filter', 'instrument', 'module',
            'zeropoint', 'chip', 'imagenumber', 'visit', 'pupil',
        ],
    )
    obstable.sort('datetime')
    return edit_visits_groups(obstable)


def create_filter_table(tables: Table, filters: list[str]) -> dict[str, Table]:
    """
    Create a dictionary of (filter, table) pairs for the full observation set.

    Parameters
    ----------
    tables : astropy.table.Table
        Observation table with a ``filter`` column.
    filters : list of str
        Filter names to extract.

    Returns
    -------
    dict
        Mapping of filter name to subtable.
    """
    return {flt: tables[tables['filter'] == flt] for flt in filters}


def xmatch_common(
    skycrd_1: SkyCoord,
    skycrd_2: SkyCoord,
    dist_limit: float = 5.0,
) -> pd.DataFrame:
    """
    Cross-match sources between two SkyCoord objects.

    Parameters
    ----------
    skycrd_1 : SkyCoord
        Positions of sources in catalog 1.
    skycrd_2 : SkyCoord
        Positions of sources in catalog 2.
    dist_limit : float, optional
        Maximum distance for a match, in arcsec.

    Returns
    -------
    pandas.DataFrame
        Columns ``idx_1``, ``idx_2``, and ``d2d`` (arcsec).
    """
    idx, d2d, _d3d = skycrd_1.match_to_catalog_sky(skycrd_2)
    xmatch_df = pd.DataFrame(
        {
            'idx_1': np.arange(len(skycrd_1)),
            'idx_2': idx,
            'd2d': d2d.to(u.arcsec).value,
        }
    )
    matched_df = xmatch_df.loc[xmatch_df.groupby('idx_2').d2d.idxmin()]
    return matched_df[matched_df['d2d'] < dist_limit]
