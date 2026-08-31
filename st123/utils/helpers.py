"""Shared helpers: coordinates, FITS bookkeeping, visits, and cross-matching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Column, Table

from st123.datamodels import HSTDataModel, InstrumentDataModel

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


# Quotes that bash does not strip (Unicode curly/smart quotes) and ASCII
# quotes sometimes left inside exported shell values (issue #3).
_COORD_QUOTE_CHARS = (
    '"',
    "'",
    '\u201c',  # "
    '\u201d',  # "
    '\u2018',  # '
    '\u2019',  # '
    '\u00ab',  # <<
    '\u00bb',  # >>
)


def _normalize_coord_token(value: str | float) -> str | float:
    """Strip surrounding whitespace/quotes from a coordinate token."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    # Peel matching or mixed quote wrappers; also drop stray leading/trailing
    # curly quotes that shell exports leave inside the value.
    while text and text[0] in _COORD_QUOTE_CHARS:
        text = text[1:].lstrip()
    while text and text[-1] in _COORD_QUOTE_CHARS:
        text = text[:-1].rstrip()
    return text.strip()


def parse_coord(ra: str | float, dec: str | float) -> SkyCoord | None:
    """
    Parse RA/Dec into an ICRS :class:`~astropy.coordinates.SkyCoord`.

    Accepts decimal degrees or sexagesimal strings (``':'`` in both values).
    Surrounding ASCII or curly/smart quotes are stripped so shell exports like
    ``export RA="09:53:42.00"`` (Unicode quotes) still parse.

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
    ra = _normalize_coord_token(ra)
    dec = _normalize_coord_token(dec)

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


def get_filter(image) -> str:
    """
    Read the filter name from a science datamodel.

    Parameters
    ----------
    image
        :class:`~st123.datamodels.instrument.InstrumentDataModel` or FITS path.

    Returns
    -------
    str
        Lowercase filter name (``FILTNAM1``, ``FILTER``, or ``FILTER1``/``FILTER2``;
        ``PHOTMODE`` as a last resort for stripped JHAT products).
    """
    from st123.datamodels.instrument import as_datamodel

    return as_datamodel(image).filter_name


def get_module(image) -> str:
    """
    Read the JWST module identifier from a science datamodel.

    NIRCam exposures use ``MODULE`` (``A`` / ``B``) or the letter embedded in
    ``DETECTOR`` (``NRCB1`` -> ``b``). MIRI has no module keyword and is
    returned as ``miri``.

    Parameters
    ----------
    image
        Datamodel or FITS path.

    Returns
    -------
    str
        Lowercase module identifier (``a``, ``b``, ``miri``, ...).
    """
    from st123.datamodels.instrument import as_datamodel

    return as_datamodel(image).module_name


def get_instrument(image) -> str:
    """
    Read the instrument name from a science datamodel.

    Parameters
    ----------
    image
        Datamodel or FITS path.

    Returns
    -------
    str
        Lowercase ``INSTRUME`` value (or ``PHOTMODE`` / ``APERTURE`` fallback).
    """
    from st123.datamodels.instrument import as_datamodel

    return as_datamodel(image).instrument_name


def get_chip(image) -> int | str:
    """
    Return the detector chip identifier from a science datamodel.

    For multi-extension HST images, ambiguous cases default to chip ``1``.
    JWST products typically expose ``DETECTOR`` instead of ``CCDCHIP``.

    Parameters
    ----------
    image
        Datamodel or FITS path.

    Returns
    -------
    int or str
        Chip number or detector name; ``1`` when no chip keyword is found.
    """
    from st123.datamodels.instrument import as_datamodel

    return as_datamodel(image).chip


def get_detector_chip(filename) -> str | None:
    """
    Get the detector chip token from a file name.

    Parameters
    ----------
    filename
        Datamodel, file name, or path.

    Returns
    -------
    str or None
        Detector chip (e.g. ``nrcb1``, ``nrcblong``, ``mirimage``), or ``None``.
    """
    from st123.datamodels.instrument import InstrumentDataModel, _filename_chip

    if isinstance(filename, InstrumentDataModel):
        return filename.filename_chip
    return _filename_chip(filename)


# Full-frame MIRI imager SCI size (ny, nx). Subarrays / cutouts are unsupported
# by ``mirimask`` and by the alignment->DOLPHOT path.
MIRI_FULL_FRAME_SCI_SHAPE = (1024, 1032)


def is_full_frame_miri(image) -> bool:
    """
    Return whether a science datamodel is a full-frame MIRI imager product.

    Accepts SCI arrays of shape ``(1024, 1032)`` or ``(1, 1024, 1032)``.
    Explicit non-``FULL`` ``SUBARRAY`` header values are rejected even when
    dimensions are ambiguous. Non-MIRI files return ``False``.

    Parameters
    ----------
    image
        Datamodel or FITS path.

    Returns
    -------
    bool
        ``True`` when the file is usable as a full-frame MIRI imager frame.
    """
    from st123.datamodels.instrument import as_datamodel
    from st123.datamodels.jwst.miri import MIRIDataModel

    try:
        model = as_datamodel(image)
    except Exception:
        return False
    if not isinstance(model, MIRIDataModel):
        return False
    try:
        return bool(model.is_full_frame())
    except Exception:
        return False


def is_mirimask_compatible(image) -> bool:
    """Alias for :func:`is_full_frame_miri` (DOLPHOT ``mirimask`` requirement)."""
    return is_full_frame_miri(image)


def get_zpt(
    image,
    ccdchip: int | str = 1,
    zptype: str = 'abmag',
) -> float | None:
    """
    Compute the photometric zero point from ``PHOTFLAM`` and ``PHOTPLAM``.

    Parameters
    ----------
    image
        Datamodel or FITS path.
    ccdchip : int or str, optional
        CCD chip or detector name when multiple science HDUs are present.
    zptype : str, optional
        ``'abmag'`` for AB magnitude or ``'st'`` for ST magnitude.

    Returns
    -------
    float or None
        Zero point in magnitudes, or ``None`` when header keywords are missing.
    """
    from st123.datamodels.instrument import as_datamodel

    return as_datamodel(image).zeropoint(chip=ccdchip, zptype=zptype)


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
    images : list of datamodel or path-like
        Science products to consider.
    reffilter : str or None, optional
        Force a specific filter (must appear in
        :meth:`InstrumentDataModel.all_filters`).
    avoid_wfpc2 : bool, optional
        Exclude WFPC2 instrument/filter pairs when alternatives exist.
    refinst : str or None, optional
        Restrict candidates to this instrument substring.

    Returns
    -------
    list of str
        Paths of images in the best instrument/filter group.
    """
    from st123.datamodels.instrument import as_datamodel, path_of

    known = InstrumentDataModel.all_filters()
    best_filters = list(HSTDataModel.BEST_REFERENCE_FILTERS)
    if reffilter and reffilter.upper() in known:
        best_filters = [reffilter.lower()]
    best_types = list(InstrumentDataModel.BEST_FILTER_TYPES)

    models = [as_datamodel(im) for im in images]
    filts = [m.filter_name for m in models]
    insts = [
        m.instrument_name.replace('_full', '').replace('_sub', '')
        for m in models
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
        for model in models:
            filt = model.filter_name
            inst = model.instrument_name.split('_')[0]
            if filt in val and inst in val:
                exp = model.exptime
                if exp is None:
                    raise KeyError(f'No exposure time in {path_of(model)}')
                exposure += float(exp)
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
    for im, model in zip(images, models):
        filt = model.filter_name
        inst = model.instrument_name.replace('_full', '').replace('_sub', '')
        if f'{filt}_{inst}' == best_filt_inst:
            reference_images.append(im)
    return reference_images


def get_sky_pgons(table: Table) -> np.ndarray:
    """
    Parse ``S_REGION`` sky polygons from images listed in a table.

    When ``S_REGION`` is missing, fall back to the SCI WCS footprint
    (``WCS.calc_footprint``) so mixed JWST/HST planning still works.

    Parameters
    ----------
    table : astropy.table.Table
        Table with an ``image`` column of datamodels or FITS paths.

    Returns
    -------
    numpy.ndarray
        Object-dtype array of Shapely polygons (one per row).
    """
    from st123.datamodels.instrument import as_datamodel

    pgons = []
    for im in table['image']:
        pgons.append(as_datamodel(im).sky_polygon())
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


def input_list(input_images: list) -> Table:
    """
    Build an observation metadata table from existing science datamodels.

    Reads exposure time, datetime, filter, instrument, module, chip, zero
    point, visit, and pupil from each datamodel, then assigns visits/groups
    via :func:`edit_visits_groups`.

    Parameters
    ----------
    input_images : list of datamodel or path-like
        Science products; missing files are skipped.

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
    from st123.datamodels.instrument import as_datamodel, path_of

    models = []
    for image in input_images:
        model = as_datamodel(image)
        if model.path.exists():
            models.append(model)
    if not models:
        raise ValueError(
            'No existing input images found. Check --base-dir points at the '
            'dataset root (.../<object>) or reduction workdir containing jhat/.'
        )

    img = [str(path_of(m)) for m in models]
    exp = [m.exptime if m.exptime is not None else 0.0 for m in models]
    dat = [m.obs_datetime for m in models]
    fil = [m.filter_name for m in models]
    ins = [m.instrument_name for m in models]
    module = [m.module_name for m in models]
    chip = [m.chip for m in models]
    zpt = [m.zeropoint(chip=c, zptype='abmag') for m, c in zip(models, chip)]
    visit = [m.visit_id for m in models]
    pupil = [m.pupil for m in models]
    image_number = [0] * len(models)

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
