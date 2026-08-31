#!/usr/bin/env python3
"""Build mosaics / coadds from JHAT-aligned frames (no DOLPHOT prep)."""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path

from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    add_filters,
    add_instruments_arg,
    add_sky_coord_args,
    configure_logging_from_args,
    create_parser as build_parser,
    dataset_label,
    expand_mission_instruments,
    parse_filter_list,
    parse_instruments,
    resolve_instruments_with_telescope,
    resolve_reduction_dir,
)
from st123.utils.logging import shutdown_logging
from st123.datamodels import MIRIDataModel, NIRCamDataModel, as_datamodel

logger = logging.getLogger(__name__)

_JWST_MOSAIC_INSTRUMENTS = frozenset({'nircam', 'nrc', 'miri'})
_HST_MOSAIC_INSTRUMENTS = frozenset({'acs', 'wfc3', 'wfpc2', 'wfc'})


def create_parser():
    parser = build_parser(
        description=(
            'Mosaic / coadd JHAT frames (JWST resample or HST AstroDrizzle) into '
            'a shared reference/group_*/ref_* box layout. With --instruments '
            'NIRCAM MIRI ACS WFC3 (or ALL), plans boxes once from combined '
            'footprints then mosaics each mission into those directories. '
            'DOLPHOT staging is a separate step: dolphot-prep.'
        ),
    )
    parser.add_argument(
        '--telescope',
        choices=('jwst', 'hst'),
        default=None,
        help=(
            'Mission shorthand, equivalent to the default --instruments set: '
            'hst -> ACS WFC3 WFPC2; jwst -> NIRCAM MIRI. When omitted, mosaics '
            'JWST JHAT frames onto shared stamps (or use --instruments).'
        ),
    )
    add_base_dir(
        parser,
        required=False,
        default='.',
        help=(
            'Project root (.../<object>) or reduction workdir. Coadds are '
            'written under <reduction>/reference/. Aliases: --basedir, '
            '--workdir, --data-dir.'
        ),
    )
    parser.add_argument(
        '--nmax',
        type=int,
        default=150,
        help=(
            'Maximum number of overlapping science images per mosaic box '
            '(default: 150). Applies to the shared JWST+HST box planner. '
            'With --footprint-weights auto, footprints from missions not in '
            'the science set also count toward this limit.'
        ),
    )
    parser.add_argument(
        '--full-group',
        action='store_true',
        help=(
            'Skip overlap box-splitting (ref_0, ref_1, ...). For each group, '
            'mosaic every JHAT frame of each filter into one coadd under '
            'reference/group_<G>/ref_full/ (JWST: coadd_<G>_full_<filter>_i2d.fits; '
            'HST: coadd_<G>_full_<inst>_<filter>_drc.fits).'
        ),
    )
    parser.add_argument(
        '--spec_groups',
        type=str,
        default=None,
        help='List of images to be grouped into an extra mosaic group',
    )
    parser.add_argument(
        '--footprint-weights',
        choices=('auto', 'none'),
        default='auto',
        help=(
            'Box-split weighting. auto (default): also count overlapping '
            'footprints from missions not already in the science JHAT set '
            'toward --nmax. none: science JHAT only.'
        ),
    )
    add_instruments_arg(
        parser,
        help=(
            'Instruments to mosaic. Mission aliases: hst (= ACS WFC3 WFPC2), '
            'jwst (= NIRCAM MIRI), all (= NIRCAM MIRI ACS WFC3 WFPC2). Mixed '
            'JWST+HST lists share one group_*/ref_* plan then mosaic each '
            'mission into those boxes. Explicit --instruments overrides '
            '--telescope. Alias: --instrument.'
        ),
    )
    add_sky_coord_args(parser, required=False)
    parser.add_argument(
        '--force-miri',
        action='store_true',
        help=(
            'Mosaic MIRI even when the field has no NIRCam. By default, '
            'MIRI-only JWST fields skip JWST mosaic when NIRCAM+MIRI are '
            'requested.'
        ),
    )
    add_filters(
        parser,
        help=(
            'Comma-separated filter list to mosaic (e.g. F150W,F444W,F160W). '
            'When omitted: mosaic every JWST filter present in each shared '
            'stamp, and every HST instrument+filter group. When set, only '
            'matching JWST and/or HST filters are mosaicked onto the stamp.'
        ),
    )
    parser.add_argument(
        '--box-id',
        action='append',
        default=None,
        help=(
            'Only mosaic these box ids (repeatable), e.g. --box-id 5. '
            'Applied after shared planning so footprints stay consistent.'
        ),
    )
    parser.add_argument(
        '--contains-ra',
        type=float,
        default=None,
        help='Only mosaic boxes whose sky footprint contains this RA (deg).',
    )
    parser.add_argument(
        '--contains-dec',
        type=float,
        default=None,
        help='Only mosaic boxes whose sky footprint contains this Dec (deg).',
    )
    parser.add_argument(
        '--require-coverage',
        action='store_true',
        help=(
            'For auto-split ref_* boxes only: keep frames that cover a sky '
            'point (--contains-ra/dec or the stamp center). Implied already '
            'for --center-ra/dec custom stamps and --existing-box remosaics '
            '(those modes always drop frames outside the stamp FoV / center).'
        ),
    )
    parser.add_argument(
        '--require-coverage-ra',
        type=float,
        default=None,
        help=(
            'RA (deg) that every input frame must cover (with '
            '--require-coverage-dec). Overrides the implied stamp center for '
            '--center-ra/dec / --existing-box.'
        ),
    )
    parser.add_argument(
        '--require-coverage-dec',
        type=float,
        default=None,
        help=(
            'Dec (deg) that every input frame must cover (with '
            '--require-coverage-ra). Overrides the implied stamp center for '
            '--center-ra/dec / --existing-box.'
        ),
    )
    parser.add_argument(
        '--visit-coadds',
        action='store_true',
        help=(
            'HST only: also write visit-tagged sidecars '
            '(coadd_*_<visit>_drc/drz.fits). Default is one combined coadd '
            'per instrument+filter (all visits stacked), like NIRCam.'
        ),
    )
    parser.add_argument(
        '--drop-outlier-visits',
        action='store_true',
        help=(
            'HST only: drop cross-visit outliers from the primary '
            'instrument+filter stack. Recommended when remosaicing multi-epoch '
            'fields (e.g. --existing-box). Default keeps all visits in the '
            'combined coadd. Bad EXPFLAG frames are always excluded.'
        ),
    )
    add_common_runtime(parser, ncores=True, ncores_default=4, plot=False, verbose=True)
    return parser



def resolve_existing_box_path(
    base_dir: str | Path,
    existing_box: str | Path,
) -> Path:
    """
    Resolve ``--existing-box`` under a reduction workdir or project root.

    Relative values are tried as ``<base>/<path>``, ``<base>/reference/<path>``,
    and the ``reduction/`` variants of those (canonical ``group_*/ref_*``).
    Absolute paths are returned unchanged when they exist as directories.
    """
    from st123.stages.mosaic.mosaic import resolve_existing_box_dir

    return resolve_existing_box_dir(base_dir, existing_box)


def _resolve_coverage_sky(
    args,
    *,
    plan=None,
    box=None,
) -> tuple[float | None, float | None]:
    """
    Sky point that every JHAT frame must cover, or ``(None, None)``.

    Priority:

    1. Explicit ``--require-coverage-ra/dec``
    2. ``--center-ra/dec`` (implied for custom stamps)
    3. Stamp center when ``--existing-box`` remosaics a shared FoV
    4. Opt-in ``--require-coverage`` for auto-split boxes (uses
       ``--contains-ra/dec`` or the stamp center)

    Custom ``--center-ra/dec`` stamps and ``--existing-box`` remosaics always
    filter frames to the stamp FoV (and its center); ``--require-coverage`` is
    only needed for auto-planned ``ref_*`` boxes.
    """
    ra = getattr(args, 'require_coverage_ra', None)
    dec = getattr(args, 'require_coverage_dec', None)
    if ra is not None and dec is not None:
        return float(ra), float(dec)
    if (ra is None) ^ (dec is None):
        raise ValueError(
            '--require-coverage-ra and --require-coverage-dec must be given together'
        )

    cra = getattr(args, 'center_ra', None)
    cdec = getattr(args, 'center_dec', None)
    if cra is not None and cdec is not None:
        return float(cra), float(cdec)

    from st123.stages.mosaic.mosaic import stamp_sky_center

    def _stamp_center() -> tuple[float | None, float | None]:
        target = box
        if target is None and plan is not None and getattr(plan, 'boxes', None):
            target = plan.boxes[0]
        if target is not None and getattr(target, 'wcs', None) is not None:
            return stamp_sky_center(target.wcs)
        return None, None

    # --existing-box: match the on-disk stamp FoV; require its center.
    if getattr(args, 'existing_box', None):
        return _stamp_center()

    if not bool(getattr(args, 'require_coverage', False)):
        return None, None

    cra = getattr(args, 'contains_ra', None)
    cdec = getattr(args, 'contains_dec', None)
    if cra is not None and cdec is not None:
        return float(cra), float(cdec)

    return _stamp_center()


def _parse_stamp_size(raw: str | None) -> float | tuple[float, float] | None:
    """Parse ``--stamp-size`` into arcsec float or (width, height)."""
    if raw is None:
        return None
    text = str(raw).strip().lower().replace('"', '').replace("'", '')
    if not text:
        return None
    if ',' in text:
        a, b = text.split(',', 1)
        return (float(a), float(b))
    return float(text)


def _load_stamp_ref_wcs(path: str | Path):
    """Load a WCS from stamp_wcs.fits or a coadd SCI extension."""
    from astropy.wcs import WCS

    from st123.stages.mosaic.mosaic import STAMP_WCS_BASENAME, load_stamp_wcs

    p = Path(path)
    if p.is_dir():
        return load_stamp_wcs(p)
    if p.name == STAMP_WCS_BASENAME or p.suffix.lower() == '.fits':
        with as_datamodel(p).open() as hdul:
            if 'SCI' in hdul:
                sci = hdul['SCI']
                w = WCS(sci.header, naxis=2)
                ny, nx = sci.data.shape[-2:]
            else:
                w = WCS(hdul[0].header, naxis=2)
                nx = int(hdul[0].header.get('MOSNX') or hdul[0].header.get('NAXIS1') or 0)
                ny = int(hdul[0].header.get('MOSNY') or hdul[0].header.get('NAXIS2') or 0)
                if (nx <= 0 or ny <= 0) and hdul[0].data is not None:
                    ny, nx = hdul[0].data.shape[-2:]
            if nx > 0 and ny > 0:
                w.pixel_shape = (nx, ny)
                w._naxis = [nx, ny]
            return w
    raise FileNotFoundError(f'Cannot load stamp reference WCS from {path}')


def _filter_plan_boxes(plan, args):
    """Restrict *plan.boxes* by --box-id and/or --contains-ra/dec."""
    from shapely.geometry import Point

    from st123.stages.mosaic.mosaic import stamp_sky_polygon

    boxes = list(plan.boxes)
    box_ids = getattr(args, 'box_id', None)
    ra = getattr(args, 'contains_ra', None)
    dec = getattr(args, 'contains_dec', None)

    # Sky selection is authoritative when ids and sky disagree.
    if ra is not None and dec is not None:
        kept = []
        pt = Point(float(ra), float(dec))
        for box in boxes:
            if box.wcs is None:
                continue
            try:
                poly = stamp_sky_polygon(box.wcs)
            except Exception:
                continue
            if poly.contains(pt) or poly.intersects(pt.buffer(1e-8)):
                kept.append(box)
        if box_ids:
            want = {str(b) for b in box_ids}
            by_id = [b for b in kept if str(b.box_id) in want]
            if by_id:
                kept = by_id
            elif kept:
                logger.warning(
                    'Box id(s) %s do not match the box containing '
                    '(RA,Dec)=(%.6f,%.6f); using sky match box_id=%s',
                    box_ids,
                    float(ra),
                    float(dec),
                    [b.box_id for b in kept],
                )
        boxes = kept
    elif box_ids:
        want = {str(b) for b in box_ids}
        boxes = [b for b in boxes if str(b.box_id) in want]

    if len(boxes) != len(plan.boxes):
        logger.info(
            'Box filter: keeping %d/%d box(es)%s%s',
            len(boxes),
            len(plan.boxes),
            f' box_id={box_ids}' if box_ids else '',
            f' contains=({ra},{dec})' if ra is not None else '',
        )
    plan.boxes = boxes
    return plan


def resolve_mosaic_instruments(
    instruments_raw: list[str] | None,
) -> list[str] | None:
    """Normalize ``--instruments``; expand ``hst`` / ``jwst`` / ``all``."""
    if not instruments_raw:
        return None
    return expand_mission_instruments(parse_instruments(instruments_raw))


def partition_mosaic_instruments(
    instruments: list[str],
) -> tuple[list[str], list[str]]:
    """Split into (jwst_instruments, hst_instruments)."""
    jwst: list[str] = []
    hst: list[str] = []
    unknown: list[str] = []
    for inst in instruments:
        key = str(inst).split('_')[0].strip().lower()
        name = str(inst).strip().upper()
        if key == 'nrc':
            name = 'NIRCAM'
            key = 'nircam'
        if key in _JWST_MOSAIC_INSTRUMENTS:
            if name not in jwst:
                jwst.append(name)
        elif key in _HST_MOSAIC_INSTRUMENTS:
            if name not in hst:
                hst.append(name)
        else:
            unknown.append(str(inst))
    if unknown:
        raise ValueError(
            'Unrecognized mosaic instrument(s): '
            f'{unknown}; expected NIRCAM, MIRI, ACS, WFC3, WFPC2 (or ALL)'
        )
    return jwst, hst


def default_jwst_filters_for_instruments(jwst_instruments: list[str]) -> list[str]:
    """Dual-mode default JWST filter list for requested instruments."""
    filters: list[str] = []
    upper = {str(i).upper() for i in jwst_instruments}
    if 'NIRCAM' in upper or 'NRC' in upper:
        filters.extend(NIRCamDataModel.MOSAIC_FILTERS)
    if 'MIRI' in upper:
        filters.extend(MIRIDataModel.MOSAIC_FILTERS)
    # Preserve order, drop dupes
    out: list[str] = []
    for f in filters:
        fl = f.lower()
        if fl not in out:
            out.append(fl)
    return out


def needs_mosaic_orchestration(instruments: list[str] | None) -> bool:
    """True when --instruments requests a multi-mission or explicit HST/JWST set."""
    if not instruments:
        return False
    try:
        jwst, hst = partition_mosaic_instruments(instruments)
    except ValueError:
        return False
    # Any explicit multi-instrument list uses the orchestrator so defaults apply.
    return bool(jwst) or bool(hst)


def _collect_mission_jhat(
    base_dir: Path,
    mission: str,
) -> list[str]:
    """Absolute JHAT paths for ``jwst`` or ``hst`` under *base_dir*."""
    from st123.scripts.utils.options import resolve_jhat_dir

    jhat_dir = resolve_jhat_dir(base_dir, mission, create=False)
    if not jhat_dir.is_dir():
        return []
    paths = sorted(glob.glob(os.path.join(str(jhat_dir), '*jhat.fits')))
    if mission == 'hst':
        paths = [
            p
            for p in paths
            if not os.path.basename(p).lower().startswith('jw')
            and not os.path.basename(p).lower().startswith('coadd_')
        ]
    return [str(Path(p).resolve()) for p in paths]


def _run_hst_mosaic(
    args,
    *,
    instruments: list[str] | None = None,
    ncores: int | None = None,
    plan=None,
    filters: list[str] | None = None,
) -> int:
    """AstroDrizzle HST frames into ``reference/group_*/ref_*`` boxes."""
    from st123.stages.mosaic.hst_drizzle import drizzle_project, drizzle_project_boxed
    from st123.utils.logging import _quiet_third_party_loggers

    _quiet_third_party_loggers()

    from st123.scripts.utils.options import resolve_jhat_dir

    base_dir = Path(resolve_reduction_dir(args.base_dir))
    jhat_dir = resolve_jhat_dir(base_dir, 'hst', create=False)
    if instruments is None:
        instruments = parse_instruments(getattr(args, 'instruments', None))
    cores = int(args.ncores if ncores is None else ncores)
    if filters is None:
        filters = parse_filter_list(getattr(args, 'filters', None))
        if filters:
            filters = [f.lower() for f in filters]
    if args.verbose:
        logger.info('Dataset: %s', dataset_label(args.base_dir))
        logger.info(
            'HST drizzle: %s -> %s/group_*/ref_*',
            jhat_dir,
            base_dir / 'reference',
        )
        if instruments:
            logger.info('HST instruments: %s', ', '.join(instruments))
        if filters:
            logger.info('HST filters: %s', ', '.join(filters))
    if not jhat_dir.is_dir():
        logger.error('missing jhat_hst/ or jhat/ under %s', base_dir)
        return 1

    try:
        if plan is None:
            results = drizzle_project(
                jhat_dir,
                base_dir / 'reference',
                num_cores=cores,
                instruments=instruments,
                filters=filters,
                nmax=int(getattr(args, 'nmax', 150) or 150),
                full_group=bool(getattr(args, 'full_group', False)),
                footprint_weights=str(
                    getattr(args, 'footprint_weights', 'auto')
                ),
                drop_outlier_visits=bool(
                    getattr(args, 'drop_outlier_visits', False)
                ),
                visit_coadds=bool(getattr(args, 'visit_coadds', False)),
                require_coverage_ra=getattr(args, '_cov_ra', None),
                require_coverage_dec=getattr(args, '_cov_dec', None),
            )
        else:
            results = drizzle_project_boxed(
                plan,
                instruments=instruments,
                filters=filters,
                num_cores=cores,
                raise_on_unify_fail=False,
                assume_aligned=True,
                drop_outlier_visits=bool(
                    getattr(args, 'drop_outlier_visits', False)
                ),
                visit_coadds=bool(getattr(args, 'visit_coadds', False)),
                require_coverage_ra=getattr(args, '_cov_ra', None),
                require_coverage_dec=getattr(args, '_cov_dec', None),
            )
    except RuntimeError as exc:
        logger.error('%s', exc)
        return 1

    if not results:
        return 1
    n_ok = sum(1 for r in results if r['status'] == 'ok')
    n_fail = sum(1 for r in results if r['status'] != 'ok')
    for r in results:
        if r['status'] == 'ok':
            logger.info(
                'OK %s/%s -> %s (%d frames)',
                r['instrument'],
                r['filter'],
                r['output'],
                len(r['frames']),
            )
        else:
            logger.error(
                'FAIL %s/%s: %s',
                r['instrument'],
                r['filter'],
                r.get('error'),
            )
    logger.info('HST drizzle done: %d ok, %d failed', n_ok, n_fail)
    return 0 if n_ok and n_fail == 0 else (0 if n_ok else 1)


def _box_wcs_header(mosaic_wcs, bbox, *, box=None):
    """Resolve the shared stamp WCS (and header) for one mosaic box."""
    from st123.stages.mosaic.mosaic import (
        _ensure_pc_cdelt_header,
        ensure_box_stamp_wcs,
        slice_box_wcs,
    )

    if box is not None:
        box_wcs = ensure_box_stamp_wcs(box, mosaic_wcs=mosaic_wcs, bbox=bbox)
    else:
        box_wcs = slice_box_wcs(mosaic_wcs, bbox)
    wcs_hdr = _ensure_pc_cdelt_header(box_wcs.to_header(), box_wcs)
    naxis1, naxis2 = box_wcs.pixel_shape or (
        int(box_wcs._naxis[0]),
        int(box_wcs._naxis[1]),
    )
    wcs_hdr['NAXIS1'], wcs_hdr['NAXIS2'] = int(naxis1), int(naxis2)
    return box_wcs, wcs_hdr


def _forced_filter_tables(reftable, forced_filters: list[str]) -> dict:
    """Build a non-empty filter->table map for requested filters present in box."""
    from st123.utils.helpers import create_filter_table

    available = {str(f).lower() for f in reftable['filter']}
    selected = [f for f in forced_filters if f in available]
    missing = [f for f in forced_filters if f not in available]
    if missing:
        logger.warning(
            'Requested filters not present in this box (skipping): %s',
            ', '.join(missing),
        )
    if not selected:
        return {}
    filter_table = create_filter_table(reftable, selected)
    return {k: v for k, v in filter_table.items() if len(v) > 0}


def _run_default_sw_coadd(
    *,
    filter_table: dict,
    box_outdir: str,
    wcs_hdr,
    group_id: int,
    box_index: int | str,
    subimages,
    base_dir_arg,
    verbose: bool,
) -> str:
    """Optimal SW-filter selection path: PSF-match and write one coadd."""
    from st123.stages.mosaic.mosaic import (
        apply_wcs_to_coadd,
        coadd,
        convolve_images,
        copy_files,
        create_coadd_mosaic,
        create_gwcs,
        mosaic_coadd_basename,
        update_path,
        write_dolphot_frame_list,
    )

    filter_keys = sorted(filter_table.keys())
    target_filter = filter_keys[-1]

    gwcs_path = create_gwcs(outdir=box_outdir, sci_header=wcs_hdr)
    copy_files(filter_table, box_outdir)
    filter_table = update_path(filter_table, box_outdir)

    convolve_images(filter_table, target_filter)

    mosaics = []
    for filter_name in filter_keys:
        driz_image = create_coadd_mosaic(
            filter_table[filter_name],
            outdir=box_outdir,
            filt=filter_name,
            gwcs_file=gwcs_path,
        )
        mosaics.append(driz_image)

    coadd_filename = os.path.join(
        box_outdir,
        mosaic_coadd_basename(group_id, box_index, target_filter),
    )
    coadd(mosaics, target_filter, coadd_filename)
    apply_wcs_to_coadd(coadd_filename)

    write_dolphot_frame_list(
        box_outdir,
        refimage=coadd_filename,
        frames=subimages,
        group=int(group_id),
        box=box_index,
    )
    if verbose:
        logger.info(
            'Wrote coadd %s; run dolphot-prep --base-dir %s',
            coadd_filename,
            base_dir_arg,
        )
    return coadd_filename


def _run_forced_filter_coadds(
    *,
    filter_table: dict,
    box_wcs,
    box_outdir: str,
    group_id: int,
    box_index: int | str,
    subimages,
    base_dir_arg,
    verbose: bool,
    ncores: int = 1,
) -> list[str]:
    """
    Mosaic each requested filter onto the shared sky footprint.

    Pixel scale follows JWST channel recommendations; RA/Dec coverage matches
    ``box_wcs``. When ``ncores > 1``, per-filter ``Image3Pipeline`` runs are
    parallelized (each filter still uses a single Image3 process internally).
    """
    from st123.stages.mosaic.mosaic import (
        JWST_ALIGN_MAX_ARCSEC,
        _ensure_pc_cdelt_header,
        copy_files,
        mosaic_pixel_scale_arcsec,
        run_jwst_filter_image3_jobs,
        unify_jwst_astrometric_frame,
        update_path,
        write_dolphot_frame_list,
    )

    # Copy JHAT frames into the box (never mutate the global jhat/ store).
    # Alignment is owned by ``align``; mosaic only coadds onto the stamp WCS.
    out_path = Path(box_outdir)
    local_tables: dict = {}
    for filter_name, ftable in filter_table.items():
        single = {filter_name: ftable}
        copy_files(single, box_outdir)
        local_tables[filter_name] = update_path(single, box_outdir)[filter_name]

    wcs_header = _ensure_pc_cdelt_header(box_wcs.to_header(), box_wcs)
    naxis1, naxis2 = box_wcs.pixel_shape or (
        int(box_wcs._naxis[0]),
        int(box_wcs._naxis[1]),
    )
    wcs_header['NAXIS1'] = int(naxis1)
    wcs_header['NAXIS2'] = int(naxis2)

    jobs: list[dict] = []
    for filter_name in sorted(local_tables.keys()):
        ftable = local_tables[filter_name]
        inst = str(ftable['instrument'][0])
        pixscale = mosaic_pixel_scale_arcsec(filter_name, inst)
        jobs.append(
            {
                'filter_name': filter_name,
                'instrument': inst,
                'images': [str(p) for p in ftable['image']],
                'box_outdir': box_outdir,
                'group_id': int(group_id),
                'box_index': box_index,
                'wcs_header': list(wcs_header.cards),
                'pixel_scale': float(pixscale),
                'base_dir_arg': base_dir_arg,
                'verbose': bool(verbose),
            }
        )

    results = run_jwst_filter_image3_jobs(
        jobs, ncores=max(1, int(ncores or 1)), parallel=True
    )
    written: list[str] = []
    n_fail = 0
    for rec in results:
        if rec.get('status') == 'ok' and rec.get('coadd'):
            written.append(str(rec['coadd']))
        else:
            n_fail += 1
            logger.error(
                'JWST mosaic failed for filter %s: %s',
                rec.get('filter'),
                rec.get('error') or 'unknown',
            )
    if n_fail and not written:
        raise RuntimeError(
            f'All {n_fail} JWST filter Image3 job(s) failed in box {box_index}'
        )

    # One manifest per box: all JHAT frames, first coadd as DOLPHOT reference.
    if written:
        write_dolphot_frame_list(
            box_outdir,
            refimage=written[0],
            frames=subimages,
            group=int(group_id),
            box=box_index,
        )
        # Report pairwise residual only -- do not remosaic / shift coadds here.
        unify = unify_jwst_astrometric_frame(
            box_outdir,
            max_residual_arcsec=JWST_ALIGN_MAX_ARCSEC,
            remosaic=False,
            box_wcs=box_wcs,
            apply_shifts=False,
        )
        max_mas = 1000.0 * float(unify.get('final_max_abs_arcsec') or 0.0)
        if not unify.get('ok'):
            logger.warning(
                'JWST coadd residual for box %s (max |Delta|=%.1f mas); '
                'coadds kept (align owns frame) -- see %s',
                box_index,
                max_mas,
                unify.get('qa_path'),
            )
        elif verbose:
            logger.info(
                'JWST coadd QA OK for box %s (max |Delta|=%.1f mas)',
                box_index,
                max_mas,
            )
    return written


def _run_jwst_mosaic(
    args,
    *,
    forced_filters: list[str] | None = None,
    ncores: int | None = None,
    plan=None,
) -> int:
    """JWST resample coadds into ``reference/group_*/ref_*`` boxes."""
    import numpy as np

    from st123.stages.mosaic.mosaic import (
        plan_mosaic_boxes,
        split_observations,
    )
    from st123.scripts.utils.options import resolve_jhat_dir
    from st123.utils.helpers import input_list

    cores = int(args.ncores if ncores is None else ncores)

    base_dir_path = Path(resolve_reduction_dir(args.base_dir))
    base_dir = str(base_dir_path)
    jhat_dir = str(resolve_jhat_dir(base_dir_path, 'jwst', create=False))
    nmax = args.nmax
    spec_group_file = args.spec_groups
    if forced_filters is None:
        forced_filters = parse_filter_list(args.filters)
        if forced_filters:
            forced_filters = [f.lower() for f in forced_filters]
    else:
        forced_filters = [f.lower() for f in forced_filters]
    full_group = bool(getattr(args, 'full_group', False))
    footprint_weights = str(getattr(args, 'footprint_weights', 'auto')).lower()
    if args.verbose:
        logger.info('Dataset: %s', dataset_label(args.base_dir))
        logger.info('Reduction workdir: %s', base_dir)
        logger.info('JHAT indir: %s', jhat_dir)
        if full_group:
            logger.info('Box mode: full-group (ref_full, no overlap split)')
        if forced_filters:
            logger.info('Forced filters: %s', ', '.join(forced_filters))
        else:
            logger.info(
                'Filter mode: all JWST filters present in each shared stamp'
            )

    if plan is None:
        inputfiles = _collect_mission_jhat(base_dir_path, 'jwst')
        if not inputfiles:
            logger.error(
                'no *jhat.fits under %s. '
                'Pass --base-dir to the dataset root (.../<object>) or the '
                'reduction workdir that contains jhat_jwst/ or jhat/.',
                jhat_dir,
            )
            return 1
        plan = plan_mosaic_boxes(
            base_dir_path,
            inputfiles,
            nmax=nmax,
            full_group=full_group,
            footprint_weights=footprint_weights,
            spec_group_file=spec_group_file,
            verbose=bool(args.verbose),
        )
        n_before = len(plan.boxes)
        plan = _filter_plan_boxes(plan, args)
        filtered = bool(getattr(args, 'box_id', None)) or (
            getattr(args, 'contains_ra', None) is not None
            and getattr(args, 'contains_dec', None) is not None
        )
        if filtered and n_before and not plan.boxes:
            logger.error('No mosaic boxes remain after --box-id / --contains-* filter')
            return 1

    for box in plan.boxes:
        jwst_frames = box.frames_for_mission('jwst')
        if not jwst_frames:
            continue
        try:
            cov_ra = getattr(args, '_cov_ra', None)
            cov_dec = getattr(args, '_cov_dec', None)
            if cov_ra is None and cov_dec is None:
                cov_ra, cov_dec = _resolve_coverage_sky(args, plan=plan, box=box)
        except ValueError as exc:
            logger.error('%s', exc)
            return 2
        if cov_ra is not None and cov_dec is not None:
            from st123.stages.mosaic.mosaic import filter_frames_covering_point

            before = len(jwst_frames)
            jwst_frames = filter_frames_covering_point(
                jwst_frames, float(cov_ra), float(cov_dec)
            )
            if args.verbose:
                logger.info(
                    'JWST coverage filter group %s box %s: %d -> %d frame(s)',
                    box.group_id,
                    box.box_id,
                    before,
                    len(jwst_frames),
                )
            if not jwst_frames:
                logger.warning(
                    'No JWST frames cover RA=%.6f Dec=%.6f in box %s; skipping',
                    cov_ra,
                    cov_dec,
                    box.box_id,
                )
                continue
        box_table = input_list(jwst_frames)
        # Default: every filter in the shared stamp on one sky grid + unify.
        group_forced = forced_filters
        if group_forced is None:
            group_forced = sorted(
                {str(f).lower() for f in box_table['filter']}
            )
            if args.verbose:
                logger.info(
                    'Shared-stamp filters for group %s box %s: %s',
                    box.group_id,
                    box.box_id,
                    ', '.join(group_forced),
                )

        filter_table = _forced_filter_tables(box_table, group_forced)

        if not filter_table:
            logger.warning(
                'No mosaickable JWST filters for group %s box %s; skipping',
                box.group_id,
                box.box_id,
            )
            continue

        mosaic_wcs = box.wcs
        if mosaic_wcs is None:
            split_local = split_observations(table=box_table, N_max=10**9)
            split_local.as_full_group()
            mosaic_wcs = split_local.wcs
            bbox = split_local.split_boxes[0]
        else:
            bbox = box.bbox

        box_outdir = str(box.outdir)
        os.makedirs(box_outdir, exist_ok=True)
        box_wcs, _wcs_hdr = _box_wcs_header(mosaic_wcs, bbox, box=box)

        _run_forced_filter_coadds(
            filter_table=filter_table,
            box_wcs=box_wcs,
            box_outdir=box_outdir,
            group_id=int(box.group_id),
            box_index=box.box_id,
            subimages=jwst_frames,
            base_dir_arg=args.base_dir,
            verbose=args.verbose,
            ncores=cores,
        )
    return 0


def run_orchestrated_mosaic(
    args: argparse.Namespace,
    instruments: list[str],
) -> int:
    """
    Multi-mission mosaic into a shared ``group_*/ref_*`` plan.

    Plans boxes once from the combined JWST+HST JHAT set, then runs JWST and
    HST legs sequentially into those directories (avoids nested-process deadlock).
    """
    from st123.datamodels import JWSTDataModel, MIRIDataModel, NIRCamDataModel
    from st123.stages.mosaic.mosaic import plan_mosaic_boxes

    try:
        jwst_inst, hst_inst = partition_mosaic_instruments(instruments)
    except ValueError as exc:
        logger.error('%s', exc)
        return 2

    base_dir = Path(resolve_reduction_dir(args.base_dir))
    force_miri = bool(getattr(args, 'force_miri', False))
    want_nircam = any(NIRCamDataModel.matches(i) for i in jwst_inst)
    want_miri = any(MIRIDataModel.matches(i) for i in jwst_inst)
    data_root = Path(args.base_dir).expanduser().resolve()
    n_nrc, n_miri = JWSTDataModel.coverage_under(
        data_root / 'download' / 'JWST',
        data_root / 'reduction' / 'raw',
        data_root / 'raw',
    )
    if (
        jwst_inst
        and not force_miri
        and want_nircam
        and want_miri
        and n_miri > 0
        and n_nrc == 0
    ):
        logger.warning(
            'MIRI-only JWST field (NIRCam frames=%d, MIRI frames=%d); '
            'skipping JWST mosaic. Pass --force-miri to override.',
            n_nrc,
            n_miri,
        )
        jwst_inst = []
        if not hst_inst:
            return 0

    ncores = max(1, int(getattr(args, 'ncores', 4) or 4))
    user_filters = parse_filter_list(getattr(args, 'filters', None))
    # None -> every JWST filter present in each shared stamp (per-box).
    # Explicit --filters still restricts; curated defaults remain available via
    # default_jwst_filters_for_instruments() for callers that want that list.
    jwst_filters = None
    if jwst_inst and user_filters:
        jwst_filters = [f.lower() for f in user_filters]

    logger.info(
        'Orchestrated mosaic: instruments=%s (shared group_*/ref_* stamps)',
        ', '.join(instruments),
    )
    if jwst_inst:
        if jwst_filters:
            logger.info('JWST filters: %s', ', '.join(jwst_filters))
        else:
            logger.info(
                'JWST filters: all filters present in each shared stamp'
            )
    if hst_inst:
        logger.info('HST instruments: %s', ', '.join(hst_inst))

    science_files: list[str] = []
    if jwst_inst:
        science_files.extend(_collect_mission_jhat(base_dir, 'jwst'))
    if hst_inst:
        science_files.extend(_collect_mission_jhat(base_dir, 'hst'))
    # De-dupe preserving order
    seen: set[str] = set()
    uniq_files: list[str] = []
    for p in science_files:
        if p not in seen:
            seen.add(p)
            uniq_files.append(p)
    if not uniq_files:
        # No JWST JHAT after MIRI-only skip (or empty tree): soft-skip JWST-only.
        if not hst_inst and not jwst_inst:
            logger.warning(
                'No JHAT frames for mosaic under %s; skipping', base_dir
            )
            return 0
        logger.error(
            'No JHAT frames found for instruments %s under %s',
            instruments,
            base_dir,
        )
        return 1

    existing_box = getattr(args, 'existing_box', None)
    center_ra = getattr(args, 'center_ra', None)
    center_dec = getattr(args, 'center_dec', None)
    if existing_box and (center_ra is not None or center_dec is not None):
        logger.error('Use only one of --existing-box or --center-ra/--center-dec')
        return 2
    if (center_ra is None) ^ (center_dec is None):
        logger.error('--center-ra and --center-dec must be given together')
        return 2

    if existing_box:
        from st123.stages.mosaic.mosaic import plan_existing_box

        try:
            box_path = resolve_existing_box_path(base_dir, existing_box)
        except FileNotFoundError as exc:
            logger.error('%s', exc)
            return 2
        # Stamp FoV + center coverage are always applied inside plan_existing_box.
        # Optional --require-coverage-ra/dec overrides the stamp center.
        try:
            cov_ra, cov_dec = _resolve_coverage_sky(args)
        except ValueError as exc:
            logger.error('%s', exc)
            return 2
        plan = plan_existing_box(
            base_dir,
            box_path,
            uniq_files,
            require_coverage_ra=cov_ra,
            require_coverage_dec=cov_dec,
            verbose=bool(args.verbose),
        )
    elif center_ra is not None and center_dec is not None:
        from st123.stages.mosaic.mosaic import plan_centered_box

        ref_wcs = None
        stamp_ref = getattr(args, 'stamp_ref', None)
        if stamp_ref:
            ref_path = Path(stamp_ref)
            if not ref_path.is_absolute():
                ref_path = (base_dir / ref_path).resolve()
            ref_wcs = _load_stamp_ref_wcs(ref_path)
        plan = plan_centered_box(
            base_dir,
            uniq_files,
            float(center_ra),
            float(center_dec),
            size_arcsec=_parse_stamp_size(getattr(args, 'stamp_size', None)),
            box_id=str(getattr(args, 'stamp_id', None) or 'sn'),
            ref_wcs=ref_wcs,
            verbose=bool(args.verbose),
        )
    else:
        plan = plan_mosaic_boxes(
            base_dir,
            uniq_files,
            nmax=int(getattr(args, 'nmax', 150) or 150),
            full_group=bool(getattr(args, 'full_group', False)),
            footprint_weights=str(getattr(args, 'footprint_weights', 'auto')),
            spec_group_file=getattr(args, 'spec_groups', None),
            verbose=bool(args.verbose),
        )
        n_before = len(plan.boxes)
        plan = _filter_plan_boxes(plan, args)
        filtered = bool(getattr(args, 'box_id', None)) or (
            getattr(args, 'contains_ra', None) is not None
            and getattr(args, 'contains_dec', None) is not None
        )
        if filtered and n_before and not plan.boxes:
            logger.error(
                'No mosaic boxes remain after --box-id / --contains-* filter'
            )
            return 1
    logger.info(
        'Shared mosaic plan: %d box(es) under %s',
        len(plan.boxes),
        plan.reference_dir,
    )
    try:
        cov_ra, cov_dec = _resolve_coverage_sky(args, plan=plan)
    except ValueError as exc:
        logger.error('%s', exc)
        return 2
    args._cov_ra = cov_ra
    args._cov_dec = cov_dec
    if cov_ra is not None and cov_dec is not None:
        logger.info(
            'Frame coverage filter: RA=%.6f Dec=%.6f',
            cov_ra,
            cov_dec,
        )

    jobs: dict[str, int] = {}
    # Sequential legs: JWST first (writes i2d anchors), then HST into same boxes.
    if jwst_inst:
        try:
            jobs['jwst'] = _run_jwst_mosaic(
                args,
                forced_filters=jwst_filters,
                ncores=ncores,
                plan=plan,
            )
        except Exception as exc:
            logger.exception('JWST mosaic failed: %s', exc)
            jobs['jwst'] = 1
    if hst_inst:
        try:
            jobs['hst'] = _run_hst_mosaic(
                args,
                instruments=hst_inst,
                ncores=ncores,
                plan=plan,
            )
        except Exception as exc:
            logger.exception('HST mosaic failed: %s', exc)
            jobs['hst'] = 1

    logger.info('Orchestrated mosaic summary:')
    for name, rc in jobs.items():
        logger.info('  %s: rc=%d', name, int(rc))
    if not jobs:
        logger.error('No mosaic stages were requested')
        return 1
    return 0 if all(int(rc) == 0 for rc in jobs.values()) else 1


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    configure_logging_from_args(args, 'mosaic')
    try:
        # --telescope hst|jwst == default --instruments for that mission.
        multi = resolve_instruments_with_telescope(
            getattr(args, 'instruments', None),
            getattr(args, 'telescope', None),
            resolve_instruments_fn=resolve_mosaic_instruments,
        )
        if multi is not None and needs_mosaic_orchestration(multi):
            return run_orchestrated_mosaic(args, multi)

        if (
            getattr(args, 'telescope', None) is not None
            and str(args.telescope).lower() == 'hst'
        ):
            return _run_hst_mosaic(args)

        return _run_jwst_mosaic(args)
    finally:
        shutdown_logging()


if __name__ == '__main__':
    raise SystemExit(main())
