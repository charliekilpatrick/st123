"""Tests for multi-telescope mosaic footprint weights and ACS/WFC3 filter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import shapely
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from st123.mosaic.footprints import (
    build_weight_pgons,
    collect_footprint_weight_paths,
    project_sky_polygons_to_wcs,
    sky_polygons_from_fits,
)
from st123.mosaic.hst_drizzle import drizzle_project, group_hst_frames
from st123.mosaic.mosaic import split_observations
from st123.scripts import mosaic as mosaic_script


def _poly_s_region(ra0: float, dec0: float, dra: float = 0.01, ddec: float = 0.01) -> str:
    coords = [
        (ra0, dec0),
        (ra0 + dra, dec0),
        (ra0 + dra, dec0 + ddec),
        (ra0, dec0 + ddec),
    ]
    flat = ' '.join(f'{x} {y}' for x, y in coords)
    return f'POLYGON ICRS  {flat}'


def _write_sci_with_sregion(
    path: Path,
    *,
    instrument: str | None = None,
    s_region: str | None = None,
    n_sci: int = 1,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU()
    if instrument:
        primary.header['INSTRUME'] = instrument
    hdus = [primary]
    for i in range(n_sci):
        sci = fits.ImageHDU(np.ones((8, 8), dtype=np.float32), name='SCI')
        if s_region:
            # Slightly shift chips so multi-SCI expands to multiple polygons
            if n_sci > 1 and i > 0:
                sci.header['S_REGION'] = _poly_s_region(10.0 + 0.02 * i, 20.0)
            else:
                sci.header['S_REGION'] = s_region
        hdus.append(sci)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def test_mosaic_parser_footprint_weights_and_instruments():
    parser = mosaic_script.create_parser()
    args = parser.parse_args(
        [
            '--telescope',
            'hst',
            '--base-dir',
            '/b',
            '--instruments',
            'ACS',
            'WFC3',
            '--footprint-weights',
            'none',
        ]
    )
    assert args.instruments == ['ACS', 'WFC3']
    assert args.footprint_weights == 'none'

    args2 = parser.parse_args(['--base-dir', '/b'])
    assert args2.footprint_weights == 'auto'
    assert args2.instruments is None


def test_mosaic_telescope_equals_default_instruments():
    from st123.scripts.utils.options import resolve_instruments_with_telescope

    assert resolve_instruments_with_telescope(
        None,
        'hst',
        resolve_instruments_fn=mosaic_script.resolve_mosaic_instruments,
    ) == ['ACS', 'WFC3', 'WFPC2']
    assert resolve_instruments_with_telescope(
        None,
        'jwst',
        resolve_instruments_fn=mosaic_script.resolve_mosaic_instruments,
    ) == ['NIRCAM', 'MIRI']
    assert mosaic_script.needs_mosaic_orchestration(['ACS', 'WFC3', 'WFPC2'])
    assert mosaic_script.needs_mosaic_orchestration(['NIRCAM', 'MIRI'])


def test_mosaic_resolve_instruments_all():
    assert mosaic_script.resolve_mosaic_instruments(['ALL']) == [
        'NIRCAM',
        'MIRI',
        'ACS',
        'WFC3',
        'WFPC2',
    ]
    assert mosaic_script.resolve_mosaic_instruments(['hst']) == [
        'ACS',
        'WFC3',
        'WFPC2',
    ]
    assert mosaic_script.default_jwst_filters_for_instruments(
        ['NIRCAM', 'MIRI']
    ) == [
        'f150w',
        'f150w2',
        'f187n',
        'f200w',
        'f300m',
        'f335m',
        'f360m',
        'f430m',
        'f444w',
        'f770w',
        'f1000w',
        'f1130w',
        'f2100w',
    ]


def test_orchestrated_mosaic_runs_jwst_and_hst_sequential(tmp_path):
    from st123.mosaic.mosaic import MosaicPlan

    calls: list[str] = []

    def _jwst(args, *, forced_filters=None, ncores=None, plan=None):
        calls.append(
            f'jwst:{",".join(forced_filters or [])}:{ncores}:plan={plan is not None}'
        )
        return 0

    def _hst(args, *, instruments=None, ncores=None, plan=None):
        calls.append(
            f'hst:{",".join(instruments or [])}:{ncores}:plan={plan is not None}'
        )
        return 0

    ns = mosaic_script.create_parser().parse_args(
        [
            '--base-dir',
            str(tmp_path),
            '--instruments',
            'NIRCAM',
            'MIRI',
            'ACS',
            'WFC3',
            '--ncores',
            '8',
            '-v',
        ]
    )
    plan = MosaicPlan(
        base_dir=tmp_path,
        reference_dir=tmp_path / 'reference',
        table=Table(),
        boxes=[],
    )
    with (
        patch.object(
            mosaic_script,
            '_collect_mission_jhat',
            side_effect=lambda base, mission: [f'/{mission}_a_jhat.fits'],
        ),
        patch(
            'st123.mosaic.mosaic.plan_mosaic_boxes', return_value=plan
        ),
        patch.object(mosaic_script, '_run_jwst_mosaic', side_effect=_jwst),
        patch.object(mosaic_script, '_run_hst_mosaic', side_effect=_hst),
    ):
        rc = mosaic_script.run_orchestrated_mosaic(
            ns, ['NIRCAM', 'MIRI', 'ACS', 'WFC3']
        )
    assert rc == 0
    # Sequential: JWST first (writes i2d anchors), then HST; shared plan; full ncores.
    assert calls[0].startswith('jwst:')
    assert calls[1].startswith('hst:ACS,WFC3:')
    assert all(':plan=True' in c for c in calls)
    assert any(c.endswith(':8:plan=True') for c in calls if c.startswith('jwst:'))
    assert any(c.endswith(':8:plan=True') for c in calls if c.startswith('hst:'))
    jwst_call = next(c for c in calls if c.startswith('jwst:'))
    # Default: no --filters → per-box "all present" (forced_filters=None).
    assert jwst_call.startswith('jwst::')


def test_mosaic_main_instruments_dispatches_orchestrator(tmp_path):
    with patch.object(
        mosaic_script, 'run_orchestrated_mosaic', return_value=0
    ) as mock_orch:
        rc = mosaic_script.main(
            [
                '--base-dir',
                str(tmp_path),
                '--instruments',
                'NIRCAM',
                'MIRI',
                'ACS',
                'WFC3',
                '--nmax',
                '150',
            ]
        )
    assert rc == 0
    mock_orch.assert_called_once()
    assert mock_orch.call_args.args[1] == [
        'NIRCAM',
        'MIRI',
        'ACS',
        'WFC3',
    ]


def test_collect_footprint_weight_paths_hst_acs_wfc3_skips_wfpc2(tmp_path: Path):
    work = tmp_path / 'reduction'
    jhat = work / 'jhat_hst'
    region = _poly_s_region(185.7, 15.8)
    acs = _write_sci_with_sregion(
        jhat / 'j9xxx_acs_jhat.fits', instrument='ACS', s_region=region, n_sci=2
    )
    wfc3 = _write_sci_with_sregion(
        jhat / 'iexxx_wfc3_jhat.fits', instrument='WFC3', s_region=region
    )
    _write_sci_with_sregion(
        jhat / 'u2xxx_wfpc2_jhat.fits', instrument='WFPC2', s_region=region
    )
    # JWST name must be ignored even under jhat_hst
    _write_sci_with_sregion(
        jhat / 'jw012345_nrcb1_jhat.fits', instrument='NIRCAM', s_region=region
    )

    found = collect_footprint_weight_paths(work)
    assert set(found) == {acs.resolve(), wfc3.resolve()}


def test_collect_miri_raw_only_when_no_miri_jhat(tmp_path: Path):
    work = tmp_path / 'reduction'
    region = _poly_s_region(185.7, 15.8)
    raw = work / 'raw'
    raw_miri = _write_sci_with_sregion(
        raw / 'jw01234567001_02101_00001_mirimage_cal.fits',
        s_region=region,
    )
    # No MIRI JHAT yet → raw counts
    found = collect_footprint_weight_paths(work)
    assert found == [raw_miri.resolve()]

    jhat = work / 'jhat_jwst'
    miri_jhat = _write_sci_with_sregion(
        jhat / 'jw01234567001_02101_00001_mirimage_jhat.fits',
        s_region=region,
    )
    found2 = collect_footprint_weight_paths(work)
    assert found2 == [miri_jhat.resolve()]
    assert raw_miri.resolve() not in found2


def test_sky_polygons_per_sci_chip(tmp_path: Path):
    region = _poly_s_region(10.0, 20.0)
    path = _write_sci_with_sregion(
        tmp_path / 'acs_jhat.fits', instrument='ACS', s_region=region, n_sci=2
    )
    pgons = sky_polygons_from_fits(path)
    assert len(pgons) == 2


def test_boxsplit_weights_force_earlier_split():
    """Weight footprints that fill a box force subdivision below N_max."""
    # Four non-overlapping science boxes in a 2x2 grid (pixel frame).
    sci_pgons = [
        shapely.box(0, 0, 10, 10),
        shapely.box(20, 0, 30, 10),
        shapely.box(0, 20, 10, 30),
        shapely.box(20, 20, 30, 30),
    ]
    # Weights cover the same areas → each science frame has a twin weight.
    weight_pgons = [shapely.box(*p.bounds) for p in sci_pgons]

    table = Table({'image': [f'/fake/sci_{i}.fits' for i in range(4)]})
    # Minimal WCS; polygons are already in pixel space.
    w = WCS(naxis=2)
    w.wcs.crpix = [15.0, 15.0]
    w.wcs.cdelt = [-0.001, 0.001]
    w.wcs.crval = [180.0, 0.0]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']

    # Without weights: 4 science < N_max=6 → one box
    split_plain = split_observations(
        table=table,
        N_max=6,
        polygons=sci_pgons,
        wcs_opt=w,
        min_overlap=0.01,
    )
    split_plain.boxsplit()
    assert len(split_plain.split_boxes) == 1

    # With weights: 4 sci + 4 weight = 8 >= 6 → must split
    split_w = split_observations(
        table=table,
        N_max=6,
        polygons=sci_pgons,
        wcs_opt=w,
        weight_pgons=weight_pgons,
        min_overlap=0.01,
    )
    split_w.boxsplit()
    assert len(split_w.split_boxes) >= 2


def test_build_weight_pgons_projects_sky(tmp_path: Path):
    region = _poly_s_region(185.73, 15.82, dra=0.02, ddec=0.02)
    path = _write_sci_with_sregion(
        tmp_path / 'wfc3_jhat.fits', instrument='WFC3', s_region=region
    )
    w = WCS(naxis=2)
    w.wcs.crpix = [50.0, 50.0]
    w.wcs.cdelt = [-0.001, 0.001]
    w.wcs.crval = [185.73, 15.82]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    pix = build_weight_pgons([path], w)
    assert len(pix) == 1
    assert pix[0].area > 0


def test_drizzle_project_instruments_empty_when_only_wfpc2(tmp_path: Path):
    """--instruments ACS WFC3 with only WFPC2 JHAT → no groups."""
    from st123.mosaic.mosaic import MosaicBox, MosaicPlan

    jhat = tmp_path / 'jhat_hst'
    out = tmp_path / 'reference'
    wfpc2 = _write_sci_with_sregion(
        jhat / 'wfpc2_f814w_jhat.fits',
        instrument='WFPC2',
        s_region=_poly_s_region(1.0, 2.0),
    )
    with fits.open(wfpc2, mode='update') as hdul:
        hdul[0].header['FILTNAM1'] = 'F814W'
    assert ('wfpc2', 'f814w') in group_hst_frames([wfpc2])
    box = MosaicBox(
        group_id=0,
        box_id=0,
        outdir=out / 'group_0' / 'ref_0',
        bbox=None,
        frames=[str(wfpc2)],
    )
    plan = MosaicPlan(
        base_dir=tmp_path,
        reference_dir=out,
        table=Table(),
        boxes=[box],
    )
    with patch('st123.mosaic.mosaic.plan_mosaic_boxes', return_value=plan):
        results = drizzle_project(
            jhat, out, instruments=['ACS', 'WFC3'], num_cores=1
        )
    assert results == []


def test_drizzle_project_instruments_skips_wfpc2(tmp_path: Path):
    from st123.mosaic.mosaic import MosaicBox, MosaicPlan, mosaic_hst_coadd_basename

    jhat = tmp_path / 'jhat_hst'
    out = tmp_path / 'reference'
    box_outdir = out / 'group_0' / 'ref_5'
    box_outdir.mkdir(parents=True)
    acs = _write_sci_with_sregion(
        jhat / 'acs_f814w_jhat.fits',
        instrument='ACS',
        s_region=_poly_s_region(1.0, 2.0),
    )
    with fits.open(acs, mode='update') as hdul:
        hdul[0].header['FILTER'] = 'F814W'
    wfpc2 = _write_sci_with_sregion(
        jhat / 'wfpc2_f814w_jhat.fits',
        instrument='WFPC2',
        s_region=_poly_s_region(1.0, 2.0),
    )
    with fits.open(wfpc2, mode='update') as hdul:
        hdul[0].header['FILTNAM1'] = 'F814W'

    coadd = box_outdir / mosaic_hst_coadd_basename(0, 5, 'acs', 'f814w')
    harm = {
        'ok': True,
        'method': 'single',
        'anchor': str(acs),
        'pre': {'max_abs_arcsec': 0.01},
        'post': {'max_abs_arcsec': 0.01},
    }
    qa = {'ok': True, 'max_abs_arcsec': 0.01, 'failed_pairs': [], 'scope': 'full_group'}
    box = MosaicBox(
        group_id=0,
        box_id=5,
        outdir=box_outdir,
        bbox=None,
        frames=[str(acs), str(wfpc2)],
    )
    plan = MosaicPlan(
        base_dir=tmp_path,
        reference_dir=out,
        table=Table(),
        boxes=[box],
    )

    with (
        patch('st123.mosaic.mosaic.plan_mosaic_boxes', return_value=plan),
        patch(
            'st123.alignment.hst_jhat.find_hst_l3_refcat', return_value=None
        ),
        patch(
            'st123.alignment.hst_jhat.harmonize_hst_group_wcs', return_value=harm
        ),
        patch(
            'st123.alignment.hst_jhat.validate_hst_group_internal_alignment',
            return_value=qa,
        ),
        patch(
            'st123.mosaic.hst_drizzle.drizzle_filter_group', return_value=coadd
        ) as mock_driz,
        patch(
            'st123.mosaic.hst_drizzle._write_group_frame_list', return_value=None
        ),
        patch(
            'st123.mosaic.hst_drizzle.unify_hst_astrometric_frame',
            return_value={'ok': True},
        ),
    ):
        results = drizzle_project(
            jhat, out, instruments=['ACS', 'WFC3'], num_cores=1
        )

    assert mock_driz.call_count == 1
    assert len(results) == 1
    assert results[0]['instrument'] == 'acs'
    assert results[0]['filter'] == 'f814w'
    assert results[0]['status'] == 'ok'
    assert results[0]['output'] == str(coadd)
    assert coadd.name == 'coadd_0_5_acs_f814w_drc.fits'


def test_project_sky_polygons_to_wcs_roundtrip():
    sky = [shapely.box(10.0, 20.0, 10.01, 20.01)]
    w = WCS(naxis=2)
    w.wcs.crpix = [100.0, 100.0]
    w.wcs.cdelt = [-0.001, 0.001]
    w.wcs.crval = [10.005, 20.005]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    pix = project_sky_polygons_to_wcs(sky, w)
    assert len(pix) == 1
    assert pix[0].is_valid
