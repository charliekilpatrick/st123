"""
Regression tests for mosaic pipeline failures seen on NGC3310.

Covers:
1. ``resolve_reduction_dir`` must prefer ``reduction/`` over a project-root
   ``reference/`` symlink (otherwise mosaic finds zero jhat frames).
2. Empty input lists must fail clearly (no ``IndexError``).
3. ``create_gwcs`` must emit ``FITSImagingWCSTransform`` for jwst>=1.20 resample.
4. ``create_coadd_mosaic`` always requires / applies that GWCS.
5. ``nircammask`` must never pass the unsupported ``-etctime`` flag.
6. Phot prep must exclude ``*.sky.fits`` from mask/sky inputs.
7. Live NGC3310 products (when present) match the expected mosaic layout.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import asdf
import pytest
from astropy.io import fits
from astropy.table import Table
from gwcs import FITSImagingWCSTransform

from st123.mosaic.mosaic import create_coadd_mosaic, create_gwcs
from st123.photometry.dolphot_prep import apply_nircammask, science_fits_paths
from st123.scripts import mosaic as mosaic_script
from st123.scripts.utils.options import resolve_reduction_dir
from st123.utils.helpers import input_list

NGC3310_ROOT = Path('/data/ckilpatrick/JWST/NGC3310')
NGC3310_REDUCTION = NGC3310_ROOT / 'reduction'
NGC3310_JHAT = NGC3310_REDUCTION / 'jhat'
NGC3310_REF0 = NGC3310_REDUCTION / 'reference' / 'group_0' / 'ref_0'
NGC3310_PHOT = NGC3310_REDUCTION / 'phot_0_0'

requires_ngc3310 = pytest.mark.skipif(
    not NGC3310_JHAT.is_dir(),
    reason='NGC3310 reduction tree not available on this host',
)

EXPECTED_JHAT_COUNT = 80
EXPECTED_PHOT_SCIENCE = 81  # 80 jhat + coadd
EXPECTED_SW_COADD_FILTER = 'f150w2'


def _sample_wcs_header() -> fits.Header:
    return fits.Header(
        {
            'CRPIX1': 100.0,
            'CRPIX2': 200.0,
            'CRVAL1': 160.0,
            'CRVAL2': 45.0,
            'CDELT1': -8.64e-6,
            'CDELT2': 8.64e-6,
            'PC1_1': 1.0,
            'PC1_2': 0.0,
            'PC2_1': 0.0,
            'PC2_2': 1.0,
            'NAXIS1': 300,
            'NAXIS2': 400,
        }
    )


def _make_ngc3310_like_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Project root with JWST/, reduction/jhat/, and reference/ symlink."""
    project = tmp_path / 'NGC3310'
    reduction = project / 'reduction'
    jhat = reduction / 'jhat'
    jhat.mkdir(parents=True)
    (project / 'JWST').mkdir()
    (reduction / 'reference').mkdir(parents=True)
    (project / 'reference').symlink_to(reduction / 'reference')
    return project, reduction, jhat


# ---------------------------------------------------------------------------
# 1. Path resolution
# ---------------------------------------------------------------------------


def test_resolve_reduction_dir_prefers_reduction_with_reference_symlink(
    tmp_path: Path,
):
    project, reduction, _jhat = _make_ngc3310_like_project(tmp_path)
    assert resolve_reduction_dir(project) == reduction.resolve()
    assert resolve_reduction_dir(reduction) == reduction.resolve()


def test_mosaic_main_finds_jhat_under_reduction_not_project_root(tmp_path: Path):
    """
    The IndexError regression: project-root reference/ symlink made mosaic
    look in <project>/jhat (empty) instead of <project>/reduction/jhat.
    """
    project, reduction, jhat = _make_ngc3310_like_project(tmp_path)
    fake = jhat / 'jw_test_nrcb1_jhat.fits'
    fake.write_text('x')

    with patch(
        'st123.scripts.mosaic.input_list', side_effect=RuntimeError('STOP')
    ) as mock_input:
        with pytest.raises(RuntimeError, match='STOP'):
            mosaic_script.main(['--base-dir', str(project), '--ncores', '1'])

    files = mock_input.call_args.args[0]
    assert len(files) == 1
    assert Path(files[0]).resolve() == fake.resolve()
    assert 'reduction' in Path(files[0]).parts


# ---------------------------------------------------------------------------
# 2. Empty inputs
# ---------------------------------------------------------------------------


def test_input_list_empty_raises_value_error():
    with pytest.raises(ValueError, match='No existing input images'):
        input_list([])


def test_input_list_missing_paths_raise_value_error(tmp_path: Path):
    missing = str(tmp_path / 'nope_jhat.fits')
    with pytest.raises(ValueError, match='No existing input images'):
        input_list([missing])


def test_mosaic_main_returns_1_when_jhat_missing(tmp_path: Path):
    project, _reduction, _jhat = _make_ngc3310_like_project(tmp_path)
    # jhat/ exists but is empty
    rc = mosaic_script.main(['--base-dir', str(project), '--ncores', '1'])
    assert rc == 1


# ---------------------------------------------------------------------------
# 3–4. GWCS / Image3 output_wcs
# ---------------------------------------------------------------------------


def test_create_gwcs_is_fits_imaging_transform(tmp_path: Path):
    path = create_gwcs(outdir=str(tmp_path), sci_header=_sample_wcs_header())
    wcs = asdf.open(path)['wcs']
    assert isinstance(wcs.forward_transform, FITSImagingWCSTransform)
    # jwst 1.20 finalize path
    from jwst.resample.resample import ResampleImage
    from stdatamodels.jwst import datamodels

    model = datamodels.ImageModel((8, 8))
    model.meta.wcs = wcs
    ResampleImage.update_fits_wcsinfo(model)
    assert model.meta.wcsinfo.crpix1 == 100.0


def test_create_coadd_mosaic_always_sets_output_wcs(tmp_path: Path):
    table = Table({'image': [str(tmp_path / 'a.fits')]})
    image3 = MagicMock()
    image3.resample = MagicMock()
    image3.tweakreg = MagicMock()
    image3.skymatch = MagicMock()
    image3.source_catalog = MagicMock()
    with (
        patch('st123.mosaic.mosaic.patch_jwst_for_photutils3'),
        patch('st123.mosaic.mosaic.asn_from_list') as asn_mod,
        patch(
            'st123.mosaic.mosaic.calwebb_image3.Image3Pipeline',
            return_value=image3,
        ),
    ):
        asn = MagicMock()
        asn.dump.return_value = ('name', '{}')
        asn_mod.asn_from_list.return_value = asn
        create_coadd_mosaic(
            table,
            outdir=str(tmp_path),
            filt='f150w2',
            sci_header=_sample_wcs_header(),
        )
    assert image3.resample.output_wcs == str(tmp_path / 'mosaic_gwcs.asdf')
    assert Path(image3.resample.output_wcs).is_file()


def test_create_coadd_mosaic_rejects_missing_gwcs_file(tmp_path: Path):
    table = Table({'image': ['a.fits']})
    missing = tmp_path / 'missing_gwcs.asdf'
    with pytest.raises(FileNotFoundError, match='gwcs_file not found'):
        create_coadd_mosaic(
            table,
            outdir=str(tmp_path),
            filt='f150w2',
            gwcs_file=str(missing),
        )


# ---------------------------------------------------------------------------
# 5–6. DOLPHOT prep flags / sky exclusion
# ---------------------------------------------------------------------------


def test_science_fits_paths_excludes_sky_products(tmp_path: Path):
    (tmp_path / 'a_jhat.fits').write_text('a')
    (tmp_path / 'a_jhat.sky.fits').write_text('sky')
    (tmp_path / 'coadd_0_0_f150w2_i2d.fits').write_text('c')
    (tmp_path / 'coadd_0_0_f150w2_i2d.sky.fits').write_text('sky')
    paths = science_fits_paths(tmp_path)
    names = {Path(p).name for p in paths}
    assert names == {'a_jhat.fits', 'coadd_0_0_f150w2_i2d.fits'}


@patch('st123.photometry.dolphot_prep.subprocess.run')
def test_apply_nircammask_command_has_no_etctime(mock_run, tmp_path: Path):
    fits_path = tmp_path / 'x_nrcb1_jhat.fits'
    fits_path.write_text('')
    apply_nircammask([fits_path], dolphot_bin='/data/software/dolphot/bin')
    cmd = mock_run.call_args.args[0]
    assert cmd[0].endswith('nircammask')
    assert '-etctime' not in cmd
    assert '-noetctime' not in cmd  # ETC time is the DOLPHOT default


@pytest.mark.skipif(
    not Path('/data/software/dolphot/bin/nircammask').is_file(),
    reason='DOLPHOT nircammask binary not installed',
)
def test_installed_nircammask_usage_has_no_etctime_flag():
    proc = subprocess.run(
        ['/data/software/dolphot/bin/nircammask'],
        check=False,
        capture_output=True,
        text=True,
    )
    usage = (proc.stdout or '') + (proc.stderr or '')
    assert 'nircammask' in usage and 'Usage:' in usage
    assert '-noetctime' in usage
    # Strip the real flag so a bare "-etctime" token would still be caught.
    assert '-etctime' not in usage.replace('-noetctime', '')


# ---------------------------------------------------------------------------
# 7. Live NGC3310 dataset expectations
# ---------------------------------------------------------------------------


@requires_ngc3310
def test_ngc3310_resolve_reduction_dir_despite_reference_symlink():
    assert NGC3310_ROOT.joinpath('reference').is_symlink()
    assert resolve_reduction_dir(NGC3310_ROOT) == NGC3310_REDUCTION.resolve()
    assert resolve_reduction_dir(NGC3310_REDUCTION) == NGC3310_REDUCTION.resolve()


@requires_ngc3310
def test_ngc3310_jhat_inventory_for_mosaic():
    files = sorted(NGC3310_JHAT.glob('*jhat.fits'))
    assert len(files) == EXPECTED_JHAT_COUNT
    table = input_list([str(p) for p in files])
    assert len(table) == EXPECTED_JHAT_COUNT
    assert set(table['group']) == {0}
    filters = {str(f) for f in table['filter']}
    assert EXPECTED_SW_COADD_FILTER in filters
    assert {'f150w', 'f200w', 'f444w'} <= filters


@requires_ngc3310
def test_ngc3310_sw_filter_table_targets_f150w2_coadd():
    from st123.mosaic.mosaic import split_observations

    files = sorted(str(p) for p in NGC3310_JHAT.glob('*jhat.fits'))
    table = input_list(files)
    split = split_observations(table=table[table['group'] == 0], N_max=150)
    split.boxsplit()
    filter_tables = split.get_sw_filter_tables(tol=0.05)
    assert len(split.split_boxes) == 1
    keys = [str(k) for k in filter_tables[0].keys()]
    assert keys == [EXPECTED_SW_COADD_FILTER]
    assert len(filter_tables[0][EXPECTED_SW_COADD_FILTER]) == 16


@requires_ngc3310
def test_ngc3310_mosaic_products_present_and_gwcs_compatible():
    coadd = NGC3310_REF0 / f'coadd_0_0_{EXPECTED_SW_COADD_FILTER}_i2d.fits'
    i2d = NGC3310_REF0 / f'out_{EXPECTED_SW_COADD_FILTER}' / f'{EXPECTED_SW_COADD_FILTER}_i2d.fits'
    gwcs_path = NGC3310_REF0 / 'mosaic_gwcs.asdf'
    frame_list = NGC3310_REF0 / 'dolphot_frames.txt'
    # dolphot.param is produced by dolphot-prep, not mosaic
    param = NGC3310_PHOT / 'dolphot.param'

    assert coadd.is_file() and coadd.stat().st_size > 0
    assert i2d.is_file() and i2d.stat().st_size > 0
    assert gwcs_path.is_file()
    assert frame_list.is_file()
    assert param.is_file()

    wcs = asdf.open(gwcs_path)['wcs']
    assert isinstance(wcs.forward_transform, FITSImagingWCSTransform)

    from jwst.resample.resample import ResampleImage
    from stdatamodels.jwst import datamodels

    model = datamodels.ImageModel((8, 8))
    model.meta.wcs = wcs
    ResampleImage.update_fits_wcsinfo(model)
    assert model.meta.wcsinfo.crpix1 is not None


@requires_ngc3310
def test_ngc3310_phot_prep_products_match_science_frames():
    science = science_fits_paths(NGC3310_PHOT)
    skies = sorted(NGC3310_PHOT.glob('*.sky.fits'))
    assert len(science) == EXPECTED_PHOT_SCIENCE
    assert len(skies) == EXPECTED_PHOT_SCIENCE
    # Every science frame has a matching sky product.
    for path in science:
        assert path.endswith('.fits')
        sky = Path(path[: -len('.fits')] + '.sky.fits')
        assert sky.is_file(), f'missing sky for {path}'


@requires_ngc3310
def test_ngc3310_mosaic_cli_resolves_same_workdir_as_explicit_reduction():
    """``--base-dir $BASE`` and ``--base-dir $BASE/reduction`` must agree."""
    from st123.scripts.utils.options import resolve_reduction_dir as resolve

    assert resolve(NGC3310_ROOT) == resolve(NGC3310_REDUCTION)
    # Smoke: mosaic empty-check would pass (jhat present) for both roots.
    for base in (NGC3310_ROOT, NGC3310_REDUCTION):
        red = resolve(base)
        n = len(list((red / 'jhat').glob('*jhat.fits')))
        assert n == EXPECTED_JHAT_COUNT
