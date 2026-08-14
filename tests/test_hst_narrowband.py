"""Tests for HST narrowband mitigation (relaxed JHAT → HST_REL → PIPELINE)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from st123.alignment import hst_jhat as hj


def _wcs_header(*, crval=(150.0, 2.0), cdelt=0.05 / 3600.0) -> fits.Header:
    w = WCS(naxis=2)
    w.wcs.crpix = [16.0, 16.0]
    w.wcs.crval = list(crval)
    w.wcs.cdelt = [-cdelt, cdelt]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return w.to_header()


def _write_frame(
    path: Path,
    *,
    instrument: str,
    filter_name: str,
    crval=(150.0, 2.0),
    shape=(32, 32),
    rootname: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = instrument
    primary.header['TELESCOP'] = 'HST'
    if instrument.upper() == 'ACS':
        primary.header['FILTER1'] = 'CLEAR1L'
        primary.header['FILTER2'] = filter_name.upper()
    else:
        primary.header['FILTER'] = filter_name.upper()
    if rootname:
        primary.header['ROOTNAME'] = rootname
    data = np.zeros(shape, dtype=np.float32)
    # A few bright pixels so DAOStarFinder can detect sources if needed.
    data[10, 10] = 1000.0
    data[12, 14] = 900.0
    data[18, 20] = 800.0
    data[22, 8] = 850.0
    data[8, 24] = 950.0
    sci = fits.ImageHDU(data, header=_wcs_header(crval=crval), name='SCI')
    fits.HDUList([primary, sci]).writeto(path, overwrite=True)
    return path


def test_is_hst_narrowband_filter():
    assert hj.is_hst_narrowband_filter('F660N')
    assert hj.is_hst_narrowband_filter('f657n')
    assert hj.is_hst_narrowband_filter('F502N')
    assert hj.is_hst_narrowband_filter('FQ437N')
    assert not hj.is_hst_narrowband_filter('F814W')
    assert not hj.is_hst_narrowband_filter('F550M')
    assert not hj.is_hst_narrowband_filter('F606W')
    assert not hj.is_hst_narrowband_filter('')
    assert not hj.is_hst_narrowband_filter(None)


def test_write_hst_jhat_from_raw_and_pipeline_fallback(tmp_path: Path):
    raw = _write_frame(
        tmp_path / 'raw' / 'j9b031kyq_flc.fits',
        instrument='ACS',
        filter_name='F660N',
        rootname='j9b031kyq',
    )
    out = tmp_path / 'jhat'
    jhat = hj.write_hst_jhat_from_raw(raw, out)
    assert jhat.name == 'j9b031kyq_jhat.fits'
    assert jhat.is_file()

    pipe = hj.align_hst_narrowband_pipeline_fallback(raw, out)
    assert pipe['ok']
    assert pipe['align_mode'] == 'PIPELINE'
    with fits.open(pipe['outpath']) as hdul:
        assert hdul[0].header['ALGNMODE'] == 'PIPELINE'
        assert hdul[0].header['ST123NAR'] is True


def test_select_hst_narrowband_parent_prefers_broadband_same_visit(tmp_path: Path):
    narrow = _write_frame(
        tmp_path / 'raw' / 'j9b031kyq_flc.fits',
        instrument='ACS',
        filter_name='F660N',
        rootname='j9b031kyq',
    )
    bb_jhat = _write_frame(
        tmp_path / 'jhat' / 'j9b031aaq_jhat.fits',
        instrument='ACS',
        filter_name='F814W',
        rootname='j9b031aaq',
    )
    other_jhat = _write_frame(
        tmp_path / 'jhat' / 'ie9801xcq_jhat.fits',
        instrument='WFC3',
        filter_name='F606W',
        rootname='ie9801xcq',
    )
    nb_ok = _write_frame(
        tmp_path / 'jhat' / 'if0415swq_jhat.fits',
        instrument='WFC3',
        filter_name='F657N',
        rootname='if0415swq',
    )
    ok_results = [
        {'status': 'ok', 'path': str(tmp_path / 'raw' / 'x_flc.fits'), 'outpath': str(nb_ok)},
        {
            'status': 'ok',
            'path': str(tmp_path / 'raw' / 'ie9801xcq_flc.fits'),
            'outpath': str(other_jhat),
        },
        {
            'status': 'ok',
            'path': str(tmp_path / 'raw' / 'j9b031aaq_flc.fits'),
            'outpath': str(bb_jhat),
        },
    ]
    parent = hj.select_hst_narrowband_parent(narrow, ok_results)
    assert parent is not None
    assert parent.name == 'j9b031aaq_jhat.fits'


def test_refine_hst_narrowband_to_refcat_applies_crval(tmp_path: Path):
    """CRVAL refine should pull a coherent ~0.1\" residual onto the refcat."""
    jhat = _write_frame(
        tmp_path / 'jhat' / 'j9b031kyq_jhat.fits',
        instrument='ACS',
        filter_name='F660N',
        rootname='j9b031kyq',
        crval=(150.0, 2.0),
        shape=(64, 64),
    )
    # Synthetic phot at pixel centers; refcat at sky of a +0.1" shifted WCS.
    from astropy.wcs import WCS as _WCS
    import pandas as pd

    with fits.open(jhat) as hdul:
        w = _WCS(hdul[1].header, hdul, naxis=2)
    xs = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 15.0, 25.0, 35.0, 45.0, 55.0])
    ys = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 45.0, 50.0, 12.0, 18.0])
    ra, dec = w.pixel_to_world_values(xs, ys)
    # Refcat is 0.08" east of the image WCS → image needs +dRA.
    dra = 0.08 / 3600.0
    ref = pd.DataFrame({'ra': np.asarray(ra) + dra, 'dec': np.asarray(dec), 'mag': np.arange(len(xs))})
    phot = pd.DataFrame(
        {
            'x': xs,
            'y': ys,
            'ra': ra,
            'dec': dec,
            'mag': np.arange(len(xs)),
        }
    )
    phot.to_csv(tmp_path / 'jhat' / 'j9b031kyq.phot.txt', sep=' ', index=False)
    refcat = tmp_path / 'ref.phot.txt'
    ref.to_csv(refcat, sep=' ', index=False)

    with fits.open(jhat) as hdul:
        crval1_0 = float(hdul[1].header['CRVAL1'])

    out = hj.refine_hst_narrowband_to_refcat(jhat, refcat, per_chip=False)
    assert out['residual_arcsec'] is not None
    assert out['residual_arcsec'] < 0.05
    with fits.open(jhat) as hdul:
        assert float(hdul[1].header['CRVAL1']) > crval1_0
        assert hdul[0].header.get('ST123NRM') is not None


def test_recover_failed_hst_narrowbands_rel_then_pipeline(tmp_path: Path):
    raw_nb = _write_frame(
        tmp_path / 'raw' / 'j9b031kyq_flc.fits',
        instrument='ACS',
        filter_name='F660N',
        rootname='j9b031kyq',
        crval=(150.0, 2.0),
    )
    raw_bb = _write_frame(
        tmp_path / 'raw' / 'j9b031aaq_flc.fits',
        instrument='ACS',
        filter_name='F814W',
        rootname='j9b031aaq',
        crval=(150.0, 2.0),
    )
    bb_jhat = tmp_path / 'jhat' / 'j9b031aaq_jhat.fits'
    bb_jhat.parent.mkdir(parents=True)
    # Parent JHAT slightly offset so a relative correction is meaningful.
    _write_frame(
        bb_jhat,
        instrument='ACS',
        filter_name='F814W',
        rootname='j9b031aaq',
        crval=(150.0 + 0.5 / 3600.0, 2.0),
    )

    results = [
        {
            'path': str(raw_bb),
            'status': 'ok',
            'error': None,
            'outpath': str(bb_jhat),
        },
        {
            'path': str(raw_nb),
            'status': 'failed',
            'error': 'RuntimeError: Only 0 objects pass the initial cut',
            'outpath': None,
        },
    ]

    def _fake_rel(raw_path, parent_jhat, outdir, **kwargs):
        del kwargs
        jhat = hj.write_hst_jhat_from_raw(raw_path, outdir)
        hj.write_hst_narrowband_provenance(
            jhat,
            align_mode='HST_REL',
            aligned_to=str(parent_jhat),
            n_match=12,
            abs_arcsec=0.12,
            dra_arcsec=-0.1,
            ddec_arcsec=0.05,
        )
        return {
            'ok': True,
            'outpath': str(jhat),
            'align_mode': 'HST_REL',
            'aligned_to': str(parent_jhat),
            'n_match': 12,
            'abs_arcsec': 0.12,
            'error': None,
        }

    with (
        patch.object(hj, 'align_hst_narrowband_relative', side_effect=_fake_rel),
        patch.object(
            hj,
            'finalize_hst_narrowband_group',
            return_value={'ok': True, 'residuals': [], 'med_residual_arcsec': 0.04},
        ),
    ):
        hj.recover_failed_hst_narrowbands(results, tmp_path / 'jhat')

    nb_row = results[1]
    assert nb_row['status'] == 'ok'
    assert nb_row['align_mode'] == 'HST_REL'
    assert Path(nb_row['outpath']).is_file()
    with fits.open(nb_row['outpath']) as hdul:
        assert hdul[0].header['ALGNMODE'] == 'HST_REL'


def test_recover_pipeline_when_relative_fails(tmp_path: Path):
    raw_nb = _write_frame(
        tmp_path / 'raw' / 'j9b031l0q_flc.fits',
        instrument='ACS',
        filter_name='F660N',
        rootname='j9b031l0q',
    )
    results = [
        {
            'path': str(raw_nb),
            'status': 'failed',
            'error': 'RuntimeError: Only 0 objects pass the initial cut',
            'outpath': None,
        }
    ]
    # No broadband parents → skip HST_REL, use PIPELINE.
    hj.recover_failed_hst_narrowbands(results, tmp_path / 'jhat')
    assert results[0]['status'] == 'ok'
    assert results[0]['align_mode'] == 'PIPELINE'
    assert Path(results[0]['outpath']).is_file()


def test_align_hst_raw_dir_uses_narrowband_recovery(tmp_path: Path):
    raw = tmp_path / 'raw'
    jhat = tmp_path / 'jhat'
    bb = _write_frame(
        raw / 'j9b031aaq_flc.fits',
        instrument='ACS',
        filter_name='F814W',
        rootname='j9b031aaq',
    )
    nb = _write_frame(
        raw / 'j9b031kyq_flc.fits',
        instrument='ACS',
        filter_name='F660N',
        rootname='j9b031kyq',
    )

    def _fake_align(frame, outdir, **kwargs):
        del kwargs
        frame = Path(frame)
        if 'kyq' in frame.name:
            raise RuntimeError('Only 0 objects pass the initial cut, at least 3 required!')
        out = hj._jhat_hst_output_path(frame, outdir)
        _write_frame(
            out,
            instrument='ACS',
            filter_name='F814W',
            rootname='j9b031aaq',
        )
        return out

    def _rel(raw_path, parent_jhat, outdir, **kwargs):
        del kwargs
        outp = hj.write_hst_jhat_from_raw(raw_path, outdir)
        hj.write_hst_narrowband_provenance(
            outp,
            align_mode='HST_REL',
            aligned_to=str(parent_jhat),
            n_match=8,
            abs_arcsec=0.2,
        )
        return {
            'ok': True,
            'outpath': str(outp),
            'align_mode': 'HST_REL',
            'aligned_to': str(parent_jhat),
            'n_match': 8,
            'abs_arcsec': 0.2,
            'error': None,
        }

    with (
        patch.object(hj, 'align_hst_image', side_effect=_fake_align),
        patch.object(hj, 'harmonize_hst_jhat_dir', return_value=[]),
        patch.object(hj, 'align_hst_narrowband_relative', side_effect=_rel),
        patch.object(
            hj,
            'finalize_hst_narrowband_group',
            return_value={'ok': True, 'residuals': [], 'med_residual_arcsec': 0.04},
        ),
        patch.object(hj, 'find_hst_l3_refcat', return_value=None),
        patch.object(hj, 'find_hst_abs_ref_image', return_value=None),
    ):
        results = hj.align_hst_raw_dir(
            raw,
            jhat,
            instruments=['ACS'],
            soft_fail=True,
            gaia=False,
            photfilename=tmp_path / 'dummy.phot.txt',
        )

    by_name = {Path(r['path']).name: r for r in results}
    assert by_name[bb.name]['status'] == 'ok'
    assert by_name[nb.name]['status'] == 'ok'
    assert by_name[nb.name]['align_mode'] == 'HST_REL'
    assert Path(by_name[nb.name]['outpath']).is_file()
