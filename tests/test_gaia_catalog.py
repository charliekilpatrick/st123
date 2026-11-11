"""Tests for Gaia catalog helpers (Vizier primary, ESA TAP fallback)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.table import Table

from st123.stages.alignment.gaia_catalog import (
    GAIA_MIN_CONE_RADIUS_DEG,
    VIZIER_GAIA_DR3,
    VIZIER_MIRRORS,
    _apply_min_cone_radius,
    _normalize_tap_gaia,
    _normalize_vizier_gaia,
    _vizier_catalog_for_dr,
    default_gaia_cache_dir,
    ensure_gaia_catalog,
    fetch_gaia_cone,
    gaia_cache_paths,
    install_jhat_gaia_vizier_patch,
    jhat_get_gaia_sources,
    load_gaia_cache,
    query_gaia,
    write_gaia_refcat,
)


def _toy_sci(path, *, ra=150.0, dec=2.0):
    from astropy.io import fits
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.crpix = [50.5, 50.5]
    w.wcs.cdelt = np.array([-0.001, 0.001])
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    data = np.zeros((100, 100), dtype=np.float32)
    hdu = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=data, header=w.to_header(), name='SCI'),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)
    return path


def _vizier_table():
    return Table(
        {
            'RA_ICRS': [150.0],
            'DE_ICRS': [2.0],
            'pmRA': [0.0],
            'pmDE': [0.0],
            'e_pmRA': [0.1],
            'e_pmDE': [0.1],
            'Gmag': [15.0],
            'Source': [99],
        }
    )


def _tap_table():
    return Table(
        {
            'source_id': [42],
            'ra': [150.0],
            'dec': [2.0],
            'pmra': [0.0],
            'pmdec': [0.0],
            'pmra_error': [0.1],
            'pmdec_error': [0.1],
            'phot_g_mean_mag': [15.0],
            'phot_bp_mean_mag': [15.5],
            'phot_rp_mean_mag': [14.5],
            'ref_epoch': [2016.0],
        }
    )


def test_normalize_vizier_gaia_columns():
    raw = Table(
        {
            'RA_ICRS': [10.0, 11.0],
            'DE_ICRS': [20.0, 21.0],
            'pmRA': [1.0, 2.0],
            'pmDE': [-1.0, -2.0],
            'e_pmRA': [0.1, 0.2],
            'e_pmDE': [0.1, 0.2],
            'Gmag': [15.0, 16.0],
            'Source': [1, 2],
        }
    )
    out = _normalize_vizier_gaia(raw)
    assert 'ra' in out.colnames
    assert 'dec' in out.colnames
    assert 'phot_g_mean_mag' in out.colnames
    assert 'pmra' in out.colnames
    assert list(out['ra']) == [10.0, 11.0]


def test_normalize_tap_gaia_aliases():
    out = _normalize_tap_gaia(_tap_table())
    assert 'BPmag' in out.colnames
    assert 'RPmag' in out.colnames
    assert 'Epoch' in out.colnames
    assert float(out['BPmag'][0]) == pytest.approx(15.5)


def test_vizier_catalog_for_dr():
    assert _vizier_catalog_for_dr('gaiadr3') == VIZIER_GAIA_DR3
    assert 'gaia2' in _vizier_catalog_for_dr('gaiadr2').lower()


def test_gaia_min_cone_radius_floor():
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    coord = SkyCoord(177.66, 55.35, unit='deg')
    _, r = _apply_min_cone_radius(coord, 0.039 * u.deg)
    assert float(r.to_value(u.deg)) == GAIA_MIN_CONE_RADIUS_DEG
    _, r2 = _apply_min_cone_radius(coord, 0.2 * u.deg)
    assert float(r2.to_value(u.deg)) == pytest.approx(0.2)


def test_default_gaia_cache_dir_under_reduction(tmp_path):
    reduction = tmp_path / 'reduction'
    (reduction / 'raw').mkdir(parents=True)
    img = _toy_sci(reduction / 'reference_prelim' / 'coadd.fits')
    assert default_gaia_cache_dir(img) == reduction / 'gaia'


def test_query_gaia_uses_vizier_backend(tmp_path):
    path = _toy_sci(tmp_path / 'toy_sci.fits')

    viz_table = _vizier_table()
    mock_result = MagicMock()
    mock_result.__len__.return_value = 1
    mock_result.__getitem__.return_value = viz_table

    with (
        patch('astroquery.vizier.Vizier') as mock_viz_cls,
        patch(
            'st123.stages.alignment.gaia_catalog.cut_gaia_sources',
            side_effect=lambda _img, tb: tb,
        ),
    ):
        mock_viz_cls.return_value.query_region.return_value = mock_result
        out = query_gaia(
            str(path), telescope='hst', backend='vizier', use_cache=False
        )

    mock_viz_cls.return_value.query_region.assert_called_once()
    assert len(out) == 1
    assert 'ra' in out.colnames
    assert float(out['ra'][0]) == pytest.approx(150.0)


def test_ensure_gaia_catalog_caches_and_reuses(tmp_path):
    reduction = tmp_path / 'reduction'
    (reduction / 'raw').mkdir(parents=True)
    img1 = _toy_sci(reduction / 'reference_prelim' / 'a.fits', ra=150.0, dec=2.0)
    img2 = _toy_sci(reduction / 'reference_prelim' / 'b.fits', ra=150.01, dec=2.0)
    cache_dir = reduction / 'gaia'

    viz_table = Table(
        {
            'RA_ICRS': [150.0, 150.01],
            'DE_ICRS': [2.0, 2.0],
            'pmRA': [0.0, 0.0],
            'pmDE': [0.0, 0.0],
            'e_pmRA': [0.1, 0.1],
            'e_pmDE': [0.1, 0.1],
            'Gmag': [15.0, 16.0],
            'Source': [1, 2],
        }
    )
    mock_result = MagicMock()
    mock_result.__len__.return_value = 1
    mock_result.__getitem__.return_value = viz_table

    with patch('astroquery.vizier.Vizier') as mock_viz_cls:
        mock_viz_cls.return_value.query_region.return_value = mock_result
        path1 = ensure_gaia_catalog(
            [img1, img2], telescope='hst', cache_dir=cache_dir, backend='vizier'
        )
        path2 = ensure_gaia_catalog(
            [img1, img2], telescope='hst', cache_dir=cache_dir, backend='vizier'
        )

    assert path1 == path2
    assert mock_viz_cls.return_value.query_region.call_count == 1
    ecsv, meta_path, radec = gaia_cache_paths(cache_dir)
    assert ecsv.is_file()
    assert meta_path.is_file()
    assert radec.is_file()
    table, meta = load_gaia_cache(cache_dir)
    assert table is not None and len(table) == 2
    assert meta is not None and meta['n'] == 2

    with (
        patch('astroquery.vizier.Vizier') as mock_viz_cls2,
        patch(
            'st123.stages.alignment.gaia_catalog.cut_gaia_sources',
            side_effect=lambda _img, tb: tb,
        ),
    ):
        out = query_gaia(
            str(img1),
            telescope='hst',
            backend='vizier',
            cache_dir=cache_dir,
        )
    mock_viz_cls2.return_value.query_region.assert_not_called()
    assert len(out) == 2


def test_write_gaia_refcat_from_cache(tmp_path):
    reduction = tmp_path / 'reduction'
    (reduction / 'raw').mkdir(parents=True)
    img = _toy_sci(reduction / 'reference_prelim' / 'coadd.fits')
    cache_dir = reduction / 'gaia'
    viz_table = _vizier_table()
    viz_table['Source'] = [1]
    mock_result = MagicMock()
    mock_result.__len__.return_value = 1
    mock_result.__getitem__.return_value = viz_table
    out = tmp_path / 'ref.phot.txt'
    with patch('astroquery.vizier.Vizier') as mock_viz_cls:
        mock_viz_cls.return_value.query_region.return_value = mock_result
        ensure_gaia_catalog([img], telescope='hst', cache_dir=cache_dir)
        path = write_gaia_refcat(img, out, telescope='hst', cache_dir=cache_dir)
    assert path.is_file()
    text = path.read_text().strip().splitlines()
    assert text[0].startswith('ra dec mag')
    assert len(text) >= 2


def test_fetch_gaia_cone_rejects_unknown_backend():
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    coord = SkyCoord(150.0, 2.0, unit='deg')
    with pytest.raises(ValueError, match='not allowed'):
        fetch_gaia_cone(coord, 0.1 * u.deg, backend='mast')  # type: ignore[arg-type]


def test_fetch_gaia_cone_tap_backend():
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    coord = SkyCoord(150.0, 2.0, unit='deg')
    with patch(
        'st123.stages.alignment.gaia_catalog._query_gaia_tap_chain',
        return_value=(_tap_table(), 'tap'),
    ) as mock_tap:
        out = fetch_gaia_cone(coord, 0.05 * u.deg, backend='tap')
    mock_tap.assert_called_once()
    assert len(out) == 1
    assert out.meta.get('st123_gaia_backend') == 'tap'
    assert float(out['ra'][0]) == pytest.approx(150.0)


def test_fetch_gaia_cone_auto_prefers_aip_cone():
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    coord = SkyCoord(150.0, 2.0, unit='deg')
    with (
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_aip_cone',
            return_value=_tap_table(),
        ) as mock_aip,
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_vizier',
        ) as mock_viz,
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_esa_tap',
        ) as mock_esa,
    ):
        out = fetch_gaia_cone(coord, 0.05 * u.deg, backend='auto')
    mock_aip.assert_called_once()
    mock_viz.assert_not_called()
    mock_esa.assert_not_called()
    assert out.meta.get('st123_gaia_backend') == 'tap_aip_cone'
    assert len(out) == 1


def test_fetch_gaia_cone_auto_falls_back_to_esa_before_vizier():
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    coord = SkyCoord(150.0, 2.0, unit='deg')
    with (
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_aip_cone',
            side_effect=RuntimeError('AIP cone down'),
        ),
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_aip_tap',
            side_effect=RuntimeError('pyvo missing'),
        ),
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_esa_tap',
            return_value=_tap_table(),
        ) as mock_esa,
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_vizier',
        ) as mock_viz,
    ):
        out = fetch_gaia_cone(coord, 0.05 * u.deg, backend='auto')
    mock_esa.assert_called_once()
    mock_viz.assert_not_called()
    assert out.meta.get('st123_gaia_backend') == 'tap'
    assert len(out) == 1


def test_fetch_gaia_cone_tap_prefers_aip_cone():
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    coord = SkyCoord(150.0, 2.0, unit='deg')
    with (
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_aip_cone',
            return_value=_tap_table(),
        ) as mock_aip,
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_aip_tap',
        ) as mock_aip_tap,
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_esa_tap',
        ) as mock_esa,
    ):
        out = fetch_gaia_cone(coord, 0.05 * u.deg, backend='tap')
    mock_aip.assert_called_once()
    mock_aip_tap.assert_not_called()
    mock_esa.assert_not_called()
    assert out.meta.get('st123_gaia_backend') == 'tap_aip_cone'
    assert len(out) == 1


def test_fetch_gaia_cone_tap_falls_through_to_esa():
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    coord = SkyCoord(150.0, 2.0, unit='deg')
    with (
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_aip_cone',
            side_effect=RuntimeError('AIP down'),
        ),
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_aip_tap',
            side_effect=RuntimeError('pyvo missing'),
        ),
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_esa_tap',
            return_value=_tap_table(),
        ) as mock_esa,
    ):
        out = fetch_gaia_cone(coord, 0.05 * u.deg, backend='tap')
    mock_esa.assert_called_once()
    assert out.meta.get('st123_gaia_backend') == 'tap'
    assert len(out) == 1


def test_fetch_gaia_cone_vizier_does_not_fall_back():
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    coord = SkyCoord(150.0, 2.0, unit='deg')
    with (
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_vizier',
            side_effect=RuntimeError('All Vizier mirrors failed'),
        ),
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_tap_chain'
        ) as mock_tap,
        pytest.raises(RuntimeError, match='Vizier mirrors failed'),
    ):
        fetch_gaia_cone(coord, 0.05 * u.deg, backend='vizier')
    mock_tap.assert_not_called()


def test_vizier_retries_then_uses_next_mirror():
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    viz_table = _vizier_table()
    viz_table['Source'] = [1]
    mock_result = MagicMock()
    mock_result.__len__.return_value = 1
    mock_result.__getitem__.return_value = viz_table

    n_calls = {'n': 0}

    def _vizier_factory(*_a, **kwargs):
        inst = MagicMock()
        server = kwargs.get('vizier_server', '')

        def _query(*_qa, **_qk):
            n_calls['n'] += 1
            # Fail first mirror entirely, succeed on second.
            if server == VIZIER_MIRRORS[0]:
                raise ConnectionError('transient')
            return mock_result

        inst.query_region.side_effect = _query
        return inst

    coord = SkyCoord(150.0, 2.0, unit='deg')
    with (
        patch('astroquery.vizier.Vizier', side_effect=_vizier_factory),
        patch('time.sleep'),
    ):
        out = fetch_gaia_cone(coord, 0.05 * u.deg, backend='vizier')
    assert len(out) == 1
    assert n_calls['n'] >= 2
    assert float(out['ra'][0]) == pytest.approx(150.0)
    assert out.meta.get('st123_gaia_backend') == 'vizier'


def test_jhat_gaia_patch_uses_vizier_and_leaves_tap_available():
    viz_table = _vizier_table()
    viz_table['Source'] = [42]
    mock_result = MagicMock()
    mock_result.__len__.return_value = 1
    mock_result.__getitem__.return_value = viz_table

    with (
        # auto prefers AIP/ESA; force the Vizier path for this identity check.
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_aip_cone',
            side_effect=RuntimeError('skip aip cone for test'),
        ),
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_aip_tap',
            side_effect=RuntimeError('skip aip tap for test'),
        ),
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_esa_tap',
            side_effect=RuntimeError('skip esa for test'),
        ),
        patch('astroquery.vizier.Vizier') as mock_viz_cls,
    ):
        mock_viz_cls.return_value.query_region.return_value = mock_result
        install_jhat_gaia_vizier_patch()
        import jhat.simple_jwst_phot as sjp

        assert sjp.get_GAIA_sources.__name__ == jhat_get_gaia_sources.__name__
        df, racol, deccol = sjp.get_GAIA_sources(150.0, 2.0, 0.05, mjd=None)
    assert racol == 'ra' and deccol == 'dec'
    assert len(df) == 1
    assert 'g' in df.columns
    mock_viz_cls.return_value.query_region.assert_called()

    # ESA TAP must remain callable so auto-fallback can use it.
    from astroquery.gaia import Gaia

    assert callable(getattr(Gaia, 'launch_job_async', None))
    assert not getattr(Gaia.launch_job_async, '_st123_tap_blocked', False)


def test_jhat_get_gaia_sources_falls_back_to_tap():
    with (
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_aip_cone',
            side_effect=RuntimeError('AIP cone down'),
        ),
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_aip_tap',
            side_effect=RuntimeError('pyvo missing'),
        ),
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_esa_tap',
            return_value=_normalize_tap_gaia(_tap_table()),
        ) as mock_esa,
        patch(
            'st123.stages.alignment.gaia_catalog._query_gaia_vizier',
        ) as mock_viz,
    ):
        df, racol, deccol = jhat_get_gaia_sources(150.0, 2.0, 0.05, mjd=None)
    mock_esa.assert_called_once()
    mock_viz.assert_not_called()
    assert racol == 'ra' and deccol == 'dec'
    assert len(df) == 1
    assert float(df['ra'].iloc[0]) == pytest.approx(150.0)
