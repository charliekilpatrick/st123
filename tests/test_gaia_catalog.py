"""Tests for Vizier-backed Gaia catalog helpers and field cache."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.table import Table

from st123.alignment.gaia_catalog import (
    GAIA_MIN_CONE_RADIUS_DEG,
    VIZIER_GAIA_DR3,
    VIZIER_MIRRORS,
    _apply_min_cone_radius,
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

    viz_table = Table(
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
    mock_result = MagicMock()
    mock_result.__len__.return_value = 1
    mock_result.__getitem__.return_value = viz_table

    with (
        patch('astroquery.vizier.Vizier') as mock_viz_cls,
        patch(
            'st123.alignment.gaia_catalog.cut_gaia_sources',
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
            'st123.alignment.gaia_catalog.cut_gaia_sources',
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
    viz_table = Table(
        {
            'RA_ICRS': [150.0],
            'DE_ICRS': [2.0],
            'pmRA': [0.0],
            'pmDE': [0.0],
            'e_pmRA': [0.1],
            'e_pmDE': [0.1],
            'Gmag': [15.0],
            'Source': [1],
        }
    )
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


def test_fetch_gaia_cone_rejects_tap_backend():
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    coord = SkyCoord(150.0, 2.0, unit='deg')
    with pytest.raises(ValueError, match='Vizier only'):
        fetch_gaia_cone(coord, 0.1 * u.deg, backend='tap')  # type: ignore[arg-type]


def test_vizier_retries_then_uses_next_mirror():
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    viz_table = Table(
        {
            'RA_ICRS': [150.0],
            'DE_ICRS': [2.0],
            'pmRA': [0.0],
            'pmDE': [0.0],
            'e_pmRA': [0.1],
            'e_pmDE': [0.1],
            'Gmag': [15.0],
            'Source': [1],
        }
    )
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


def test_jhat_gaia_patch_blocks_tap_and_uses_vizier():
    viz_table = Table(
        {
            'RA_ICRS': [150.0],
            'DE_ICRS': [2.0],
            'pmRA': [0.0],
            'pmDE': [0.0],
            'e_pmRA': [0.1],
            'e_pmDE': [0.1],
            'Gmag': [15.0],
            'Source': [42],
        }
    )
    mock_result = MagicMock()
    mock_result.__len__.return_value = 1
    mock_result.__getitem__.return_value = viz_table

    with patch('astroquery.vizier.Vizier') as mock_viz_cls:
        mock_viz_cls.return_value.query_region.return_value = mock_result
        install_jhat_gaia_vizier_patch()
        from astroquery.gaia import Gaia

        with pytest.raises(RuntimeError, match='TAP is disabled'):
            Gaia.launch_job_async('SELECT 1')
        with pytest.raises(RuntimeError, match='TAP is disabled'):
            Gaia.launch_job('SELECT 1')

        import jhat.simple_jwst_phot as sjp

        assert sjp.get_GAIA_sources.__name__ == jhat_get_gaia_sources.__name__
        df, racol, deccol = sjp.get_GAIA_sources(150.0, 2.0, 0.05, mjd=None)
    assert racol == 'ra' and deccol == 'dec'
    assert len(df) == 1
    assert 'g' in df.columns
    mock_viz_cls.return_value.query_region.assert_called()
