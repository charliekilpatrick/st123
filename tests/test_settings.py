"""Unit tests for ``st123.utils.settings`` defaults and filter catalogs."""

from __future__ import annotations

from st123.utils import settings


def test_acceptable_filters_covers_all_missions():
    """Header-style uppercase names for each supported facility."""
    for name in (
        'F606W',  # HST shared
        'F275W',  # WFC3/UVIS
        'F160W',  # WFC3/IR
        'F200W',  # NIRCam
        'F770W',  # MIRI
        'F062',  # Roman/WFI
        'VIS',  # Euclid
        'YE',
        'NIR_J',
    ):
        assert name in settings.acceptable_filters


def test_filters_by_instrument_telescope_tags():
    expected = {
        'WFPC2': 'HST',
        'ACS': 'HST',
        'WFC3': 'HST',
        'NIRCAM': 'JWST',
        'MIRI': 'JWST',
        'WFI': 'Roman',
        'VIS': 'Euclid',
        'NISP': 'Euclid',
    }
    for instrument, telescope in expected.items():
        meta = settings.FILTERS_BY_INSTRUMENT[instrument]
        assert meta['telescope'] == telescope
        assert len(meta['filters']) > 0


def test_acceptable_filters_is_union_of_instrument_lists():
    from_groups = set()
    for key in ('WFPC2', 'ACS', 'WFC3', 'NIRCAM', 'MIRI', 'WFI'):
        from_groups.update(settings.FILTERS_BY_INSTRUMENT[key]['filters'])
    from_groups.update(settings.EUCLID_FILTERS)
    assert set(settings.acceptable_filters) == from_groups


def test_acceptable_filters_unique_and_uppercase():
    assert len(settings.acceptable_filters) == len(set(settings.acceptable_filters))
    assert all(name == name.upper() for name in settings.acceptable_filters)


def test_best_reference_filters_are_lowercase_subset():
    assert settings.BEST_REFERENCE_FILTERS[0] == 'f606w'
    for filt in settings.BEST_REFERENCE_FILTERS:
        assert filt == filt.lower()
        assert filt.upper() in settings.acceptable_filters


def test_mast_and_alignment_defaults():
    assert settings.DEFAULT_HST_FILTERS == ('F275W', 'F555W', 'F814W')
    assert 'NIRCAM' in settings.DEFAULT_JWST_INSTRUMENTS
    assert settings.DEFAULT_DOWNLOAD_LAYOUT == 'telescope/instrument/filter/obsid'
    assert settings.DEFAULT_MAX_REFERENCE_DISPERSION_MAS == 70.0
    assert settings.FILTER_MAX_REFERENCE_DISPERSION_MAS['F560W'] is None
    assert settings.FILTER_MAX_REFERENCE_DISPERSION_MAS['F1000W'] == 35.0
    assert settings.DEFAULT_PAIR_OUTDIR == 'alignment_output'


def test_jhat_and_dolphot_param_dicts():
    assert settings.strict_jwst_params['refcat_racol'] == 'ra'
    assert settings.base_params['FitSky'] == '2'
    assert settings.miri_params['raper'] == '3'
    assert settings.nircam_calcsky_params['rin'] == 15
    assert settings.miri_calcsky_params['rin'] == 10
