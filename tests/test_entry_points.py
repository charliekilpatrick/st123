"""Validate console-script registration and script module entry points."""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points

import pytest

EXPECTED_SCRIPTS = {
    'download': 'st123.scripts.download:main',
    'align': 'st123.scripts.align:main',
    'mosaic': 'st123.scripts.mosaic:main',
    'link-raw': 'st123.scripts.link_raw:main',
    'image-overlap': 'st123.scripts.image_overlap:main',
    'region': 'st123.scripts.region:main',
    'catalog': 'st123.scripts.catalog:main',
    'dolphot-prep': 'st123.scripts.dolphot:main',
    'run-dolphot': 'st123.scripts.run_dolphot:main',
    'dolphot-hdf5': 'st123.scripts.dolphot_hdf5:main',
    'dolphot-warmstart-prep': 'st123.scripts.dolphot_warmstart:main',
    'dolphot-warmstart': 'st123.scripts.dolphot_warmstart:main',
    'coadd-phot': 'st123.scripts.coadd_phot:main',
}

SCRIPT_MODULES = [
    'st123.scripts.download',
    'st123.scripts.align',
    'st123.scripts.mosaic',
    'st123.scripts.link_raw',
    'st123.scripts.image_overlap',
    'st123.scripts.region',
    'st123.scripts.catalog',
    'st123.scripts.dolphot',
    'st123.scripts.run_dolphot',
    'st123.scripts.dolphot_hdf5',
    'st123.scripts.dolphot_warmstart',
    'st123.scripts.coadd_phot',
]


def test_console_scripts_registered():
    eps = entry_points()
    group = eps.select(group='console_scripts') if hasattr(eps, 'select') else eps.get(
        'console_scripts', []
    )
    by_name = {ep.name: ep.value for ep in group}
    for name, value in EXPECTED_SCRIPTS.items():
        assert name in by_name, f'missing console script: {name}'
        assert by_name[name] == value


@pytest.mark.parametrize('module_name', SCRIPT_MODULES)
def test_script_modules_expose_main_and_parser(module_name):
    mod = importlib.import_module(module_name)
    assert callable(getattr(mod, 'main', None)), f'{module_name} missing main'
    assert callable(getattr(mod, 'create_parser', None)), f'{module_name} missing create_parser'
    parser = mod.create_parser()
    # --help should not SystemExit with code other than 0 when we catch it
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(['--help'])
    assert exc.value.code == 0
