"""Validate console-script registration and script module entry points."""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points

import pytest

EXPECTED_SCRIPTS = {
    'download': 'st123.scripts.download:main',
    'jwst-download': 'st123.scripts.download:main_jwst_download',
    'align': 'st123.scripts.align:main',
    'mosaic': 'st123.scripts.mosaic:main',
    'link-raw': 'st123.scripts.link_raw:main',
    'image-overlap': 'st123.scripts.image_overlap:main',
    'region': 'st123.scripts.region:main',
    'illuminated-s-region': 'st123.scripts.region:main',
    'relative-align': 'st123.scripts.relative_align:main',
    'alignment-wrap': 'st123.scripts.alignment_wrap:main',
    'apply-gwcs': 'st123.scripts.apply_gwcs:main',
    'catalog': 'st123.scripts.catalog:main',
}

SCRIPT_MODULES = [
    'st123.scripts.download',
    'st123.scripts.jwst_download',
    'st123.scripts.align',
    'st123.scripts.mosaic',
    'st123.scripts.link_raw',
    'st123.scripts.image_overlap',
    'st123.scripts.region',
    'st123.scripts.relative_align',
    'st123.scripts.alignment_wrap',
    'st123.scripts.apply_gwcs',
    'st123.scripts.catalog',
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
