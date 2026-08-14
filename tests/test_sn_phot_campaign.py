"""Tests for examples/sn_phot campaign MIRI-only JWST skip helpers.

``examples/`` is gitignored (local-only), so these tests skip on CI / clean
checkouts where ``run_campaign.py`` is absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_CAMPAIGN = _REPO / 'examples' / 'sn_phot' / 'run_campaign.py'

pytestmark = pytest.mark.skipif(
    not _CAMPAIGN.is_file(),
    reason='examples/sn_phot/run_campaign.py not present (examples/ is local-only)',
)


def _load_campaign():
    spec = importlib.util.spec_from_file_location('sn_phot_run_campaign', _CAMPAIGN)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def campaign():
    return _load_campaign()


def test_drop_jwst_process_stages(campaign):
    stages = list(campaign.JWST_STAGES)
    dropped = campaign.drop_jwst_process_stages(stages)
    assert 'download-jwst' in dropped
    assert 'download-hst' in dropped
    assert 'align-hst' in dropped
    assert 'align-jwst' not in dropped
    assert 'mosaic-jwst' not in dropped
    assert 'dolphot-jwst' not in dropped
    assert 'dolphot-prep-jwst' not in dropped


def test_campaign_should_skip_jwst_processing(campaign, tmp_path):
    raw = tmp_path / 'reduction' / 'raw'
    raw.mkdir(parents=True)
    (raw / 'jw02666006001_02101_00001_mirimage_cal.fits').write_bytes(b'x')
    assert campaign.campaign_should_skip_jwst_processing(tmp_path)
    assert not campaign.campaign_should_skip_jwst_processing(
        tmp_path, force_miri=True
    )

    empty = tmp_path / 'empty'
    empty.mkdir()
    assert campaign.campaign_should_skip_jwst_processing(empty)


def test_run_parser_accepts_force_miri(campaign):
    parser = campaign.build_parser()
    args = parser.parse_args(['run', 'named', '--force-miri'])
    assert args.force_miri is True


def test_hst_dolphot_prep_omits_outdir(campaign, tmp_path, monkeypatch):
    """Multi-box fields need default per-box outdirs (no --outdir)."""
    recorded: list[list[str]] = []

    def _fake_run(cmd, *, dry_run, cwd=None):
        recorded.append(list(cmd))

    monkeypatch.setattr(campaign, '_run', _fake_run)
    monkeypatch.setattr(campaign, '_which_or_die', lambda name: name)

    ns = type('NS', (), {'ncores': 4, 'dry_run': True})()
    out_dirs = campaign.step_dolphot_prep_hst(tmp_path, ns)
    assert recorded
    assert recorded[0][:3] == ['dolphot-prep', '--instruments', 'hst']
    assert '--outdir' not in recorded[0]
    assert out_dirs == [tmp_path / 'dolphot' / 'hst_0_0']


def test_hst_dolphot_run_uses_run_dolphot(campaign, tmp_path, monkeypatch):
    recorded: list[list[str]] = []

    def _fake_run(cmd, *, dry_run, cwd=None):
        recorded.append(list(cmd))

    monkeypatch.setattr(campaign, '_run', _fake_run)
    monkeypatch.setattr(campaign, '_which_or_die', lambda name: name)

    ns = type('NS', (), {'ncores': 8, 'dry_run': True})()
    campaign.step_run_dolphot_hst(tmp_path, ns)
    assert recorded == [
        [
            'run-dolphot',
            '--base-dir',
            str(tmp_path),
            '--instruments',
            'hst',
            '--ncores',
            '8',
            '-v',
        ]
    ]
