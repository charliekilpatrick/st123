"""Repo checkout must not grow dataset layout dirs during tests."""

from __future__ import annotations

from pathlib import Path

from st123.scripts.utils.options import ensure_project_layout

REPO_ROOT = Path(__file__).resolve().parents[1]
LAYOUT_DIR_NAMES = ('download', 'logs', 'reduction')


def test_repo_root_has_no_download_logs_reduction():
    for name in LAYOUT_DIR_NAMES:
        assert not (REPO_ROOT / name).exists(), (
            f'{name}/ must not exist in the repository root; tests use tmp_path'
        )


def test_pytest_tmp_path_is_outside_repo(tmp_path: Path):
    repo = REPO_ROOT.resolve()
    scratch = tmp_path.resolve()
    assert scratch != repo
    assert repo not in scratch.parents
    host = Path('/data/ckilpatrick')
    if host.is_dir():
        assert host in scratch.parents or scratch.parent == host


def test_ensure_project_layout_uses_tmp_not_repo(tmp_path: Path):
    project = tmp_path / 'dataset'
    ensure_project_layout(project)
    for name in LAYOUT_DIR_NAMES:
        assert (project / name).is_dir()
        assert not (REPO_ROOT / name).exists()
