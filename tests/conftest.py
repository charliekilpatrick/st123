"""Shared fixtures for st123 stage / library tests.

Dataset layout directories (``download/``, ``logs/``, ``reduction/``) must
never be created inside the git checkout. Pytest temporary paths live under
``/data/ckilpatrick/st123-pytest`` on this host (elsewhere: pytest's default
temp root). The scratch tree is removed at session end.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from helpers import write_illuminated_fits, write_ref_with_s_region

REPO_ROOT = Path(__file__).resolve().parents[1]
LAYOUT_DIR_NAMES = ('download', 'logs', 'reduction')
DEFAULT_SCRATCH = Path('/data/ckilpatrick/st123-pytest')
_SCRATCH_USED: Path | None = None


def _writable_scratch() -> Path | None:
    """Return a host scratch tree outside the repo, or None (use pytest default)."""
    override = os.environ.get('ST123_PYTEST_SCRATCH')
    candidate = Path(override) if override else DEFAULT_SCRATCH
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / '.write_probe'
        probe.write_text('ok')
        probe.unlink()
    except OSError:
        return None
    # Never nest pytest temps inside the checkout.
    resolved = candidate.resolve()
    if resolved == REPO_ROOT.resolve() or REPO_ROOT.resolve() in resolved.parents:
        return None
    return resolved


def pytest_configure(config: pytest.Config) -> None:
    global _SCRATCH_USED
    if getattr(config.option, 'basetemp', None):
        return
    scratch = _writable_scratch()
    if scratch is None:
        return
    _SCRATCH_USED = scratch
    config.option.basetemp = str(scratch / 'run')


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    scratch = _SCRATCH_USED
    if scratch is None:
        env = os.environ.get('ST123_PYTEST_SCRATCH')
        scratch = Path(env) if env else (
            DEFAULT_SCRATCH if DEFAULT_SCRATCH.parent.is_dir() else None
        )
    if scratch is None or not scratch.is_dir():
        return
    shutil.rmtree(scratch, ignore_errors=True)


@pytest.fixture(autouse=True)
def _st123_no_repo_layout_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep dataset layout dirs out of the git checkout.

    Redirect ``setup_script_logging(None)`` away from ``{repo}/logs`` when cwd
    is the checkout, and refuse ``ensure_project_layout`` on the repo root.
    """
    import st123.scripts.utils.options as options_mod
    import st123.utils.logging as logging_mod

    orig_logs_dir = logging_mod._logs_directory
    orig_layout = options_mod.ensure_project_layout
    repo = REPO_ROOT.resolve()

    def _logs_directory(base_dir: str | Path | None) -> Path:
        if base_dir is None and Path.cwd().resolve() == repo:
            return tmp_path / 'logs'
        return orig_logs_dir(base_dir)

    def _ensure_project_layout(base_dir: str | Path) -> Path:
        resolved = Path(base_dir).expanduser().resolve()
        if resolved == repo:
            pytest.fail(
                f'ensure_project_layout({base_dir!r}) would create '
                'download/, logs/, and reduction/ inside the repository. '
                'Pass tmp_path (pytest scratch under /data/ckilpatrick) as --base-dir.'
            )
        return orig_layout(base_dir)

    monkeypatch.setattr(logging_mod, '_logs_directory', _logs_directory)
    monkeypatch.setattr(options_mod, 'ensure_project_layout', _ensure_project_layout)

    yield
    leaked = [name for name in LAYOUT_DIR_NAMES if (REPO_ROOT / name).is_dir()]
    for name in leaked:
        shutil.rmtree(REPO_ROOT / name, ignore_errors=True)
    if leaked:
        pytest.fail(
            'Test created dataset layout dirs inside the repository root '
            f'{REPO_ROOT}: {", ".join(leaked)}. Use tmp_path / --base-dir under '
            'the pytest scratch tree (see tests/conftest.py).'
        )


@pytest.fixture
def illuminated_fits(tmp_path: Path) -> Path:
    return write_illuminated_fits(tmp_path / 'science_cal.fits', include_s_region=False)


@pytest.fixture
def ref_i2d_fits(tmp_path: Path) -> Path:
    return write_ref_with_s_region(tmp_path / 'coadd_i2d.fits')
