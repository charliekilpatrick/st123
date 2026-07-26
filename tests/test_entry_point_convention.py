"""Enforce that executable ``__main__`` blocks live under ``st123/scripts/``.

Vendored code under ``extdeps/`` and tests are ignored.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Only allowed location for ``if __name__ == "__main__"`` entry points.
ALLOWED_MAIN_PREFIXES = (
    'st123/scripts/',
)

SKIP_PREFIXES = (
    'extdeps/',
    'tests/',
    '.cursor/',
    '.git/',
    'build/',
    'dist/',
)

MAIN_PATTERN = re.compile(
    r"""if\s+__name__\s*==\s*['\"]__main__['\"]""",
)


def _iter_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob('*.py'):
        rel = path.relative_to(root).as_posix()
        if any(rel.startswith(p) for p in SKIP_PREFIXES):
            continue
        if '__pycache__' in path.parts:
            continue
        files.append(path)
    return sorted(files)


def find_disallowed_main_blocks(root: Path = REPO_ROOT) -> list[str]:
    """Return human-readable violations for ``__main__`` blocks outside scripts."""
    violations: list[str] = []
    for path in _iter_python_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding='utf-8')
        except OSError:
            continue
        if not MAIN_PATTERN.search(text):
            continue
        if any(rel.startswith(p) for p in ALLOWED_MAIN_PREFIXES):
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            violations.append(f'{rel}: contains __main__ guard but failed to parse')
            continue
        has_main = False
        for node in tree.body:
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == '__name__'
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == '__main__'
            ):
                has_main = True
                break
        if has_main:
            violations.append(
                f'{rel}: move the ``if __name__ == "__main__"`` entry point to '
                f'st123/scripts/'
            )
    return violations


def test_no_main_blocks_outside_st123_scripts():
    violations = find_disallowed_main_blocks()
    assert not violations, (
        'Entry-point convention violated:\n  - '
        + '\n  - '.join(violations)
        + '\n\nAll runnable CLIs with a ``__main__`` block must live under '
        'st123/scripts/.'
    )


def test_required_cli_modules_exist():
    """Preserve packaged CLI modules under st123/scripts/."""
    assert (REPO_ROOT / 'st123' / 'scripts' / 'align.py').is_file()
    assert (REPO_ROOT / 'st123' / 'scripts' / 'download.py').is_file()
    assert (REPO_ROOT / 'st123' / 'scripts' / 'utils' / 'options.py').is_file()
    # Shared helpers must not sit beside entry-point modules.
    assert not (REPO_ROOT / 'st123' / 'scripts' / 'options.py').exists()
    assert not (REPO_ROOT / 'st123' / 'scripts' / 'alignment_wrap.py').exists()
    assert not (REPO_ROOT / 'st123' / 'scripts' / 'relative_align.py').exists()
    assert not (REPO_ROOT / 'st123' / 'scripts' / 'jwst_download.py').exists()
    # Alignment library is a single module (no wrap/relative/parallel split).
    align_pkg = REPO_ROOT / 'st123' / 'alignment'
    assert (align_pkg / 'align.py').is_file()
    for gone in (
        'relative_align.py',
        'alignment_wrap.py',
        'alignment_parallel.py',
        'alignment_fallback.py',
        'calibrators.py',
    ):
        assert not (align_pkg / gone).exists(), gone
    # Repo-root scripts/ must not be reintroduced.
    assert not (REPO_ROOT / 'scripts').exists()


@pytest.mark.parametrize(
    'module_name',
    [
        'st123.scripts.align',
        'st123.scripts.download',
    ],
)
def test_packaged_entry_modules_expose_main(module_name: str):
    import importlib

    mod = importlib.import_module(module_name)
    assert callable(getattr(mod, 'main', None))
    assert callable(getattr(mod, 'create_parser', None))
