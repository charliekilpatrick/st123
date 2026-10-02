"""Enforce ASCII-only Python source under ``st123/`` and ``tests/``.

Ruff has no global ASCII-source rule. ``PLC2401`` / ``PLC2403`` catch
non-ASCII names and ``RUF001``-``RUF003`` catch ambiguous homoglyphs.
This module scans every ``.py`` byte for codepoints above 127.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

SKIP_PARTS = frozenset({
    '__pycache__',
    'notebooks',
})


def _iter_python_files(root: Path = REPO_ROOT) -> list[Path]:
    files: list[Path] = []
    for directory in (root / 'st123', root / 'tests'):
        if not directory.is_dir():
            continue
        for path in directory.rglob('*.py'):
            if SKIP_PARTS.intersection(path.parts):
                continue
            files.append(path)
    return sorted(files)


def find_non_ascii_lines(root: Path = REPO_ROOT) -> list[str]:
    """Return ``relative:line: ...`` entries for non-ASCII source lines."""
    violations: list[str] = []
    for path in _iter_python_files(root):
        try:
            text = path.read_text(encoding='utf-8')
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.isascii():
                continue
            chars = sorted({ch for ch in line if not ch.isascii()})
            shown = ', '.join(f'U+{ord(ch):04X}' for ch in chars)
            violations.append(f'{rel}:{lineno}: non-ASCII {shown}')
    return violations


def test_find_non_ascii_lines_reports_hits(tmp_path: Path):
    (tmp_path / 'st123').mkdir()
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'st123' / 'bad.py').write_text(
        '# em dash \u2014 here\n', encoding='utf-8'
    )
    (tmp_path / 'st123' / 'ok.py').write_text(
        "curly = '\\u201cquoted\\u201d'\n", encoding='utf-8'
    )
    hits = find_non_ascii_lines(tmp_path)
    assert hits == ['st123/bad.py:1: non-ASCII U+2014']


def test_python_source_is_ascii():
    violations = find_non_ascii_lines()
    assert not violations, (
        'Python source under st123/ and tests/ must be ASCII-only. '
        'Use --, ..., ->, and ASCII quotes, or \\uXXXX escapes:\n  '
        + '\n  '.join(violations)
    )


def test_ruff_unicode_rules():
    """Run ruff's name / homoglyph Unicode rules when ruff is installed."""
    ruff = shutil.which('ruff')
    if ruff is None:
        pytest.skip('ruff is not on PATH; install the [dev] extra')
    result = subprocess.run(
        [ruff, 'check', 'st123', 'tests'],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        output = (result.stdout or '') + (result.stderr or '')
        pytest.fail(f'ruff Unicode rules failed:\n{output}')
