"""Tests for st123.utils.logging (POTPyRI-style setup)."""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from astropy.table import Table

from st123.utils.logging import (
    COLORS,
    COLOR_SEQ,
    RESET_SEQ,
    ColoredFormatter,
    FILE_FORMAT,
    STREAM_FORMAT,
    _quiet_third_party_loggers,
    _reset_premature_astropy_logger,
    get_logger,
    setup_script_logging,
    shutdown_logging,
)


@pytest.fixture(autouse=True)
def _clean_st123_logger():
    yield
    shutdown_logging()


def test_setup_creates_dated_uid_log_under_base_dir(tmp_path: Path):
    log_path = setup_script_logging(tmp_path, 'align', verbose=False)
    assert log_path.parent == (tmp_path / 'logs').resolve()
    assert log_path.name.startswith('align_')
    assert log_path.suffix == '.log'
    assert re.match(
        r'^align_\d{8}_\d{6}_[0-9a-f]{8}\.log$',
        log_path.name,
    )
    assert log_path.is_file()
    text = log_path.read_text()
    assert 'Logging to' in text
    assert RESET_SEQ not in text


def test_logger_writes_info_without_ansi(tmp_path: Path):
    log_path = setup_script_logging(tmp_path, 'mosaic')
    logger = get_logger('st123.tests.demo')
    logger.info('hello %s', 'world')
    text = log_path.read_text()
    assert 'hello world' in text
    assert 'INFO' in text
    assert RESET_SEQ not in text
    assert '::' in text


def test_stream_formatter_colors_levelname():
    fmt = ColoredFormatter(STREAM_FORMAT, use_color=True)
    record = logging.LogRecord(
        name='st123.test',
        level=logging.INFO,
        pathname='demo.py',
        lineno=10,
        msg='colored',
        args=(),
        exc_info=None,
    )
    out = fmt.format(record)
    assert COLOR_SEQ % (30 + COLORS['INFO']) in out
    assert 'colored' in out


def test_file_formatter_has_no_ansi_and_utc_converter(tmp_path: Path):
    setup_script_logging(tmp_path, 'align')
    package = logging.getLogger('st123')
    file_handlers = [
        h for h in package.handlers if isinstance(h, logging.FileHandler)
    ]
    assert len(file_handlers) == 1
    formatter = file_handlers[0].formatter
    assert isinstance(formatter, ColoredFormatter)
    assert formatter.converter is time.gmtime
    record = logging.LogRecord(
        name='st123.test',
        level=logging.WARNING,
        pathname='demo.py',
        lineno=3,
        msg='plain',
        args=(),
        exc_info=None,
    )
    out = formatter.format(record)
    assert RESET_SEQ not in out
    assert 'WARNING' in out


def test_second_setup_does_not_duplicate_handlers(tmp_path: Path):
    setup_script_logging(tmp_path, 'align')
    setup_script_logging(tmp_path, 'align')
    package = logging.getLogger('st123')
    assert len(package.handlers) == 2  # stream + file


def test_cwd_logs_when_base_dir_missing(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log_path = setup_script_logging(None, 'region')
    assert log_path.parent == (tmp_path / 'logs').resolve()
    assert log_path.name.startswith('region_')


def test_verbose_enables_debug(tmp_path: Path):
    log_path = setup_script_logging(tmp_path, 'catalog', verbose=True)
    logger = get_logger(__name__)
    logger.debug('debug-line')
    assert 'debug-line' in log_path.read_text()


def test_align_main_creates_log(tmp_path: Path):
    from st123.scripts import align as align_script

    empty = Table(
        {'group': np.array([], dtype=int), 'visit': np.array([], dtype='U8')}
    )
    with (
        patch('st123.alignment.align.get_input_images', return_value=[]),
        patch('st123.scripts.align.input_list', return_value=empty),
        patch('st123.alignment.align.visit_filter_dict', return_value={}),
    ):
        rc = align_script.main(
            ['--base-dir', str(tmp_path), '--mode', 'visit', '--ncores', '1']
        )
    assert rc == 0
    logs = list((tmp_path / 'logs').glob('align_*.log'))
    assert len(logs) == 1
    assert 'Logging to' in logs[0].read_text()


def test_capture_output_routes_stdout_to_log_file(tmp_path: Path):
    from st123.utils.logging import capture_output

    log_path = setup_script_logging(tmp_path, 'capture')
    with capture_output():
        print('jhat-style line')
        print('second line', file=__import__('sys').stderr)
    text = log_path.read_text()
    assert 'jhat-style line' in text
    assert 'second line' in text


def test_run_logged_subprocess_captures_stdout(tmp_path: Path):
    from st123.utils.logging import run_logged_subprocess

    log_path = setup_script_logging(tmp_path, 'subproc')
    result = run_logged_subprocess(
        ['python', '-c', 'print("dolphot-ish")'],
        check=True,
    )
    assert result.returncode == 0
    assert 'dolphot-ish' in log_path.read_text()
    assert 'Running:' in log_path.read_text()


def test_file_handler_always_debug(tmp_path: Path):
    setup_script_logging(tmp_path, 'align', verbose=False)
    package = logging.getLogger('st123')
    file_handlers = [
        h for h in package.handlers if isinstance(h, logging.FileHandler)
    ]
    assert file_handlers[0].level == logging.DEBUG
    stream_handlers = [
        h for h in package.handlers if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    assert stream_handlers[0].level == logging.INFO


def test_quiet_third_party_does_not_precreate_astropy_logger():
    """Regression: pre-creating ``astropy`` breaks ``import astropy``."""
    _reset_premature_astropy_logger()
    # Simulate a bad prior state then ensure quiet repairs it.
    plain = logging.Logger('astropy')
    logging.Logger.manager.loggerDict['astropy'] = plain
    assert not hasattr(logging.getLogger('astropy'), '_set_defaults')
    _quiet_third_party_loggers()
    assert 'astropy' not in logging.Logger.manager.loggerDict


def test_setup_logging_then_astropy_import_in_subprocess(tmp_path: Path):
    """Mosaic-like order: configure logging, then import astropy (fresh interp)."""
    code = f"""
import logging
from st123.utils.logging import setup_script_logging
setup_script_logging({str(tmp_path)!r}, 'mosaic')
assert 'astropy' not in logging.Logger.manager.loggerDict
from astropy.io import fits  # noqa: F401
assert hasattr(logging.getLogger('astropy'), '_set_defaults')
"""
    proc = subprocess.run(
        [sys.executable, '-c', code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
