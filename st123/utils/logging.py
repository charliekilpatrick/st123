"""
POTPyRI-style logging for st123.

Configures the ``st123`` logger hierarchy with a colored console handler and a
UTC file handler. Library code should use::

    import logging
    logger = logging.getLogger(__name__)

Scripts call :func:`setup_script_logging` (or
:func:`st123.scripts.utils.options.configure_logging_from_args`) once in
``main()``.

External tools that write to stdout/stderr (JHAT, ``jwst`` Image3, DOLPHOT
binaries, MAST clients) should run under :func:`capture_output` or
:func:`run_logged_subprocess` so their chatter is recorded in the log file
(DEBUG) without flooding the console unless ``--verbose``.

Log files land in ``{base-dir}/logs/{script}_{YYYYMMDD_HHMMSS}_{uid}.log``.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import time
import uuid
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Sequence

# Foreground ANSI color indices (30 + n).
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

RESET_SEQ = '\033[0m'
COLOR_SEQ = '\033[1;%dm'
BOLD_SEQ = '\033[1m'

COLORS = {
    'WARNING': YELLOW,
    'INFO': GREEN,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED,
}

PACKAGE_LOGGER_NAME = 'st123'
EXTERNAL_LOGGER_NAME = f'{PACKAGE_LOGGER_NAME}.external'
LOG_FILE_ENV = 'ST123_LOG_FILE'
_QUIET_THIRD_PARTY = (
    'stpipe',
    'jwst',
    'CRDS',
    'pysiaf',
    'astropy',
    'ccdproc',
)

_ST_FMT = '[$BOLD%(filename)s::%(lineno)d$RESET] [%(levelname)s]  %(message)s'
_F_FMT = '[$BOLD%(asctime)s::%(filename)s::%(lineno)d$RESET] [%(levelname)s] %(message)s'


def formatter_message(message: str, use_color: bool = True) -> str:
    """
    Replace ``$RESET`` / ``$BOLD`` placeholders in a format string.

    Parameters
    ----------
    message : str
        Format string containing ``$RESET`` and/or ``$BOLD`` tokens.
    use_color : bool, optional
        If True, substitute ANSI escape codes; otherwise strip the tokens.

    Returns
    -------
    str
        Format string with placeholders expanded or removed.
    """
    if use_color:
        return message.replace('$RESET', RESET_SEQ).replace('$BOLD', BOLD_SEQ)
    return message.replace('$RESET', '').replace('$BOLD', '')


STREAM_FORMAT = formatter_message(_ST_FMT, True)
FILE_FORMAT = formatter_message(_F_FMT, False)


class ColoredFormatter(logging.Formatter):
    """
    :class:`logging.Formatter` that optionally colors ``levelname``.

    Parameters
    ----------
    msg : str
        Log record format string (may include ANSI placeholders).
    use_color : bool, optional
        Colorize the level name when True.
    """

    def __init__(self, msg: str, use_color: bool = True) -> None:
        super().__init__(msg)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        # Copy so concurrent handlers do not share a mutated levelname.
        record = logging.makeLogRecord(record.__dict__)
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            record.levelname = (
                COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            )
        elif not self.use_color and RESET_SEQ in levelname:
            record.levelname = levelname[7:].replace(RESET_SEQ, '')
        return super().format(record)


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return a logger under the ``st123`` hierarchy.

    Parameters
    ----------
    name : str or None, optional
        Logger name. ``None`` returns the package root logger. Names without
        an ``st123.`` prefix are registered as ``st123.<name>``.

    Returns
    -------
    logging.Logger
        Logger instance under the ``st123`` namespace.
    """
    if name is None or name == PACKAGE_LOGGER_NAME:
        return logging.getLogger(PACKAGE_LOGGER_NAME)
    if name.startswith(f'{PACKAGE_LOGGER_NAME}.'):
        return logging.getLogger(name)
    return logging.getLogger(f'{PACKAGE_LOGGER_NAME}.{name}')


def _logs_directory(base_dir: str | Path | None) -> Path:
    """Return ``{project_root}/logs`` or ``{cwd}/logs`` when base_dir is unset."""
    if base_dir is None:
        return Path.cwd() / 'logs'
    from st123.scripts.utils.options import resolve_project_root

    return resolve_project_root(Path(base_dir)) / 'logs'


def _make_log_path(log_dir: Path, script_name: str) -> Path:
    datestr = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    uid = uuid.uuid4().hex[:8]
    safe_script = str(script_name).strip().replace(' ', '-') or 'st123'
    return log_dir / f'{safe_script}_{datestr}_{uid}.log'


def _clear_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)


def _reset_premature_astropy_logger() -> None:
    """
    Drop a plain ``logging.Logger('astropy')`` created before Astropy loads.

    Astropy's ``_init_log`` requires ``AstropyLogger`` (with ``_set_defaults``).
    Calling ``logging.getLogger('astropy')`` earlier freezes a standard Logger
    in the manager cache and breaks the next ``import astropy``.
    """
    mgr = logging.Logger.manager
    existing = mgr.loggerDict.get('astropy')
    if existing is None or isinstance(existing, logging.PlaceHolder):
        return
    if hasattr(existing, '_set_defaults'):
        return
    try:
        for handler in list(getattr(existing, 'handlers', [])):
            try:
                existing.removeHandler(handler)
            except Exception:
                pass
    except Exception:
        pass
    mgr.loggerDict.pop('astropy', None)


def _quiet_third_party_loggers() -> None:
    """Raise third-party logger levels; never pre-create Astropy's logger."""
    _reset_premature_astropy_logger()
    mgr = logging.Logger.manager
    for name in _QUIET_THIRD_PARTY:
        # Do not instantiate ``astropy`` before AstropyLogger is registered.
        if name == 'astropy' and 'astropy' not in mgr.loggerDict:
            continue
        logging.getLogger(name).setLevel(logging.WARNING)


def _file_formatter() -> ColoredFormatter:
    fmt = ColoredFormatter(FILE_FORMAT, use_color=False)
    fmt.converter = time.gmtime
    return fmt


class _LineLoggerIO(io.TextIOBase):
    """Text stream that forwards complete lines to a :class:`logging.Logger`."""

    encoding = 'utf-8'

    def __init__(self, logger: logging.Logger, level: int):
        super().__init__()
        self._logger = logger
        self._level = level
        self._buf = ''

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:  # type: ignore[override]
        if not s:
            return 0
        if not isinstance(s, str):
            s = s.decode('utf-8', errors='replace')
        self._buf += s
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            line = line.rstrip('\r')
            if line:
                self._logger.log(self._level, '%s', line)
        return len(s)

    def flush(self) -> None:
        if self._buf:
            line = self._buf.rstrip('\r\n')
            self._buf = ''
            if line:
                self._logger.log(self._level, '%s', line)


@contextmanager
def capture_output(
    *,
    logger: logging.Logger | str | None = None,
    stdout_level: int = logging.DEBUG,
    stderr_level: int = logging.DEBUG,
    discard: bool = False,
) -> Iterator[None]:
    """
    Redirect ``stdout`` / ``stderr`` into the logging system (or discard).

    Parameters
    ----------
    logger : logging.Logger, str, or None, optional
        Destination logger (default: ``st123.external``). Ignored when
        ``discard`` is True.
    stdout_level : int, optional
        Log level for stdout lines (DEBUG is persisted in the log file even
        when the console handler is INFO).
    stderr_level : int, optional
        Log level for stderr lines.
    discard : bool, optional
        If True, send streams to ``os.devnull`` (import-time quieting only).

    Yields
    ------
    None
    """
    if discard:
        with open(os.devnull, 'w', encoding='utf-8') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                previous_disable = logging.root.manager.disable
                logging.disable(logging.CRITICAL)
                try:
                    yield
                finally:
                    logging.disable(previous_disable)
        return

    if isinstance(logger, logging.Logger):
        log = logger
    elif isinstance(logger, str):
        log = logging.getLogger(logger)
    else:
        log = logging.getLogger(EXTERNAL_LOGGER_NAME)

    out = _LineLoggerIO(log, stdout_level)
    err = _LineLoggerIO(log, stderr_level)
    with redirect_stdout(out), redirect_stderr(err):
        try:
            yield
        finally:
            out.flush()
            err.flush()


def suppress_output() -> Iterator[None]:
    """
    Discard stdout and stderr (import-time banners only).

    Prefer :func:`capture_output` for subprocess or runtime capture.

    Yields
    ------
    None
    """
    return capture_output(discard=True)


def run_logged_subprocess(
    cmd: Sequence[str],
    *,
    logger: logging.Logger | None = None,
    check: bool = True,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    label: str | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """
    Run a subprocess, logging the command and captured stdout/stderr.

    stdout/stderr lines are logged at DEBUG; a non-zero exit also logs stderr
    at ERROR (or raises if ``check`` is True).

    Parameters
    ----------
    cmd : sequence of str
        Command argv passed to :func:`subprocess.run`.
    logger : logging.Logger or None, optional
        Logger for command and output lines (default: ``st123.external``).
    check : bool, optional
        Raise :class:`subprocess.CalledProcessError` on non-zero exit.
    cwd : str, Path, or None, optional
        Working directory for the subprocess.
    env : dict of str to str or None, optional
        Environment overrides for the subprocess.
    label : str or None, optional
        Human-readable command label for the INFO log line; defaults to
        ``' '.join(cmd)``.
    **kwargs
        Additional keyword arguments forwarded to :func:`subprocess.run`.

    Returns
    -------
    subprocess.CompletedProcess
        Completed process with ``stdout`` and ``stderr`` captured as text.
    """
    log = logger or logging.getLogger(EXTERNAL_LOGGER_NAME)
    display = label or ' '.join(str(c) for c in cmd)
    log.info('Running: %s', display)
    result = subprocess.run(
        list(cmd),
        check=False,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        **kwargs,
    )
    if result.stdout:
        for line in result.stdout.splitlines():
            if line:
                log.debug('%s', line)
    if result.stderr:
        level = logging.ERROR if result.returncode else logging.DEBUG
        for line in result.stderr.splitlines():
            if line:
                log.log(level, '%s', line)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def configure_worker_logging() -> None:
    """
    Attach an append FileHandler in spawn workers (same path as the parent).

    Reads :data:`LOG_FILE_ENV`. No-op when unset or already configured.
    Workers do not attach a StreamHandler (parent owns the console).

    Returns
    -------
    None
    """
    path = os.environ.get(LOG_FILE_ENV)
    if not path:
        return
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    for handler in package_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == Path(path).resolve():
                    return
            except Exception:
                pass

    package_logger.setLevel(logging.DEBUG)
    package_logger.propagate = False
    file_handler = logging.FileHandler(path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_file_formatter())
    package_logger.addHandler(file_handler)
    _quiet_third_party_loggers()


def setup_script_logging(
    base_dir: str | Path | None,
    script_name: str,
    *,
    verbose: bool = False,
) -> Path:
    """
    Attach colored stream + UTC file handlers to the ``st123`` logger.

    The file handler always accepts DEBUG so external-tool capture is
    persisted; the console stays INFO unless ``verbose``.

    Parameters
    ----------
    base_dir : str, Path, or None
        Project root used to resolve ``{base_dir}/logs/``; ``None`` uses
        ``{cwd}/logs/``.
    script_name : str
        Script stem embedded in the log filename.
    verbose : bool, optional
        If True, emit DEBUG on the console as well as the log file.

    Returns
    -------
    Path
        Absolute path to the new log file.
    """
    from st123.scripts.utils.options import ensure_writable_dir

    log_dir = ensure_writable_dir(_logs_directory(base_dir), label='logs')
    log_path = _make_log_path(log_dir, script_name).resolve()

    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    _clear_handlers(package_logger)
    package_logger.setLevel(logging.DEBUG)
    package_logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream_handler.setFormatter(ColoredFormatter(STREAM_FORMAT, use_color=True))

    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_file_formatter())

    package_logger.addHandler(stream_handler)
    package_logger.addHandler(file_handler)
    _quiet_third_party_loggers()

    os.environ[LOG_FILE_ENV] = str(log_path)
    package_logger.info('Logging to %s', log_path)
    return log_path


def shutdown_logging() -> None:
    """
    Flush and close handlers attached to the ``st123`` logger.

    Clears :data:`LOG_FILE_ENV` from the environment.

    Returns
    -------
    None
    """
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    _clear_handlers(package_logger)
    os.environ.pop(LOG_FILE_ENV, None)
