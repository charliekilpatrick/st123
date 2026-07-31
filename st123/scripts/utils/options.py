"""
Unified CLI options for st123 console entry points.

Canonical path / runtime flags
------------------------------
``--base-dir``
    Primary project / dataset directory. The object name is the basename
    of this path (e.g. ``.../NGC3310``). There is no separate ``--obj``.
``--ncores``
    Parallel worker count (alias: ``--workers``).
``--plot`` / ``--verbose`` / ``--version`` / ``--dry-run``
    Shared runtime switches.

Legacy directory aliases (``--workdir``, ``--data-dir``, ``--download-dir``,
``--basedir``, ``--outdir``) map onto ``--base-dir``.

Image targets
-------------
When a CLI needs one or more science / target FITS paths, use ``--image``
(via :func:`add_image_arg`). Do **not** invent instrument- or verb-specific
path flags such as ``--miri`` or ``--align`` for that role.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from st123 import __version__ as ST123_VERSION

# ---------------------------------------------------------------------------
# Path layout helpers (project root vs reduction workdir)
# ---------------------------------------------------------------------------

_REDUCTION_MARKERS = ('raw', 'jhat', 'align', 'reference')


def as_path(value: object | None) -> Path | None:
    """
    Convert a CLI path value to :class:`~pathlib.Path`, or ``None``.

    Parameters
    ----------
    value : object or None
        Raw path-like value from argparse (or ``None``).

    Returns
    -------
    pathlib.Path or None
        Expanded path when *value* is not ``None``.
    """
    if value is None:
        return None
    return Path(str(value)).expanduser()


def resolve_base_dir(value: object | None, *, default: str | Path | None = None) -> Path:
    """
    Return an absolute ``--base-dir`` path.

    Parameters
    ----------
    value : object or None
        User-supplied ``--base-dir`` value.
    default : str or pathlib.Path or None, optional
        Fallback when *value* is ``None``.

    Returns
    -------
    pathlib.Path
        Resolved absolute path.

    Raises
    ------
    ValueError
        If both *value* and *default* are ``None``.
    """
    if value is None:
        if default is None:
            raise ValueError('--base-dir is required')
        value = default
    return Path(str(value)).expanduser().resolve()


def looks_like_reduction_dir(path: Path) -> bool:
    """
    Return True if *path* already looks like a reduction/workdir root.

    Parameters
    ----------
    path : pathlib.Path
        Candidate directory.

    Returns
    -------
    bool
        True when any of ``raw/``, ``jhat/``, ``align/``, or ``reference/``
        exists under *path*.
    """
    return any((path / name).exists() for name in _REDUCTION_MARKERS)


def resolve_reduction_dir(base_dir: Path) -> Path:
    """
    Map ``--base-dir`` to the NIRCam reduction workdir.

    * If ``base_dir`` looks like a project root (``JWST/`` and/or
      ``reduction/`` present), return ``base_dir / 'reduction'``. This is
      checked before workdir markers so a project-level ``reference/``
      symlink (from :func:`ensure_dataset_reference_link`) does not make
      the project root look like the reduction workdir.
    * Else if ``base_dir`` already contains ``raw/``, ``jhat/``, ``align/``,
      or ``reference/``, it is the reduction root (legacy ``--workdir``).
    * Otherwise treat ``base_dir`` itself as the reduction workdir.

    Parameters
    ----------
    base_dir : pathlib.Path
        Project root or reduction workdir from ``--base-dir``.

    Returns
    -------
    pathlib.Path
        Absolute reduction workdir path.
    """
    base = Path(base_dir).expanduser().resolve()
    if (base / 'JWST').is_dir() or (base / 'reduction').is_dir():
        return (base / 'reduction').resolve()
    if looks_like_reduction_dir(base):
        return base
    return base


def resolve_project_root(base_dir: Path) -> Path:
    """
    Map ``--base-dir`` to the dataset / project root (parent of ``JWST/``).

    If ``base_dir`` is a reduction workdir named ``reduction``, return its
    parent; otherwise return ``base_dir`` itself.

    Parameters
    ----------
    base_dir : pathlib.Path
        Project root or reduction workdir from ``--base-dir``.

    Returns
    -------
    pathlib.Path
        Absolute project root path.
    """
    base = Path(base_dir).expanduser().resolve()
    if base.name == 'reduction' and (
        looks_like_reduction_dir(base) or not (base / 'JWST').is_dir()
    ):
        return base.parent
    return base


def instrument_raw_dir(base_dir: Path, instrument: str) -> Path:
    """
    Return ``{project}/JWST/{Instrument}`` for symlink sources.

    Parameters
    ----------
    base_dir : pathlib.Path
        Project root or reduction workdir from ``--base-dir``.
    instrument : str
        Instrument name (e.g. ``NIRCAM``, ``MIRI``).

    Returns
    -------
    pathlib.Path
        Absolute ``JWST/<Instrument>`` directory under the project root.
    """
    root = resolve_project_root(base_dir)
    key = str(instrument).strip().upper()
    if key in ('NIRCAM', 'NRC'):
        name = 'NIRCam'
    elif key == 'MIRI':
        name = 'MIRI'
    elif key == 'NIRISS':
        name = 'NIRISS'
    else:
        name = instrument
    return root / 'JWST' / name


def dataset_label(base_dir: Path | str) -> str:
    """
    Return the object / dataset name from ``--base-dir`` (project basename).

    Parameters
    ----------
    base_dir : pathlib.Path or str
        Project root or reduction workdir.

    Returns
    -------
    str
        Basename of the resolved project root.
    """
    return resolve_project_root(base_dir).name


def default_phot_dir(base_dir: Path, *, group: int = 0, box: int = 0) -> Path:
    """
    Return ``{reduction}/phot_{group}_{box}`` (no extra object subdirectory).

    Parameters
    ----------
    base_dir : pathlib.Path
        Project root or reduction workdir from ``--base-dir``.
    group : int, optional
        Mosaic overlap group index.
    box : int, optional
        Mosaic box index within the group.

    Returns
    -------
    pathlib.Path
        Default NIRCam DOLPHOT staging directory.
    """
    return resolve_reduction_dir(base_dir) / f'phot_{group}_{box}'


def default_warmstart_outdir(base_dir: Path, *, group: int = 0, box: int = 0) -> Path:
    """
    Return ``{project}/dolphot/nircam_miri_{group}_{box}``.

    Parameters
    ----------
    base_dir : pathlib.Path
        Project root or reduction workdir from ``--base-dir``.
    group : int, optional
        Mosaic overlap group index.
    box : int, optional
        Mosaic box index within the group.

    Returns
    -------
    pathlib.Path
        Default warm-start DOLPHOT output directory.
    """
    return resolve_project_root(base_dir) / 'dolphot' / f'nircam_miri_{group}_{box}'


def default_miri_outdir(base_dir: Path, *, group: int = 0, box: int = 0) -> Path:
    """
    Return ``{project}/dolphot/miri_{group}_{box}``.

    Parameters
    ----------
    base_dir : pathlib.Path
        Project root or reduction workdir from ``--base-dir``.
    group : int, optional
        Mosaic overlap group index.
    box : int, optional
        Mosaic box index within the group.

    Returns
    -------
    pathlib.Path
        Default MIRI-only DOLPHOT output directory.
    """
    return resolve_project_root(base_dir) / 'dolphot' / f'miri_{group}_{box}'


def default_alignment_summary(base_dir: Path) -> Path:
    """
    Return ``{project}/{label}_alignment_summary.txt``.

    Parameters
    ----------
    base_dir : pathlib.Path
        Project root or reduction workdir from ``--base-dir``.

    Returns
    -------
    pathlib.Path
        Default alignment summary table path.
    """
    root = resolve_project_root(base_dir)
    return root / f'{root.name}_alignment_summary.txt'


# ---------------------------------------------------------------------------
# Parser factory
# ---------------------------------------------------------------------------


def st123_version_string() -> str:
    return f'st123 {ST123_VERSION}'


def create_parser(
    description: str,
    *,
    prog: str | None = None,
    epilog: str | None = None,
) -> argparse.ArgumentParser:
    """Create a parser with ``--version`` pre-attached."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        epilog=epilog,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--version',
        action='version',
        version=st123_version_string(),
    )
    return parser


def add_common_runtime(
    parser: argparse.ArgumentParser,
    *,
    ncores: bool = True,
    ncores_default: int = 1,
    plot: bool = True,
    verbose: bool = True,
    dry_run: bool = False,
) -> argparse.ArgumentParser:
    """Attach shared runtime flags (``--ncores``, ``--plot``, ``--verbose``, …)."""
    group = parser.add_argument_group('common runtime')
    if ncores:
        group.add_argument(
            '--ncores',
            '--workers',
            dest='ncores',
            type=int,
            default=ncores_default,
            help=(
                'Number of parallel workers / CPU cores (alias: --workers). '
                'For DOLPHOT launch commands this sets MaxThreads.'
            ),
        )
    if plot:
        group.add_argument(
            '--plot',
            action='store_true',
            help='Generate diagnostic / preview plots where applicable.',
        )
    if verbose:
        group.add_argument(
            '-v',
            '--verbose',
            action='store_true',
            help='Increase command-line logging.',
        )
    if dry_run:
        group.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be done without writing outputs.',
        )
    return parser


def add_base_dir(
    parser: argparse.ArgumentParser,
    *,
    required: bool = False,
    default: str | None = None,
    help: str | None = None,
    aliases: Sequence[str] = (
        '--basedir',
        '--workdir',
        '--data-dir',
        '--download-dir',
    ),
) -> argparse.ArgumentParser:
    """
    Attach canonical ``--base-dir`` (with legacy aliases).

    Default aliases: ``--basedir``, ``--workdir``, ``--data-dir``,
    ``--download-dir``. Pass ``aliases`` to extend or replace (e.g. include
    ``--outdir`` for download).
    """
    group = parser.add_argument_group('paths')
    option_strings = ['--base-dir', *aliases]
    # De-duplicate while preserving order
    seen: set[str] = set()
    opts: list[str] = []
    for opt in option_strings:
        if opt not in seen:
            seen.add(opt)
            opts.append(opt)
    group.add_argument(
        *opts,
        dest='base_dir',
        type=str,
        default=default,
        required=required and default is None,
        help=help
        or (
            'Primary project or working directory. '
            f'Aliases: {", ".join(opts[1:])}.'
        ),
    )
    return parser


def add_mast_token(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        '--token',
        default=None,
        type=str,
        help=(
            'MAST authorization token for proprietary data '
            '(https://auth.mast.stsci.edu/info). '
            'Also read from MAST_API_TOKEN / MAST_TOKEN.'
        ),
    )
    return parser


def add_filters(parser: argparse.ArgumentParser, *, help: str | None = None) -> argparse.ArgumentParser:
    parser.add_argument(
        '--filters',
        type=str,
        default=None,
        help=help
        or 'Optional comma-separated filter list (e.g. F560W,F770W).',
    )
    return parser


def add_image_arg(
    parser: argparse.ArgumentParser,
    *,
    required: bool = False,
    nargs: str | int | None = None,
    default: object | None = None,
    help: str | None = None,
    dest: str = 'image',
    group: Any | None = None,
) -> argparse.ArgumentParser:
    """
    Attach canonical ``--image`` for science / target FITS path(s).

    Prefer this over instrument- or verb-specific flags (``--miri``,
    ``--align``, …) whenever a CLI needs a target image path.
    """
    target = group if group is not None else parser
    kwargs: dict = {
        'dest': dest,
        'type': str,
        'default': default,
        'required': required and default is None,
        'help': help
        or 'Science / target FITS image path(s).',
    }
    if nargs is not None:
        kwargs['nargs'] = nargs
    target.add_argument('--image', **kwargs)
    return parser


def add_pair_align_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Attach single-frame relative-alignment options.

    Used by ``align --mode pair`` (or whenever ``--ref`` / ``--image`` are set)::

        align --ref coadd_i2d.fits --image mirimage_cal.fits \\
          --base-dir alignment_output [--photfile cat.phot.txt]
    """
    group = parser.add_argument_group('pair alignment')
    group.add_argument(
        '--ref',
        type=str,
        default=None,
        help=(
            'Reference image used to build the photometry catalog '
            '(e.g. coadd i2d). Enables pair mode with --image.'
        ),
    )
    add_image_arg(
        parser,
        group=group,
        help=(
            'Science image to align to --ref '
            '(e.g. MIRI *_cal.fits or NIRCam *_i2d.fits).'
        ),
    )
    group.add_argument(
        '--photfile',
        type=str,
        default=None,
        help=(
            'Reuse an existing reference .phot.txt catalog instead of building '
            'one from --ref (pair mode).'
        ),
    )
    return parser


def parse_instruments(raw: Sequence[str] | None) -> list[str] | None:
    """Split ``--instruments`` values on commas and/or spaces."""
    if raw is None:
        return None
    out: list[str] = []
    for token in raw:
        for part in str(token).replace(',', ' ').split():
            name = part.strip()
            if name:
                out.append(name)
    return out or None


def parse_filter_list(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [p.strip() for p in str(raw).split(',') if p.strip()] or None


def require_base_dir(args: argparse.Namespace, *, default: str | Path | None = None) -> Path:
    """
    Resolve ``args.base_dir`` or raise a print-friendly :class:`ValueError`.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI namespace (expects ``base_dir``).
    default : str or pathlib.Path or None, optional
        Fallback when ``args.base_dir`` is unset.

    Returns
    -------
    pathlib.Path
        Resolved absolute ``--base-dir``.
    """
    value = getattr(args, 'base_dir', None)
    return resolve_base_dir(value, default=default)


def configure_logging_from_args(
    args: argparse.Namespace,
    script_name: str,
) -> Path:
    """
    Configure package logging from a parsed CLI namespace.

    Uses ``args.base_dir`` when set (project-root ``logs/``); otherwise
    ``./logs``. Honors ``args.verbose`` when present.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI namespace.
    script_name : str
        Short script label used in the log filename.

    Returns
    -------
    pathlib.Path
        Path to the created log file.
    """
    from st123.utils.logging import setup_script_logging

    base = getattr(args, 'base_dir', None)
    verbose = bool(getattr(args, 'verbose', False))
    return setup_script_logging(base, script_name, verbose=verbose)


def sync_legacy_ncores(args: argparse.Namespace) -> argparse.Namespace:
    """
    Expose ``args.workers`` as an alias of ``args.ncores`` for older call sites.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI namespace that may include ``ncores``.

    Returns
    -------
    argparse.Namespace
        The same namespace, with ``workers`` set when ``ncores`` is present.
    """
    if hasattr(args, 'ncores'):
        args.workers = args.ncores
    return args
