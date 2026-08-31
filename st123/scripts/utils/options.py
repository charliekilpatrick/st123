"""
Unified CLI options API for st123 **stages**.

Command-line units (``align``, ``mosaic``, ``download``, ``dolphot-prep``, ...)
are stages, not pipelines. Every stage parser is built from
:func:`create_parser` (the stage primitive's option constructor). Stage
modules must not create a second ``ArgumentParser`` or call
``add_argument``; they attach named ``add_*`` helpers from this module
and read dests. Library packages never parse argv.

Pipelines (sequences of stages) are a later layer and must not grow their
own CLI parsers.

Canonical path / runtime flags
------------------------------
``--base-dir``
    Primary project / dataset directory. The object name is the basename
    of this path (e.g. ``.../NGC3310``). There is no separate ``--obj``.
``--ncores``
    Parallel worker count (alias: ``--workers``).
``--instruments`` / ``--instrument``
    Shared instrument list. Mission aliases: ``hst`` -> ACS WFC3 WFPC2,
    ``jwst`` -> NIRCAM MIRI, ``all`` -> NIRCAM MIRI ACS WFC3 WFPC2.
``--ra`` / ``--dec`` / ``--radius``
    Shared sky coordinates (required by download; accepted elsewhere so
    the same argv works across pipeline stages).
``--existing-box`` / ``--center-ra`` / ``--center-dec`` / ``--stamp-*``
    Shared stamp / mosaic-box identity. Mosaic uses them to build or remosaic
    a ``reference/group_*/ref_*`` grid; align uses ``--existing-box`` coadds as
    the MIRI JHAT reference (and skips a redundant full-field mosaic). Other
    commands accept the same flags for uniform scripting.
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
import sys
from pathlib import Path
from typing import Any, Sequence

from st123 import __version__ as ST123_VERSION
from st123.utils.settings import DOWNLOAD_DIR_NAME

# ---------------------------------------------------------------------------
# Path layout helpers (project root vs reduction workdir)
# ---------------------------------------------------------------------------

_REDUCTION_MARKERS = ('raw', 'jhat', 'align', 'reference')
_PROJECT_SUBDIRS = (DOWNLOAD_DIR_NAME, 'reduction', 'logs')


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


def ensure_writable_dir(path: Path | str, *, label: str | None = None) -> Path:
    """
    Create *path* (and parents) if missing; raise on permission / OS errors.

    Parameters
    ----------
    path : pathlib.Path or str
        Directory to create.
    label : str or None, optional
        Short name used in the error message (e.g. ``'download'``).

    Returns
    -------
    pathlib.Path
        Absolute path to the directory.

    Raises
    ------
    PermissionError
        If the directory cannot be created (permissions or other ``OSError``).
    """
    dest = Path(path).expanduser()
    kind = label or 'directory'
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PermissionError(
            f'Cannot create {kind} directory {dest}: {exc}. '
            'Check that you have write permission for this path '
            '(and its parents).'
        ) from exc
    return dest.resolve()


def ensure_project_layout(base_dir: Path | str) -> Path:
    """
    Ensure the standard project directories exist under ``--base-dir``.

    Creates ``download/``, ``reduction/``, and ``logs/`` (and the project
    root itself when missing). Callers should not require a manual
    ``mkdir -p``.

    When *base_dir* is already a legacy reduction workdir (contains
    ``raw/`` / ``jhat/`` / ... and is not named ``reduction``), only
    ``logs/`` is created there so we do not nest a second layout.

    Parameters
    ----------
    base_dir : pathlib.Path or str
        Project root or reduction workdir from ``--base-dir``.

    Returns
    -------
    pathlib.Path
        Absolute project root (or legacy workdir root).

    Raises
    ------
    PermissionError
        If any required directory cannot be created.
    """
    base = Path(base_dir).expanduser()
    # resolve() without requiring the path to exist yet.
    try:
        base_res = base.resolve(strict=False)
    except TypeError:
        base_res = base.resolve()
    root = resolve_project_root(base_res)

    # Legacy workdir passed as --base-dir: keep products in-place.
    if looks_like_reduction_dir(base_res) and base_res.name != 'reduction':
        ensure_writable_dir(base_res / 'logs', label='logs')
        return root

    ensure_writable_dir(root, label='project')
    for name in _PROJECT_SUBDIRS:
        ensure_writable_dir(root / name, label=name)
    return root


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
    Map ``--base-dir`` to the reduction workdir.

    * If ``base_dir`` looks like a project root (``download/``, ``JWST/``,
      ``HST/``, and/or ``reduction/`` present), return
      ``base_dir / 'reduction'``. This is checked before workdir markers so a
      project-level ``reference/`` symlink (from
      :func:`ensure_dataset_reference_link`) does not make the project root
      look like the reduction workdir.
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
    if (
        (base / DOWNLOAD_DIR_NAME).is_dir()
        or (base / 'JWST').is_dir()
        or (base / 'HST').is_dir()
        or (base / 'reduction').is_dir()
    ):
        return (base / 'reduction').resolve()
    if looks_like_reduction_dir(base):
        return base
    return base


def resolve_jhat_dir(
    work_dir: Path,
    telescope: str,
    *,
    create: bool = False,
) -> Path:
    """
    Resolve the JHAT output/input directory for *telescope*.

    Dual-mode (JWST+HST) campaigns prefer separate trees so HST mosaic globs
    do not pick up ``jw*_jhat.fits`` (and vice versa):

    * HST -> ``reduction/jhat_hst`` (falls back to legacy ``reduction/jhat``)
    * JWST -> ``reduction/jhat_jwst`` (falls back to legacy ``reduction/jhat``)

    When *create* is True, ensure the preferred directory exists and, if legacy
    ``jhat/`` is absent, add a ``jhat`` -> preferred symlink for older tooling.
    """
    work = Path(work_dir).expanduser().resolve()
    tel = str(telescope).strip().lower()
    is_hst = tel in {'hst', 'hubble'}
    preferred_name = 'jhat_hst' if is_hst else 'jhat_jwst'
    preferred = work / preferred_name
    legacy = work / 'jhat'

    def _has_jhat(d: Path) -> bool:
        return d.is_dir() and any(d.glob('*_jhat.fits'))

    def _legacy_matches_telescope(d: Path) -> bool:
        if not d.is_dir():
            return False
        names = [p.name.lower() for p in d.glob('*_jhat.fits')]
        if not names:
            return False
        if is_hst:
            return any(not n.startswith('jw') for n in names)
        return any(n.startswith('jw') for n in names)

    if create:
        # Prefer dedicated tree when already populated; otherwise keep writing
        # into a mid-campaign mixed/legacy jhat/ so we do not orphan products.
        if _has_jhat(preferred):
            return preferred
        if _legacy_matches_telescope(legacy):
            return legacy
        preferred.mkdir(parents=True, exist_ok=True)
        if not legacy.exists():
            try:
                legacy.symlink_to(preferred_name)
            except OSError:
                pass
        return preferred

    if _has_jhat(preferred):
        return preferred
    if legacy.is_dir():
        return legacy
    return preferred


def resolve_project_root(base_dir: Path) -> Path:
    """
    Map ``--base-dir`` to the dataset / project root (parent of ``download/``,
    ``JWST/``, ``HST/``).

    If ``base_dir`` is a reduction workdir named ``reduction``, or the
    download tree named ``download``, return its parent; otherwise return
    ``base_dir`` itself.

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
        looks_like_reduction_dir(base)
        or not (
            (base / DOWNLOAD_DIR_NAME).is_dir()
            or (base / 'JWST').is_dir()
            or (base / 'HST').is_dir()
        )
    ):
        return base.parent
    if base.name == DOWNLOAD_DIR_NAME and (
        (base.parent / 'reduction').is_dir()
        or (base / 'JWST').is_dir()
        or (base / 'HST').is_dir()
        or (base.parent / 'JWST').is_dir()
        or (base.parent / 'HST').is_dir()
    ):
        return base.parent
    return base


def instrument_raw_dir(
    base_dir: Path,
    instrument: str,
    *,
    telescope: str | None = None,
) -> Path:
    """
    Return MAST product root for an instrument under the project.

    Prefer ``{project}/download/{Telescope}/{Instrument}`` (canonical). Fall
    back to legacy ``{project}/{Telescope}/{Instrument}`` when only that tree
    exists.

    Parameters
    ----------
    base_dir : pathlib.Path
        Project root or reduction workdir from ``--base-dir``.
    instrument : str
        Instrument name (e.g. ``NIRCAM``, ``MIRI``, ``WFC3``, ``ACS``).
    telescope : str or None, optional
        ``JWST`` or ``HST``. When ``None``, inferred from *instrument*
        (HST for ACS/WFC3/WFPC2; JWST otherwise).

    Returns
    -------
    pathlib.Path
        Absolute ``.../<Telescope>/<Instrument>`` directory under the project.
    """
    root = resolve_project_root(base_dir)
    key = str(instrument).strip().upper()
    tel = (telescope or '').strip().upper() or None
    if key in ('NIRCAM', 'NRC'):
        name = 'NIRCam'
        tel = tel or 'JWST'
    elif key == 'MIRI':
        name = 'MIRI'
        tel = tel or 'JWST'
    elif key == 'NIRISS':
        name = 'NIRISS'
        tel = tel or 'JWST'
    elif key in ('ACS',):
        name = 'ACS'
        tel = tel or 'HST'
    elif key in ('WFC3',):
        name = 'WFC3'
        tel = tel or 'HST'
    elif key in ('WFPC2',):
        name = 'WFPC2'
        tel = tel or 'HST'
    else:
        name = instrument
        tel = tel or 'JWST'
    preferred = root / DOWNLOAD_DIR_NAME / tel / name
    legacy = root / tel / name
    if preferred.is_dir():
        return preferred
    if legacy.is_dir():
        return legacy
    return preferred


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


def default_phot_dir(
    base_dir: Path, *, group: int = 0, box: int | str = 0
) -> Path:
    """
    Return ``{reduction}/phot_{group}_{box}`` (no extra object subdirectory).

    Parameters
    ----------
    base_dir : pathlib.Path
        Project root or reduction workdir from ``--base-dir``.
    group : int, optional
        Mosaic overlap group index.
    box : int or str, optional
        Mosaic box index or stamp id within the group (e.g. ``0`` or ``sn``).

    Returns
    -------
    pathlib.Path
        Default NIRCam DOLPHOT staging directory.
    """
    return resolve_reduction_dir(base_dir) / f'phot_{group}_{box}'


def default_warmstart_outdir(
    base_dir: Path, *, group: int = 0, box: int | str = 0
) -> Path:
    """
    Return ``{project}/dolphot/nircam_miri_{group}_{box}``.

    Parameters
    ----------
    base_dir : pathlib.Path
        Project root or reduction workdir from ``--base-dir``.
    group : int, optional
        Mosaic overlap group index.
    box : int or str, optional
        Mosaic box index or stamp id within the group (e.g. ``0`` or ``sn``).

    Returns
    -------
    pathlib.Path
        Default NIRCam->MIRI warm-start DOLPHOT output directory.
    """
    return resolve_project_root(base_dir) / 'dolphot' / f'nircam_miri_{group}_{box}'


def default_hst_warmstart_outdir(
    base_dir: Path, *, group: int = 0, box: int | str = 0
) -> Path:
    """
    Return ``{project}/dolphot/nircam_hst_{group}_{box}``.

    Parameters
    ----------
    base_dir : pathlib.Path
        Project root or reduction workdir from ``--base-dir``.
    group : int, optional
        Mosaic overlap group index.
    box : int or str, optional
        Mosaic box index or stamp id within the group (e.g. ``0`` or ``sn``).

    Returns
    -------
    pathlib.Path
        Default NIRCam->HST warm-start DOLPHOT output directory.
    """
    return resolve_project_root(base_dir) / 'dolphot' / f'nircam_hst_{group}_{box}'


def default_miri_outdir(
    base_dir: Path, *, group: int = 0, box: int | str = 0
) -> Path:
    """
    Return ``{project}/dolphot/miri_{group}_{box}``.

    Parameters
    ----------
    base_dir : pathlib.Path
        Project root or reduction workdir from ``--base-dir``.
    group : int, optional
        Mosaic overlap group index.
    box : int or str, optional
        Mosaic box index or stamp id within the group (e.g. ``0`` or ``sn``).

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
    """Create a parser with the uniform stage option set.

    Every stage starts from this primitive. Shared identity flags
    (``--existing-box``, ``--center-ra``, ``--stamp-id``, ...) are attached
    here so the same argv works across stages. Path, instrument, sky, and
    runtime groups use the ``add_*`` helpers (same dest names). Stage
    modules must not construct a second parser.
    """
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
    add_stamp_box_args(parser)
    return parser


def add_stamp_box_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Attach stamp / mosaic-box identity flags.

    Always available on every entry point (via :func:`create_parser`) so a
    pipeline can pass ``--existing-box group_0/ref_sn`` (or ``--center-ra`` /
    ``--stamp-id``) uniformly. Stages that do not mosaic or reference-align
    accept the flags and ignore them.
    """
    group = parser.add_argument_group('stamp / mosaic box')
    group.add_argument(
        '--existing-box',
        type=str,
        default=None,
        help=(
            'Existing reference/group_*/ref_* stamp directory. Accepts '
            'group_0/ref_sn, reference/group_0/ref_sn, or an absolute path. '
            'mosaic remosaics onto that stamp WCS (frames outside the FoV '
            'or not covering its center are dropped). align --instruments '
            'MIRI uses that stamp\'s coadd*i2d.fits as the JHAT reference '
            'and skips a redundant full-field NIRCam mosaic when those '
            'coadds already exist.'
        ),
    )
    group.add_argument(
        '--center-ra',
        type=float,
        default=None,
        help=(
            'Custom stamp center RA (deg), with --center-dec. mosaic builds '
            'one stamp of --stamp-size (directory ref_<stamp-id>). Other '
            'commands accept the flag for uniform scripting.'
        ),
    )
    group.add_argument(
        '--center-dec',
        type=float,
        default=None,
        help='Custom stamp center declination (deg), with --center-ra.',
    )
    group.add_argument(
        '--stamp-size',
        type=str,
        default=None,
        help=(
            'Custom stamp size in arcsec: S (square) or W,H. Default ~68x51 '
            'or the size of --stamp-ref when given. Used by mosaic; accepted '
            'elsewhere for uniform scripting.'
        ),
    )
    group.add_argument(
        '--stamp-ref',
        type=str,
        default=None,
        help=(
            'Optional coadd/i2d or stamp_wcs.fits whose orientation and pixel '
            'scale are copied into the --center-ra/dec stamp.'
        ),
    )
    group.add_argument(
        '--stamp-id',
        type=str,
        default='sn',
        help=(
            'Box id for a --center-ra/dec stamp (directory ref_<id>; '
            'default sn).'
        ),
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
    """
    Attach shared runtime flags (``--ncores``, ``--plot``, ``--verbose``, ...).

    ``--ncores`` is always accepted so pipelines can pass a uniform
    ``--ncores "$NCORES"`` across entry points, even when a command is not
    yet parallelized (*ncores* is kept for API compatibility and ignored
    when False would previously have omitted the flag).
    """
    group = parser.add_argument_group('common runtime')
    # Always expose --ncores (ncores=False is a no-op for backward compat).
    _ = ncores
    group.add_argument(
        '--ncores',
        '--workers',
        dest='ncores',
        type=int,
        default=ncores_default,
        help=(
            'Number of parallel workers / CPU cores (alias: --workers). '
            'Accepted by all entry points for uniform scripting; unused '
            'when a command is not parallelized. For mosaic JWST: concurrent '
            'per-filter Image3Pipeline runs. For mosaic HST: AstroDrizzle '
            'cores / filter pool. For dolphot-prep: pool size for independent '
            'splitgroups / *mask / calcsky subprocesses (stages stay ordered). '
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
    ``--align``, ...) whenever a CLI needs a target image path.
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


def expand_mission_instruments(
    tokens: Sequence[str] | None,
) -> list[str] | None:
    """
    Expand mission aliases in an instrument token list.

    * ``hst`` -> ACS, WFC3, WFPC2
    * ``jwst`` -> NIRCAM, MIRI
    * ``all`` -> NIRCAM, MIRI, ACS, WFC3, WFPC2
    * ``nrc`` -> NIRCAM

    Other tokens are upper-cased and de-duplicated (order preserved).
    """
    from st123.datamodels import HSTDataModel, JWSTDataModel

    if tokens is None:
        return None
    out: list[str] = []

    def _add(name: str) -> None:
        key = str(name).strip().upper()
        if key and key not in out:
            out.append(key)

    all_instruments = JWSTDataModel.INSTRUMENTS + HSTDataModel.INSTRUMENTS
    for token in tokens:
        key = str(token).strip().upper()
        if not key:
            continue
        if key == 'HST':
            for name in HSTDataModel.INSTRUMENTS:
                _add(name)
            continue
        if key == 'JWST':
            for name in JWSTDataModel.INSTRUMENTS:
                _add(name)
            continue
        if key == 'ALL':
            for name in all_instruments:
                _add(name)
            continue
        if key == 'NRC':
            _add('NIRCAM')
            continue
        _add(key)
    return out or None


def resolve_instruments(raw: Sequence[str] | None) -> list[str] | None:
    """Parse ``--instruments`` and expand mission aliases (``hst`` / ``jwst`` / ``all``)."""
    return expand_mission_instruments(parse_instruments(raw))


def resolve_photometry_instrument(
    raw: Sequence[str] | None,
    *,
    default: str = 'nircam',
) -> str:
    """
    Map ``--instruments`` to a dolphot-prep / run-dolphot mode string.

    Mission aliases are preserved as modes when given alone (``hst``, ``jwst``).
    A multi-camera HST list (``ACS WFC3`` or expanded ``hst``) becomes ``hst``.
    ``jwst`` alone maps to ``nircam`` (primary JWST phot discovery tree).
    """
    tokens = parse_instruments(raw)
    if not tokens:
        return default
    if len(tokens) == 1:
        key = tokens[0].strip().lower()
        if key == 'nrc':
            return 'nircam'
        if key == 'jwst':
            return 'nircam'
        if key in {'hst', 'nircam', 'miri', 'acs', 'wfc3', 'wfpc2'}:
            return key
    expanded = expand_mission_instruments(tokens) or []
    hst = [u for u in expanded if u in {'ACS', 'WFC3', 'WFPC2', 'WFC'}]
    jwst = [u for u in expanded if u in {'NIRCAM', 'MIRI'}]
    if hst and not jwst:
        return 'hst' if len(hst) > 1 else hst[0].lower()
    if jwst and not hst:
        return jwst[0].lower() if len(jwst) == 1 else 'nircam'
    raise ValueError(
        'Cannot mix HST and JWST in one dolphot-prep / run-dolphot call; '
        f'got {expanded}. Run each mission separately.'
    )


def add_instruments_arg(
    parser: argparse.ArgumentParser,
    *,
    help: str | None = None,
    default: object | None = None,
) -> argparse.ArgumentParser:
    """
    Attach canonical ``--instruments`` (alias ``--instrument``).

    Mission aliases: ``hst`` (= ACS WFC3 WFPC2), ``jwst`` (= NIRCAM MIRI),
    ``all`` (= NIRCAM MIRI ACS WFC3 WFPC2).
    """
    parser.add_argument(
        '--instruments',
        '--instrument',
        nargs='+',
        dest='instruments',
        default=default,
        metavar='INSTR',
        help=help
        or (
            'Instruments or mission aliases: hst (= ACS WFC3 WFPC2), '
            'jwst (= NIRCAM MIRI), all (= NIRCAM MIRI ACS WFC3 WFPC2), '
            'or explicit names (ACS, WFC3, NIRCAM, ...). '
            'Alias: --instrument. Space- or comma-separated.'
        ),
    )
    return parser


def add_sky_coord_args(
    parser: argparse.ArgumentParser,
    *,
    required: bool = False,
    ra_default: object | None = None,
    dec_default: object | None = None,
    radius_default: float = 3.0,
) -> argparse.ArgumentParser:
    """
    Attach ``--ra`` / ``--dec`` / ``--radius``.

    Always available on pipeline CLIs for uniform scripting; commands that do
    not perform a cone search ignore them.
    """
    group = parser.add_argument_group('sky coordinates')
    group.add_argument(
        '--ra',
        type=str,
        default=ra_default,
        required=required and ra_default is None,
        help='Target ICRS right ascension (degrees or sexagesimal).',
    )
    group.add_argument(
        '--dec',
        type=str,
        default=dec_default,
        required=required and dec_default is None,
        help='Target ICRS declination (degrees or sexagesimal).',
    )
    group.add_argument(
        '--radius',
        type=float,
        default=radius_default,
        help=(
            'MAST cone-search radius in arcminutes (default: '
            f'{radius_default}). Used by download; accepted elsewhere for '
            'uniform scripting.'
        ),
    )
    return parser


def default_instruments_for_telescope(
    telescope: str | None,
) -> list[str] | None:
    """
    Mission default instrument list for ``--telescope``.

    These are treated as identical to the matching ``--instruments`` lists:

    * ``hst`` -> ACS, WFC3, WFPC2 (:attr:`HSTDataModel.INSTRUMENTS`)
    * ``jwst`` -> NIRCAM, MIRI (:attr:`JWSTDataModel.INSTRUMENTS`)

    Parameters
    ----------
    telescope : str or None
        ``hst``, ``jwst``, or ``None`` when the flag was omitted.

    Returns
    -------
    list of str or None
        Upper-case instrument names, or ``None`` when *telescope* is unset.

    Raises
    ------
    ValueError
        If *telescope* is set but not ``hst`` / ``jwst``.
    """
    if telescope is None:
        return None
    key = str(telescope).strip().lower()
    if not key:
        return None
    if key == 'hst':
        from st123.datamodels import HSTDataModel

        return list(HSTDataModel.INSTRUMENTS)
    if key == 'jwst':
        from st123.datamodels import JWSTDataModel

        return list(JWSTDataModel.INSTRUMENTS)
    raise ValueError(
        f'Unsupported telescope {telescope!r}; choose from hst, jwst'
    )


def resolve_instruments_with_telescope(
    instruments_raw: Sequence[str] | None,
    telescope: str | None,
    *,
    resolve_instruments_fn,
) -> list[str] | None:
    """
    Resolve CLI instruments, expanding ``--telescope`` when instruments omitted.

    Explicit ``--instruments`` wins. Otherwise ``--telescope hst|jwst`` expands
    to the mission default set (same as passing that instrument list).
    """
    multi = resolve_instruments_fn(instruments_raw)
    if multi is not None:
        return multi
    return default_instruments_for_telescope(telescope)


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

    When ``args.base_dir`` is set, also ensures the standard project layout
    (``download/``, ``reduction/``, ``logs/``) exists. Permission errors
    exit with code 1 and a clear message (no stack trace).

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
    if base is not None:
        try:
            ensure_project_layout(base)
        except PermissionError as exc:
            print(f'ERROR: {exc}', file=sys.stderr)
            raise SystemExit(1) from exc
    try:
        return setup_script_logging(base, script_name, verbose=verbose)
    except PermissionError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc


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
