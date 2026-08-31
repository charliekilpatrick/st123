"""Helpers for downloading HST and JWST imaging from MAST."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from astropy.coordinates import SkyCoord
from astropy.units import Quantity

from st123.datamodels import HSTDataModel, JWSTDataModel, MIRIDataModel, NIRCamDataModel
from st123.stages.download.mast import (
    DEFAULT_DOWNLOAD_LAYOUT,
    download_hst_observations,
    download_jwst_observations,
    filter_jwst_observations_by_stage,
    normalize_filter_name,
    query_hst,
    query_jwst,
    resolve_mast_token,
)
from st123.utils.logging import capture_output

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MastDownloadResult:
    """Outcome of a MAST download query (HST or JWST)."""

    n_observations: int
    skipped_miri_only: bool = False
    n_failed: int = 0

    @property
    def incomplete(self) -> bool:
        """True when some matched observations failed after retries."""
        return int(self.n_failed) > 0

    def __int__(self) -> int:
        return int(self.n_observations)

    def __bool__(self) -> bool:
        return bool(self.n_observations) or bool(self.skipped_miri_only)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.n_observations == other
        if isinstance(other, MastDownloadResult):
            return (
                self.n_observations == other.n_observations
                and self.skipped_miri_only == other.skipped_miri_only
                and self.n_failed == other.n_failed
            )
        return NotImplemented


def suppress_stdout():
    """
    Mediate stdout/stderr through logging (legacy name).

    Prefer :func:`st123.utils.logging.capture_output` at new call sites.

    Returns
    -------
    contextmanager
        Context manager that redirects stdout/stderr into the logging system.
    """
    return capture_output()


def resolve_outdir(
    outdir: str | Path | None,
    *,
    create: bool = True,
    obj: str | None = None,
) -> str:
    """
    Resolve the download output directory.

    Parameters
    ----------
    outdir : str or pathlib.Path
        Explicit download tree root (required). The ``download`` CLI passes
        ``<base-dir>/download``; library callers should do the same when
        writing the canonical layout.
    create : bool, optional
        If True (default), create ``outdir`` (and parents) when missing.
    obj : str or None, optional
        Deprecated. Ignored; kept only for older call signatures.

    Returns
    -------
    str
        Output directory path as a string.

    Raises
    ------
    ValueError
        If ``outdir`` is None.
    PermissionError
        If the directory cannot be created.
    """
    del obj  # legacy keyword compatibility
    if outdir is None:
        raise ValueError('outdir / --base-dir is required')
    outdir_s = str(outdir)
    if create:
        try:
            os.makedirs(outdir_s, exist_ok=True)
        except OSError as exc:
            raise PermissionError(
                f'Cannot create download directory {outdir_s}: {exc}. '
                'Check that you have write permission for this path '
                '(and its parents).'
            ) from exc
    return outdir_s


def query_mast_jwst(
    coord: SkyCoord,
    outdir: str | Path,
    radius: Quantity,
    stage: int = 2,
    token: str | None = None,
    instruments: Sequence[str] | None = None,
    *,
    layout: str = DEFAULT_DOWNLOAD_LAYOUT,
    dry_run: bool = False,
    allowed_filters: Optional[Sequence[str]] = None,
    force_miri: bool = False,
) -> MastDownloadResult:
    """
    Query MAST and download available JWST imaging.

    MIRI downloads always keep imager ``*mirimage*`` products only.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Target coordinates.
    outdir : str or pathlib.Path
        Download tree root (typically ``<base-dir>/download``).
    radius : astropy.units.Quantity
        Search radius.
    stage : int, optional
        JWST calibration stage (``2`` = CAL, ``3`` = I2D).
    token : str or None, optional
        Optional MAST API token. When set (or via ``MAST_API_TOKEN``),
        authenticates with ``Observations.login`` and includes proprietary
        observations.
    instruments : sequence of str or None, optional
        Instrument name substrings (e.g. ``NIRCAM``, ``MIRI``). ``None`` uses
        package defaults.
    layout : str, optional
        Per-observation directory layout. Default
        ``telescope/instrument/filter/obsid``
        (``JWST/MIRI/F560W/<obsid>``). Also accepts ``filter/obsid`` and
        ``filter_obsid``.
    dry_run : bool, optional
        List matching products without downloading.
    allowed_filters : sequence of str or None, optional
        Optional filter whitelist (e.g. ``F560W``).
    force_miri : bool, optional
        When False (default), MIRI-only fields (no NIRCam) are skipped if
        both NIRCam and MIRI were requested. Pass True (or request MIRI
        alone via ``instruments=['MIRI']``) to download MIRI anyway.

    Returns
    -------
    MastDownloadResult
        ``n_observations`` downloaded (or listed in dry-run). When a
        MIRI-only field is skipped, ``skipped_miri_only`` is True and
        ``n_observations`` is 0.
    """
    outdir_s = str(outdir)
    os.makedirs(outdir_s, exist_ok=True)
    token = resolve_mast_token(token)
    kwargs = {'radius': radius, 'token': token}
    if instruments is not None:
        kwargs['instruments'] = instruments

    logger.info(
        'Querying MAST JWST imaging within %s of %s',
        radius,
        coord.to_string('decimal'),
    )
    with capture_output():
        obs_table = query_jwst(coord, **kwargs)
    if allowed_filters:
        wanted = {normalize_filter_name(f) for f in allowed_filters}
        keep = [
            normalize_filter_name(f) in wanted for f in obs_table['filters']
        ]
        obs_table = obs_table[keep]

    n_matched = len(obs_table)
    obs_table = filter_jwst_observations_by_stage(obs_table, stage)
    n_skipped = n_matched - len(obs_table)
    logger.info(
        'Found %d JWST observation(s) with calib_level>=%d%s',
        len(obs_table),
        stage,
        f' ({n_skipped} skipped: no matching products expected)' if n_skipped else '',
    )
    if len(obs_table) == 0:
        return MastDownloadResult(0)

    requested = instruments if instruments is not None else JWSTDataModel.INSTRUMENTS
    want_nircam = any(NIRCamDataModel.matches(name) for name in requested)
    want_miri = any(MIRIDataModel.matches(name) for name in requested)
    has_nircam = False
    has_miri = False
    if 'instrument_name' in getattr(obs_table, 'colnames', []):
        for name in obs_table['instrument_name']:
            if NIRCamDataModel.matches(name):
                has_nircam = True
            if MIRIDataModel.matches(name):
                has_miri = True
            if has_nircam and has_miri:
                break
    if (
        not force_miri
        and want_nircam
        and want_miri
        and has_miri
        and not has_nircam
    ):
        logger.warning(
            'MIRI-only JWST field (no NIRCam); skipping JWST download. '
            'Pass --force-miri (or --instruments MIRI) to download MIRI.'
        )
        return MastDownloadResult(0, skipped_miri_only=True)

    # Pass token again so download authenticates even if called standalone.
    with capture_output():
        n = download_jwst_observations(
            obs_table,
            outdir=outdir_s,
            stage=stage,
            token=token,
            layout=layout,
            dry_run=dry_run,
        )
    return MastDownloadResult(int(n))


def query_mast_hst(
    coord: SkyCoord,
    outdir: str | Path,
    radius: Quantity,
    token: str | None = None,
    instruments: Sequence[str] | None = None,
    *,
    layout: str = DEFAULT_DOWNLOAD_LAYOUT,
    dry_run: bool = False,
    allowed_filters: Optional[Sequence[str]] = None,
    use_galaxy_size: bool = False,
) -> MastDownloadResult:
    """
    Query MAST and download available HST imaging.

    WFPC2 ``c1m`` DQ companions are always downloaded (required for drizzle /
    ``wfpc2mask``). ``ACS`` includes ACS/WFC (``flc``) plus ACS/HRC and
    ACS/SBC (``flt``).

    Parameters
    ----------
    coord : SkyCoord
        Target coordinates.
    outdir : str or pathlib.Path
        Download tree root (typically ``<base-dir>/download``).
    radius : Quantity
        Search radius.
    token : str or None, optional
        Optional MAST API token.
    instruments : sequence of str or None, optional
        Instrument substrings (default ACS, WFC3, WFPC2).
    layout : str, optional
        Per-observation directory layout.
    dry_run : bool, optional
        List matching products without downloading.
    allowed_filters : sequence of str or None, optional
        Optional filter whitelist. ``None`` keeps all imaging filters.
    use_galaxy_size : bool, optional
        Derive radius from PGC size when ``True`` and radius handling allows.

    Returns
    -------
    MastDownloadResult
        ``n_observations`` ready for the pipeline (newly downloaded /
        listed, or already fully present on disk). ``n_failed`` counts
        observations that still failed after product-list / download
        retries (partial inventory -> incomplete).
    """
    outdir_s = str(outdir)
    os.makedirs(outdir_s, exist_ok=True)
    token = resolve_mast_token(token)
    inst = list(instruments) if instruments is not None else list(HSTDataModel.INSTRUMENTS)
    filters = None
    if allowed_filters:
        filters = [normalize_filter_name(f) for f in allowed_filters]

    logger.info(
        'Querying MAST HST imaging within %s of %s (instruments=%s)',
        radius,
        coord.to_string('decimal'),
        ', '.join(inst),
    )
    with capture_output():
        obs_table = query_hst(
            coord,
            radius=radius,
            filters=filters,
            instruments=inst,
            use_galaxy_size=use_galaxy_size,
            token=token,
        )
    logger.info('Found %d HST imaging observation(s)', len(obs_table))
    if len(obs_table) == 0:
        return MastDownloadResult(0)

    with capture_output():
        return download_hst_observations(
            obs_table,
            outdir=outdir_s,
            token=token,
            layout=layout,
            dry_run=dry_run,
        )


def query_and_download_miri(
    coord: SkyCoord,
    *,
    download_dir: str | Path,
    radius: Quantity,
    stage: int = 2,
    obj: str | None = None,
    allowed_filters: Optional[Sequence[str]] = None,
    dry_run: bool = False,
    token: Optional[str] = None,
) -> int:
    """
    Download public MIRI imager products into
    ``<download_dir>/download/JWST/MIRI/<FILTER>/<obsid>/<filename>``.

    This is the canonical layout expected by ``align --mode reference``.
    ``obj`` is deprecated; the dataset label is ``download_dir.name``.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Target coordinates.
    download_dir : str or pathlib.Path
        Dataset / project root (same role as CLI ``--base-dir``). Products
        are written under ``<download_dir>/download/``.
    radius : astropy.units.Quantity
        Search radius.
    stage : int, optional
        JWST calibration stage (``2`` = CAL, ``3`` = I2D).
    obj : str or None, optional
        Deprecated label override; defaults to ``Path(download_dir).name``.
    allowed_filters : sequence of str or None, optional
        Optional filter whitelist.
    dry_run : bool, optional
        List matching products without downloading.
    token : str or None, optional
        Optional MAST API token.

    Returns
    -------
    int
        Number of observation product sets downloaded (or listed in dry-run).
    """
    from st123.utils.settings import DOWNLOAD_DIR_NAME

    project = Path(download_dir).expanduser().resolve()
    outdir = project / DOWNLOAD_DIR_NAME
    label = obj or project.name
    logger.info('Target: %s', label)
    logger.info('Coordinates: %s', coord.to_string('hmsdms'))
    logger.info('Search radius: %s', radius)
    logger.info('Download directory: %s', outdir)
    result = query_mast_jwst(
        coord,
        outdir=str(outdir),
        radius=radius,
        stage=stage,
        token=token,
        instruments=['MIRI'],
        layout='telescope/instrument/filter/obsid',
        dry_run=dry_run,
        allowed_filters=allowed_filters,
        force_miri=True,
    )
    return int(result)
