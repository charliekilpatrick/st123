"""Helpers for downloading JWST imaging from MAST."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence

from st123.mast.mast import (
    DEFAULT_DOWNLOAD_LAYOUT,
    download_jwst_observations,
    filter_jwst_observations_by_stage,
    normalize_filter_name,
    query_jwst,
    resolve_mast_token,
)


@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def resolve_outdir(outdir, *, create: bool = True, obj=None):
    '''
    Resolve the download output directory.

    Parameters:
    ----------
    outdir : str
        Explicit output directory (required).
    create : bool
        If True (default), create ``outdir`` (and parents) when missing.
    obj : str or None
        Deprecated. Ignored; kept only for older call signatures.

    Returns:
    -------
    str
        Absolute or relative output directory path
    '''
    del obj  # legacy keyword compatibility
    if outdir is None:
        raise ValueError('outdir / --base-dir is required')
    if create:
        try:
            os.makedirs(outdir, exist_ok=True)
        except OSError as exc:
            raise PermissionError(
                f'Cannot create download directory {outdir!r}: {exc}'
            ) from exc
    return outdir


def query_mast_jwst(
    coord,
    outdir,
    radius,
    stage=2,
    token=None,
    instruments=None,
    *,
    layout: str = DEFAULT_DOWNLOAD_LAYOUT,
    mirimage_only: bool = False,
    dry_run: bool = False,
    allowed_filters: Optional[Sequence[str]] = None,
):
    '''
    Query MAST and download available JWST imaging.

    Parameters:
    ----------
    coord : astropy.coordinates.SkyCoord
        target coordinates
    outdir : str
        output directory for downloads
    radius : astropy.units.Quantity
        search radius
    stage : int
        JWST calibration stage (2=CAL, 3=I2D)
    token : str or None
        Optional MAST API token. When set (or via MAST_API_TOKEN), authenticates
        with ``Observations.login`` and includes proprietary observations.
    instruments : sequence of str or None
        Instrument name substrings (e.g. NIRCAM, MIRI). None uses defaults.
    layout : str
        Per-observation directory layout. Default
        ``telescope/instrument/filter/obsid``
        (``JWST/MIRI/F560W/<obsid>``). Also accepts ``filter/obsid`` and
        ``filter_obsid``.
    mirimage_only : bool
        Restrict products to MIRI imager ``*mirimage*`` files.
    dry_run : bool
        List matching products without downloading.
    allowed_filters : sequence of str or None
        Optional filter whitelist (e.g. ``F560W``).

    Returns:
    -------
    int
        Number of observation product sets downloaded.
    '''
    os.makedirs(outdir, exist_ok=True)
    token = resolve_mast_token(token)
    kwargs = {'radius': radius, 'token': token}
    if instruments is not None:
        kwargs['instruments'] = instruments

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
    print(
        f'Found {len(obs_table)} JWST observation(s) with '
        f'calib_level>={stage}'
        + (f' ({n_skipped} skipped: no matching products expected)' if n_skipped else '')
    )
    if len(obs_table) == 0:
        return 0

    # Pass token again so download authenticates even if called standalone.
    return download_jwst_observations(
        obs_table,
        outdir=outdir,
        stage=stage,
        token=token,
        layout=layout,
        mirimage_only=mirimage_only,
        dry_run=dry_run,
    )


def query_and_download_miri(
    coord,
    *,
    download_dir: str | Path,
    radius,
    stage: int = 2,
    obj: str | None = None,
    allowed_filters: Optional[Sequence[str]] = None,
    dry_run: bool = False,
    token: Optional[str] = None,
) -> int:
    """
    Download public MIRI imager products into
    ``<download_dir>/JWST/MIRI/<FILTER>/<obsid>/``.

    This is the canonical layout expected by ``align --mode reference``.
    ``obj`` is deprecated; the dataset label is ``download_dir.name``.
    """
    download_dir = Path(download_dir).expanduser().resolve()
    label = obj or download_dir.name
    print(f'Target: {label}')
    print(f'Coordinates: {coord.to_string("hmsdms")}')
    print(f'Search radius: {radius}')
    print(f'Download directory: {download_dir}')
    return query_mast_jwst(
        coord,
        outdir=str(download_dir),
        radius=radius,
        stage=stage,
        token=token,
        instruments=('MIRI',),
        layout=DEFAULT_DOWNLOAD_LAYOUT,
        mirimage_only=True,
        dry_run=dry_run,
        allowed_filters=allowed_filters,
    )
