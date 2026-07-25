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


def resolve_outdir(obj, outdir=None):
    '''
    Resolve the download output directory.

    Parameters:
    ----------
    obj : str
        Object name used for the default path
    outdir : str or None
        Explicit output directory. If None, uses ``jwst_data/<obj>``.

    Returns:
    -------
    str
        Absolute or relative output directory path
    '''
    if outdir is None:
        outdir = os.path.join('jwst_data', obj)
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

    print(f'Found {len(obs_table)} JWST observation(s)')
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
    obj: str = 'target',
    allowed_filters: Optional[Sequence[str]] = None,
    dry_run: bool = False,
    token: Optional[str] = None,
) -> int:
    """
    Download public MIRI imager products into
    ``<download_dir>/JWST/MIRI/<FILTER>/<obsid>/``.

    This is the canonical layout expected by ``alignment_wrap``.
    """
    download_dir = Path(download_dir).expanduser().resolve()
    print(f'Target: {obj}')
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
