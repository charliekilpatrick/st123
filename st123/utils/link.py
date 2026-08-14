"""Symlink helpers for reduction ``raw/`` trees."""

from __future__ import annotations

import glob
import logging
import os
from typing import Sequence

logger = logging.getLogger(__name__)


def create_symlink(src: str, dst: str) -> None:
    """
    Create a symlink from ``src`` to ``dst``.

    Creates the destination parent directory when needed. If ``dst`` already
    exists as a file or symlink, it is replaced; if it exists as a non-link
    path that is still present, a warning is logged and no link is created.

    Parameters
    ----------
    src : str
        Absolute or relative path to the source file.
    dst : str
        Absolute or relative path of the symlink to create.

    Returns
    -------
    None
    """
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not os.path.exists(dst):
        try:
            os.symlink(src, dst)
        except FileExistsError:
            os.unlink(dst)
            os.symlink(src, dst)
    else:
        logger.warning('%s already exists', dst)


def remove_proc_files(files: Sequence[str], proc_dir: str) -> list[str]:
    """
    Drop science files that already appear under ``proc_dir/raw/``.

    Comparison uses resolved real paths so symlinks into the reduction tree
    match their source FITS files.

    Parameters
    ----------
    files : sequence of str
        Candidate FITS paths (typically under a JWST download tree).
    proc_dir : str
        Reduction directory that may contain ``raw/*.fits`` symlinks or copies
        of already-staged frames.

    Returns
    -------
    list of str
        Members of ``files`` that are not already present under
        ``proc_dir/raw/``.
    """
    proc_files = glob.glob(os.path.join(proc_dir, 'raw', '*.fits'), recursive=True)
    proc_files = [os.path.realpath(i) for i in proc_files]
    return list(set(files) - set(proc_files))
