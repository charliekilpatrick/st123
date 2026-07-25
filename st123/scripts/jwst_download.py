#!/usr/bin/env python3
"""CLI entry for MIRI-oriented MAST downloads (legacy ``jwst_download`` name).

Prefer::

    python -m st123.scripts.download \\
      --ra ... --dec ... --obj NGC3310 \\
      --download-dir /data/rwisenbaker/jwst_data/NGC3310 \\
      --radius 3 --stage 2 --instruments MIRI \\
      --layout telescope/instrument/filter/obsid

Or this module / the ``jwst-download`` console script after ``pip install -e .``::

    python -m st123.scripts.jwst_download \\
      --ra ... --dec ... --obj NGC3310 \\
      --download-dir /data/rwisenbaker/jwst_data/NGC3310 \\
      --radius 3 --stage 2
"""

from __future__ import annotations

from st123.scripts.download import create_parser, main_jwst_download

__all__ = ['create_parser', 'main', 'main_jwst_download']


def main(argv: list[str] | None = None) -> int:
    """Alias so entry-point checks and ``python -m`` share one ``main``."""
    return main_jwst_download(argv)


if __name__ == '__main__':
    raise SystemExit(main())
