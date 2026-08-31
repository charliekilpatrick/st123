"""
st123 stages: download, alignment, mosaic, and photometry.

Each subpackage is library code for one stage. Import the domain API from
the subpackage (lazy exports)::

    from st123.stages.download import query_jwst, parse_s_region
    from st123.stages.alignment import run_jhat
    from st123.stages.mosaic import plan_mosaic_boxes
    from st123.stages.photometry import prepare_mosaic_phot_job
    from st123.datamodels import open_datamodel, sanitize_science_fits

Command-line wrappers remain in :mod:`st123.scripts` until they are folded
into these packages. Shared run/logging/options behavior is
:class:`~st123.stages.stage.Stage`. Pipelines live in :mod:`st123.pipelines`.
"""

from __future__ import annotations

from st123.stages.stage import Stage

__all__ = ['Stage']
