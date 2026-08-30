###
API
###

Package overview
================

.. automodule:: st123
   :members:
   :undoc-members:
   :show-inheritance:

Stages
======

Library code for each processing stage lives under :mod:`st123.stages`.
The shared primitive is :class:`~st123.stages.stage.Stage`. CLI wrappers
remain in :mod:`st123.scripts`.

.. automodule:: st123.stages
   :members:
   :undoc-members:

.. automodule:: st123.stages.stage
   :members:
   :undoc-members:
   :show-inheritance:

Download
========

.. automodule:: st123.stages.download
   :members:
   :undoc-members:
   :imported-members:

.. automodule:: st123.stages.download.mast
   :members:
   :undoc-members:

.. automodule:: st123.stages.download.download
   :members:
   :undoc-members:

Alignment
=========

.. automodule:: st123.stages.alignment
   :members:
   :undoc-members:
   :imported-members:

Mosaic
======

.. automodule:: st123.stages.mosaic
   :members:
   :undoc-members:
   :imported-members:

Photometry
==========

.. automodule:: st123.stages.photometry
   :members:
   :undoc-members:
   :imported-members:

.. automodule:: st123.stages.photometry.dolphot
   :members:
   :undoc-members:

.. automodule:: st123.stages.photometry.warmstart
   :members:
   :undoc-members:

.. automodule:: st123.stages.photometry.catalog
   :members:
   :undoc-members:

Datamodels
==========

On-disk FITS product types. Not a CLI stage. :func:`~st123.datamodels.open_datamodel`
returns the most specific class (JWST-NIRCam, HST-WFC3-IR, ...).

.. automodule:: st123.datamodels
   :members:
   :undoc-members:
   :imported-members:

.. automodule:: st123.datamodels.instrument
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: st123.datamodels.jwst
   :members:
   :undoc-members:
   :imported-members:
   :show-inheritance:

.. automodule:: st123.datamodels.hst
   :members:
   :undoc-members:
   :imported-members:
   :show-inheritance:

Pipelines
=========

Campaign pipelines that sequence stages. :class:`~st123.pipelines.pipeline.Pipeline`
is the wrapper; concrete pipelines are not implemented yet.

.. automodule:: st123.pipelines
   :members:
   :undoc-members:

.. automodule:: st123.pipelines.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
=========

.. automodule:: st123.utils.settings
   :members:
   :undoc-members:

.. automodule:: st123.utils.logging
   :members:
   :undoc-members:

.. automodule:: st123.utils.helpers
   :members:
   :undoc-members:
