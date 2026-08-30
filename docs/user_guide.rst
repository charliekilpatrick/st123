##########
User guide
##########

Typical workflow
================

1. **Query / download** products from MAST (``download``, or library
   helpers in :mod:`st123.stages.download`).
2. **Align** JWST frames with JHAT (``align`` and related helpers under
   :mod:`st123.stages.alignment`).
3. **Mosaic / coadd** (``mosaic``) under :mod:`st123.stages.mosaic`. Each
   ``reference/group_*/ref_*`` box gets a shared sky stamp
   (``stamp_wcs.fits``); JWST filters and HST drizzle products are built on
   that footprint so missions stay registered.
4. **Prepare DOLPHOT** (``dolphot-prep``) and run the external ``dolphot``
   binary. Warm-start options via ``dolphot-warmstart``
   (:mod:`st123.stages.photometry`): NIRCam->MIRI (``--target miri``, default) or
   NIRCam reference/catalog -> HST science (``--target hst``). See the README
   "HST one-target end-to-end" and "HST warmstart from NIRCam" sections for
   recommended globals (``UseWCS=2``, ``Align=0``, ``Force1=1``) and QA.
5. **Catalog** photometry products (``catalog``), or scrape a ``.phot`` with
   :func:`st123.stages.photometry.dolphot.nearest_phot_source`.

Work-directory layout
=====================

CLIs generally take a ``--base-dir`` (or similar) pointing at a reduction
root. Logs are written under ``<base-dir>/logs/`` when package logging is
enabled. Instrument / filter products are typically arranged under
telescope and instrument directories after MAST download.

Stages
======

Installed CLI entry points are **stages** (see ``pyproject.toml``
``[project.scripts]``). They are not pipelines; a pipeline layer that
composes stages is future work.

* ``download`` - MAST query / download
* ``align`` - JHAT alignment
* ``mosaic`` - mosaicking / coadds
* ``link-raw``, ``image-overlap``, ``region`` - path / footprint helpers
* ``dolphot-prep`` - DOLPHOT mask / calcsky / paramfile prep
  (console name kept distinct from the ``dolphot`` binary)
* ``run-dolphot`` - execute prepared DOLPHOT directories
* ``dolphot-warmstart`` - NIRCam->MIRI or NIRCam->HST warm-start setup
* ``catalog`` - combined photometry catalogs

Every stage starts from the same parser primitive
(:func:`st123.scripts.utils.options.create_parser`). Library implementations
live under :mod:`st123.stages`; campaign sequences will use
:class:`st123.pipelines.pipeline.Pipeline` (not yet implemented). Flags are
registered in the options API, not in the stage module. Stamp / mosaic-box
flags (``--existing-box``, ``--center-ra`` / ``--center-dec``,
``--stamp-size``, ``--stamp-id``, …) are accepted uniformly even when a given
stage ignores them. Shared knobs such as ``--ncores`` propagate into DOLPHOT
``MaxThreads`` and other parallel stages where applicable.

Run ``<stage> --help`` for the full option set.

Supported instruments
=====================

Filter / instrument tables live in :mod:`st123.utils.settings`
(``acceptable_filters``, ``FILTERS_BY_INSTRUMENT``), covering HST
(WFPC2, ACS, WFC3), JWST (NIRCam, MIRI), Roman, and Euclid groups used
for validation and MAST filtering.

Tips
====

* Prefer an editable install (``pip install -e .``) so the repo-local
  JHAT under ``extdeps/jhat/`` is used.
* Put the DOLPHOT ``bin`` directory on ``PATH`` before ``dolphot-prep`` /
  running ``dolphot``.
* For network tests or live MAST queries, set a MAST token via the usual
  environment / ``astroquery`` mechanisms.
