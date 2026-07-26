##########
User guide
##########

Typical workflow
================

1. **Query / download** products from MAST (``download``, or library
   helpers in :mod:`st123.mast`).
2. **Align** JWST frames with JHAT (``align`` and related helpers under
   :mod:`st123.alignment`).
3. **Mosaic / coadd** (``mosaic``) under :mod:`st123.mosaic`.
4. **Prepare DOLPHOT** (``dolphot-prep``) and run the external ``dolphot``
   binary; optionally warm-start MIRI from NIRCam
   (``dolphot-warmstart``, :mod:`st123.photometry`).
5. **Catalog** photometry products (``catalog``).

Work-directory layout
=====================

CLIs generally take a ``--base-dir`` (or similar) pointing at a reduction
root. Logs are written under ``<base-dir>/logs/`` when package logging is
enabled. Instrument / filter products are typically arranged under
telescope and instrument directories after MAST download.

Console scripts
===============

Installed entry points (see ``pyproject.toml`` ``[project.scripts]``):

* ``download`` — MAST query / download
* ``align`` — JHAT alignment
* ``mosaic`` — mosaicking / coadds
* ``link-raw``, ``image-overlap``, ``region`` — path / footprint helpers
* ``dolphot-prep`` — DOLPHOT mask / calcsky / paramfile prep
  (console name kept distinct from the ``dolphot`` binary)
* ``dolphot-warmstart`` — NIRCam→MIRI warm-start setup
* ``catalog`` — combined photometry catalogs

Run ``<command> --help`` for options. Shared knobs such as ``--ncores``
propagate into DOLPHOT ``MaxThreads`` and other parallel stages where
applicable.

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
