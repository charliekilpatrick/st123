#################################
Welcome to st123's documentation!
#################################

**st123** is a Python toolkit for space-telescope imaging: query and download
from MAST, align with a repository-local build of JHAT, build mosaics /
coadds, prepare DOLPHOT runs, and scrape photometry catalogs. Designed for
**HST, JWST, and Roman**, with current workflows strongest for JWST (and HST
reference / MAST helpers).

* :doc:`installation` - Python 3.11-3.12, Conda, optional DOLPHOT / JWST.
* :doc:`user_guide` - Work-directory layout, CLI overview, outputs.
* :doc:`api` - package modules (MAST, alignment, mosaic, photometry, utils).

Quick install
=============

From the repository root (see :doc:`installation` for detail):

.. code-block:: bash

   conda create -n st123 python=3.12 pip
   conda activate st123
   pip install -e .

Build this documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs && make html

The `documentation workflow
<https://github.com/charliekilpatrick/st123/actions/workflows/documentation.yml>`__
publishes to `GitHub Pages <https://charliekilpatrick.github.io/st123/>`__.

Repository layout
=================

- **st123/** - installable package: ``stages/`` (download, alignment, mosaic,
  photometry + ``stage.py`` primitive), ``datamodels/`` (HST/JWST FITS
  product types), ``pipelines/`` (``Pipeline``
  stub), ``scripts/`` (CLI wrappers), ``utils/``, ``notebooks/``.
- **extdeps/jhat/** - custom JHAT tree used by editable installs.
- **tests/** - ``pytest`` suite.
- **docs/** - Sphinx (reStructuredText + MyST Markdown): this site and changelog.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   user_guide
   changelog
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
