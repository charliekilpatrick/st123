############
Installation
############

Requirements
============

* **Python 3.11–3.12** (CI targets 3.12; ``requires-python`` is ``>=3.11,<3.13``)
* Conda is recommended for a clean environment
* Optional: a compiled **DOLPHOT** binary on ``PATH`` for PSF photometry
* Optional: **JWST** / **STPSF** stacks when running JWST-specific pipelines

Conda environment
=================

From the repository root:

.. code-block:: bash

   conda create -n st123 python=3.12 pip
   conda activate st123
   pip install -e .

Editable install pulls the custom JHAT under ``extdeps/jhat/`` via
``pyproject.toml`` path dependencies when present.

Optional extras
===============

.. code-block:: bash

   # Documentation (Sphinx + RTD theme)
   pip install -e ".[docs]"

   # Test dependencies
   pip install -e ".[test]"

   # Dev tooling (ruff / mypy / pre-commit, if defined)
   pip install -e ".[dev]"

Verify
======

.. code-block:: bash

   python -c "import st123; print(st123.__version__)"
   download --help
   align --help
   dolphot-prep --help

Building the docs
=================

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html
   # open build/html/index.html

On pushes to ``main``, the GitHub Actions workflow
``.github/workflows/documentation.yml`` builds this tree and deploys
``docs/build/html`` to the ``gh-pages`` branch
(`https://charliekilpatrick.github.io/st123/
<https://charliekilpatrick.github.io/st123/>`__).
