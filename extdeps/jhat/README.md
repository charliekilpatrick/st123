# JHAT (custom build for st123)

This directory is a **custom, repository-local build of
[JHAT](https://github.com/arminrest/jhat)** (JWST/HST Alignment Tool) for use
with **st123**. It is **not** the unmodified PyPI / upstream package.

Upstream project: [arminrest/jhat](https://github.com/arminrest/jhat)  
Docs: https://jhat.readthedocs.io/

## Why this fork exists

st123’s MIRI↔NIRCam relative alignment pipeline (`st123.alignment`,
filter-specific calibrators, iterative refine, and `jhat_params` overrides)
depends on a JHAT install that:

- Uses modern **photutils** APIs (`photutils.aperture`, `photutils.detection`, …)
  compatible with the pinned stack in the root `requirements.txt`.
- Is installed **from this tree** so the same JHAT code is used in development,
  CI, and production reductions (no silent drift to a different PyPI wheel).

Do **not** `pip install jhat` from PyPI for this repository unless you
intentionally want upstream instead of this custom build.

## Install

From the **st123 repository root**:

```bash
pip install -e ./extdeps/jhat
pip install -e .
```

Or in one step (recommended):

```bash
pip install -e ./extdeps/jhat -e .
```

Verify the import resolves to this tree:

```bash
python -c "import jhat, pathlib; print(jhat.__file__)"
# expect: .../st123/extdeps/jhat/jhat/__init__.py
```

## Version

Package version is marked as a st123 local build (see `setup.py` /
`jhat.__version__`), distinct from the generic PyPI `jhat` release line.

## License / credit

JHAT is developed by Armin Rest, Justin Pierel, and collaborators. Please cite
upstream JHAT when publishing results that use it (see
https://jhat.readthedocs.io/).
