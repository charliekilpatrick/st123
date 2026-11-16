"""
st123 **CLI wrappers** for stages.

Each module here is a command-line wrapper (``align``, ``mosaic``,
``dolphot-prep``, ...) around library code in :mod:`st123.stages`. Shared
parser / logging helpers live in :mod:`st123.scripts.utils`. These wrappers
will eventually merge into the corresponding stage packages.

Campaign sequences belong in :mod:`st123.pipelines`, not here.

Only modules that expose a console-script ``main()`` belong in this package
root.
"""
