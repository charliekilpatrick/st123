# External dependencies (`extdeps`)

Vendored / customized third-party packages used by **st123**.

| Path | Package | Role |
| --- | --- | --- |
| [`jhat/`](jhat/) | **jhat** (custom) | JWST/HST Alignment Tool used by `st123.stages.alignment` |

These packages are **not** installed from PyPI by default. The root
`requirements.txt` pulls in the custom JHAT build via a PEP 508 path
dependency (`jhat @ file:./extdeps/jhat`) when you run `pip install -e .`.
