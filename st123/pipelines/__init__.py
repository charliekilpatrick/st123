"""
st123 pipelines: sequences of stages.

See :class:`~st123.pipelines.pipeline.Pipeline`. Campaign-specific subclasses
are a later layer; do not add CLI entry points here.
"""

from __future__ import annotations

from st123.pipelines.pipeline import Pipeline

__all__ = ['Pipeline']
