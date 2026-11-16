"""Euclid datamodels: shared Euclid class plus VIS and NIR (NISP)."""

from __future__ import annotations

from st123.datamodels.euclid.euclid import EuclidDataModel
from st123.datamodels.euclid.nir import EuclidNIRDataModel
from st123.datamodels.euclid.vis import EuclidVISDataModel

__all__ = [
    'EuclidDataModel',
    'EuclidNIRDataModel',
    'EuclidVISDataModel',
]
