"""Roman datamodels: shared Roman class plus WFI."""

from __future__ import annotations

from st123.datamodels.roman.roman import RomanDataModel
from st123.datamodels.roman.wfi import RomanWFIDataModel

__all__ = [
    'RomanDataModel',
    'RomanWFIDataModel',
]
