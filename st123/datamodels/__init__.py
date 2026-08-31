"""
On-disk science-product datamodels for st123.

:class:`InstrumentDataModel` is the common FITS-path handle. Telescope
subclasses (:class:`JWSTDataModel`, :class:`HSTDataModel`,
:class:`EuclidDataModel`, :class:`RomanDataModel`) own shared sanitize
and quality behavior. Instrument classes (NIRCam, MIRI, WFPC2, ACS/WFC,
ACS/HRC, WFC3/UVIS, WFC3/IR, Euclid VIS / NIR, Roman WFI) only override
identity and instrument-specific cues.

Use :func:`open_datamodel` / :func:`as_datamodel` to construct the most
specific class for a file. Instrument, filter, header keywords, WCS /
geometric distortion, and quality gates are methods on that handle::

    from st123.datamodels import as_datamodel
    frame = as_datamodel('jwst_nircam_cal.fits')
    frame.filter_name
    frame.wavelength_um
    frame.keyword('INSTRUME')
    with frame.open() as hdul:  # sanitizes, then yields the FITS handle
        ...
    frame.sci_wcs()
    frame.distortion_keywords()

WFPC2 ``c0m`` science and ``c1m`` DQ are one :class:`WFPC2DataModel`
(opening either file yields the pair).
"""

from __future__ import annotations

from st123.datamodels.euclid import (
    EuclidDataModel,
    EuclidNIRDataModel,
    EuclidVISDataModel,
)
from st123.datamodels.hst import (
    ACSDataModel,
    ACSHRCDataModel,
    ACSWFCDataModel,
    HSTDataModel,
    WFC3IRDataModel,
    WFC3UVISDataModel,
    WFPC2DataModel,
    filter_good_hst_frames,
    is_good_hst_alignment,
    is_good_hst_expflag,
    is_hst_science_path,
    log_rejected_hst_frames,
    prune_bad_hst_expflag,
    read_expflag,
)
from st123.datamodels.instrument import (
    DataModelLike,
    InstrumentDataModel,
    as_datamodel,
    as_datamodels,
    classify_image_kind,
    filter_paths_for_stage,
    materialize_fits_wcs_keywords,
    open_datamodel,
    path_of,
)
from st123.datamodels.jwst import (
    JWSTDataModel,
    MIRIDataModel,
    NIRCamDataModel,
    sanitize_jwst_l2,
)
from st123.datamodels.roman import RomanDataModel, RomanWFIDataModel

__all__ = [
    'ACSDataModel',
    'ACSHRCDataModel',
    'ACSWFCDataModel',
    'EuclidDataModel',
    'EuclidNIRDataModel',
    'EuclidVISDataModel',
    'HSTDataModel',
    'InstrumentDataModel',
    'JWSTDataModel',
    'MIRIDataModel',
    'NIRCamDataModel',
    'RomanDataModel',
    'RomanWFIDataModel',
    'WFC3IRDataModel',
    'WFC3UVISDataModel',
    'WFPC2DataModel',
    'DataModelLike',
    'as_datamodel',
    'as_datamodels',
    'classify_image_kind',
    'filter_good_hst_frames',
    'filter_paths_for_stage',
    'is_good_hst_alignment',
    'is_good_hst_expflag',
    'is_hst_science_path',
    'log_rejected_hst_frames',
    'materialize_fits_wcs_keywords',
    'open_datamodel',
    'path_of',
    'prune_bad_hst_expflag',
    'read_expflag',
    'sanitize_jwst_l2',
    'sanitize_science_fits',
]


def sanitize_science_fits(image, *, materialize_headers: bool = True):
    """Dispatch sanitization by telescope via :func:`as_datamodel`."""
    return as_datamodel(image).sanitize(materialize_headers=materialize_headers)
