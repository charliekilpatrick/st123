#!/usr/bin/env python3
"""
Relative JHAT alignment of one JWST image to a reference, with dispersion metrics.

Thin CLI wrapper around :mod:`st123.alignment.relative_align`.

Example:

    python -m st123.scripts.relative_align \\
        --ref /path/to/coadd_i2d.fits \\
        --align /path/to/mirimage_cal.fits \\
        --outdir alignment_output
"""

from __future__ import annotations

from st123.alignment.relative_align import (  # noqa: F401
    build_master_ref_catalog,
    build_ref_catalog,
    create_parser,
    has_jwst_gwcs,
    load_sci_data_wcs,
    main,
    phot_catalog_path,
    read_dispersion_mas,
    refine_alignment_iteratively,
    resolve_outdir,
    run_alignment,
    stage_photfile,
    write_jhat_phot_table,
)

__all__ = [
    'build_master_ref_catalog',
    'build_ref_catalog',
    'create_parser',
    'has_jwst_gwcs',
    'load_sci_data_wcs',
    'main',
    'phot_catalog_path',
    'read_dispersion_mas',
    'refine_alignment_iteratively',
    'resolve_outdir',
    'run_alignment',
    'stage_photfile',
    'write_jhat_phot_table',
]


if __name__ == '__main__':
    raise SystemExit(main())
