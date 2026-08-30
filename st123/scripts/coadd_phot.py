#!/usr/bin/env python3
"""Forced aperture photometry on JWST Level-3 / coadd images."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from st123.scripts.utils.options import (
    add_common_runtime,
    add_image_arg,
    configure_logging_from_args,
    create_parser as build_parser,
)
from st123.utils.logging import shutdown_logging

logger = logging.getLogger(__name__)


def create_parser():
    parser = build_parser(
        description=(
            'Forced aperture photometry on a JWST Level-3 / coadd image. '
            'Aperture radius and correction come from CRDS APCORR at EE=0.90 '
            '(NIRCam) or EE=0.80 (MIRI). Centers are supplied by the caller '
            '(pixel and/or equatorial); no source detection is performed.'
        ),
    )
    add_image_arg(
        parser,
        required=True,
        help='Level-3 / coadd FITS image (SCI in MJy/sr).',
    )
    parser.add_argument(
        '--xy',
        type=str,
        default=None,
        help='Single pixel position as X,Y (0-based FITS SCI pixels).',
    )
    parser.add_argument(
        '--ra',
        type=float,
        default=None,
        help='Single ICRS right ascension in degrees.',
    )
    parser.add_argument(
        '--dec',
        type=float,
        default=None,
        help='Single ICRS declination in degrees.',
    )
    parser.add_argument(
        '--coords',
        type=str,
        default=None,
        help=(
            'ASCII/ECSV table of positions with columns x,y and/or ra,dec '
            '(sky columns preferred when both are present).'
        ),
    )
    parser.add_argument(
        '--ee',
        type=float,
        default=None,
        help=(
            'Encircled-energy fraction override (0-1, or percent >1). '
            'Default: 0.90 NIRCam / 0.80 MIRI.'
        ),
    )
    parser.add_argument(
        '--outfile',
        type=str,
        default=None,
        help=(
            'Output catalog path (default: '
            '<image_stem>_forced_phot.ecsv).'
        ),
    )
    add_common_runtime(parser, plot=False, verbose=True)
    return parser


def _parse_xy(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in str(text).replace(' ', '').split(',') if p.strip()]
    if len(parts) != 2:
        raise ValueError(f'--xy must be X,Y; got {text!r}')
    return float(parts[0]), float(parts[1])


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    configure_logging_from_args(args, 'coadd-phot')
    try:
        from st123.stages.photometry.aperture import (
            forced_aperture_photometry,
            read_coords_table,
        )

        image = str(Path(args.image).expanduser().resolve())
        if not Path(image).is_file():
            raise FileNotFoundError(f'Image not found: {image}')

        kwargs: dict = {'ee_fraction': args.ee}
        n_sources = 0
        if args.coords:
            coord_kwargs, mode = read_coords_table(args.coords)
            kwargs.update(coord_kwargs)
            n_sources = len(next(iter(coord_kwargs.values())))
            logger.info(
                'Loaded %d position(s) from %s (%s)',
                n_sources,
                args.coords,
                mode,
            )
        else:
            has_xy = args.xy is not None
            has_sky = args.ra is not None or args.dec is not None
            if has_sky and (args.ra is None or args.dec is None):
                raise ValueError('Provide both --ra and --dec')
            if not has_xy and not has_sky:
                raise ValueError(
                    'Provide --xy and/or --ra/--dec, or --coords FILE'
                )
            if has_xy:
                x, y = _parse_xy(args.xy)
                kwargs['x'] = x
                kwargs['y'] = y
            if has_sky:
                kwargs['ra'] = args.ra
                kwargs['dec'] = args.dec
            n_sources = 1

        table = forced_aperture_photometry(image, **kwargs)
        outfile = args.outfile
        if not outfile:
            outfile = str(
                Path(image).with_name(f'{Path(image).stem}_forced_phot.ecsv')
            )
        outfile = str(Path(outfile).expanduser().resolve())
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        table.write(outfile, format='ascii.ecsv', overwrite=True)
        logger.info(
            'Wrote forced photometry for %d source(s) -> %s',
            len(table),
            outfile,
        )
        if args.verbose and len(table):
            row = table[0]
            logger.info(
                'First row: ra=%.6f dec=%.6f flux=%.4f+/-%.4f uJy '
                'AB=%.3f+/-%.3f (EE=%.2f r=%.2f px apcorr=%.4f)',
                float(row['ra']),
                float(row['dec']),
                float(row['flux_ujy']),
                float(row['flux_ujy_err']),
                float(row['abmag']),
                float(row['abmag_err']),
                float(row['ee_fraction']),
                float(row['radius_px']),
                float(row['apcorr']),
            )
        return 0
    except Exception as exc:
        logger.error('%s', exc)
        if getattr(args, 'verbose', False):
            logger.exception('coadd-phot failed')
        return 1
    finally:
        shutdown_logging()


if __name__ == '__main__':
    sys.exit(main())
