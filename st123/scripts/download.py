#!/usr/bin/env python3
"""Download imaging products from MAST (HST and JWST)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from st123.scripts.utils.options import (
    add_base_dir,
    add_common_runtime,
    add_filters,
    add_mast_token,
    configure_logging_from_args,
    create_parser as build_parser,
    parse_filter_list,
    parse_instruments,
)
from st123.utils.logging import shutdown_logging
from st123.utils.settings import (
    DEFAULT_HST_INSTRUMENTS,
    DEFAULT_JWST_INSTRUMENTS,
    DOWNLOAD_DIR_NAME,
)

logger = logging.getLogger(__name__)

_VALID_TELESCOPES = frozenset({'hst', 'jwst'})
_JWST_INSTRUMENT_NAMES = frozenset(
    {'nircam', 'miri', 'niriss', 'nirspec', 'fgs'}
)
_HST_INSTRUMENT_NAMES = frozenset(
    {'acs', 'wfc3', 'wfpc2', 'wfc', 'nicmos', 'stis', 'cos', 'foc', 'hsp'}
)


def parse_telescopes(raw: Sequence[str] | str | None) -> list[str]:
    """
    Normalize ``--telescope`` values to a de-duplicated ``['hst'|'jwst', …]``.

    Accepts space- and/or comma-separated tokens. ``None`` / empty → ``['jwst']``.
    """
    if raw is None:
        return ['jwst']
    if isinstance(raw, str):
        tokens: list[str] = [raw]
    else:
        tokens = list(raw)
    out: list[str] = []
    for token in tokens:
        for part in str(token).replace(',', ' ').split():
            name = part.strip().lower()
            if not name:
                continue
            if name not in _VALID_TELESCOPES:
                raise ValueError(
                    f'Unsupported telescope {part!r}; choose from hst, jwst'
                )
            if name not in out:
                out.append(name)
    return out or ['jwst']


def infer_telescopes_from_instruments(
    instruments: Sequence[str],
) -> list[str]:
    """
    Infer ``['hst'|'jwst', …]`` from known instrument names (order of first hit).
    """
    out: list[str] = []
    for inst in instruments:
        key = str(inst).split('/')[0].strip().lower()
        if not key:
            continue
        if key in _JWST_INSTRUMENT_NAMES and 'jwst' not in out:
            out.append('jwst')
        elif key in _HST_INSTRUMENT_NAMES and 'hst' not in out:
            out.append('hst')
    if not out:
        raise ValueError(
            'Cannot infer --telescope from instruments '
            f'{list(instruments)}; use known HST (ACS, WFC3, …) or '
            'JWST (NIRCAM, MIRI, …) names, or pass --telescope explicitly'
        )
    return out


def resolve_telescopes(
    telescope_arg: Sequence[str] | str | None,
    instruments: Sequence[str] | None,
) -> list[str]:
    """
    Choose missions to download.

    Explicit ``--telescope`` wins. Otherwise, infer from ``--instruments``.
    With neither, default to JWST.
    """
    if telescope_arg is not None:
        return parse_telescopes(telescope_arg)
    if instruments:
        return infer_telescopes_from_instruments(instruments)
    return ['jwst']


def partition_instruments_by_telescope(
    instruments: Sequence[str] | None,
    telescopes: Sequence[str],
) -> dict[str, list[str]]:
    """
    Assign instruments to each requested telescope.

    When *instruments* is ``None``, each telescope gets its mission default.
    Named instruments are routed by known HST/JWST membership. Unknown names
    are assigned only when a single telescope was requested.
    """
    tels = [str(t).lower() for t in telescopes]
    if instruments is None:
        mapping = {
            'hst': list(DEFAULT_HST_INSTRUMENTS),
            'jwst': list(DEFAULT_JWST_INSTRUMENTS),
        }
        return {tel: list(mapping[tel]) for tel in tels}

    by_tel: dict[str, list[str]] = {tel: [] for tel in tels}
    unknown: list[str] = []
    for inst in instruments:
        key = str(inst).split('/')[0].strip().lower()
        if not key:
            continue
        if key in _JWST_INSTRUMENT_NAMES and 'jwst' in by_tel:
            by_tel['jwst'].append(str(inst))
        elif key in _HST_INSTRUMENT_NAMES and 'hst' in by_tel:
            by_tel['hst'].append(str(inst))
        else:
            unknown.append(str(inst))

    if unknown:
        if len(tels) == 1:
            by_tel[tels[0]].extend(unknown)
            logger.warning(
                'Assigning unrecognized instrument(s) %s to %s',
                unknown,
                tels[0].upper(),
            )
        else:
            raise ValueError(
                'Cannot assign instrument(s) '
                f'{unknown} with multiple --telescope values; '
                'use known HST (ACS, WFC3, WFPC2, …) or JWST '
                '(NIRCAM, MIRI, …) names'
            )
    return by_tel


def create_parser():
    parser = build_parser(
        description=(
            'Download imaging from MAST (HST and/or JWST). Pass --token (or set '
            'MAST_API_TOKEN) to authenticate and include proprietary data. '
            'Canonical layout is '
            '<base-dir>/download/<telescope>/<instrument>/<filter>/<obsid>/'
            '<filename> (e.g. …/download/HST/ACS/F814W/<obsid>/'
            'jey335ehq_flc.fits). Nested mastDownload/ trees are flattened. '
            'After a successful download, symlinks products into '
            '<base-dir>/reduction/raw (same as link-raw). '
            'The object name is the basename of --base-dir.'
        ),
    )
    add_base_dir(
        parser,
        required=True,
        aliases=(
            '--basedir',
            '--workdir',
            '--data-dir',
            '--download-dir',
            '--outdir',
        ),
        help=(
            'Project / dataset root (object name = basename). Products go '
            'under <base-dir>/download/…. Aliases: --basedir, --workdir, '
            '--data-dir, --download-dir, --outdir.'
        ),
    )
    parser.add_argument('--ra', type=str, required=True, help='RA of the target')
    parser.add_argument('--dec', type=str, required=True, help='DEC of the target')
    parser.add_argument(
        '--radius',
        type=float,
        default=3.0,
        help=(
            'MAST cone-search radius in arcminutes (default: 3.0). '
            'Passed through unchanged for both HST and JWST.'
        ),
    )
    parser.add_argument(
        '--telescope',
        nargs='+',
        default=None,
        metavar='TEL',
        help=(
            'Optional mission filter: hst and/or jwst. Omit when '
            '--instruments already imply the mission(s) '
            '(NIRCAM/MIRI → JWST; ACS/WFC3/WFPC2 → HST). '
            'With neither --telescope nor --instruments, defaults to jwst.'
        ),
    )
    parser.add_argument(
        '--stage', type=int, default=2, help='JWST calibration stage (2=CAL, 3=I2D)'
    )
    parser.add_argument(
        '--instruments',
        nargs='+',
        default=None,
        help=(
            'Instruments to include (also selects telescope when '
            '--telescope is omitted). Defaults: JWST→NIRCAM MIRI; '
            'HST→ACS WFC3 WFPC2. Space- or comma-separated mixed lists '
            'are fine (e.g. NIRCAM MIRI ACS WFC3).'
        ),
    )
    parser.add_argument(
        '--layout',
        choices=(
            'telescope/instrument/filter/obsid',
            'filter/obsid',
            'filter_obsid',
        ),
        default='telescope/instrument/filter/obsid',
        help=(
            'Per-observation directory layout under '
            f'<base-dir>/{DOWNLOAD_DIR_NAME}/.'
        ),
    )
    add_filters(parser)
    parser.add_argument(
        '--mirimage-only',
        action='store_true',
        help='Keep only MIRI imager (*mirimage*) products (JWST only).',
    )
    parser.add_argument(
        '--force-miri',
        action='store_true',
        help=(
            'Download MIRI even when the field has no NIRCam. By default, '
            'MIRI-only JWST fields are skipped when NIRCAM+MIRI are requested '
            '(use --instruments MIRI for the same effect).'
        ),
    )
    parser.add_argument(
        '--skip-link-raw',
        action='store_true',
        help=(
            'Do not symlink downloaded FITS into <base-dir>/reduction/raw '
            'after download (default: run link-raw automatically).'
        ),
    )
    add_mast_token(parser)
    add_common_runtime(parser, plot=False, dry_run=True)
    return parser


def _download_one_telescope(
    *,
    telescope: str,
    instruments: Sequence[str],
    coord,
    outdir: str,
    args,
    allowed_filters,
    query_mast_hst,
    query_mast_jwst,
    u,
):
    """
    Run MAST query/download for one mission.

    Returns
    -------
    int or MastDownloadResult
        HST returns an int product count. JWST returns
        :class:`~st123.mast.download.MastDownloadResult`.
    """
    radius = float(args.radius) * u.arcmin
    logger.info(
        '%s MAST cone search: center=%s radius=%s',
        telescope.upper(),
        coord.to_string('decimal'),
        radius,
    )
    if telescope == 'hst':
        return int(
            query_mast_hst(
                coord,
                outdir=outdir,
                radius=radius,
                token=args.token,
                instruments=list(instruments),
                layout=args.layout,
                dry_run=args.dry_run,
                allowed_filters=allowed_filters,
            )
        )

    mirimage_only = bool(args.mirimage_only)
    layout = args.layout
    if (
        instruments is not None
        and len(instruments) == 1
        and str(instruments[0]).upper() == 'MIRI'
    ):
        if layout in (
            'telescope/instrument/filter/obsid',
            'filter/obsid',
        ):
            mirimage_only = True
    return query_mast_jwst(
        coord,
        outdir=outdir,
        radius=radius,
        stage=args.stage,
        token=args.token,
        instruments=list(instruments),
        layout=layout,
        mirimage_only=mirimage_only,
        dry_run=args.dry_run,
        allowed_filters=allowed_filters,
        force_miri=bool(getattr(args, 'force_miri', False)),
    )


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    configure_logging_from_args(args, 'download')
    try:
        # Deferred: astroquery / MAST stack.
        from astropy import units as u

        from st123.mast.download import (
            query_mast_hst,
            query_mast_jwst,
            resolve_outdir,
        )
        from st123.mast.mast import normalize_filter_name
        from st123.scripts.link_raw import link_raw_tree
        from st123.utils.helpers import parse_coord

        coord = parse_coord(args.ra, args.dec)
        if coord is None:
            return 1

        project_root = Path(args.base_dir).expanduser()
        try:
            outdir = resolve_outdir(project_root / DOWNLOAD_DIR_NAME)
        except (PermissionError, ValueError) as exc:
            logger.error('%s', exc)
            return 1

        instruments = parse_instruments(args.instruments)
        try:
            telescopes = resolve_telescopes(args.telescope, instruments)
            inst_by_tel = partition_instruments_by_telescope(
                instruments, telescopes
            )
        except ValueError as exc:
            logger.error('%s', exc)
            return 1

        # Drop telescopes with an empty instrument list after partitioning.
        active = [(t, inst_by_tel[t]) for t in telescopes if inst_by_tel[t]]
        skipped = [t for t in telescopes if not inst_by_tel[t]]
        for tel in skipped:
            logger.warning(
                'Skipping %s: no instruments assigned from %s',
                tel.upper(),
                list(instruments) if instruments else '(defaults)',
            )
        if not active:
            logger.error('No telescopes left to download after instrument routing')
            return 1

        allowed = None
        if args.filters:
            allowed = [
                normalize_filter_name(f)
                for f in parse_filter_list(args.filters) or []
            ]

        counts: dict[str, object] = {}
        skipped_miri_only: set[str] = set()
        try:
            for telescope, tel_instruments in active:
                logger.info(
                    'Downloading %s instruments=%s → %s',
                    telescope.upper(),
                    ', '.join(tel_instruments),
                    outdir,
                )
                result = _download_one_telescope(
                    telescope=telescope,
                    instruments=tel_instruments,
                    coord=coord,
                    outdir=outdir,
                    args=args,
                    allowed_filters=allowed,
                    query_mast_hst=query_mast_hst,
                    query_mast_jwst=query_mast_jwst,
                    u=u,
                )
                counts[telescope] = result
                if (
                    telescope == 'jwst'
                    and getattr(result, 'skipped_miri_only', False)
                ):
                    skipped_miri_only.add(telescope)
        except (RuntimeError, OSError) as exc:
            logger.error('%s', exc)
            return 1

        def _count_n(value: object) -> int:
            return int(value)

        failed = [
            t
            for t, n in counts.items()
            if _count_n(n) == 0 and t not in skipped_miri_only
        ]
        if failed:
            logger.error(
                'No matching products for: %s '
                '(empty search or all downloads failed)',
                ', '.join(t.upper() for t in failed),
            )
            if len(failed) == len(counts):
                return 1
            # Partial success: continue to link whatever landed, but exit 1.
            partial_fail = True
        else:
            partial_fail = False

        # Mirror link-raw per requested instrument (never instrument=ALL —
        # that would re-link leftover trees like WFPC2 from earlier runs).
        if args.dry_run:
            logger.info('Dry run: skipping link-raw into reduction/raw')
        elif args.skip_link_raw:
            logger.info('Skipping link-raw (--skip-link-raw)')
        else:
            for telescope, tel_instruments in active:
                if telescope in skipped_miri_only:
                    continue
                if _count_n(counts.get(telescope, 0)) == 0:
                    continue
                for link_inst in tel_instruments:
                    logger.info(
                        'Linking downloaded FITS into reduction/raw '
                        '(telescope=%s instrument=%s)',
                        telescope.upper(),
                        link_inst,
                    )
                    try:
                        n_links = link_raw_tree(
                            base_dir=str(project_root),
                            telescope=telescope.upper(),
                            instrument=link_inst,
                            verbose=args.verbose,
                        )
                        logger.info(
                            'Linked %d FITS file(s) under reduction/raw '
                            '(%s/%s)',
                            n_links,
                            telescope.upper(),
                            link_inst,
                        )
                    except ValueError as exc:
                        logger.error(
                            'link-raw after download failed: %s', exc
                        )
                        return 1

        return 1 if partial_fail else 0
    finally:
        shutdown_logging()


if __name__ == '__main__':
    raise SystemExit(main())
