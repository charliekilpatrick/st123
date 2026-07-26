from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shapely
import shapely.ops
import stpsf
from asdf import AsdfFile
from astropy import coordinates as coord
from astropy import nddata
from astropy import units as u
from astropy import wcs
from astropy.convolution import convolve_fft
from astropy.io import fits
from astropy.modeling import models
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats as scs
from astropy.table import Table
from ccdproc import Combiner
from gwcs import FITSImagingWCSTransform, coordinate_frames as cf
from gwcs import WCS as g_wcs
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
from jwst.pipeline import calwebb_image3
from matplotlib.patches import Polygon
from photutils.psf.matching import SplitCosineBellWindow, create_matching_kernel
from reproject.mosaicking import find_optimal_celestial_wcs

from st123.mast import parse_s_region
from st123.utils.compatibility import patch_jwst_for_photutils3
from st123.utils.helpers import create_filter_table, input_list
from st123.utils.logging import capture_output

logger = logging.getLogger(__name__)


def mp_init(
    init_success: int = 0,
    init_failed: int = 0,
    init_success_files: list[str] | None = None,
) -> None:
    """
    Initialize module-level counters for multiprocessing mosaic workers.

    Parameters
    ----------
    init_success : int, optional
        Initial success count.
    init_failed : int, optional
        Initial failure count.
    init_success_files : list of str, optional
        Paths of successfully processed files.

    Returns
    -------
    None
    """
    if init_success_files is None:
        init_success_files = []
    global success
    global failed
    global success_files
    success = init_success
    failed = init_failed
    success_files = init_success_files

class split_observations(object):
    """
    Split an input list into spatial boxes for Level-3 mosaic / coadd groups.

    Footprints are derived from ``S_REGION`` polygons (or caller-supplied
    shapely geometries) and recursively subdivided until each box contains at
    most ``N_max`` images.
    """

    def __init__(
        self,
        table: Table,
        N_max: int = 150,
        min_overlap: float = 0.2,
        pad: float = 15,
        polygons: Sequence[Any] | None = None,
        wcs_opt: wcs.WCS | None = None,
    ) -> None:
        """
        Parameters
        ----------
        table : Table
            Input list with an ``image`` column.
        N_max : int, optional
            Maximum images per mosaic box.
        min_overlap : float, optional
            Minimum fractional overlap required to assign an image to a box.
        pad : float, optional
            Pixel padding applied to each accepted box.
        polygons : sequence, optional
            Precomputed shapely polygons (one per table row).
        wcs_opt : astropy.wcs.WCS, optional
            Pixel WCS matching ``polygons`` when supplied.
        """
        self.table = table
        self.N_max = N_max
        self.min_overlap = min_overlap
        self.pad = pad

        if polygons is None:
            self.wcs, self.pgons, self.centroids = self.get_pgons(table)
        else:
            self.pgons = np.array(polygons)
            self.wcs = wcs_opt
            self.centroids = np.array([i.centroid for i in polygons])

        self.split_boxes = []
        self.subimages = []
        self.reftables = []
        self.refpgons = []
        self.filtertables = []
        self.physical_split = True
        self.check_box = None

    def get_pgons(self, table):
        image_hdus = [fits.open(i)[1] for i in table['image']]
        wcs_out, shape_out = find_optimal_celestial_wcs(image_hdus, auto_rotate = True)

        wcs_header = wcs_out.to_header()
        wcs_header['NAXIS1'] = shape_out[1]
        wcs_header['NAXIS2'] = shape_out[0]
        wcs_opt = wcs.WCS(wcs_header)
        
        pgons, centroids = [], []
        for im in table['image']:
            region = fits.open(im)['SCI'].header['S_REGION']
            sky = np.asarray(parse_s_region(region).exterior.coords[:-1])
            x, y = wcs_opt.all_world2pix(sky[:, 0], sky[:, 1], 0)
            xy_coords = np.column_stack((x, y))
            pgons.append(shapely.Polygon(xy_coords))
            centroids.append(shapely.Polygon(xy_coords).centroid)

        pgons, centroids = np.array(pgons), np.array(centroids)

        return wcs_opt, pgons, centroids
    
    def line_split(self, bounds, xs, ys, split_size=150, split_by='x', physical=False):
        lstrings = []
        
        if split_by == 'x':
            if physical:
                i = (bounds[0]+bounds[2])/2
                lstrings.append(shapely.LineString([[i, bounds[1]], [i, bounds[3]]]))
            else:
                cst = np.argsort(xs)
                xs, ys = np.array(xs)[cst], np.array(ys)[cst]
                split_lines = xs[0::split_size][1:]
                for i in split_lines:
                    lstrings.append(shapely.LineString([[i, bounds[1]], [i, bounds[3]]]))

        elif split_by == 'y':
            if physical:
                i = (bounds[1]+bounds[3])/2
                lstrings.append(shapely.LineString([[bounds[0], i], [bounds[2], i]]))
            else:
                cst = np.argsort(ys)
                xs, ys = np.array(xs)[cst], np.array(ys)[cst]
                split_lines = ys[0::split_size][1:]
                for i in split_lines:
                    lstrings.append(shapely.LineString([[bounds[0], i], [bounds[2], i]]))

        else:
            raise ValueError("Must be split by x or y")
        
        lstrings = shapely.MultiLineString(lstrings)

        return lstrings
    
    def find_intersections(self, bbox=None, min_overlap=0.05):

        if bbox is None:
            bbox = self.split_boxes

        int_area = np.array([bbox.intersection(p).area/p.area for p in self.pgons])
        mask = int_area > min_overlap
        return mask
    
    def get_bbox(self, pgons=None):
        
        if pgons is None:
            pgons = self.pgons

        bounds = shapely.unary_union(pgons).bounds
        bbox_coords = [[bounds[0], bounds[1]], [bounds[0], bounds[3]], [bounds[2], bounds[3]], [bounds[2], bounds[1]]]
        bbox = shapely.Polygon(bbox_coords)

        return bbox
    
    def pad_box(self, box, pad=0):

        bounds = box.bounds
        padded_box_coords = [[bounds[0]-pad, bounds[1]-pad], [bounds[0]-pad, bounds[3]+pad], [bounds[2]+pad, bounds[3]+pad], [bounds[2]+pad, bounds[1]-pad]]
        padded_box_coords = np.array(padded_box_coords)
        padded_box_coords[padded_box_coords < 0] = 0
        bbox = shapely.Polygon(padded_box_coords)

        return bbox
    
    def boxsplit(self, bbox=None, split_n=None):

        self.check_box = bbox

        if bbox is None:
            bbox = self.get_bbox(pgons=self.pgons)

        mask = self.find_intersections(bbox, min_overlap=self.min_overlap)
        if mask.sum() < self.N_max:
            bbox = self.pad_box(bbox, pad=self.pad)
            self.split_boxes.append(bbox)
            self.subimages.append(self.table[mask]['image'])
            ref_mask = self.find_intersections(bbox, min_overlap=0.0)
            self.reftables.append(self.table[ref_mask])
            self.refpgons.append(self.pgons[ref_mask])
            return None

        else:
            spl_pgons, spl_centroids  = self.pgons[mask], self.centroids[mask]
            bounds = bbox.bounds
            width, height = bounds[3] - bounds[1], bounds[2] - bounds[0]
            split_by = 'x' if height > width else 'y'
            if split_n is None:
                split_n = len(spl_pgons)//2 + 1

            xs = [i.x for i in spl_centroids]
            ys = [i.y for i in spl_centroids]
            lstrings = self.line_split(bounds, xs, ys, split_size=split_n, split_by=split_by, physical=self.physical_split)
            
            split_bbox = []
            for ln in lstrings.geoms:
                linesplit = shapely.ops.split(bbox, ln)
                if len(linesplit.geoms) > 1:
                    split_bbox.append(linesplit.geoms[0])
                    bbox = linesplit.geoms[1]
                else:
                    bbox = linesplit.geoms[0]
            split_bbox.append(bbox)

            for split_box in split_bbox:
                if self.check_box == split_box:
                    logger.warning(
                        'Cannot split further, minmimum group size is %s',
                        mask.sum(),
                    )
                    split_box = self.pad_box(split_box, pad=self.pad)
                    self.split_boxes.append(split_box)
                    self.subimages.append(self.table[mask]['image'])
                    ref_mask = self.find_intersections(split_box, min_overlap=0.0)
                    self.reftables.append(self.table[ref_mask])
                    self.refpgons.append(self.pgons[ref_mask])
                    continue
                _ = self.boxsplit(split_box)
            
            return None
        
    def get_sw_filter_tables(self, tol = 0.05):
        for b in range(len(self.split_boxes)):
            bbox, reftable, refpgons = self.split_boxes[b], self.reftables[b], self.refpgons[b]
            filters = np.unique(reftable['filter'])
            swmask = np.array([int(i[1:4]) for i in filters]) < 215
            swmask = swmask & np.array(['n' not in i for i in filters])

            filter_footprints = []
            for filter_name in filters[swmask]:
                filter_footprints.append(shapely.unary_union(refpgons[reftable['filter'] == filter_name]))
            filter_footprints = np.array(filter_footprints)
            shortwave_union = shapely.unary_union(filter_footprints)
            
            net_ref_pgon = shapely.unary_union(refpgons).intersection(bbox)
            best_tol = net_ref_pgon.difference(shortwave_union).area/net_ref_pgon.area
            tol = max(tol, best_tol)
            total_ref_area, uncovered_frac = net_ref_pgon.area, 1
            ref_filters = []

            while uncovered_frac > tol:
                max_int = np.argmax([i.intersection(net_ref_pgon).area/net_ref_pgon.area for i in filter_footprints])
                footprint = filter_footprints[max_int]
                ref_filters.append(filters[swmask][max_int])
                uncovered_frac = net_ref_pgon.difference(footprint).area/total_ref_area
                net_ref_pgon = net_ref_pgon.difference(footprint)

            filter_table = create_filter_table(reftable, ref_filters)
            self.filtertables.append(filter_table)

        return self.filtertables
    
    def add_plt_patch(self, pgon, ax, facecolor = 'lightblue', edgecolor = 'blue', alpha = 0.3):
        vertices = np.array(pgon.exterior.xy)
        polygon = Polygon(vertices.T, closed=True, facecolor=facecolor, edgecolor=edgecolor, alpha = alpha)
        ax.add_patch(polygon)

    def plot_obs(self, cents = False):
        fig, ax = plt.subplots(1, 1)
        for polygon in self.pgons:
            self.add_plt_patch(polygon, ax)
        for polygon in self.split_boxes:
            self.add_plt_patch(polygon, ax, facecolor = 'none', edgecolor = 'black', alpha = 1)
        if cents:
            ax.scatter([i.x for i in self.centroids], [i.y for i in self.centroids], color = 'mediumvioletred', s = 2)
        xmin, xmax = min([min(i.exterior.xy[0]) for i in self.pgons]), max([max(i.exterior.xy[0]) for i in self.pgons])
        ymin, ymax = min([min(i.exterior.xy[1]) for i in self.pgons]), max([max(i.exterior.xy[1]) for i in self.pgons])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('x (pix)')
        ax.set_ylabel('y (pix)')
        
        
def get_pgons(
    table: Table,
) -> tuple[wcs.WCS, np.ndarray, np.ndarray]:
    """
    Build pixel-space shapely footprints for each row in an input list.

    Parameters
    ----------
    table : Table
        Input list with an ``image`` column.

    Returns
    -------
    wcs_opt : astropy.wcs.WCS
        Optimal celestial WCS projected to pixels.
    pgons : numpy.ndarray
        Shapely polygons, one per image.
    centroids : numpy.ndarray
        Polygon centroids in pixel coordinates.
    """
    image_hdus = [fits.open(i)[1] for i in table['image']]
    wcs_out, shape_out = find_optimal_celestial_wcs(image_hdus, auto_rotate = True)

    wcs_header = wcs_out.to_header()
    wcs_header['NAXIS1'] = shape_out[1]
    wcs_header['NAXIS2'] = shape_out[0]
    wcs_opt = wcs.WCS(wcs_header)
    
    pgons, centroids = [], []
    for im in table['image']:
        region = fits.open(im)['SCI'].header['S_REGION']
        sky = np.asarray(parse_s_region(region).exterior.coords[:-1])
        x, y = wcs_opt.all_world2pix(sky[:, 0], sky[:, 1], 0)
        xy_coords = np.column_stack((x, y))
        pgons.append(shapely.Polygon(xy_coords))
        centroids.append(shapely.Polygon(xy_coords).centroid)

    pgons, centroids = np.array(pgons), np.array(centroids)

    return wcs_opt, pgons, centroids
        
def create_default_mosaic(
    inputfiles: Sequence[str],
    outdir: str,
    filt: str,
) -> None:
    """
    Create a Level-3 drizzled mosaic from Level-2 inputs with default JWST options.

    Parameters
    ----------
    inputfiles : sequence of str
        Level-2 science FITS paths.
    outdir : str
        Output directory for association and pipeline products.
    filt : str
        Filter name used to select inputs and name outputs.

    Returns
    -------
    None
    """
    patch_jwst_for_photutils3()
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    table = input_list(inputfiles)
    table = table[table['filter'] == filt]
    asn_file = f'{outdir}/{filt}.json'
    base_filenames = np.array([os.path.basename(r['image']) for r in table])
    asn3 = asn_from_list.asn_from_list(base_filenames,
        rule=DMS_Level3_Base, product_name=f'{filt}')

    with open(asn_file, 'w') as outfile:
        name, serialized = asn3.dump(format='json')
        outfile.write(serialized)

    image3 = calwebb_image3.Image3Pipeline()

    outdir_level3 = os.path.join(outdir, f'out_{filt}')
    if not os.path.exists(outdir_level3):
        os.makedirs(outdir_level3)

    image3.output_dir = outdir_level3
    image3.save_results = True
    image3.tweakreg.skip = True
    image3.skymatch.skip = True
    image3.skymatch.match_down = False
    image3.source_catalog.skip=False
    image3.resample.pixfrac = 1.0
    image3.pixel_scale = 0.0311
    image3.weight_type = 'ivm'

    with capture_output():
        image3.run(asn_file)

def create_coadd_mosaic(
    table: Table,
    outdir: str,
    filt: str,
    *,
    sci_header: fits.Header | None = None,
    wcs_out: wcs.WCS | None = None,
    shape_out: tuple[int, int] | None = None,
    gwcs_file: str | None = None,
) -> str:
    """
    Create a Level-3 drizzled mosaic with a shared output GWCS.

    A ``FITSImagingWCSTransform``-based output WCS is always applied
    (via :func:`create_gwcs`) so resample can write FITS WCS keywords
    under jwst>=1.20. Provide an existing ``gwcs_file``, or inputs to
    build one (``sci_header``, or ``wcs_out`` + ``shape_out``).

    Parameters
    ----------
    table : Table
        Input list of images to resample.
    outdir : str
        Output directory for association and pipeline products.
    filt : str
        Filter name for the resampled product.
    sci_header : fits.Header, optional
        FITS WCS header used to build ``mosaic_gwcs.asdf`` when
        ``gwcs_file`` is not given.
    wcs_out : astropy.wcs.WCS, optional
        Astropy WCS used with ``shape_out`` when ``gwcs_file`` /
        ``sci_header`` are not given.
    shape_out : tuple of int, optional
        ``(NAXIS2, NAXIS1)`` for ``wcs_out``.
    gwcs_file : str, optional
        Path to an existing GWCS asdf file (from :func:`create_gwcs`).

    Returns
    -------
    str
        Path to the resampled ``*_i2d.fits`` image.
    """
    patch_jwst_for_photutils3()
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if gwcs_file is None:
        gwcs_file = create_gwcs(
            outdir=outdir,
            sci_header=sci_header,
            wcs_out=wcs_out,
            shape_out=shape_out,
        )
    elif not os.path.exists(gwcs_file):
        raise FileNotFoundError(f'gwcs_file not found: {gwcs_file}')

    asn_file = f'{outdir}/{filt}.json'
    base_filenames = np.array([os.path.basename(r['image']) for r in table])
    asn3 = asn_from_list.asn_from_list(base_filenames,
        rule=DMS_Level3_Base, product_name=f'{filt}')

    with open(asn_file, 'w') as outfile:
        name, serialized = asn3.dump(format='json')
        outfile.write(serialized)

    image3 = calwebb_image3.Image3Pipeline()

    outdir_level3 = os.path.join(outdir, f'out_{filt}')
    if not os.path.exists(outdir_level3):
        os.makedirs(outdir_level3)

    image3.output_dir = outdir_level3
    image3.save_results = True
    image3.tweakreg.skip = True
    image3.skymatch.skip = True
    image3.skymatch.match_down = False
    image3.source_catalog.skip=False
    image3.resample.pixfrac = 1.0
    image3.pixel_scale = 0.0311
    image3.weight_type = 'ivm'
    image3.resample.output_wcs = gwcs_file

    with capture_output():
        image3.run(asn_file)

    filepath = f'{outdir}/out_{filt}/{filt}_i2d.fits'
    return filepath


def create_gwcs(
    outdir: str,
    sci_header: fits.Header | None = None,
    wcs_out: wcs.WCS | None = None,
    shape_out: tuple[int, int] | None = None,
    return_gwcs: bool = False,
) -> str | g_wcs:
    """
    Convert an astropy WCS to GWCS and write or return it.

    Parameters
    ----------
    outdir : str
        Directory for ``mosaic_gwcs.asdf`` when ``return_gwcs`` is False.
    sci_header : fits.Header, optional
        FITS WCS header defining the output mosaic grid.
    wcs_out : astropy.wcs.WCS, optional
        Astropy WCS converted with ``shape_out`` when ``sci_header`` is omitted.
    shape_out : tuple of int, optional
        ``(NAXIS2, NAXIS1)`` shape paired with ``wcs_out``.
    return_gwcs : bool, optional
        When True, return the in-memory GWCS object instead of writing asdf.

    Returns
    -------
    str or gwcs.wcs.WCS
        Path to ``mosaic_gwcs.asdf``, or the GWCS object when
        ``return_gwcs`` is True.
    """

    if sci_header:
        pass
    elif wcs_out:
        sci_header = wcs_out.to_header()
        sci_header['NAXIS1'] = shape_out[1]
        sci_header['NAXIS2'] = shape_out[0]
    else:
        raise ValueError("Please provide header or wcs object")

    # jwst>=1.20 ResampleImage.update_fits_wcsinfo expects a
    # FITSImagingWCSTransform (with .crpix/.cdelt/.crval/.pc). A plain
    # CompoundModel of Shift|Affine|Scale|TAN|Rotate raises
    # AttributeError: Attribute "crpix" not found (jwst#10377).
    # crpix here is 0-indexed detector pixels (FITS CRPIX minus 1).
    matrix = np.array(
        [
            [sci_header['PC1_1'], sci_header['PC1_2']],
            [sci_header['PC2_1'], sci_header['PC2_2']],
        ]
    )
    det2sky = FITSImagingWCSTransform(
        models.Pix2Sky_TAN(),
        crpix=[sci_header['CRPIX1'] - 1, sci_header['CRPIX2'] - 1],
        crval=[sci_header['CRVAL1'], sci_header['CRVAL2']],
        cdelt=[sci_header['CDELT1'], sci_header['CDELT2']],
        pc=matrix,
    )
    det2sky.name = 'linear_transform'

    detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"),
                                unit=(u.pix, u.pix))
    sky_frame = cf.CelestialFrame(reference_frame=coord.ICRS(), name='world',
                                unit=(u.deg, u.deg))

    pipeline = [(detector_frame, det2sky),
                (sky_frame, None)
            ]
    wcsobj = g_wcs(pipeline)
    wcsobj.bounding_box = ((0, sci_header['NAXIS1']), (0, sci_header['NAXIS2']))

    if return_gwcs:
        return wcsobj

    else:
        #write gwcs to asdf file
        tree = {"wcs": wcsobj}
        wcs_file = AsdfFile(tree)
        gwcs_path = f"{outdir}/mosaic_gwcs.asdf"
        wcs_file.write_to(gwcs_path)

    return gwcs_path

def find_optimal_wcs(
    filter_table: dict[str, Table],
) -> tuple[wcs.WCS, tuple[int, int]]:
    """
    Find the optimal celestial WCS spanning all images in a filter table dict.

    Parameters
    ----------
    filter_table : dict
        Filter name mapped to an input-list :class:`~astropy.table.Table`.

    Returns
    -------
    wcs_out : astropy.wcs.WCS
        Optimal output WCS.
    shape_out : tuple of int
        ``(NAXIS2, NAXIS1)`` shape for the mosaic grid.
    """
    images = np.hstack([filter_table[i]['image'].value for i in filter_table.keys()])
    image_hdus = [fits.open(i)[1] for i in images]
    wcs_out, shape_out = find_optimal_celestial_wcs(image_hdus, auto_rotate = True)

    return wcs_out, shape_out

def create_psf_kernel(
    ref_filter: str,
    in_filter: str,
    ovs: int = 5,
    fov: int = 81,
) -> np.ndarray:
    """
    Build a photutils PSF-matching kernel between two NIRCam filters.

    Parameters
    ----------
    ref_filter : str
        Reference filter name (e.g. ``F277W``).
    in_filter : str
        Source filter to match to ``ref_filter``.
    ovs : int, optional
        STPSF oversampling factor.
    fov : int, optional
        STPSF field of view in pixels.

    Returns
    -------
    numpy.ndarray
        Matching kernel for :func:`astropy.convolution.convolve_fft`.
    """
    nrc = stpsf.NIRCam()
    nrc.filter = in_filter.upper()
    if nrc.filter == 'F150W2':
        nrc.SHORT_WAVELENGTH_MAX = 2.39e-6
    nrc.detector = 'NRCA3'
    psf_src = nrc.calc_psf(oversample=ovs, fov_pixels=fov) 

    #use detector distorted version
    psf_src_dat = psf_src[3].data/psf_src[3].data.sum()

    nrc.filter = ref_filter.upper()
    if nrc.filter == 'F150W2':
        nrc.SHORT_WAVELENGTH_MAX = 2.39e-6
    psf_ref = nrc.calc_psf(oversample=ovs, fov_pixels=fov)
    psf_ref_dat = psf_ref[3].data/psf_ref[3].data.sum()

    window = SplitCosineBellWindow(1.5, 1.3)
    psf_kernel = create_matching_kernel(psf_src_dat, psf_ref_dat, window=window) 

    return psf_kernel

def convolve_images(
    filter_table: dict[str, Table],
    target_filter: str,
) -> None:
    """
    PSF-match and overwrite science images to ``target_filter`` in place.

    Parameters
    ----------
    filter_table : dict
        Filter name mapped to an input-list :class:`~astropy.table.Table`.
    target_filter : str
        Reference filter for PSF matching.

    Returns
    -------
    None
    """
    for filt in filter_table.keys():
        if filt.upper() == target_filter.upper():
            continue
        psf_kernel = create_psf_kernel(target_filter, filt)
        tbl = filter_table[filt]
        for im in tbl['image']:
            hdu = fits.open(im)
            sci, err = hdu['SCI'].data, hdu['ERR'].data
            sci_con = convolve_fft(sci, psf_kernel, normalize_kernel=True)
            err_con = convolve_fft(err, psf_kernel, normalize_kernel=True)
            sci_header = hdu['SCI'].header
            sci_header['filter'] = target_filter.upper()

            hdu['SCI'].header, hdu['SCI'].data = sci_header, sci_con
            hdu['ERR'].data = err_con
            hdu.writeto(im, overwrite=True)

def create_ccddata(file: str) -> CCDData:
    """
    Load a JWST i2d FITS file as :class:`~astropy.nddata.CCDData`.

    Parameters
    ----------
    file : str
        Path to a Level-3 or coadd FITS file with SCI/ERR extensions.

    Returns
    -------
    CCDData
        Science data, uncertainty, WCS, and zero mask.
    """
    hdu = fits.open(file)
    sci_data = hdu['SCI'].data
    
    uncertainty = nddata.StdDevUncertainty(array = hdu['ERR'].data)
    data_unit = u.MJy/u.sr
    w = wcs.WCS(hdu['SCI'].header)
    mask = sci_data == 0
    ccd_data = CCDData(data = sci_data, uncertainty = uncertainty, 
                       wcs = w, unit = data_unit)
    
    return ccd_data

def update_photmjsr(
    ccddata: Sequence[CCDData],
    phots: Sequence[float],
) -> float:
    """
    Estimate a combined PHOTMJSR from weighted coadd inputs.

    Parameters
    ----------
    ccddata : sequence of CCDData
        Per-image science arrays in MJy/sr.
    phots : sequence of float
        Per-image ``PHOTMJSR`` header values.

    Returns
    -------
    float
        Sigma-clipped median conversion factor (MJy/sr per count).
    """
    ccd_mjsr = np.sum([ccd.data for ccd in ccddata], axis = 0)
    ccd_cps = np.sum([ccd.data/phot for ccd, phot in list(zip(ccddata, phots))], axis = 0)
    mjsr = ccd_mjsr/ccd_cps
    _, mjsr_med, _ = scs(mjsr)

    return mjsr_med

def coadd(
    ref_files: Sequence[str],
    filt: str,
    filename: str = 'coadd_i2d.fits',
) -> None:
    """
    Inverse-variance coadd Level-3 images with ccdproc.

    Parameters
    ----------
    ref_files : sequence of str
        Input ``*_i2d.fits`` paths to combine.
    filt : str
        Filter name written to the coadd primary header.
    filename : str, optional
        Output coadd FITS path.

    Returns
    -------
    None
    """
    #edit specific header keys
    hdu_template = fits.open(ref_files[0])
    hdr_update = {'EFFEXPTM': [], 'TMEASURE': [], 'DURATION': []}
    filters, phots = [], []
    #WHT data for coadded image
    wht_data = []
    
    for file in ref_files:
        hdul = fits.open(file)
        for key in list(hdr_update.keys()):
            hdr_update[key].append(fits.getval(file, key, ext = 0))
        filter_name = fits.getval(file, 'FILTER', ext = 0)
        filters.append(filter_name)
        phots.append(fits.getval(file, 'PHOTMJSR', ext = 1))
        # #inverse variance weighting
        wht_data.append(hdul['WHT'].data/fits.getval(file, 'DURATION', ext = 0))
        hdul.close()

    combiner_weights = np.array(wht_data)
    combiner_weights /= np.sum(combiner_weights, axis = 0)
    
    for i, weight in enumerate(combiner_weights):
        invalid = np.isnan(weight) | np.isinf(weight)
        weight[invalid] = 0
        combiner_weights[i] = weight 
    combiner_weights = np.array(combiner_weights)
    
    #coadd images using ccdproc
    ccddata_ = []
    for file in ref_files:
        ccddata_.append(create_ccddata(file))
        
    combiner = Combiner(ccddata_)
    combiner.weights = combiner_weights
    combined_sum = combiner.sum_combine()

    #SCI and ERR data for coadded image
    coadd_data = combined_sum.data
    det_mask = coadd_data == 0
    quad_err = np.sqrt(np.sum([(wht_*ccd.uncertainty.array)**2 for ccd, wht_ in zip(ccddata_, combiner_weights)], axis = 0))
        
    primary_header, sci_header = hdu_template['PRIMARY'].header, hdu_template['SCI'].header
    err_header, wht_header = hdu_template['ERR'].header, hdu_template['WHT'].header
    primary_header['FILENAME'] = filename
    primary_header['FILTER'] = filt.upper()
    
    exptime_wt = [np.nanmean(i) for i in combiner_weights]
    for key in list(hdr_update.keys()):
        hdr_update[key] = np.sum(hdr_update[key])
        primary_header[key] = hdr_update[key]

    sci_header['PHOTMJSR'] = update_photmjsr(ccddata_, phots)
    sci_header['XPOSURE'] = hdr_update['EFFEXPTM']
    sci_header['TELAPSE'] = hdr_update['DURATION']

    primary_hdu = fits.PrimaryHDU(header = primary_header)
    sci_hdu = fits.ImageHDU(data = coadd_data, header = sci_header, name = 'SCI')
    err_hdu = fits.ImageHDU(data = quad_err, header = err_header, name = 'ERR')
    wht_hdu = fits.ImageHDU(data = np.sum(wht_data, axis = 0), header = wht_header, name = 'WHT')
    
    coadd_hdul = fits.HDUList([primary_hdu, sci_hdu, err_hdu, wht_hdu])
    coadd_hdul.writeto(filename, overwrite = True)
    hdu_template.close()

def create_dirs(base_dir: str, n: int = 1) -> dict[int, str]:
    """
    Create ``reference/group_*`` directories under a mosaic base directory.

    When ``base_dir`` is named ``reduction``, also symlink ``../reference`` to
    ``reduction/reference`` for align reference mode.

    Parameters
    ----------
    base_dir : str
        Mosaic reduction root (typically ``.../reduction``).
    n : int, optional
        Number of group directories to create.

    Returns
    -------
    dict
        Group index mapped to ``reference/group_<index>`` path.
    """
    out_dict = dict.fromkeys(range(n))
    os.makedirs(base_dir, exist_ok=True)
    for i in range(n):
        outdir = os.path.join(base_dir, f'reference/group_{i}')
        out_dict[i] = outdir
        os.makedirs(outdir, exist_ok=True)

    # When reducing under <data-root>/reduction, expose reference/ at the
    # dataset root so align --mode reference can find coadds without a manual ln/mkdir.
    ensure_dataset_reference_link(base_dir)

    return out_dict


def ensure_dataset_reference_link(base_dir: str) -> str | None:
    """
    Symlink dataset-root ``reference`` to ``reduction/reference``.

    Parameters
    ----------
    base_dir : str
        Reduction directory; no-op unless its basename is ``reduction``.

    Returns
    -------
    str or None
        Dataset-root reference path when linked or already present, else
        ``None``.
    """
    base = Path(base_dir).resolve()
    if base.name != 'reduction':
        return None
    src = base / 'reference'
    os.makedirs(src, exist_ok=True)
    dst = base.parent / 'reference'
    if dst.exists() or dst.is_symlink():
        return str(dst)
    try:
        os.symlink(src, dst)
    except OSError:
        return None
    return str(dst)

def copy_files(filter_table: dict[str, Table], outdir: str) -> None:
    """
    Copy all science images listed in a filter table dict into ``outdir``.

    Parameters
    ----------
    filter_table : dict
        Filter name mapped to an input-list :class:`~astropy.table.Table`.
    outdir : str
        Destination directory.

    Returns
    -------
    None
    """
    infiles = np.hstack([filter_table[i]['image'].value for i in filter_table.keys()])
    for file in infiles:
        shutil.copy(file, outdir)

def update_path(
    filter_table: dict[str, Table],
    outdir: str,
) -> dict[str, Table]:
    """
    Rewrite ``image`` paths in a filter table dict to basenames under ``outdir``.

    Parameters
    ----------
    filter_table : dict
        Filter name mapped to an input-list :class:`~astropy.table.Table`.
    outdir : str
        Directory containing copied FITS files.

    Returns
    -------
    dict
        Updated filter table (same object, mutated in place).
    """
    for flt in filter_table.keys():
        tbl = filter_table[flt]
        filenames = [os.path.basename(i['image']) for i in tbl]
        tbl['image'] = [os.path.join(outdir, i) for i in filenames]
        filter_table[flt] = tbl
    
    return filter_table

def write_dolphot_frame_list(
    box_outdir: str,
    *,
    refimage: str,
    frames: Sequence[str],
    group: int = 0,
    box: int = 0,
) -> str:
    """
    Write a manifest of coadd + JHAT frames for ``dolphot-prep --from-mosaic``.

    Parameters
    ----------
    box_outdir : str
        Mosaic box directory (``reference/group_G/ref_B``).
    refimage : str
        Path to the coadd ``*_i2d.fits`` reference.
    frames : sequence
        JHAT frame paths belonging to this box.
    group, box : int, optional
        Indices used by ``dolphot-prep`` for ``phot_{group}_{box}``.

    Returns
    -------
    str
        Path to ``dolphot_frames.txt``.
    """
    out = os.path.join(box_outdir, 'dolphot_frames.txt')
    with open(out, 'w') as fh:
        fh.write(f'# group={int(group)} box={int(box)}\n')
        fh.write(f'# ref {os.path.abspath(refimage)}\n')
        for path in frames:
            fh.write(f'{os.path.abspath(path)}\n')
    return out


def edit_spec_groups(
    table: Table,
    spec_group_file: str,
) -> Table:
    """
    Assign a new mosaic group index to files listed in a text manifest.

    Parameters
    ----------
    table : Table
        Input list with ``image`` and ``group`` columns.
    spec_group_file : str
        Text file of basenames to move into group ``max(group)+1``.

    Returns
    -------
    Table
        Updated input list.
    """
    files = np.loadtxt(spec_group_file, dtype=str)
    ngrp = np.max(table['group'])
    basenames = np.array([os.path.basename(i) for i in table['image']])
    for fl in files:
        table['group'][basenames == fl] = ngrp+1

    return table

def assign_gwcs(box_outdir: str, wcs_hdr: fits.Header) -> g_wcs:
    """
    Build a GWCS object for a mosaic box from a FITS WCS header.

    Parameters
    ----------
    box_outdir : str
        Mosaic box directory passed to :func:`create_gwcs`.
    wcs_hdr : fits.Header
        SCI extension WCS header from a coadd product.

    Returns
    -------
    gwcs.wcs.WCS
        GWCS object suitable for JWST datamodel ``meta.wcs``.
    """
    wcsobj = create_gwcs(outdir=box_outdir, sci_header=wcs_hdr, return_gwcs=True)

    return wcsobj


def apply_wcs_to_coadd(coadd_file: str, output: str | None = None) -> str:
    """
    Attach a :func:`create_gwcs` GWCS object to a coadd datamodel.

    By default the coadd is updated in place so mosaic products already
    carry a pipeline-compatible GWCS (no separate ``apply-gwcs`` step).

    Parameters
    ----------
    coadd_file : str
        Path to a coadd ``*_i2d.fits`` product
    output : str or None
        Optional alternate save path; default overwrites ``coadd_file``

    Returns
    -------
    str
        Path to the saved coadd with GWCS attached
    """
    from jwst import datamodels

    out_path = output or coadd_file
    with fits.open(coadd_file) as hdul:
        wcs_hdr = hdul['SCI'].header
    im = datamodels.open(coadd_file)
    wcsobj = assign_gwcs(box_outdir=os.path.dirname(coadd_file), wcs_hdr=wcs_hdr)
    im.meta.wcs = wcsobj
    im.save(out_path)
    return out_path

