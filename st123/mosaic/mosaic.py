import glob, os
from st123.utils.helpers import create_filter_table, input_list
import numpy as np
from multiprocessing import Pool
from jwst.pipeline import calwebb_image3

import glob,os
from astropy.io import fits
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
import shapely
import shapely.ops
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import shutil
from pathlib import Path

from astropy.modeling import models
from astropy import coordinates as coord
from astropy import units as u
from astropy.table import Column
from gwcs import coordinate_frames as cf
from astropy import wcs
from gwcs import WCS as g_wcs
from asdf import AsdfFile

from astropy.nddata import CCDData
from astropy import nddata
from ccdproc import Combiner

import stpsf
from astropy.stats import sigma_clipped_stats as scs
#from photutils.psf.matching import resize_psf, SplitCosineBellWindow, create_matching_kernel, CosineBellWindow, TukeyWindow, TopHatWindow, HanningWindow
from photutils.psf.matching import SplitCosineBellWindow, create_matching_kernel, CosineBellWindow, TukeyWindow, TopHatWindow, HanningWindow
from astropy.convolution import convolve, convolve_fft
from reproject.mosaicking import find_optimal_celestial_wcs
import subprocess
from st123.utils.helpers import get_detector_chip
from st123.mast import parse_s_region
from st123.utils.settings import *  # noqa: F403


def mp_init(init_success: int = 0,
            init_failed: int = 0,
            init_success_files: list =[]):
    global success
    global failed
    global success_files
    success = init_success
    failed = init_failed
    success_files = init_success_files

class split_observations(object):
    def __init__(self, table, N_max=150, min_overlap=0.2, pad=15, polygons=None, wcs_opt=None):

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
                    print(f"WARNING: Cannot split further, minmimum group size is {mask.sum()}")
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
        
        
def get_pgons(table):
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
        
def create_default_mosaic(inputfiles, outdir, filt):
    '''
    Create level3 drizzled mosaic from level2 input files
    using the JWST pipeline with default options

    Parameters
    ----------
    input_files : list
        List of input level2 files
    outdir: str
        Output directory

    Returns
    -------
    None
    '''
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

    image3.run(asn_file)

def create_coadd_mosaic(table, outdir, filt, centroid=None, 
                        output_shape=None, gwcs_file=None):
    '''
    Create level3 drizzled mosaic from level2 input files
    using the JWST pipeline. Centroid and output_shape or gwcs_file
    can be passed in to create uniformly drizzled mosaics

    Parameters
    ----------
    table : astropy.table.table.Table
        Table containing all images to be resampled
    outdir: str
        Output directory
    filt: str
        Filter of images to be resampled
    centroid: optional, shapely.geometry.point.Point
        Centroid of the drizzled field
    output_shape: optional, tuple
        (xbound, ybound) for the drizzled field
    gwcs_file: optional, str
        Path to gwcs file to drizzle uniformly

    Returns
    -------
    filepath: str
        Path to resampled i2d image
    '''
    if not os.path.exists(outdir):
        os.makedirs(outdir)

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
    if gwcs_file:
        image3.resample.output_wcs = gwcs_file

    elif centroid:
        image3.resample.crval = (centroid.x , centroid.y)
        image3.resample.output_shape = output_shape[0], output_shape[1]

    image3.run(asn_file)
    
    filepath = f'{outdir}/out_{filt}/{filt}_i2d.fits'
    return filepath


def create_gwcs(outdir, sci_header=None, wcs_out=None, shape_out=None, return_gwcs=False):
    '''
    Convert astropy WCS to a GWCS object and write it into an asdf file

    Parameters
    ----------
    wcs_out : astropy.wcs.WCS
        Astropy WCS to be converted to gwcs
    shape_out : tuple
        (NAXIS2, NAXIS1) shape of output mosaic

    Returns
    -------
    gwcs_path: str
        Path to the GWCS asdf file
    '''

    if sci_header:
        pass
    elif wcs_out:
        sci_header = wcs_out.to_header()
        sci_header['NAXIS1'] = shape_out[1]
        sci_header['NAXIS2'] = shape_out[0]
    else:
        raise ValueError("Please provide header or wcs object")

    shift_by_crpix = models.Shift(-(sci_header['CRPIX1'] - 1)) & models.Shift(-(sci_header['CRPIX2'] - 1))
    matrix = np.array([[sci_header['PC1_1'], sci_header['PC1_2']],
                    [sci_header['PC2_1'] , sci_header['PC2_2']]])
    rotation = models.AffineTransformation2D(matrix , translation=[0, 0])

    tan = models.Pix2Sky_TAN()
    pixelscale = models.Scale(sci_header['CDELT1']) & models.Scale(sci_header['CDELT2'])
    celestial_rotation =  models.RotateNative2Celestial(sci_header['CRVAL1'], sci_header['CRVAL2'], 180)

    det2sky = shift_by_crpix | rotation | pixelscale | tan | celestial_rotation
    det2sky.name = "linear_transform"

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

def find_optimal_wcs(filter_table):
    images = np.hstack([filter_table[i]['image'].value for i in filter_table.keys()])
    image_hdus = [fits.open(i)[1] for i in images]
    wcs_out, shape_out = find_optimal_celestial_wcs(image_hdus, auto_rotate = True)

    return wcs_out, shape_out

def create_psf_kernel(ref_filter: str, in_filter: str, ovs=5, fov=81):
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

def convolve_images(filter_table, target_filter):
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

def create_ccddata(file):
    hdu = fits.open(file)
    sci_data = hdu['SCI'].data
    
    uncertainty = nddata.StdDevUncertainty(array = hdu['ERR'].data)
    data_unit = u.MJy/u.sr
    w = wcs.WCS(hdu['SCI'].header)
    mask = sci_data == 0
    ccd_data = CCDData(data = sci_data, uncertainty = uncertainty, 
                       wcs = w, unit = data_unit)
    
    return ccd_data

def update_photmjsr(ccddata, phots):
    ccd_mjsr = np.sum([ccd.data for ccd in ccddata], axis = 0)
    ccd_cps = np.sum([ccd.data/phot for ccd, phot in list(zip(ccddata, phots))], axis = 0)
    mjsr = ccd_mjsr/ccd_cps
    _, mjsr_med, _ = scs(mjsr)

    return mjsr_med

def coadd(ref_files, filt, filename = 'coadd_i2d.fits'):
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

def create_dirs(base_dir, n=1):
    out_dict = dict.fromkeys(range(n))
    for i in range(n):
        outdir = os.path.join(base_dir, f'reference/group_{i}')
        out_dict[i] = outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    
    return out_dict

def copy_files(filter_table, outdir):
    infiles = np.hstack([filter_table[i]['image'].value for i in filter_table.keys()])
    for file in infiles:
        shutil.copy(file, outdir)

def update_path(filter_table, outdir):
    for flt in filter_table.keys():
        tbl = filter_table[flt]
        filenames = [os.path.basename(i['image']) for i in tbl]
        tbl['image'] = [os.path.join(outdir, i) for i in filenames]
        filter_table[flt] = tbl
    
    return filter_table

def setup_paramfile(phot_outdir, refimage, files):
    '''
    Generate the dolphot parameter file

    Parameters
    ----------
    basedir : str
        Base directory to search for files
    files : list
        List of files to be included in the paramfile

    Returns
    -------
    None
    '''
    
    shutil.copy(refimage, phot_outdir)
    for src_file in files:
        shutil.copy(src_file, phot_outdir)

    phot_image_base = [os.path.basename(r).replace('.fits', '') for r in files]
    phot_image_det = ['long' if 'long' in get_detector_chip(r) else 'short' for r in files]
    N_img = len(files)

    with open(f'{phot_outdir}/dolphot.param', 'w') as f:
        f.write('Nimg = {}\n'.format(N_img))
        f.write('img0_file = {}\n'.format(os.path.basename(refimage).replace('.fits', '')))
        for i, (img, det) in enumerate(zip(phot_image_base, phot_image_det)):
            f.write('img{}_file = {}\n'.format(i+1, img))
            if det == 'short':
                for key, val in short_params.items():
                    f.write('img{}_{} = {}\n'.format(i+1, key, val))
            if det == 'long':
                for key, val in long_params.items():
                    f.write('img{}_{} = {}\n'.format(i+1, key, val))
        for key, val in base_params.items():
            f.write('{} = {}\n'.format(key, val))

def apply_nircammask(files):
    '''
    Apply nircammask from dolphot to input files to mask pixels in 
    images using DQ mask
    
    Parameters
    ----------
    files : list
        List of files to apply nircammask to
        
    Returns
    -------
    None
    '''
    cmd = ['nircammask', '-etctime']
    for fl in files:
        # cmd += f' {fl}'
        cmd.append(fl)

    subprocess.run(cmd)  

def calc_sky(files):
    '''
    Calculate the sky for input files using calcsky in dolphot

    Parameters
    ----------
    files : list
        List of files to calculate the sky for

    Returns
    -------
    None
    '''
    cmd = 'calcsky {fits_base} {rin} {rout} {step} {sigma_low} {sigma_high}'
    for fl in files:
        #read params from options later
        fits_base = fl.replace('.fits','')
        print(fits_base)
        rin = 15
        rout = 25
        step = -64
        sigma_low = 2.25
        sigma_high = 2.00
        cmd_fl = cmd.format(fits_base=fits_base,rin=rin,rout=rout,
                            step=step,sigma_low=sigma_low,sigma_high=sigma_high)
        subprocess.run(cmd_fl,shell=True)

def edit_spec_groups(table, spec_group_file):
    files = np.loadtxt(spec_group_file, dtype=str)
    ngrp = np.max(table['group'])
    basenames = np.array([os.path.basename(i) for i in table['image']])
    for fl in files:
        table['group'][basenames == fl] = ngrp+1

    return table

def assign_gwcs(box_outdir, wcs_hdr):
    wcsobj = create_gwcs(outdir=box_outdir, sci_header=wcs_hdr, return_gwcs=True)

    return wcsobj


def apply_wcs_to_coadd(coadd_file):
    """Attach a GWCS object to a coadd datamodel and save a corrected file."""
    from jwst import datamodels

    new_file = coadd_file.replace('coadd_', 'coadd_corrected_')
    with fits.open(coadd_file) as hdul:
        wcs_hdr = hdul['SCI'].header
    im = datamodels.open(coadd_file)
    wcsobj = assign_gwcs(box_outdir=os.path.dirname(coadd_file), wcs_hdr=wcs_hdr)
    im.meta.wcs = wcsobj
    im.save(new_file)
    return new_file

