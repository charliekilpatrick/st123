"""Shared helpers: coordinates, FITS bookkeeping, visits, and cross-matching.

Merged from the former ``st123.util`` and ``st123.utils`` modules.
"""

import warnings
warnings.filterwarnings('ignore')
import stwcs
import glob
import sys
import os
import shutil
import time
import filecmp
import astroquery
import progressbar
import copy
import requests
import random
import astropy.wcs as wcs
import numpy as np
from contextlib import contextmanager
from astropy import units as u
from astropy.utils.data import clear_download_cache,download_file
from astropy.io import fits
from astropy.table import Table, Column, unique
from astropy.time import Time
from astroscrappy import detect_cosmics
from stwcs import updatewcs
from scipy.interpolate import interp1d
import pandas as pd
from astropy.coordinates import SkyCoord
import shapely

from st123.mast import parse_s_region

def make_banner(message):
    print('\n\n' + message + '\n' + '#' * 80 + '\n' + '#' * 80 + '\n\n')


def is_number(num):
    try:
        float(num)
    except (TypeError, ValueError):
        return False
    return True


def parse_coord(ra, dec):
    if (not (is_number(ra) and is_number(dec)) and
            (':' not in str(ra) and ':' not in str(dec))):
        print(f'ERROR: cannot interpret: {ra} {dec}')
        return None

    if ':' in str(ra) and ':' in str(dec):
        unit = (u.hourangle, u.deg)
    else:
        unit = (u.deg, u.deg)

    try:
        return SkyCoord(ra, dec, frame='icrs', unit=unit)
    except ValueError:
        print(f'ERROR: Cannot parse coordinates: {ra} {dec}')
        return None


acceptable_filters = [
    'F220W','F250W','F330W','F344N','F435W','F475W','F550M','F555W',
    'F606W','F625W','F660N','F660N','F775W','F814W','F850LP','F892N',
    'F098M','F105W','F110W','F125W','F126N','F127M','F128N','F130N','F132N',
    'F139M','F140W','F153M','F160W','F164N','F167N','F200LP','F218W','F225W',
    'F275W','F280N','F300X','F336W','F343N','F350LP','F373N','F390M','F390W',
    'F395N','F410M','F438W','F467M','F469N','F475X','F487N','F547M',
    'F600LP','F621M','F625W','F631N','F645N','F656N','F657N','F658N','F665N',
    'F673N','F680N','F689M','F763M','F845M','F953N','F122M','F160BW','F185W',
    'F218W','F255W','F300W','F375N','F380W','F390N','F437N','F439W','F450W',
    'F569W','F588N','F622W','F631N','F673N','F675W','F702W','F785LP','F791W',
    'F953N','F1042M']

def get_filter(image):
    #check jwst header keywords for filters
    #just 'FILTER' may suffice
    try:
        f = str(fits.getval(image, 'FILTER'))
    except:
        f = str(fits.getval(image, 'FILTER1'))
        if 'clear' in f.lower():
            f = str(fits.getval(image, 'FILTER2'))
    return(f.lower())

def get_module(image):
    #check jwst header keywords for module
    try:
        f = str(fits.getval(image, 'MODULE'))
    except:
        f = str(fits.getval(image, 'DETECTOR'))
        f = f.split('NRC')[1][0]
    return(f.lower())

def get_instrument(image):
    hdu = fits.open(image, mode='readonly')
    inst = hdu[0].header['INSTRUME'].lower()
    # det = hdu[0].header['DETECTOR'].lower()
    # out = f'{inst}_{det}'
    out = f'{inst}'
    return(out)

def get_chip(image):
    # Returns the chip (i.e., 1 for UVIS1, 2 for UVIS2, 1 for WFPC2/PC, 2-4 for
    # WFPC2/WFC 2-4, etc., default=1 for ambiguous images with more than one
    # chip in the hdu)
    
    #jwst does not seem to have chips (nircam at least)
    hdu = fits.open(image)
    chip = None
    for h in hdu:
        if 'CCDCHIP' in h.header.keys():
            if not chip: chip=h.header['CCDCHIP']
            else: chip=1
        elif 'DETECTOR' in h.header.keys():
            if not chip: chip=h.header['DETECTOR']
            else: chip=1

    if not chip: chip=1

    return(chip)

def get_detector_chip(filename):
    '''
    Get the detector chip from the file name

    Parameters
    ----------
    filename : str
        File name

    Returns
    -------
    detector_chip : str
        Detector chip (e.g. ``nrcb1``, ``nrcblong``, ``mirimage``)
    '''
    fl_split = os.path.basename(filename).split('_')
    mask = ['nrc' in x for x in fl_split]
    if any(mask):
        idx = mask.index(True)
        return fl_split[idx]

    # MIRI imager products use the ``mirimage`` detector token.
    mask_miri = ['mirimage' in x.lower() for x in fl_split]
    if any(mask_miri):
        idx = mask_miri.index(True)
        return fl_split[idx]

    return None

def get_zpt(image, ccdchip=1, zptype='abmag'):
    # For a given image and optional ccdchip, determine the photometric zero
    # point in AB mag from PHOTFLAM and PHOTPLAM.
    # ZP_AB = -2.5*np.log10(PHOTFLAM)-5*np.log10(PHOTPLAM)-2.408
    
    #jwst does not seem to have chips
    #nircam zero points: 
    #https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-absolute-flux-calibration-and-zeropoints#gsc.tab=0
    hdu = fits.open(image, mode='readonly')
    inst = get_instrument(image).lower()
    use_hdu = None
    zpt = None

    # Get hdus that contain PHOTPLAM and PHOTFLAM
    sci = []
    for i,h in enumerate(hdu):
        keys = list(h.header.keys())
        if ('PHOTPLAM' in keys and 'PHOTFLAM' in keys):
            sci.append(h)

    if len(sci)==1: use_hdu=sci[0]
    elif len(sci)>1:
        chips = []
        for h in sci:
            if 'acs' in inst or 'wfc3' in inst:
                if 'CCDCHIP' in h.header.keys():
                    if h.header['CCDCHIP']==ccdchip:
                        chips.append(h)
            else:
                if 'DETECTOR' in h.header.keys():
                    if h.header['DETECTOR']==ccdchip:
                        chips.append(h)
        if len(chips)>0: use_hdu = chips[0]

    if use_hdu:
        photplam = float(use_hdu.header['PHOTPLAM'])
        photflam = float(use_hdu.header['PHOTFLAM'])
        if 'ab' in zptype:
            zpt = -2.5*np.log10(photflam)-5*np.log10(photplam)-2.408
        elif 'st' in zptype:
            zpt = -2.5*np.log10(photflam)-21.1

    return(zpt)

# For an input obstable, sort all files into instrument, visit, and filter
# so we can group them together for the final output from dolphot
def add_visit_info(obstable):
    # First add empty 'visit' column to obstable
    obstable['visit'] = [int(0)] * len(obstable)

    # Sort obstable by date so we assign visits in chronological order
    obstable.sort('datetime')

    # Time tolerance for a 'visit' -- how many days apart are obs?
    tol = 1

    # Iterate through each file in the obstable
    for row in obstable:
        inst = row['instrument']
        mjd = Time(row['datetime']).mjd
        filt = row['filter']
        filename = row['image']
        module = row['module']

        # If this is the first one we're making, assign it to visit 1
        if all([obs['visit'] == 0 for obs in obstable]):
            row['visit'] = int(1)
        else:
            instmask = obstable['instrument'] == inst
            timemask = [abs(Time(obs['datetime']).mjd - mjd) < tol
                            for obs in obstable]
            filtmask = [f == filt for f in obstable['filter']]
            nzero = obstable['visit'] != 0 # Ignore unassigned visits
            # mask = [all(l) for l in zip(instmask, timemask, filtmask, nzero)]
            mask = [all(l) for l in zip(instmask, timemask, nzero)]

            # If no matches, then we need to define a new visit
            if not any(mask):
                # no matches. create new visit number = max(visit)+1
                row['visit'] = int(np.max(obstable['visit']) + 1)
            else:
                # otherwise, confirm that all visits of obstable[mask] are same
                if (len(list(set(obstable[mask]['visit']))) != 1):
                    error = 'ERROR: visit numbers are incorrectly assigned.'
                    print(error)
                    return(None)
                else:
                    # visit number is equal to other values in set
                    row['visit'] = list(set(obstable[mask]['visit']))[0]

    return(obstable)

def organize_visit_tables(obstable, byvisit=False):

    tables = []
    if byvisit:
        for visit in list(set(obstable['visit'].data)):
            mask = obstable['visit'] == visit
            tables.append(obstable[mask])
    else:
        tables.append(obstable)

    return(tables)

def organize_reduction_tables(obstable, byvisit=False, bymodule=False):

    tables = []
    if bymodule:
        for mod in list(set(obstable['module'].data)):
            mask = obstable['module'] == mod
            tables.append(organize_visit_tables(obstable[mask], byvisit=byvisit))
    else:
        tables.append(organize_visit_tables(obstable, byvisit=byvisit))

    return(tables)

def pick_deepest_images(images, reffilter=None, avoid_wfpc2=False, refinst=None):
    # Best possible filter for a dolphot reference image in the approximate
    # order I would want to use for a reference image.  You can also use
    # to force the script to pick a reference image from a specific filter.
    best_filters = ['f606w','f555w','f814w','f350lp','f110w','f105w',
        'f336w']

    # If we gave an input filter for reference, override best_filters
    if reffilter:
        if reffilter.upper() in acceptable_filters:
            # Automatically set the best filter to only this value
            best_filters = [reffilter.lower()]

    # Best filter suffixes in the approximate order we would want to use to
    # generate a template.
    best_types = ['lp', 'w', 'x', 'm', 'n']
    
    # First group images together by filter/instrument
    filts = [get_filter(im) for im in images]
    insts = [get_instrument(im).replace('_full','').replace('_sub','')
        for im in images]

    if refinst:
        mask = [refinst.lower() in i for i in insts]
        if any(mask):
            filts = list(np.array(filts)[mask])
            insts = list(np.array(insts)[mask])

    # Group images together by unique instrument/filter pairs and then
    # calculate the total exposure time for all pairs.
    unique_filter_inst = list(set(['{}_{}'.format(a_, b_)
        for a_, b_ in zip(filts, insts)]))

    # Don't construct reference image from acs/hrc if avoidable
    if any(['hrc' not in val for val in unique_filter_inst]):
        # remove all elements with hrc
        new = [val for val in unique_filter_inst if 'hrc' not in val]
        unique_filter_inst = new

    # Do same for WFPC2 if avoid_wfpc2=True
    if avoid_wfpc2:
        if any(['wfpc2' not in val for val in unique_filter_inst]):
            # remove elements with WFPC2
            new = [val for val in unique_filter_inst if 'wfpc2' not in val]
            unique_filter_inst = new

    total_exposure = []
    for val in unique_filter_inst:
        exposure = 0
        for im in images:
            if (get_filter(im) in val and
                get_instrument(im).split('_')[0] in val):
                exposure += fits.getval(im,'EFFEXPTM')
        total_exposure.append(exposure)

    best_filt_inst = ''
    best_exposure = 0
    
    # First type to generate a reference image from the 'best' filters.
    for filt in best_filters:
        if any(filt in s for s in unique_filter_inst):
            vals = filter(lambda x: filt in x, unique_filter_inst)
            for v in vals:
                exposure = total_exposure[unique_filter_inst.index(v)]
                if exposure > best_exposure:
                    best_filt_inst = v
                    best_exposure = exposure

    # Now try to generate a reference image for types in best_types.
    for filt_type in best_types:
        if not best_filt_inst:
            if any(filt_type in s for s in unique_filter_inst):
                vals = filter(lambda x: filt_type in x, unique_filter_inst)
                for v in vals:
                    exposure = total_exposure[unique_filter_inst.index(v)]
                    if exposure > best_exposure:
                        best_filt_inst = v
                        best_exposure = exposure

    # Now get list of images with best_filt_inst.
    reference_images = []
    for im in images:
        filt = get_filter(im)
        inst = get_instrument(im).replace('_full','').replace('_sub','')
        if (filt+'_'+inst == best_filt_inst):
            reference_images.append(im)

    return(reference_images)

def get_sky_pgons(table):
    pgons = []
    for im in table['image']:
        region = fits.open(im)['SCI'].header['S_REGION']
        pgons.append(parse_s_region(region))
    return np.array(pgons, dtype=object)

def edit_visits_groups(table):
    unique_visits = np.unique(table['visit'])
    visit_mapping = {old: new for new, old in enumerate(unique_visits)}
    table['visit'] = [visit_mapping[v] for v in table['visit']]

    table.add_column(Column(name='group', data=[None]*len(table)))
    pgons = get_sky_pgons(table)

    net_field = shapely.unary_union(pgons)
    if type(net_field) == shapely.geometry.polygon.Polygon:
        net_field = shapely.MultiPolygon([net_field])

    for i, component in enumerate(net_field.geoms):
        int_area = np.array([poly.intersection(component).area/poly.area for poly in pgons])
        mask = int_area > 0
        table['group'][mask] = i

    for visit in np.unique(table['visit']):
        groups = np.unique(table[table['visit'] == visit]['group'])
        if len(groups) > 1:
            table['group'][table['visit'] == visit] = min(groups)

    unique_groups = np.unique(table['group'])
    group_mapping = {old: new for new, old in enumerate(unique_groups)}
    table['group'] = [group_mapping[g] for g in table['group']]
    table.sort(['visit'])

    return table

def input_list(input_images):
    img = input_images
    zptype = 'abmag'
    good = []
    image_number = []
    #check if image exists (check in archive otherwise)
    for image in img:
        success = True
        if not os.path.exists(image):
            success = False
        if success:
            good.append(image)
    img = copy.copy(good)

    hdu = fits.open(img[0])
    h = hdu[0].header

    exp = [fits.getval(image,'EFFEXPTM') for image in img] #exposure time
    if 'DATE-OBS' in h.keys() and 'TIME-OBS' in h.keys(): 
        dat = [fits.getval(image,'DATE-OBS') + 'T' +
               fits.getval(image,'TIME-OBS') for image in img] #datetime
    elif 'EXPSTART' in h.keys():
        dat = [Time(fits.getval(image, 'EXPSTART'),
            format='mjd').datetime.strftime('%Y-%m-%dT%H:%M:%S') #datetime if DATE-OBS is missing
            for image in img]

    fil = [get_filter(image) for image in img]
    ins = [get_instrument(image) for image in img]
    # det = ['_'.join(get_instrument(image).split('_')[:2]) for image in img]
    module = [get_module(image) for image in img]
    chip= [get_chip(image) for image in img]
    zpt = [get_zpt(i, ccdchip=c, zptype=zptype) for i,c in zip(img,chip)]
    visit = [fits.getval(i, 'VISIT_ID', ext=0) for i in img]
    pupil = [fits.getval(i, 'PUPIL', ext=0) for i in img]

    if not image_number:
        image_number = [0 for image in img]

    obstable = Table([img,exp,dat,fil,ins,module,zpt,chip,image_number, visit, pupil],
        names=['image','exptime','datetime','filter','instrument', 'module',
         'zeropoint','chip','imagenumber', 'visit', 'pupil'])
    obstable.sort('datetime')
    obstable = edit_visits_groups(obstable)
        
    return obstable

def create_filter_table(tables, filters):
    '''
    Create a dictionary of (filter, table) pairs for
    the full observation set

    This is useful to collect all observations in a 
    patricular filter when the observation is a mix

    Parameters
    ----------
    tables : list
        List of tables
    filters : list
        Filter of each table

    Returns
    -------
    filter_table : dict
        Dictionary of (filter, table) pairs
    '''
    filter_table = dict.fromkeys(filters)
    for flt in filter_table.keys():
        # flt_tables = [tbl_ for tbl_ in tables if tbl_['filter'][0] == flt]
        filter_table[flt] = tables[tables['filter'] == flt] #vstack(flt_tables)

    return filter_table

def xmatch_common(skycrd_1, skycrd_2, dist_limit = 5.0):
    """
    crossmatch sources between two SkyCoord objects
    
    Parameters
    ------------
    
    skycrd_1 : SkyCoord
        positions of sources in catalog 1 as a SkyCoord object
    skycrd_2 : SkyCoord
        positions of sources in catalog 2 as a SkyCoord object
    dist_limit : float
        maximum distance for a match, in arcsec
        
    Returns
    ------------
    
    dist_matched_df : pd.DataFrame
        dataframe containing three columns with indices in catalog 1, 
        indices in catalog 2 and distances in arcsec respectively
    """
    idx, d2d, d3d = skycrd_1.match_to_catalog_sky(skycrd_2)
    # Store plain floats (arcsec) so pandas comparisons stay dimensionless.
    d = {
        'idx_1': np.arange(0, len(skycrd_1)),
        'idx_2': idx,
        'd2d': d2d.to(u.arcsec).value,
    }
    xmatch_df = pd.DataFrame(data=d)
    matched_df = xmatch_df.loc[xmatch_df.groupby('idx_2').d2d.idxmin()]
    dist_matched_df = matched_df[matched_df['d2d'] < dist_limit]

    return dist_matched_df