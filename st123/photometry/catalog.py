import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import glob, os
import itertools

def get_filters(columns):
    """
    get unique filters and lines from dolphot column files

    Parameters
    ----------
    columns : str
        path to dolphot column file

    Returns
    -------
    lines : list
        list of lines from dolphot column file
    filters : list
        list of unique filters
    """
    with open(columns, 'r') as f:
        lines = f.readlines()
    lines = np.array(lines)

    filters = []
    for line in lines:
        if 'Normalized count rate, NIRCAM' in line:
            filters.append(line.split('NIRCAM_')[-1].split('\n')[0])

    return lines, filters

def map_columns(columns):
    """
    generate dictionaries mapping column names to column indices from dolphot column files

    Parameters
    ----------
    columns : str
        path to dolphot column file

    Returns
    -------
    col_dict : dict
        dictionary mapping column names to column indices for combined photometry
    filter_cols : dict
        dictionary mapping filter names to counts, error and flag column indices for
        individual images
    """
    #combined photometry columns
    column_strings = ['Object X position', 'Object Y position', 'Signal-to-noise',
                      'Object sharpness', 'Crowding', 'Object type']
    column_key = ['X', 'Y', 'SNR', 'Sharpness', 'Crowding', 'Type']

    #get unique filters and lines from column file
    lines, filters = get_filters(columns)

    #define keys and strings for instrumental magnitudes and uncertainties
    mag_strings = [f'Instrumental VEGAMAG magnitude, NIRCAM_{filt}' for filt in filters]
    magerr_strings = [f'Magnitude uncertainty, NIRCAM_{filt}' for filt in filters]
    mag_key, magerr_key = [i + '_mag' for i in filters], [i + '_err' for i in filters]

    keys = column_key+mag_key+magerr_key
    strings = column_strings+mag_strings+magerr_strings

    col_dict = {key: [] for key in keys}

    #get column indices for combined photometry
    for key, string in zip(keys, strings):
        col = lines[np.char.find(lines, string) > 0]
        if len(col) > 0:
            col_dict[key] = int(col[0].split('.')[0]) - 1
        else:
            col_dict[key] = None

    #get column indices for individual images
    filter_cols = {key: [] for key in filters}
    for filter_name in filters:
        phot_keys = ['Counts', 'Err', 'Flag']
        per_filter_cols = {key: [] for key in phot_keys}
        idx_cts, idx_err, idx_flag = [], [], []
        for line in lines:
            #get column indices for counts, errors and flags
            #count uncertainty is used to get the index since 'Normalized count rate' is not unique
            if (filter_name in line) & ('Normalized count rate uncertainty' in line):
                idx = int(line.split(' ')[0].split('.')[0]) - 1
                idx_cts.append(str(idx - 1))
                idx_err.append(str(idx))
                idx_flag.append(str(idx + 9))
        #first index corresponds to combined photometry
        per_filter_cols['Counts'] = idx_cts[1:]
        per_filter_cols['Err'] = idx_err[1:]
        per_filter_cols['Flag'] = idx_flag[1:]
        filter_cols[filter_name] = per_filter_cols

    return col_dict, filters, filter_cols

def save_photfiles(photfile_path, outdir, obj, chunksize = 100000):
    """
    save photometry files with cuts applied to smaller csv files

    Parameters
    ----------
    photfile_path : str
        path to directory containing dolphot photometry files
    outdir : str
        path to directory to save csv files
    obj : str
        object name
    chunksize : int
        number of rows to read from photometry file at a time

    Returns
    -------
    None
    """
    photfiles = sorted(glob.glob(os.path.join(photfile_path, '*phot')))[:1]
    for i, photfile in enumerate(photfiles):
        column_file = photfile + '.columns'
        #map columns to indices
        col_idx, filters, _ = map_columns(column_file)

        #read photometry file in chunks
        photdf = pd.read_csv(photfile, sep = '\s+', memory_map = True,
                             header = None, iterator = True, chunksize = chunksize)

        for j, chunk in tqdm(enumerate(photdf)):
            #apply cuts
            cuts = (chunk[col_idx['SNR']] >= 10) & \
                    ((chunk[col_idx['Sharpness']])**2 <= 0.01) & \
                    (chunk[col_idx['Crowding']] <= 0.5) & \
                    (chunk[col_idx['Type']] <=2)

            #generate index by combining x and y positions
            #the x, y positions are rounded to the floor of the value since dolphot output
            #does not use the same star center from different runs
            df_idx = ["{:.2f}_{:.2f}".format(x, y) for x, y in zip(np.floor(np.array(chunk[2])), np.floor(np.array(chunk[3])))]
            chunk['idx'] = df_idx
            chunk.set_index('idx', inplace = True)
            chunk[cuts].to_csv(f'{outdir}/{obj}_{i}_{j}.csv', mode = 'a', header = False)

def create_common_catalog(common_ids, dfs, columns, outfile):
    """
    save combined photometry for common sources to a csv file

    Parameters
    ----------
    common_ids : list
        list of unique source indices shared across catalogs
    dfs : list
        list of dataframes containing photometry
    columns : list
        list of column files

    Returns
    -------
    None
    """
    combined_data = []
    all_filters = ['F115W', 'F150W', 'F200W', 'F277W', 'F360M']

    for source_id in tqdm(common_ids):
        phot_row = []
        positions = []
        for filt in all_filters:
            invvar_sum, weight_sum = 0, 0
            for df, col in zip(dfs, columns):
                if source_id not in df.index:
                    continue
                col_dict, filters, filter_cols = map_columns(col)
                positions.append([df[str(col_dict['X'])].loc[source_id], df[str(col_dict['Y'])].loc[source_id]])
                if filt not in filters:
                    continue
                #weighted average of magnitudes
                counts = np.array(df[filter_cols[filt]['Counts']].loc[source_id])
                err = np.array(df[filter_cols[filt]['Err']].loc[source_id])
                flag = np.array(df[filter_cols[filt]['Flag']].loc[source_id])
                good = (flag < 8) & (counts > 0)
                good_counts, good_err = counts[good], err[good]
                invvar_sum += np.sum(good_counts / good_err**2)
                weight_sum += np.sum(1 / good_err**2)

            if invvar_sum > 0:
                mag = -2.5*np.log10(invvar_sum/weight_sum)
                #dolphot calculates magnitude uncertainties as below
                magerr = 1.0857362*(1/np.sqrt(weight_sum))/(invvar_sum/weight_sum)
            else:
                mag, magerr = 99.99, 99.99
            phot_row.extend([mag, magerr])
        phot_row = [np.mean(np.array(positions)[:, 0]), np.mean(np.array(positions)[:, 1])] + phot_row
        combined_data.append(phot_row)

    combined_data = np.array(combined_data)
    #save combined photometry to csv
    df_col = ['X', 'Y'] + [[f'{filt}_mag', f'{filt}_err'] for filt in all_filters]
    df_col = list(itertools.chain(*df_col))
    df = pd.DataFrame(combined_data, columns = df_col)
    df.to_csv(outfile, index = False)
