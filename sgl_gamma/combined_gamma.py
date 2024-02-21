import numpy as np
import pandas as pd
import os
from astropy.table import Table
from utils_plot import plot_point_with_fit
import warnings
warnings.filterwarnings("ignore")
import io

def weighted_mean(values, errors):

    weights = 1 / errors**2  # Assuming errors are standard deviations

    return np.average(values, weights=weights)

def uncertainty(values, errors):

    mean_value = np.mean(values)
    weights = 1 / errors**2  # Assuming errors are standard deviations
    
    return np.sqrt(np.sum(weights * (values - mean_value)**2) / np.sum(weights)) / mean_value


def load_tabs(path_ann,path_gp ):
   
    tab_ann = Table.read(path_ann, format='csv')
    tab_gp = Table.read(path_gp, format='csv')

    return tab_ann, tab_gp

def wmean_and_error(path_ann, path_gp, column='dd'):
    tab_ann, tab_gp = load_tabs(path_ann, path_gp)

    wmean_list = []
    wmean_error_list = []

    # Get all the indices, gp has fewer indices so use that condition 
    indices = set(tab_gp[tab_gp['dd_GP'] > 0]['index'])

    if len(indices) != len(tab_gp[tab_gp['dd_GP'] > 0]):
        raise Exception('Some entries have duplicate indices')

    for index in indices:
        ann_subset = tab_ann[tab_ann['index'] == index]
        gp_subset = tab_gp[tab_gp['index'] == index]
        
        values = np.concatenate([gp_subset[f'{column}_GP'].data, ann_subset[f'{column}_ANN'].data])
        errors = np.concatenate([gp_subset[f'{column}_error_GP'].data, ann_subset[f'{column}_error_ANN'].data])

        # Assuming weighted_mean and uncertainty functions work with lists of values and errors
        mean = weighted_mean(values, errors)
        uncertainty_value = uncertainty(values, errors)

        wmean_list.append(mean)
        wmean_error_list.append(uncertainty_value)

    return wmean_list, wmean_error_list

def combined_dd(path_table_GP, path_table_ANN, output_folder, return_table_path=True):

    if not os.path.exists(output_folder):
        print(f'making dir {output_folder}')
        os.makedirs(output_folder, exist_ok=True)

    table_GP = Table.read(path_table_GP, format='csv')
    
    table_GP = table_GP[table_GP['dd_GP'] > 0]

    wmean_list, wmean_error_list = wmean_and_error(path_table_ANN, path_table_GP)
    
    table_GP['dd_wmean'] = np.array(wmean_list)
    table_GP['dd_error_wmean'] = np.array(wmean_error_list)


    table_GP.write(os.path.join(output_folder, 'SGLTable_combined_ANNGP.csv'), overwrite = True,format='csv')
     
    if return_table_path:
        return os.path.join(output_folder, 'SGLTable_combined_ANNGP.csv')


def lenstable_cut_by_z(lens_table_path, 
                       cc_file ='Hz-35.txt', 
                       return_tab=False,
                        return_tab_path = False ):

    # Get the path of the current script
    script_path = os.path.abspath(__file__)
    # Get the parent directory of the script
    path_project =  os.path.dirname(os.path.dirname(script_path))

    format = lens_table_path.split('.')[-1]
    lens_table = Table.read(lens_table_path, format=format)
    cc_path = os.path.join(path_project, 'Data' ,cc_file )

    n = cc_file.split('-')[1][:2]
    
    # Open file with the correct encoding
    with io.open(os.path.join(path_project, cc_path), 'r', encoding='utf-8') as f:
        Z, _, _ = np.loadtxt(f, unpack=True)
        
    z_max = np.max(Z)

    output_path = os.path.join(path_project, 'Data', f'SGLTable-{n}.{format}')

    lens_table[lens_table['zs']<=z_max].write(output_path, 
                                              format =format, overwrite = True)
    
    if return_tab:
        return lens_table
    if return_tab_path:
        return output_path