import numpy as np
import pandas as pd
import os
from astropy.table import Table
from utils_plot import plot_point_with_fit

def weighted_mean(values, errors):

    weights = 1 / errors**2  # Assuming errors are standard deviations

    return np.average(values, weights=weights)

def uncertainty(values, errors):

    mean_value = values.mean()
    weights = 1 / errors**2  # Assuming errors are standard deviations
    
    return np.sqrt(np.sum(weights * (values - mean_value)**2) / np.sum(weights)) / mean_value


def load_tabs(path_ann,path_gp ):
   
    tab_ann = Table.read(path_ann)
    tab_gp = Table.read(path_gp)

    return tab_ann, tab_gp

def wmean_and_error(path_ann, path_gp, column = 'dd'):

    tab_ann, tab_gp = load_tabs(path_ann, path_gp)

    wmean_list = []
    wmean_error_list = []

    #get all the lens names, gp has fewer lenses so use that condition 
    lens_names = set(tab_gp[tab_gp['dd_GP'] > 0]['lensName'])

    if len(lens_names) != len(tab_gp[tab_gp['dd_GP'] > 0]):
        raise Exception('Some lenses have duplicate names')

    for ln in lens_names:
        ann_subset = tab_ann[tab_ann['lensName'] == ln]
        gp_subset = tab_gp[tab_gp['lensName'] == ln]

        values = np.array([gp_subset[f'{column}_GP'], ann_subset[f'{column}_ANN']])
        errors = np.array([gp_subset[f'{column}_error_GP'], ann_subset[f'{column}_error_ANN']])

        # Assuming weighted_mean and uncertainty functions work with lists of values and errors
        mean = weighted_mean(values, errors)
        uncertainty_value = uncertainty(values, errors)

        wmean_list.append(mean)
        wmean_error_list.append(uncertainty_value)

    return wmean_list, wmean_error_list

if __name__ == "__main__":

    output_path = '/home/grespanm/github/SLcosmological_parameters/SGL_gamma/Output'
    path_ann = os.path.join(output_path, 'ANN', 'SGLTable_ANN.fits')
    path_gp = os.path.join(output_path, 'GP', 'SGLTable_GP.fits')
    
    wmean_list, wmean_error_list = wmean_and_error(path_ann, path_gp)
    tab_gp = Table.read(path_gp)
    x = tab_gp[tab_gp['dd_GP'] > 0]['zl']
    plot_point_with_fit(x,wmean_list, wmean_error_list )