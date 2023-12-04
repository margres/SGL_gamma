from astropy.table import Table
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from getdist import plots, MCSamples
from rec_gamma_mcmc import MCMC
from mcmc_utils import lnprobfit
 
# Calculate the best-fit (mean/median) values and uncertainties
def get_param_stats(samples, param_index):
    """
    Calculate the median and the 68% confidence interval for a given parameter from all walkers.

    Parameters:
    samples (numpy.ndarray): MCMC samples array with shape (walkers, steps, parameters).
    param_index (int): The index of the parameter in the samples array.

    Returns:
    tuple: Median, lower bound, and upper bound of the 68% confidence interval.
    """
    # Flatten the first two dimensions (walkers, steps)
    flat_samples = samples.reshape(-1, samples.shape[-1])

    median = np.median(flat_samples[:, param_index])
    err_lower = median-np.percentile(flat_samples[:, param_index], 16)
    err_upper = np.percentile(flat_samples[:, param_index], 84)-median
    return median, err_lower, err_upper

def plot_directfit(samples, output_folder=None):

    # If you have parameter names and want to include them in the plot
    param_names = [r'$\gamma_0$', r'$\gamma_1$']
    param_labels = ['\gamma_0', '\gamma_1']

    median_list = [] # median, err_lower, err_upper
    for i in range(samples.shape[-1]):  # Loop over parameters
        median_list.append(get_param_stats(samples, i))



    # Convert your samples to MCSamples object
    mcmc_samples = MCSamples(samples=samples, names=param_names, labels=param_labels)

    # Initialize the GetDist plotter
    g = plots.getSubplotPlotter(width_inch=10)
    g.settings.title_limit_fontsize = 15
    g.settings.axes_fontsize = 15
    g.settings.axes_labelsize = 15

    # Triangle plot with filled contours
    g.triangle_plot(mcmc_samples, param_names, filled=True, title_limit=1)
    
    for ax1 in g.subplots[:,0]:
        ax1.axvline(median_list[0][0], color='red', ls='--')

    ax2 = g.subplots[1, 1]
    ax2.axvline(median_list[1][0], color='red', ls='--')

    ax2 = g.subplots[1, 0]
    ax2.axhline(median_list[1][0], color='red', ls='--')

    if output_folder is not None:
        # Optionally, save the plot
        plt.savefig(os.path.join(output_folder , "gamma_evo_CCGP.png"))

    # Show the plot
    plt.show(block=False)
 


def main(lens_table_path , path_project ,
         show_plot=True, model='GP', lnprob_touse=lnprobfit, 
         ndim=2, x_ini=[2.0, 0] ):
    
    #if lens_table_path is None:
    #    lens_table_path = os.path.join(path_project,'Data','Combined_table.fits')

    output_folder = os.path.join(path_project, 'Output', 'Gamma_DirectFit' )

    if not os.path.exists(output_folder):
        print(f'making dir {output_folder}')
        os.makedirs(output_folder, exist_ok=True)


    mcmc = MCMC(lens_table_path, path_project, model=model, output_folder=output_folder, 
                lnprob_touse=lnprob_touse, ndim=ndim, x_ini = x_ini)

    try:
        all_samples, _ =  mcmc.load_samples_and_ln_probs()
    except FileNotFoundError:
        mcmc.run_mcmc() 
    
    all_samples, _ =  mcmc.load_samples_and_ln_probs()
    all_samples = np.squeeze(all_samples)
    #
    # print(np.shape(all_samples))
    
    if show_plot:
        plot_directfit(all_samples, output_folder)


if __name__ == "__main__":

    path_project = '/home/grespanm/github/SLcosmological_parameters/sgl_gamma'
    lens_table_path = os.path.join(path_project, 'Data' , 'LensTable02.fits')
    main(lens_table_path , path_project)