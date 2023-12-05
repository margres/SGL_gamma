from scipy.special import gamma
import numpy as np
from astropy.constants import c
from scipy.integrate import quad, trapz
from scipy.stats import median_abs_deviation
from getdist import plots, MCSamples
import matplotlib.pyplot as plt
import os
c = c.to('km/s').value

def f_gama(x):
        a = -1 / np.sqrt(np.pi)
        b = ((5 - 2 * x) * (1 - x)) / (3 - x)
        c = gamma(x - 1) / gamma(x - (3 / 2))
        d = gamma((x - 1) / 2) / gamma(x / 2)

        return  a * b * c * np.square(d)

def lnprior(x):
    
    if 1.51<x<2.49:
        return 0.0
    return -np.inf

def d_th(x,theta, theta_ap, sigma):

    a = (theta*c**2) / (4 * np.pi * sigma**2)
    
    return a* np.power((theta / theta_ap), x-2) / f_gama(x)


def solve_for_gamma(x, theta, theta_ap, sigma, dd):
    return dd - d_th(x, theta, theta_ap, sigma)

def chi_2(x, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
    num = solve_for_gamma(x, theta, theta_ap, sigma, dd)**2
    denom_th = 4 * (abs_delta_sigma_ap / sigma)**2 + ((1 - x) * 0.05)**2 * d_th(x, theta, theta_ap, sigma)
    denom_obs = (abs_delta_dd)**2
    return num / (denom_obs + denom_th)


def lnlike( x, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
    return np.exp(-chi_2(x, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd) * 0.5)

def lnprob( x, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
    lp = lnprior(x)
    if np.any(~np.isfinite(lp)):
        return -np.inf
    return lp + np.sum(np.log(lnlike(x, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd)))


##### utils for the direct fit ###############


def gamma_z1(g0,g1,z):
    return g0+g1*z

def lnprobdirect(g, zl, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
    g0, g1 = g
    total_lnlike = 0
   
    # Iterate over each row in the fits table
    for i in range(len(theta)):
        x = gamma_z1(g0, g1, zl[i])

        # Check if x is within the desired range
        if not (1.51 < x < 2.49):
            return -np.inf

        total_lnlike += np.log(lnlike(x, theta[i], theta_ap[i], sigma[i], dd[i], 
                                      abs_delta_sigma_ap[i], abs_delta_dd[i]))

#     lp = lnprior(g, zl)
#     if not np.isfinite(lp):
#         return -np.inf    
    return total_lnlike

############### linear fit #################### 

def weighted_chi_square(theta, z, y, y_err):
    """
    Calculate the weighted chi-square statistic.

    Parameters:
    x (numpy.ndarray): Independent variable.
    y (numpy.ndarray): Observed dependent variable.
    y_err (numpy.ndarray): Uncertainties in the observed y-values.
    model_func (function): A function that takes x and returns the expected y-values.

    Returns:
    float: Chi-square statistic.
    """
    g0,g1 = theta
    y_expected = gamma_z1(g0,g1,z)
    chi_square = np.sum(((y - y_expected) ** 2) / (y_err ** 2))
    return chi_square

    
def lnlike_linear(theta, z, y, y_err):
    
    return np.exp(-weighted_chi_square(theta, z, y, y_err)*0.5)


def lnproblinear(theta, x, y, yerr):
    g0, g1 = theta
    total_lnlike = 0

    # Iterate over each row in the fits table
    for i in range(len(x)):
        gamma = gamma_z1(g0, g1, x[i])


        # Check if x is within the desired range
        if not (1.51 < gamma < 2.49):
            return -np.inf

        total_lnlike += np.log(lnlike_linear(theta, x[i], y[i], yerr[i]))

#     lp = lnprior(g, zl)
#     if not np.isfinite(lp):
#         return -np.inf    
    return total_lnlike


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
    
    print(samples.shape)


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
        plt.savefig(os.path.join(output_folder , "gamma_evo_GP.png"))

    # Show the plot
    plt.show(block=False)