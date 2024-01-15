import numpy as np
import matplotlib.pyplot as plt
import os
from getdist import plots, MCSamples
import pandas as pd
import seaborn as sns

def fit_line(x, y, y_err):

    # Perform a weighted linear fit
    coefficients = np.polyfit(x, y, 1, w=1/np.array(y_err))
    m = coefficients[0]  # Slope of the line
    b = coefficients[1]  # Intercept of the line
    
    return m, b


def plot_point_with_fit(x, y, y_err, 
    x_label='z$_L$',
    y_label = '$\gamma$',
    plot_name = 'linear_fit_gamma.png',
    label = None, 
    output_folder=None,
    m=None,
    b=None,
    plot_residuals = True):

    if m is None and b is None:
        # Perform a weighted linear fit
        m, b = fit_line(x, y, y_err)
        
    elif m is not None and b is not None:
        print('Using the given m and b values')
    else:
        raise ValueError('I need both values of m and b')
    
    pd.DataFrame({'b': b,
        'm': m}).to_csv(os.path.join(output_folder,'m_b_linear_fit.csv' ))
    
    # Generate points for the best fit line
    xfit = np.linspace(min(x)-delta_x, max(x)+delta_x, 100)
    yfit = m * xfit + b
    delta_x = x[2]-x[10]
    # Calculate standard deviation of residuals
    residuals = y - (m * x + b)
    std_residuals = np.std(residuals)

    # Calculate upper and lower errorbars 
    upper_error = yfit + 1 * std_residuals
    lower_error = yfit - 1 * std_residuals

    sns.set(style="whitegrid")

    fig = plt.figure(dpi=200)
    
    plt.plot(xfit, yfit, 'k--', lw=1.5, label=r'$\rm y = \rm %0.3fx + %0.3f$' % (m, b))
    if plot_residuals:
        plt.fill_between(xfit, upper_error, lower_error, alpha=0.1, color="k", edgecolor="none")
    plt.errorbar(x, y, yerr=y_err, fmt='o', color= 'firebrick', markersize=5, 
                 capsize=2, elinewidth=1, label=f'{label}', alpha =0.4)
    plt.xlabel(x_label, fontsize = 15)
    plt.ylabel(y_label, fontsize = 15)
    plt.xlim(min(xfit)-delta_x, max(xfit)+delta_x)
    fig.legend(loc='upper center')#loc='upper center', bbox_to_anchor=(0.5, 1.1))

    if output_folder is not None:
        plt.savefig(os.path.join(output_folder, plot_name),
                    transparent=False, facecolor='white', bbox_inches='tight')
    #plt.show(block=False)



def calculate_marginal_median_and_mad(samples):
    """
    Calculate the marginal median and median absolute deviation for each parameter.

    :param samples: A numpy array of MCMC samples with shape (nwalkers, nsteps, ndim)
    :return: Two numpy arrays containing the marginal medians and MADs for each parameter
    """
    # Reshape the samples to a 2D array (nwalkers*nsteps, ndim)
    nwalkers, nsteps, ndim = samples.shape
    flattened_samples = samples.reshape(-1, ndim)

    # Calculate marginal medians
    marginal_medians = np.median(flattened_samples, axis=0)

    # Calculate median absolute deviations
    mad = np.median(np.abs(flattened_samples - marginal_medians), axis=0)

    return marginal_medians, mad

def add_dollar_signs(param_labels):
    return [f"${label}$" for label in param_labels]

def plot_GetDist(samples, param_labels, output_folder ):

    marginal_medians, mads = calculate_marginal_median_and_mad(samples)
    #print(marginal_medians)
    pd.DataFrame({'median': marginal_medians, 
                  'mad': mads}).to_csv(os.path.join(output_folder,'median_mad_posterior_dist.csv' ))

    param_names = add_dollar_signs(param_labels)

    # Convert your samples to MCSamples object
    mcmc_samples = MCSamples(samples=samples, names=param_names, labels=param_labels)

    # Initialize the GetDist plotter
    g = plots.getSubplotPlotter(width_inch=10)
    g.settings.title_limit_fontsize = 15
    g.settings.axes_fontsize = 15
    g.settings.axes_labelsize = 15

    # Triangle plot with filled contours
    g.triangle_plot(mcmc_samples, param_names, filled=True)

    # Add markers and lines for best fit values
    n_params = len(marginal_medians)
    for i, val in enumerate(marginal_medians):
        # Add red dashed line to 1D subplots if the subplot exists
        if g.subplots[i, i] is not None:
            g.subplots[i, i].axvline(val, color='red', ls='--')

        # Add lines to 2D subplots if the subplot exists
        for j in range(n_params):
            if j != i:
                if g.subplots[i, j] is not None:
                    g.subplots[i, j].axhline(val, color='red', ls='--')
                if g.subplots[j, i] is not None:
                    g.subplots[j, i].axvline(val, color='red', ls='--')

    # Manually set titles for the 1D marginal distributions
    for i, param_name in enumerate(param_names):
        median = marginal_medians[i]
        mad = mads[i]
        title = f"{param_name} = {median:.3f} ± {mad:.3f}"
        ax = g.subplots[i, i]
        ax.set_title(title, fontsize=12)

    
    if output_folder is not None:
        # Optionally, save the plot
        plt.savefig(os.path.join(output_folder , "Posterior_Dist.png"), bbox_inches="tight", dpi=200)

   #plt.show( block=False)


