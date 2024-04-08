import numpy as np
import matplotlib.pyplot as plt
import os
from getdist import plots, MCSamples
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


def plot_hist_bins(lens_table, bin_edges, 
                   color_points, output_folder, 
                   plot_name = 'hist_zl.png'):

    sns.set(style="whitegrid")
    # Plot the histogram
    plt.hist(lens_table['zl'], bins=bin_edges, color=color_points, alpha=0.7)
    plt.xlabel('$z_l$')
    plt.ylabel('Counts')
    plt.title('Histogram with Fixed Number of Elements per Bin')
    plt.savefig(os.path.join(output_folder, plot_name),
                 transparent=False, facecolor='white', bbox_inches='tight')
    #plt.show(block=False)
    plt.close()
    
def fit_line(x, y, y_err):

    # Perform a weighted linear fit
    coefficients = np.polyfit(x, y, 1, w=1/np.array(y_err))
    m = coefficients[0]  # Slope of the line
    b = coefficients[1]  # Intercept of the line
    
    return m, b


def plot_point_with_fit(x, y, y_err, 
    x_label='$z_L$',
    y_label = '$\gamma$',
    plot_name = 'linear_fit_gamma.png',
    label = 'SGL', 
    output_folder=None,
    m=None,
    b=None,
    color_points = 'firebrick',
    plot_residuals = True,
    correlation=False,
    close_plot=True):

    if m is None and b is None:
        # Perform a weighted linear fit
        m, b = fit_line(x, y, y_err)
        
    elif m is not None and b is not None:
        print('Using the given m and b values')
    else:
        raise ValueError('I need both values of m and b')
    
    if output_folder is not None:
        pd.DataFrame({'b': [b],
            'm': [m]}).to_csv(os.path.join(output_folder,'m_b_linear_fit.csv' ))
    
    # Generate points for the best fit line
    x_range = np.ptp(x) 
    delta_x = 0.1 * x_range

    xfit = np.linspace(min(x)-delta_x, max(x)+delta_x, 100)
    yfit = m * xfit + b

    sns.set(style="whitegrid")

    fig = plt.figure(dpi=200)
    
    plt.plot(xfit, yfit, 'k--', lw=1.5, label=f'{y_label} = {round(m, 3)}{x_label} + {round(b, 3)}')
    #plt.plot(xfit, yfit, 'k--', lw=1.5, label=r'$\rm y = \rm %0.3fx + %0.3f$' % (m, b))

    if correlation:
        # Calculate Pearson correlation coefficient
        correlation_coefficient, _ = pearsonr(x, y)

        # Print correlation coefficient on the plot
        plt.annotate(f'Correlation: {round(correlation_coefficient, 3)}', 
                 xy=(0.9, 0.05), xycoords='axes fraction',
                 fontsize=10, ha='right', va='bottom',
                 bbox=dict(boxstyle='round', fc='w', alpha=0.7))
        
    if plot_residuals:

        # Calculate standard deviation of residuals
        residuals = y - (m * x + b)
        std_residuals = np.std(residuals)

        # Calculate upper and lower errorbars 
        upper_error = yfit + 1 * std_residuals
        lower_error = yfit - 1 * std_residuals

        plt.fill_between(xfit, upper_error, lower_error, 
                         alpha=0.1, color="k", edgecolor="none", label = "1$\sigma$ residuals")
    
    plt.errorbar(x, y, yerr=y_err, fmt='o', color= color_points, markersize=5, 
                 capsize=2, elinewidth=1, label=f'{label}', alpha =0.4)
    
    plt.xlabel(x_label, fontsize = 15)
    plt.ylabel(y_label, fontsize = 15)

    plt.xlim(min(x)-delta_x, max(x)+delta_x)

    fig.legend(loc='lower left',bbox_to_anchor=((0.12, 0.12)))#loc='upper center', bbox_to_anchor=(0.5, 1.1))

    if output_folder is not None:
        plt.savefig(os.path.join(output_folder, plot_name).replace('$', '').replace('\\', '') ,
                    transparent=False, facecolor='white', )
    if close_plot:
        plt.close()


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

def plot_GetDist(samples, param_labels, output_folder, plot_name= 'Posterior_Dist.png' ):


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
        title = f"{param_name} = {round(median,3)} ± {round(mad,3)}"
        ax = g.subplots[i, i]
        ax.set_title(title, fontsize=12)

    
    if output_folder is not None:
        # Optionally, save the plot
        plt.savefig(os.path.join(output_folder , plot_name), bbox_inches="tight", dpi=200)
    plt.close()

   #plt.show( block=False)


