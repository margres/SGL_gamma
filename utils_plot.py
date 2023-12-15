import numpy as np
import matplotlib.pyplot as plt
import os
from getdist import plots, MCSamples


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
    label = '', 
    output_folder=None):

    # Perform a weighted linear fit
    m, b = fit_line(x, y, y_err)

    # Generate points for the best fit line
    xfit = np.linspace(0, 1, 100)
    yfit = m * xfit + b

    # Calculate standard deviation of residuals
    residuals = y - (m * x + b)
    std_residuals = np.std(residuals)

    # Calculate upper and lower errorbars 
    upper_error = yfit + 1 * std_residuals
    lower_error = yfit - 1 * std_residuals

    fig = plt.figure(dpi=100)

    plt.plot(xfit, yfit, 'k--', lw=1, label=r'Best fit: $\rm y = \rm %0.2fx + %0.2f$' % (m, b))
    plt.fill_between(xfit, upper_error, lower_error, alpha=0.1, color="k", edgecolor="none")
    plt.errorbar(x, y, yerr=y_err, fmt='.', markersize=8, capsize=2, elinewidth=2, label=f'{label}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    fig.legend()#loc='upper center', bbox_to_anchor=(0.5, 1.1))

    if output_folder is not None:
        plt.savefig(os.path.join(output_folder, plot_name),
                    transparent=False, facecolor='white', bbox_inches='tight')
    plt.show(block=False)

# Example usage:
# Assuming you have x, y, and y_err data arrays and other parameters
# plot_fit(x, y, y_err, bin_width=0.1, model='YourModel', output_folder='path/to/output')


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

def setGetDist(samples, param_names, param_labels, output_folder = None):


    # Convert your samples to MCSamples object
    mcmc_samples = MCSamples(samples=samples, names=param_names, labels=param_labels)

    # Initialize the GetDist plotter
    g = plots.getSubplotPlotter(width_inch=10)
    g.settings.title_limit_fontsize = 15
    g.settings.axes_fontsize = 15
    g.settings.axes_labelsize = 15

    # Triangle plot with filled contours
    g.triangle_plot(mcmc_samples, param_names, filled=True, title_limit=1)

    if output_folder is not None:
        # Optionally, save the plot
        plt.savefig(os.path.join(output_folder , "ProbDist.png"))

    # Show the plot
    plt.show(block=False)


def plot_directfit(samples, param_names, output_folder=None):

    # If you have parameter names and want to include them in the plot
    param_labels = param_names #['\gamma_0', '\gamma_1']

    median_list = [] # median, err_lower, err_upper 
    for i in range(samples.shape[-1]):  # Loop over parameters
        median_list.append(get_param_stats(samples, i))
    
    print(samples.shape)

    setGetDist(samples, param_names, param_labels, output_folder)

    '''
    for ax1 in g.subplots[:,0]:
        ax1.axvline(median_list[0][0], color='red', ls='--')

    ax2 = g.subplots[1, 1]
    ax2.axvline(median_list[1][0], color='red', ls='--')

    ax2 = g.subplots[1, 0]
    ax2.axhline(median_list[1][0], color='red', ls='--')
    '''



