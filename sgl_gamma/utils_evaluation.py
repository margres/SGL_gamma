import numpy as  np

def AIC(n_parameters, max_log_likelihood):
    """
    Calculate the Bayesian Information Criterion (BIC).

    Parameters:
    - n_parameters: Number of parameters in the model.
    - max_log_likelihood: Maximum log-likelihood of the model.

    Returns:
    - BIC value for the model.
    """
    return 2*n_parameters - 2 * max_log_likelihood

def BIC(n_data_points, n_parameters, max_log_likelihood):
    """
    Calculate the Bayesian Information Criterion (BIC).

    Parameters:
    - n_data_points: Number of data points.
    - n_parameters: Number of parameters in the model.
    - max_log_likelihood: Maximum log-likelihood of the model.

    Returns:
    - BIC value for the model.
    """
    return n_parameters * np.log(n_data_points) - 2 * max_log_likelihood

def get_marginal(sample):

    hist, bin_edges = np.histogram(sample, bins='auto', density=True)
    # Find the bin with the highest frequency
    max_bin_index = np.argmax(hist)
    # Estimate the mode as the midpoint of the bin with the highest frequency
    marginal_peak = 0.5 * (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1])
    
    return marginal_peak

def marginal_peak_and_mad(samples, ndim=None):
    """
    Calculate the marginal peak (mode) and median absolute deviation for each parameter.

    :param samples: A numpy array of MCMC samples with shape (nwalkers, nsteps, ndim)
    :return: Two numpy arrays containing the marginal peaks and MADs for each parameter
    """
    if len(np.shape(samples))>1:
        # Reshape the samples to a 2D array (nwalkers*nsteps, ndim)
        nwalkers, nsteps, ndim = samples.shape
        samples = samples.reshape(-1, ndim)

    marginal_peaks = np.empty(ndim)

    for i in range(ndim):
        marginal_peaks[i] =  get_marginal(samples[:, i])

    # Calculate median absolute deviations
    mad = np.median(np.abs(samples - np.median(samples, axis=0)), axis=0)

    return marginal_peaks, mad