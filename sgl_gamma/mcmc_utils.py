from scipy.special import gamma
import numpy as np
from astropy.constants import c
from scipy.integrate import quad, trapz
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt
import os
import math

c = c.to('km/s').value
#beta used as 
beta_manga = 0.002

def f_gamma(x):
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
    
    return a* np.power((theta / theta_ap), x-2) / f_gamma(x)


def solve_for_gamma(x, theta, theta_ap, sigma, dd):
    return dd - d_th(x, theta, theta_ap, sigma)

def chi_2(x, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
    num = solve_for_gamma(x, theta, theta_ap, sigma, dd)**2
    denom_th = 4 * (abs_delta_sigma_ap / sigma)**2 + ((1 - x) * 0.05)**2 * d_th(x, theta, theta_ap, sigma)
    denom_obs = (abs_delta_dd)**2
    return num / (denom_obs + denom_th)

def lnlike(x, *args):
   
    if len(args) == 8:
        return np.exp(-chi_2_K(x, *args) * 0.5)
    else:
        return np.exp(-chi_2(x, *args) * 0.5)
    
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

    for i in range(len(x)):
        gamma = gamma_z1(g0, g1, x[i])

        # Check if x is within the desired range
        if not (1.51 < gamma < 2.49):
            return -np.inf

        total_lnlike += np.log(lnlike_linear(theta, x[i], y[i], yerr[i]))

    return total_lnlike


#### Koopmans lens model #####

def chi_2_K(x,theta, theta_ap, sigma,dd,abs_delta_sigma_ap,abs_delta_dd,delta, beta):
    
    num = solve_for_gamma_K(x,theta, theta_ap, sigma,dd,delta, beta)**2
    denom_th = 4*(abs_delta_sigma_ap/sigma)**2 + ((1-x)*0.05)**2 * d_th_prime(x,theta, theta_ap, sigma,delta, beta)
    denom_obs =  (abs_delta_dd)**2
    
    return num/(denom_obs + denom_th)

def f_prime(gamma_val, delta, beta):

    # Adding the missing term 1 / (2 * sqrt(pi))
    prefactor = 1. / (2. * math.sqrt(math.pi))
    term1 = (gamma_val + delta - 5.) * (gamma_val + delta - 2. - 2. * beta) / (delta - 3.)
    term2_numerator = gamma((gamma_val + delta - 2.) / 2.) * gamma((gamma_val + delta) / 2.)
    term2_denominator = gamma((gamma_val + delta) / 2.) * gamma((gamma_val + delta - 3.) / 2.) - beta * gamma((gamma_val + delta - 2.) / 2.) * gamma((gamma_val + delta - 1.) / 2.)
    term3 = gamma((delta - 1.) / 2.) * gamma((gamma_val - 1.) / 2.) / (gamma(delta / 2.) * gamma(gamma_val / 2.))
    
    return prefactor * term1 * (term2_numerator / term2_denominator) * term3

def solve_for_gamma_K(x,theta, theta_ap, sigma,dd,delta, beta):
    
    return dd - d_th_prime(x,theta, theta_ap, sigma,delta, beta)

def d_th_prime(x,theta, theta_ap, sigma,delta, beta):
    
    a = (theta*c**2) / (4 * np.pi * sigma**2)
    
    return a* np.power((theta / theta_ap), x-2) / f_prime(x, delta, beta)



def lnprob_K_2D(g, zl, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
    g0, d0 = g
    beta = beta_manga
    total_lnlike = 0

    # Iterate over each row in the fits table
    for i in range(len(theta)):
        x = g0
        delta = d0
        f_g = f_prime(x, delta, beta)
        # Check if x, delta, and beta are within the desired range
        if not (1.2 < x < 2.8 and 1.2 < delta < 2.8 and f_g>0.):
            return -np.inf

        # Check if f_prime result is NaN
        f_prime_result = f_prime(x, delta, beta)
        if np.isnan(f_prime_result):
            return -np.inf

        # Compute the log-likelihood
        total_lnlike += np.log(lnlike(x, theta[i], theta_ap[i], sigma[i], dd[i], 
                                      abs_delta_sigma_ap[i], abs_delta_dd[i], delta, beta))


    return total_lnlike

def lnprob_K_3D(g, zl, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
    g0, d0, beta = g
    total_lnlike = 0

    # Define the lower limit (a), peak (c), and upper limit (b) of the triangular distribution for beta
    a, c, b = -0.356, 0.002, 0.512  # Adjust these values as needed
    
    # Check if beta is within the bounds of the triangular distribution
    if not (a <= beta <= b):
        return -np.inf

    # Compute the log of the triangular prior for beta
    if a <= beta <= c:
        ln_prior_beta = np.log(2 * (beta - a) / ((b - a) * (c - a)))
    else: # c < beta <= b
        ln_prior_beta = np.log(2 * (b - beta) / ((b - a) * (b - c)))

    # Iterate over each row in the fits table
    for i in range(len(theta)):
        x = g0
        delta = d0
        f_g = f_prime(x, delta, beta)

        # Check if x, delta, and f_g are within the desired range
        if not (1.2 < x < 2.8 and 1.2 < delta < 2.8 and f_g > 0.):
            return -np.inf

        # Check if f_prime result is NaN
        f_prime_result = f_prime(x, delta, beta)
        if np.isnan(f_prime_result):
            return -np.inf

        # Compute the log-likelihood for this row
        total_lnlike += np.log(lnlike(x, theta[i], theta_ap[i], sigma[i], dd[i], 
                                      abs_delta_sigma_ap[i], abs_delta_dd[i], delta, beta))

    # Return the total log-likelihood plus the log-prior for beta
    return total_lnlike + ln_prior_beta
'''

def lnprob_K_3D(g, zl, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
    g0, d0,beta = g
    total_lnlike = 0

    # Iterate over each row in the fits table
    for i in range(len(theta)):
        x = g0
        delta = d0
        f_g = f_prime(x, delta, beta)
        # Check if x, delta, and beta are within the desired range
        if not (1.2 < x < 2.8 and 1.2 < delta < 2.8 and -0.287<beta<0.505 and f_g>0.):
            return -np.inf

        # Check if f_prime result is NaN
        f_prime_result = f_prime(x, delta, beta)
        if np.isnan(f_prime_result):
            return -np.inf

        # Compute the log-likelihood
        total_lnlike += np.log(lnlike(x, theta[i], theta_ap[i], sigma[i], dd[i], 
                                      abs_delta_sigma_ap[i], abs_delta_dd[i], delta, beta))


    return total_lnlike


'''


def lnprob_K_4D(g, zl, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
    g0, g1, d0,d1 = g
    beta = beta_manga
    total_lnlike = 0

    # Iterate over each row in the fits table
    for i in range(len(theta)):
        x = gamma_z1(g0, g1, zl[i])
        delta = delta_z1(d0,d1,zl[i])
        f_g = f_prime(x, delta, beta)
        # Check if x, delta, and beta are within the desired range
        if not (1.2 < x < 2.8 and 1.2 < delta < 2.8  and f_g>0.):
            return -np.inf

        # Check if f_prime result is NaN
        f_prime_result = f_prime(x, delta, beta)
        if np.isnan(f_prime_result):
            return -np.inf

        # Compute the log-likelihood
        total_lnlike += np.log(lnlike(x, theta[i], theta_ap[i], sigma[i], dd[i], 
                                      abs_delta_sigma_ap[i], abs_delta_dd[i], delta, beta))
    return total_lnlike

def lnprob_K_5D(g, zl, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
    
    #5d with flat prior
    
    g0, g1, d0,d1,beta = g
    total_lnlike = 0

    # Iterate over each row in the fits table
    for i in range(len(theta)):

        x = gamma_z1(g0, g1, zl[i])
        delta = delta_z1(d0,d1,zl[i])
        f_g = f_prime(x, delta, beta)
        # Check if x, delta, and beta are within the desired range
        if not (1.2 < x < 2.8 and 1.2 < delta < 2.8 and -0.287<beta<0.505 and f_g>0.):
            return -np.inf

        # Check if f_prime result is NaN
        f_prime_result = f_prime(x, delta, beta)
        if np.isnan(f_prime_result):
            return -np.inf

        # Compute the log-likelihood
        total_lnlike += np.log(lnlike(x, theta[i], theta_ap[i], sigma[i], dd[i], 
                                      abs_delta_sigma_ap[i], abs_delta_dd[i], delta, beta))

    return total_lnlike

# def lnprob_K_5D(g, zl, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
#     g0, g1, d0, d1, beta = g
#     total_lnlike = 0

#     # Define the lower limit (a), peak (c), and upper limit (b) of the triangular distribution for beta
#     a, c, b = -0.356, 0.002, 0.512  # Adjust these values as needed
    
#     # Check if beta is within the bounds of the triangular distribution
#     if not (a <= beta <= b):
#         return -np.inf

#     # Compute the log of the triangular prior for beta
#     if a <= beta <= c:
#         ln_prior_beta = np.log(2 * (beta - a) / ((b - a) * (c - a)))
#     else: # c < beta <= b
#         ln_prior_beta = np.log(2 * (b - beta) / ((b - a) * (b - c)))

#     # Iterate over each row in the fits table
#     for i in range(len(theta)):
#         x = gamma_z1(g0, g1, zl[i])
#         delta = delta_z1(d0, d1, zl[i])
#         f_g = f_prime(x, delta, beta)

#         # Check if x, delta, and f_g are within the desired range
#         if not (1.2 < x < 2.8 and 1.2 < delta < 2.8 and f_g > 0.):
#             return -np.inf

#         # Check if f_prime result is NaN
#         f_prime_result = f_prime(x, delta, beta)
#         if np.isnan(f_prime_result):
#             return -np.inf

#         # Compute the log-likelihood for this row
#         total_lnlike += np.log(lnlike(x, theta[i], theta_ap[i], sigma[i], dd[i], 
#                                       abs_delta_sigma_ap[i], abs_delta_dd[i], delta, beta))

#     # Return the total log-likelihood plus the log-prior for beta
#     return total_lnlike + ln_prior_beta

# def delta_z1(d0,d1,z):
#     return d0+d1*z

# def lnprob_K_5D(g, zl, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
#     g0, g1, d0, d1, beta = g
#     total_lnlike = 0
    
#     # Define the mean (mu) and standard deviation (sigma) of the Gaussian distribution for beta
#     mu, sigma_beta = 0.22, 0.2  # Adjust these values as needed
    
#     # Compute the log of the Gaussian prior for beta
#     ln_prior_beta = -0.5 * np.log(2 * np.pi * sigma_beta**2) - ((beta - mu)**2 / (2 * sigma_beta**2))
    
#     # Iterate over each row in the fits table
#     for i in range(len(theta)):
#         x = gamma_z1(g0, g1, zl[i])
#         delta = delta_z1(d0, d1, zl[i])
#         f_g = f_prime(x, delta, beta)
        
#         # Check if x, delta, and f_g are within the desired range
#         if not (1.2 < x < 2.8 and 1.2 < delta < 2.8 and f_g > 0.):
#             return -np.inf
        
#         # Check if f_prime result is NaN
#         if np.isnan(f_g):
#             return -np.inf
        
#         # Compute the log-likelihood for this row
#         total_lnlike += np.log(lnlike(x, theta[i], theta_ap[i], sigma[i], dd[i],
#                                       abs_delta_sigma_ap[i], abs_delta_dd[i], delta, beta))
    
#     # Return the total log-likelihood plus the log-prior for beta
#     return total_lnlike + ln_prior_beta


###============================================================================

def delta_z1_scale(d0,d1,z):
    return d0+d1*z/(1.+z)
def gamma_z1_scale(g0,g1,z):
    return g0+g1*z/(1.+z)

def lnprob_K_5D_scale(g, zl, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
    ## scale evolved
    g0, g1, d0, d1, beta = g
    total_lnlike = 0
    
    # Define the mean (mu) and standard deviation (sigma) of the Gaussian distribution for beta
    mu, sigma_beta = 0.22, 0.2  # Adjust these values as needed
    
    # Compute the log of the Gaussian prior for beta
    ln_prior_beta = -0.5 * np.log(2 * np.pi * sigma_beta**2) - ((beta - mu)**2 / (2 * sigma_beta**2))
    
    # Iterate over each row in the fits table
    for i in range(len(theta)):
        x = gamma_z1_scale(g0, g1, zl[i])
        delta = delta_z1_scale(d0, d1, zl[i])
        f_g = f_prime(x, delta, beta)
        
        # Check if x, delta, and f_g are within the desired range
        if not (1.2 < x < 2.8 and 1.2 < delta < 2.8 and f_g > 0.):
            return -np.inf
        
        # Check if f_prime result is NaN
        if np.isnan(f_g):
            return -np.inf
        
        # Compute the log-likelihood for this row
        total_lnlike += np.log(lnlike(x, theta[i], theta_ap[i], sigma[i], dd[i],
                                      abs_delta_sigma_ap[i], abs_delta_dd[i], delta, beta))
    
    # Return the total log-likelihood plus the log-prior for beta
    return total_lnlike + ln_prior_beta

###============================================================================

def delta_z1_log(d0,d1,z):
    
     return d0+d1*np.log(1.+z)
    
def gamma_z1_log(g0,g1,z):
    
    return g0+g1**np.log(1.+z)

def lnprob_K_5D_log(g, zl, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
     ## log evolved
     
    g0, g1, d0, d1, beta = g
    total_lnlike = 0

    # Define the mean (mu) and standard deviation (sigma) of the Gaussian distribution for beta
    mu, sigma_beta = 0.22, 0.2  # Adjust these values as needed
    
     # Compute the log of the Gaussian prior for beta
    ln_prior_beta = -0.5 * np.log(2 * np.pi * sigma_beta**2) - ((beta - mu)**2 / (2 * sigma_beta**2))
    
    # Iterate over each row in the fits table
    for i in range(len(theta)):
        x = gamma_z1_log(g0, g1, zl[i])
        delta = delta_z1_log(d0, d1, zl[i])
        f_g = f_prime(x, delta, beta)
        
         # Check if x, delta, and f_g are within the desired range
        if not (1.2 < x < 2.8 and 1.2 < delta < 2.8 and f_g > 0.):
            return -np.inf
        
        # Check if f_prime result is NaN
        if np.isnan(f_g):
            return -np.inf
        
         # Compute the log-likelihood for this row
        total_lnlike += np.log(lnlike(x, theta[i], theta_ap[i], sigma[i], dd[i],
                                       abs_delta_sigma_ap[i], abs_delta_dd[i], delta, beta))
    
     # Return the total log-likelihood plus the log-prior for beta
    return total_lnlike + ln_prior_beta


# def gamma_z2(g0,gz,z,gs,sigma):
#     return g0+gz*(z-0.3)+gs*(np.log10(sigma)-np.log10(246.44))

# def delta_z2(d0,dz,z,ds,sigma):
#     return d0+dz*(z-0.3)+ds*(np.log10(sigma)-np.log10(246.44))

# def lnprob_K_6D(g, zl, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
#     g0, gz, gs, d0, dz, ds = g
#     total_lnlike = 0
#     beta = 0.22
    
#     # # Define the mean (mu) and standard deviation (sigma) of the Gaussian distribution for beta
#     # mu, sigma_beta = 0.22, 0.2  # Adjust these values as needed
    
#     # # Compute the log of the Gaussian prior for beta
#     # ln_prior_beta = -0.5 * np.log(2 * np.pi * sigma_beta**2) - ((beta - mu)**2 / (2 * sigma_beta**2))
    
#     # Iterate over each row in the fits table
#     for i in range(len(theta)):
#         x = gamma_z2(g0,gz,zl[i],gs,sigma[i])
#         delta = delta_z2(d0,dz,zl[i],ds,sigma[i])
#         f_g = f_prime(x, delta, beta)
        
#         # Check if x, delta, and f_g are within the desired range
#         if not (1.2 < x < 2.8 and 1.2 < delta < 2.8 and f_g > 0.):
#             return -np.inf
        
#         # Check if f_prime result is NaN
#         if np.isnan(f_g):
#             return -np.inf
        
#         # Compute the log-likelihood for this row
#         total_lnlike += np.log(lnlike(x, theta[i], theta_ap[i], sigma[i], dd[i],
#                                       abs_delta_sigma_ap[i], abs_delta_dd[i], delta, beta))
    
#     # Return the total log-likelihood plus the log-prior for beta
#     return total_lnlike

# def lnprob_K_7D(g, zl, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
#     g0, gz, gs, d0, dz, ds, beta = g
#     total_lnlike = 0
    
#     # Define the mean (mu) and standard deviation (sigma) of the Gaussian distribution for beta
#     mu, sigma_beta = 0.22, 0.2  # Adjust these values as needed
    
#     # Compute the log of the Gaussian prior for beta
#     ln_prior_beta = -0.5 * np.log(2 * np.pi * sigma_beta**2) - ((beta - mu)**2 / (2 * sigma_beta**2))
    
#     # Iterate over each row in the fits table
#     for i in range(len(theta)):
#         x = gamma_z2(g0,gz,zl[i],gs,sigma[i])
#         delta = delta_z2(d0,dz,zl[i],ds,sigma[i])
#         f_g = f_prime(x, delta, beta)
        
#         # Check if x, delta, and f_g are within the desired range
#         if not (1.2 < x < 2.8 and 1.2 < delta < 2.8 and f_g > 0.):
#             return -np.inf
        
#         # Check if f_prime result is NaN
#         if np.isnan(f_g):
#             return -np.inf
        
#         # Compute the log-likelihood for this row
#         total_lnlike += np.log(lnlike(x, theta[i], theta_ap[i], sigma[i], dd[i],
#                                       abs_delta_sigma_ap[i], abs_delta_dd[i], delta, beta))
    
#     # Return the total log-likelihood plus the log-prior for beta
#     return total_lnlike + ln_prior_beta


# def delta_z1(d0,d1,z):
#     return d0+d1*z
