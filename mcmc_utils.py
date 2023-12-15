from scipy.special import gamma
import numpy as np
from astropy.constants import c
from scipy.integrate import quad, trapz
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt
import os
import math

c = c.to('km/s').value

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

'''
def lnlike( x, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
    return np.exp(-chi_2(x, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd) * 0.5)
'''


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


def lnprob_K(g, zl, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):

    g0, g1, delta, beta = g
    total_lnlike = 0

    # Iterate over each row in the fits table
    for i in range(len(theta)):
        x = gamma_z1(g0, g1, zl[i])
        f_g = f_prime(x, delta, beta)

        # Check if x, delta, and beta are within the desired range
        if not (1.2 < x < 2.8 and 1.8 < delta < 2.8 and -3. < beta < 1. and f_g>0.):
            return -np.inf

        # Check if f_prime result is NaN
        f_prime_result = f_prime(x, delta, beta)
        if np.isnan(f_prime_result):
            return -np.inf

        # Compute the log-likelihood
        total_lnlike += np.log(lnlike(x, theta[i], theta_ap[i], sigma[i], dd[i], 
                                      abs_delta_sigma_ap[i], abs_delta_dd[i], delta, beta))

    return total_lnlike


def lnprob_K_fixbeta(g, zl, theta, theta_ap, sigma, dd, abs_delta_sigma_ap, abs_delta_dd):
    g0, g1, d0,d1 = g
    beta = 0.18
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


def delta_z1(d0,d1,z):
    return d0+d1*z
