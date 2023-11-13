from scipy.special import gamma
import numpy as np
from astropy.constants import c

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





