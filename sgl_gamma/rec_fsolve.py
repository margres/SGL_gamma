from scipy.optimize import fsolve
import numpy as np
from scipy.integrate import trapz
from mcmc_utils import f_gamma
from astropy import units as u
import os 
from astropy.table import Table

c = 299792.458 

def DA_reconstruct_z(z,zlist,hzlist,sighzlist):
    ## for 1 z, calculate the Da from reconstructed hzlist
    ## with error
    # check if z is in the redshift range of reconstructed hzlist
    if z<=max(zlist):
        # calculate the mean value
        DA_r = c/(1.+zlist[int(z*10000)])*trapz(
                1.0/hzlist[:int(z*10000)+1],x=zlist[:int(z*10000)+1])
        # calculate the error
        int_cell = abs(-1./np.array(hzlist)**2.0*np.array(sighzlist))
        int_z = trapz(int_cell[:int(z*10000)+1],zlist[:int(z*10000)+1])
        DA_sig_r = c/(1.+zlist[int(z*10000)])*int_z
    else:
        DA_r,DA_sig_r = np.nan,np.nan
        print("%f is out of the reconstruct bound." % z )
    return [DA_r,DA_sig_r]


def Dls_reconstruct_z(z1, z2, zlist, hzlist, sighzlist):

    # Check if z1 and z2 are within the bounds of zlist
    if z1 < min(zlist) or z2 > max(zlist):
        print("z1 or z2 is out of the bounds, saving as nan")
        return np.nan, np.nan
    # Find the closest indices in zlist for z1 and z2
    idx1 = np.abs(np.array(zlist) - z1).argmin()
    idx2 = np.abs(np.array(zlist) - z2).argmin()
    if idx1 > idx2:
        print("z1 should be less than z2.")
        return np.nan,np.nan
    # Calculate the integral using the trapezoidal rule
    DA_r = c / (1. + z2) * trapz(1.0 / hzlist[idx1:idx2+1], x=zlist[idx1:idx2+1])
    # Calculate the uncertainty by propagation of the uncertainty
    int_cell = np.abs(-1. / np.array(hzlist[idx1:idx2+1]) ** 2.0 * np.array(sighzlist[idx1:idx2+1]))
    int_z = trapz(int_cell, zlist[idx1:idx2+1])
    DA_sig_r = c / (1. + z2) * int_z
    return DA_r, DA_sig_r


def dis_ratio_lensing(thetaE,theta_ap,sigma_ap,pmgamma):

    # distance ratio from isotropic powerlow lens model
    p1 = thetaE/(4.*np.pi)
    p2 = c**2./sigma_ap**2.
    p3 = (thetaE/theta_ap)**(pmgamma-2.)/f_gamma(pmgamma)
    return p1*p2*p3

def dis_ratio_rec(zl,zs,res_GP):
    ## calculate the distance ratio from reconstructed hzlist
    # without error
    zlist =res_GP[:,0]
    hzlist = res_GP[:,1]
    sighzlist = res_GP[:,2]

    # Pre-allocate your result array
    ratios = np.empty(zl.shape[0])
    # Iterate over each row in your data
    for i in range(zl.shape[0]):
        z1 = zl[i]
        z2 = zs[i]
        Dls = Dls_reconstruct_z(z1,z2,zlist,hzlist,sighzlist)[0]
        Ds = DA_reconstruct_z(z2,zlist,hzlist,sighzlist)[0]
        # Compute and sok pertore the ratio
        ratios[i] = Dls/Ds

    return ratios

def func_to_solve(pmgamma,thetaE,theta_ap,sigma_ap,Da_r):
    return Da_r-dis_ratio_lensing(thetaE,theta_ap,sigma_ap,pmgamma)


def add_fsolve_table(path_project, path_table, x0=2):

    table  = Table.read(path_table)

    zl = table['zl']
    zs = table['zs']
    theta_E = table['theta_E'] * u.arcsec.to('radian')
    theta_ap = table['theta_ap'] * u.arcsec.to('radian')
    sigma_ap = table['sigma_ap']
    
    res_GP = np.load(os.path.join(path_project,'Output' ,'GP', 'hz_reconstructed_GP.npy'))
    # calculate the distance ratio
    Da_ratio = dis_ratio_rec(zl,zs,res_GP)

    gamma = []
    for i in range(0,len(Da_ratio)):

        root = fsolve(func_to_solve,x0,args=(theta_E[i] ,theta_ap[i],sigma_ap[i],Da_ratio[i]))

        if len(np.shape(root))>1:
            print('2 solutions')

        gamma.append(root[0])

    table['gamma_fsolve'] = gamma

    table.write(path_table, overwrite = True)
    
    print(f'fsolve results added to table {path_table}')

if __name__ == "__main__":

    path_project= '/home/grespanm/github/SLcosmological_parameters/SGL_gamma/'
    lens_table_path = os.path.join(path_project, 'Output' ,'GP','SGLTable_GP.fits')

    add_fsolve_table(path_project , lens_table_path)