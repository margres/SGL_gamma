import numpy as np
from astropy.table import Table
import io
import os
import matplotlib.pyplot as plt
import random
from gapp import gp, covariance
from scipy.integrate import quad,trapz

random.seed(42)

class GP:
    def __init__(self, 
                 lens_table_path,
                 path_project,
                 table_CC='Hz-34.txt', folder_data='Data',
                 ):
        
        self.path_project = path_project
        self.output_folder =  os.path.join(self.path_project,'Output', 'GP')
        self.output_table = os.path.join(self.output_folder, 'LensTable_dd_GP.fits')
        self.lens_table_path = lens_table_path
        self.lens_table = Table.read(self.lens_table_path)

        # Open file with the correct encoding
        with io.open(os.path.join(self.path_project, folder_data, table_CC), 'r', encoding='utf-8') as f:
            Z, Hz, Sigma = np.loadtxt(f, unpack=True)

        if not os.path.exists(self.output_folder):
            print(f'making dir {self.output_folder}')
            os.makedirs(self.output_folder, exist_ok=True)

        self.Z = Z
        self.Hz = Hz
        self.Sigma = Sigma
        self.zmin = 0
        self.zmax = np.max(Z)


    def plot_GP_result(self, zrec, hzrec, sighzrec):
        # latex rendering text fonts
        # plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    
        # ====== Create figure size in inches
        fig, ax = plt.subplots(figsize=(12., 8.))
    
        # ========= Define axes
        plt.xlabel(r"$z$", fontsize=22)
        plt.ylabel(r"$H(z)$ ($\mathrm{km} \; \mathrm{s}^{-1} \; \mathrm{Mpc}^{-1}$)", fontsize=22)
        plt.xlim(self.zmin, 2.1)
    
        # ========== Plotting the real data points and reconstructed H(z) curves - from 1 to 3sigma
        errorbar_cc = plt.errorbar(self.Z, self.Hz, yerr=self.Sigma, fmt='o', color='black')
        fitcurve01, = ax.plot(zrec, hzrec)
        fitcurve01_1sig = ax.fill_between(zrec, hzrec + 1. * sighzrec, hzrec - 1. * sighzrec,
                                          facecolor='#F08080', alpha=0.80, interpolate=True)
        ax.legend([errorbar_cc, (fitcurve01, fitcurve01_1sig)],
                  [r"Cosmology Chronometer", r"$GPfit+1\sigma$"], fontsize='22', loc='upper left')
        plt.title("H(z) reconstruction with 34 CC")
    
        # =========== saving the plot
        
        plt.savefig(os.path.join(self.output_folder, 'Hz_cc_reconstructed.png'))
    
        plt.show(block=False)


    def Dls_reconstruct_z(self, z1, z2, zlist, hzlist, sighzlist):
        c = 299792.458 # km/s
        # Check if z1 and z2 are within the bounds of zlist
        if z1 < min(zlist) or z2 > max(zlist):
            print("z1 or z2 is out of the bounds, saving as -1")
            return np.nan, np.nan
        # Find the closest indices in zlist for z1 and z2
        idx1 = np.abs(np.array(zlist) - z1).argmin()
        idx2 = np.abs(np.array(zlist) - z2).argmin()
        if idx1 > idx2:
            print("z1 should be less than z2.")
            return ,np.nan
        # Calculate the integral using the trapezoidal rule
        DA_r = c / (1. + z2) * trapz(1.0 / hzlist[idx1:idx2+1], x=zlist[idx1:idx2+1])
        # Calculate the uncertainty by propagation of the uncertainty
        int_cell = np.abs(-1. / np.array(hzlist[idx1:idx2+1]) ** 2.0 * np.array(sighzlist[idx1:idx2+1]))
        int_z = trapz(int_cell, zlist[idx1:idx2+1])
        DA_sig_r = c / (1. + z2) * int_z
        return DA_r, DA_sig_r

    def calculate_uncertainty(self, a, b, a_err, b_err):
        """
        Calculate the uncertainty of the division of two independent quantities with uncertainties.
        Parameters:
        a (float): Value of the first quantity.
        b (float): Value of the second quantity.
        a_err (float): Uncertainty in the first quantity.
        b_err (float): Uncertainty in the second quantity.
        Returns:
        tuple: A tuple containing the result of the division and its uncertainty.
        """
        # Prevent division by zero
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        # Calculate the result of the division
        c = a / b
        # Calculate the relative uncertainties
        rel_uncertainty_a = a_err / a if a != 0 else 0
        rel_uncertainty_b = b_err / b if b != 0 else 0
        # Calculate the absolute uncertainty in the result using the formula for propagation of uncertainty
        c_err = np.abs(c) * np.sqrt(rel_uncertainty_a**2 + rel_uncertainty_b**2)
        return c, c_err


    def main(self, plot_results=True):

        g1 = gp.GaussianProcess(self.Z, self.Hz, self.Sigma,
                                covfunction=covariance.Matern92, cXstar=(self.zmin, self.zmax, 19651))

        (rec1, theta1) = g1.gp(thetatrain='True')

        zrec = rec1[:, 0]
        hzrec = rec1[:, 1]
        sighzrec = rec1[:, 2]

        dd_CC_GP = []
        dd_CC_GP_err = []
        for index, row in enumerate(self.lens_table):
            Dls = self.Dls_reconstruct_z(row['zl'], row['zs'], zrec, hzrec, sighzrec)[0]
            Ds = self.Dls_reconstruct_z(0.0, row['zs'], zrec, hzrec, sighzrec)[0]
            Dls_err = self.Dls_reconstruct_z(row['zl'], row['zs'], zrec, hzrec, sighzrec)[1]
            Ds_err = self.Dls_reconstruct_z(0.0, row['zs'], zrec, hzrec, sighzrec)[1]
            dd_CC_GP.append(self.calculate_uncertainty(Dls, Ds, Dls_err, Ds_err)[0])
            dd_CC_GP_err.append(self.calculate_uncertainty(Dls, Ds, Dls_err, Ds_err)[1])

        print(f"Saving distance reconstruction in {os.path.join(self.output_folder,f'Lens_table_GP.fits')}")
        self.lens_table['dd_from_GP'] = dd_CC_GP
        self.lens_table['dd_error_from_GP'] = dd_CC_GP_err
        self.lens_table.write(self.output_table, overwrite =True)

       
        # ======= printing the reconstructed H(z) at the lowest point, i.e., zmin=0, and its relative uncertainty
        # print(f'z = {zrec[0]}, H0 = {hzrec[0]}, sigH0 = {sighzrec[0]}, sigH0/H0(%) = {(sighzrec[0] / hzrec[0]) * 100.}')
        print('z = {}, H0 = {}, sigH0 = {}, sigH0/H0(%) = {}'.format(zrec[0], hzrec[0], sighzrec[0], (sighzrec[0] / hzrec[0]) * 100.))
        # print(f'saving results GP as {os.path.join(self.path_project, "hz_reconstructed_GP.txt")}')
        print('saving results GP as {}'.format(os.path.join(self.path_project, "hz_reconstructed_GP.npy")))
        # ========== saving the reconstructed hz
        np.save(os.path.join(self.output_folder, 'hz_reconstructed_GP.npy'), rec1)

        if plot_results:
            print('plotting the results')
            self.plot_GP_result(zrec, hzrec, sighzrec)


if __name__ == "__main__":

    path_project='/home/grespanm/github/SLcosmological_parameters/SGL_gamma/'
    lens_table_path = os.path.join(path_project, 'Data' , 'LensTable02.fits')

    GP = GP(lens_table_path=lens_table_path, path_project=path_project)
    GP.run()
    print("Done!")
