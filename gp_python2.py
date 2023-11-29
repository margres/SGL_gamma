import numpy as np
import io
import os
import matplotlib.pyplot as plt
import random
from gapp import gp, covariance

random.seed(42)

class GP:
    def __init__(self, table_CC='Hz-34.txt', folder_data='Data',
                 path_project='/home/grespanm/github/SLcosmological_parameters/',
                 folder_figures='figures'
                 ):
        self.path_project = path_project

        # Open file with the correct encoding
        with io.open(os.path.join(self.path_project, folder_data, table_CC), 'r', encoding='utf-8') as f:
            Z, Hz, Sigma = np.loadtxt(f, unpack=True)
        self.folder_figures = folder_figures
        self.Z = Z
        self.Hz = Hz
        self.Sigma = Sigma
        self.zmin = 0
        self.zmax = np.max(Z)

    def plot_GP_result(self, zrec, hzrec, sighzrec):

        # latex rendering text fonts
        plt.rc('text', usetex=True)
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
        if not os.path.exists(os.path.join(self.path_project, self.folder_figures)):
            print(f'making dir {os.path.join(self.path_project, self.folder_figures)}')
            os.makedirs(os.path.join(self.path_project, self.folder_figures), exist_ok=True)

        plt.savefig(os.path.join(self.path_project, self.folder_figures, 'Hz_cc_reconstrct_34_Matern92.png'))

        plt.show()

    def run(self, plot_results=True):

        g1 = gp.GaussianProcess(self.Z, self.Hz, self.Sigma,
                                covfunction=covariance.Matern92, cXstar=(self.zmin, self.zmax, 19651))

        (rec1, theta1) = g1.gp(thetatrain='True')

        zrec = rec1[:, 0]
        hzrec = rec1[:, 1]
        sighzrec = rec1[:, 2]

        # ======= printing the reconstructed H(z) at the lowest point, i.e., zmin=0, and its relative uncertainty
        print(f'z = {zrec[0]}, H0 = {hzrec[0]}, sigH0 = {sighzrec[0]}, sigH0/H0(%) = {(sighzrec[0] / hzrec[0]) * 100.}')

        print(f'saving results GP as {os.path.join(self.path_project, "hz_reconstructed_GP.txt")}')
        # ========== saving the reconstructed hz
        np.save(os.path.join(self.path_project, "hz_reconstructed_GP.txt"), rec1)

        if plot_results:
            print('plotting the results')
            self.plot_GP_result(zrec, hzrec, sighzrec)


if __name__ == "__main__":
    GP = GP()
    GP.run()
