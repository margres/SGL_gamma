import emcee
import os
import numpy as np
from scipy import integrate
from astropy.table import Table
import time
from astropy import units as u
from astropy import constants as const
import matplotlib.pyplot as plt
from datetime import datetime
from mcmc_utils import lnprob, lnprobdirect, lnproblinear, lnprob_K, lnprob_K_fixbeta
import random
from multiprocessing import Pool
from utils_plot import plot_point_with_fit, plot_directfit


random.seed(42)

class MCMC:

    def __init__(self, lens_table_path, path_project, 
                 output_folder = None, 
                 model='',
                 mode = '',
                 bin_width=None, elements_per_bin=None, nsteps=5000, 
                 nwalkers=500, ndim=None, ncpu=None,  burnin = None,
                 all_ln_probs=None,all_samples=None,
                 lnprob_touse = lnprob, x_ini=[2]
                 ):
        
        self.model = model
        self.mode = mode
        self.lens_table_path = lens_table_path
        self.bin_width = bin_width
        self.elements_per_bin = elements_per_bin
        self.nsteps = nsteps
        self.nwalkers = nwalkers
        self.x_ini = x_ini

        if ndim is None:
            self.ndim = len(self.x_ini)
        else:
            self.ndim = ndim
    
        if ncpu is None:
            self.ncpu = os.cpu_count()
        
        if mode=='linear':
            self.lnprob_touse = lnproblinear
        elif mode == 'direct':
            self.lnprob_touse = lnprobdirect
        elif mode =='':
            self.lnprob_touse = lnprob_touse
        elif mode  == 'Koopmans':
            self.lnprob_touse = lnprob_K
        elif mode == 'Koopmans_beta':
            self.lnprob_touse = lnprob_K_fixbeta
        
        if burnin is None:
            self.burnin = int(nsteps*0.1)
        else:
            self.burnin = burnin

        # Load lens table
        self.lens_table = Table.read(self.lens_table_path)
        self.binned = True
        self.output_folder = output_folder

        if model not in ['ANN', 'GP']:
            raise ValueError('model not known, only available ANN or GP')
        
        if model == 'GP':
            # there are no values for some lenses 
            self.lens_table = self.lens_table[(self.lens_table['dd_GP']>0)] 
            print(f" Using table with {len(self.lens_table)} values")
        # Constants
        #self.c = c.to('km/s').value
        
        if (bin_width is not None) and (elements_per_bin is not None):
            raise ValueError("Only one of bin_width or elements_per_bin should be defined, not both.")
            
        if (bin_width is None) and (elements_per_bin is None):
            print('No binning  - mcmc for every element')
            self.binned = False
            if self.output_folder is None:
                self.output_folder = os.path.join(path_project, 'Output',
                                                  f"MCMC_{self.model}_singular_obj_nw_{nwalkers}_ns_{nsteps}_no_burnin")
        if bin_width is not None:
            #fixed
            if self.output_folder is None:
                self.output_folder = os.path.join(path_project,'Output',
                                                  f"MCMC_{self.model}_fixed_{self.bin_width}_nwalkers_{nwalkers}_nsteps_{nsteps}_no_burnin")
            min_z_l = self.lens_table['zl'].min()
            max_z_l = self.lens_table['zl'].max()
            self.bin_edges = np.arange(min_z_l, max_z_l + self.bin_width, self.bin_width)

        if elements_per_bin is not None:
            #adaptive
            if self.output_folder is None:
                self.output_folder = os.path.join(path_project, 'Output',
                                                  f"MCMC_{self.model}_adaptive_{self.elements_per_bin}_nw_{nwalkers}_ns_{nsteps}_no_burnin")
            self.bin_edges = np.percentile(self.lens_table['zl'], np.linspace(0, 100, self.elements_per_bin + 1))

        if self.binned:
            self.hist, _ = np.histogram(self.lens_table['zl'], bins=self.bin_edges)
            self.xcenters = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
            try:    
                self.xcenters  = np.delete(self.xcenters,np.where(self.hist==0)[0][0])
                self.hist =  np.delete(self,self.hist,np.where(self.hist==0)[0][0])
            except:
                pass
    
        if not os.path.exists(self.output_folder):
            print(f'making dir {self.output_folder}')
            os.makedirs(self.output_folder, exist_ok=True)
            self.all_samples, self.all_ln_probs = all_samples, all_ln_probs
        else:
            try:
                self.all_samples =  self.load_samples_and_ln_probs()
            except Exception as e:
                print(e)
                self.all_samples, self.all_ln_probs = all_samples, all_ln_probs

        self.output_table = os.path.join(self.output_folder, f'SGL_{self.model}_gammaMCMC.fits')
        #if burnin is None:
        #      self.burnin = int(self.nsteps * 0.1)

    
    def create_output_folder(self):
            # Check if the output folder already exists
            if os.path.exists(self.output_folder):
                # Folder exists
                user_input = input(f"The folder '{self.output_folder}' already exists. Do you want to overwrite it? (y/n): ").lower()

                if user_input == 'y':
                    # Overwrite the existing folder
                    os.makedirs(self.output_folder, exist_ok=True)
                elif user_input == 'n':
                    # Create a new folder with the current date
                    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    new_folder = f"{self.output_folder}_{current_date}"
                    os.makedirs(new_folder, exist_ok=True)
                    self.output_folder = new_folder
                    print(f"New folder '{new_folder}' created.")
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")
                    # You may want to add a loop or further handling depending on your needs.
            else:
                # Folder does not exist, create it
                os.makedirs(self.output_folder, exist_ok=True)
                print(f"Folder '{self.output_folder}' created.")

    def create_out_table(self, median, mean,mad):
        # Create Astropy table
        data = {
            'Median': median,
            'Mean': mean,
            'MAD': mad,
            'bin_center': self.xcenters,
            'bin_min_edge': self.bin_edges[:-1],
            'bin_max_edge': self.bin_edges[1:],
            'n_elem': self.hist
        }

        results_table = Table(data)

        return results_table
    
    def get_subtable(self):
        # Initialize an empty list to store subtables for each bin
        subtables = []
        # Loop through the bins
        for i in range(len(self.bin_edges) - 1):
            data =  self.lens_table['zl']
            bin_start = self.bin_edges[i]
            bin_end = self.bin_edges[i + 1]

            # Filter your data within the current bin
            mask = ((data >= bin_start) & (data <= bin_end))
            sub_table =  self.lens_table[mask]
            subtables.append(sub_table)
            #print(sub_table['zl'])
        return subtables
    
    def process_subtable(self, sub_table):
    
        # Process each subtable
        zl_list, theta_E_r_list, theta_ap_r_list, sigma_ap_list, dd_list, abs_delta_sigma_ap_list, abs_delta_dd_list = [],[], [], [], [], [], []
        
        #print(np.shape(sub_table))
        if len(np.shape(sub_table))==0:
            sub_table =[sub_table]
            
        for row in sub_table:
            zl = row['zl']
            theta_E_r = row['theta_E'] * u.arcsec.to('radian')
            theta_ap_r = row['theta_ap'] * u.arcsec.to('radian')
            sigma_ap = row['sigma_ap']
            dd = row[f'dd_{self.model}']
        
            # uncertainties, global values
            abs_delta_sigma_ap = row['sigma_ap_err']
            abs_delta_dd = row[f'dd_error_{self.model}']

            # Append parameter values for this row to lists
            zl_list.append(zl)
            theta_E_r_list.append(theta_E_r)
            theta_ap_r_list.append(theta_ap_r)
            sigma_ap_list.append(sigma_ap)
            dd_list.append(dd)
            abs_delta_sigma_ap_list.append(abs_delta_sigma_ap)
            abs_delta_dd_list.append(abs_delta_dd)
            
        return np.array(zl_list),np.array(theta_E_r_list),np.array(theta_ap_r_list),np.array(sigma_ap_list),np.array(dd_list),np.array(abs_delta_sigma_ap_list),np.array(abs_delta_dd_list)
        

    def args_linear_fit(self, sub_table):
        
        zl_list, gamma_list, gamma_mad_list = [], [],[]

        if len(np.shape(sub_table))==0:
            sub_table =[sub_table]
        
        for row in sub_table:
            zl = row['zl']
            gamma = row[f'Gamma_median_{self.model}'] 
            gamma_mad = row[f'Gamma_MAD_{self.model}']

            # Append parameter values for this row to lists
            zl_list.append(zl)
            gamma_list.append(gamma)
            gamma_mad_list.append(gamma_mad)

        return np.array(zl_list),np.array(gamma_list),np.array(gamma_mad_list)
    
    def run_mcmc(self,):

        print('################ running the mcmc ##################')
        
        # Initialize lists to accumulate samples
        all_samples = []
        all_ln_probs = []
        
        if self.binned:
            print('plot hist zl')
            self.plot_hist_bins()
            subtables = self.get_subtable()
        else:
            print(f'Number of elements in the table {len(self.lens_table)}')
            subtables = self.lens_table
            if self.mode in ['Koopmans', 'Koopmans_beta'] :
                subtables =  [subtables]

        if self.mode in ['linear', 'direct' ]:
            subtables = [subtables]

        for sub_table in subtables:
            
            if len(sub_table)<1:
                continue 

            #print(len(sub_table['zl']))
            
            #zl_arr, theta_E_r_arr, theta_ap_r_arr, sigma_ap_arr, dd_arr, abs_delta_sigma_ap_arr, abs_delta_dd_arr = self.process_subtable(sub_table)
            if self.mode == 'linear':
                print('Running linear fit')
                args = self.args_linear_fit(sub_table)
            else:
                args = self.process_subtable(sub_table)

            
            # initial guess for MCMC
            p0 = [np.random.normal(loc=self.x_ini, scale=1e-4, size=self.ndim) for _ in range(self.nwalkers)]
            
            # Set up the backend
            # Don't forget to clear it in case the file already exists
            filename = os.path.join(self.output_folder, f"mcmc_results{self.mode}.h5")
            backend = emcee.backends.HDFBackend(filename)
            backend.reset(self.nwalkers, self.ndim)
            
            with Pool(self.ncpu) as pool:
                # initial sampler
                if self.ndim in [1,4]:
                    if self.ndim == 1:
                        args = args[1:]
                    sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob_touse,
                                                #args=(theta_E_r_arr, theta_ap_r_arr, sigma_ap_arr, dd_arr, abs_delta_sigma_ap_arr, abs_delta_dd_arr),
                                                args = args,
                                                backend=backend)
                elif self.ndim == 2:
                    
                    sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob_touse,
                                args = args,
                                #(zl_arr,theta_E_r_arr, theta_ap_r_arr, sigma_ap_arr, dd_arr, abs_delta_sigma_ap_arr, abs_delta_dd_arr),
                                backend=backend)

                

            sampler.run_mcmc(p0, self.nsteps, progress=True)
        
            # only take one for each group of 10 for correlations between samples
            thin = 1
            
            if self.burnin is None:
                # autocorrelation check
                tau = sampler.get_autocorr_time()
                # Suggested burn-in
                self.burnin = int(2 * np.max(tau))
                # Suggested thinning
                thin = int(0.5 * np.min(tau))

            # Append the samples for this 'z_l' bin to the list of all samples
            all_samples.append(sampler.get_chain(discard=self.burnin, thin=thin))
            all_ln_probs.append(sampler.get_log_prob())

            # Process and plot the combined results for each 'z_l' bin here
            np.save(os.path.join(self.output_folder,f'all_samples{self.mode}.npy'),all_samples)
            #np.save(os.path.join(self.output_folder,f'all_ln_probs{self.mode}.npy'),all_ln_probs)

        self.all_samples = np.array(all_samples)
        self.all_ln_probs = np.array(all_ln_probs)

    def calculate_statistics(self):
        """
        Calculate median, standard deviation, MAD, and mean absolute deviation for a given column in MCMC samples.
        """
        median_value, mean_value,mad_value = [],[],[]
        for sample in self.all_samples:
            column_values = sample[:, 0]
            median_value.append(np.median(column_values))
            mean_value.append(np.mean(column_values))
            mad_value.append(np.median(np.abs(column_values - np.median(column_values))))
            # mean_ad_value.append(np.mean(np.abs(column_values - median_value)))

        if (self.bin_width is None) and (self.elements_per_bin is None):
            self.lens_table[f'Gamma_median_{self.model}'] = median_value
            self.lens_table[f'Gamma_MAD_{self.model}'] = mad_value
            self.lens_table.write(self.output_table, overwrite =True)
        else:
            results_table = self.create_out_table( median_value, mean_value,mad_value)
            results_table.write(self.output_table, overwrite =True)


        print(f'saved results in {self.output_folder} ')
        return median_value, mean_value, mad_value

    def plot_hist_bins(self):
        # Plot the histogram
        plt.hist(self.lens_table['zl'], bins=self.bin_edges, edgecolor='k', alpha=0.7)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram with Fixed Number of Elements per Bin')
        plt.savefig(os.path.join(self.output_folder,'hist_zl.png'), transparent=False, facecolor='white', bbox_inches='tight')
        plt.show(block=False)

    def plot_post_prob(self):

        median_x, mean_x, mad_x = self.calculate_statistics()
        
        # Define the number of subplots per row
        subplots_per_row = 4
        #num_rows = len(all_samples) // subplots_per_row
        num_samples = len(self.all_samples)
        num_rows = (num_samples + subplots_per_row - 1) // subplots_per_row

        # Calculate dynamic figsize based on the number of rows
        figsize_per_row = 5  # You can adjust this factor based on your preference
        figsize = (figsize_per_row*subplots_per_row, num_rows * figsize_per_row)

        # Create a new figure and axis objects for subplots with only the required number of subplots
        fig, axes = plt.subplots(num_rows, subplots_per_row, figsize=figsize)

        # Iterate over the MCMC samples and create subplots
        for i, samples in enumerate(self.all_samples):
            row_idx = i // subplots_per_row  # Calculate the row index
            col_idx = i % subplots_per_row   # Calculate the column index

            ax = axes[row_idx, col_idx]  # Get the current subplot axis

            hist_gamma, _, _ = ax.hist(samples[:, 0], bins=30, alpha=0.5, color='b', density=True, label='Posterior Distribution')
            ax.axvline(x=median_x[i], c='r', ls='--', lw=2.0, label='Median')
            ax.axvline(x=mean_x[i], c='g', ls='--', lw=2.0, label='Mean', alpha=0.7)
            #ax.fill_betweenx([0, 1], lower_x, upper_x, color='g', alpha=0.3, label='68% Confidence Interval')
            ax.fill_betweenx([0, max(hist_gamma)], median_x[i] - 1.4826*mad_x[i], median_x[i] + 1.4826*mad_x[i], color='g', alpha=0.3, label='MADN Region')
            ax.set_xlabel('x')
            ax.set_ylabel('Probability Density')
            #ax.set_title(f'z_l Bin {z_bins[i]:.3f} - {z_bins[i-1]:.3f} #elem {hist[i]}')
            if self.binned:
                ax.set_title(f'z_l Bin centered at {self.xcenters[i]:.3f} #elem {self.hist[i]}')

        # Remove empty subplots (if any)
        for i in range(len(self.all_samples), num_rows * subplots_per_row):
            fig.delaxes(axes.flatten()[i])

        # Create a common legend for all subplots
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=2)

        # Adjust spacing between subplots
        plt.tight_layout()

        if self.bin_width is not None:
            plt.suptitle(f'{self.model} fixed {self.bin_width}  bin',y=1.05, fontsize = 16)
        if self.elements_per_bin is not None:
            plt.suptitle(f'{self.model} adaptive # {self.elements_per_bin} per bin',y=1.05, fontsize = 16)
        plt.savefig(os.path.join(self.output_folder,'posterior_distribution.png'), transparent=False, facecolor='white', bbox_inches='tight' )
        plt.show(block=False)
        
    def load_samples_and_ln_probs(self):
        """
        Load MCMC samples and ln_probs from the specified folder path.

        Parameters:
        - folder_path: str, path to the folder containing the samples and ln_probs files.

        Returns:
        - all_samples: list of arrays, MCMC samples for each 'z_l' bin.
        - all_ln_probs: list of arrays, ln_probs for each 'z_l' bin.
        """
        samples_file_path = os.path.join(self.output_folder, f'all_samples{self.mode}.npy')
        ln_probs_file_path = os.path.join(self.output_folder, f'all_ln_probs{self.mode}.npy')

        if os.path.exists(samples_file_path): #and os.path.exists(ln_probs_file_path):
            print(f'load data from  {samples_file_path}')
            all_samples = np.load(samples_file_path, allow_pickle=True)
            #all_ln_probs = np.load(ln_probs_file_path, allow_pickle=True)

            return all_samples#, all_ln_probs
        else:
            raise FileNotFoundError(f"Files not found in the specified folder: {self.output_folder}")


    def plot_results(self):
    
        # Data
        if self.binned:
            x = self.xcenters
        else: 
            x = self.lens_table['zl']

        y, _, y_err = self.calculate_statistics()
    
        if self.bin_width is not None:
            label = f'{self.model} fixed {self.bin_width}  bin'
        if self.elements_per_bin is not None:
             label = f'{self.model} adaptive # {self.elements_per_bin} per bin'
        else:
            label = f'{self.model} singular element'


        plot_point_with_fit(x, y, y_err, 
            x_label='z$_L$',
            y_label = '$\gamma$',
            plot_name = 'linear_fit_gamma.png',
            label = '', 
            output_folder=self.output_folder)


    def main(self):
        
        #run mcmc or use stored results
        if self.all_samples is None:
            self.run_mcmc()

        # different plottings
        if self.mode in ['linear' , 'direct']: 
            param_names = [r'\gamma_0', r'\gamma_1']
            self.all_samples = np.squeeze(self.all_samples)
            plot_directfit(self.all_samples, param_names, output_folder = self.output_folder)

        elif  self.mode == 'Koopmans_beta':
            param_names = [r'$\gamma_0$', r'$\gamma_1$',r'$\delta_0$',r'$\delta_1$']
            
        else:
            self.plot_post_prob()
            self.plot_results()

# Example usage:
if __name__ == "__main__":

    path_project = '/home/grespanm/github/SLcosmological_parameters/SGL_gamma'
    lens_table_path = os.path.join(path_project, 'Data' , 'LensTable02.fits')

    #fixed bins
    mcmc_instance_binned = MCMC(lens_table_path, path_project=path_project , model='ANN', bin_width=0.1)
    mcmc_instance_binned.main()
