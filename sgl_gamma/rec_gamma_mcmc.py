import emcee
import os
import numpy as np
from scipy import integrate
import pandas as pd
import time
from astropy import units as u
from astropy import constants as const
import matplotlib.pyplot as plt
from datetime import datetime
from mcmc_utils import lnprob, lnprobdirect, lnproblinear, lnprob_K_5D, lnprob_K_4D, lnprob_K_3D, lnprob_K_2D, lnprob_K_5D_log, lnprob_K_5D_scale
import random
from multiprocessing import Pool
from utils_plot import plot_point_with_fit, plot_GetDist,plot_hist_bins
import seaborn as sns
from astropy.table import Table


seed = 42
np.random.seed(seed)

class MCMC:

    def __init__(self, lens_table_path, path_project, 
                 output_folder = None, 
                 model='',
                 mode = '',
                 bin_width=None, elements_per_bin=None, nsteps=5000, 
                 nwalkers=500, ndim=None, ncpu=None,  burnin = None,
                 all_ln_probs=None,all_samples=None,
                 lnprob_touse = None, x_ini=[2],
                 nsteps_per_checkpoint = 1000,
                 checkpoint=True,
                 param_fit = 'zl',
                 force_run = False,
                 force_plots = True,
                 color_points = 'firebrick',
                 model_name_out = None,
                 column_sigma_ap = 'sigma_ap',
                 column_theta_ap  = 'theta_ap',
                 save_outputs = True,
                 run_plots = True
                 ):
        
        self.model = model

        if model_name_out is None:
            self.model_name_out = model
        else:
            self.model_name_out = model_name_out

        self.mode = mode
        self.lens_table_path = lens_table_path
        self.bin_width = bin_width
        self.elements_per_bin = elements_per_bin
        self.nsteps = nsteps
        self.nwalkers = nwalkers
        self.x_ini = x_ini
        self.nsteps_per_checkpoint = nsteps_per_checkpoint
        self.checkpoint = checkpoint
        self.param_fit = param_fit
        self.chain_info_list = []
        self.force_plots = force_plots
        self.color_points = color_points
        self.column_sigma_ap = column_sigma_ap
        self.column_theta_ap = column_theta_ap
        self.save_outputs = save_outputs
        self.run_plots =  run_plots

        # Load lens table
        format = self.lens_table_path.split('.')[-1]
        self.lens_table = Table.read(self.lens_table_path, format=format)
        print(f'Table has {len(self.lens_table)} elements')
        self.binned = True
        self.output_folder = output_folder


        if ndim is None:
            self.ndim = len(self.x_ini)
        else:
            self.ndim = ndim
    
        if ncpu is None:   
            self.ncpu = os.cpu_count()
            print(f'ncpu not specified, using all the {self.ncpu} cpu ')
        else:
            self.ncpu = ncpu
        if lnprob_touse is None or lnprob_touse=='':
            if mode=='linear':
                self.lnprob_touse = lnproblinear
            elif mode == 'direct':
                self.lnprob_touse = lnprobdirect
            elif mode =='1D':
                self.lnprob_touse = lnprob
            elif mode == 'Koopmans_2D':
                self.lnprob_touse = lnprob_K_2D
            elif mode == 'Koopmans_3D':
                self.lnprob_touse = lnprob_K_3D
            elif mode == 'Koopmans_4D':
                self.lnprob_touse = lnprob_K_4D
            elif mode == 'Koopmans_5D':
                self.lnprob_touse = lnprob_K_5D
            else:
                raise ValueError(f'mode {mode} not available ')
        else: 
            self.lnprob_touse = lnprob_touse
            print('Using the given lnbprob')
        
        if burnin is None:
            self.burnin = int(nsteps*0.1)
        else:
            self.burnin = burnin

        if model not in ['ANN', 'GP', 'wmean']:
            raise ValueError('model not known, only available ANN or GP')
        
        if ('GP' in model) and (mode!='linear'):
            # there are no values for some lenses 
            self.lens_table = self.lens_table[(self.lens_table['dd_GP']>0)] 
            print(f" Using table with {len(self.lens_table)} values")
        # Constants
        #self.c = c.to('km/s').value
        
        if (bin_width is not None) and (elements_per_bin is not None):
            raise ValueError("Only one of bin_width or elements_per_bin should be defined, not both.")
            
        if (bin_width is None) and (elements_per_bin is None) and  self.mode =='1D':
            print('No binning  - mcmc for every element')
            self.binned = False
            self.mode = 'singular'
        elif (bin_width is None) and (elements_per_bin is None) and  self.mode !='1D' :
            self.binned = False
           
        if (bin_width is not None) and self.mode =='1D' :
            #fixed
            self.mode = 'fixed'
            a = 1./ (1 + self.lens_table['zl'])
            bins = np.arange(min(a), max(a)+ bin_width, self.bin_width)
            bin_edges = 1/bins-1
            self.bin_edges = bin_edges[::-1]
            '''
            min_z_l = self.lens_table['zl'].min()
            max_z_l = self.lens_table['zl'].max()
            self.bin_edges = np.arange(min_z_l, max_z_l + self.bin_width, self.bin_width)
            '''

        if (elements_per_bin is not None) and self.mode =='1D':
            #adaptive
            self.mode = 'adaptive'
            self.bin_edges = np.percentile(self.lens_table['zl'], np.linspace(0, 100, self.elements_per_bin + 1))

        if self.output_folder is None:
            self.output_folder = os.path.join(path_project, 'Output',
                            f"{self.model_name_out}_gamma-{self.param_fit}_{self.mode}_nw_{nwalkers}_ns_{nsteps}")

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
        elif force_run:
            self.all_samples, self.all_ln_probs = all_samples, all_ln_probs
        else:
            try:
                self.all_samples =  self.load_samples_and_ln_probs()
            except Exception as e:
                print(e)
                self.all_samples, self.all_ln_probs = all_samples, all_ln_probs

        self.output_table = os.path.join(self.output_folder, f'SGL_{self.model}_gammaMCMC.csv')

        if self.param_fit== 'theta_Edivtheta_eff':
            self.lens_table['theta_Edivtheta_eff'] = self.lens_table['theta_E'] / self.lens_table['theta_eff']
            self.lens_table.to_csv(self.output_table, index=False)
    

    def create_out_table(self, median, mean,mad):
        #print(len(median),len(mean),len(mad),len(self.xcenters),len(self.bin_edges[:-1]),len(self.bin_edges[1:]),len(self.hist))
        # Create Astropy table

        non_zero_rows = self.hist != 0
        #print(non_zero_rows)

        data = {
            f'Gamma_median_{self.model}': median,
            f'Gamma_mean_{self.model}': mean,
            f'Gamma_MAD_{self.model}': mad,
            'bin_center': self.xcenters,
            'bin_min_edge': self.bin_edges[:-1][non_zero_rows],
            'bin_max_edge': self.bin_edges[1:][non_zero_rows],
            'n_elem': self.hist[non_zero_rows]
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
    
    def get_parameters(self,row,zl_list, theta_E_r_list, theta_ap_r_list, sigma_ap_list, dd_list, abs_delta_sigma_ap_list, abs_delta_dd_list ):

        zl = row['zl']
        theta_E_r = row['theta_E'] * u.arcsec.to('radian')
        theta_ap_r = row[self.column_theta_ap] * u.arcsec.to('radian')
        sigma_ap = row[self.column_sigma_ap]
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

        return zl_list, theta_E_r_list, theta_ap_r_list, sigma_ap_list, dd_list, abs_delta_sigma_ap_list, abs_delta_dd_list
        
    def process_subtable(self, sub_table):
    
        # Process each subtable
        zl_list, theta_E_r_list, theta_ap_r_list, sigma_ap_list, dd_list, abs_delta_sigma_ap_list, abs_delta_dd_list = [],[], [], [], [], [], []
        
        #print(np.shape(sub_table))
        if len(np.shape(sub_table))==0:
            sub_table =[sub_table]

        for row in sub_table:
                zl_list, theta_E_r_list, theta_ap_r_list, sigma_ap_list, dd_list, abs_delta_sigma_ap_list, abs_delta_dd_list = self.get_parameters(row,zl_list, theta_E_r_list, theta_ap_r_list, sigma_ap_list, dd_list, abs_delta_sigma_ap_list, abs_delta_dd_list) 

        return np.array(zl_list),np.array(theta_E_r_list),np.array(theta_ap_r_list),np.array(sigma_ap_list),np.array(dd_list),np.array(abs_delta_sigma_ap_list),np.array(abs_delta_dd_list)

    def args_linear_fit(self, sub_table):
        
        param_list, gamma_list, gamma_mad_list = [], [],[]

        if len(np.shape(sub_table))==0:
            sub_table =[sub_table]
        
        for row in sub_table:
            #in the most cases this is z
            try:
                param = row[self.param_fit]
            except KeyError:
                print (f'{self.param_fit} parameter not available')
                # first two are index and lens name
                print(f' The parameter available are {print(sub_table.colnames[2:])} and theta_E/theta_eff')
                return
            gamma = row[f'Gamma_median_{self.model}'] 
            gamma_mad = row[f'Gamma_MAD_{self.model}']



            # Append parameter values for this row to lists
            param_list.append(param)
            gamma_list.append(gamma)
            gamma_mad_list.append(gamma_mad)

        return np.array(param_list),np.array(gamma_list),np.array(gamma_mad_list)
    
    def run_mcmc(self,):

        print('starting the mcmc ... \n')
        
        # Initialize lists to accumulate samples
        all_samples = []
        
        if self.binned:
            
            plot_hist_bins(self.lens_table, self.bin_edges, 
                   self.color_points, self.output_folder)
            subtables = self.get_subtable()
            
        else:
            print(f'Number of elements in the table {len(self.lens_table)}')
            subtables = self.lens_table
            if 'Koopmans' in self.mode or len(self.x_ini)>2:
                subtables =  [subtables]

        if self.mode in ['linear', 'direct' ]:
            subtables = [subtables]

        for sub_table in subtables:
            
            if len(sub_table)<1:
                continue 

            #print(len(sub_table['zl']))
            
            #zl_arr, theta_E_r_arr, theta_ap_r_arr, sigma_ap_arr, dd_arr, abs_delta_sigma_ap_arr, abs_delta_dd_arr = self.process_subtable(sub_table)
            if self.mode == 'linear':
                args = self.args_linear_fit(sub_table)
            else:
                args = self.process_subtable(sub_table)

             
            # initial guess for MCMC
            p0 = [np.random.normal(loc=self.x_ini, scale=1e-4, size=self.ndim) for _ in range(self.nwalkers)]
            
            # Set up the backend
            # Don't forget to clear it in case the file already exists
            if self.mode=='linear':
                filename = os.path.join(self.output_folder, f"mcmc_chain_linearfit.h5")
            else:
                filename = os.path.join(self.output_folder, f"mcmc_chain.h5")

            backend = emcee.backends.HDFBackend(filename)
            backend.reset(self.nwalkers, self.ndim)
            
            with Pool(self.ncpu) as pool:
                # initial sampler
                if self.ndim in [1,2,3,4,5]:
                    if self.ndim == 1:
                        args = args[1:]
                    sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob_touse,
                                                #args=(theta_E_r_arr, theta_ap_r_arr, sigma_ap_arr, dd_arr, abs_delta_sigma_ap_arr, abs_delta_dd_arr),
                                                args = args,
                                                pool=pool,
                                                backend=backend,
                                               )
                elif self.ndim == 2 and self.mode in ['linear', 'direct']:
                    
                    sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob_touse,
                                args = args,
                                pool=pool,
                                #(zl_arr,theta_E_r_arr, theta_ap_r_arr, sigma_ap_arr, dd_arr, abs_delta_sigma_ap_arr, abs_delta_dd_arr),
                                backend=backend,
                               )
                else:
                    raise ValueError('no sampler available for this number of x_ini')
                
                if self.checkpoint:   
                    print(f'Running with checkpoints every {self.nsteps_per_checkpoint} ') 
                    # Checkpoint system
                    for _ in range(0, self.nsteps, self.nsteps_per_checkpoint):
                        sampler.run_mcmc(p0, self.nsteps_per_checkpoint, store=True,progress=True)

                        # Evaluate the current state of the chain
                        try:
                            tau = sampler.get_autocorr_time(tol=0)
                            self.burnin = int(2 * np.max(tau))
                            thin = 1
                            print(f"Checkpoint: tau={tau}, burnin={self.burnin}, thin={thin}")

                            # If chain is long enough, break
                            if sampler.iteration / np.max(tau) > 50:
                                print("Sufficient chain length achieved. Stopping run.")
                                break
                        except emcee.autocorr.AutocorrError as e:
                            # Handle the case where autocorrelation time can't be reliably estimated
                            print(str(e))
                        p0 = sampler.get_last_sample()

                else:
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

                chain_info = {
                    'nsteps': len(sampler.get_chain()),
                    'nsteps_without_burn-in': len(sampler.get_chain(discard=self.burnin, thin=thin)),
                    'burn-in': self.burnin,
                    'thin': thin}
                
                self.chain_info_list.append(chain_info)

                if self.save_outputs:
                    pd.DataFrame(self.chain_info_list).to_csv(os.path.join(self.output_folder,'mcmc_chain_log.csv' ))

                # Append the samples for this 'z_l' bin to the list of all samples
                all_samples.append(sampler.get_chain(discard=self.burnin, thin=thin))
                #all_ln_probs.append(sampler.get_log_prob())
        if self.save_outputs:
            # Process and plot the combined results for each 'z_l' bin here
            np.save(os.path.join(self.output_folder,f'{self.mode}_mcmc_samples.npy'),all_samples)
        #np.save(os.path.join(self.output_folder,f'all_ln_probs{self.mode}.npy'),all_ln_probs)
        self.all_samples = np.array(all_samples)
        #elf.all_ln_probs = np.array(all_ln_probs)
        print('mcmc finished! \n ')

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
            self.lens_table.write(self.output_table, format='csv', overwrite=True)
        else:
            results_table = self.create_out_table( median_value, mean_value,mad_value)
            results_table.write(self.output_table, format='csv', overwrite=True)

        return median_value, mean_value, mad_value

    def plot_post_prob(self, output_folder, plot_name = 'posterior_distribution.png', color_points= None):

        if color_points is None:
            color_points = self.color_points

        sns.set(style="whitegrid")
        median_x, mean_x, mad_x = self.calculate_statistics()
        
        num_samples = len(self.all_samples)
        
        # Define the number of subplots per row
        if num_samples>4:
            subplots_per_row = 4
        else: 
            subplots_per_row = int(num_samples/2)
        #num_rows = len(all_samples) // subplots_per_row
       
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

            hist_gamma, _, _ = ax.hist(samples[:, 0], bins=30, alpha=0.7, color='b', density=True, label='Posterior Distribution')
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
            plt.suptitle(f'{self.model} fixed, bin width = {self.bin_width}  bin',y=1.05, fontsize = 16)
        elif self.elements_per_bin is not None:
            plt.suptitle(f'{self.model} adaptive # {self.elements_per_bin} per bin',y=1.05, fontsize = 16)
    
            plt.savefig(os.path.join(output_folder, plot_name), transparent=False, facecolor='white', bbox_inches='tight' )
            plt.close()
        
    def load_samples_and_ln_probs(self):

        """
        Load MCMC samples and ln_probs from the specified folder path.

        Parameters:
        - folder_path: str, path to the folder containing the samples and ln_probs files.

        Returns:
        - all_samples: list of arrays, MCMC samples for each 'z_l' bin.
        - all_ln_probs: list of arrays, ln_probs for each 'z_l' bin.
        """

        samples_file_path = os.path.join(self.output_folder, f'{self.mode}_mcmc_samples.npy')
        #ln_probs_file_path = os.path.join(self.output_folder, f'all_ln_probs{self.mode}.npy')

        if os.path.exists(samples_file_path): #and os.path.exists(ln_probs_file_path):
            print(f'load data from  {samples_file_path}')
            all_samples = np.load(samples_file_path, allow_pickle=True)
            #all_ln_probs = np.load(ln_probs_file_path, allow_pickle=True)

            return all_samples#, all_ln_probs
        else:
            raise FileNotFoundError(f"File {samples_file_path} not found")


    def plot_results(self, plot_name= 'linear_fit_gamma.png'):
    
        # Data
        if self.binned:
            x = self.xcenters
        else: 
            x = self.lens_table['zl']

        y, _, y_err = self.calculate_statistics()

        if self.bin_width is not None:
            label = f'bin fixed width = {self.bin_width}'
        elif self.elements_per_bin is not None:
            label = f'adaptive, $\sim$ {self.elements_per_bin} SGL per bin'
        else:
            label = f'{self.model} SGL'
       


        plot_point_with_fit(x, y, y_err, 
            x_label='z',
            y_label = '$\gamma$',
            plot_name = plot_name,
            label = label, 
            output_folder=self.output_folder,
            title = self.model
            )

    def path_exists(self, output_folder, file_name):
        os.path.exists(os.path.join(output_folder , file_name))


    def main(self):
        
        #run mcmc or use stored results`    `
        if self.all_samples is None:
            self.run_mcmc()

        self.all_samples = np.squeeze(self.all_samples)
        plot_name = f'Posterior_Dist_{self.mode}.png' 
        if self.run_plots:
            # different plottings
            if (self.mode in ['linear', 'direct']) and (not self.path_exists(self.output_folder,  plot_name ) or self.force_plots):
                param_labels = [r'\gamma_0', r'\gamma_S']
                plot_GetDist(np.squeeze(self.all_samples), param_labels, output_folder = self.output_folder,  plot_name=plot_name)

            elif (self.mode == 'Koopmans_2D') and (not self.path_exists(self.output_folder, plot_name) or self.force_plots):

                param_labels = [r'\gamma',r'\delta']
                plot_GetDist(np.squeeze(self.all_samples), param_labels, output_folder = self.output_folder,  plot_name=plot_name)

            elif (self.mode == 'Koopmans_3D') and (not self.path_exists(self.output_folder,  plot_name) or self.force_plots):

                param_labels = [r'\gamma',r'\delta', r'\beta']
                plot_GetDist(np.squeeze(self.all_samples), param_labels, output_folder = self.output_folder,  plot_name=plot_name)

            elif (self.mode == 'Koopmans_4D') and (not self.path_exists(self.output_folder,  plot_name ) or self.force_plots):

                param_labels = [r'\gamma_0', r'\gamma_S',r'\delta_0',r'\delta_S']
                plot_GetDist(np.squeeze(self.all_samples), param_labels, output_folder = self.output_folder,  plot_name=plot_name)

            elif (self.mode =='Koopmans_5D') and (not self.path_exists(self.output_folder,  plot_name) or self.force_plots):

                param_labels = [r'\gamma_0', r'\gamma_S',r'\delta_0',r'\delta_S',r'\beta']
                plot_GetDist(np.squeeze(self.all_samples), param_labels, output_folder = self.output_folder,  plot_name=plot_name)

            else:
            
                if (not self.path_exists(self.output_folder,plot_name) or self.force_plots):
                    self.plot_post_prob(output_folder = self.output_folder,  plot_name =  plot_name)

                if (not self.path_exists(self.output_folder, 'linear_fit_gamma.png') or self.force_plots):
                    self.plot_results(  plot_name = 'linear_fit_gamma.png')

