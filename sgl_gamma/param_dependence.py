from rec_hz_gp import GP
from rec_dd_ann import ANN
from rec_gamma_mcmc import MCMC
import os
from rec_fsolve import add_fsolve_table
from combined_gamma import combined_dd
from utils_plot import  plot_point_with_fit
from astropy.table import Table


# Get the path of the current script
script_path = os.path.abspath(__file__)
# Get the parent directory of the script
path_project =  os.path.dirname(os.path.dirname(script_path))

print(f'Path in which your output will be saved {path_project}')

lens_table_path = os.path.join(path_project, 'Data' , 'SGLTable.fits')
nwalkers = 200
nsteps = None #4000
#if true runs usinf the checkpoint system
checkpoint = True

## run the GP
print( '\n ************** running the GP reconstruction ************** \n')
GP = GP(lens_table_path =lens_table_path, path_project=path_project)
GP.main()
#add_fsolve_table(path_project , GP.output_table)

name_model, table  = 'GP',GP.output_table 

## run the mcmc 
mcmc = MCMC(lens_table_path = table , path_project = path_project, 
            model=name_model, nwalkers = nwalkers, nsteps = nsteps,
            checkpoint=checkpoint)
mcmc.main()

y_label = f'Gamma_median_{name_model}'
y_label_err = f'Gamma_MAD_{name_model}'

parameters_list = ['theta_eff', 'theta_ap', 'sigma_ap', 'theta_E', 'theta_E/theta_eff']

for param in parameters_list:

    path_out = os.path.join(path_project,'Output', f'Gamma_LinearFit_{name_model}_{param}' )

    print(' \n  ************** gamma linear fit ************** ' )

    mcmc_linear = MCMC(lens_table_path =  mcmc.output_table   ,
                    path_project=path_project,
                    param_directfit = param, 
                    output_folder = path_out, 
                model=name_model, nwalkers = nwalkers, nsteps = nsteps, mode='linear', x_ini=[2.0, 0],
                checkpoint=checkpoint)
    mcmc_linear.main()
    print('Done! \n')


    linear_results_path = os.path.join('/home/grespanm/github/SLcosmological_parameters/SGL_gamma/Output/',
                  f'Gamma_LinearFit_GP_{param}','median_mad_posterior_dist.csv')
    
    linear_results = Table.read(linear_results_path)
    m = linear_results['median'][1]
    b = linear_results['median'][0]

    x, y, y_err= mcmc.lens_table[param], mcmc.lens_table[y_label], mcmc.lens_table[y_label_err]
    
    plot_point_with_fit(x, y, y_err, 
    x_label= param,
    y_label = y_label,
    plot_name = f'linear_fit_{param}-{y_label}.png', 
    output_folder=os.path.dirname(linear_results_path),
    m=m,b=b)


