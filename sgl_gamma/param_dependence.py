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
nsteps = 20000
mode='linear'
checkpoint = True
# name_model  = 'GP'
name_model  = 'ANN'

if name_model =='GP':
    ## run the GP
    print( '\n ************** running the GP reconstruction ************** \n')
    GP = GP(lens_table_path =lens_table_path, path_project=path_project)
    #GP.main()
    table =GP.output_table 
    color_points = '#d00000'
elif name_model=='ANN':
    ## run the ANN
    print( '\n ************** running the ANN reconstruction ************** \n')
    ANN = ANN(path_project=path_project, lens_table_path=lens_table_path)
    #ANN.main()
    table = ANN.output_table 
    color_points  = '#0e79b2'

## run the mcmc 
mcmc = MCMC(lens_table_path = table , path_project = path_project, 
            model=name_model, nwalkers = nwalkers, nsteps = nsteps,
            checkpoint=checkpoint)
mcmc.main()


y_label = f'median $\gamma$ {name_model}'

y_name = f'Gamma_median_{name_model}'
y_err_name = f'Gamma_MAD_{name_model}'

parameters_list = ['theta_eff',  'sigma_ap', 'theta_Edivtheta_eff']
labels = ['$\theta_{eff}$',  '$\sigma_{ap}$',  '$\theta_{E}/\theta_{eff}$']

for param, lab in zip(parameters_list, labels):

    #path_out = os.path.join(path_project,'Output', f'Gamma_LinearFit_{name_model}_{param}' )

    print(f' \n  ************** {param} ************** ' )

    mcmc_linear = MCMC(lens_table_path =  mcmc.output_table   ,
                    path_project=path_project,
                    param_fit = param, 
                model=name_model, nwalkers = nwalkers, nsteps = nsteps, mode='linear', x_ini=[2.0, 0],
                checkpoint=checkpoint)
    mcmc_linear.main()
    print('Done! \n')


    linear_results_path = os.path.join(path_project, 'Output/',
                  f'{name_model}_gamma-{param}_{mode}_nw_{nwalkers}_ns_{nsteps}','median_mad_posterior_dist.csv')
    
    linear_results = Table.read(linear_results_path)
    m = linear_results['median'][1]
    b = linear_results['median'][0]

    x= mcmc_linear.lens_table[param]
    y= mcmc_linear.lens_table[y_name]
    y_err= mcmc_linear.lens_table[y_err_name]
    
    plot_point_with_fit(x, y, y_err, 
    x_label= lab,label=f'{name_model} SGL',
    y_label = y_label,correlation=True,
    plot_name = f'mcmc_linear_fit_{param}-{y_label}.png', 
    output_folder=os.path.dirname(linear_results_path),
    color_points = color_points,
    m=m,b=b)

    plot_point_with_fit(x, y, y_err, 
    x_label= lab,label=f'{name_model} SGL',
    y_label = y_label, correlation=True,
    plot_name = f'linear_fit_{param}-{y_label}.png', 
    color_points = color_points,
    output_folder=os.path.dirname(linear_results_path))

