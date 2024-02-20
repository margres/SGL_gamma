from rec_hz_gp import GP
from rec_dd_ann import ANN
from rec_gamma_mcmc import MCMC
import os
from rec_fsolve import add_fsolve_table
from combined_gamma import combined_dd, lenstable_cut_by_z


# Get the path of the current script
script_path = os.path.abspath(__file__)
# Get the parent directory of the script
path_project =  os.path.dirname(os.path.dirname(script_path))

print(f'Path in which your output will be saved {path_project}')


lens_table_path = os.path.join(path_project, 'Data' , 'SGLTable.fits')
nwalkers = 200
nsteps = 20000
#if true runs usinf the checkpoint system
checkpoint = True
ncpu = None
wmean = False
table_CC = 'Hz-34.txt'
cut_table = False
name_model_list = ['GP', 'ANN']

# shortens the table used for mcmc accordinly to max('zs') of the table_cc
if cut_table:
    lens_table_path = lenstable_cut_by_z(lens_table_path, table_CC, return_tab_path=True)


## run the GP
print( '\n ************** running the GP reconstruction ************** \n')
GP = GP(lens_table_path =lens_table_path, path_project=path_project, table_CC= table_CC )
GP.main()
add_fsolve_table(path_project , GP.output_table)
print('Done! \n')


#### run the ann 
print(' \n  ************** running the ANN reconstruction ************** ' )
ANN = ANN(path_project=path_project, 
        lens_table_path=lens_table_path)
ANN.main()
add_fsolve_table(path_project , ANN.output_table)
print('Done! \n')

print('Calculating weighted mean of dd from GP and ANN')

if wmean:
    combined_tab = combined_dd(GP.output_table, ANN.output_table, 
                output_folder= os.path.join(path_project, 'Output', 'Combined_dd' ),
                return_table_path=True)


table_list = [GP.output_table, ANN.output_table ]


for name_model, table  in zip(name_model_list, table_list) :
    if  name_model =='GP':
        color_points = '#d00000'
    if  name_model=='ANN':
        color_points = 'royalblue'
    elif name_model=='wmean':
        color_points = '#e3b23c'
    
    print (f' \n  ************** MCMC for {name_model} ************** \n ')

    print(f" \n ************** gamma from {table} for every value ************** \n")
    ## run the mcmc 
    mcmc = MCMC(lens_table_path = table , path_project = path_project, 
                model=name_model, nwalkers = nwalkers, nsteps = nsteps,
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points)
    mcmc.main()
    print('Done! \n')
    
    print(f" \n ************** gamma from {table} for fixed bins ************** \n")
    ## run the mcmc for fixed bins
    mcmc_instance_binned = MCMC(lens_table_path = table , model=name_model, bin_width=0.1,
                            path_project = path_project, nwalkers = nwalkers, nsteps = nsteps,
                            checkpoint=checkpoint, ncpu=ncpu, color_points=color_points)
    mcmc_instance_binned.main()

    print('Done! \n')

    print(f" \n ************** gamma from {table} for adaptive bins ************** \n")
    ## run the mcmc for fixed bins
    mcmc_instance_binned = MCMC(lens_table_path = table , model=name_model,
                            elements_per_bin = 15,
                            path_project = path_project, nwalkers = nwalkers, nsteps = nsteps,
                            checkpoint=checkpoint, ncpu=ncpu, color_points=color_points)
    mcmc_instance_binned.main()

    print('Done! \n')

    print(' \n  ************** gamma direct fit ************** ' )
    mcmc_direct = MCMC(lens_table_path = mcmc.output_table , 
                    path_project =path_project, 
                    model=name_model, nwalkers = nwalkers, nsteps = nsteps, mode='direct',  x_ini=[2.0, 0],
                    checkpoint=checkpoint, ncpu=ncpu, color_points=color_points)
    mcmc_direct.main()
    print('Done! \n')


    print(' \n  ************** gamma linear fit ************** ' )
    mcmc_linear = MCMC(lens_table_path =  mcmc.output_table   ,
                    path_project=path_project, 
                model=name_model, nwalkers = nwalkers, nsteps = nsteps, mode='linear', x_ini=[2.0, 0],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points)
    mcmc_linear.main()
    print('Done! \n')


            
    print(' \n  ************** Koopmans power law 2d fixed beta ************** ' ) 
    mcmc_K_beta = MCMC(lens_table_path = mcmc.output_table ,
                    path_project=path_project,
                model=name_model, nwalkers=nwalkers, nsteps = nsteps, mode='Koopmans_2D',  x_ini= [2.0,2.0],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points)
    mcmc_K_beta.main()    
    print('Done! \n')
            
    print(' \n  ************** Koopmans power law 3d  ************** ' ) 
    mcmc_K_beta = MCMC(lens_table_path = mcmc.output_table ,
                    path_project=path_project,
                model=name_model, nwalkers=nwalkers, nsteps = nsteps, mode='Koopmans_3D',  x_ini= [2.0,2.0,0.],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points)
    mcmc_K_beta.main()    
    print('Done! \n')


        
    print(' \n  ************** Koopmans power law 4d fixed beta ************** ' ) 
    mcmc_K_beta = MCMC(lens_table_path = mcmc.output_table ,
                    path_project=path_project,
                model=name_model, nwalkers=400, nsteps = nsteps, mode='Koopmans_4D',  x_ini= [2.0,0.0,2.0,0.0],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points)
    mcmc_K_beta.main()    
    print('Done! \n')

    print(' \n  ************** Koopmans power law 5d ************** ' )    
    mcmc_K = MCMC(lens_table_path = mcmc.output_table ,
                    path_project=path_project,
                model=name_model, nwalkers=400, nsteps = 6000, mode='Koopmans_5D',  x_ini= [2.0, 0.0, 2.0, 0.0, 0.0],
                checkpoint=checkpoint, ncpu=ncpu, color_points=color_points)
    mcmc_K.main()
    print('Done! \n')

