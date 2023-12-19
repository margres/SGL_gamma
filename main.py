from rec_hz_gp import GP
from rec_dd_ann import ANN
from rec_gamma_mcmc import MCMC
import os
from rec_fsolve import add_fsolve_table
from combined_gamma import combined_dd

path_project='/home/grespanm/github/SLcosmological_parameters/SGL_gamma/'
lens_table_path = os.path.join(path_project, 'Data' , 'SGLTable.fits')
nwalkers = 50
nsteps = 100


## run the GP
print( '\n ************** running the GP reconstruction ************** \n')
GP = GP(lens_table_path =lens_table_path, path_project=path_project)
#GP.main()
#add_fsolve_table(path_project , GP.output_table)
print('Done! \n')

#### run the ann 
print(' \n  ************** running the ANN reconstruction ************** ' )
ANN = ANN(path_project=path_project, lens_table_path=lens_table_path)
#ANN.main()
#add_fsolve_table(path_project , GP.output_table)
print('Done! \n')

print('Calculating weighted mean of dd from GP and ANN')

combined_dd(GP.output_table, ANN.output_table, 
            output_folder= os.path.join(path_project, 'Output', 'Combined_dd' ) )

for name_model, table  in zip(['GP', 'ANN'],[GP.output_table, ANN.output_table ]) :
    
    print (f' \n  ************** MCMC for {name_model} ************** \n ')

    print(f" \n ************** gamma from {table} for every value ************** \n")
    ## run the mcmc 
    mcmc = MCMC(lens_table_path = table , path_project = path_project, 
                model=name_model, nwalkers = nwalkers, nsteps = nsteps)
    mcmc.main()
    print('Done! \n')
    
    print(f" \n ************** gamma from {table} for fixed bins ************** \n")
    ## run the mcmc for fixed bins
    mcmc_instance_binned = MCMC(lens_table_path = table , model=name_model, bin_width=0.1,
                            path_project = path_project, nwalkers = nwalkers, nsteps = nsteps)
    mcmc_instance_binned.main()

    print('Done! \n')

    print(f" \n ************** gamma from {table} for adaptive bins ************** \n")
    ## run the mcmc for fixed bins
    mcmc_instance_binned = MCMC(lens_table_path = table , model=name_model,
                            elements_per_bin = 15,
                            path_project = path_project, nwalkers = nwalkers, nsteps = nsteps)
    mcmc_instance_binned.main()

    print('Done! \n')

    print(' \n  ************** gamma direct fit ************** ' )
    mcmc_direct = MCMC(lens_table_path = mcmc.output_table , 
                    path_project =path_project,
                    output_folder = os.path.join(path_project, 'Output', f'Gamma_DirectFit_{name_model}' ), 
                model=name_model, nwalkers = nwalkers, nsteps = nsteps, mode='direct',  x_ini=[2.0, 0])
    mcmc_direct.main()
    print('Done! \n')


    print(' \n  ************** gamma linear fit ************** ' )
    mcmc_linear = MCMC(lens_table_path =  mcmc.output_table   ,
                    path_project=path_project,
                    output_folder = os.path.join(path_project,'Output', f'Gamma_LinearFit_{name_model}' ), 
                model=name_model, nwalkers = nwalkers, nsteps = nsteps, mode='linear', x_ini=[2.0, 0])
    mcmc_linear.main()
    print('Done! \n')

            
    print(' \n  ************** Koopmans power law 2d fixed beta ************** ' ) 
    mcmc_K_beta = MCMC(lens_table_path = mcmc.output_table ,
                    path_project=path_project,
                    output_folder = os.path.join(path_project,'Output', f'Gamma_Koopmans_2D_fixed_beta_{name_model}' ), 
                model=name_model, nwalkers=nwalkers, nsteps = nsteps, mode='Koopmans_2D',  x_ini= [2.0,2.0])
    mcmc_K_beta.main()    
    print('Done! \n')


            
    print(' \n  ************** Koopmans power law 3d  ************** ' ) 
    mcmc_K_beta = MCMC(lens_table_path = mcmc.output_table ,
                    path_project=path_project,
                    output_folder = os.path.join(path_project,'Output', f'Gamma_Koopmans_3D_{name_model}' ), 
                model=name_model, nwalkers=nwalkers, nsteps = nsteps, mode='Koopmans_3D',  x_ini= [2.0,2.0,0.])
    mcmc_K_beta.main()    
    print('Done! \n')


        
    print(' \n  ************** Koopmans power law 4d fixed beta ************** ' ) 
    mcmc_K_beta = MCMC(lens_table_path = mcmc.output_table ,
                    path_project=path_project,
                    output_folder = os.path.join(path_project,'Output', f'Gamma_Koopmans_4D_fixed_beta_{name_model}' ), 
                model=name_model, nwalkers=50, nsteps = nsteps, mode='Koopmans_4D',  x_ini= [2.0,0.0,2.0,0.0])
    mcmc_K_beta.main()    
    print('Done! \n')

    print(' \n  ************** Koopmans power law 5d ************** ' )    
    mcmc_K = MCMC(lens_table_path = mcmc.output_table ,
                    path_project=path_project,
                    output_folder = os.path.join(path_project,'Output', f'Gamma_Koopmans_5D_{name_model}' ), 
                model=name_model, nwalkers=50, nsteps = nsteps, mode='Koopmans_5D',  x_ini= [2.0, 0.0, 2.0, 0.0, 0.0])
    mcmc_K.main()
    print('Done! \n')

