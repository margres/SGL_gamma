from rec_hz_gp import GP
from rec_dd_ann import ANN
from rec_gamma_mcmc import MCMC
import os


path_project='/home/grespanm/github/SLcosmological_parameters/SGL_gamma/'
lens_table_path = os.path.join(path_project, 'Data' , 'LensTable02.fits')
## run the GP
print('  ************** running the GP reconstruction **************')
GP = GP(lens_table_path =lens_table_path, path_project=path_project)
#GP.main()
#print('Done!')

print(f"Using distance reconstruction from {GP.output_table}")
## run the mcmc for fixed bins for the GP results 
mcmc_gp = MCMC(lens_table_path=GP.output_table , path_project = path_project, 
            model='GP', nwalkers =50, nsteps = 100)
mcmc_gp.main()

if False:
    #### run the ann

    print('************** running the ANN reconstruction **************')
    ANN = ANN(path_project=path_project)
    ANN.main()
    print('Done!')
    

    #### combine the results in one table

    ANN.output_table

    ## run the mcmc for fixed bins fro the ANN results
    mcmc_instance_binned = MCMC(lens_table_path=ANN.output_table , model='ANN', bin_width=0.1)
    mcmc_instance_binned.main()


    print(f"Using distance reconstruction from {ANN.output_table}")
    ## run the mcmc for fixed bins for the GP results 
    mcmc_gp = MCMC(lens_table_path=ANN.output_table , path_project = path_project, 
                model='ANN', nwalkers =50, nsteps = 100)
    mcmc_gp.main()


    print(f"Using distance reconstruction from {GP.output_table}")
    ## run the mcmc for fixed bins for the GP results 
    mcmc_gp = MCMC(lens_table_path=GP.output_table , path_project = path_project, 
                model='GP', nwalkers =50, nsteps = 100)
    mcmc_gp.main()


mcmc_direct = MCMC(lens_table_path=GP.output_table , 
                   path_project=path_project,
                   output_folder = os.path.join(path_project, 'Output', 'Gamma_DirectFit' ), 
            model='GP', nwalkers =50, nsteps = 100, mode='direct',  ndim=2, x_ini=[2.0, 0])
mcmc_direct.main()

mcmc_linear = MCMC(lens_table_path=mcmc_gp.output_table ,
                   path_project=path_project,
                   output_folder = os.path.join(path_project,'Output', 'Gamma_LinearFit' ), 
            model='GP', nwalkers =50, nsteps = 100, mode='linear',  ndim=2, x_ini=[2.0, 0])
mcmc_linear.main()
