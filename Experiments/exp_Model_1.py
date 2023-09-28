import matplotlib.image as mpimg
import sys, os
import numpy as np
import datetime
import csv 


PI = np.pi
SIN = np.sin
COS = np.cos
EXP = np.exp
LOG = np.log


def zero_2d(x, y):
    return 0.


def zero_1d(x, y):
    return 0.


def f_func(x: float, y: float):
    if 1. / 8. < x < 7. / 8. and 3. / 8. < y < 5. / 8.:
        return 1.0
    elif 3. / 8. < x < 5. / 8. and 1. / 8. < y < 7. / 8.:
        return 1.0
    else:
        return 0.0

def g_func(x: float, y: float):
    return x**2 + EXP(x * y)


def g_dx_func(x: float, y: float):
    return 2.0 * x + y * EXP(x * y)


def g_dy_func(x: float, y: float):
    return x * EXP(x * y)

def beta_func(x:float,y:float):
    return COS(18*PI*y)*SIN(18*PI*x), -COS(18*PI*x)*SIN(18*PI*y)

root_path = os.path.join(sys.path[0], '..')
sys.path.append(root_path)

import Model1.InhomoDiriBVP as DBVP

import logging
from logging import config

result_foldername='exp_Model_1'
if not os.path.exists(result_foldername):
    os.makedirs(result_foldername)

log_filename=result_foldername+'/exp_Model_1.log'
csv_filename=result_foldername+'/results.csv'

if not os.path.exists(csv_filename):
    import pandas as pd
    csv_column_names=['contrast','lambda','coarse_grid','eigen_num','over_sampling','Dg_norm_a',\
                        'Dg_norm_l2', 'Dg_ref_norm_a', 'Dg_ref_norm_l2','Dg_error_a','Dg_error_l2',\
                        'u_norm_a','u_norm_l2', 'u_ref_norm_a','u_ref_norm_l2', 'u_error_a','u_error_l2']
                    
    data = {}
    df = pd.DataFrame(data, columns=csv_column_names)
    df.to_csv(csv_filename)
    del df

config.fileConfig(os.path.join(root_path, 'Settings', 'log.conf'), defaults={'logfilename': log_filename})


def get_coeff_from_tmp(coeff_tmp, ctr: float):
    coeff = np.ones(coeff_tmp.shape)
    dim_x, dim_y = coeff_tmp.shape
    for ind_y in range(dim_y):
        for ind_x in range(dim_x):
            if coeff_tmp[ind_x, ind_y] >= 2.0:
                coeff[ind_x, ind_y] = ctr
    return coeff

logging.info('=' * 80)
logging.info("Start")

coeff_tmp = np.load(os.path.join(root_path, 'Resources', 'MediumA.npy'))

SEC_NUM = 3
SUB_SEC_NUM = 4
EIGEN_NUM = 5
FINE_GRID = 400
logging.info("Start experimenting with Model 1")

for sub_sec_ind in range(1, SUB_SEC_NUM + 1):
    for eigen_num in range(1,EIGEN_NUM):
        for ctr_exp in [3,4,5,6]:
            for coarse_grid in [10,20,40,80]:
                parameters= {"os":sub_sec_ind,
                            "eig": eigen_num,
                            "ctr":ctr_exp,
                            "H":coarse_grid,}
                
                logging.info(f"Working on {'_'.join(f'{k}{v}' for k,v in parameters.items())}...")
                logging.info("Now setting up the initial parameters...")
                ctr = 10.0**ctr_exp
                dbvp = DBVP.ProblemSetting(coarse_grid=coarse_grid,fine_grid=FINE_GRID)
                # dbvp.set_coarse_grid(coarse_grid)
                # dbvp.set_fine_grid(FINE_GRID)
                coeff = get_coeff_from_tmp(coeff_tmp, ctr)
                dbvp.set_coeff(coeff)

                dbvp.set_beta_func(beta_func)
                dbvp.set_elem_Adv_mat(beta_func)
                dbvp.set_elem_Bi_mass_mat(beta_func)

                dbvp.set_source_func(f_func)
                dbvp.set_Diri_func(g_func, g_dx_func, g_dy_func)

                logging.info("Getting the glb_A_F...")
                dbvp.get_glb_A_F()
                dbvp.eigen_num = eigen_num
                logging.info("Getting eigen pairs...")
                dbvp.get_eigen_pair()
                Lambda = np.max(dbvp.eigen_val)

                os_ly = sub_sec_ind
                dbvp.init_args(eigen_num, os_ly)
                dbvp.get_ind_map()

                logging.info("Getting the corrector...")
                corrector = dbvp.get_corr()
                logging.info("Getting the true corrector...")
                true_corrector = dbvp.get_true_corr(guess=corrector)

                Dg_ref_l2, Dg_ref_eg = dbvp.get_L2_energy_norm(corrector)
                Dg_l2, Dg_eg = dbvp.get_L2_energy_norm(true_corrector)
                Dg_error_l2, Dg_error_eg = dbvp.get_L2_energy_norm(corrector-true_corrector)

                logging.info("Getting the solution...")
                u0_ms, guess = dbvp.solve()
                logging.info("Getting the reference solution...")
                u0_ref = dbvp.solve_ref(guess=u0_ms)
                u_ref = dbvp.get_inhomo_ref(u0_ref)
                u_ms = dbvp.get_inhomo_ref(u0_ms)
                u_error_l2, u_error_eg = dbvp.get_L2_energy_norm(u0_ms - u0_ref)
                u_ms_l2, u_ms_eg = dbvp.get_L2_energy_norm(u_ms)
                u_ref_l2, u_ref_eg = dbvp.get_L2_energy_norm(u_ref)

                csv_column_names=['contrast','lambda','coarse_grid','eigen_num','over_sampling','Dg_norm_a',\
                                    'Dg_norm_l2', 'Dg_ref_norm_a', 'Dg_ref_norm_l2','Dg_error_a','Dg_error_l2',\
                                    'u_norm_a','u_norm_l2', 'u_ref_norm_a','u_ref_norm_l2', 'u_error_a','u_error_l2']
                
                data = [ctr, Lambda, coarse_grid, eigen_num, sub_sec_ind, Dg_eg, Dg_l2, Dg_ref_eg, Dg_ref_l2,\
                        Dg_error_eg, Dg_error_l2, u_ms_eg, u_ms_l2, u_ref_eg, u_ref_l2, u_error_eg, u_error_l2]
                
                new_row = dict(zip(csv_column_names,data))
                
                
                # now=datetime.now()
                # timestamp = now.strftime("%Y%m%d_%H%M%S")
                np.savez_compressed(f"{result_foldername}/{'_'.join(f'{k}{v}' for k,v in parameters.items())}.npz",
                         corrector=corrector,
                         true_corrector=true_corrector,
                         u0_ms=u0_ms,
                         u0_ref=u0_ref,
                         u_ref=u_ref,
                         u_ms=u_ms,
                         )
                with open(csv_filename,'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=new_row.keys())
                    writer.writerow(new_row)

logging.info("End\n")
