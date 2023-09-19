import matplotlib.image as mpimg
import sys, os
import numpy as np

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

log_filename='exp_Model_1.log'
result_foldername='exp_Model_1'
csv_filename=result_foldername+'/results.csv'

if not os.path.exists(result_foldername):
    os.makedirs(result_foldername)

if not os.path.exists(csv_filename):
    csv_column_names=['contrast','coarse_grid','eigen_num','over_sampling','Dg_norm_a',\
                      'Dg_norm_l2','Dg_error_a','Dg_error_l2','u0_norm_a','u0_norm_l2',\
                        'u_norm_a','u_norm_l2']

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

for eigen_num in range(1,EIGEN_NUM):
    for ctr_exp in [3,4,5,6]:
        for coarse_grid in [10,20,40,80]:
            for sec_ind in range(SEC_NUM):
                for sub_sec_ind in range(SUB_SEC_NUM):
                    ctr = 10.0**4
                    coeff = get_coeff_from_tmp(coeff_tmp, ctr)
                    logging.info("Get coefficients from the image, set contrast ratio={:.4e}".format(ctr))
                    dbvp = DBVP.ProblemSetting()
                    dbvp.set_coarse_grid(coarse_grid)
                    dbvp.set_fine_grid(FINE_GRID)
                    logging.info("Coarse grid:{0:d}x{0:d}, fine grid:{1:d}x{1:d}.".format(dbvp.coarse_grid, dbvp.fine_grid))
                    dbvp.set_coeff(coeff)

                    dbvp.set_beta_func(beta_func)
                    dbvp.set_elem_Adv_mat(beta_func)
                    dbvp.set_elem_Bi_mass_mat(beta_func)

                    dbvp.set_source_func(f_func)
                    dbvp.set_Diri_func(g_func, g_dx_func, g_dy_func)
                    dbvp.get_glb_A_F()
                    dbvp.eigen_num = eigen_num
                    dbvp.get_eigen_pair()
                    Lambda = np.max(dbvp.eigen_val)
                    logging.info("Lambda={0:.6f}".format(Lambda))
                    corr_list = [None] * (SUB_SEC_NUM + 1)
                    for sub_sec_ind in range(1, SUB_SEC_NUM + 1):
                        os_ly = sub_sec_ind
                        dbvp.init_args(EIGEN_NUM, os_ly)
                        dbvp.get_ind_map()
                        corr_list[sub_sec_ind] = dbvp.get_corr()
                        logging.info("Finish section {0:d}-{1:d}, layers={2:d}.".format(sec_ind, sub_sec_ind, dbvp.oversamp_layer))
                    corr_list[0] = dbvp.get_true_corr(guess=corr_list[SUB_SEC_NUM])
                    logging.info("Finish reference section {0:d}-{1:d}.".format(1, sub_sec_ind))
                    err_l2_list = [0.0] * SUB_SEC_NUM
                    err_eg_list = [0.0] * SUB_SEC_NUM
                    err_l2_ref, err_eg_ref = dbvp.get_L2_energy_norm(corr_list[0])
                    logging.info("Reference L2 norm:{0:.6f}, eg norm:{1:.6f}".format(err_l2_ref, err_eg_ref))
                    for sub_sec_ind in range(1, SUB_SEC_NUM + 1):
                        err_l2_list[sub_sec_ind - 1], err_eg_list[sub_sec_ind - 1] = dbvp.get_L2_energy_norm(corr_list[sub_sec_ind] - corr_list[0])
                    np.savez(f"Results/cd_ex1/op_1.npz",corr_list)
                    logging.info("L2-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_l2_list]))
                    logging.info("H1-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_eg_list]))


                    u0_ms, guess = dbvp.solve()
                    u0_ref = dbvp.solve_ref(guess=u0_ms)
                    u_ref = dbvp.get_inhomo_ref(u0_ref)
                    err_l2_abs, err_eg_abs = dbvp.get_L2_energy_norm(u0_ms - u0_ref)
                    u0_ref_l2, u0_ref_eg = dbvp.get_L2_energy_norm(u0_ref)
                    u_ref_l2, u_ref_eg = dbvp.get_L2_energy_norm(u_ref)
                    np.savez(f"Results/cd_ex3/op_1_{sec_ind}_{sub_sec_ind}.npz",u0_ms, u0_ref)

logging.info("End\n")