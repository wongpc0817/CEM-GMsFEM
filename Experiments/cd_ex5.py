import matplotlib.image as mpimg
import sys, os
import numpy as np
import math

PI = np.pi
SIN = np.sin
COS = np.cos
EXP = np.exp
LOG = np.log
CEIL = math.ceil


def zero_2d(x, y):
    return 0.


def zero_1d(x, y):
    return 0.

def beta_func(x:float,y:float):
    return COS(18*PI*y)*SIN(18*PI*x), -COS(18*PI*x)*SIN(18*PI*y)
    
def f_func(x: float, y: float):
    if 1. / 8. < x < 7. / 8. and 3. / 8. < y < 5. / 8.:
        return 1.0
    elif 3. / 8. < x < 5. / 8. and 1. / 8. < y < 7. / 8.:
        return 1.0
    else:
        return 0.0


def q_lf_func(y: float):
    return -1.0


def q_rg_func(y: float):
    return 1.0


def q_dw_func(x: float):
    if x < 0.5:
        return 1.0
    else:
        return 0.0


def q_up_func(x: float):
    if x > 0.5:
        return -1.0
    else:
        return 0.0


root_path = os.path.join(sys.path[0], '..')
sys.path.append(root_path)

import Model3.InhomoRobinBVP as RBVP

import logging
from logging import config

if len(sys.argv) == 1:
    op = 0
    log_filename = "cd_ex5.log"
elif sys.argv[1] == '1':
    op = 1
    log_filename = "cd_ex5p1.log"
elif sys.argv[1] == '2':
    op = 2
    log_filename = "cd_ex5p2.log"
elif sys.argv[1] == '3':
    op = 3
    log_filename = "cd_ex5p3.log"
else:
    raise ValueError

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

coeff_tmp = np.load(os.path.join(root_path, 'Resources', 'MediumC.npy'))

SEC_NUM = 3
SUB_SEC_NUM = 4
EIGEN_NUM = 3

if op == 0 or op == 1:
    sec_ind = 1
    ctr = 10.0**4
    coeff = get_coeff_from_tmp(coeff_tmp, ctr)
    logging.info("Get coefficients from the image, set contrast ratio={:.4e}".format(ctr))
    rbvp = RBVP.ProblemSetting(option=2)
    rbvp.TOL = 1.e-8
    logging.info("Coarse grid:{0:d}x{0:d}, fine grid:{1:d}x{1:d}.".format(rbvp.coarse_grid, rbvp.fine_grid))
    # coeff = np.ones((rbvp.fine_grid, rbvp.fine_grid))
    rbvp.set_coeff(coeff)

        
    rbvp.set_beta_func(beta_func)
    rbvp.set_elem_Adv_mat(beta_func)
    rbvp.set_elem_Bi_mass_mat(beta_func)

    rbvp.set_Robin_coeff(coeff)
    # rbvp.set_coeff(coeff)
    rbvp.set_source_func(f_func)
    rbvp.set_Robin_func(q_lf_func, q_rg_func, q_dw_func, q_up_func)
    rbvp.get_glb_A_F()
    rbvp.eigen_num = EIGEN_NUM
    rbvp.get_eigen_pair()
    Lambda = np.max(rbvp.eigen_val)
    logging.info("Lambda={0:.6f}".format(Lambda))
    corr_list = [None] * (SUB_SEC_NUM + 1)
    for sub_sec_ind in range(1, SUB_SEC_NUM + 1):
        os_ly = sub_sec_ind
        rbvp.init_args(EIGEN_NUM, os_ly)
        rbvp.get_ind_map()
        corr_list[sub_sec_ind] = rbvp.get_corr()
        logging.info("Finish section {0:d}-{1:d}, layers={2:d}.".format(sec_ind, sub_sec_ind, rbvp.oversamp_layer))
    corr_list[0] = rbvp.get_true_corr(guess=corr_list[SUB_SEC_NUM])
    logging.info("Finish reference section {0:d}-{1:d}.".format(1, sub_sec_ind))
    err_l2_list = [0.0] * SUB_SEC_NUM
    err_eg_list = [0.0] * SUB_SEC_NUM
    err_l2_ref, err_eg_ref = rbvp.get_L2_energy_norm(corr_list[0])
    logging.info("Reference L2 norm:{0:.6f}, eg norm:{1:.6f}".format(err_l2_ref, err_eg_ref))
    for sub_sec_ind in range(1, SUB_SEC_NUM + 1):
        err_l2_list[sub_sec_ind - 1], err_eg_list[sub_sec_ind - 1] = rbvp.get_L2_energy_norm(corr_list[sub_sec_ind] - corr_list[0])
    np.savez(f"Experiments/Results/cd_ex5/op_1.npz",corr_list)
    logging.info("L2-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_l2_list]))
    logging.info("H1-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_eg_list]))

if op == 0 or op == 2:
    sec_ind = 2
    ctr = 10.0**5
    coeff = get_coeff_from_tmp(coeff_tmp, ctr)
    logging.info("Get coefficients from the image, set contrast ratio={:.4e}".format(ctr))
    rbvp = RBVP.ProblemSetting(option=2)
    rbvp.TOL = 1.e-8
    logging.info("Coarse grid:{0:d}x{0:d}, fine grid:{1:d}x{1:d}.".format(rbvp.coarse_grid, rbvp.fine_grid))
    # coeff = np.ones((rbvp.fine_grid, rbvp.fine_grid))
    rbvp.set_coeff(coeff)
    rbvp.set_beta_func(beta_func)
    rbvp.set_elem_Adv_mat(beta_func)
    rbvp.set_elem_Bi_mass_mat(beta_func)

    rbvp.set_Robin_coeff(coeff)
    # rbvp.set_coeff(coeff)
    rbvp.set_source_func(f_func)
    rbvp.set_Robin_func(q_lf_func, q_rg_func, q_dw_func, q_up_func)
    rbvp.get_glb_A_F()
    rbvp.eigen_num = EIGEN_NUM
    rbvp.get_eigen_pair()
    Lambda = np.max(rbvp.eigen_val)
    logging.info("Lambda={0:.6f}".format(Lambda))
    corr_list = [None] * (SUB_SEC_NUM + 1)
    for sub_sec_ind in range(1, SUB_SEC_NUM + 1):
        os_ly = sub_sec_ind
        rbvp.init_args(EIGEN_NUM, os_ly)
        rbvp.get_ind_map()
        corr_list[sub_sec_ind] = rbvp.get_corr()
        logging.info("Finish section {0:d}-{1:d}, layers={2:d}.".format(sec_ind, sub_sec_ind, rbvp.oversamp_layer))
    corr_list[0] = rbvp.get_true_corr(guess=corr_list[SUB_SEC_NUM])
    logging.info("Finish reference section {0:d}-{1:d}.".format(1, sub_sec_ind))
    err_l2_list = [0.0] * SUB_SEC_NUM
    err_eg_list = [0.0] * SUB_SEC_NUM
    err_l2_ref, err_eg_ref = rbvp.get_L2_energy_norm(corr_list[0])
    logging.info("Reference L2 norm:{0:.6f}, eg norm:{1:.6f}".format(err_l2_ref, err_eg_ref))
    for sub_sec_ind in range(1, SUB_SEC_NUM + 1):
        err_l2_list[sub_sec_ind - 1], err_eg_list[sub_sec_ind - 1] = rbvp.get_L2_energy_norm(corr_list[sub_sec_ind] - corr_list[0])
    np.savez(f"Experiments/Results/cd_ex5/op_2.npz",corr_list)
    logging.info("L2-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_l2_list]))
    logging.info("H1-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_eg_list]))

if op == 0 or op == 3:
    sec_ind = 3
    ctr = 10.0**6
    coeff = get_coeff_from_tmp(coeff_tmp, ctr)
    logging.info("Get coefficients from the image, set contrast ratio={:.4e}".format(ctr))
    rbvp = RBVP.ProblemSetting(option=2)
    rbvp.TOL = 1.e-8
    logging.info("Coarse grid:{0:d}x{0:d}, fine grid:{1:d}x{1:d}.".format(rbvp.coarse_grid, rbvp.fine_grid))
    # coeff = np.ones((rbvp.fine_grid, rbvp.fine_grid))
    rbvp.set_coeff(coeff)
    rbvp.set_beta_func(beta_func)
    rbvp.set_elem_Adv_mat(beta_func)
    rbvp.set_elem_Bi_mass_mat(beta_func)

    rbvp.set_Robin_coeff(coeff)
    # rbvp.set_coeff(coeff)
    rbvp.set_source_func(f_func)
    rbvp.set_Robin_func(q_lf_func, q_rg_func, q_dw_func, q_up_func)
    rbvp.get_glb_A_F()
    rbvp.eigen_num = EIGEN_NUM
    rbvp.get_eigen_pair()
    Lambda = np.max(rbvp.eigen_val)
    logging.info("Lambda={0:.6f}".format(Lambda))
    corr_list = [None] * (SUB_SEC_NUM + 1)
    for sub_sec_ind in range(1, SUB_SEC_NUM + 1):
        os_ly = sub_sec_ind
        rbvp.init_args(EIGEN_NUM, os_ly)
        rbvp.get_ind_map()
        corr_list[sub_sec_ind] = rbvp.get_corr()
        logging.info("Finish section {0:d}-{1:d}, layers={2:d}.".format(sec_ind, sub_sec_ind, rbvp.oversamp_layer))
    corr_list[0] = rbvp.get_true_corr(guess=corr_list[SUB_SEC_NUM])
    logging.info("Finish reference section {0:d}-{1:d}.".format(1, sub_sec_ind))
    err_l2_list = [0.0] * SUB_SEC_NUM
    err_eg_list = [0.0] * SUB_SEC_NUM
    err_l2_ref, err_eg_ref = rbvp.get_L2_energy_norm(corr_list[0])
    logging.info("Reference L2 norm:{0:.6f}, eg norm:{1:.6f}".format(err_l2_ref, err_eg_ref))
    for sub_sec_ind in range(1, SUB_SEC_NUM + 1):
        err_l2_list[sub_sec_ind - 1], err_eg_list[sub_sec_ind - 1] = rbvp.get_L2_energy_norm(corr_list[sub_sec_ind] - corr_list[0])
    np.savez(f"Experiments/Results/cd_ex5/op_3.npz",corr_list)
    logging.info("L2-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_l2_list]))
    logging.info("H1-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_eg_list]))

logging.info("End\n")
