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


root_path = os.path.join(sys.path[0], '..')
sys.path.append(root_path)

import Model2.InhomoDiriBVP as DBVP

import logging
from logging import config

if len(sys.argv) == 1:
    op = 0
    log_filename = "ex1.log"
elif sys.argv[1] == '1':
    op = 1
    log_filename = "ex1p1.log"
elif sys.argv[1] == '2':
    op = 2
    log_filename = "ex1p2.log"
elif sys.argv[1] == '3':
    op = 3
    log_filename = "ex1p3.log"
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

coeff_tmp = np.load(os.path.join(root_path, 'Resources', 'MediumA.npy'))

SEC_NUM = 3
SUB_SEC_NUM = 4
EIGEN_NUM = 3

if op == 0 or op == 1:
    sec_ind = 1
    ctr = 10.0**4
    coeff = get_coeff_from_tmp(coeff_tmp, ctr)
    logging.info("Get coefficients from the image, set contrast ratio={:.4e}".format(ctr))
    dbvp = DBVP.ProblemSetting(option=2)
    logging.info("Coarse grid:{0:d}x{0:d}, fine grid:{1:d}x{1:d}.".format(dbvp.coarse_grid, dbvp.fine_grid))
    # coeff = np.ones((dbvp.fine_grid, dbvp.fine_grid))
    dbvp.set_coeff(coeff)
    dbvp.set_source_func(f_func)
    dbvp.set_Diri_func(g_func, g_dx_func, g_dy_func)
    dbvp.get_glb_A_F()
    dbvp.eigen_num = EIGEN_NUM
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
    logging.info("L2-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_l2_list]))
    logging.info("H1-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_eg_list]))

if op == 0 or op == 2:
    sec_ind = 2
    ctr = 10.0**5
    coeff = get_coeff_from_tmp(coeff_tmp, ctr)
    logging.info("Get coefficients from the image, set contrast ratio={:.4e}".format(ctr))
    dbvp = DBVP.ProblemSetting(option=2)
    logging.info("Coarse grid:{0:d}x{0:d}, fine grid:{1:d}x{1:d}.".format(dbvp.coarse_grid, dbvp.fine_grid))
    # coeff = np.ones((dbvp.fine_grid, dbvp.fine_grid))
    dbvp.set_coeff(coeff)
    dbvp.set_source_func(f_func)
    dbvp.set_Diri_func(g_func, g_dx_func, g_dy_func)
    dbvp.get_glb_A_F()
    dbvp.eigen_num = EIGEN_NUM
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
    logging.info("L2-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_l2_list]))
    logging.info("H1-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_eg_list]))

if op == 0 or op == 3:
    sec_ind = 3
    ctr = 10.0**6
    coeff = get_coeff_from_tmp(coeff_tmp, ctr)
    logging.info("Get coefficients from the image, set contrast ratio={:.4e}".format(ctr))
    dbvp = DBVP.ProblemSetting(option=2)
    logging.info("Coarse grid:{0:d}x{0:d}, fine grid:{1:d}x{1:d}.".format(dbvp.coarse_grid, dbvp.fine_grid))
    # coeff = np.ones((dbvp.fine_grid, dbvp.fine_grid))
    dbvp.set_coeff(coeff)
    dbvp.set_source_func(f_func)
    dbvp.set_Diri_func(g_func, g_dx_func, g_dy_func)
    dbvp.get_glb_A_F()
    dbvp.eigen_num = EIGEN_NUM
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
    logging.info("L2-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_l2_list]))
    logging.info("H1-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_eg_list]))

logging.info("End\n")
