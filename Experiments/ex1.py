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
    return EXP(x * y) + SIN(x - y) * COS(x * y) + LOG(x + y + 1.)


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

config.fileConfig(os.path.join(root_path, 'Settings', 'log.conf'))


def get_coeff_from_tmp(coeff_tmp, ctr: float):
    return 1.0 + ctr * (1.0 - coeff_tmp)


logging.info('=' * 80)
logging.info("Start")

img = mpimg.imread(os.path.join(root_path, 'Resources', 'Medium.png'))
coeff_tmp = img[:, :, 0]

SEC_NUM = 1
SUB_SEC_NUM = 2

for sec_ind in range(1, SEC_NUM + 1):
    # Section 1
    ctr = 10.0**(sec_ind + 3)
    ctr = 10
    coeff = get_coeff_from_tmp(coeff_tmp, ctr)
    logging.info("Get coefficients from the image, set contrast ratio={:.4f}".format(ctr))
    dbvp = DBVP.ProblemSetting(option=-3)
    dbvp.set_coeff(coeff)
    dbvp.set_source_func(f_func)
    dbvp.set_Diri_func(g_func, g_dx_func, g_dy_func)
    dbvp.eigen_num = 4
    dbvp.get_eigen_pair()
    Lambda = np.max(dbvp.eigen_val)
    logging.info("Lambda={0:.6f}".format(Lambda))
    corr_list = [None] * (SUB_SEC_NUM + 1)
    for sub_sec_ind in range(0, SUB_SEC_NUM + 1):
        if sub_sec_ind == 0:
            os_ly = dbvp.coarse_grid - 1
            dbvp.init_args(4, os_ly)
            corr_list[0] = dbvp.get_true_corr()
        else:
            os_ly = sub_sec_ind
            dbvp.init_args(4, os_ly)
            dbvp.get_ind_map()
            corr_list[sub_sec_ind] = dbvp.get_corr()
        logging.info("Finish experiment section {0:d}-{1:d}.".format(sec_ind, sub_sec_ind))
    dbvp.get_glb_A_F()
    err_l2_list = [None] * SUB_SEC_NUM
    err_eg_list = [None] * SUB_SEC_NUM
    for sub_sec_ind in range(1, SUB_SEC_NUM + 1):
        err_l2_list[sub_sec_ind - 1], err_eg_list[sub_sec_ind - 1] = dbvp.get_L2_energy_norm(corr_list[sub_sec_ind] - corr_list[0])
    logging.info("L2-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_l2_list]))
    logging.info("H1-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_l2_list]))
logging.info("End\n")
