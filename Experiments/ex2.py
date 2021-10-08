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


def q_lf_func(y: float):
    return y * EXP(y)


def q_rg_func(y: float):
    return SIN(y) + y
    # return 0.0


def q_dw_func(x: float):
    return x + 1.0


root_path = os.path.join(sys.path[0], '..')
sys.path.append(root_path)

import Model1.InhomoNeumBVP as NBVP

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
    coeff = get_coeff_from_tmp(coeff_tmp, ctr)
    logging.info("Get coefficients from the image, set contrast ratio={:.4f}".format(ctr))
    nbvp = NBVP.ProblemSetting(option=-3)
    nbvp.set_coeff(coeff)
    nbvp.set_source_func(f_func)
    nbvp.set_Neum_func(q_lf_func, q_rg_func, q_dw_func)
    nbvp.eigen_num = 4
    nbvp.get_eigen_pair()
    Lambda = np.max(nbvp.eigen_val)
    logging.info("Lambda={0:.6f}".format(Lambda))
    corr_list = [None] * (SUB_SEC_NUM + 1)
    for sub_sec_ind in range(0, SUB_SEC_NUM + 1):
        if sub_sec_ind == 0:
            os_ly = nbvp.coarse_grid - 1
            nbvp.init_args(4, os_ly)
            corr_list[0] = nbvp.get_true_corr()
        else:
            os_ly = sub_sec_ind
            nbvp.init_args(4, os_ly)
            nbvp.get_ind_map()
            corr_list[sub_sec_ind] = nbvp.get_corr()
        logging.info("Finish experiment section {0:d}-{1:d}.".format(sec_ind, sub_sec_ind))
    nbvp.get_glb_A_F()
    err_l2_list = [None] * SUB_SEC_NUM
    err_eg_list = [None] * SUB_SEC_NUM
    for sub_sec_ind in range(1, SUB_SEC_NUM + 1):
        err_l2_list[sub_sec_ind - 1], err_eg_list[sub_sec_ind - 1] = nbvp.get_L2_energy_norm(corr_list[sub_sec_ind] - corr_list[0])
    logging.info("L2-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_l2_list]))
    logging.info("H1-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_l2_list]))
logging.info("End\n")
