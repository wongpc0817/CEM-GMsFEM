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

import Model1.InhomoDiriBVP as DBVP

import logging
from logging import config

if len(sys.argv) == 1:
    op = 0
    log_filename = "ex3.log"
elif sys.argv[1] == '1':
    op = 1
    log_filename = "ex3p1.log"
elif sys.argv[1] == '2':
    op = 2
    log_filename = "ex3p2.log"
elif sys.argv[1] == '3':
    op = 3
    log_filename = "ex3p3.log"
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

# Change coarse grid (10, 20, 40, 80), change oversampling size
# Fix basis number=3, contrast ratio=10^4
if op == 0 or op == 1:
    SUB_SEC_NUM = 3
    EIGEN_NUM = 3
    for sub_sec_ind in range(SUB_SEC_NUM):
        ctr = 10.0**4
        coeff = get_coeff_from_tmp(coeff_tmp, ctr)
        logging.info("Get coefficients from the image, set contrast ratio={:.4e}".format(ctr))
        # dbvp = DBVP.ProblemSetting(option=4)
        dbvp = DBVP.ProblemSetting(option=sub_sec_ind + 2)
        logging.info("Coarse grid:{0:d}x{0:d}, fine grid:{1:d}x{1:d}.".format(dbvp.coarse_grid, dbvp.fine_grid))
        dbvp.set_coeff(coeff)
        dbvp.set_source_func(f_func)
        dbvp.set_Diri_func(g_func, g_dx_func, g_dy_func)
        # ol_ly = max(CEIL(4 * LOG(1 / dbvp.coarse_grid) / LOG(1 / 10)), 1)
        ol_ly = 2 + sub_sec_ind
        dbvp.init_args(EIGEN_NUM, ol_ly)
        logging.info("Coarse grid: [{0:d}x{0:d}]; fine grid: [{1:d}x{1:d}]; eigenvalue number: [{2:d}]; oversampling layers: [{3:d}].".format(dbvp.coarse_grid, dbvp.fine_grid, dbvp.eigen_num, dbvp.oversamp_layer))
        u0_ms = dbvp.solve()
        u0_ref = dbvp.solve_ref(guess=u0_ms)
        err_l2_abs, err_eg_abs = dbvp.get_L2_energy_norm(u0_ms - u0_ref)
        u0_ref_l2, u0_ref_eg = dbvp.get_L2_energy_norm(u0_ref)
        logging.info("Absolute errors: L2-norm:{0:.6f}, energy norm:{1:.6f}".format(err_l2_abs, err_eg_abs))
        logging.info("Reference L2 norm:{0:.6f}, eg norm:{1:.6f}".format(u0_ref_l2, u0_ref_eg))
    logging.info("~~End of section 1~~\n")

# Change contrast ratio 10^3, 10^4, 10^5, 10^6, change oversampling size 3, 4, 5
# Fix coarse grid=80, basis number=3
if op == 0 or op == 2:
    SEC_NUM = 4
    SUB_SEC_NUM = 3
    EIGEN_NUM = 3
    for sec_ind in range(SEC_NUM):
        ctr = 10.0**(sec_ind + 3)
        coeff = get_coeff_from_tmp(coeff_tmp, ctr)
        logging.info("Get coefficients from the image, set contrast ratio={:.4e}".format(ctr))
        # dbvp = DBVP.ProblemSetting(option=-3)
        dbvp = DBVP.ProblemSetting(option=4)
        logging.info("Coarse grid:{0:d}x{0:d}, fine grid:{1:d}x{1:d}.".format(dbvp.coarse_grid, dbvp.fine_grid))
        dbvp.set_coeff(coeff)
        dbvp.set_source_func(f_func)
        dbvp.set_Diri_func(g_func, g_dx_func, g_dy_func)
        for sub_sec_ind in range(SUB_SEC_NUM):
            dbvp.init_args(EIGEN_NUM, 2 + sub_sec_ind)
            logging.info("Coarse grid: [{0:d}x{0:d}]; fine grid: [{1:d}x{1:d}]; eigenvalue number: [{2:d}]; oversampling layers: [{3:d}].".format(dbvp.coarse_grid, dbvp.fine_grid, dbvp.eigen_num, dbvp.oversamp_layer))
            u0_ms = dbvp.solve()
            u0_ref = dbvp.solve_ref(guess=u0_ms)
            err_l2_abs, err_eg_abs = dbvp.get_L2_energy_norm(u0_ms - u0_ref)
            u0_ref_l2, u0_ref_eg = dbvp.get_L2_energy_norm(u0_ref)
            logging.info("Absolute errors: L2-norm:{0:.6f}, energy norm:{1:.6f}".format(err_l2_abs, err_eg_abs))
            logging.info("Reference L2 norm:{0:.6f}, eg norm:{1:.6f}".format(u0_ref_l2, u0_ref_eg))
        logging.info("~" * 80)
    logging.info("~~End of section 2~~\n")

# Change basis num 2, 3, 4
# Fix coarse grid=80, contrast ratio=10^4, oversampling size=7
if op == 0 or op == 3:
    SEC_NUM = 1
    for sec_ind in range(SEC_NUM):
        ctr = 10.0**4
        coeff = get_coeff_from_tmp(coeff_tmp, ctr)
        logging.info("Get coefficients from the image, set contrast ratio={:.4e}".format(ctr))
        # dbvp = DBVP.ProblemSetting(option=-3)
        dbvp = DBVP.ProblemSetting(option=4)
        logging.info("Coarse grid:{0:d}x{0:d}, fine grid:{1:d}x{1:d}.".format(dbvp.coarse_grid, dbvp.fine_grid))
        dbvp.set_coeff(coeff)
        dbvp.set_source_func(f_func)
        dbvp.set_Diri_func(g_func, g_dx_func, g_dy_func)
        dbvp.init_args(sec_ind + 1, 3)
        logging.info("Coarse grid: [{0:d}x{0:d}]; fine grid: [{1:d}x{1:d}]; eigenvalue number: [{2:d}]; oversampling layers: [{3:d}].".format(dbvp.coarse_grid, dbvp.fine_grid, dbvp.eigen_num, dbvp.oversamp_layer))
        u0_ms = dbvp.solve()
        u0_ref = dbvp.solve_ref(guess=u0_ms)
        err_l2_abs, err_eg_abs = dbvp.get_L2_energy_norm(u0_ms - u0_ref)
        u0_ref_l2, u0_ref_eg = dbvp.get_L2_energy_norm(u0_ref)
        logging.info("Absolute errors: L2-norm:{0:.6f}, energy norm:{1:.6f}".format(err_l2_abs, err_eg_abs))
        logging.info("Reference L2 norm:{0:.6f}, eg norm:{1:.6f}".format(u0_ref_l2, u0_ref_eg))
    logging.info("~~End of section 3~~\n")

logging.info("~END~")