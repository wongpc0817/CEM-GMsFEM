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
FINE_GRID=400

def zero_2d(x, y):
    return 0.


def zero_1d(x, y):
    return 0.


def f_func(x: float, y: float):
    return 1.


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

import Model1.old_inhomoDiriBVP as DBVP

import logging
from logging import config


config.fileConfig(os.path.join(root_path, 'Settings', 'log.conf'), defaults={'logfilename': 'old_plot_ex1.log'})


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

SEC_NUM = 1
SUB_SEC_NUM = 3
EIGEN_NUM = 3
for sec_ind in range(SEC_NUM):
    for sub_sec_ind in range(SUB_SEC_NUM):
        if os.path.exists(f'Plots/old_model1_sol_{sec_ind}_{2 + sub_sec_ind}.npy') and os.path.exists(f'old_Plots/model1_ref_{sec_ind}_{2 + sub_sec_ind}.npy'):
            continue
        ctr = 10.0**4
        coeff = get_coeff_from_tmp(coeff_tmp, ctr)
        logging.info("Get coefficients from the image, set contrast ratio={:.4e}".format(ctr))
        # dbvp = DBVP.ProblemSetting(option=-3)
        dbvp = DBVP.ProblemSetting(option=sec_ind + 1)
        logging.info("Coarse grid:{0:d}x{0:d}, fine grid:{1:d}x{1:d}.".format(dbvp.coarse_grid, dbvp.fine_grid))
        dbvp.set_coeff(coeff)
        dbvp.set_source_func(f_func)
        dbvp.set_Diri_func(g_func, g_dx_func, g_dy_func)
        # ol_ly = max(CEIL(4 * LOG(1 / dbvp.coarse_grid) / LOG(1 / 10)), 1)
        os_ly = 2 + sub_sec_ind
        dbvp.init_args(EIGEN_NUM, os_ly)
        logging.info("Coarse grid: [{0:d}x{0:d}]; fine grid: [{1:d}x{1:d}]; eigenvalue number: [{2:d}]; oversampling layers: [{3:d}].".format(dbvp.coarse_grid, dbvp.fine_grid, dbvp.eigen_num, dbvp.oversamp_layer))
        u0_ms, guess = dbvp.solve()
        u0_ref = dbvp.solve_ref(guess=u0_ms)
        u_ref = dbvp.get_inhomo_ref(u0_ref)
        np.save(f'Plots/old_model1_sol_{sec_ind}_{os_ly}.npy',u0_ms)
        np.save(f'Plots/old_model1_ref_{sec_ind}_{os_ly}.npy',u_ref)
        err_l2_abs, err_eg_abs = dbvp.get_L2_energy_norm(u0_ms - u0_ref)
        u0_ref_l2, u0_ref_eg = dbvp.get_L2_energy_norm(u0_ref)
        u_ref_l2, u_ref_eg = dbvp.get_L2_energy_norm(u_ref)
        logging.info("Absolute errors: L2-norm:{0:.6f}, energy norm:{1:.6f}".format(err_l2_abs, err_eg_abs))
        logging.info("Reference u_0 L2 norm:{0:.6f}, eg norm:{1:.6f}".format(u0_ref_l2, u0_ref_eg))
        logging.info("Reference u L2 norm:{0:.6f}, eg norm:{1:.6f}".format(u_ref_l2, u_ref_eg))
    logging.info("~" * 80)
logging.info("~~Now Plotting~~\n")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

for sec_ind in range(SEC_NUM):
    for sub_sec_ind in range(SUB_SEC_NUM):
        os_ly = 2 + sub_sec_ind
        u0_ms=np.load(f'Plots/old_model1_sol_{sec_ind}_{os_ly}.npy').reshape([FINE_GRID+1,FINE_GRID+1])
        u0_ref=np.load(f'Plots/old_model1_ref_{sec_ind}_{os_ly}.npy').reshape([FINE_GRID+1,FINE_GRID+1])
        # Create data
        x = np.linspace(0, 1, FINE_GRID+1)
        y = np.linspace(0, 1, FINE_GRID+1)
        x, y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, u0_ms)
        plt.savefig(f'Plots/old_model1_sol_{sec_ind}_{os_ly}.png')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, u0_ref)
        plt.savefig(f'Plots/old_model1_ref_{sec_ind}_{os_ly}.png')

logging.info("~END~")