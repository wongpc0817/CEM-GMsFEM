# Tips
# Use Anaconda Prompt to locate the work directory, then open with VSCODE, the debugging should be fine

import sys, os
root_path = os.path.join(sys.path[0], '..')
sys.path.append(root_path)

import logging
from logging import config
config.fileConfig(os.path.join(root_path, 'Settings', 'log.conf'))


logging.info("Start")

import numpy as np
PI = np.pi
SIN = np.sin

import Model2.InhomoDiriBVP as DBVP


def f_func(x: float, y: float):
    return 8.* PI**2 * SIN(2.*PI*x) * SIN(2.*PI*y)


def g_func(x: float, y: float):
    return x*y


def g_dx_func(x: float, y: float):
    return y


def g_dy_func(x: float, y: float):
    return x


def u0_func(x: float, y: float):
    return SIN(2.*PI*x) * SIN(2.*PI*y)

# u(x, y) = Sin(2 pi x)Sin(2 pi y) + x y
# u0(x, y) = Sin(2 pi x)Sin(2 pi y)


ps = DBVP.ProblemSetting(option=-3)
coeff = np.ones((ps.fine_grid, ps.fine_grid))
ps.init_args(4, 1)
u0_real = np.zeros((ps.fine_grid+1, ps.fine_grid+1))
for node_ind_x in range(ps.fine_grid+1):
    for node_ind_y in range(ps.fine_grid+1):
        x = ps.h * node_ind_x
        y = ps.h * node_ind_y
        u0_real[node_ind_x, node_ind_y] = u0_func(x, y)

logging.info("Coarse grid: [{0:d}x{0:d}]; fine grid: [{1:d}x{1:d}]; eigenvalue number: [{2:d}]; oversampling layers: [{3:d}].".format(ps.coarse_grid, ps.fine_grid, ps.eigen_num, ps.oversamp_layer))
ps.set_coeff(coeff)
ps.set_source_func(f_func)
ps.set_Diri_func(g_func, g_dx_func, g_dy_func)
u0 = ps.solve()
err_l2, err_eg = ps.get_L2_energy_norm(u0_real-u0)
logging.info("Error in L2 norm:{0:.6f}, in energy norm:{1:.6f}".format(err_l2, err_eg))
logging.info("End\n")
