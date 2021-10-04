# Tips
# Use Anaconda Prompt to locate the work directory, then open with VSCODE, the debugging should be fine

import sys, os
root_path = os.path.join(sys.path[0], '..')
sys.path.append(root_path)

import logging
from logging import config
config.fileConfig(os.path.join(root_path, 'Settings', 'log.conf'))

logging.info('='*80)
logging.info("Start")

import numpy as np
PI = np.pi
SIN = np.sin
EXP = np.exp

import Model2.InhomoDiriBVP as DBVP


def f_func(x: float, y: float):
    return 2.0 * PI**2 * SIN(PI*x) * SIN(PI*y)
    # return .0
    # return -2.0
    # return 1.0
    # return 0.0


def g_func(x: float, y: float):
    # return x*y
    # return 0.5*x**2 + 0.5*y**2
    return 0.0
    # return x**2 + x*EXP(y)
    # return x**2 + y**2


def g_dx_func(x: float, y: float):
    # return y
    # return x
    return 0.0
    # return 2.0*x + EXP(y)
    # return 2.0*x

def g_dy_func(x: float, y: float):
    # return x
    # return y
    return 0.0
    # return x*EXP(y)
    # return 2.0*y


def u0_func(x: float, y: float):
    return SIN(PI*x) * SIN(PI*y)
    # return 0.0

# u(x, y) = Sin(2 pi x)Sin(2 pi y) + x y
# u0(x, y) = Sin(2 pi x)Sin(2 pi y)


ps = DBVP.ProblemSetting(option=-2)
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
logging.info("f(x,y)=SIN(PI x) SIN(PI y), u(x,y)=0 on \partial\Omega.")
u0 = ps.solve()
# err_l2, err_eg = ps.get_L2_energy_norm(u0_real-u0)
# logging.info("L2-norm error:{0:.6f}, energy-norm error:{1:.6f}".format(err_l2, err_eg))
u0_ref = ps.solve_ref()
# u0_dep = ps.solve_depreciated()
# err_l2_ref, err_eg_ref = ps.get_L2_energy_norm(u0_real-u0_ref)
# logging.info("L2-norm error of the reference:{0:.6f}, energy-norm error of the reference:{1:.6f}".format(err_l2_ref, err_eg_ref))
# u0_dbg = ps.solve_dbg()
err_l2_num, err_eg_num = ps.get_L2_energy_norm(u0-u0_ref)
# err_l2_dbg, err_eg_dbg = ps.get_L2_energy_norm(u0_dbg-u0_ref)
logging.info("L2-norm error:{0:.6f}, energy-norm error:{1:.6f}".format(err_l2_num, err_eg_num))
# err_l2_dep, err_eg_dep = ps.get_L2_energy_norm(u0-u0_dep)
# err_l2_dbg, err_eg_dbg = ps.get_L2_energy_norm(u0_dbg-u0_ref)
# logging.info("[Depreciated]L2-norm error:{0:.6f}, energy-norm error:{1:.6f}".format(err_l2_dep, err_eg_dep))
# logging.info("L2-norm error:{0:.6f}, energy-norm error:{1:.6f}".format(err_l2_dbg, err_eg_dbg))
logging.info("End\n")
