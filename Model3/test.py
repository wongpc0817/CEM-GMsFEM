# Tips
# Use Anaconda Prompt to locate the work directory, then open with VSCODE, the debugging should be fine

import sys, os

root_path = os.path.join(sys.path[0], '..')
sys.path.append(root_path)

import logging
from logging import config

config.fileConfig(os.path.join(root_path, 'Settings', 'log.conf'))

logging.info('=' * 80)
logging.info("Start")

import numpy as np

PI = np.pi
SIN = np.sin
COS = np.cos
EXP = np.exp
LOG = np.log

import Model3.InhomoRobinBVP as NBVP


def get_os_ly(ps):
    return max(1, int(LOG(ps.coarse_grid)))


def f_func(x: float, y: float):
    return EXP(x * y) + SIN(x - y) * COS(x * y) + LOG(x + y + 1.)
    # return .0
    # return -2.0
    # return 1.0
    # return 0.0


def zero(x: float):
    return 0.0


def b_lf_func(y: float):
    return 2 + COS(PI * y)


def q_lf_func(y: float):
    return y * EXP(y)


def q_rg_func(y: float):
    return SIN(y) + y
    # return 0.0


def q_dw_func(x: float):
    return x + 1.0


# u(x, y) = Sin(2 pi x)Sin(2 pi y) + x y
# u0(x, y) = Sin(2 pi x)Sin(2 pi y)

# logging.info("-\Delta u(x,y)=EXP(x*y)+SIN(x-y)*COS(x*y)+LOG(x+y+1), u(x,y)=0 on \partial\Omega")
logging.info("-\Delta u(x,y)=EXP(x*y)+SIN(x-y)*COS(x*y)+LOG(x+y+1) in \Omega, n \cdot Grad u + bu=q on \partial\Omega,")
logging.info("b=2+COS(PI y) on {0}x(0,1); b=0 on {1}x(0,1); b=0 on (0,1)x{0}; b=0 on (0,1)x{1}")
logging.info("q=y EXP(y) on {0}x(0,1); q=SIN(y)+y on {1}x(0,1); q=x+1 on (0,1)x{0}; q=0 on (0,1)x{1}.")
logging.info('-' * 30 + 'CONFIG 1' + '-' * 30)
ps = NBVP.ProblemSetting(option=-3)
coeff = np.ones((ps.fine_grid, ps.fine_grid))
ps.set_coeff(coeff)
ps.set_source_func(f_func)
ps.set_Robin_func(q_lf_func, q_rg_func, q_dw_func, zero, b_lf_func, zero, zero, zero)
os_ly = get_os_ly(ps)
ps.init_args(4, os_ly)
# u0_real = np.zeros((ps.fine_grid+1, ps.fine_grid+1))
# for node_ind_x in range(ps.fine_grid+1):
#     for node_ind_y in range(ps.fine_grid+1):
#         x = ps.h * node_ind_x
#         y = ps.h * node_ind_y
#         u0_real[node_ind_x, node_ind_y] = u0_func(x, y)
logging.info("Coarse grid: [{0:d}x{0:d}]; fine grid: [{1:d}x{1:d}]; eigenvalue number: [{2:d}]; oversampling layers: [{3:d}].".format(ps.coarse_grid, ps.fine_grid, ps.eigen_num, ps.oversamp_layer))
u0 = ps.solve()
# err_l2, err_eg = ps.get_L2_energy_norm(u0_real-u0)
# logging.info("L2-norm error:{0:.6f}, energy-norm error:{1:.6f}".format(err_l2, err_eg))
u0_ref = ps.solve_ref()
# u0_dep = ps.solve_depreciated()
# err_l2_ref, err_eg_ref = ps.get_L2_energy_norm(u0_real-u0_ref)
# logging.info("L2-norm error of the reference:{0:.6f}, energy-norm error of the reference:{1:.6f}".format(err_l2_ref, err_eg_ref))
# u0_dbg = ps.solve_dbg()
# err = (u0 - u0_ref).reshape((-1))
# err_eg_test = np.sqrt(ps.glb_A_mat.dot(err) @ err)
err_l2_num, err_eg_num = ps.get_L2_energy_norm(u0 - u0_ref)
# err_l2_dbg, err_eg_dbg = ps.get_L2_energy_norm(u0_dbg-u0_ref)
logging.info("L2-norm error:{0:.6f}, energy-norm error:{1:.6f}".format(err_l2_num, err_eg_num))