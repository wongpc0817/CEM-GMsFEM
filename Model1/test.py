# Tips
# Use Anaconda Prompt to locate the work directory, then open with VSCODE, the debugging should be fine

import sys, os

root_path = os.path.join(sys.path[0], '..')
sys.path.append(root_path)

import logging
from logging import config

config.fileConfig(os.path.join(root_path, 'Settings', 'log.conf'), defaults={'logfilename': 'test.log'})

logging.info('=' * 80)
logging.info("Start")

import numpy as np

PI = np.pi
SIN = np.sin
COS = np.cos
EXP = np.exp
LOG = np.log

import Model1.InhomoNeumBVP as NBVP


def get_os_ly(ps):
    return max(1, int(LOG(ps.coarse_grid)))


def f_func(x: float, y: float):
    # return PI**2 * (1. - y) * COS(PI * x)
    return 0.


def q_lf_func(y: float):
    # return y * EXP(y)
    # return 0.0
    return y - 1.


def q_rg_func(y: float):
    # return SIN(y) + y
    # return 0.0
    return 1. - y


def q_dw_func(x: float):
    # return 0.0
    # return x + 1.0
    # return COS(PI * x)
    return x


def u(x, y):
    # return (1.0 - y) * COS(PI * x)
    return (1. - y) * x


# u0(x, y) = Sin(2 pi x)Sin(2 pi y)

# logging.info("-\Delta u(x,y)=EXP(x*y)+SIN(x-y)*COS(x*y)+LOG(x+y+1), u(x,y)=0 on \partial\Omega")
logging.info("-\Delta u(x,y)=EXP(x*y)+SIN(x-y)*COS(x*y)+LOG(x+y+1) in \Omega.")
logging.info("q=y EXP(y) on {0}x(0,1); q=SIN(y)+y on {1}x(0,1); q=x+1.0 on (0,1)x{0}.")
logging.info('-' * 30 + 'CONFIG 1' + '-' * 30)
ps = NBVP.ProblemSetting(option=-4)
coeff = np.ones((ps.fine_grid, ps.fine_grid))
ps.set_coeff(coeff)
ps.set_source_func(f_func)
ps.set_Neum_func(q_lf_func, q_rg_func, q_dw_func)

u_real = np.zeros((ps.tot_node, ))
for node_ind in range(ps.tot_node):
    node_ind_y, node_ind_x = divmod(node_ind, ps.fine_grid + 1)
    y, x = ps.h * node_ind_y, ps.h * node_ind_x
    u_real[node_ind] = u(x, y)

# ps.get_glb_A_F()
# ps.eigen_num = 3
# SUB_SEC_NUM = 4
# ps.get_eigen_pair()
# corr_list = [None] * (SUB_SEC_NUM + 1)
# err_l2_list = [0.0] * SUB_SEC_NUM
# err_eg_list = [0.0] * SUB_SEC_NUM
# for sub_sec_ind in range(SUB_SEC_NUM + 1):
#     if sub_sec_ind == 0:
#         corr_list[0] = ps.get_true_corr()
#     else:
#         ps.init_args(3, sub_sec_ind)
#         ps.get_ind_map()
#         corr_list[sub_sec_ind] = ps.get_corr()
#         err_l2_list[sub_sec_ind - 1], err_eg_list[sub_sec_ind - 1] = ps.get_L2_energy_norm(corr_list[sub_sec_ind] - corr_list[0])
# logging.info("L2-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_l2_list]))
# logging.info("H1-norm errors:" + "  ".join(["{:.6f}".format(err) for err in err_eg_list]))
# os_ly = get_os_ly(ps)
ps.init_args(4, 5)
logging.info("Coarse grid: [{0:d}x{0:d}]; fine grid: [{1:d}x{1:d}]; eigenvalue number: [{2:d}]; oversampling layers: [{3:d}].".format(ps.coarse_grid, ps.fine_grid, ps.eigen_num, ps.oversamp_layer))
# ps.get_eigen_pair()
# true_glb_corr = ps.get_true_corr()
u0 = ps.solve()
# glb_corr = ps.get_corr()
# err_l2, err_eg = ps.get_L2_energy_norm(u0_real-u0)
# logging.info("L2-norm error:{0:.6f}, energy-norm error:{1:.6f}".format(err_l2, err_eg))
u0_ref = ps.solve_ref()
# u0_dep = ps.solve_depreciated()
# err_l2_ref, err_eg_ref = ps.get_L2_energy_norm(u0_real-u0_ref)
# logging.info("L2-norm error of the reference:{0:.6f}, energy-norm error of the reference:{1:.6f}".format(err_l2_ref, err_eg_ref))
# u0_dbg = ps.solve_dbg()
# err = (u0 - u0_ref).reshape((-1))
# err_eg_test = np.sqrt(ps.glb_A_mat.dot(err) @ err)
# err_l2_corr, err_eg_corr = ps.get_L2_energy_norm(true_glb_corr - glb_corr)
err_l2_ref, err_eg_ref = ps.get_L2_energy_norm(u0_ref - u_real)
err_l2_cem, err_eg_cem = ps.get_L2_energy_norm(u0 - u_real)
err_l2_num, err_eg_num = ps.get_L2_energy_norm(u0 - u0_ref)
ref_l2, ref_eg = ps.get_L2_energy_norm(u0_ref)
# err_l2_dbg, err_eg_dbg = ps.get_L2_energy_norm(u0_dbg-u0_ref)
logging.info("L2-norm error:{0:.6f}, energy-norm error:{1:.6f}".format(err_l2_num / ref_l2, err_eg_num / ref_eg))
logging.info("u_h-u: L2-norm error:{0:.6f}, energy-norm error:{1:.6f}".format(err_l2_ref, err_eg_ref))
logging.info("u_H-u: L2-norm error:{0:.6f}, energy-norm error:{1:.6f}".format(err_l2_cem, err_eg_cem))

# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator

# X = np.linspace(0., 1., ps.fine_grid + 1)
# Y = np.linspace(0., 1., ps.fine_grid + 1)
# XX, YY = np.meshgrid(X, Y)

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(X, Y, u_real.reshape((ps.fine_grid + 1, -1)), cmap=cm.coolwarm, linewidth=0, antialiased=False)
# plt.show()
# surf = ax.plot_surface(X, Y, u0.reshape((ps.fine_grid + 1, -1)), cmap=cm.coolwarm, linewidth=0, antialiased=False)
# plt.show()
# surf = ax.plot_surface(X, Y, u0_ref.reshape((ps.fine_grid + 1, -1)), cmap=cm.coolwarm, linewidth=0, antialiased=False)
# plt.show()
