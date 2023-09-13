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

import Model1.InhomoDiriBVP as DBVP


def get_coeff_from_tmp(coeff_tmp, ctr: float):
    coeff = np.ones(coeff_tmp.shape)
    dim_x, dim_y = coeff_tmp.shape
    for ind_y in range(dim_y):
        for ind_x in range(dim_x):
            if coeff_tmp[ind_x, ind_y] >= 2.0:
                coeff[ind_x, ind_y] = ctr
    return coeff


coeff_tmp = np.load(os.path.join(root_path, 'Resources', 'MediumA.npy'))

SEC_NUM = 1
SUB_SEC_NUM = 3
EIGEN_NUM = 3

if not os.path.isdir('Plots/cd_ex3'):
    os.makedirs('Plots/cd_ex3')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Solution Plots
for op in [1,2,3]:
    for sec_ind in range(SEC_NUM):
            for sub_sec_ind in range(SUB_SEC_NUM):
                data = np.load(f"Results/cd_ex3/op_1_{sec_ind}_{sub_sec_ind}.npz")

                u0_ms=data['arr_0'].reshape([FINE_GRID+1,FINE_GRID+1])
                u0_ref=data['arr_1'].reshape([FINE_GRID+1,FINE_GRID+1])

                x=np.linspace(0,1,FINE_GRID+1)
                y=np.linspace(0,1,FINE_GRID+1)
                x,y=np.meshgrid(x,y)
                fig=plt.figure()
                ax=fig.add_subplot(111,projection='3d')
                ax.plot_surface(x,y,u0_ms)
                fig.savefig(f'Plots/cd_ex3/op_{op}_{sec_ind}_{sub_sec_ind}_u0_ms.png')
                fig=plt.figure()
                ax=fig.add_subplot(111,projection='3d')
                ax.plot_surface(x,y,u0_ref)
                fig.savefig(f'Plots/cd_ex3/op_{op}_{sec_ind}_{sub_sec_ind}_u0_ref.png')