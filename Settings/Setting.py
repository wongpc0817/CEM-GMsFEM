import numpy as np

FINE_GRID = 400
COARSE1 = 10
COARSE2 = 20
COARSE3 = 40
COARSE4 = 80
N_V = 4

QUAD_ORDER = 5
QUAD_CORD, QUAD_WGHT = np.polynomial.legendre.leggauss(QUAD_ORDER)


def get_locbase_val(loc_ind: int, x: float, y: float):
    val = -1.0
    if loc_ind == 0:
        val = 0.25 * (1.0 - x) * (1.0 - y)
    elif loc_ind == 1:
        val = 0.25 * (1.0 + x) * (1.0 - y)
    elif loc_ind == 2:
        val = 0.25 * (1.0 - x) * (1.0 + y)
    elif loc_ind == 3:
        val = 0.25 * (1.0 + x) * (1.0 + y)
    else:
        raise ValueError("Invalid option")
    return val


def get_locbase_grad_val(loc_ind: int, x: float, y: float):
    grad_val_x, grad_val_y = -1.0, -1.0
    if loc_ind == 0:
        grad_val_x = -0.25 * (1.0 - y)
        grad_val_y = -0.25 * (1.0 - x)
    elif loc_ind == 1:
        grad_val_x = 0.25 * (1.0 - y)
        grad_val_y = -0.25 * (1.0 + x)
    elif loc_ind == 2:
        grad_val_x = -0.25 * (1.0 + y)
        grad_val_y = 0.25 * (1.0 - x)
    elif loc_ind == 3:
        grad_val_x = 0.25 * (1.0 + y)
        grad_val_y = 0.25 * (1.0 + x)
    else:
        raise ValueError("Invalid option")
    return grad_val_x, grad_val_y


def get_loc_stiff(loc_coeff: float, loc_ind_i: int, loc_ind_j: int):
    val = 0.0
    for quad_ind_x in range(QUAD_ORDER):
        for quad_ind_y in range(QUAD_ORDER):
            quad_cord_x, quad_cord_y = QUAD_CORD[quad_ind_x], QUAD_CORD[quad_ind_y]
            quad_wght_x, quad_wght_y = QUAD_WGHT[quad_ind_x], QUAD_WGHT[quad_ind_y]
            grad_val_ix, grad_val_iy = get_locbase_grad_val(loc_ind_i, quad_cord_x, quad_cord_y)
            grad_val_jx, grad_val_jy = get_locbase_grad_val(loc_ind_j, quad_cord_x, quad_cord_y)
            val += loc_coeff * (grad_val_ix * grad_val_jx + grad_val_iy * grad_val_jy) * quad_wght_x * quad_wght_y
    return val


def get_loc_mass(h: float, loc_kappa: float, loc_ind_i: int, loc_ind_j: int):
    val = 0.0
    for quad_ind_x in range(QUAD_ORDER):
        for quad_ind_y in range(QUAD_ORDER):
            quad_cord_x, quad_cord_y = QUAD_CORD[quad_ind_x], QUAD_CORD[quad_ind_y]
            quad_wght_x, quad_wght_y = QUAD_WGHT[quad_ind_x], QUAD_WGHT[quad_ind_y]
            val_i = get_locbase_val(loc_ind_i, quad_cord_x, quad_cord_y)
            val_j = get_locbase_val(loc_ind_j, quad_cord_x, quad_cord_y)
            val += 0.25 * h**2 * loc_kappa * val_i * val_j * quad_wght_x * quad_wght_y
    return val


class Setting:
    def upd(self, option: int = 1):
        self.coarse_grid = 0
        if option == 1:
            self.fine_grid = FINE_GRID
            self.coarse_grid = COARSE1
        elif option == 2:
            self.fine_grid = FINE_GRID
            self.coarse_grid = COARSE2
        elif option == 3:
            self.fine_grid = FINE_GRID
            self.coarse_grid = COARSE3
        elif option == 4:
            self.fine_grid = FINE_GRID
            self.coarse_grid = COARSE4
        elif option == -1:
            self.fine_grid = 8
            self.coarse_grid = 2
        elif option == -2:
            self.fine_grid = 16
            self.coarse_grid = 4
        elif option == -3:
            self.fine_grid = 32
            self.coarse_grid = 8
        elif option == -4:
            self.fine_grid = 64
            self.coarse_grid = 16
        elif option == -5:
            self.fine_grid = 128
            self.coarse_grid = 32
        elif option == -6:
            self.fine_grid = 256
            self.coarse_grid = 64
        else:
            raise ValueError("Invalid option")
        self.fine_elem = self.fine_grid**2
        self.coarse_elem = self.coarse_grid**2
        self.sub_grid = self.fine_grid // self.coarse_grid
        self.sub_elem = self.sub_grid**2
        self.tot_node = (self.fine_grid + 1)**2
        self.H = 1.0 / float(self.coarse_grid)
        self.h = 1.0 / float(self.fine_grid)
        # Save the local Laplace stiffness matrix i.e., \int_{K_h} \nabla L_i \cdot \nabla L_j dx
        # Save the local mass matrix i.e., \int_{K_h} L_i L_j dx
        self.elem_Lap_stiff_mat = np.zeros((N_V, N_V))
        self.elem_Bi_mass_mat = np.zeros((N_V, N_V))
        for loc_ind_i in range(N_V):
            for loc_ind_j in range(N_V):
                self.elem_Lap_stiff_mat[loc_ind_i, loc_ind_j] = get_loc_stiff(1.0, loc_ind_i, loc_ind_j)
                self.elem_Bi_mass_mat[loc_ind_i, loc_ind_j] = get_loc_mass(self.h, 1.0, loc_ind_i, loc_ind_j)

    def __init__(self, option: int = 1):
        self.upd(option)
