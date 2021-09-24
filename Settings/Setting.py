import numpy as np

FINE_GRID = 400
COARSE1 = 10
COARSE2 = 20
COARSE3 = 40
COARSE4 = 80
N_V = 4

QUAD_ORDER = 5
QUAD_CORD, QUAD_WGHT = np.polynomial.legendre.leggauss(QUAD_ORDER)


class Setting:
    def __init__(self, option: int = 1):
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
            self.fine_grid = 4
            self.coarse_grid = 2
        else:
            raise ValueError("Invalid option")
        self.fine_elem = self.fine_grid**2
        self.coarse_elem = self.coarse_grid**2
        self.sub_grid = self.fine_grid // self.coarse_grid
        self.sub_elem = self.sub_grid**2
        self.H = 1.0 / float(self.coarse_grid)
        self.h = 1.0 / float(self.fine_grid)


def get_locbase_val(loc_ind: int, x: float, y: float):
    val = -1.0
    assert (-1.0 <= x <= 1.0 and -1.0 <= y <= 1.0)
    if loc_ind == 0:
        val = 0.25 * (1.0-x) * (1.0-y)
    elif loc_ind == 1:
        val = 0.25 * (1.0+x) * (1.0-y)
    elif loc_ind == 2:
        val = 0.25 * (1.0-x) * (1.0+y)
    elif loc_ind == 3:
        val = 0.25 * (1.0+x) * (1.0+y)
    else:
        raise ValueError("Invalid option")
    return val


def get_locbase_grad_val(loc_ind: int, x: float, y: float):
    grad_val_x, grad_val_y = -1.0, -1.0
    assert (-1.0 <= x <= 1.0 and -1.0 <= y <= 1.0)
    if loc_ind == 0:
        grad_val_x = -0.25 * (1.0-y)
        grad_val_y = -0.25 * (1.0-x)
    elif loc_ind == 1:
        grad_val_x = 0.25 * (1.0-y)
        grad_val_y = -0.25 * (1.0+x)
    elif loc_ind == 2:
        grad_val_x = -0.25 * (1.0+y)
        grad_val_y = 0.25 * (1.0-x)
    elif loc_ind == 3:
        grad_val_x = 0.25 * (1.0+y)
        grad_val_y = 0.25 * (1.0+x)
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
            val += loc_coeff * (grad_val_ix*grad_val_jx+grad_val_iy*grad_val_jy) * quad_wght_x * quad_wght_y
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
