import Code.Settings.Setting as ST
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


class ProblemSetting(ST.Setting):
    def upd_eigen_num(self, n: int = 4):
        self.eigen_num = n

    def upd_oversamp_layer(self, k: int = 1):
        self.oversamp_layer = k

    def set_coeff(self, coeff: np.ndarray):
        assert coeff.shape == (self.fine_grid, self.fine_grid)
        self.coeff = coeff
        self.kappa = 24.0 * self.coarse_grid**2 * self.coeff
        # Magic formula from the paper

    def get_eigen_pair(self, eigen_num: int):
        assert eigen_num > 0
        self.eigen_num = eigen_num
        fd_num = (self.sub_grid + 1)**2
        loc_data_len = ST.N_V**2
        eigen_vec = np.zeros((fd_num, self.coarse_elem * self.eigen_num))
        eigen_val = np.zeros((self.coarse_elem * self.eigen_num, ))
        S_mat_list = [] # A list of S matrices, saved here for futural usages
        for coarse_elem_ind in range(self.coarse_elem):
            coarse_elem_ind_y, coarse_elem_ind_x = divmod(coarse_elem_ind, self.coarse_grid)
            I = np.zeros((self.sub_elem * loc_data_len, ), dtype=np.int32)
            J = np.zeros((self.sub_elem * loc_data_len, ), dtype=np.int32)
            A_val, S_val = np.zeros((self.sub_elem * loc_data_len, )), np.zeros((self.sub_elem * loc_data_len, ))
            for sub_elem_ind in range(self.sub_elem):
                sub_elem_ind_y, sub_elem_ind_x = divmod(sub_elem_ind, self.sub_grid)
                fine_elem_ind_x = coarse_elem_ind_x * self.sub_grid + sub_elem_ind_x
                fine_elem_ind_y = coarse_elem_ind_y * self.sub_grid + sub_elem_ind_y
                loc_coeff = self.coeff[fine_elem_ind_x, fine_elem_ind_y]
                loc_kappa = self.kappa[fine_elem_ind_x, fine_elem_ind_y]
                
                I_loc, J_loc = np.zeros((loc_data_len, ), dtype=np.int32), np.zeros((loc_data_len, ), dtype=np.int32)
                A_val_loc, S_val_loc = np.zeros((loc_data_len, )), np.zeros((loc_data_len, ))
                for loc_ind_j in range(ST.N_V):  # i=0,1,2,3
                    for loc_ind_i in range(ST.N_V):  # j=0,1,2,3
                        temp_ind = loc_ind_j*ST.N_V + loc_ind_i
                        iy, ix = divmod(loc_ind_i, 2)
                        I_loc[temp_ind] = (sub_elem_ind_y+iy) * (self.sub_grid+1) + sub_elem_ind_x + ix
                        jy, jx = divmod(loc_ind_j, 2)
                        J_loc[temp_ind] = (sub_elem_ind_y+jy) * (self.sub_grid+1) + sub_elem_ind_x + jx
                        A_val_loc[temp_ind] = ST.get_loc_stiff(loc_coeff, loc_ind_i, loc_ind_j) # scaling
                        S_val_loc[temp_ind] = ST.get_loc_mass(self.h, loc_kappa, loc_ind_i, loc_ind_j) # scaling
                I[sub_elem_ind * loc_data_len: (sub_elem_ind+1) * loc_data_len] = I_loc
                J[sub_elem_ind * loc_data_len: (sub_elem_ind+1) * loc_data_len] = J_loc
                A_val[sub_elem_ind * loc_data_len: (sub_elem_ind+1) * loc_data_len] = A_val_loc
                S_val[sub_elem_ind * loc_data_len: (sub_elem_ind+1) * loc_data_len] = S_val_loc
            A_mat_coo = coo_matrix((A_val, (I, J)), shape=(fd_num, fd_num)) # A nicer method of constructing FEM matrices
            S_mat_coo = coo_matrix((S_val, (I, J)), shape=(fd_num, fd_num)) # A nicer method of constructing FEM matrices
            A_mat = A_mat_coo.tocsr()
            S_mat = S_mat_coo.tocsr()
            val, vec = eigsh(A_mat, k=self.eigen_num, M=S_mat, sigma=-1.0, which='LM') # Refer Scipy documents
            eigen_val[coarse_elem_ind * self.eigen_num: (coarse_elem_ind+1) * self.eigen_num] = val
            eigen_vec[:, coarse_elem_ind * self.eigen_num: (coarse_elem_ind+1) * self.eigen_num] = vec
            S_mat_list.append(S_mat)
        return eigen_val, eigen_vec, S_mat_list

