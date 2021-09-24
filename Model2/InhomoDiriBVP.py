import Code.Settings.Setting as ST
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


class ProblemSetting(ST.Setting):
    def init_args(self, n: int = 4, k: int = 1):
        self.eigen_num = n
        self.oversamp_layer = k

    def upd_eigen_num(self, n: int = 4):
        self.eigen_num = n

    def upd_oversamp_layer(self, k: int = 1):
        self.oversamp_layer = k

    def set_coeff(self, coeff: np.ndarray):
        assert coeff.shape == (self.fine_grid, self.fine_grid)
        self.coeff = coeff
        self.kappa = 24.0 * self.coarse_grid**2 * self.coeff
        # Magic formula from the paper

    def get_eigen_pair(self):
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

    def set_Diri_func(self, Diri_func):
        self.Diri_func = Diri_func

    def get_Diri_corr(self, oversamp_layer: int = 1):
        assert oversamp_layer > 0
        self.oversamp_layer = oversamp_layer
        for coarse_elem_ind in range(self.coarse_elem):
            # Get the mapping [global node index]->[freedom index]
            ind_map_dic = {}
            fd_ind = 0
            coarse_elem_ind_y, coarse_elem_ind_x = divmod(coarse_elem_ind, self.coarse_grid)
            for coarse_ngh_elem_ind in range((2*self.oversamp_layer+1)**2): 
            # A loop over elements in the oversampling layer
                coarse_ngh_elem_ind_off_y, coarse_ngh_elem_ind_off_x = divmod(coarse_ngh_elem_ind, 2*self.oversamp_layer+1)
                coarse_ngh_elem_ind_y = coarse_elem_ind_y + coarse_ngh_elem_ind_off_y - self.oversamp_layer  
                coarse_ngh_elem_ind_x = coarse_elem_ind_x + coarse_ngh_elem_ind_off_x - self.oversamp_layer
                # Neighboring element in the global coordinate
                if (0 <= coarse_ngh_elem_ind_y < self.coarse_grid) and (0 <= coarse_ngh_elem_ind_x < self.coarse_grid):
                    for sub_elem_ind in range(self.sub_elem):
                        sub_elem_ind_y, sub_elem_ind_x = divmod(sub_elem_ind, self.sub_grid)
                        fine_elem_ind_y = coarse_ngh_elem_ind_y*self.sub_grid + sub_elem_ind_y
                        fine_elem_ind_x = coarse_ngh_elem_ind_x*self.sub_grid + sub_elem_ind_x
                        # Coordinates of fine elements are always defined globally
                        for loc_ind in range(ST.N_V):
                            loc_ind_y, loc_ind_x = divmod(loc_ind, 2)
                            node_ind_y = fine_elem_ind_y + loc_ind_y
                            node_ind_x = fine_elem_ind_x + loc_ind_x
                            node_ind = node_ind_y*(self.fine_grid+1) + node_ind_x
                            if node_ind_y == 0 or node_ind_y == self.fine_grid or node_ind_x == 0 or node_ind_x == self.fine_grid:
                            # Global Dirichlet boundary
                                ind_map_dic[node_ind] = -1
                            elif coarse_ngh_elem_ind_off_x == 0 and sub_elem_ind_x == 0 and loc_ind_x == 0:
                            # Local left Dirichlet boundary
                                ind_map_dic[node_ind] = -1
                            elif coarse_ngh_elem_ind_off_x == 2*self.oversamp_layer and sub_elem_ind_x == self.sub_grid and loc_ind_x == 1:
                            # Local right Dirichlet boundary
                                ind_map_dic[node_ind] = -1
                            elif coarse_ngh_elem_ind_off_y == 0 and sub_elem_ind_y == 0 and loc_ind_y == 0:
                            # Local down Dirichlet boundary
                                ind_map_dic[node_ind] = -1
                            elif coarse_ngh_elem_ind_off_y == 2*self.oversamp_layer and sub_elem_ind_y == self.sub_grid and loc_ind_y == 1:
                            # Local up Dirichlet boundary
                                ind_map_dic[node_ind] = -1
                            elif not node_ind in ind_map_dic:
                            # This is a freedom node and has not been recorded
                                ind_map_dic[node_ind] = fd_ind
                                fd_ind += 1
            print(dict(sorted(ind_map_dic.items())), fd_ind)
                    

                         
                    






    

