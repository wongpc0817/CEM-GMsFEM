import Code.Settings.Setting as ST
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import lgmres


class ProblemSetting(ST.Setting):
    def init_args(self, n: int = 4, k: int = 1):
        self.eigen_num = n
        self.oversamp_layer = k
        self.tot_fd_num = self.eigen_num * self.coarse_elem


    def upd_eigen_num(self, n: int = 4):
        self.eigen_num = n


    def upd_oversamp_layer(self, k: int = 1):
        self.oversamp_layer = k


    def set_coeff(self, coeff: np.ndarray):
        assert coeff.shape == (self.fine_grid, self.fine_grid)
        self.coeff = coeff
        self.kappa = 24.0 * self.coarse_grid**2 * self.coeff
        # Magic formula from the paper


    def set_Diri_func(self, Diri_func_gx, Diri_func_gy):
        self.Diri_func_gx = Diri_func_gx
        self.Diri_func_gy = Diri_func_gy


    def set_source_func(self, source_func):
        self.source_func = source_func


    def get_Diri_quad_fine_elem(self, fine_elem_ind_x, fine_elem_ind_y, loc_ind):
        # Compute \int_{K_h} A\nabla g\cdot \nable L_i
        val = 0.0
        h = self.h
        center_x, center_y = 0.5*h*(2*fine_elem_ind_x+1), 0.5*h*(2*fine_elem_ind_y+1)
        for quad_ind_x in range(ST.QUAD_ORDER):
            for quad_ind_y in range(ST.QUAD_ORDER):
                quad_cord_x, quad_cord_y = ST.QUAD_CORD[quad_ind_x], ST.QUAD_CORD[quad_ind_y]
                quad_wght_x, quad_wght_y = ST.QUAD_WGHT[quad_ind_x], ST.QUAD_WGHT[quad_ind_y]
                quad_real_cord_x = center_x + 0.5*h*quad_cord_x
                quad_real_cord_y = center_y + 0.5*h*quad_cord_y
                test_grad_x, test_grad_y = ST.get_locbase_grad_val(loc_ind, quad_cord_x, quad_cord_y)
                Diri_grad_x = self.Diri_func_gx(quad_real_cord_x, quad_real_cord_y) 
                Diri_grad_y = self.Diri_func_gy(quad_real_cord_x, quad_real_cord_y)
                val += 0.5 * h * quad_wght_x * quad_wght_y * (Diri_grad_x*test_grad_x+Diri_grad_y*test_grad_y)
                # Be careful with scaling
        return val


    def get_source_quad_fine_elem(self, fine_elem_ind_x, fine_elem_ind_y, loc_ind):
        return 0.0


    def get_stiff_quad_by_node_val(self, fine_elem_ind_x, fine_elem_ind_y, node_val_i, node_val_j):
        # Compute \int_{K_h} A\nabla u \cdot \nabla v, where the values of u, v on nodes of the fine element K_h is given
        val = 0.0
        for quad_ind_x in range(ST.QUAD_ORDER):
            for quad_ind_y in range(ST.QUAD_ORDER):
                quad_cord_x, quad_cord_y = ST.QUAD_CORD[quad_ind_x], ST.QUAD_CORD[quad_ind_y]
                quad_wght_x, quad_wght_y = ST.QUAD_WGHT[quad_ind_x], ST.QUAD_WGHT[quad_ind_y]
                grad_x_of_i, grad_y_of_i, grad_x_of_j, grad_y_of_j = 0.0, 0.0, 0.0, 0.0
                for loc_ind in range(ST.N_V):
                    Lag_grad_x, Lag_grad_y = ST.get_locbase_grad_val(loc_ind, quad_cord_x, quad_cord_y)
                    grad_x_of_i += node_val_i[loc_ind] * Lag_grad_x
                    grad_y_of_i += node_val_i[loc_ind] * Lag_grad_y
                    grad_x_of_j += node_val_j[loc_ind] * Lag_grad_x
                    grad_y_of_j += node_val_j[loc_ind] * Lag_grad_y
                val += self.coeff[fine_elem_ind_x, fine_elem_ind_y] * (grad_x_of_i*grad_x_of_j+grad_y_of_i*grad_y_of_j)
        return val       


    def get_eigen_pair(self):
        assert self.eigen_num > 0
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
            for eigen_ind in range(self.eigen_num):
                norm_s = np.sqrt(np.inner(S_mat.dot(vec[:, eigen_ind]), vec[:, eigen_ind]))
                vec[:, eigen_ind] = vec[:, eigen_ind] / norm_s
            # Normalize the eigenvectors
            eigen_val[coarse_elem_ind * self.eigen_num: (coarse_elem_ind+1) * self.eigen_num] = val
            eigen_vec[:, coarse_elem_ind * self.eigen_num: (coarse_elem_ind+1) * self.eigen_num] = vec
            S_mat_list.append(S_mat)
        self.eigen_val = eigen_val
        self.eigen_vec = eigen_vec
        self.S_mat_list = S_mat_list


    def get_ind_map(self):
        assert self.oversamp_layer > 0
        self.ind_map_list = []
        # A list of ind_map, ind_map[coarse_elem_ind] is the ind_map
        self.loc_fd_num = np.zeros((self.coarse_elem, ), dtype=np.int32)
        # The number of freedom degrees of local problems
        for coarse_elem_ind in range(self.coarse_elem):
            # Get the mapping ind_map_dic[glb_node_ind] = loc_fd_ind
            ind_map_dic = {}
            fd_ind = 0
            coarse_elem_ind_y, coarse_elem_ind_x = divmod(coarse_elem_ind, self.coarse_grid)
            for coarse_ngh_elem_ind_off in range((2*self.oversamp_layer+1)**2): 
            # A loop over elements in the oversampling layer
                coarse_ngh_elem_ind_off_y, coarse_ngh_elem_ind_off_x = divmod(coarse_ngh_elem_ind_off, 2*self.oversamp_layer+1)
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
                            node_ind = (fine_elem_ind_y + loc_ind_y)*(self.fine_grid+1) + fine_elem_ind_x + loc_ind_x
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
            self.ind_map_list.append(ind_map_dic)
            self.loc_fd_num[coarse_elem_ind_x] = fd_ind


    def get_coarse_ngh_elem_ind(off_ind: int, coarse_elem_ind_x, coarse_elem_ind_y):
            coarse_ngh_elem_ind_off_y, coarse_ngh_elem_ind_off_x = divmod(off_ind, 2*self.oversamp_layer+1)
            coarse_ngh_elem_ind_y = coarse_elem_ind_y + coarse_ngh_elem_ind_off_y - self.oversamp_layer  
            coarse_ngh_elem_ind_x = coarse_elem_ind_x + coarse_ngh_elem_ind_off_x - self.oversamp_layer
            is_elem_indomain = 0 <= coarse_ngh_elem_ind_x < self.coarse_grid and 0 <= coarse_elem_ind_y < self.coarse_grid
            if is_elem_indomain:
                return coarse_elem_ind_y*self.coarse_grid + coarse_ngh_elem_ind_x
            else:
                return -1 


    def get_corr_basis(self):
        assert self.oversamp_layer > 0 and self.eigen_num > 0
        assert len(self.ind_map_list) > 0
        self.corr_list = []
        self.basis_list = []
        max_fine_elem_num = (2*self.oversamp_layer+1)**2 * self.sub_elem
        # The maximal number of fine elements that a oversampled region contains
        loc_data_len = ST.N_V**2
        for coarse_elem_ind in range(self.coarse_elem):
            coarse_elem_ind_y, coarse_elem_ind_x = divmod(coarse_elem_ind, self.coarse_grid)
            fd_num = self.loc_fd_num[coarse_elem_ind]
            ind_map_dic = sef.ind_map.list[coarse_elem_ind]
            I = -np.ones((max_fine_elem_num * loc_data_len, ), dtype=np.int32)
            J = -np.ones((max_fine_elem_num * loc_data_len, ), dtype=np.int32)
            V = -np.zeros((max_fine_elem_num * loc_data_len))
            marker = 0
            # Vectors I, J and V are used to construct a coo_matrix, integer marker determines the valid range of vectors,
            # such as I[:marker], J[:marker]. Because we do not know how many indeces and values need to be inserted. 
            rhs_corr = np.zeros((fd_num, ))
            rhs_basis = np.zeros((fd_num, self.eigen_num))
            # Right-hand vectors of correctors and multiscale bases 
            for coarse_ngh_elem_ind_off in range((2*self.oversamp_layer+1)**2): 
            # A loop over elements in the oversampling layer
                coarse_ngh_elem_ind_off_y, coarse_ngh_elem_ind_off_x = divmod(coarse_ngh_elem_ind_off, 2*self.oversamp_layer+1)
                coarse_ngh_elem_ind_y = coarse_elem_ind_y + coarse_ngh_elem_ind_off_y - self.oversamp_layer  
                coarse_ngh_elem_ind_x = coarse_elem_ind_x + coarse_ngh_elem_ind_off_x - self.oversamp_layer
                coarse_ngh_elem_ind = coarse_elem_ind_y*self.coarse_grid + coarse_ngh_elem_ind_x
                # Neighboring element in the global coordinate
                is_elem_indomain = (0 <= coarse_ngh_elem_ind_y < self.coarse_grid) and (0 <= coarse_ngh_elem_ind_x < self.coarse_grid)
                if is_elem_indomain:
                # Select the real neighboring elements in the domain
                    S_mat = self.S_mat_list[coarse_ngh_elem_ind]
                    # Retreive the S matrix of the current neighboring coarse element
                    P_mat = S_mat.dot(self.eigen_vec[:, coarse_ngh_elem_ind*self.eigen_num: (coarse_ngh_elem_ind+1)*self.eigen_num])
                    # Let v=\sum_i v_i L_i where L_i is the Lagrange basis corresponding to i-th node.
                    # Then s(\pi v, \phi^eigen_ind) = \sum_i v_iP[i, eigen_ind]
                    for sub_elem_ind in range(self.sub_elem):
                        sub_elem_ind_y, sub_elem_ind_x = divmod(sub_elem_ind, self.sub_grid)
                        fine_elem_ind_y = coarse_ngh_elem_ind_y*self.sub_grid + sub_elem_ind_y
                        fine_elem_ind_x = coarse_ngh_elem_ind_x*self.sub_grid + sub_elem_ind_x
                        # Coordinates of fine elements are always defined globally
                        for loc_ind_i in range(ST.N_V):
                            loc_ind_iy, loc_ind_ix = divmod(loc_ind, 2)
                            node_ind_i = (fine_elem_ind_y+loc_ind_iy)*(self.fine_grid+1) + fine_elem_ind_x + loc_ind_ix
                            # Coordinates of nodes are always defined globally
                            node_sub_ind_i = (sub_elem_ind_y+loc_ind_iy)*(self.sub_grid+1) + sub_elem_ind_x + loc_ind_ix
                            # The node index w.r.t. the current coarse element [0, (self.sub_grid+1)*(self.sub_grid+1))
                            assert node_ind_i in ind_map_dic
                            fd_ind_i = ind_map_dic[node_ind_i]
                            # The freedom index of the node in the oversampled region
                            for loc_ind_j in range(ST.N_V):
                                loc_ind_jy, loc_ind_jx = divmod(loc_ind, 2)
                                node_ind_j = (fine_elem_ind_y+loc_ind_jy)*(self.fine_grid+1) + fine_elem_ind_x + loc_ind_jx
                                node_sub_ind_j = (sub_elem_ind_y+loc_ind_jy)*(self.sub_grid+1) + sub_elem_ind_x + loc_ind_jx
                                assert node_ind_j in ind_map_dic
                                fd_ind_j = ind_map_dic[node_ind_j]
                                if fd_ind_i >= 0 and fd_ind_j >= 0:
                                # If true, those two nodes are freedom nodes
                                    loc_coeff = self.coeff[fine_elem_ind_x, fine_elem_ind_y]
                                    temp = ST.get_loc_stiff(loc_coeff, loc_ind_i, loc_ind_j)
                                    # The frist term a(L_i, L_j)
                                    temp += np.inner(P_mat[node_sub_ind_i, :], P_mat[node_sub_ind_j, :])
                                    # The second term s(\pi L_i, \pi L_j)
                                    I[marker] = fd_ind_i
                                    J[marker] = fd_ind_j
                                    V[marker] = temp
                                    marker += 1
                                    # Prepare data for constructing coo_matrix

                            if fd_ind_i >= 0 and coarse_elem_ind == coarse_ngh_elem_ind:
                            # Construct the right-hand vectors for solving bases and Dirichlet boundary correctors
                            # Note that only the integral on K_i is summed from the paper
                                rhs_corr[fd_ind_i] += self.get_Diri_quad_fine_elem(fine_elem_ind_x, fine_elem_ind_y, loc_ind)
                                for eigen_ind in range(self.eigen_num):
                                    rhs_basis[fd_ind_i, eigen_ind] += P_mat[node_sub_ind_i, eigen_ind]
                                    # Use the definition of P_mat
            Op_mat_coo = coo_matrix((V[:marker], (I[:marker], J[:marker])), shape=(fd_num, fd_num))
            Op_mat = Op_mat_coo.tocsr()
            corr, info = lgmres(Op_mat, rhs_corr)
            assert info == 0
            for eigen_ind in range(self.eigen_num):
                basis, info = lgmres(Op_mat, rhs_basis[:, eigen_ind])
                assert info == 0
            self.corr_list.append(corr)
            self.basis_list.append(basis)


    def solve(self):
        assert self.oversamp_layer > 0 and self.eigen_num > 0
        assert len(self.ind_map_list) > 0
        assert len(self.corr_list) > 0 and len(self.basis_list) > 0
        max_data_len = self.coarse_elem * (2*self.oversamp_layer+1)**4
        I, J = -np.ones((max_data_len, ), dtype=np.int32), -np.ones((max_data_len, ), dtype=np.int32)
        V = np.zeros((max_data_len))
        marker = 0
        # As previously introduced, vectors I, J and V are used to construct the coo_matrix of the final linear system
        for coarse_elem_ind in range(self.coarse_elem):
            coarse_elem_ind_y, coarse_elem_ind_x = divmod(coarse_elem_ind, self.coarse_grid)
            for coarse_ngh_elem_ind_off_i in range((2*self.oversamp_layer+1)**2): 
                coarse_ngh_elem_ind_i = self.get_coarse_ngh_elem_ind(coarse_elem_ind_off_i, coarse_elem_ind_x, coarse_elem_ind_y)
                for coarse_elem_ind_off_j in range((2*self.oversamp_layer+1)**2):
                    coarse_ngh_elem_ind_j = self.get_coarse_ngh_elem_ind(coarse_elem_ind_off_j, coarse_elem_ind_x, coarse_elem_ind_y)
                    # This loop over K_n is used to compute \int_{K_n} A\nabla \Phi_j^s \cdot \nabla \Phi_i^t,
                    # where \supp(\Phi_j^s) \cap K_n \neq \empty and \supp(\Phi_j^t) \cap \neq \empty.
                    if coarse_ngh_elem_ind_i >= 0 and coarse_ngh_elem_ind_j >= 0:
                        for eigen_ind_i in range(self.eigen_num):
                            fd_ind_i = coarse_ngh_elem_ind_i*self.eigen_num + eigen_ind_i
                            for eigen_ind_j in range(self.eigen_num):
                                fd_ind_j = coarse_ngh_elem_ind_j*self.eigen_num + eigen_ind_j
                                for sub_elem_ind in range(self.sub_elem):
                                    sub_elem_ind_y, sub_elem_ind_x = divmod(sub_elem_ind, self.sub_grid)
                                    fine_elem_ind_y = coarse_elem_ind_y*self.sub_grid + sub_elem_ind_y
                                    fine_elem_ind_x = coarse_elem_ind_x*self.sub_grid + sub_elem_ind_x
                                    node_val_i = [0.0, 0.0, 0.0, 0.0]
                                    node_val_j = [0.0, 0.0, 0.0, 0.0]
                                    for loc_ind in range(ST.N_V):
                                        loc_ind_i, loc_ind_i = divmod(loc_ind, 2)
                                        node_ind = node_ind = node_ind_y*(self.fine_grid+1) + node_ind_x

                                                            