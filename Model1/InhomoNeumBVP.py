import Settings.Setting as ST
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import LinearOperator

import logging


class ProblemSetting(ST.Setting):
    def init_args(self, n: int = 4, k: int = 1):
        self.eigen_num = n
        self.oversamp_layer = k
        self.tot_fd_num = self.eigen_num * self.coarse_elem

    def set_coeff(self, coeff: np.ndarray):
        self.coeff = coeff
        self.kappa = 24.0 * self.coarse_grid**2 * self.coeff

    def set_beta_func(self,beta_func):
        self.beta_func=beta_func

    def set_source_func(self, source_func):
        self.source_func = source_func

    def set_Neum_func(self, Neum_func_lf, Neum_func_rg, Neum_func_dw):
        self.Neum_func_lf = Neum_func_lf
        self.Neum_func_rg = Neum_func_rg
        self.Neum_func_dw = Neum_func_dw

    def get_coarse_ngh_elem_lim(self, coarse_elem_ind):
        coarse_elem_ind_y, coarse_elem_ind_x = divmod(coarse_elem_ind, self.coarse_grid)
        coarse_ngh_elem_lf_lim = max(0, coarse_elem_ind_x - self.oversamp_layer)
        coarse_ngh_elem_rg_lim = min(self.coarse_grid, coarse_elem_ind_x + self.oversamp_layer + 1)
        coarse_ngh_elem_dw_lim = max(0, coarse_elem_ind_y - self.oversamp_layer)
        coarse_ngh_elem_up_lim = min(self.coarse_grid, coarse_elem_ind_y + self.oversamp_layer + 1)
        return coarse_ngh_elem_lf_lim, coarse_ngh_elem_rg_lim, coarse_ngh_elem_dw_lim, coarse_ngh_elem_up_lim

    def get_Neum_quad_Lag(self, fine_elem_ind_x, fine_elem_ind_y, loc_ind):
        # Compute $\int_{\partial K_h} q L_i d\sigma$ 
        val = 0.0
        h = self.h
        if fine_elem_ind_x == 0 and loc_ind in [0, 2]:
            center_y = 0.5 * h * (2 * fine_elem_ind_y + 1)
            for quad_ind in range(ST.QUAD_ORDER):
                quad_cord = ST.QUAD_CORD[quad_ind]
                quad_wght = ST.QUAD_WGHT[quad_ind]
                quad_real_cord_y = center_y + 0.5 * h * quad_cord
                q_val = self.Neum_func_lf(quad_real_cord_y)
                test_val = ST.get_locbase_val(loc_ind, -1.0, quad_cord)
                val += 0.5 * h * quad_wght * q_val * test_val
        if fine_elem_ind_x == self.fine_grid - 1 and loc_ind in [1, 3]:
            center_y = 0.5 * h * (2 * fine_elem_ind_y + 1)
            for quad_ind in range(ST.QUAD_ORDER):
                quad_cord = ST.QUAD_CORD[quad_ind]
                quad_wght = ST.QUAD_WGHT[quad_ind]
                quad_real_cord_y = center_y + 0.5 * h * quad_cord
                q_val = self.Neum_func_rg(quad_real_cord_y)
                test_val = ST.get_locbase_val(loc_ind, 1.0, quad_cord)
                val += 0.5 * h * quad_wght * q_val * test_val
        if fine_elem_ind_y == 0 and loc_ind in [0, 1]:
            center_x = 0.5 * h * (2 * fine_elem_ind_x + 1)
            for quad_ind in range(ST.QUAD_ORDER):
                quad_cord = ST.QUAD_CORD[quad_ind]
                quad_wght = ST.QUAD_WGHT[quad_ind]
                quad_real_cord_x = center_x + 0.5 * h * quad_cord
                q_val = self.Neum_func_dw(quad_real_cord_x)
                test_val = ST.get_locbase_val(loc_ind, quad_cord, -1.0)
                val += 0.5 * h * quad_wght * q_val * test_val
        return val

    def get_source_quad_Lag(self, fine_elem_ind_x, fine_elem_ind_y, loc_ind):
        val = 0.0
        h = self.h
        center_x, center_y = 0.5 * h * (2 * fine_elem_ind_x + 1), 0.5 * h * (2 * fine_elem_ind_y + 1)
        for quad_ind_x in range(ST.QUAD_ORDER):
            for quad_ind_y in range(ST.QUAD_ORDER):
                quad_cord_x, quad_cord_y = ST.QUAD_CORD[quad_ind_x], ST.QUAD_CORD[quad_ind_y]
                quad_wght_x, quad_wght_y = ST.QUAD_WGHT[quad_ind_x], ST.QUAD_WGHT[quad_ind_y]
                quad_real_cord_x = center_x + 0.5 * h * quad_cord_x
                quad_real_cord_y = center_y + 0.5 * h * quad_cord_y
                test_val = ST.get_locbase_val(loc_ind, quad_cord_x, quad_cord_y)
                f_val = self.source_func(quad_real_cord_x, quad_real_cord_y)
                val += 0.25 * h**2 * quad_wght_x * quad_wght_y * f_val * test_val
        return val

    def get_glb_vec(self, coarse_elem_ind, vec):
        ind_map_rev = self.ind_map_rev_list[coarse_elem_ind]
        glb_vec = np.zeros((self.tot_node, ))
        for loc_fd_ind, node_ind in ind_map_rev.items():
            glb_vec[node_ind] = vec[loc_fd_ind]
        return glb_vec

    def get_L2_energy_norm(self, u):
        assert self.glb_A_mat != None
        val0 = self.h * np.linalg.norm(u)
        val1 = np.sqrt(u @ self.glb_A_mat.dot(u))
        return val0, val1

    def get_eigen_pair(self):
        assert self.eigen_num > 0
        fd_num = (self.sub_grid + 1)**2
        eigen_vec = np.zeros((fd_num, self.coarse_elem * self.eigen_num))
        eigen_val = np.zeros((self.coarse_elem * self.eigen_num, ))
        S_mat_list = [None] * self.coarse_elem  # A list of S matrices, saved here for futural usages
        # A_mat_list = [None] * self.coarse_elem
        for coarse_elem_ind in range(self.coarse_elem):
            coarse_elem_ind_y, coarse_elem_ind_x = divmod(coarse_elem_ind, self.coarse_grid)
            max_data_len = self.sub_elem * ST.N_V**2
            I = np.zeros((max_data_len, ), dtype=np.int32)
            J = np.zeros((max_data_len, ), dtype=np.int32)
            A_val, S_val = np.zeros((max_data_len, )), np.zeros((max_data_len, ))
            marker = 0
            for sub_elem_ind in range(self.sub_elem):
                sub_elem_ind_y, sub_elem_ind_x = divmod(sub_elem_ind, self.sub_grid)
                fine_elem_ind_x = coarse_elem_ind_x * self.sub_grid + sub_elem_ind_x
                fine_elem_ind_y = coarse_elem_ind_y * self.sub_grid + sub_elem_ind_y
                loc_coeff = self.coeff[fine_elem_ind_x, fine_elem_ind_y]
                loc_kappa = self.kappa[fine_elem_ind_x, fine_elem_ind_y]
                # I_loc, J_loc = np.zeros((loc_data_len, ), dtype=np.int32), np.zeros((loc_data_len, ), dtype=np.int32)
                # A_val_loc, S_val_loc = np.zeros((loc_data_len, )), np.zeros((loc_data_len, ))
                for loc_ind_i in range(ST.N_V):  # i=0,1,2,3
                    for loc_ind_j in range(ST.N_V):  # j=0,1,2,3
                        # temp_ind = loc_ind_j*ST.N_V + loc_ind_i
                        iy, ix = divmod(loc_ind_i, 2)
                        # I_loc[temp_ind] = (sub_elem_ind_y+iy) * (self.sub_grid+1) + sub_elem_ind_x + ix
                        I[marker] = (sub_elem_ind_y + iy) * (self.sub_grid + 1) + sub_elem_ind_x + ix
                        jy, jx = divmod(loc_ind_j, 2)
                        # J_loc[temp_ind] = (sub_elem_ind_y+jy) * (self.sub_grid+1) + sub_elem_ind_x + jx
                        J[marker] = (sub_elem_ind_y + jy) * (self.sub_grid + 1) + sub_elem_ind_x + jx
                        # A_val_loc[temp_ind] = ST.get_loc_stiff(loc_coeff, loc_ind_i, loc_ind_j) # scaling
                        # S_val_loc[temp_ind] = ST.get_loc_mass(self.h, loc_kappa, loc_ind_i, loc_ind_j) # scaling
                        A_val[marker] = loc_coeff * (self.elem_Lap_stiff_mat[loc_ind_i, loc_ind_j])
                        S_val[marker] = loc_kappa * (self.elem_Bi_mass_mat[loc_ind_i, loc_ind_j])
                        marker += 1
                # I[sub_elem_ind * loc_data_len: (sub_elem_ind+1) * loc_data_len] = I_loc
                # J[sub_elem_ind * loc_data_len: (sub_elem_ind+1) * loc_data_len] = J_loc
                # A_val[sub_elem_ind * loc_data_len: (sub_elem_ind+1) * loc_data_len] = A_val_loc
                # S_val[sub_elem_ind * loc_data_len: (sub_elem_ind+1) * loc_data_len] = S_val_loc
            A_mat_coo = coo_matrix((A_val[:marker], (I[:marker], J[:marker])), shape=(fd_num, fd_num))  # A nicer method of constructing FEM matrices
            S_mat_coo = coo_matrix((S_val[:marker], (I[:marker], J[:marker])), shape=(fd_num, fd_num))  # A nicer method of constructing FEM matrices
            A_mat = A_mat_coo.tocsc()
            S_mat = S_mat_coo.tocsc()
            val, vec = eigsh(A_mat, k=self.eigen_num, M=S_mat, sigma=-1.0, which='LM')  # Refer Scipy documents
            # All eigenvectors are orthogonal to each other w.r.t. S_mat
            # for eigen_ind in range(self.eigen_num):
            #     norm_s = np.sqrt(np.inner(S_mat.dot(vec[:, eigen_ind]), vec[:, eigen_ind]))
            #     vec[:, eigen_ind] = vec[:, eigen_ind] / norm_s
            # Normalize the eigenvectors
            eigen_val[coarse_elem_ind * self.eigen_num:(coarse_elem_ind + 1) * self.eigen_num] = val
            eigen_vec[:, coarse_elem_ind * self.eigen_num:(coarse_elem_ind + 1) * self.eigen_num] = vec
            S_mat_list[coarse_elem_ind] = S_mat
            # A_mat_list[coarse_elem_ind] = A_mat
        self.eigen_val = eigen_val
        self.eigen_vec = eigen_vec
        self.S_mat_list = S_mat_list
        # self.A_mat_list = A_mat_list

    def get_ind_map(self):
        assert self.oversamp_layer > 0
        self.ind_map_list = [None] * self.coarse_elem
        self.ind_map_rev_list = [None] * self.coarse_elem
        # A list of ind_map, ind_map[coarse_elem_ind] is the ind_map
        # and the reverse map list
        self.loc_fd_num = np.zeros((self.coarse_elem, ), dtype=np.int32)
        # The number of freedom degrees of local problems
        for coarse_elem_ind in range(self.coarse_elem):
            # Get the mapping ind_map_dic[glb_node_ind] = loc_fd_ind
            # and the reverse mapping ind_map_rev_dic[loc_fd_ind] = glb_node_ind
            ind_map_dic = {}
            ind_map_rev_dic = {}
            fd_ind = 0
            lf_lim, rg_lim, dw_lim, up_lim = self.get_coarse_ngh_elem_lim(coarse_elem_ind)
            if lf_lim == 0:
                node_ind_x_lf_lim = 0
            else:
                node_ind_x_lf_lim = lf_lim * self.sub_grid + 1

            if rg_lim == self.coarse_grid:
                node_ind_x_rg_lim = self.fine_grid + 1
            else:
                node_ind_x_rg_lim = rg_lim * self.sub_grid

            if dw_lim == 0:
                node_ind_y_dw_lim = 0
            else:
                node_ind_y_dw_lim = dw_lim * self.sub_grid + 1

            node_ind_y_up_lim = up_lim * self.sub_grid

            for node_ind_y in range(node_ind_y_dw_lim, node_ind_y_up_lim):
                for node_ind_x in range(node_ind_x_lf_lim, node_ind_x_rg_lim):
                    node_ind = node_ind_y * (self.fine_grid + 1) + node_ind_x
                    ind_map_dic[node_ind] = fd_ind
                    ind_map_rev_dic[fd_ind] = node_ind
                    fd_ind += 1
            self.ind_map_list[coarse_elem_ind] = ind_map_dic
            self.ind_map_rev_list[coarse_elem_ind] = ind_map_rev_dic
            self.loc_fd_num[coarse_elem_ind] = fd_ind

    def get_corr_basis(self):
        assert self.oversamp_layer > 0 and self.eigen_num > 0
        assert len(self.ind_map_list) > 0
        glb_corr = np.zeros((self.tot_node, ))
        basis_list = [None] * self.coarse_elem
        max_data_len = (2 * self.oversamp_layer + 1)**2 * ((self.sub_grid + 1)**4 + self.sub_elem * ST.N_V**2)
        prc_flag = 1
        for coarse_elem_ind in range(self.coarse_elem):
            fd_num = self.loc_fd_num[coarse_elem_ind]
            ind_map_dic = self.ind_map_list[coarse_elem_ind]
            I = -np.ones((max_data_len, ), dtype=np.int32)
            J = -np.ones((max_data_len, ), dtype=np.int32)
            V = np.zeros((max_data_len, ))
            marker = 0
            rhs_corr = np.zeros((fd_num, ))
            rhs_basis = np.zeros((fd_num, self.eigen_num))
            guess = np.zeros(rhs_basis.shape)
            lf_lim, rg_lim, dw_lim, up_lim = self.get_coarse_ngh_elem_lim(coarse_elem_ind)
            for coarse_ngh_elem_ind_y in range(dw_lim, up_lim):
                for coarse_ngh_elem_ind_x in range(lf_lim, rg_lim):
                    coarse_ngh_elem_ind = coarse_ngh_elem_ind_y * self.coarse_grid + coarse_ngh_elem_ind_x
                    S_mat = self.S_mat_list[coarse_ngh_elem_ind]
                    eigen_vec = self.eigen_vec[:, coarse_ngh_elem_ind * self.eigen_num:(coarse_ngh_elem_ind + 1) * self.eigen_num]
                    P_mat = S_mat.dot(eigen_vec)
                    Q_mat = P_mat.dot(P_mat.T)
                    node_sub_ind_list = -np.ones(((self.sub_grid + 1)**2, ), dtype=np.int32)
                    fd_ind_list = -np.ones(((self.sub_grid + 1)**2, ), dtype=np.int32)
                    marker_ = 0
                    for node_sub_ind in range((self.sub_grid + 1)**2):
                        node_sub_ind_y, node_sub_ind_x = divmod(node_sub_ind, self.sub_grid + 1)
                        node_ind_y = coarse_ngh_elem_ind_y * self.sub_grid + node_sub_ind_y
                        node_ind_x = coarse_ngh_elem_ind_x * self.sub_grid + node_sub_ind_x
                        node_ind = node_ind_y * (self.fine_grid + 1) + node_ind_x
                        if node_ind in ind_map_dic:
                            fd_ind = ind_map_dic[node_ind]
                            node_sub_ind_list[marker_] = node_sub_ind
                            fd_ind_list[marker_] = fd_ind
                            marker_ += 1
                    for ind_i in range(marker_):
                        node_sub_ind_i = node_sub_ind_list[ind_i]
                        fd_ind_i = fd_ind_list[ind_i]
                        for ind_j in range(marker_):
                            node_sub_ind_j = node_sub_ind_list[ind_j]
                            fd_ind_j = fd_ind_list[ind_j]
                            I[marker] = fd_ind_i
                            J[marker] = fd_ind_j
                            V[marker] = Q_mat[node_sub_ind_i, node_sub_ind_j]
                            marker += 1
                        if coarse_ngh_elem_ind == coarse_elem_ind:
                            for eigen_ind in range(self.eigen_num):
                                rhs_basis[fd_ind_i, eigen_ind] += P_mat[node_sub_ind_i, eigen_ind]
                                guess[fd_ind_i, eigen_ind] += eigen_vec[node_sub_ind_i, eigen_ind]
                    for sub_elem_ind in range(self.sub_elem):
                        sub_elem_ind_y, sub_elem_ind_x = divmod(sub_elem_ind, self.sub_grid)
                        fine_elem_ind_y = coarse_ngh_elem_ind_y * self.sub_grid + sub_elem_ind_y
                        fine_elem_ind_x = coarse_ngh_elem_ind_x * self.sub_grid + sub_elem_ind_x
                        loc_coeff = self.coeff[fine_elem_ind_x, fine_elem_ind_y]
                        for loc_ind_i in range(ST.N_V):
                            loc_ind_iy, loc_ind_ix = divmod(loc_ind_i, 2)
                            node_ind_i = (fine_elem_ind_y + loc_ind_iy) * (self.fine_grid + 1) + fine_elem_ind_x + loc_ind_ix
                            if node_ind_i in ind_map_dic:
                                fd_ind_i = ind_map_dic[node_ind_i]
                                for loc_ind_j in range(ST.N_V):
                                    loc_ind_jy, loc_ind_jx = divmod(loc_ind_j, 2)
                                    node_ind_j = (fine_elem_ind_y + loc_ind_jy) * (self.fine_grid + 1) + fine_elem_ind_x + loc_ind_jx
                                    if node_ind_j in ind_map_dic:
                                        fd_ind_j = ind_map_dic[node_ind_j]
                                        I[marker] = fd_ind_i
                                        J[marker] = fd_ind_j
                                        V[marker] = loc_coeff * (self.elem_Lap_stiff_mat[loc_ind_i, loc_ind_j])
                                        marker += 1
                                if coarse_ngh_elem_ind == coarse_elem_ind:
                                    rhs_corr[fd_ind_i] += self.get_Neum_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
            Op_mat_coo = coo_matrix((V[:marker], (I[:marker], J[:marker])), shape=(fd_num, fd_num))
            Op_mat = Op_mat_coo.tocsc()
            # logging.info("Construct the linear system [{0:d}]/[{1:d}], [{2:d}x{2:d}]".format(coarse_elem_ind, self.coarse_elem, fd_num))
            ilu = spilu(Op_mat)
            Mx = lambda x: ilu.solve(x)
            pre_M = LinearOperator((fd_num, fd_num), Mx)
            corr, info = lgmres(Op_mat, rhs_corr, tol=self.TOL, M=pre_M)
            assert info == 0
            glb_corr += self.get_glb_vec(coarse_elem_ind, corr)
            basis_wrt_coarse_elem = np.zeros(rhs_basis.shape)
            for eigen_ind in range(self.eigen_num):
                basis, info = lgmres(Op_mat, rhs_basis[:, eigen_ind], x0=guess[:, eigen_ind], tol=self.TOL, M=pre_M)
                assert info == 0
                basis_wrt_coarse_elem[:, eigen_ind] = basis
            basis_list[coarse_elem_ind] = basis_wrt_coarse_elem
            # logging.info("Finish [{0:d}]/[{1:d}]".format(coarse_elem_ind, self.coarse_elem))
            if coarse_elem_ind > prc_flag / 10 * self.coarse_elem:
                logging.info("......{0:.2f}%".format(coarse_elem_ind / self.coarse_elem * 100.))
                prc_flag += 1
        self.glb_corr = glb_corr
        self.basis_list = basis_list

    def get_glb_A_F(self):
        max_data_len = self.fine_elem * ST.N_V**2
        I = -np.ones((max_data_len, ), dtype=np.int32)
        J = -np.ones((max_data_len, ), dtype=np.int32)
        V = np.zeros((max_data_len, ))
        V1 = np.zeros((max_data_len, ))
        marker = 0
        tot_node = self.tot_node
        glb_F_vec = np.zeros((tot_node, ))
        for fine_elem_ind in range(self.fine_elem):
            fine_elem_ind_y, fine_elem_ind_x = divmod(fine_elem_ind, self.fine_grid)
            loc_coeff = self.coeff[fine_elem_ind_x, fine_elem_ind_y]
            for loc_ind_i in range(ST.N_V):
                loc_ind_iy, loc_ind_ix = divmod(loc_ind_i, 2)
                node_ind_i = (fine_elem_ind_y + loc_ind_iy) * (self.fine_grid + 1) + fine_elem_ind_x + loc_ind_ix
                for loc_ind_j in range(ST.N_V):
                    loc_ind_jy, loc_ind_jx = divmod(loc_ind_j, 2)
                    node_ind_j = (fine_elem_ind_y + loc_ind_jy) * (self.fine_grid + 1) + fine_elem_ind_x + loc_ind_jx
                    I[marker] = node_ind_i
                    J[marker] = node_ind_j
                    V[marker] = loc_coeff * self.elem_Lap_stiff_mat[loc_ind_i, loc_ind_j]
                    V1[marker] = loc_coeff * self.elem_Adv_mat[loc_ind_i, loc_ind_j]
                    marker += 1
                glb_F_vec[node_ind_i] += self.get_source_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
                glb_F_vec[node_ind_i] += self.get_Neum_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
        glb_A_mat_coo = coo_matrix((V[:marker], (I[:marker], J[:marker])), shape=(tot_node, tot_node))
        glb_A_mat = glb_A_mat_coo.tocsc()
        glb_A_mat_coo1 = coo_matrix((V1[:marker], (I[:marker], J[:marker])), shape=(tot_node, tot_node))
        glb_A_mat1 = glb_A_mat_coo1.tocsc()
        self.glb_A_mat = glb_A_mat
        self.glb_F_vec = glb_F_vec
        self.glb_Adv_mat = glb_A_mat1

    def get_glb_basis_spmat(self):
        max_data_len = np.sum(self.loc_fd_num) * self.eigen_num
        I = -np.ones((max_data_len, ), dtype=np.int32)
        J = -np.ones((max_data_len, ), dtype=np.int32)
        V = np.zeros((max_data_len, ))
        marker = 0
        for coarse_elem_ind in range(self.coarse_elem):
            for eigen_ind in range(self.eigen_num):
                fd_ind = coarse_elem_ind * self.eigen_num + eigen_ind
                for loc_fd_ind, node_ind in self.ind_map_rev_list[coarse_elem_ind].items():
                    I[marker] = node_ind
                    J[marker] = fd_ind
                    V[marker] = self.basis_list[coarse_elem_ind][loc_fd_ind, eigen_ind]
                    marker += 1
        glb_basis_spmat = csc_matrix((V[:marker], (I[:marker], J[:marker])), shape=(self.tot_node, self.tot_fd_num))
        glb_basis_spmat_T = csc_matrix((V[:marker], (J[:marker], I[:marker])), shape=(self.tot_fd_num, self.tot_node))
        self.glb_basis_spmat = glb_basis_spmat
        self.glb_basis_spmat_T = glb_basis_spmat_T

    def solve(self, guess=[]):
        assert self.oversamp_layer > 0 and self.eigen_num > 0
        self.get_eigen_pair()
        logging.info("Finish getting all eigenvalue-vector pairs.")
        self.get_ind_map()
        logging.info("Finish getting maps of [global node index] to [local freedom index].")
        self.get_corr_basis()
        logging.info("Finish getting the Neumann corrector and multiscale bases.")
        self.get_glb_A_F()
        logging.info("Finish getting the global stiffness matrix and right-hand vector.")
        self.get_glb_basis_spmat()
        logging.info("Finish collecting all the bases in a sparse matrix formation.")
        A_mat = self.glb_basis_spmat_T * (self.glb_A_mat+self.glb_Adv_mat) * self.glb_basis_spmat
        rhs = self.glb_basis_spmat_T.dot(self.glb_F_vec - self.glb_A_mat.dot(self.glb_corr))
        logging.info("Finish constructing the final linear system.")
        ilu = spilu(A_mat)
        Mx = lambda x: ilu.solve(x)
        pre_M = LinearOperator((self.tot_fd_num, self.tot_fd_num), Mx)
        if len(guess) == self.tot_fd_num:
            x0 = guess
        else:
            x0 = np.zeros((self.tot_fd_num, ))
        omega, info = lgmres(A_mat, rhs, tol=self.TOL, M=pre_M, x0=x0)
        if info != 0:
            logging.critical("Fail to solve the final linear system, info={0:d}".format(info))
            raise AssertionError
        u = self.glb_basis_spmat.dot(omega)
        u += self.glb_corr
        logging.info("Finish solving the final linear system.")
        return u, omega

    def solve_ref(self, guess=[]):
        def get_fd_ind(fine_elem_ind_x, fine_elem_ind_y, loc_ind):
            if fine_elem_ind_y == self.fine_grid - 1 and loc_ind in [2, 3]:
                return -1
            else:
                loc_ind_y, loc_ind_x = divmod(loc_ind, 2)
                return (fine_elem_ind_y + loc_ind_y) * (self.fine_grid + 1) + fine_elem_ind_x + loc_ind_x

        max_data_len = self.fine_elem * ST.N_V**2
        I = -np.ones((max_data_len, ), dtype=np.int32)
        J = -np.ones((max_data_len, ), dtype=np.int32)
        V = np.zeros((max_data_len, ))
        marker = 0
        fd_num = self.fine_grid * (self.fine_grid + 1)
        rhs = np.zeros((fd_num, ))
        x0 = np.zeros((fd_num, ))
        if len(guess) > 0:
            x0 = guess[:fd_num]

        for fine_elem_ind in range(self.fine_elem):
            fine_elem_ind_y, fine_elem_ind_x = divmod(fine_elem_ind, self.fine_grid)
            loc_coeff = self.coeff[fine_elem_ind_x, fine_elem_ind_y]
            for loc_ind_i in range(ST.N_V):
                fd_ind_i = get_fd_ind(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
                if fd_ind_i >= 0:
                    for loc_ind_j in range(ST.N_V):
                        fd_ind_j = get_fd_ind(fine_elem_ind_x, fine_elem_ind_y, loc_ind_j)
                        if fd_ind_j >= 0:
                            I[marker] = fd_ind_i
                            J[marker] = fd_ind_j
                            V[marker] = loc_coeff * self.elem_Lap_stiff_mat[loc_ind_i, loc_ind_j]
                            + loc_coeff* self.elem_Adv_mat[loc_ind_i, loc_ind_j]
                            marker += 1
                    rhs[fd_ind_i] += self.get_source_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
                    rhs[fd_ind_i] += self.get_Neum_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
        A_mat_coo = coo_matrix((V[:marker], (I[:marker], J[:marker])), shape=(fd_num, fd_num))
        A_mat = A_mat_coo.tocsc()
        logging.info("Finish constructing the final linear system of the reference problem.")
        ilu = spilu(A_mat)
        Mx = lambda x: ilu.solve(x)
        pre_M = LinearOperator((fd_num, fd_num), Mx)
        u_ref_inner, info = lgmres(A_mat, rhs, tol=self.TOL, x0=x0, M=pre_M)
        assert info == 0
        u_ref = np.zeros((self.tot_node, ))
        u_ref[:fd_num] = u_ref_inner
        logging.info("Finish solving the final linear system of the reference problem.")
        return u_ref

    def get_corr(self):
        assert self.oversamp_layer > 0 and self.eigen_num > 0
        assert len(self.ind_map_list) > 0
        glb_corr = np.zeros((self.tot_node, ))
        max_data_len = (2 * self.oversamp_layer + 1)**2 * ((self.sub_grid + 1)**4 + self.sub_elem * ST.N_V**2)
        prc_flag = 1
        for coarse_elem_ind in range(self.coarse_elem):
            fd_num = self.loc_fd_num[coarse_elem_ind]
            ind_map_dic = self.ind_map_list[coarse_elem_ind]
            I = -np.ones((max_data_len, ), dtype=np.int32)
            J = -np.ones((max_data_len, ), dtype=np.int32)
            V = np.zeros((max_data_len, ))
            marker = 0
            rhs_corr = np.zeros((fd_num, ))
            lf_lim, rg_lim, dw_lim, up_lim = self.get_coarse_ngh_elem_lim(coarse_elem_ind)
            for coarse_ngh_elem_ind_x in range(lf_lim, rg_lim):
                for coarse_ngh_elem_ind_y in range(dw_lim, up_lim):
                    coarse_ngh_elem_ind = coarse_ngh_elem_ind_y * self.coarse_grid + coarse_ngh_elem_ind_x
                    S_mat = self.S_mat_list[coarse_ngh_elem_ind]
                    eigen_vec = self.eigen_vec[:, coarse_ngh_elem_ind * self.eigen_num:(coarse_ngh_elem_ind + 1) * self.eigen_num]
                    P_mat = S_mat.dot(eigen_vec)
                    Q_mat = P_mat.dot(P_mat.T)
                    node_sub_ind_list = [-1] * (self.sub_grid + 1)**2
                    fd_ind_list = [-1] * (self.sub_grid + 1)**2
                    marker_ = 0
                    for node_sub_ind_y in range(self.sub_grid + 1):
                        for node_sub_ind_x in range(self.sub_grid + 1):
                            node_sub_ind = node_sub_ind_y * (self.sub_grid + 1) + node_sub_ind_x
                            node_ind_y = coarse_ngh_elem_ind_y * self.sub_grid + node_sub_ind_y
                            node_ind_x = coarse_ngh_elem_ind_x * self.sub_grid + node_sub_ind_x
                            node_ind = node_ind_y * (self.fine_grid + 1) + node_ind_x
                            if node_ind in ind_map_dic:
                                fd_ind = ind_map_dic[node_ind]
                                node_sub_ind_list[marker_] = node_sub_ind
                                fd_ind_list[marker_] = fd_ind
                                marker_ += 1
                    for ind_i in range(marker_):
                        node_sub_ind_i = node_sub_ind_list[ind_i]
                        fd_ind_i = fd_ind_list[ind_i]
                        for ind_j in range(marker_):
                            node_sub_ind_j = node_sub_ind_list[ind_j]
                            fd_ind_j = fd_ind_list[ind_j]
                            I[marker] = fd_ind_i
                            J[marker] = fd_ind_j
                            V[marker] = Q_mat[node_sub_ind_i, node_sub_ind_j]
                            marker += 1
                    for sub_elem_ind in range(self.sub_elem):
                        sub_elem_ind_y, sub_elem_ind_x = divmod(sub_elem_ind, self.sub_grid)
                        fine_elem_ind_y = coarse_ngh_elem_ind_y * self.sub_grid + sub_elem_ind_y
                        fine_elem_ind_x = coarse_ngh_elem_ind_x * self.sub_grid + sub_elem_ind_x
                        loc_coeff = self.coeff[fine_elem_ind_x, fine_elem_ind_y]
                        for loc_ind_i in range(ST.N_V):
                            loc_ind_iy, loc_ind_ix = divmod(loc_ind_i, 2)
                            node_ind_i = (fine_elem_ind_y + loc_ind_iy) * (self.fine_grid + 1) + fine_elem_ind_x + loc_ind_ix
                            if node_ind_i in ind_map_dic:
                                fd_ind_i = ind_map_dic[node_ind_i]
                                for loc_ind_j in range(ST.N_V):
                                    loc_ind_jy, loc_ind_jx = divmod(loc_ind_j, 2)
                                    node_ind_j = (fine_elem_ind_y + loc_ind_jy) * (self.fine_grid + 1) + fine_elem_ind_x + loc_ind_jx
                                    if node_ind_j in ind_map_dic:
                                        fd_ind_j = ind_map_dic[node_ind_j]
                                        I[marker] = fd_ind_i
                                        J[marker] = fd_ind_j
                                        V[marker] = loc_coeff * (self.elem_Lap_stiff_mat[loc_ind_i, loc_ind_j])
                                        # + loc_coeff * self.elem_Adv_mat[loc_ind_i, loc_ind_j]
                                        marker += 1
                                if coarse_ngh_elem_ind == coarse_elem_ind:
                                    rhs_corr[fd_ind_i] += self.get_Neum_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
            Op_mat_coo = coo_matrix((V[:marker], (I[:marker], J[:marker])), shape=(fd_num, fd_num))
            Op_mat = Op_mat_coo.tocsc()
            # logging.info("Construct the linear system [{0:d}]/[{1:d}], [{2:d}x{2:d}]".format(coarse_elem_ind, self.coarse_elem, fd_num))
            ilu = spilu(Op_mat)
            Mx = lambda x: ilu.solve(x)
            pre_M = LinearOperator((fd_num, fd_num), Mx)
            corr, info = lgmres(Op_mat, rhs_corr, tol=self.TOL, M=pre_M)
            assert info == 0
            glb_corr += self.get_glb_vec(coarse_elem_ind, corr)
            if coarse_elem_ind > prc_flag / 10 * self.coarse_elem:
                logging.info("......{0:.2f}%".format(coarse_elem_ind / self.coarse_elem * 100.))
                prc_flag += 1
            # logging.info("Finish [{0:d}]/[{1:d}]".format(coarse_elem_ind, self.coarse_elem))
        return glb_corr

    def get_true_corr(self, guess=[]):
        def get_fd_ind(fine_elem_ind_x, fine_elem_ind_y, loc_ind):
            if fine_elem_ind_y == self.fine_grid - 1 and loc_ind in [2, 3]:
                return -1
            else:
                loc_ind_y, loc_ind_x = divmod(loc_ind, 2)
                return (fine_elem_ind_y + loc_ind_y) * (self.fine_grid + 1) + fine_elem_ind_x + loc_ind_x

        fd_num = self.fine_grid * (self.fine_grid + 1)
        max_data_len = self.coarse_elem * ((self.sub_grid + 1)**4 + self.sub_elem * ST.N_V**2)
        I = -np.ones((max_data_len, ), dtype=np.int32)
        J = -np.ones((max_data_len, ), dtype=np.int32)
        V = np.zeros((max_data_len, ))
        marker = 0
        rhs_corr = np.zeros((fd_num, ))
        x0 = np.zeros((fd_num, ))
        if len(guess) > 0:
            x0 = guess[:fd_num]
        for coarse_elem_ind in range(self.coarse_elem):
            coarse_elem_ind_y, coarse_elem_ind_x = divmod(coarse_elem_ind, self.coarse_grid)
            S_mat = self.S_mat_list[coarse_elem_ind]
            eigen_vec = self.eigen_vec[:, coarse_elem_ind * self.eigen_num:(coarse_elem_ind + 1) * self.eigen_num]
            P_mat = S_mat.dot(eigen_vec)
            Q_mat = P_mat.dot(P_mat.T)
            node_sub_ind_list = [-1] * (self.sub_grid + 1)**2
            fd_ind_list = [-1] * (self.sub_grid + 1)**2
            marker_ = 0
            for node_sub_ind_y in range(self.sub_grid + 1):
                for node_sub_ind_x in range(self.sub_grid + 1):
                    node_sub_ind = node_sub_ind_y * (self.sub_grid + 1) + node_sub_ind_x
                    node_ind_y = coarse_elem_ind_y * self.sub_grid + node_sub_ind_y
                    node_ind_x = coarse_elem_ind_x * self.sub_grid + node_sub_ind_x
                    if node_ind_y < self.fine_grid:
                        fd_ind = node_ind_y * (self.fine_grid + 1) + node_ind_x
                    else:
                        fd_ind = -1
                    if fd_ind >= 0:
                        node_sub_ind_list[marker_] = node_sub_ind
                        fd_ind_list[marker_] = fd_ind
                        marker_ += 1
            for ind_i in range(marker_):
                node_sub_ind_i = node_sub_ind_list[ind_i]
                fd_ind_i = fd_ind_list[ind_i]
                for ind_j in range(marker_):
                    node_sub_ind_j = node_sub_ind_list[ind_j]
                    fd_ind_j = fd_ind_list[ind_j]
                    I[marker] = fd_ind_i
                    J[marker] = fd_ind_j
                    V[marker] = Q_mat[node_sub_ind_i, node_sub_ind_j]
                    marker += 1
            for sub_elem_ind in range(self.sub_elem):
                sub_elem_ind_y, sub_elem_ind_x = divmod(sub_elem_ind, self.sub_grid)
                fine_elem_ind_y = coarse_elem_ind_y * self.sub_grid + sub_elem_ind_y
                fine_elem_ind_x = coarse_elem_ind_x * self.sub_grid + sub_elem_ind_x
                loc_coeff = self.coeff[fine_elem_ind_x, fine_elem_ind_y]
                for loc_ind_i in range(ST.N_V):
                    fd_ind_i = get_fd_ind(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
                    if fd_ind_i >= 0:
                        for loc_ind_j in range(ST.N_V):
                            fd_ind_j = get_fd_ind(fine_elem_ind_x, fine_elem_ind_y, loc_ind_j)
                            if fd_ind_j >= 0:
                                I[marker] = fd_ind_i
                                J[marker] = fd_ind_j
                                V[marker] = loc_coeff * (self.elem_Lap_stiff_mat[loc_ind_i, loc_ind_j])
                                # + loc_coeff * self.elem_Adv_mat[loc_ind_i, loc_ind_j]
                                marker += 1
                        rhs_corr[fd_ind_i] += self.get_Neum_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
        Op_mat_coo = coo_matrix((V[:marker], (I[:marker], J[:marker])), shape=(fd_num, fd_num))
        Op_mat = Op_mat_coo.tocsc()
        ilu = spilu(Op_mat)
        Mx = lambda x: ilu.solve(x)
        pre_M = LinearOperator((fd_num, fd_num), Mx)
        corr, info = lgmres(Op_mat, rhs_corr, tol=self.TOL, M=pre_M, x0=x0)
        assert info == 0
        glb_corr = np.zeros((self.tot_node, ))
        glb_corr[:fd_num] = corr
        return glb_corr