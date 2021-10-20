from scipy.sparse import linalg
import Settings.Setting as ST
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import spsolve

import logging


class ProblemSetting(ST.Setting):
    def init_args(self, eigen_num: int = 4, os_ly: int = 1):
        self.eigen_num = eigen_num
        self.oversamp_layer = os_ly
        self.tot_fd_num = self.eigen_num * self.coarse_elem

    def set_coeff(self, coeff: np.ndarray):
        self.coeff = coeff
        self.kappa = 24.0 * self.coarse_grid**2 * self.coeff
        # Magic formula from the paper

    def set_Diri_func(self, Diri_func_g, Diri_func_gx, Diri_func_gy):
        self.Diri_func_g = Diri_func_g
        self.Diri_func_gx = Diri_func_gx
        self.Diri_func_gy = Diri_func_gy

    def set_source_func(self, source_func):
        self.source_func = source_func

    def get_Diri_quad_Lag(self, fine_elem_ind_x, fine_elem_ind_y, loc_ind):
        # Compute \int_{K_h} A\nabla g\cdot \nable L_i dx
        val = 0.0
        h = self.h
        center_x, center_y = 0.5 * h * (2 * fine_elem_ind_x + 1), 0.5 * h * (2 * fine_elem_ind_y + 1)
        coeff = self.coeff[fine_elem_ind_x, fine_elem_ind_y]
        for quad_ind_x in range(ST.QUAD_ORDER):
            for quad_ind_y in range(ST.QUAD_ORDER):
                quad_cord_x, quad_cord_y = ST.QUAD_CORD[quad_ind_x], ST.QUAD_CORD[quad_ind_y]
                quad_wght_x, quad_wght_y = ST.QUAD_WGHT[quad_ind_x], ST.QUAD_WGHT[quad_ind_y]
                quad_real_cord_x = center_x + 0.5 * h * quad_cord_x
                quad_real_cord_y = center_y + 0.5 * h * quad_cord_y
                test_grad_x, test_grad_y = ST.get_locbase_grad_val(loc_ind, quad_cord_x, quad_cord_y)
                Diri_grad_x = self.Diri_func_gx(quad_real_cord_x, quad_real_cord_y)
                Diri_grad_y = self.Diri_func_gy(quad_real_cord_x, quad_real_cord_y)
                val += 0.5 * h * quad_wght_x * quad_wght_y * coeff * (Diri_grad_x * test_grad_x + Diri_grad_y * test_grad_y)
                # Be careful with scaling
        return val

    def get_coarse_ngh_elem_lim(self, coarse_elem_ind):
        coarse_elem_ind_y, coarse_elem_ind_x = divmod(coarse_elem_ind, self.coarse_grid)
        coarse_ngh_elem_lf_lim = max(0, coarse_elem_ind_x - self.oversamp_layer)
        coarse_ngh_elem_rg_lim = min(self.coarse_grid, coarse_elem_ind_x + self.oversamp_layer + 1)
        coarse_ngh_elem_dw_lim = max(0, coarse_elem_ind_y - self.oversamp_layer)
        coarse_ngh_elem_up_lim = min(self.coarse_grid, coarse_elem_ind_y + self.oversamp_layer + 1)
        return coarse_ngh_elem_lf_lim, coarse_ngh_elem_rg_lim, coarse_ngh_elem_dw_lim, coarse_ngh_elem_up_lim

    def get_source_quad_Lag(self, fine_elem_ind_x, fine_elem_ind_y, loc_ind):
        # Compute \int_{K_h} f L_i dx
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

    def get_L2_energy_norm(self, u):
        assert self.glb_A_mat != None
        val0 = self.h * np.linalg.norm(u)
        val1 = np.sqrt(u @ self.glb_A_mat.dot(u))
        return val0, val1

    def get_eigen_pair(self):
        assert self.eigen_num > 0
        fd_num = (self.sub_grid + 1)**2
        loc_data_len = ST.N_V**2
        eigen_vec = np.zeros((fd_num, self.coarse_elem * self.eigen_num))
        eigen_val = np.zeros((self.coarse_elem * self.eigen_num, ))
        S_mat_list = [None] * self.coarse_elem  # A list of S matrices, saved here for futural usages
        # A_mat_list = [None] * self.coarse_elem
        for coarse_elem_ind in range(self.coarse_elem):
            coarse_elem_ind_y, coarse_elem_ind_x = divmod(coarse_elem_ind, self.coarse_grid)
            max_data_len = self.sub_elem * loc_data_len
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
            node_ind_x_lf_lim = lf_lim * self.sub_grid + 1
            node_ind_x_rg_lim = rg_lim * self.sub_grid
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

    def get_glb_vec(self, coarse_elem_ind, vec):
        ind_map_rev = self.ind_map_rev_list[coarse_elem_ind]
        glb_vec = np.zeros((self.tot_node, ))
        for loc_fd_ind, node_ind in ind_map_rev.items():
            glb_vec[node_ind] = vec[loc_fd_ind]
        return glb_vec

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
                                    rhs_corr[fd_ind_i] += self.get_Diri_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
            Op_mat_coo = coo_matrix((V[:marker], (I[:marker], J[:marker])), shape=(fd_num, fd_num))
            Op_mat = Op_mat_coo.tocsc()
            # logging.info("Construct the linear system [{0:d}]/[{1:d}], [{2:d}x{2:d}]".format(coarse_elem_ind, self.coarse_elem, fd_num))
            # ilu = spilu(Op_mat)
            # Mx = lambda x: ilu.solve(x)
            # pre_M = LinearOperator((fd_num, fd_num), Mx)
            corr, info = lgmres(Op_mat, rhs_corr, tol=self.TOL)
            assert info == 0
            glb_corr += self.get_glb_vec(coarse_elem_ind, corr)
            basis_wrt_coarse_elem = np.zeros(rhs_basis.shape)
            for eigen_ind in range(self.eigen_num):
                basis, info = lgmres(Op_mat, rhs_basis[:, eigen_ind], x0=guess[:, eigen_ind], tol=self.TOL)
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
        marker = 0
        tot_node = self.tot_node
        glb_F_vec = np.zeros((tot_node, ))
        for fine_elem_ind in range(self.fine_elem):
            fine_elem_ind_y, fine_elem_ind_x = divmod(fine_elem_ind, self.fine_grid)
            coeff = self.coeff[fine_elem_ind_x, fine_elem_ind_y]
            elem_stiff_mat = coeff * self.elem_Lap_stiff_mat
            for loc_ind_i in range(ST.N_V):
                loc_ind_iy, loc_ind_ix = divmod(loc_ind_i, 2)
                node_ind_i = (fine_elem_ind_y + loc_ind_iy) * (self.fine_grid + 1) + fine_elem_ind_x + loc_ind_ix
                for loc_ind_j in range(ST.N_V):
                    loc_ind_jy, loc_ind_jx = divmod(loc_ind_j, 2)
                    node_ind_j = (fine_elem_ind_y + loc_ind_jy) * (self.fine_grid + 1) + fine_elem_ind_x + loc_ind_jx
                    I[marker] = node_ind_i
                    J[marker] = node_ind_j
                    V[marker] = elem_stiff_mat[loc_ind_i, loc_ind_j]
                    marker += 1
                glb_F_vec[node_ind_i] += self.get_source_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
                glb_F_vec[node_ind_i] -= self.get_Diri_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
        glb_A_mat_coo = coo_matrix((V[:marker], (I[:marker], J[:marker])), shape=(tot_node, tot_node))
        glb_A_mat = glb_A_mat_coo.tocsc()
        self.glb_A_mat = glb_A_mat
        self.glb_F_vec = glb_F_vec

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
        logging.info("Finish getting the Dirichlet corrector and multiscale bases.")
        self.get_glb_A_F()
        logging.info("Finish getting the global stiffness matrix and right-hand vector.")
        self.get_glb_basis_spmat()
        logging.info("Finish collecting all the bases in a sparse matrix formation.")
        A_mat = self.glb_basis_spmat_T * self.glb_A_mat * self.glb_basis_spmat
        rhs = self.glb_basis_spmat_T.dot(self.glb_F_vec + self.glb_A_mat.dot(self.glb_corr))
        logging.info("Finish constructing the final linear system.")
        ilu = spilu(A_mat)
        Mx = lambda x: ilu.solve(x)
        pre_M = LinearOperator((self.tot_fd_num, self.tot_fd_num), Mx)
        if len(guess) == self.tot_fd_num:
            x0 = guess
        else:
            x0 = np.zeros((self.tot_fd_num, ))
        # omega, info = lgmres(A_mat, rhs, tol=self.TOL, M=pre_M, x0=x0, maxiter=10000)
        omega = spsolve(A_mat, rhs)
        # if info != 0:
        #     logging.critical("Fail to solve the final linear system, info={0:d}".format(info))
        #     raise AssertionError
        u = self.glb_basis_spmat.dot(omega)
        u -= self.glb_corr
        logging.info("Finish solving the final linear system.")
        return u, omega
        # assert self.oversamp_layer > 0 and self.eigen_num > 0
        # self.get_eigen_pair()
        # assert len(self.eigen_val) > 0
        # logging.info("Finish getting all eigenvalue-vector pairs.")
        # self.get_ind_map()
        # assert len(self.ind_map_list) > 0
        # logging.info("Finish getting maps of [global node index] to [local freedom index].")
        # self.get_corr_basis()
        # assert len(self.basis_list) > 0
        # logging.info("Finish getting the Dirichlet corrector and multiscale bases.")
        # self.get_glb_A_F()
        # assert len(self.glb_F_vec) > 0
        # logging.info("Finish getting the global stiffness matrix and right-hand vector.")
        # self.get_glb_basis_spmat()
        # assert self.glb_basis_spmat != None and self.glb_basis_spmat_T != None
        # logging.info("Finish collecting all the bases in a sparse matrix formation.")
        # os_ly = self.oversamp_layer
        # max_data_len = self.coarse_elem * self.eigen_num**2 * (4*os_ly+1)**2
        # # max_data_len = self.coarse_elem**2 * self.eigen_num**2
        # I, J = -np.ones((max_data_len, ), dtype=np.int32), -np.ones((max_data_len, ), dtype=np.int32)
        # V = np.zeros((max_data_len))
        # marker = 0
        # rhs = np.zeros((self.tot_fd_num, ))
        # for coarse_elem_ind_i in range(self.coarse_elem):
        #     coarse_elem_ind_iy, coarse_elem_ind_ix = divmod(coarse_elem_ind_i, self.coarse_grid)
        #     for eigen_ind_i in range(self.eigen_num):
        #         fd_ind_i = coarse_elem_ind_i*self.eigen_num + eigen_ind_i
        #         loc_basis_i = self.basis_list[coarse_elem_ind_i][:, eigen_ind_i]
        #         glb_basis_i = self.get_glb_vec(coarse_elem_ind_i, loc_basis_i)
        #         lf_lim = max(0, coarse_elem_ind_ix-2*os_ly)
        #         rg_lim = min(self.coarse_grid, coarse_elem_ind_ix+2*os_ly+1)
        #         dw_lim = max(0, coarse_elem_ind_iy-2*os_ly)
        #         up_lim = min(self.coarse_grid, coarse_elem_ind_iy+2*os_ly+1)
        #         # lf_lim = 0
        #         # rg_lim = self.coarse_grid
        #         # dw_lim = 0
        #         # up_lim = self.coarse_grid
        #         for coarse_elem_ind_jx in range(lf_lim, rg_lim):
        #             for coarse_elem_ind_jy in range(dw_lim, up_lim):
        #                 coarse_elem_ind_j = coarse_elem_ind_jy*self.coarse_grid + coarse_elem_ind_jx
        #                 for eigen_ind_j in range(self.eigen_num):
        #                     fd_ind_j = coarse_elem_ind_j*self.eigen_num + eigen_ind_j
        #                     loc_basis_j = self.basis_list[coarse_elem_ind_j][:, eigen_ind_j]
        #                     glb_basis_j = self.get_glb_vec(coarse_elem_ind_j, loc_basis_j)
        #                     I[marker] = fd_ind_i
        #                     J[marker] = fd_ind_j
        #                     V[marker] = np.inner(self.glb_A_mat.dot(glb_basis_i), glb_basis_j)
        #                     # print("coarse_elem_ind_i:{0:d}, eigen_ind_i:{1:d}, coarse_elem_ind_j:{2:d}, eigen_ind_j:{3:d}, value:{4:.5f}".format(coarse_elem_ind_i, eigen_ind_i, coarse_elem_ind_j, eigen_ind_j, V[marker]))
        #                     marker += 1
        #         rhs[fd_ind_i] = np.inner(self.glb_F_vec, glb_basis_i) + np.inner(self.glb_A_mat.dot(glb_basis_i), self.glb_corr)
        # A_mat_coo = coo_matrix((V[:marker], (I[:marker], J[:marker])), shape=(self.tot_fd_num, self.tot_fd_num))
        # A_mat = A_mat_coo.tocsr()
        # A_mat = self.glb_basis_spmat_T * self.glb_A_mat * self.glb_basis_spmat
        # rhs = self.glb_basis_spmat_T.dot(self.glb_F_vec + self.glb_A_mat.dot(self.glb_corr))
        # logging.info("Finish constructing the final linear system.")
        # ilu = spilu(A_mat)
        # Mx = lambda x: ilu.solve(x)
        # pre_M = LinearOperator((self.tot_fd_num, self.tot_fd_num), Mx)
        # omega, info = lgmres(A_mat, rhs, tol=self.TOL, M=pre_M)
        # assert info == 0
        # u0 = self.glb_basis_spmat.dot(omega)
        # for coarse_elem_ind in range(self.coarse_elem):
        #     for eigen_ind in range(self.eigen_num):
        #         fd_ind = coarse_elem_ind*self.eigen_num + eigen_ind
        #         loc_basis = self.basis_list[coarse_elem_ind][:, eigen_ind]
        #         u0 += omega[fd_ind] * self.get_glb_vec(coarse_elem_ind, loc_basis)
        u0 -= self.glb_corr
        # logging.info("Finish solving the final linear system.")
        # return u0

    def solve_ref(self, guess=[]):
        def get_fd_ind(fine_elem_ind_x, fine_elem_ind_y, loc_ind):
            if fine_elem_ind_x == 0 and loc_ind in [0, 2]:
                return -1
            elif fine_elem_ind_x == (self.fine_grid - 1) and loc_ind in [1, 3]:
                return -1
            elif fine_elem_ind_y == 0 and loc_ind in [0, 1]:
                return -1
            elif fine_elem_ind_y == (self.fine_grid - 1) and loc_ind in [2, 3]:
                return -1
            else:
                loc_ind_y, loc_ind_x = divmod(loc_ind, 2)
                return (fine_elem_ind_y + loc_ind_y - 1) * (self.fine_grid - 1) + fine_elem_ind_x + loc_ind_x - 1

        I, J = -np.ones((self.fine_elem * ST.N_V**2), dtype=np.int32), -np.ones((self.fine_elem * ST.N_V**2), dtype=np.int32)
        V = np.zeros((self.fine_elem * ST.N_V**2))
        marker = 0
        fd_num = (self.fine_grid - 1)**2
        rhs = np.zeros((fd_num, ))
        x0 = np.zeros((fd_num, ))

        if len(guess) > 0:
            for node_ind_y in range(self.fine_grid + 1):
                for node_ind_x in range(self.fine_grid + 1):
                    if 0 < node_ind_y < self.fine_grid and 0 < node_ind_x < self.fine_grid:
                        node_ind = node_ind_y * (self.fine_grid + 1) + node_ind_x
                        fd_ind = (node_ind_y - 1) * (self.fine_grid - 1) + node_ind_x - 1
                        x0[fd_ind] = guess[node_ind]

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
                            marker += 1
                    rhs[fd_ind_i] += self.get_source_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
                    rhs[fd_ind_i] -= self.get_Diri_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
        A_mat_coo = coo_matrix((V[:marker], (I[:marker], J[:marker])), shape=(fd_num, fd_num))
        A_mat = A_mat_coo.tocsc()
        logging.info("Finish constructing the final linear system of the reference problem.")
        ilu = spilu(A_mat)
        Mx = lambda x: ilu.solve(x)
        pre_M = LinearOperator((fd_num, fd_num), Mx)
        u0_ref_inner, info = lgmres(A_mat, rhs, tol=self.TOL, x0=x0, M=pre_M)
        assert info == 0
        u0_ref = np.zeros((self.tot_node, ))
        for node_ind in range(self.tot_node):
            node_ind_y, node_ind_x = divmod(node_ind, self.fine_grid + 1)
            if (0 < node_ind_x < self.fine_grid) and (0 < node_ind_y < self.fine_grid):
                u0_ref[node_ind] = u0_ref_inner[(node_ind_y - 1) * (self.fine_grid - 1) + node_ind_x - 1]
        logging.info("Finish solving the final linear system of the reference problem.")
        return u0_ref

    def get_inhomo_ref(self, u0_ref):
        u = np.zeros((self.tot_node, ))
        h = self.h
        for node_ind in range(self.tot_node):
            node_ind_y, node_ind_x = divmod(node_ind, self.fine_grid + 1)
            y, x = node_ind_y * h, node_ind_x * h
            u[node_ind] = u0_ref[node_ind] + self.Diri_func_g(x, y)
        return u

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
            for coarse_ngh_elem_ind_y in range(dw_lim, up_lim):
                for coarse_ngh_elem_ind_x in range(lf_lim, rg_lim):
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
                                        marker += 1
                                if coarse_ngh_elem_ind == coarse_elem_ind:
                                    rhs_corr[fd_ind_i] += self.get_Diri_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
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
            if fine_elem_ind_x == 0 and loc_ind in [0, 2]:
                return -1
            elif fine_elem_ind_x == (self.fine_grid - 1) and loc_ind in [1, 3]:
                return -1
            elif fine_elem_ind_y == 0 and loc_ind in [0, 1]:
                return -1
            elif fine_elem_ind_y == (self.fine_grid - 1) and loc_ind in [2, 3]:
                return -1
            else:
                loc_ind_y, loc_ind_x = divmod(loc_ind, 2)
                return (fine_elem_ind_y + loc_ind_y - 1) * (self.fine_grid - 1) + fine_elem_ind_x + loc_ind_x - 1

        fd_num = (self.fine_grid - 1)**2
        max_data_len = self.coarse_elem * ((self.sub_grid + 1)**4 + self.sub_elem * ST.N_V**2)
        I = -np.ones((max_data_len, ), dtype=np.int32)
        J = -np.ones((max_data_len, ), dtype=np.int32)
        V = np.zeros((max_data_len, ))
        marker = 0
        rhs_corr = np.zeros((fd_num, ))
        x0 = np.zeros((fd_num, ))
        if len(guess) > 0:
            for node_ind_y in range(self.fine_grid + 1):
                for node_ind_x in range(self.fine_grid + 1):
                    if 0 < node_ind_y < self.fine_grid and 0 < node_ind_x < self.fine_grid:
                        node_ind = node_ind_y * (self.fine_grid + 1) + node_ind_x
                        fd_ind = (node_ind_y - 1) * (self.fine_grid - 1) + node_ind_x - 1
                        x0[fd_ind] = guess[node_ind]

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
                    if 0 < node_ind_x < self.fine_grid and 0 < node_ind_y < self.fine_grid:
                        fd_ind = (node_ind_y - 1) * (self.fine_grid - 1) + node_ind_x - 1
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
                                marker += 1
                        rhs_corr[fd_ind_i] += self.get_Diri_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
        Op_mat_coo = coo_matrix((V[:marker], (I[:marker], J[:marker])), shape=(fd_num, fd_num))
        Op_mat = Op_mat_coo.tocsc()
        ilu = spilu(Op_mat)
        Mx = lambda x: ilu.solve(x)
        pre_M = LinearOperator((fd_num, fd_num), Mx)
        corr, info = lgmres(Op_mat, rhs_corr, tol=self.TOL, M=pre_M, x0=x0)
        assert info == 0
        glb_corr = np.zeros((self.tot_node, ))
        for node_ind in range(self.tot_node):
            node_ind_y, node_ind_x = divmod(node_ind, self.fine_grid + 1)
            if (0 < node_ind_x < self.fine_grid) and (0 < node_ind_y < self.fine_grid):
                glb_corr[node_ind] = corr[(node_ind_y - 1) * (self.fine_grid - 1) + node_ind_x - 1]
        return glb_corr


########################
########################
# depreciated contents #
########################
########################

    def get_Diri_quad_node_val(self, fine_elem_ind_x, fine_elem_ind_y, loc_val):
        # Compute \int_{K_h} A\nabla g\cdot \nable v, where v = v_0L_0+v_1L_1+v_2L_2+v_3L_3
        val = 0.0
        for loc_ind in range(ST.N_V):
            val += loc_val[loc_ind] * self.get_Diri_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind)
        return val

    def get_source_quad_node_val(self, fine_elem_ind_x, fine_elem_ind_y, loc_val):
        val = 0.0
        for loc_ind in range(ST.N_V):
            val += loc_val[loc_ind] * self.get_source_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind)
        return val

    def get_stiff_quad_node_val(self, fine_elem_ind_x, fine_elem_ind_y, node_val_i, node_val_j):
        # Compute \int_{K_h} A\nabla u \cdot \nabla v, where the values of u, v on nodes of the fine element K_h is given
        coeff = self.coeff[fine_elem_ind_x, fine_elem_ind_y]
        elem_stiff_mat = coeff * self.elem_Lap_stiff_mat
        val = np.inner(elem_stiff_mat.dot(node_val_i), node_val_j)
        return val

    def get_coarse_ngh_elem_ind(self, off_ind: int, coarse_elem_ind_x, coarse_elem_ind_y):
        coarse_ngh_elem_ind_off_y, coarse_ngh_elem_ind_off_x = divmod(off_ind, 2 * self.oversamp_layer + 1)
        coarse_ngh_elem_ind_y = coarse_elem_ind_y + coarse_ngh_elem_ind_off_y - self.oversamp_layer
        coarse_ngh_elem_ind_x = coarse_elem_ind_x + coarse_ngh_elem_ind_off_x - self.oversamp_layer
        is_elem_indomain = (0 <= coarse_ngh_elem_ind_x < self.coarse_grid) and (0 <= coarse_ngh_elem_ind_y < self.coarse_grid)
        if is_elem_indomain:
            return coarse_ngh_elem_ind_y * self.coarse_grid + coarse_ngh_elem_ind_x
        else:
            return -1

    def get_glb_node_val(self, coarse_elem_ind, vec):
        # Get the zero-extension of v \in V_i^m
        ind_map = self.ind_map_list[coarse_elem_ind]
        glb_node_val = np.zeros((self.fine_grid + 1, self.fine_grid + 1))
        for node_ind_x in range(self.fine_grid + 1):
            for node_ind_y in range(self.fine_grid + 1):
                node_ind = node_ind_y * (self.fine_grid + 1) + node_ind_x
                if node_ind in ind_map:
                    rlt_node_ind = ind_map[node_ind]
                    if rlt_node_ind >= 0:
                        glb_node_val[node_ind_x, node_ind_y] = vec[rlt_node_ind]
        return glb_node_val

    def get_glb_basis(self, coarse_elem_ind, eigen_ind, basis_list):
        b = basis_list[coarse_elem_ind][:, eigen_ind]
        glb_b = self.get_glb_vec(coarse_elem_ind, b)
        return glb_b.reshape((self.fine_grid + 1, -1))

    def get_corr_basis_dr(self):
        assert self.oversamp_layer > 0 and self.eigen_num > 0
        assert len(self.ind_map_list) > 0
        glb_corr = np.zeros((self.tot_node, ))
        basis_list = [None] * self.coarse_elem
        for coarse_elem_ind in range(self.coarse_elem):
            coarse_elem_ind_y, coarse_elem_ind_x = divmod(coarse_elem_ind, self.coarse_grid)
            fd_num = self.loc_fd_num[coarse_elem_ind]
            ind_map_dic = self.ind_map_list[coarse_elem_ind]
            Op_mat = np.zeros((fd_num, fd_num))
            rhs_corr = np.zeros((fd_num, ))
            rhs_basis = np.zeros((fd_num, self.eigen_num))
            for coarse_ngh_elem_ind_off in range((2 * self.oversamp_layer + 1)**2):
                coarse_ngh_elem_ind_off_y, coarse_ngh_elem_ind_off_x = divmod(coarse_ngh_elem_ind_off, 2 * self.oversamp_layer + 1)
                coarse_ngh_elem_ind_y = coarse_elem_ind_y + coarse_ngh_elem_ind_off_y - self.oversamp_layer
                coarse_ngh_elem_ind_x = coarse_elem_ind_x + coarse_ngh_elem_ind_off_x - self.oversamp_layer
                coarse_ngh_elem_ind = coarse_ngh_elem_ind_y * self.coarse_grid + coarse_ngh_elem_ind_x
                is_elem_indomain = (0 <= coarse_ngh_elem_ind_y < self.coarse_grid) and (0 <= coarse_ngh_elem_ind_x < self.coarse_grid)
                if is_elem_indomain:
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
                            Op_mat[fd_ind_i, fd_ind_j] += Q_mat[node_sub_ind_i, node_sub_ind_j]
                        if coarse_ngh_elem_ind == coarse_elem_ind:
                            for eigen_ind in range(self.eigen_num):
                                rhs_basis[fd_ind_i, eigen_ind] += P_mat[node_sub_ind_i, eigen_ind]
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
                                        Op_mat[fd_ind_i, fd_ind_j] += loc_coeff * (self.elem_Lap_stiff_mat[loc_ind_i, loc_ind_j])
                                if coarse_ngh_elem_ind == coarse_elem_ind:
                                    rhs_corr[fd_ind_i] += self.get_Diri_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
            logging.info("Construct the linear system [{0:d}]/[{1:d}], [{2:d}x{2:d}]".format(coarse_elem_ind, self.coarse_elem, fd_num))
            corr = np.linalg.solve(Op_mat, rhs_corr)
            glb_corr += self.get_glb_vec(coarse_elem_ind, corr)
            basis_wrt_coarse_elem = np.zeros(rhs_basis.shape)
            for eigen_ind in range(self.eigen_num):
                basis = np.linalg.solve(Op_mat, rhs_basis[:, eigen_ind])
                basis_wrt_coarse_elem[:, eigen_ind] = basis
            basis_list[coarse_elem_ind] = basis_wrt_coarse_elem
            logging.info("Finish [{0:d}]/[{1:d}]".format(coarse_elem_ind, self.coarse_elem))
        self.glb_corr = glb_corr
        self.basis_list = basis_list

    def get_corr_basis_depreciated(self):
        # An important information
        # when constructing the stiffness matrix A_{i,j} = a(L_i, L_j)+s(\pi L_i, \pi L_j)
        # s(\pi L_i, \pi L_j) is not zero even nodes i, j are not in a same fine element!
        assert self.oversamp_layer > 0 and self.eigen_num > 0
        assert len(self.ind_map_list) > 0
        corr_list = [None] * self.coarse_elem
        basis_list = [None] * self.coarse_elem
        glb_basis_list = [None] * self.tot_fd_num
        # The maximal number of fine elements that a oversampled region contains
        # max_data_len = (2*self.oversamp_layer+1)**2 * (self.sub_elem*ST.N_V**2 + (self.sub_grid+1)**4)
        max_data_len = (2 * self.oversamp_layer + 1)**2 * (self.sub_grid + 1)**4
        for coarse_elem_ind in range(self.coarse_elem):
            coarse_elem_ind_y, coarse_elem_ind_x = divmod(coarse_elem_ind, self.coarse_grid)
            fd_num = self.loc_fd_num[coarse_elem_ind]
            ind_map_dic = self.ind_map_list[coarse_elem_ind]
            I = -np.ones((max_data_len, ), dtype=np.int32)
            J = -np.ones((max_data_len, ), dtype=np.int32)
            V = np.zeros((max_data_len, ))
            marker = 0
            # Vectors I, J and V are used to construct a coo_matrix, integer marker determines the valid range of vectors,
            # such as I[:marker], J[:marker]. Because we do not know how many indeces and values need to be inserted.
            Op_mat_test = np.zeros((fd_num, fd_num))
            rhs_corr = np.zeros((fd_num, ))
            rhs_basis = np.zeros((fd_num, self.eigen_num))
            # Right-hand vectors of correctors and multiscale bases
            for coarse_ngh_elem_ind_off in range((2 * self.oversamp_layer + 1)**2):
                # A loop over elements in the oversampling layer
                coarse_ngh_elem_ind_off_y, coarse_ngh_elem_ind_off_x = divmod(coarse_ngh_elem_ind_off, 2 * self.oversamp_layer + 1)
                coarse_ngh_elem_ind_y = coarse_elem_ind_y + coarse_ngh_elem_ind_off_y - self.oversamp_layer
                coarse_ngh_elem_ind_x = coarse_elem_ind_x + coarse_ngh_elem_ind_off_x - self.oversamp_layer
                coarse_ngh_elem_ind = coarse_ngh_elem_ind_y * self.coarse_grid + coarse_ngh_elem_ind_x
                # Neighboring element in the global coordinate
                is_elem_indomain = (0 <= coarse_ngh_elem_ind_y < self.coarse_grid) and (0 <= coarse_ngh_elem_ind_x < self.coarse_grid)
                # Select the real neighboring elements in the domain
                if is_elem_indomain:
                    # Retreive the S matrix of the current neighboring coarse element
                    S_mat = self.S_mat_list[coarse_ngh_elem_ind]
                    A_mat = self.A_mat_list[coarse_ngh_elem_ind]
                    # Let v=\sum_i v_i L_i where L_i is the Lagrange basis corresponding to i-th node.
                    # Then s(\pi v, \phi^eigen_ind) = \sum_i v_i P[i, eigen_ind]
                    eigen_vec = self.eigen_vec[:, coarse_ngh_elem_ind * self.eigen_num:(coarse_ngh_elem_ind + 1) * self.eigen_num]
                    P_mat = S_mat @ eigen_vec
                    # Get nodes in the current coarse element
                    node_in_coarse_elem_dic = {}
                    for sub_elem_ind in range(self.sub_elem):
                        sub_elem_ind_y, sub_elem_ind_x = divmod(sub_elem_ind, self.sub_grid)
                        # Coordinates of fine elements are always defined globally
                        fine_elem_ind_y = coarse_ngh_elem_ind_y * self.sub_grid + sub_elem_ind_y
                        fine_elem_ind_x = coarse_ngh_elem_ind_x * self.sub_grid + sub_elem_ind_x
                        for loc_ind_i in range(ST.N_V):
                            loc_ind_iy, loc_ind_ix = divmod(loc_ind_i, 2)
                            # Coordinates of nodes are always defined globally
                            node_ind_i = (fine_elem_ind_y + loc_ind_iy) * (self.fine_grid + 1) + fine_elem_ind_x + loc_ind_ix
                            # The node index w.r.t. the current coarse element [0, (self.sub_grid+1)*(self.sub_grid+1))
                            node_sub_ind_i = (sub_elem_ind_y + loc_ind_iy) * (self.sub_grid + 1) + sub_elem_ind_x + loc_ind_ix
                            assert node_ind_i in ind_map_dic
                            fd_ind_i = ind_map_dic[node_ind_i]
                            # The freedom index of the node in the oversampled region
                            if fd_ind_i >= 0:
                                node_in_coarse_elem_dic[node_sub_ind_i] = fd_ind_i
                                # Add the illegal node into the dic
                                # for loc_ind_j in range(ST.N_V):
                                #     loc_ind_jy, loc_ind_jx = divmod(loc_ind_j, 2)
                                #     node_ind_j = (fine_elem_ind_y+loc_ind_jy)*(self.fine_grid+1) + fine_elem_ind_x + loc_ind_jx
                                #     node_sub_ind_j = (sub_elem_ind_y+loc_ind_jy)*(self.sub_grid+1) + sub_elem_ind_x + loc_ind_jx
                                #     assert node_ind_j in ind_map_dic
                                #     fd_ind_j = ind_map_dic[node_ind_j]
                                #     if fd_ind_j >= 0: # If true, those two nodes are freedom nodes
                                #         loc_coeff = self.coeff[fine_elem_ind_x, fine_elem_ind_y]
                                #         I[marker] = fd_ind_i
                                #         J[marker] = fd_ind_j
                                #         # temp = ST.get_loc_stiff(loc_coeff, loc_ind_i, loc_ind_j)
                                #         # The frist term a(L_i, L_j)
                                #         V[marker] = loc_coeff * (self.elem_Lap_stiff_mat[loc_ind_i, loc_ind_j])
                                #         marker += 1
                                if coarse_ngh_elem_ind == coarse_elem_ind:
                                    # The Dirichlet corrector, can be constructed on fine elements
                                    rhs_corr[fd_ind_i] += self.get_Diri_quad_Lag(fine_elem_ind_x, fine_elem_ind_y, loc_ind_i)
                    # if (len(node_in_coarse_elem_dic) > 16):
                    #     print("Watch out:{0:d}".format(len(node_in_coarse_elem_dic)))
                    # The second term s(\pi L_i, \pi L_j)
                    # must loop over all illegal nodes in the current element!
                    for node_sub_ind_i in node_in_coarse_elem_dic:
                        fd_ind_i = node_in_coarse_elem_dic[node_sub_ind_i]
                        for node_sub_ind_j in node_in_coarse_elem_dic:
                            fd_ind_j = node_in_coarse_elem_dic[node_sub_ind_j]
                            I[marker] = fd_ind_i
                            J[marker] = fd_ind_j
                            V[marker] += A_mat[node_sub_ind_i, node_sub_ind_j]
                            V[marker] += np.inner(P_mat[node_sub_ind_i, :], P_mat[node_sub_ind_j, :])
                            Op_mat_test[fd_ind_i, fd_ind_j] += V[marker]
                            marker += 1
                        if coarse_ngh_elem_ind == coarse_elem_ind:
                            # Construct the right-hand vectors for solving bases and Dirichlet boundary correctors
                            # Note that only the integral on K_i is summed from the paper
                            for eigen_ind in range(self.eigen_num):
                                rhs_basis[fd_ind_i, eigen_ind] += P_mat[node_sub_ind_i, eigen_ind]
                                # Use the definition of P_mat
            Op_mat_coo = coo_matrix((V[:marker], (I[:marker], J[:marker])), shape=(fd_num, fd_num))
            Op_mat = Op_mat_coo.tocsc()
            corr, info = lgmres(Op_mat, rhs_corr, tol=self.TOL)
            assert info == 0
            corr_list[coarse_elem_ind] = corr
            basis_wrt_coarse_elem = np.zeros(rhs_basis.shape)
            for eigen_ind in range(self.eigen_num):
                basis, info = lgmres(Op_mat, rhs_basis[:, eigen_ind], tol=self.TOL)
                assert info == 0
                basis_wrt_coarse_elem[:, eigen_ind] = basis
                fd_ind = coarse_elem_ind * self.eigen_num + eigen_ind
                glb_basis_list[fd_ind] = self.get_glb_node_val(coarse_elem_ind, basis)
            basis_list[coarse_elem_ind] = basis_wrt_coarse_elem
        # Get the final Dirichlet corrector \sum_i^N D_i^m
        glb_corr = np.zeros((self.fine_grid + 1, self.fine_grid + 1))
        for coarse_elem_ind in range(self.coarse_elem):
            glb_corr += self.get_glb_node_val(coarse_elem_ind, corr_list[coarse_elem_ind])
        self.glb_corr = glb_corr
        self.basis_list = basis_list
        self.glb_basis_list = glb_basis_list
        # Basis index = eigen_num * coarse_elem_ind + eigen_ind

    def get_node_val_basis(self, fine_elem_ind_x, fine_elem_ind_y, fd_ind):
        # Get node values on a fine element,
        # return [v_0, v_1, v_2, v_3]
        # fd_ind = coarse_elem_ind * eigen_num + eigen_ind
        coarse_elem_ind, eigen_ind = divmod(fd_ind, self.eigen_num)
        ind_map = self.ind_map_list[coarse_elem_ind]
        basis = self.basis_list[coarse_elem_ind][:, eigen_ind]
        loc_val = np.zeros((ST.N_V, ))
        for loc_ind in range(ST.N_V):
            loc_ind_y, loc_ind_x = divmod(loc_ind, 2)
            node_ind = (fine_elem_ind_y + loc_ind_y) * (self.fine_grid + 1) + fine_elem_ind_x + loc_ind_x
            assert node_ind in ind_map
            rlt_fd_ind = ind_map[node_ind]
            if rlt_fd_ind >= 0:
                loc_val[loc_ind] = basis[rlt_fd_ind]
        return loc_val

    def get_node_val_glb_basis(self, fine_elem_ind_x, fine_elem_ind_y, fd_ind):
        loc_val = np.zeros((ST.N_V, ))
        loc_val[0] = self.glb_basis_list[fd_ind][fine_elem_ind_x, fine_elem_ind_y]
        loc_val[1] = self.glb_basis_list[fd_ind][fine_elem_ind_x + 1, fine_elem_ind_y]
        loc_val[2] = self.glb_basis_list[fd_ind][fine_elem_ind_x, fine_elem_ind_y + 1]
        loc_val[3] = self.glb_basis_list[fd_ind][fine_elem_ind_x + 1, fine_elem_ind_y + 1]
        return loc_val

    def get_node_val_corr(self, fine_elem_ind_x, fine_elem_ind_y):
        loc_val = np.zeros((ST.N_V, ))
        loc_val[0] = self.glb_corr[fine_elem_ind_x, fine_elem_ind_y]
        loc_val[1] = self.glb_corr[fine_elem_ind_x + 1, fine_elem_ind_y]
        loc_val[2] = self.glb_corr[fine_elem_ind_x, fine_elem_ind_y + 1]
        loc_val[3] = self.glb_corr[fine_elem_ind_x + 1, fine_elem_ind_y + 1]
        return loc_val

    def solve_depreciated(self):
        assert self.oversamp_layer > 0 and self.eigen_num > 0
        self.get_eigen_pair()
        assert len(self.eigen_val) > 0
        logging.info("Finish getting all eigenvalue-vector pairs.")
        self.get_ind_map()
        assert len(self.ind_map_list) > 0
        logging.info("Finish getting maps of [global node index] to [local freedom index].")
        self.get_corr_basis_depreciated()
        assert len(self.basis_list) > 0
        logging.info("Finish getting the Dirichlet corrector and multiscale bases.")
        max_data_len = self.coarse_elem * (2 * self.oversamp_layer + 1)**4 * self.eigen_num**2
        I, J = -np.ones((max_data_len, ), dtype=np.int32), -np.ones((max_data_len, ), dtype=np.int32)
        V = np.zeros((max_data_len))
        marker = 0
        # As previously introduced, vectors I, J and V are used to construct the coo_matrix of the final linear system
        rhs = np.zeros((self.tot_fd_num, ))
        # The right-hand vector of the final linear system
        for coarse_elem_ind in range(self.coarse_elem):
            coarse_elem_ind_y, coarse_elem_ind_x = divmod(coarse_elem_ind, self.coarse_grid)
            # Get the basis indeces that not valish on the current coarse element
            coarse_ngh_elem_list = []
            for coarse_elem_ind_off in range((2 * self.oversamp_layer + 1)**2):
                coarse_ngh_elem_ind = self.get_coarse_ngh_elem_ind(coarse_elem_ind_off, coarse_elem_ind_x, coarse_elem_ind_y)
                if coarse_ngh_elem_ind >= 0:
                    coarse_ngh_elem_list.append(coarse_ngh_elem_ind)
            for coarse_ngh_elem_ind_i in coarse_ngh_elem_list:
                for eigen_ind_i in range(self.eigen_num):
                    fd_ind_i = coarse_ngh_elem_ind_i * self.eigen_num + eigen_ind_i
                    for coarse_ngh_elem_ind_j in coarse_ngh_elem_list:
                        for eigen_ind_j in range(self.eigen_num):
                            fd_ind_j = coarse_ngh_elem_ind_j * self.eigen_num + eigen_ind_j
                            I[marker] = fd_ind_i
                            J[marker] = fd_ind_j
                            # This is \int_{K_h} A\nabla \Phi_j^s \cdot \nabla \Phi_i^t and right-hand \int_{K_h} f \Phi_i^t \dx x et al.
                            for sub_elem_ind in range(self.sub_elem):
                                sub_elem_ind_y, sub_elem_ind_x = divmod(sub_elem_ind, self.sub_grid)
                                fine_elem_ind_y = coarse_elem_ind_y * self.sub_grid + sub_elem_ind_y
                                fine_elem_ind_x = coarse_elem_ind_x * self.sub_grid + sub_elem_ind_x
                                node_val_i = self.get_node_val_basis(fine_elem_ind_x, fine_elem_ind_y, fd_ind_i)  # Node values of \Phi_i^t on a fine element
                                node_val_j = self.get_node_val_basis(fine_elem_ind_x, fine_elem_ind_y, fd_ind_j)  # Node values of \Phi_i^t on a fine element
                                # node_val_i = self.get_node_val_glb_basis(fine_elem_ind_x, fine_elem_ind_y, fd_ind_i)
                                # node_val_j = self.get_node_val_glb_basis(fine_elem_ind_x, fine_elem_ind_y, fd_ind_j)
                                V[marker] += self.get_stiff_quad_node_val(fine_elem_ind_x, fine_elem_ind_y, node_val_i, node_val_j)
                            marker += 1
                    # Begin to construct the right-hand vector
                    for sub_elem_ind in range(self.sub_elem):
                        sub_elem_ind_y, sub_elem_ind_x = divmod(sub_elem_ind, self.sub_grid)
                        fine_elem_ind_y = coarse_elem_ind_y * self.sub_grid + sub_elem_ind_y
                        fine_elem_ind_x = coarse_elem_ind_x * self.sub_grid + sub_elem_ind_x
                        node_val_i = self.get_node_val_basis(fine_elem_ind_x, fine_elem_ind_y, fd_ind_i)  # Node values of \Phi_i^t on a fine element
                        corr_val = self.get_node_val_corr(fine_elem_ind_x, fine_elem_ind_y)
                        rhs[fd_ind_i] += self.get_source_quad_node_val(fine_elem_ind_x, fine_elem_ind_y, node_val_i)
                        rhs[fd_ind_i] -= self.get_Diri_quad_node_val(fine_elem_ind_x, fine_elem_ind_y, node_val_i)
                        # Compute \int_{K_h} A \nabla D^m\cdot \nabla \Phi_i^t \dx
                        rhs[fd_ind_i] += self.get_stiff_quad_node_val(fine_elem_ind_x, fine_elem_ind_y, corr_val, node_val_i)
        A_mat_coo = coo_matrix((V[:marker], (I[:marker], J[:marker])), shape=(self.tot_fd_num, self.tot_fd_num))
        A_mat = A_mat_coo.tocsc()
        logging.info("Finish constructing the final linear system.")
        omega, info = lgmres(A_mat, rhs, tol=self.TOL)
        assert info == 0
        # u_0 = \sum x_i^s \Phi_i^s - D^m g
        # u_0 = omega - [global Dirichlet corrector]
        u0 = np.zeros((self.fine_grid + 1, self.fine_grid + 1))
        for coarse_elem_ind in range(self.coarse_elem):
            for eigen_ind in range(self.eigen_num):
                fd_ind = coarse_elem_ind * self.eigen_num + eigen_ind
                basis = self.basis_list[coarse_elem_ind][:, eigen_ind]
                u0 += omega[fd_ind] * self.get_glb_node_val(coarse_elem_ind, basis)
        u0 -= self.glb_corr
        logging.info("Finish solving the final linear system.")
        self.rhs_for_dbg = rhs
        self.A_mat_for_dbg = A_mat.todense()
        return u0

    def solve_dbg(self):
        A_mat = np.zeros((self.tot_fd_num, self.tot_fd_num))
        rhs = np.zeros((self.tot_fd_num, ))
        for fine_elem_ind in range(self.fine_elem):
            fine_elem_ind_y, fine_elem_ind_x = divmod(fine_elem_ind, self.fine_grid)
            for fd_ind_i in range(self.tot_fd_num):
                node_val_i = self.get_node_val_glb_basis(fine_elem_ind_x, fine_elem_ind_y, fd_ind_i)
                for fd_ind_j in range(self.tot_fd_num):
                    node_val_j = self.get_node_val_glb_basis(fine_elem_ind_x, fine_elem_ind_y, fd_ind_j)
                    A_mat[fd_ind_i, fd_ind_j] += self.get_stiff_quad_node_val(fine_elem_ind_x, fine_elem_ind_y, node_val_i, node_val_j)
                corr_val = self.get_node_val_corr(fine_elem_ind_x, fine_elem_ind_y)
                rhs[fd_ind_i] += self.get_source_quad_node_val(fine_elem_ind_x, fine_elem_ind_y, node_val_i)
                rhs[fd_ind_i] -= self.get_Diri_quad_node_val(fine_elem_ind_x, fine_elem_ind_y, node_val_i)
                # Compute \int_{K_h} A \nabla D^m\cdot \nabla \Phi_i^t \dx
                rhs[fd_ind_i] += self.get_stiff_quad_node_val(fine_elem_ind_x, fine_elem_ind_y, corr_val, node_val_i)
        logging.info("Finish constructing the final linear system of the debug problem.")
        u0 = np.linalg.solve(A_mat, rhs)
        u0_dbg = np.zeros((self.fine_grid + 1, self.fine_grid + 1))
        for fd_ind in range(self.tot_fd_num):
            u0_dbg += u0[fd_ind] * self.glb_basis_list[fd_ind]
        u0_dbg -= self.glb_corr
        logging.info("Finish solving the final linear system of the debug problem.")
        return u0_dbg