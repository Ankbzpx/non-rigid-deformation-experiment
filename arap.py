import numpy as np
import igl
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg


def boundary_condition_bary(V, F, b_fid, b_bary_coords):
    '''
    Boundary condition barycentric
        V: mesh vertices
        F: mesh faces
        b_fid: boundary faces
        b_bary_coords: barycentric coordinate for each boundary face
    '''
    NV = len(V)
    NH = len(b_fid)
    b_vid = F[b_fid].reshape(-1)

    C = scipy.sparse.coo_matrix(
        (b_bary_coords.reshape(-1), (np.repeat(np.arange(NH), 3), b_vid)),
        shape=(NH, NV)).tocsc()

    b_mask = np.ones(NV)
    b_mask[b_vid] = 0
    b_mask = b_mask.astype(bool)

    return C, b_mask


def boundary_condition(V, b_vid):
    '''
    Boundary condition
        V: mesh vertices
        b_vid: boundary vertex index
    '''
    NV = len(V)
    NH = len(b_vid)
    C = scipy.sparse.coo_matrix((np.ones(NH), (np.arange(NH), b_vid)),
                                shape=(NH, NV)).tocsc()

    b_mask = np.ones(NV)
    b_mask[b_vid] = 0
    b_mask = b_mask.astype(bool)

    return C, b_mask


class VertsLinearSolver:

    def __init__(self,
                 C_upper: scipy.sparse.spmatrix,
                 V: np.ndarray,
                 F: np.ndarray,
                 b_vid: np.ndarray | None = None,
                 b_fid: np.ndarray | None = None,
                 b_bary_coords: np.ndarray | None = None):
        C_lower = []
        b_mask = np.ones(len(V)).astype(bool)
        if b_vid is not None:
            C_v, b_v = boundary_condition(V, b_vid)
            C_lower.append(C_v)
            b_mask = np.logical_and(b_mask, b_v)
        if b_fid is not None:
            C_f, b_f = boundary_condition_bary(V, F, b_fid, b_bary_coords)
            C_lower.append(C_f)
            b_mask = np.logical_and(b_mask, b_f)

        C = scipy.sparse.vstack([C_upper[b_mask]] + C_lower)

        self.b_mask = b_mask
        self.NBC = NV - np.sum(b_mask)
        self.C_T = C.T
        self.solve_factorized = scipy.sparse.linalg.factorized(C.T @ C)

    def verify_bc_dim(self, BC: np.ndarray) -> bool:
        assert self.NBC == len(BC)


class BiLaplacian(VertsLinearSolver):

    def __init__(self,
                 V: np.ndarray,
                 F: np.ndarray,
                 b_vid: np.ndarray | None = None,
                 b_fid: np.ndarray | None = None,
                 b_bary_coords: np.ndarray | None = None):
        L: scipy.sparse.csc_matrix = igl.cotmatrix(V, F)
        # Hybrid voronoi that guarantees positive area
        M: scipy.sparse.csc_matrix = igl.massmatrix(V, F,
                                                    igl.MASSMATRIX_TYPE_VORONOI)
        M_inv = scipy.sparse.diags(1 / M.diagonal())
        C_upper = L @ M_inv @ L
        super().__init__(C_upper, V, F, b_vid, b_fid, b_bary_coords)
        self.B_upper = np.zeros((np.sum(self.b_mask), 3))

    def solve(self, BC: np.ndarray):
        self.verify_bc_dim(BC)
        B = np.vstack([self.B_upper, BC])
        return self.solve_factorized(self.C_T @ B)


class AsRigidAsPossible(VertsLinearSolver):

    def __init__(self,
                 V: np.ndarray,
                 F: np.ndarray,
                 b_vid: np.ndarray | None = None,
                 b_fid: np.ndarray | None = None,
                 b_bary_coords: np.ndarray | None = None):
        L: scipy.sparse.csc_matrix = igl.cotmatrix(V, F)
        # Negative diagonal
        C_upper = -L
        super().__init__(C_upper, V, F, b_vid, b_fid, b_bary_coords)

        V_cot_adj_coo = scipy.sparse.coo_array(L)
        valid_entries_mask = V_cot_adj_coo.col != V_cot_adj_coo.row
        # col major
        E_i = V_cot_adj_coo.col[valid_entries_mask]
        E_j = V_cot_adj_coo.row[valid_entries_mask]
        E_weight = V_cot_adj_coo.data[valid_entries_mask]
        E_unique, E_count = np.unique(E_i, return_counts=True)
        assert (E_unique != np.arange(len(V))).sum() == 0
        split_indices = np.cumsum(E_count)[:-1]

        self.V = V
        self.E_i_list = np.split(E_i, split_indices)
        self.E_j_list = np.split(E_j, split_indices)
        self.E_weight_list = np.split(E_weight, split_indices)

    def solve(self,
              BC: np.ndarray,
              V_arap: np.ndarray,
              max_iters=8) -> np.ndarray:
        self.verify_bc_dim(BC)
        for _ in range(max_iters):
            # TODO: add stiffness weight
            B_upper = self.build_arap_rhs(V_arap)[self.b_mask]
            B = np.vstack([B_upper, BC])
            V_arap = self.solve_factorized(self.C_T @ B)
        return V_arap

    def build_arap_rhs(self, V_arap: np.ndarray) -> np.ndarray:
        arap_rhs = []
        for (e_i, e_j, e_weight) in zip(self.E_i_list, self.E_j_list,
                                        self.E_weight_list):
            eij = self.V[e_i] - self.V[e_j]
            eij_ = V_arap[e_i] - V_arap[e_j]
            cov = eij_.T @ np.diag(e_weight) @ eij

            U, S, V_T = np.linalg.svd(cov)
            R = U @ V_T

            # Reflection
            if np.linalg.det(R) < 0:
                E = np.eye(len(cov))
                min_idx = np.argmin(S)
                E[min_idx, min_idx] = -1
                R = U @ E @ V_T

            b = (e_weight[:, None] * eij @ R.T).sum(0)
            arap_rhs.append(b)
        arap_rhs = np.stack(arap_rhs)
        return arap_rhs


if __name__ == '__main__':
    import polyscope as ps
    from icecream import ic
    from scipy.spatial.transform import Rotation
    from mesh_helper import read_obj

    bar = read_obj('data/bar.obj')

    V = bar.vertices
    F = bar.faces

    NV = len(V)
    v_ids = np.arange(NV)
    handle_ids_0 = bar.vertices[:, 2] > 95
    handle_ids_1 = bar.vertices[:, 2] < 28

    # boundary indices
    boundary_ids = np.concatenate([v_ids[handle_ids_0], v_ids[handle_ids_1]])

    # deformation transformation
    R_deform = Rotation.from_rotvec(np.array([0, 0, 2 * np.pi / 3])).as_matrix()
    t_deform = np.array([0, -10, -10])

    V_deform = np.copy(V)
    V_deform[handle_ids_1] = V_deform[handle_ids_1] @ R_deform.T + t_deform

    bilaplacian = BiLaplacian(V, F, boundary_ids)
    V_init = bilaplacian.solve(V_deform[boundary_ids])

    arap = AsRigidAsPossible(V, F, boundary_ids)
    V_arap = arap.solve(V_init[boundary_ids], V_init)

    arap_igl = igl.ARAP(V, F, 3, boundary_ids)
    V_arap_igl = arap_igl.solve(V_init[boundary_ids], V_init)

    ps.init()
    ps.register_surface_mesh('bar', V, F, enabled=False)
    ps.register_surface_mesh('bar_deform', V_deform, F, enabled=False)
    ps.register_surface_mesh('bar_init', V_init, F, enabled=False)
    ps.register_surface_mesh('bar_arap', V_arap, F)
    ps.register_surface_mesh('bar_arap_igl', V_arap_igl, F, enabled=False)
    ps.show()
