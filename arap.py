import numpy as np
import igl
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
from icecream import ic
import jax.numpy as jnp
from jax import vmap
from jax.lax import dynamic_slice


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


class LinearVertexSolver:

    def __init__(self,
                 C_upper: scipy.sparse.spmatrix,
                 V: np.ndarray,
                 F: np.ndarray,
                 V_weight: np.ndarray | None = None,
                 b_vid: np.ndarray | None = None,
                 b_v_bounded=True,
                 b_v_weight: np.ndarray | None = None,
                 b_fid: np.ndarray | None = None,
                 b_bary_coords: np.ndarray | None = None,
                 b_f_bounded=True,
                 b_f_weight: np.ndarray | None = None):

        if V_weight is not None:
            assert len(V_weight) == len(V)
        else:
            V_weight = np.ones(len(V))

        C_lower = []
        weights = []
        b_mask = np.ones(len(V)).astype(bool)
        if b_vid is not None:
            C_v, b_v_mask = boundary_condition(V, b_vid)
            C_lower.append(C_v)

            if b_v_weight is not None:
                assert len(b_v_weight) == len(b_vid)
            else:
                b_v_weight = np.ones(len(b_vid))
            weights.append(b_v_weight)

            if b_v_bounded:
                b_mask = np.logical_and(b_mask, b_v_mask)

        if b_fid is not None:
            C_f, b_f_mask = boundary_condition_bary(V, F, b_fid, b_bary_coords)
            C_lower.append(C_f)

            if b_f_weight is not None:
                assert len(b_f_weight) == len(b_fid)
            else:
                b_f_weight = np.ones(len(b_fid))
            weights.append(b_f_weight)

            if b_f_bounded:
                b_mask = np.logical_and(b_mask, b_f_mask)

        C = scipy.sparse.vstack([C_upper[b_mask]] + C_lower)
        C_T = C.T.tocsc()
        weights = np.concatenate([V_weight[b_mask]] + weights)
        W = scipy.sparse.diags(weights)

        self.b_mask = b_mask
        self.NBC = C.shape[0] - np.sum(b_mask)
        self.C_T = C_T
        self.W = W
        self.solve_factorized = scipy.sparse.linalg.factorized(C_T @ W @ C)

    def verify_bc_dim(self, BC: np.ndarray) -> bool:
        assert self.NBC == len(BC)


class BiLaplacian(LinearVertexSolver):

    def __init__(self,
                 V: np.ndarray,
                 F: np.ndarray,
                 V_weight: np.ndarray | None = None,
                 b_vid: np.ndarray | None = None,
                 b_v_bounded=True,
                 b_v_weight: np.ndarray | None = None,
                 b_fid: np.ndarray | None = None,
                 b_bary_coords: np.ndarray | None = None,
                 b_f_bounded=True,
                 b_f_weight: np.ndarray | None = None):
        L: scipy.sparse.csc_matrix = igl.cotmatrix(V, F)
        # Hybrid voronoi that guarantees positive area
        M: scipy.sparse.csc_matrix = igl.massmatrix(V, F,
                                                    igl.MASSMATRIX_TYPE_VORONOI)
        M_inv = scipy.sparse.diags(1 / M.diagonal())
        C_upper = L @ M_inv @ L
        super().__init__(C_upper, V, F, V_weight, b_vid, b_v_bounded,
                         b_v_weight, b_fid, b_bary_coords, b_f_bounded,
                         b_f_weight)
        self.B_upper = np.zeros((np.sum(self.b_mask), 3))

    def solve(self, BC: np.ndarray):
        self.verify_bc_dim(BC)
        B = np.vstack([self.B_upper, BC])
        return self.solve_factorized(self.C_T @ self.W @ B)


class AsRigidAsPossible(LinearVertexSolver):

    def __init__(self,
                 V: np.ndarray,
                 F: np.ndarray,
                 V_weight: np.ndarray | None = None,
                 b_vid: np.ndarray | None = None,
                 b_v_bounded=True,
                 b_v_weight: np.ndarray | None = None,
                 b_fid: np.ndarray | None = None,
                 b_bary_coords: np.ndarray | None = None,
                 b_f_bounded=True,
                 b_f_weight: np.ndarray | None = None):
        L: scipy.sparse.csc_matrix = igl.cotmatrix(V, F)
        # Negative diagonal
        C_upper = -L
        super().__init__(C_upper, V, F, V_weight, b_vid, b_v_bounded,
                         b_v_weight, b_fid, b_bary_coords, b_f_bounded,
                         b_f_weight)

        V_cot_adj_coo = scipy.sparse.coo_array(L)
        valid_entries_mask = V_cot_adj_coo.col != V_cot_adj_coo.row
        # col major
        E_i = V_cot_adj_coo.col[valid_entries_mask]
        E_j = V_cot_adj_coo.row[valid_entries_mask]
        E_weight = V_cot_adj_coo.data[valid_entries_mask]
        E_unique, E_count = np.unique(E_i, return_counts=True)
        assert (E_unique != np.arange(len(V))).sum() == 0
        split_indices = np.cumsum(E_count)[:-1]

        pad_width = np.max(E_count)
        E_i_list = np.split(E_i, split_indices)
        E_j_list = np.split(E_j, split_indices)
        E_weight_list = np.split(E_weight, split_indices)

        def pad_list_of_array(list_of_array):
            return np.array([
                np.concatenate([el, np.zeros(pad_width - len(el))])
                for el in list_of_array
            ])

        E_i = pad_list_of_array(E_i_list).astype(np.int64)
        E_j = pad_list_of_array(E_j_list).astype(np.int64)
        E_weight = pad_list_of_array(E_weight_list)

        self.Eij = jnp.array(V[E_i] - V[E_j])
        self.E_i = jnp.array(E_i)
        self.E_j = jnp.array(E_j)
        self.E_weight = jnp.array(E_weight)

    def solve(self,
              BC: np.ndarray,
              V_arap: np.ndarray,
              max_iters=8) -> np.ndarray:
        self.verify_bc_dim(BC)
        for _ in range(max_iters):
            # TODO: add stiffness weight
            B_upper = self.build_arap_rhs(V_arap)[self.b_mask]
            B = np.vstack([B_upper, BC])
            V_arap = self.solve_factorized(self.C_T @ self.W @ B)
        return V_arap

    def build_arap_rhs(self, V_arap: np.ndarray) -> np.ndarray:
        V_arap = jnp.array(V_arap)
        Eij_ = V_arap[self.E_i] - V_arap[self.E_j]

        def arap_rhs(eij, eij_, e_weight):
            cov = eij_.T @ jnp.diag(e_weight) @ eij
            U, S, V_T = jnp.linalg.svd(cov)
            R = U @ V_T

            # Reflection
            E = jnp.eye(len(cov))
            min_idx = jnp.argmin(S)
            E = E.at[min_idx, min_idx].set(jnp.sign(jnp.linalg.det(R)))
            R = U @ E @ V_T

            return (e_weight[:, None] * eij @ R.T).sum(0)

        B = vmap(arap_rhs)(self.Eij, Eij_, self.E_weight)
        return np.asarray(B)


if __name__ == '__main__':
    import polyscope as ps
    from scipy.spatial.transform import Rotation
    from mesh_helper import read_obj

    bar = read_obj('data/bar.obj')

    V = bar.vertices
    F = bar.faces
    per_face_vertex = V[F]

    NV = len(V)
    NF = len(F)
    v_ids = np.arange(NV)
    handle_ids_0 = bar.vertices[:, 2] > 95
    handle_ids_1 = bar.vertices[:, 2] < 28

    handle_ids_0_f_mask = np.sum((per_face_vertex[..., 2] < 28), 1) == 3
    handle_ids_1_f_mask = np.sum((per_face_vertex[..., 2] > 95), 1) == 3

    f_ids = np.arange(NF)
    boundary_fid = np.concatenate(
        [f_ids[handle_ids_0_f_mask], f_ids[handle_ids_1_f_mask]])
    bary_coords = np.vstack([
        np.ones((np.sum(handle_ids_0_f_mask), 3)),
        np.ones((np.sum(handle_ids_1_f_mask), 3))
    ]) / 3

    BC = (per_face_vertex[boundary_fid] * bary_coords[..., None]).sum(1)

    # deformation transformation
    R_deform = Rotation.from_rotvec(np.array([0, 0, 2 * np.pi / 3])).as_matrix()
    t_deform = np.array([0, -10, -10])

    BC_split = np.sum(handle_ids_0_f_mask)
    BC[BC_split:] = BC[BC_split:] @ R_deform.T + t_deform

    bilaplacian = BiLaplacian(V,
                              F,
                              b_fid=boundary_fid,
                              b_bary_coords=bary_coords)
    V_init = bilaplacian.solve(BC)

    arap = AsRigidAsPossible(V,
                             F,
                             b_fid=boundary_fid,
                             b_bary_coords=bary_coords)
    V_arap = arap.solve(BC, V_init)

    ps.init()
    ps.register_surface_mesh('bar', V, F, enabled=False)
    ps.register_surface_mesh('bar_init', V_init, F, enabled=False)
    ps.register_surface_mesh('bar_arap', V_arap, F)
    ps.show()
