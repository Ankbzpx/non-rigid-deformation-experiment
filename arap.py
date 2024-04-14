import numpy as np
import igl
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import sparseqr
from icecream import ic
import jax.numpy as jnp
from jax import vmap, jit


def boundary_condition_bary(V, F, b_fid, b_bary_coords):
    '''
    Helper to build Boundary condition from barycentric coordinates

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
    Helper to Boundary condition from vertex index

        V: mesh vertices
        b_vid: vertex boundary index
    '''
    NV = len(V)
    NH = len(b_vid)
    C = scipy.sparse.coo_matrix((np.ones(NH), (np.arange(NH), b_vid)),
                                shape=(NH, NV)).tocsc()

    b_mask = np.ones(NV)
    b_mask[b_vid] = 0
    b_mask = b_mask.astype(bool)

    return C, b_mask


def reduce_full_rank(C):
    Q, R, E, rank = sparseqr.qr(C)
    Q = Q.tocsc()[:, :rank]
    R = R.tocsc()
    P = sparseqr.permutation_vector_to_matrix(E)

    # Q @ R = C @ P => Q @ R @ M.T = C
    # C @ x = d => R @ M.T @ x = Q.T @ d
    return (R @ P.T)[:rank], lambda x: Q.T @ x


class LinearVertexSolver:
    '''
    Solve C @ V = B with respect to boundary condition V_b = B_b
        Specifically, we build
            C = [A, C_b]^T
            B = [B, B_b]^T
        then solve
            C^T @ C @ X = C^T @ B

        A: objective
        V: mesh vertices
        F: mesh faces
        V_weight: per vertex weight
        b_vid: vertex boundary index
        b_fid: boundary faces
        b_bary_coords: barycentric coordinate for each boundary face
    '''

    def __init__(self,
                 A: scipy.sparse.spmatrix,
                 V: np.ndarray,
                 F: np.ndarray,
                 V_weight: np.ndarray | None = None,
                 b_vid: np.ndarray | None = None,
                 b_fid: np.ndarray | None = None,
                 b_bary_coords: np.ndarray | None = None):

        if V_weight is not None:
            assert len(V_weight) == len(V)
        else:
            V_weight = np.ones(len(V))

        constraints = []
        if b_vid is not None:
            C_v, b_v_mask = boundary_condition(V, b_vid)
            constraints.append(C_v)

        if b_fid is not None:
            C_f, b_f_mask = boundary_condition_bary(V, F, b_fid, b_bary_coords)
            constraints.append(C_f)

        C_T = scipy.sparse.vstack(constraints)

        # Use QR to reduce the rank of constraints
        C_T, self.reduce = reduce_full_rank(C_T)

        W = scipy.sparse.diags(V_weight)
        M = scipy.sparse.vstack([
            scipy.sparse.hstack([W @ A, C_T.T]),
            scipy.sparse.hstack(
                [C_T,
                 scipy.sparse.csc_matrix((C_T.shape[0], C_T.shape[0]))])
        ]).tocsc()
        self.W = W

        self.solve_factorized = scipy.sparse.linalg.factorized(M)

    def solve_(self, b, d):
        return self.solve_factorized(np.vstack([self.W @ b, d]))[:len(b)]


class BiLaplacian(LinearVertexSolver):

    def __init__(self,
                 V: np.ndarray,
                 F: np.ndarray,
                 V_weight: np.ndarray | None = None,
                 b_vid: np.ndarray | None = None,
                 b_fid: np.ndarray | None = None,
                 b_bary_coords: np.ndarray | None = None):
        L: scipy.sparse.csc_matrix = igl.cotmatrix(V, F)
        # Hybrid voronoi that guarantees positive area
        M: scipy.sparse.csc_matrix = igl.massmatrix(V, F,
                                                    igl.MASSMATRIX_TYPE_VORONOI)
        M_inv = scipy.sparse.diags(1 / M.diagonal())
        A = L @ M_inv @ L
        super().__init__(A, V, F, V_weight, b_vid,
                         b_fid, b_bary_coords)
        self.n = len(V)

    def solve(self, BC: np.ndarray):
        d = self.reduce(BC)
        return self.solve_(np.zeros((self.n, 3)), d)


# https://igl.ethz.ch/projects/ARAP/arap_web.pdf
class AsRigidAsPossible(LinearVertexSolver):

    def __init__(self,
                 V: np.ndarray,
                 F: np.ndarray,
                 V_weight: np.ndarray | None = None,
                 b_vid: np.ndarray | None = None,
                 b_fid: np.ndarray | None = None,
                 b_bary_coords: np.ndarray | None = None):
        L: scipy.sparse.csc_matrix = igl.cotmatrix(V, F)
        # Negative diagonal
        A = -L
        super().__init__(A, V, F, V_weight, b_vid,
                         b_fid, b_bary_coords)
        # get one-ring neighbour from cotangent matrix
        V_cot_adj_coo = scipy.sparse.coo_array(L)
        valid_entries_mask = V_cot_adj_coo.col != V_cot_adj_coo.row
        # col major
        E_i = V_cot_adj_coo.col[valid_entries_mask]
        E_j = V_cot_adj_coo.row[valid_entries_mask]
        E_weight = V_cot_adj_coo.data[valid_entries_mask]
        E_unique, E_count = np.unique(E_i, return_counts=True)
        assert (E_unique != np.arange(len(V))).sum() == 0
        split_indices = np.cumsum(E_count)[:-1]

        E_i_list = np.split(E_i, split_indices)
        E_j_list = np.split(E_j, split_indices)
        E_weight_list = np.split(E_weight, split_indices)

        # pad so it can be vmapped
        pad_width = np.max(E_count)

        def pad_list_of_array(list_of_array):
            return np.array([
                np.concatenate([el, np.zeros(pad_width - len(el))])
                for el in list_of_array
            ])

        E_i = pad_list_of_array(E_i_list).astype(np.int64)
        E_j = pad_list_of_array(E_j_list).astype(np.int64)
        E_weight = pad_list_of_array(E_weight_list)

        self.Eij = jnp.array(V[E_i] - V[E_j])
        self.E_i = E_i
        self.E_j = E_j
        self.E_weight = jnp.array(E_weight)

    def solve(self,
              BC: np.ndarray,
              V_arap: np.ndarray,
              max_iters=8) -> np.ndarray:
        d = self.reduce(BC)
        for _ in range(max_iters):
            # minimize R
            b = self.build_arap_rhs(V_arap)
            V_arap = self.solve_(b, d)
        return V_arap

    def build_arap_rhs(self, V_arap: np.ndarray) -> np.ndarray:
        Eij_ = jnp.array(V_arap[self.E_i] - V_arap[self.E_j])

        # \sum_{j \in \mathcal{N}(i)} w_{ij} R_i (p_i - p_j)
        @jit
        def arap_rhs(eij, eij_, e_weight):
            cov = eij_.T @ jnp.diag(e_weight) @ eij
            U, S, V_T = jnp.linalg.svd(cov)
            R = U @ V_T

            # Handle reflection
            E = jnp.eye(len(cov))
            min_idx = jnp.argmin(S)
            E = E.at[min_idx, min_idx].set(jnp.sign(jnp.linalg.det(R)))
            R = U @ E @ V_T

            return (e_weight[:, None] * eij @ R.T).sum(0)

        RHS = vmap(arap_rhs)(self.Eij, Eij_, self.E_weight)
        return np.asarray(RHS)


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
