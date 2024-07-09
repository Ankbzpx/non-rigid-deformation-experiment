import numpy as np
import igl
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import sparseqr
import jax.numpy as jnp
import jax
from jax import vmap, jit
from jax.scipy.spatial.transform import Rotation
from sksparse.cholmod import cholesky_AAt

import polyscope as ps
from icecream import ic


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


# TODO: Implement simultaneous hard and soft constraints
class LinearVertexSolver:
    '''
    Solve C @ V = B with respect to boundary condition V_b = B_b
        For hard constraints, we use Lagrange multiplier
        For soft constraints, we build
                M = [A, C]^T
                B = [b, d]^T
            then solve
                M^T @ M @ X = M^T @ B

        A: objective
        V: mesh vertices
        F: mesh faces
        V_weight: per vertex weight
        b_vid: vertex boundary index
        b_fid: boundary faces
        b_bary_coords: barycentric coordinate for each boundary face
        soft_weight: treat as soft constraint if the weight is larger than 0
        soft_exclude: exclude soft constraints in primary objective
    '''

    def __init__(self,
                 A: scipy.sparse.spmatrix,
                 V: np.ndarray,
                 F: np.ndarray,
                 V_weight: np.ndarray | None = None,
                 b_vid: np.ndarray | None = None,
                 b_fid: np.ndarray | None = None,
                 b_bary_coords: np.ndarray | None = None,
                 soft_weight: float = 0.,
                 soft_exclude: bool = True):

        if V_weight is not None:
            assert len(V_weight) == len(V)
        else:
            V_weight = np.ones(len(V))

        constraints = []
        b_mask = np.ones(len(V)).astype(bool)
        if b_vid is not None:
            C_v, b_v_mask = boundary_condition(V, b_vid)
            constraints.append(C_v)

            if soft_exclude:
                b_mask = np.logical_and(b_mask, b_v_mask)

        if b_fid is not None:
            C_f, b_f_mask = boundary_condition_bary(V, F, b_fid, b_bary_coords)
            constraints.append(C_f)

            if soft_exclude:
                b_mask = np.logical_and(b_mask, b_f_mask)

        C_T = scipy.sparse.vstack(constraints)

        if isinstance(soft_weight, float) or isinstance(soft_weight, int):
            soft_weight = soft_weight * np.ones(C_T.shape[0])

        self.is_soft = soft_weight.sum() > 0
        if self.is_soft:
            M = scipy.sparse.vstack([A[b_mask], C_T])
            M_T = M.T.tocsc()
            W = scipy.sparse.diags(
                np.concatenate([V_weight[b_mask], soft_weight]))

            self.solve_factorized = scipy.sparse.linalg.factorized(M_T @ W @ M)
            self.M_T = M_T
            self.W = W
            self.b_mask = b_mask
        else:
            # Use QR to reduce the rank of constraints
            C_T, self.reduce = reduce_full_rank(C_T)
            W = scipy.sparse.diags(V_weight)
            M = scipy.sparse.vstack([
                scipy.sparse.hstack([W @ A, C_T.T]),
                scipy.sparse.hstack([
                    C_T,
                    scipy.sparse.csc_matrix((C_T.shape[0], C_T.shape[0]))
                ])
            ]).tocsc()
            self.W = W
            self.solve_factorized = scipy.sparse.linalg.factorized(M)

    def solve_(self, b, BC):
        if self.is_soft:
            return self.solve_factorized(
                self.M_T @ self.W @ np.vstack([b[self.b_mask], BC]))
        else:
            d = self.reduce(BC)
            return self.solve_factorized(np.vstack([self.W @ b, d]))[:len(b)]


class BiLaplacian(LinearVertexSolver):

    def __init__(self,
                 V: np.ndarray,
                 F: np.ndarray,
                 V_weight: np.ndarray | None = None,
                 b_vid: np.ndarray | None = None,
                 b_fid: np.ndarray | None = None,
                 b_bary_coords: np.ndarray | None = None,
                 soft_weight: float | np.ndarray = 0.,
                 soft_exclude: bool = True):
        L: scipy.sparse.csc_matrix = igl.cotmatrix(V, F)
        # Hybrid voronoi that guarantees positive area
        M: scipy.sparse.csc_matrix = igl.massmatrix(V, F,
                                                    igl.MASSMATRIX_TYPE_VORONOI)
        M_inv = scipy.sparse.diags(1 / M.diagonal())
        A = L @ M_inv @ L
        super().__init__(A, V, F, V_weight, b_vid, b_fid, b_bary_coords,
                         soft_weight, soft_exclude)
        self.n = len(V)

    def solve(self, BC: np.ndarray):
        return self.solve_(np.zeros((self.n, 3)), BC)


@jit
def fit_cov(eij, eij_, e_weight):
    return eij_.T @ jnp.diag(e_weight) @ eij


@jit
def fit_R(cov):
    U, S, V_T = jnp.linalg.svd(cov)
    R = U @ V_T

    # Handle reflection
    E = jnp.eye(len(cov))
    min_idx = jnp.argmin(S)
    E = E.at[min_idx, min_idx].set(jnp.sign(jnp.linalg.det(R)))
    R = U @ E @ V_T
    return R


@jit
def fit_cov_SR(e_weight, R_j, w_sr):
    return 2 * w_sr * (e_weight[:, None, None] * R_j).sum(0)


@jit
def fit_cov_surrogate(d, R, n_p, n_q, w_sr):
    d_norm = jnp.linalg.norm(d)
    h = R @ n_p - d * ((n_q + R @ n_p) @ d) / (d_norm**2)
    return w_sr * d_norm**2 * n_p[:, None] @ h[:, None].T


# \sum_{j \in \mathcal{N}(i)} 0.5 * w_{ij} (R_i + R_j) @ (p_i - p_j)
@jit
def arap_rhs(R_i, R_j, eij, e_weight):
    return (0.5 * e_weight[:, None] *
            jnp.einsum('bn,bmn->bm', eij, R_i + R_j)).sum(0)


# ARAP: https://igl.ethz.ch/projects/ARAP/arap_web.pdf
# SR-ARAP: https://zoharl3.github.io/publ/arap.pdf
# TODO: Implement triangle edge sets for SR-ARAP
class AsRigidAsPossible(LinearVertexSolver):

    def __init__(self,
                 V: np.ndarray,
                 F: np.ndarray,
                 V_weight: np.ndarray | None = None,
                 b_vid: np.ndarray | None = None,
                 b_fid: np.ndarray | None = None,
                 b_bary_coords: np.ndarray | None = None,
                 soft_weight: np.ndarray | float = 0.,
                 soft_exclude: bool = True,
                 smooth_rotation=False):
        L: scipy.sparse.csc_matrix = igl.cotmatrix(V, F)
        # Negative diagonal
        super().__init__(-L, V, F, V_weight, b_vid, b_fid, b_bary_coords,
                         soft_weight, soft_exclude)
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

        self.smooth_rotation = smooth_rotation
        if smooth_rotation:
            self.A = igl.doublearea(V, F).sum() / 2

    def solve(self,
              V_arap: np.ndarray,
              BC: np.ndarray,
              max_iters=8,
              w_sr=1e-5) -> np.ndarray:
        # Should not save as a class state
        Rs = np.repeat(np.eye(3)[None, ...], len(V_arap), axis=0)
        for _ in range(max_iters):
            # minimize R
            b, Rs = self.build_arap_rhs(Rs, V_arap, w_sr)
            V_arap = self.solve_(b, BC)
        return V_arap

    def build_arap_rhs(self, Rs: np.ndarray, V_arap: np.ndarray,
                       w_sr: float) -> np.ndarray:
        Eij_ = jnp.array(V_arap[self.E_i] - V_arap[self.E_j])

        cov = vmap(fit_cov)(self.Eij, Eij_, self.E_weight)
        if self.smooth_rotation:
            cov += vmap(fit_cov_SR,
                        in_axes=(0, 0, None))(self.E_weight, Rs[self.E_j],
                                              w_sr * self.A)

        Rs = vmap(fit_R)(cov)

        Rs_i = Rs[self.E_i]
        Rs_j = Rs[self.E_j]
        RHS = vmap(arap_rhs)(Rs_i, Rs_j, self.Eij, self.E_weight)
        return np.asarray(RHS), Rs


# Replace entries in sparse matrix by coefficient weighted identity blocks
def unroll_identity_block(A, dim):
    H, W = A.shape
    A_coo = scipy.sparse.coo_array(A)
    A_unroll_row = ((dim * A_coo.row)[..., None] +
                    np.arange(dim)[None, ...]).reshape(-1)
    A_unroll_col = ((dim * A_coo.col)[..., None] +
                    np.arange(dim)[None, ...]).reshape(-1)
    A_unroll_data = np.repeat(A_coo.data, dim)

    return scipy.sparse.csc_array((A_unroll_data, (A_unroll_row, A_unroll_col)),
                                  shape=(dim * H, dim * W))


@jit
def R3_to_rotvec(R):
    return Rotation.from_matrix(R).as_rotvec()


@jit
def skew_symmetric3(rotvec):
    return jnp.array([[0, -rotvec[2], rotvec[1]], [rotvec[2], 0, -rotvec[0]],
                      [-rotvec[1], rotvec[0], 0]])


@jit
def rotvec_to_R3_Rodrigues(rotvec):
    rotvec_norm = jnp.linalg.norm(rotvec)
    A = skew_symmetric3(rotvec / rotvec_norm)
    return jnp.eye(
        3) + jnp.sin(rotvec_norm) * A + (1 - jnp.cos(rotvec_norm)) * A @ A


# First order approximation
@jit
def rotvec_to_R3_approx(rotvec):
    return jnp.eye(3) + skew_symmetric3(rotvec)


@jit
def rotvec_to_R3(rotvec):
    return Rotation.from_rotvec(rotvec).as_matrix()


# https://arxiv.org/pdf/2405.20188
class SymmetricPointToPlane:

    def __init__(self,
                 V: np.ndarray,
                 F: np.ndarray,
                 b_vid: np.ndarray | None = None,
                 b_fid: np.ndarray | None = None,
                 b_bary_coords: np.ndarray | None = None,
                 smooth_rotation=True):

        constraints = []
        self.b_v_mask = None
        if b_vid is not None:
            C_v, b_mask = boundary_condition(V, b_vid)
            self.b_v_mask = np.logical_not(b_mask)
            constraints.append(C_v)

        if b_fid is not None:
            C_f, _ = boundary_condition_bary(V, F, b_fid, b_bary_coords)
            constraints.append(C_f)

        self.C = scipy.sparse.vstack(constraints)

        # ARAP
        L: scipy.sparse.csc_matrix = igl.cotmatrix(V, F)
        self.L_unroll = unroll_identity_block(-L, 3)

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

        self.smooth_rotation = smooth_rotation
        self.A = igl.doublearea(V, F).sum() / 2

    def build_robust_weight(self, P, Q):
        D = P - Q
        dists = np.linalg.norm(D, axis=1)
        sigma = np.median(dists)
        return np.exp(-(dists**2) / (2 * sigma**2))

    def build_arap_rhs(self, Rs: np.ndarray, D: np.ndarray, weight: np.ndarray,
                       N_p: np.ndarray, N_q: np.ndarray, V_arap: np.ndarray,
                       w_sr: float, w_arap: float):
        Eij_ = jnp.array(V_arap[self.E_i] - V_arap[self.E_j])

        # Local
        cov = vmap(fit_cov)(self.Eij, Eij_, self.E_weight)
        if self.smooth_rotation:
            cov += vmap(fit_cov_SR,
                        in_axes=(0, 0, None))(self.E_weight, Rs[self.E_j],
                                              w_sr * self.A)
        Rs = vmap(fit_R)(cov)

        rotvecs = vmap(R3_to_rotvec)(Rs)
        R_p = vmap(rotvec_to_R3)(self.C @ rotvecs)

        cov_P = (self.C @ cov.reshape(-1, 9)).reshape(-1, 3, 3)
        cov_P = w_arap * cov_P + vmap(fit_cov_surrogate)(D, R_p, N_p, N_q,
                                                         weight)
        R_p = vmap(fit_R)(cov_P)

        # Update R_s with new R_p, at least for non barycentric ones
        if self.b_v_mask is not None:
            b_v_count = self.b_v_mask.sum()
            Rs = Rs.at[self.b_v_mask].set(R_p[:b_v_count])

        Rs_i = Rs[self.E_i]
        Rs_j = Rs[self.E_j]
        RHS = vmap(arap_rhs)(Rs_i, Rs_j, self.Eij, self.E_weight)
        return np.asarray(RHS).reshape(-1), Rs, R_p

    def build_global(self, weight, R_p, N_p, Q, N_q):
        W = scipy.sparse.diags(weight)
        N_sym = np.einsum('bmn,bn->bm', R_p, N_p) + N_q

        h, w = self.C.shape
        C_coo = scipy.sparse.coo_array(self.C)
        N_row = np.repeat(C_coo.row, 3)
        N_col = ((3 * C_coo.col)[..., None] +
                 np.arange(3)[None, ...]).reshape(-1)
        N_data = np.repeat(C_coo.data, 3) * N_sym[C_coo.row].reshape(-1)
        N = scipy.sparse.csc_array((N_data, (N_row, N_col)), shape=(h, 3 * w))
        C = W @ N
        d = W @ np.einsum('bn,bn->b', N_sym, Q)

        M = scipy.sparse.vstack([self.L_unroll, C])
        return M, d

    # TODO: Use symbolic factorization
    def solve(self,
              V_arap: np.ndarray,
              N_p: np.ndarray,
              Q: np.ndarray,
              N_q: np.ndarray,
              max_iters=3,
              robust_weight=True,
              w_arap=1,
              w_sr=1e-5):
        Rs = np.repeat(np.eye(3)[None, ...], len(V_arap), axis=0)

        factor = None
        for _ in range(max_iters):
            P = self.C @ V_arap
            weight = self.build_robust_weight(
                P, Q) if robust_weight else 1.0 * np.ones(len(Q))
            b, Rs, R_p = self.build_arap_rhs(Rs, P - Q, weight, N_p, N_q,
                                             V_arap, w_sr, w_arap)
            M, d = self.build_global(weight, R_p, N_p, Q, N_q)
            N_p = np.einsum('bmn,bn->bm', R_p, N_p)

            W_sqrt = scipy.sparse.diags(
                np.concatenate(
                    [np.sqrt(w_arap) * np.ones(len(b)),
                     np.ones(len(d))]))
            A = (M.T @ W_sqrt).tocsc()

            if factor is None:
                factor = cholesky_AAt(A)
            else:
                # Symbolic factorization
                factor.cholesky_AAt_inplace(A)

            V_arap = factor(A @ W_sqrt @ np.concatenate([b, d])).reshape(-1, 3)

        return V_arap


if __name__ == '__main__':
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
    R_deform = Rotation.from_rotvec(np.array([np.pi, 0, 0])).as_matrix()
    t_deform = np.array([0, -20, 120])

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
    V_arap = arap.solve(V_init, BC)

    sr_arap = AsRigidAsPossible(V,
                                F,
                                b_fid=boundary_fid,
                                b_bary_coords=bary_coords,
                                smooth_rotation=True)
    V_sr_arap = sr_arap.solve(V_init, BC)

    ps.init()
    ps.register_surface_mesh('bar', V, F, enabled=False)
    ps.register_surface_mesh('bar_init', V_init, F, enabled=False)
    ps.register_surface_mesh('bar_arap', V_arap, F)
    ps.register_surface_mesh('bar_arap_sr', V_sr_arap, F)
    ps.show()
