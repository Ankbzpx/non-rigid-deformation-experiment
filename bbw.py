import numpy as np
import igl
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
from mesh_helper import read_obj
import polyscope as ps
from icecream import ic
import scipy.optimize
import osqp    # Doc: https://osqp.org/docs/interfaces/python.html#python-interface
import multiprocessing
from joblib import Parallel, delayed

from arap import boundary_condition, boundary_condition_bary


class BoundedBiharmonicWeights:

    def __init__(self, V, F, sp_fid: np.ndarray | None = None, sp_weight=1e2):
        L: scipy.sparse.csc_matrix = igl.cotmatrix(V, F)
        M: scipy.sparse.csc_matrix = igl.massmatrix(V, F,
                                                    igl.MASSMATRIX_TYPE_VORONOI)
        M_inv = scipy.sparse.diags(1 / M.diagonal())
        Q = L @ M_inv @ L

        if sp_fid is not None:
            f_weights = np.zeros(len(F))
            f_weights[sp_fid] = sp_weight
            f_weights = f_weights * igl.doublearea(V, F)

            M_tile = scipy.sparse.diags(np.repeat(f_weights, 3))
            G = igl.grad(V, F)
            Q += G.T @ M_tile @ G

        self.Q = Q
        self.V = V
        self.F = F
        self.NV = len(V)
        self.item_per_batch = 32

    def solve_bbw(self, A: scipy.sparse.spmatrix,
                  handle_weights: np.ndarray) -> np.ndarray:
        prob = osqp.OSQP()
        bbw_weights = []
        for i in range(handle_weights.shape[1]):
            l = np.concatenate([handle_weights[:, i], np.zeros(self.NV)])
            u = np.concatenate([handle_weights[:, i], np.ones(self.NV)])
            if i == 0:
                prob.setup(P=self.Q, A=A, l=l, u=u, verbose=False)
            else:
                prob.update(l=l, u=u)
            prob.warm_start(x=np.zeros((self.NV)))
            res = prob.solve()
            assert res.info.status == 'solved'
            bbw_weights.append(res.x)
        return np.stack(bbw_weights, -1)

    def compute(self,
                b_vid: np.ndarray | None = None,
                b_fid: np.ndarray | None = None,
                b_bary_coords: np.ndarray | None = None) -> np.ndarray:

        A = []
        handle_size = 0
        if b_vid is not None:
            handle_size += len(b_vid)
            A_v, _ = boundary_condition(self.V, b_vid)
            A.append(A_v)
        if b_fid is not None:
            handle_size += len(b_fid)
            A_f, _ = boundary_condition_bary(self.V, self.F, b_fid,
                                             b_bary_coords)
            A.append(A_f)
        A = scipy.sparse.vstack(A + [scipy.sparse.identity(self.NV)]).tocsc()

        handle_weights = np.eye(handle_size)
        total_size = handle_weights.shape[1]
        batch_size = total_size // self.item_per_batch
        weights_batch_list = np.split(handle_weights[:, :batch_size *
                                                     self.item_per_batch],
                                      batch_size,
                                      axis=1)
        if total_size % self.item_per_batch != 0:
            weights_batch_list += [
                handle_weights[:, batch_size * self.item_per_batch:]
            ]

        bbw_weights = np.concatenate(
            Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self.solve_bbw)(A, weights_batch)
                for weights_batch in weights_batch_list),
            axis=1)

        return bbw_weights / np.sum(bbw_weights, 1, keepdims=True)


if __name__ == '__main__':
    np.random.seed(0)
    bar = read_obj('data/bar.obj')

    V = bar.vertices
    F = bar.faces

    NV = len(V)
    v_ids = np.arange(NV)

    handle_size = 100
    handle_idx = np.random.choice(np.arange(NV), handle_size, replace=False)
    handles = V[handle_idx]
    handle_weights = np.eye(handle_size)
    displacement = 5 * igl.per_vertex_normals(V, F)[handle_idx]

    bbw = BoundedBiharmonicWeights(V, F)
    bbw_weights = bbw.compute(handle_idx)

    V_deform = bbw_weights @ displacement + V

    ps.init()
    ps.register_surface_mesh('V', V, F)
    ps.register_surface_mesh('V_deform', V_deform, F)
    ps.register_point_cloud('Handles', V_deform[handle_idx])
    ps.show()
