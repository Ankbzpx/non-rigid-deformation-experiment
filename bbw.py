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
import time

from arap import boundary_condition, boundary_condition_bary

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

    L: scipy.sparse.csc_matrix = igl.cotmatrix(V, F)
    M: scipy.sparse.csc_matrix = igl.massmatrix(V, F,
                                                igl.MASSMATRIX_TYPE_VORONOI)
    M_inv = scipy.sparse.diags(1 / M.diagonal())
    Q = L @ M_inv @ L
    A_eq, _ = boundary_condition(V, handle_idx)
    A = scipy.sparse.vstack([A_eq, scipy.sparse.identity(NV)]).tocsc()

    item_per_batch = 32
    total_size = handle_weights.shape[0]
    batch_size = total_size // item_per_batch
    weights_batch_list = np.split(handle_weights[:, :batch_size *
                                                 item_per_batch],
                                  batch_size,
                                  axis=1)
    if total_size % item_per_batch != 0:
        weights_batch_list += [handle_weights[:, batch_size * item_per_batch:]]

    def solve_bbw(weights):
        prob = osqp.OSQP()
        bbw_weights = []
        for i in range(weights.shape[1]):
            l = np.concatenate([weights[:, i], np.zeros(NV)])
            u = np.concatenate([weights[:, i], np.ones(NV)])
            if i == 0:
                prob.setup(Q, A=A, l=l, u=u, verbose=False)
            else:
                prob.update(l=l, u=u)
            prob.warm_start(x=np.zeros((NV)))
            res = prob.solve()
            assert res.info.status == 'solved'
            bbw_weights.append(res.x)
        return np.stack(bbw_weights, -1)

    bbw_weights = np.concatenate(Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(solve_bbw)(weights_batch)
        for weights_batch in weights_batch_list),
                                 axis=1)

    bbw_weights = bbw_weights / np.sum(bbw_weights, 1, keepdims=True)

    V_deform = bbw_weights @ displacement + V

    ps.init()
    ps.register_surface_mesh('V', V, F)
    ps.register_surface_mesh('V_deform', V_deform, F)
    ps.register_point_cloud('Handles', V_deform[handle_idx])
    ps.show()
