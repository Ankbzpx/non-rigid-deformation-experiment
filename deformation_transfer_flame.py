import numpy as np
import igl
import pickle

from non_rigid_deformation_arap_flame import deformation_gradient
import scipy.sparse
from scipy.sparse.linalg import factorized

import polyscope as ps
from icecream import ic

if __name__ == '__main__':
    flame_path = 'templates/flame2023.pkl'

    with open(flame_path, 'rb') as f:
        data = pickle.load(f, encoding="latin1")

    V = np.float64(data['v_template'])
    F = np.int64(data['f'])

    shapedirs = np.float64(data['shapedirs'])

    np.random.seed(0)
    shape_idx = 1
    weight = -5

    V_deform = V + weight * shapedirs[..., shape_idx]

    flame_path = 'results/head1_match_arap_nicp.obj'
    V_temp, _ = igl.read_triangle_mesh(flame_path)

    J = deformation_gradient(V, V_deform, F)
    S = np.vstack([J[..., 0], J[..., 1], J[..., 2]])

    G = igl.grad(V_temp, F)
    d_area = igl.doublearea(V_temp, F)
    D = scipy.sparse.diags(np.hstack([d_area, d_area, d_area]) * 0.5)

    # 3929
    eyeball_center_l_idx = 3929
    eyeball_center_r_idx = 3930

    # 546
    eyeball_l_idx = 3931
    eyeball_r_idx = 3931 + 546

    eyeball_l_offset = V_temp[eyeball_l_idx:eyeball_r_idx] - V_temp[
        eyeball_center_l_idx][None, :]
    eyeball_r_offset = V_temp[eyeball_r_idx:] - V_temp[eyeball_center_r_idx][
        None, :]

    bc_vert_id = np.array([0, eyeball_l_idx, eyeball_r_idx])

    A = scipy.sparse.csc_array(
        (np.ones(len(bc_vert_id)), (np.arange(len(bc_vert_id)), bc_vert_id)),
        shape=(len(bc_vert_id), len(V_temp)))

    b = V_temp[bc_vert_id]

    Q = scipy.sparse.vstack([G.T @ D @ G, A])
    c = np.vstack([G.T @ D @ S, b])

    solve = factorized((Q.T @ Q).tocsc())
    V_transfer = solve(Q.T @ c)

    # Try to fix eyeball
    V_transfer[eyeball_l_idx:eyeball_r_idx] = eyeball_l_offset + V_transfer[
        eyeball_center_l_idx][None, :]
    V_transfer[eyeball_r_idx:] = eyeball_r_offset + V_transfer[
        eyeball_center_r_idx][None, :]

    ps.init()
    ps.register_surface_mesh(f'mesh source', V_temp, F)
    ps.register_surface_mesh(f'mesh target', V_transfer, F)
    # ps.register_surface_mesh(f'basemesh source', V, F)
    # ps.register_surface_mesh(f'basemesh target', V_deform, F)
    ps.show()
