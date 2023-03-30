import numpy as np
from icecream import ic
import trimesh
import polyscope as ps

# Depends on GPU memory
kMaxQuerySize = 2e9


def match_correspondence(P, Q, matched):
    P_copy = np.copy(P)
    Q_copy = np.copy(Q)

    P = P[matched[:, 0]]
    Q = Q[matched[:, 0]]

    P_mu = np.average(P, 0, keepdims=True)
    P_bar = P - P_mu
    Q = scan.vertices[known_correspondences[:, 1]]
    Q_mu = np.average(Q, 0, keepdims=True)
    Q_bar = Q - Q_mu

    cov = Q_bar.T @ P_bar / len(known_correspondences)
    U, S, V_T = np.linalg.svd(cov)
    R = U @ V_T

    # Reflection
    E = np.ones(len(cov))
    E[-1] = np.sign(np.linalg.det(R))
    R = R * E

    demo = np.sum(P_bar * P_bar) / len(known_correspondences)

    s = np.sum(S) / demo
    t = Q_mu - s * P_mu @ R.T

    return s * P_copy @ R.T + t, Q_copy


if __name__ == '__main__':
    # Must pass "process=False" "maintain_order=True" if using trimesh
    # See: https://github.com/mikedh/trimesh/issues/147
    template: trimesh.Trimesh = trimesh.load('data/mastermodel_3d.obj',
                                             process=False,
                                             maintain_order=True)
    scan: trimesh.Trimesh = trimesh.load('data/scan.ply',
                                         process=False,
                                         maintain_order=True)

    known_correspondences = np.array([[10274, 974177], [10304, 1254944],
                                      [12331, 965655], [18558, 742479],
                                      [123, 968848], [256, 920117],
                                      [4778, 1076107], [15193, 644677],
                                      [13022, 908601], [20009, 235969],
                                      [9692, 362182], [133, 867122]])

    VX, VY = match_correspondence(template.vertices,
                                  scan.vertices,
                                  matched=known_correspondences)

    trimesh.Trimesh(VX, template.faces).export('template_corase_match.obj')
    trimesh.Trimesh(VY, scan.faces).export('scan_corase_match.obj')

    ps.init()
    ps.register_surface_mesh("template", VX, template.faces)
    ps.register_surface_mesh("scan", VY, scan.faces)
    ps.register_point_cloud("template kpt", VX[known_correspondences[:, 0]])
    ps.register_point_cloud("scan kpt", VY[known_correspondences[:, 1]])
    ps.show()
