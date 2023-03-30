import numpy as np
import trimesh
import polyscope as ps


def match_correspondence(P, Q, matches):
    P_copy = np.copy(P)

    P = P[matches[:, 0]]
    Q = Q[matches[:, 0]]

    P_mu = np.average(P, 0, keepdims=True)
    P_bar = P - P_mu
    Q = scan.vertices[matches[:, 1]]
    Q_mu = np.average(Q, 0, keepdims=True)
    Q_bar = Q - Q_mu

    cov = Q_bar.T @ P_bar / len(matches)
    U, S, V_T = np.linalg.svd(cov)
    R = U @ V_T

    # Reflection
    E = np.ones(len(cov))
    E[-1] = np.sign(np.linalg.det(R))
    R = R * E

    demo = np.sum(P_bar * P_bar) / len(matches)

    s = np.sum(S) / demo
    t = Q_mu - s * P_mu @ R.T

    return s * P_copy @ R.T + t


if __name__ == '__main__':
    # Must pass "process=False" "maintain_order=True" if using trimesh
    # See: https://github.com/mikedh/trimesh/issues/147
    template: trimesh.Trimesh = trimesh.load('data/mastermodel_3d.obj',
                                             process=False,
                                             maintain_order=True)
    scan: trimesh.Trimesh = trimesh.load('data/scan.ply',
                                         process=False,
                                         maintain_order=True)

    # hand pick vertices (double click)
    # ps.init()
    # ps.register_point_cloud("template", template.vertices)
    # ps.show()

    # ps.init()
    # ps.register_point_cloud("scan", scan.vertices)
    # ps.show()

    # exit()

    # TODO: Auto detect or finetone the matches
    matches = np.array([[10274, 974177], [10304, 1254944], [12331, 965655],
                        [18558, 742479], [123, 968848], [256, 920117],
                        [4778, 1076107], [15193, 644677], [13022, 908601],
                        [20009, 235969], [9692, 362182], [133, 867122],
                        [18463, 303717], [8155, 298427], [15065, 616624],
                        [4749, 1095784]])

    VX = match_correspondence(template.vertices, scan.vertices, matches=matches)

    trimesh.Trimesh(VX, template.faces).export('template_corase_match.obj')

    ps.init()
    ps.register_surface_mesh("template", VX, template.faces)
    ps.register_surface_mesh("scan", scan.vertices, scan.faces)
    ps.register_point_cloud("template kpt", VX[matches[:, 0]])
    ps.register_point_cloud("scan kpt", scan.vertices[matches[:, 1]])
    ps.show()
