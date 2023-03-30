import numpy as np
from icecream import ic
import trimesh
import polyscope as ps
import closest_neighbour

# Depends on GPU memory
kMaxQuerySize = 2e9


# Approximated closest neighbour (Exact one on subset)
# TODO: Use point to plane distance
def closest_neighbour_approx(P, Q):
    Q_max_query_size = int(kMaxQuerySize / len(P))

    if len(Q) < Q_max_query_size:
        _, closest_Q_indices, _, _ = closest_neighbour.compute(P, Q)
        return closest_Q_indices
    else:
        query_index = np.random.choice(np.arange(len(Q)), Q_max_query_size,
                                       False)
        Q_query = Q[query_index]
        _, closest_Q_indices, _, _ = closest_neighbour.compute(P, Q_query)
        return query_index[closest_Q_indices]


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


# Reference: https://web.stanford.edu/class/cs273/refs/umeyama.pdf
def icp(P, Q, max_iters=20, reweight=False):
    assert np.linalg.matrix_rank(P) >= P.shape[1] - 1
    assert np.linalg.matrix_rank(Q) >= Q.shape[1] - 1

    reweight_threshold = 5
    threshold_weight = np.power(reweight_threshold, -0.5) if reweight else 1.
    N = len(P)
    weight = threshold_weight * np.ones(N)
    avg_residual = np.inf

    for i in range(max_iters):
        weight_sum = np.sum(weight)

        P_mu = np.average(P, 0, keepdims=True)
        Q_mu = np.average(Q, 0, keepdims=True)

        P_bar = (P - P_mu)
        Q_bar = (Q - Q_mu)

        closest_Q_indices = closest_neighbour_approx(P_bar, Q_bar)
        Q_bar_weighted = Q_bar[closest_Q_indices] * weight[..., None]
        P_bar_weighted = P_bar * weight[..., None]
        cov = Q_bar_weighted.T @ P_bar_weighted / N / weight_sum

        U, S, V_T = np.linalg.svd(cov)
        R = U @ V_T

        # Reflection
        E = np.ones(len(cov))
        E[-1] = np.sign(np.linalg.det(R))
        R = R * E

        demo = np.sum(P_bar_weighted * P_bar_weighted) / N / weight_sum

        s = np.sum(S) / demo
        t = Q_mu - s * P_mu @ R.T

        P = s * P @ R.T + t

        residual = ((P - Q[closest_Q_indices])**2).sum(1)

        # Reweight (Huber loss)
        if reweight:
            weight = np.where(residual < reweight_threshold,
                              np.power(residual, -0.5), threshold_weight)

        avg_residual = np.average(residual)
        print(f"Step {i}, avg_residual {avg_residual}")

        if avg_residual < 1e-6:
            break

    return P, Q


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
