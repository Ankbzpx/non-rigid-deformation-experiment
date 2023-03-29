import numpy as np
from icecream import ic
import igl
import polyscope as ps
import closest_neighbour

kMaxQuerySize = 2e9


# Exact closest neighbour but limited by cuda memory
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


# Reference: https://web.stanford.edu/class/cs273/refs/umeyama.pdf
def icp(P,
        Q,
        matched=None,
        matched_weight_scale=2,
        max_iters=10,
        reweight=False):
    assert np.linalg.matrix_rank(P) >= P.shape[1] - 1
    assert np.linalg.matrix_rank(Q) >= Q.shape[1] - 1

    reweight_threshold = 5
    threshold_weight = np.power(reweight_threshold, -0.5) if reweight else 1.
    N = len(P)
    weight = threshold_weight * np.ones(N)
    avg_residual = np.inf

    if matched is not None:
        N_matched = len(matched)
        weight_matched = matched_weight_scale * \
            threshold_weight * np.ones(N_matched)

    for i in range(max_iters):
        weight_sum = np.sum(weight)

        Q_mu = np.average(Q, 0, keepdims=True)
        P_mu = np.average(P, 0, keepdims=True)

        Q_bar = (Q - Q_mu)
        P_bar = (P - P_mu)

        closest_Q_indices = closest_neighbour_approx(P_bar, Q_bar)
        Q_bar_weighted = Q_bar[closest_Q_indices] * weight[..., None]
        P_bar_weighted = P_bar * weight[..., None]
        cov = Q_bar_weighted.T @ P_bar_weighted / N / weight_sum

        if matched is not None:
            weight_sum_matched = np.sum(weight_matched)
            Q_matched = Q[matched[:, 0]]
            P_macthed = P[matched[:, 1]]
            Q_bar_matched = (Q_matched - Q_mu) * \
                weight_matched[..., None]
            P_bar_matched = (P_macthed - P_mu) * \
                weight_matched[..., None]
            cov_matched = Q_bar_matched.T @ P_bar_matched / N_matched / weight_sum_matched
            cov += cov_matched

        U, S, V_T = np.linalg.svd(cov)
        R = U @ V_T

        # reflection
        E = np.ones(len(cov))
        E[-1] = np.sign(np.linalg.det(R))
        R = R * E

        demo = np.sum(P_bar_weighted * P_bar_weighted) / N / weight_sum

        if matched is not None:
            demo += np.sum(P_bar_matched * P_bar_matched) / \
                N_matched / weight_sum_matched

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
    VX, FX = igl.read_triangle_mesh('data/mastermodel_3d.obj')
    VY, FY = igl.read_triangle_mesh('data/scan.ply')

    # known_correspondences = np.array([[0, 0], [100, 100], [200, 200]])
    VX, VY = icp(VX, VY)

    ps.init()
    ps.register_surface_mesh("template", VX, FX)
    ps.register_surface_mesh("target", VY, FY)
    ps.show()
