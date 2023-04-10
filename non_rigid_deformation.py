import torch
import torch.nn as nn
import numpy as np
import trimesh
from mesh_helper import read_obj, write_obj
import polyscope as ps
import json
import igl
from icecream import ic
import scipy
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.transforms.so3 import so3_exp_map
from VectorAdam.vectoradam import VectorAdam
from torch.optim import Adam


# https://github.com/nmwsharp/diffusion-net/blob/master/src/diffusion_net/utils.py#L50
def sparse_np_to_torch(A) -> torch.Tensor:
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices),
                                    torch.FloatTensor(values),
                                    torch.Size(shape)).coalesce()


def gradient(y, x, grad_outputs=None) -> list[torch.Tensor]:
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x],
                               grad_outputs=grad_outputs,
                               create_graph=True)[0]
    return grad


def jacobian(output, input) -> torch.Tensor:
    elements = output.split(1, -1)
    return torch.stack([gradient(ele, input) for ele in elements], -2)


def compute_edge_vec(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    per_face_verts = verts[faces]
    v1 = per_face_verts[:, 0]
    v2 = per_face_verts[:, 1]
    v3 = per_face_verts[:, 2]

    e1 = v2 - v1
    e2 = v3 - v1

    # TODO: use QR for efficient evaluation
    # edge_vec = np.stack([e1, e2], -1)
    # ic(np.linalg.qr(edge_vec[0]))

    FN = torch.cross(e1, e2, axis=-1)
    v4 = v1 + FN / torch.sqrt(torch.linalg.norm(FN, axis=-1, keepdims=True))
    e3 = v4 - v1

    return torch.stack([e1, e2, e3], -1)


if __name__ == '__main__':
    template = read_obj('results/template_icp_match.obj')
    lms_data = np.array(json.load(open('results/template_icp_match_lms.txt')))
    lms_fid = np.int64(lms_data[:, 0])
    lms_uv = np.float64(lms_data[:, 1:])

    scan: trimesh.Trimesh = trimesh.load('data/scan.ply',
                                         process=False,
                                         maintain_order=True)
    scan_lms_data = json.load(open('data/scan_3d.txt'))
    scan_lms = np.stack(
        [np.array([lm['x'], lm['y'], lm['z']]) for lm in scan_lms_data])

    V = template.vertices
    F = template.faces

    NV = len(V)
    NF = len(F)

    # Cotangent laplacian
    L = igl.cotmatrix(V, F)
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    M_inv = scipy.sparse.diags(1 / M.diagonal())
    laplacian_cot = M_inv @ L

    # Uniform laplacian
    A = igl.adjacency_matrix(F)
    A_sum = np.sum(A, axis=1)
    A_diag = scipy.sparse.spdiags(A_sum.squeeze(-1), 0, NV, NV)
    laplacian_uniform = A - A_diag

    # edge info
    L_coo = scipy.sparse.coo_array(L)
    edges = np.stack([L_coo.row, L_coo.col], -1)
    edge_weights = L_coo.data

    # Annoyingly lms are NOT vetices, so use incident triangle vertices instead
    lm_tri_verts_indice = np.unique(F[lms_fid])

    # If load using trimesh, the internal processing would cause landmark mismatch
    template_tri = trimesh.Trimesh(V, F)
    face_adjacency = np.copy(template_tri.face_adjacency)

    handle_indices = np.load('results/bi.npy')
    weights = np.load('results/bbw.npy')
    weights[weights < 1e-6] = 0
    # Partition of unity
    weights = weights / weights.sum(1, keepdims=True)
    weights = scipy.sparse.csr_matrix(weights)
    exclude_indices = np.load('results/ei.npy')

    Ti = np.arange(NF, dtype=np.int64)
    diag_pairs = np.stack([Ti, Ti], -1)

    # quad adjacency constraint
    # face_adj_quad_pairs = np.vstack([np.stack([Ti[::2], Ti[1::2]], -1), np.stack([Ti[1::2], Ti[::2]], -1)])
    # face_adj_quad_weight = np.ones(len(face_adj_quad_pairs))

    # face_adj_quad_pairs = np.vstack([face_adj_quad_pairs, diag_pairs])
    # face_adj_weight = np.concatenate([face_adj_quad_weight, -np.ones(NF)])

    # F_adj_quad = scipy.sparse.coo_array(
    #     (face_adj_weight, (face_adj_quad_pairs[:, 0], face_adj_quad_pairs[:, 1])),
    #     shape=(NF, NF)).tocsc()

    # TODO: use quad face adjacency
    # triangle-triangle adjacency matrix
    TT = igl.triangle_triangle_adjacency(F)[0]

    face_adj_pairs = np.vstack([
        np.stack([Ti, TT[:, 0]], -1),
        np.stack([Ti, TT[:, 1]], -1),
        np.stack([Ti, TT[:, 2]], -1)
    ])
    boundary_indices = np.sum(face_adj_pairs == -1, 1) != 0
    boundary_face_indices = np.unique(face_adj_pairs[:, 0][boundary_indices])

    diag_weight = -np.ones(NF) * 3
    diag_weight[boundary_face_indices] = -2

    face_adj_pairs = face_adj_pairs[np.logical_not(boundary_indices)]
    face_adj_weight = np.ones(len(face_adj_pairs))

    face_adj_pairs = np.vstack([face_adj_pairs, diag_pairs])
    face_adj_weight = np.concatenate([face_adj_weight, diag_weight])

    F_adj = scipy.sparse.coo_array(
        (face_adj_weight, (face_adj_pairs[:, 0], face_adj_pairs[:, 1])),
        shape=(NF, NF)).tocsc()

    # to torch
    valid_mask = torch.ones(len(V))
    valid_mask[exclude_indices] = 0
    valid_mask = valid_mask.cuda().float()

    verts = torch.from_numpy(V).float().cuda()
    faces = torch.from_numpy(F).long().cuda()
    face_adjacency = torch.from_numpy(face_adjacency).long().cuda()
    edges = torch.from_numpy(edges).long().cuda()
    edge_weights = torch.from_numpy(edge_weights).float().cuda()

    W = compute_edge_vec(verts, faces)
    W_inv = torch.linalg.inv(W)

    handle_indices = torch.from_numpy(handle_indices).long().cuda()
    weights = sparse_np_to_torch(weights).cuda()

    laplacian_cot = sparse_np_to_torch(laplacian_cot).cuda()
    laplacian_uniform = sparse_np_to_torch(laplacian_uniform).cuda()
    L = sparse_np_to_torch(L).cuda()
    F_adj = sparse_np_to_torch(F_adj).to_sparse_csr().cuda()
    # F_adj_quad = sparse_np_to_torch(F_adj_quad).to_sparse_csr().cuda()

    lms_fid = torch.from_numpy(lms_fid).long().cuda()
    lms_uv = torch.from_numpy(lms_uv).float().cuda()
    scan_lms = torch.from_numpy(scan_lms).float().cuda()
    verts_scan = torch.from_numpy(scan.vertices).float().cuda()
    vertex_normals_scan = torch.from_numpy(np.copy(
        scan.vertex_normals)).float().cuda()

    def closest_neighbour(points, thr=5e-4):
        candidates_idx = knn_points(points[None, ...],
                                    verts_scan[None, ...],
                                    K=32).idx
        closest_candidates = knn_gather(verts_scan[None, ...], candidates_idx)
        closest_candidate_normals = knn_gather(vertex_normals_scan[None, ...],
                                               candidates_idx)

        dists = torch.einsum('bncd,bncd->bnc',
                             points[None, :, None, :] - closest_candidates,
                             closest_candidate_normals).abs()
        closest_dist, closest_idx = torch.min(dists, dim=-1, keepdim=True)
        closest_idx = torch.gather(candidates_idx, -1, closest_idx)

        mask_robust = closest_dist < thr
        idx_robust = closest_idx[mask_robust][None, ..., None]
        verts_scan_closest = knn_gather(verts_scan[None, ...],
                                        idx_robust).squeeze(-2).squeeze(0)

        return mask_robust, verts_scan_closest

    # log_Rs = nn.Parameter(
    #     torch.zeros(3, device='cuda')[None,
    #                                   ...].repeat_interleave(verts.shape[0], 0))

    delta_t = nn.Parameter(
        torch.zeros(3, device='cuda')[None, ...].repeat_interleave(
            len(handle_indices), 0))

    # optimizer = torch.optim.Adam([log_Rs, delta_t], lr=1e-4)
    optimizer = VectorAdam([delta_t], lr=1e-3)

    verts_deformed = verts
    weight_close = 0.5
    weight_lms = 1.0
    weight_amips = 1.0
    weight_smooth = 1e-3

    for _ in range(1):
        mask_robust, verts_scan_closest = closest_neighbour(
            verts_deformed, 5e-4)

        for iter in range(100):
            optimizer.zero_grad()

            # per_vertex_R = so3_exp_map(log_Rs)

            # verts_deformed = weights @ delta_t + torch.einsum(
            #     'ni,nji->ni', verts, per_vertex_R)

            verts_deformed = weights @ delta_t + verts

            loss_closest = (
                (verts_scan_closest -
                 verts_deformed[mask_robust[0, :, 0]])**2).sum(1).mean()

            per_landmark_face_verts = verts_deformed[faces[lms_fid]]
            A = per_landmark_face_verts[:, 0, :]
            B = per_landmark_face_verts[:, 1, :]
            C = per_landmark_face_verts[:, 2, :]
            lms = C + (A - C) * lms_uv[:, 0][:, None] + (
                B - C) * lms_uv[:, 1][:, None]
            loss_lms = ((scan_lms - lms)**2).sum(1).mean()

            loss_laplace = laplacian_cot.matmul(verts_deformed).abs().mean()

            # per_face_verts = verts_deformed[faces]
            # ba = per_face_verts[:, 0, :] - per_face_verts[:, 1, :]
            # ca = per_face_verts[:, 0, :] - per_face_verts[:, 2, :]
            # face_normals = torch.cross(ba, ca)
            # face_normals = face_normals / (
            #     torch.linalg.norm(face_normals, dim=-1, keepdim=True) + 1e-6)
            # normal_residual = 1 - (face_normals[face_adjacency[:, 0]] *
            #                        face_normals[face_adjacency[:, 1]]).sum(-1)
            # loss_normal = (normal_residual**2).mean()

            # e_0 = edges[:, 0]
            # e_1 = edges[:, 1]
            # edge_vec = verts[e_0] - verts[e_1]
            # edge_vec_deformed = verts_deformed[e_0] - verts_deformed[
            #     e_1]
            # edge_R = per_vertex_R[e_0]
            # deform_residual = (edge_vec_deformed -
            #                    torch.einsum('ni,nji->ni', edge_vec, edge_R))**2
            # loss_arap = (edge_weights * deform_residual.sum(-1)).mean()

            # A = verts_deformed[faces[:, 0]]
            # B = verts_deformed[faces[:, 1]]
            # C = verts_deformed[faces[:, 2]]
            # e1 = B - A
            # e2 = C - A
            # e1_uv = torch.stack([(e1 * F1).sum(-1), (e1 * F2).sum(-1)], -1)
            # e2_uv = torch.stack([(e2 * F1).sum(-1), (e2 * F2).sum(-1)], -1)
            # E = torch.stack([e1_uv, e2_uv], -1)

            W_deformed = compute_edge_vec(verts_deformed, faces)
            J = torch.bmm(W_deformed, W_inv)
            J_tr = torch.linalg.matrix_norm(J)**2
            J_det = torch.linalg.det(J)
            # https://app.box.com/s/h6650u6vnxf581hl2rodr3enzf7silex
            pos_det = (J_det >= 0.0).float()
            loss_amips = ((J_tr * (1e-6 + torch.pow(J_det, -2 / 3)) - 3) *
                          pos_det).sum() / pos_det.sum()

            # one ring smooth of deformation gradient
            loss_smooth = (F_adj @ J.reshape(-1, 9)).sum(-1).abs().mean()

            loss = weight_close * loss_closest + weight_lms * loss_lms + weight_amips * loss_amips + weight_smooth * loss_smooth

            print(f"Iteration {iter}, Loss {loss.item()}")

            loss.backward()
            # log_Rs.grad *= valid_mask[:, None]
            optimizer.step()

        with torch.no_grad():
            # verts_deformed = weights @ delta_t.detach() + torch.einsum(
            #     'ni,nji->ni', verts, so3_exp_map(log_Rs.detach()))
            verts_deformed = weights @ delta_t.detach() + verts

    mask_robust, verts_scan_closest = closest_neighbour(verts_deformed)
    verts_deformed = verts_deformed.detach().cpu().numpy()

    b = np.where(mask_robust[0, :, 0].detach().cpu().numpy())[0]
    bc = verts_scan_closest.detach().cpu().numpy()

    # b = lm_tri_verts_indice
    # bc = verts_deformed[lm_tri_verts_indice]

    b = np.concatenate([exclude_indices, b])
    bc = np.concatenate([verts_deformed[exclude_indices], bc])

    b, b_idx = np.unique(b, return_index=True)
    bc = bc[b_idx]

    arap = igl.ARAP(V, F, 3, b)
    verts_arap = arap.solve(bc, verts_deformed)

    ps.init()
    ps.register_surface_mesh("Original",
                             template.vertices,
                             template.faces,
                             enabled=False)
    ps.register_surface_mesh("Deformed", verts_deformed, template.faces)
    ps.register_surface_mesh("ARAP", verts_arap, template.faces, enabled=False)
    # ps.register_surface_mesh("Scan", scan.vertices, scan.faces)
    ps.show()

    template.vertices = verts_deformed
    write_obj('results/nicp.obj', template)
    # template.vertices = verts_arap
    # write_obj('results/arap.obj', template)
