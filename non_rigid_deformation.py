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
# from pytorch3d.transforms.so3 import so3_exp_map
from VectorAdam.vectoradam import VectorAdam
# from torch.optim import Adam
from pytorch3d import _C
from functools import partial


# https://github.com/nmwsharp/diffusion-net/blob/master/src/diffusion_net/utils.py#L50
def sparse_np_to_torch(A) -> torch.Tensor:
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices),
                                    torch.FloatTensor(values),
                                    torch.Size(shape)).coalesce()


def compute_deformation(verts: torch.Tensor,
                        faces: torch.Tensor) -> torch.Tensor:
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


def uniform_vert_normals(
        vertices: torch.Tensor, faces: torch.Tensor,
        vert_face_adjacency: list[torch.Tensor]) -> torch.Tensor:
    per_face_verts = vertices[faces]
    face_normals = torch.cross(per_face_verts[:, 1] - per_face_verts[:, 0],
                               per_face_verts[:, 2] - per_face_verts[:, 0],
                               dim=-1)
    face_normals = face_normals / torch.linalg.norm(
        face_normals, dim=-1, keepdims=True)
    return torch.stack(
        [face_normals[vf_idx].mean(dim=0) for vf_idx in vert_face_adjacency])


def closest_point_on_triangle(verts: torch.Tensor,
                              target_per_face_verts: torch.Tensor,
                              target_vertex_normals: torch.Tensor,
                              min_triangle_area=5e-3):
    max_points = len(verts)
    first_idx = torch.tensor([0], device=verts.device).long()
    dists, indices = _C.point_face_dist_forward(verts, first_idx,
                                                target_per_face_verts,
                                                first_idx, max_points,
                                                min_triangle_area)
    # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.4264&rep=rep1&type=pdf
    tris = target_per_face_verts[indices]
    tri_normals = target_vertex_normals[indices]
    pt_proj = verts - torch.einsum('ab,ab->a', verts - tris[:, 0, :],
                                   tri_normals)[..., None] * tri_normals

    return pt_proj, dists, indices


def closest_point_triangle_match(verts: torch.Tensor,
                                 faces: torch.Tensor,
                                 vert_face_adjacency: list[torch.Tensor],
                                 target_per_face_verts: torch.Tensor,
                                 target_vertex_normals: torch.Tensor,
                                 exclude_indices: torch.Tensor,
                                 dist_thr=5e-4,
                                 cos_thr=0.0) -> list[torch.Tensor]:
    pt_proj, dists, indices = closest_point_on_triangle(verts,
                                                        target_per_face_verts,
                                                        target_vertex_normals)
    vert_normals = uniform_vert_normals(verts, faces, vert_face_adjacency)
    cos = torch.einsum('ab,ab->a', target_vertex_normals[indices], vert_normals)
    valid_mask = torch.logical_and(dists < dist_thr, cos > cos_thr)
    valid_mask[exclude_indices] = False

    return valid_mask, pt_proj[valid_mask], dists[valid_mask], indices[valid_mask]


if __name__ == '__main__':
    template = read_obj('results/template_icp_match.obj')
    lms_data = np.array(json.load(open('results/template_icp_match_lms.txt')))
    lms_fid = np.int64(lms_data[:, 0])
    lms_uv = np.float64(lms_data[:, 1:])

    scan: trimesh.Trimesh = trimesh.load('data/scan_decimated.obj',
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

    f_ids = np.arange(NF, dtype=np.int64)
    diag_pairs = np.stack([f_ids, f_ids], -1)

    # quad adjacency constraint
    # face_adj_quad_pairs = np.vstack([np.stack([f_ids[::2], f_ids[1::2]], -1), np.stack([f_ids[1::2], f_ids[::2]], -1)])
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
        np.stack([f_ids, TT[:, 0]], -1),
        np.stack([f_ids, TT[:, 1]], -1),
        np.stack([f_ids, TT[:, 2]], -1)
    ])
    boundary_indices = np.sum(face_adj_pairs == -1, 1) != 0
    boundary_face_indices = np.unique(face_adj_pairs[:, 0][boundary_indices])

    diag_weight = -np.ones(NF) * 3
    diag_weight[boundary_face_indices] = -2

    face_adj_pairs = face_adj_pairs[np.logical_not(boundary_indices)]
    face_adj_weight = np.ones(len(face_adj_pairs))

    face_adj_pairs = np.vstack([face_adj_pairs, diag_pairs])
    face_adj_weight = np.concatenate([face_adj_weight, diag_weight])

    F_weight = scipy.sparse.spdiags(igl.doublearea(V, F), 0, NF, NF)
    F_adj = scipy.sparse.coo_array(
        (face_adj_weight, (face_adj_pairs[:, 0], face_adj_pairs[:, 1])),
        shape=(NF, NF)).tocsc()

    # to torch
    valid_mask = torch.ones(len(V))
    valid_mask[exclude_indices] = 0
    valid_mask = valid_mask.cuda().float()
    exclude_indices_torch = torch.from_numpy(exclude_indices).cuda().long()

    verts = torch.from_numpy(V).float().cuda()
    faces = torch.from_numpy(F).long().cuda()
    face_adjacency = torch.from_numpy(face_adjacency).long().cuda()
    edges = torch.from_numpy(edges).long().cuda()
    edge_weights = torch.from_numpy(edge_weights).float().cuda()
    VF, NI = igl.vertex_triangle_adjacency(F, NV)
    vert_face_adjacency = [
        torch.from_numpy(vf_indices).long().cuda()
        for vf_indices in np.split(VF, NI[1:-1])
    ]

    W = compute_deformation(verts, faces)
    W_inv = torch.linalg.inv(W)

    handle_indices = torch.from_numpy(handle_indices).long().cuda()
    weights = sparse_np_to_torch(weights).cuda()

    laplacian_cot = sparse_np_to_torch(laplacian_cot).cuda()
    laplacian_uniform = sparse_np_to_torch(laplacian_uniform).cuda()
    L = sparse_np_to_torch(L).cuda()
    F_adj = sparse_np_to_torch(F_adj).to_sparse_csr().cuda()
    F_weight = sparse_np_to_torch(F_weight).to_sparse_csr().cuda()
    # F_adj_quad = sparse_np_to_torch(F_adj_quad).to_sparse_csr().cuda()

    lms_fid = torch.from_numpy(lms_fid).long().cuda()
    lms_uv = torch.from_numpy(lms_uv).float().cuda()
    scan_lms = torch.from_numpy(scan_lms).float().cuda()
    per_face_verts_scan = torch.from_numpy(
        scan.vertices[scan.faces]).float().cuda()
    face_normals_scan = torch.from_numpy(np.copy(
        scan.face_normals)).float().cuda()
    J_I = torch.eye(3).float().cuda()

    # point_face_dist_forward

    # def uniform_vert_normals(vertices: torch.Tensor) -> torch.Tensor:
    #     per_face_verts = vertices[faces]
    #     face_normals = torch.cross(per_face_verts[:, 1] - per_face_verts[:, 0],
    #                                per_face_verts[:, 2] - per_face_verts[:, 0],
    #                                dim=-1)
    #     face_normals = face_normals / torch.linalg.norm(
    #         face_normals, dim=-1, keepdims=True)
    #     return torch.stack([
    #         face_normals[vf_idx].mean(dim=0) for vf_idx in vert_face_adjacency
    #     ])
    closest_match = partial(closest_point_triangle_match,
                            faces=faces,
                            vert_face_adjacency=vert_face_adjacency,
                            target_per_face_verts=per_face_verts_scan,
                            target_vertex_normals=face_normals_scan,
                            exclude_indices=exclude_indices_torch)

    # log_Rs = nn.Parameter(
    #     torch.zeros(3, device='cuda')[None,
    #                                   ...].repeat_interleave(verts.shape[0], 0))

    delta_t = nn.Parameter(
        torch.zeros(3, device='cuda')[None, ...].repeat_interleave(
            len(handle_indices), 0))

    # optimizer = torch.optim.Adam([log_Rs, delta_t], lr=1e-4)
    optimizer = VectorAdam([delta_t], lr=1e-3)

    verts_deformed = verts

    weight_close = 0
    weight_lms = 1.0
    weight_amips = 1e3
    weight_identity = 0.5
    weight_smooth = 1e1

    for iter in range(5):
        if iter == 0:
            weight_close = 0
        else:
            if weight_close == 0:
                weight_close = 5e-1
            else:
                weight_close *= 2
                weight_close = min(weight_close, 5e1)

        valid_mask, verts_scan_closest, _ = closest_match(verts_deformed,
                                                          cos_thr=0.5)

        for iter in range(100):
            optimizer.zero_grad()

            # per_vertex_R = so3_exp_map(log_Rs)

            # verts_deformed = weights @ delta_t + torch.einsum(
            #     'ni,nji->ni', verts, per_vertex_R)

            verts_deformed = weights @ delta_t + verts

            loss_closest = ((verts_scan_closest -
                             verts_deformed[valid_mask])**2).sum(1).mean()

            per_landmark_face_verts = verts_deformed[faces[lms_fid]]
            A = per_landmark_face_verts[:, 0, :]
            B = per_landmark_face_verts[:, 1, :]
            C = per_landmark_face_verts[:, 2, :]
            lms = C + (A - C) * lms_uv[:, 0][:, None] + (
                B - C) * lms_uv[:, 1][:, None]
            loss_lms = ((scan_lms - lms)**2).sum(1).mean()

            # loss_laplace = laplacian_cot.matmul(verts_deformed).abs().mean()

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

            W_deformed = compute_deformation(verts_deformed, faces)
            # deformation gradient
            J = torch.bmm(W_deformed, W_inv)
            J_tr = torch.linalg.matrix_norm(J)**2
            J_det = torch.linalg.det(J)
            # https://app.box.com/s/h6650u6vnxf581hl2rodr3enzf7silex
            pos_det = (J_det >= 0.0).float()
            loss_amips = (F_weight @ (J_tr /
                                      (1e-6 + torch.pow(J_det, 2 / 3)) - 3) *
                          pos_det).sum() / pos_det.sum()

            # one ring smooth of deformation gradient
            loss_smooth = (
                F_weight @ F_adj @ J.reshape(-1, 9)).sum(-1).abs().mean()

            # deformation gradient to be identity
            loss_identity = (F_weight @ (J - J_I[None, ...]).reshape(
                -1, 9)).sum(-1).abs().mean()

            loss = (weight_close * loss_closest) + (weight_lms * loss_lms) + (
                weight_amips * loss_amips) + (weight_identity * loss_identity
                                             ) + (weight_smooth * loss_smooth)

            print(f"Iteration {iter}, Loss {loss.item()}")

            loss.backward()
            # log_Rs.grad *= valid_mask[:, None]
            optimizer.step()

        with torch.no_grad():
            # verts_deformed = weights @ delta_t.detach() + torch.einsum(
            #     'ni,nji->ni', verts, so3_exp_map(log_Rs.detach()))
            verts_deformed = weights @ delta_t.detach() + verts

    valid_mask, verts_scan_closest = closest_match(verts_deformed,
                                                   dist_thr=5e-4,
                                                   cos_thr=0.9)
    V_deformed = verts_deformed.detach().cpu().numpy()

    B = np.where(valid_mask.detach().cpu().numpy())[0]
    BC = verts_scan_closest.detach().cpu().numpy()

    B = np.concatenate([exclude_indices, B])
    BC = np.concatenate([V_deformed[exclude_indices], BC])

    B, b_idx = np.unique(B, return_index=True)
    BC = BC[b_idx]

    arap = igl.ARAP(V, F, 3, B)
    V_arap = arap.solve(BC, V_deformed)

    ps.init()
    ps.register_surface_mesh("Original",
                             template.vertices,
                             template.faces,
                             enabled=False)
    ps.register_surface_mesh("Deformed", V_deformed, template.faces)
    ps.register_surface_mesh("ARAP", V_arap, template.faces, enabled=False)
    ps.register_surface_mesh("Scan", scan.vertices, scan.faces, enabled=False)
    ps.register_point_cloud("Boundary", BC, enabled=False)
    ps.show()

    # template.vertices = V_deformed
    # write_obj('results/nicp.obj', template)
    # template.vertices = verts_arap
    # write_obj('results/arap.obj', template)
