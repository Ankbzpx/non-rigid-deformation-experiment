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

    # adj_lists = igl.adjacency_list(F)
    # weight_lists = []
    # for i in range(NV):
    #     weight_list = [L[i, adj_idx] for adj_idx in adj_lists[i]]
    #     weight_lists.append(weight_list)

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

    # to torch
    verts = torch.from_numpy(V).float().cuda()
    faces = torch.from_numpy(F).long().cuda()
    face_adjacency = torch.from_numpy(face_adjacency).long().cuda()
    edges = torch.from_numpy(edges).long().cuda()
    edge_weights = torch.from_numpy(edge_weights).float().cuda()

    handle_indices = torch.from_numpy(handle_indices).long().cuda()
    weights = sparse_np_to_torch(weights).cuda()

    laplacian_cot = sparse_np_to_torch(laplacian_cot).cuda()
    laplacian_uniform = sparse_np_to_torch(laplacian_uniform).cuda()
    L = sparse_np_to_torch(L).cuda()

    lms_fid = torch.from_numpy(lms_fid).long().cuda()
    lms_uv = torch.from_numpy(lms_uv).float().cuda()
    scan_lms = torch.from_numpy(scan_lms).float().cuda()
    verts_scan = torch.from_numpy(scan.vertices).float().cuda()

    # adj_lists = [torch.tensor(adj_list).long().cuda() for adj_list in adj_lists]
    # weight_lists = [torch.tensor(weight_list).float().cuda() for weight_list in weight_lists]
    log_Rs = nn.Parameter(
        torch.zeros(3, device='cuda')[None,
                                      ...].repeat_interleave(verts.shape[0], 0))

    delta_t = nn.Parameter(
        torch.zeros(3, device='cuda')[None, ...].repeat_interleave(
            len(handle_indices), 0))

    optimizer = torch.optim.Adam([log_Rs, delta_t], lr=1e-3)
    verts_deformed = verts

    for _ in range(20):
        knn = knn_points(verts_deformed[None, ...], verts_scan[None, ...])
        dists = knn.dists
        idx = knn.idx

        mask_robust = dists < dists.mean()
        idx_robust = idx[mask_robust][None, ..., None]
        verts_scan_closest = knn_gather(verts_scan[None, ...],
                                        idx_robust).squeeze(-2).squeeze(0)

        for iter in range(100):
            optimizer.zero_grad()

            per_vertex_R = so3_exp_map(log_Rs)

            verts_transformed = weights @ delta_t + torch.einsum(
                'ni,nji->ni', verts, per_vertex_R)

            loss_closest = (
                (verts_scan_closest -
                 verts_transformed[mask_robust[0, :, 0]])**2).sum(1).mean()

            per_landmark_face_verts = verts_transformed[faces[lms_fid]]
            A = per_landmark_face_verts[:, 0, :]
            B = per_landmark_face_verts[:, 1, :]
            C = per_landmark_face_verts[:, 2, :]
            lms = C + (A - C) * lms_uv[:, 0][:, None] + (
                B - C) * lms_uv[:, 1][:, None]
            loss_lms = ((scan_lms - lms)**2).sum(1).mean()

            loss_laplace = laplacian_uniform.matmul(
                verts_transformed).abs().mean()

            per_face_verts = verts_transformed[faces]
            ba = per_face_verts[:, 0, :] - per_face_verts[:, 1, :]
            ca = per_face_verts[:, 0, :] - per_face_verts[:, 2, :]
            face_normals = torch.cross(ba, ca)
            face_normals = face_normals / (
                torch.linalg.norm(face_normals, dim=-1, keepdim=True) + 1e-6)
            normal_residual = 1 - (face_normals[face_adjacency[:, 0]] *
                                   face_normals[face_adjacency[:, 1]]).sum(-1)
            loss_normal = (normal_residual**2).mean()

            e_0 = edges[:, 0]
            e_1 = edges[:, 1]
            edge_vec = verts[e_0] - verts[e_1]
            edge_vec_transformed = verts_transformed[e_0] - verts_transformed[
                e_1]
            edge_R = 0.5 * (per_vertex_R[e_0] + per_vertex_R[e_1])
            deform_residual = (edge_vec_transformed -
                               torch.einsum('ni,nji->ni', edge_vec, edge_R))**2
            loss_arap = (edge_weights * deform_residual.sum(-1)).mean()
            loss = 0.5 * loss_closest + loss_lms + 1e2 * loss_arap

            print(f"Iteration {iter}, Loss {loss.item()}")

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            verts_deformed = weights @ delta_t.detach() + torch.einsum(
                'ni,nji->ni', verts, so3_exp_map(log_Rs.detach()))

    knn = knn_points(verts_deformed[None, ...], verts_scan[None, ...])
    dists = knn.dists
    idx = knn.idx
    ic(dists.median())
    ic(dists.mean())
    mask_robust = dists < 5e-4
    idx_robust = idx[mask_robust][None, ..., None]
    verts_scan_closest = knn_gather(verts_scan[None, ...],
                                    idx_robust).squeeze(-2).squeeze(0)

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
    ps.register_surface_mesh("Original", template.vertices, template.faces)
    ps.register_surface_mesh("Deformed", verts_deformed, template.faces)
    ps.register_surface_mesh("ARAP", verts_arap, template.faces)
    ps.register_surface_mesh("Scan", scan.vertices, scan.faces)
    ps.show()

    template.vertices = verts_deformed
    write_obj('results/nicp.obj', template)
    template.vertices = verts_arap
    write_obj('results/arap.obj', template)
