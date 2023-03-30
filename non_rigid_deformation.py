import torch
import torch.nn as nn
import numpy as np
import pytorch3d.io
import trimesh
import scipy, math
import pickle
from tqdm import tqdm
from tqdm import trange
import pytorch3d.structures as structures
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points, knn_gather, laplacian
from icecream import ic
import polyscope as ps
import robust_laplacian
import scipy.sparse
from pytorch3d.structures import Meshes
from pytorch3d.transforms.so3 import so3_exp_map


def find_icp_verts_corr(vmap_tar,
                        nmap_tar,
                        vert_src,
                        icp_near_size=32,
                        icp_theta_thresh=np.pi / 6,
                        icp_dist_thresh=0.05):    # 基于最近邻KD树的关键点寻找
    src_normals = trimesh.Trimesh(vertices=vert_src,
                                  faces=source_model.faces,
                                  process=False).vertex_normals
    kdtree = scipy.spatial.cKDTree(vmap_tar)    # 用于快速最近邻查找的KD-树
    dists, indices = kdtree.query(vert_src, k=icp_near_size)
    tar_normals = nmap_tar[indices.reshape(-1)].reshape(-1, icp_near_size, 3)

    cosine = np.einsum('ijk,ik->ij', tar_normals, src_normals)
    valid = (dists < icp_dist_thresh) & (cosine > math.cos(icp_theta_thresh))

    valid_indices = np.argmax(valid, axis=1)
    indices_corr = indices[np.arange(valid.shape[0]), valid_indices]
    tar_verts = vmap_tar[indices_corr]

    return tar_verts


def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices),
                                    torch.FloatTensor(values),
                                    torch.Size(shape)).coalesce()


if __name__ == '__main__':
    # Must pass "process=False" "maintain_order=True" if using trimesh
    # See: https://github.com/mikedh/trimesh/issues/147
    template: trimesh.Trimesh = trimesh.load('template_corase_match.obj',
                                             process=False,
                                             maintain_order=True)
    scan: trimesh.Trimesh = trimesh.load('scan_corase_match_simplified.obj',
                                         process=False,
                                         maintain_order=True)

    verts_template = torch.from_numpy(template.vertices)[None,
                                                         ...].float().cuda()
    faces_template = torch.from_numpy(template.faces)[None, ...].long().cuda()

    # L, M = robust_laplacian.mesh_laplacian(np.array(template.vertices), np.array(template.faces))
    # M_inv = scipy.sparse.diags(1 / M.diagonal())
    # delta_L = M_inv @ L
    # delta_L = sparse_np_to_torch(delta_L).to_sparse_csr().cuda()

    delta_L = Meshes(verts=verts_template,
                     faces=faces_template).laplacian_packed().to_sparse_csr()

    verts_scan = torch.from_numpy(scan.vertices)[None, ...].float().cuda()
    faces_scan = torch.from_numpy(scan.faces)[None, ...].long().cuda()

    log_Rs = nn.Parameter(
        torch.zeros(3, device='cuda')[None, ...].repeat_interleave(
            verts_template.shape[1], 0))
    ts = nn.Parameter(
        torch.zeros(3, device='cuda')[None, None, ...].repeat_interleave(
            verts_template.shape[1], 1))

    optimizer = torch.optim.AdamW([log_Rs, ts], lr=1e-3)

    for _ in range(1):
        idx = knn_points(verts_template, verts_scan).idx
        verts_scan_closest = knn_gather(verts_scan, idx).squeeze(-2)

        for i in range(50):
            optimizer.zero_grad()

            verts_template_transformed = torch.einsum(
                'bni,nji->bni', verts_template, so3_exp_map(log_Rs)) + ts

            residual = ((verts_template_transformed -
                         verts_scan_closest)**2).sum(-1)

            loss_laplace = delta_L.matmul(
                verts_template_transformed).abs().mean()

            loss_l2 = residual.mean()

            loss_chamfer, _ = chamfer_distance(verts_template_transformed,
                                               verts_scan)

            loss = loss_chamfer + loss_laplace

            print(f"Iteration {i}, Loss {loss.item()}")

            loss.backward()
            optimizer.step()

        verts_template = torch.einsum('bni,nji->bni', verts_template,
                                      so3_exp_map(
                                          log_Rs.detach())) + ts.detach()

        ps.init()
        ps.register_surface_mesh("Before", template.vertices, template.faces)
        ps.register_surface_mesh("Refined",
                                 verts_template.detach().cpu().numpy()[0],
                                 template.faces)
        ps.register_surface_mesh("Scan", scan.vertices, scan.faces)
        ps.show()
