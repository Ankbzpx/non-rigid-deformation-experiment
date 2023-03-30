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
import igl


# https://github.com/nmwsharp/diffusion-net/blob/master/src/diffusion_net/utils.py#L50
def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices),
                                    torch.FloatTensor(values),
                                    torch.Size(shape)).coalesce()


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x],
                               grad_outputs=grad_outputs,
                               create_graph=True)[0]
    return grad


def jacobian(output, input):
    elements = output.split(1, -1)
    return torch.stack([gradient(ele, input) for ele in elements], -2)


if __name__ == '__main__':
    # Must pass "process=False" "maintain_order=True" if using trimesh
    # See: https://github.com/mikedh/trimesh/issues/147
    template: trimesh.Trimesh = trimesh.load('template_corase_match.obj',
                                             process=False,
                                             maintain_order=True)
    scan: trimesh.Trimesh = trimesh.load('scan_corase_match_simplified.obj',
                                         process=False,
                                         maintain_order=True)

    face_adjacency = torch.from_numpy(template.face_adjacency).long().cuda()

    verts_template = torch.from_numpy(template.vertices)[None,
                                                         ...].float().cuda()
    faces_template = torch.from_numpy(template.faces).long().cuda()

    # V = np.array(template.vertices)
    # F = np.array(template.faces)
    # L, M = robust_laplacian.mesh_laplacian(V, F)
    # M_inv = scipy.sparse.diags(1 / M.diagonal())
    # delta_L = M_inv @ L
    # delta_L = sparse_np_to_torch(delta_L).to_sparse_csr().cuda()

    delta_L = Meshes(
        verts=verts_template,
        faces=faces_template[None, ...]).laplacian_packed().to_sparse_csr()

    verts_scan = torch.from_numpy(scan.vertices)[None, ...].float().cuda()
    faces_scan = torch.from_numpy(scan.faces)[None, ...].long().cuda()

    As = nn.Parameter(
        torch.eye(3,
                  device='cuda')[None, None,
                                 ...].repeat_interleave(verts_template.shape[1],
                                                        1))
    ts = nn.Parameter(
        torch.zeros(3, device='cuda')[None, None, ...].repeat_interleave(
            verts_template.shape[1], 1))

    optimizer = torch.optim.Adam([As, ts], lr=1e-3)

    for _ in range(1):
        verts_template = verts_template.requires_grad_(True)

        idx = knn_points(verts_template, verts_scan).idx
        verts_scan_closest = knn_gather(verts_scan, idx).squeeze(-2)

        for i in range(100):
            optimizer.zero_grad()

            verts_template_transformed = torch.einsum('bni,bnji->bni',
                                                      verts_template, As) + ts

            per_face_verts = verts_template_transformed[0][faces_template]
            ba = per_face_verts[:, 0, :] - per_face_verts[:, 1, :]
            ca = per_face_verts[:, 0, :] - per_face_verts[:, 2, :]
            face_normals = torch.cross(ba, ca)
            face_normals = face_normals / torch.linalg.norm(
                face_normals, dim=1, keepdim=True)
            normal_residual = 1 - (face_normals[face_adjacency[:, 0]] *
                                   face_normals[face_adjacency[:, 1]]).sum(-1)
            loss_normal = (normal_residual**2).mean()

            J = jacobian(verts_template_transformed, verts_template)
            J_tr = torch.linalg.matrix_norm(J)**2
            J_det = torch.linalg.det(J)
            #https://app.box.com/s/h6650u6vnxf581hl2rodr3enzf7silex
            loss_amips = (J_tr / torch.pow(J_det, 2 / 3)).mean()

            loss_laplace = delta_L.matmul(
                verts_template_transformed).abs().mean()

            loss_l2 = ((verts_template_transformed -
                        verts_scan_closest)**2).sum(-1).mean()

            loss_chamfer, _ = chamfer_distance(verts_template_transformed,
                                               verts_scan)

            loss_deform = ((verts_template_transformed -
                            verts_template)**2).sum(-1).mean(-1)

            loss = loss_l2 + 0.25 * loss_deform + loss_amips
            # + 1e-2 * loss_normal + 1e-2 * loss_laplace

            print(f"Iteration {i}, Loss {loss.item()}")

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            verts_template = torch.einsum('bni,bnji->bni', verts_template,
                                          As) + ts

        ps.init()
        ps.register_surface_mesh("Before", template.vertices, template.faces)
        ps.register_surface_mesh("Refined",
                                 verts_template.detach().cpu().numpy()[0],
                                 template.faces)
        ps.register_surface_mesh("Scan", scan.vertices, scan.faces)
        ps.show()
