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

    verts_template = torch.from_numpy(template.vertices)[None,
                                                         ...].float().cuda()
    faces_template = torch.from_numpy(template.faces)[None, ...].long().cuda()

    # V = np.array(template.vertices)
    # F = np.array(template.faces)
    # L, M = robust_laplacian.mesh_laplacian(V, F)
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

    verts_template = verts_template.requires_grad_(True)

    for _ in range(1):
        idx = knn_points(verts_template, verts_scan).idx
        verts_scan_closest = knn_gather(verts_scan, idx).squeeze(-2)

        for i in range(50):
            optimizer.zero_grad()

            verts_template_transformed = torch.einsum(
                'bni,nji->bni', verts_template, so3_exp_map(log_Rs)) + ts

            J = jacobian(verts_template_transformed, verts_template)

            J_tr = torch.linalg.matrix_norm(J)**2
            J_det = torch.linalg.det(J)

            #https://app.box.com/s/h6650u6vnxf581hl2rodr3enzf7silex
            loss_amips = torch.exp(0.125 * (J_tr - 1) + 0.5 *
                                   (J_det + 1 / J_det)).mean()

            residual = ((verts_template_transformed -
                         verts_scan_closest)**2).sum(-1)

            loss_laplace = delta_L.matmul(
                verts_template_transformed).abs().mean()

            loss_l2 = residual.mean()

            loss_chamfer, _ = chamfer_distance(verts_template_transformed,
                                               verts_scan)

            loss = loss_amips

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
