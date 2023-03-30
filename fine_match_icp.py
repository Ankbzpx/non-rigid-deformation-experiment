import torch
from pytorch3d.ops import iterative_closest_point
import trimesh
import numpy as np

from icecream import ic
import polyscope as ps

if __name__ == '__main__':
    # Must pass "process=False" "maintain_order=True" if using trimesh
    # See: https://github.com/mikedh/trimesh/issues/147
    template: trimesh.Trimesh = trimesh.load('template_corase_match.obj',
                                             process=False,
                                             maintain_order=True)
    scan: trimesh.Trimesh = trimesh.load('scan_corase_match.obj',
                                         process=False,
                                         maintain_order=True)

    # b x n x 3
    V_source = torch.from_numpy(np.copy(template.vertices[None,
                                                          ...])).float().cuda()
    F_source = torch.from_numpy(np.copy(template.faces[None, ...])).cuda()

    # b x m x 3
    V_target = torch.from_numpy(np.copy(scan.vertices[None,
                                                      ...])).float().cuda()
    F_target = torch.from_numpy(np.copy(scan.faces[None, ...])).cuda()

    # seem to overfit as iteration increases
    T = iterative_closest_point(V_source,
                                V_target,
                                max_iterations=1,
                                verbose=True).RTs

    V_source = (T.s * torch.einsum('bnc,bci->bni', V_source, T.R) + T.T)

    template_matched = trimesh.Trimesh(V_source[0].cpu().numpy(),
                                       template.faces)
    template_matched.export('template_fine_match.obj')

    ps.init()
    ps.register_surface_mesh('template_fine_match', template_matched.vertices,
                             template_matched.faces)
    ps.register_surface_mesh('scan', scan.vertices, scan.faces)
    ps.show()
