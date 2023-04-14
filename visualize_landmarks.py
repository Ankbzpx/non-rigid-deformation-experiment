import numpy as np
import polyscope as ps
import json
from icecream import ic
import trimesh
import torch
from pytorch3d import _C

from mesh_helper import read_obj, load_face_landmarks
from non_rigid_deformation import closest_point_on_triangle

if __name__ == '__main__':
    template = read_obj('results/template_icp_match.obj')
    template_lms = load_face_landmarks(template,
                                       'results/template_icp_match_lms.txt')

    scan: trimesh.Trimesh = trimesh.load('data/scan.ply',
                                         process=False,
                                         maintain_order=True)
    scan_lms_data = json.load(open('data/scan_3d.txt'))
    scan_lms = np.stack(
        [np.array([lm['x'], lm['y'], lm['z']]) for lm in scan_lms_data])

    scan_lms_proj = closest_point_on_triangle(
        torch.from_numpy(scan_lms).float().cuda(),
        torch.from_numpy(scan.vertices[scan.faces]).float().cuda(),
        torch.from_numpy(np.copy(
            scan.face_normals)).float().cuda())[0].detach().cpu().numpy()

    ps.init()
    ps.register_surface_mesh('template',
                             template.vertices,
                             template.faces,
                             enabled=False)
    ps.register_point_cloud('template_lms',
                            template_lms,
                            radius=2e-3,
                            enabled=False)
    ps.register_surface_mesh('scan', scan.vertices, scan.faces)
    ps.register_point_cloud('scan_lms', scan_lms, radius=2e-3)
    ps.register_point_cloud('scan_lms_proj', scan_lms_proj, radius=2e-3)
    ps.show()
