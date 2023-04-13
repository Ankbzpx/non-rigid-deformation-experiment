import numpy as np
import polyscope as ps
from icecream import ic
import igl
import json

from mesh_helper import read_obj, write_obj
from arap import AsRigidAsPossible
import torch
from non_rigid_deformation import closest_point_triangle_match
import trimesh
from functools import partial
from tqdm import tqdm

if __name__ == '__main__':
    template = read_obj('results/template_icp_match.obj')
    lms_data = np.array(json.load(open('results/template_icp_match_lms.txt')))
    lms_fid = np.int64(lms_data[:, 0])
    lms_uv = np.float64(lms_data[:, 1:])

    lms_bary_coords = np.stack(
        [lms_uv[:, 0], lms_uv[:, 1], 1 - lms_uv[:, 0] - lms_uv[:, 1]], -1)

    V = template.vertices
    F = template.faces

    NV = len(V)
    NF = len(F)

    face_groups = np.split(template.faces, template.polygon_groups[1:])

    # 3 Tongue
    # 6 Ears
    # 10 Right Eye
    # 13 Left Eye
    # 27 head
    exclude_indices = np.unique(
        np.concatenate([
            face_groups[3].reshape(-1), face_groups[10].reshape(-1),
            face_groups[13].reshape(-1)
        ]))

    scan: trimesh.Trimesh = trimesh.load('data/scan_decimated.obj',
                                         process=False,
                                         maintain_order=True)
    scan_lms_data = json.load(open('data/scan_3d.txt'))
    scan_lms = np.stack(
        [np.array([lm['x'], lm['y'], lm['z']]) for lm in scan_lms_data])

    arap = AsRigidAsPossible(V,
                             F,
                             b_fid=lms_fid,
                             b_bary_coords=lms_bary_coords,
                             b_f_bounded=False)
    V_arap = arap.solve(scan_lms, V)

    # ps.init()
    # ps.register_surface_mesh("template", V, F)
    # ps.register_surface_mesh("template_arap", V_arap, F)
    # ps.show()

    # template.vertices = V_arap
    # write_obj("results/lm_arap.obj", template)

    faces = torch.from_numpy(F).long().cuda()
    VF, NI = igl.vertex_triangle_adjacency(F, NV)
    vert_face_adjacency = [
        torch.from_numpy(vf_indices).long().cuda()
        for vf_indices in np.split(VF, NI[1:-1])
    ]
    per_face_verts_scan = torch.from_numpy(
        scan.vertices[scan.faces]).float().cuda()
    face_normals_scan = torch.from_numpy(np.copy(
        scan.face_normals)).float().cuda()
    exclude_indices = torch.from_numpy(exclude_indices).cuda().long()
    closest_match = partial(closest_point_triangle_match,
                            faces=faces,
                            vert_face_adjacency=vert_face_adjacency,
                            target_per_face_verts=per_face_verts_scan,
                            target_vertex_normals=face_normals_scan,
                            exclude_indices=exclude_indices)

    def get_closest_match(verts: np.ndarray, dist_thr=5e-4, cos_thr=0.0):
        valid_mask, pt_matched = closest_match(
            torch.from_numpy(verts).float().cuda(),
            dist_thr=dist_thr,
            cos_thr=cos_thr)
        B = torch.where(valid_mask)[0].detach().cpu().numpy()
        BC = pt_matched.detach().cpu().numpy()
        return B, BC

    max_iter = 20
    dist_thrs = np.linspace(1e-3, 1e-5, max_iter)
    cos_thrs = np.linspace(0, 0.95, max_iter)

    for i in tqdm(range(max_iter)):
        dist_thr = dist_thrs[i]
        cos_thr = cos_thrs[i]
        B, BC = get_closest_match(V_arap, dist_thr=dist_thr, cos_thr=cos_thr)

        arap = AsRigidAsPossible(V,
                                 F,
                                 b_vid=B,
                                 b_v_bounded=False,
                                 b_fid=lms_fid,
                                 b_bary_coords=lms_bary_coords,
                                 b_f_bounded=False)
        V_arap = arap.solve(np.vstack([BC, scan_lms]), V_arap)

    ps.init()
    ps.register_surface_mesh("template", V, F)
    ps.register_surface_mesh("template_arap", V_arap, F)
    ps.register_surface_mesh("scan", scan.vertices, scan.faces)
    ps.show()

    template.vertices = V_arap
    write_obj("results/lm_nicp_arap.obj", template)
