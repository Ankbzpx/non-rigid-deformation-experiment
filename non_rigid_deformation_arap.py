import numpy as np
import polyscope as ps
from icecream import ic
import igl
import json

from mesh_helper import read_obj, write_obj
from arap import AsRigidAsPossible
import torch
from non_rigid_deformation import closest_point_triangle_match, closest_point_on_triangle
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

    # handle_group_ids = [0, 16, 18, 19, 21, 22]
    # boundary_handle_list = [
    #     np.unique(igl.boundary_facets(face_groups[id]))
    #     for id in handle_group_ids
    # ]
    # boundary_handle_indices = np.unique(np.concatenate(boundary_handle_list))

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

    V_2d = template.uvs[template.face_uvs_idx].mean(1)

    mouth_left_corner_mask = np.logical_and(
        np.logical_and(V_2d[:, 0] > 0.4163, V_2d[:, 0] < 0.4522),
        np.logical_and(V_2d[:, 1] > 0.4335, V_2d[:, 1] < 0.4714))

    mouth_right_corner_mask = np.logical_and(
        np.logical_and(V_2d[:, 0] > 1.0 - 0.4522, V_2d[:, 0] < 1.0 - 0.4163),
        np.logical_and(V_2d[:, 1] > 0.4335, V_2d[:, 1] < 0.4714))

    mouth_corner_mask = np.logical_or(mouth_left_corner_mask,
                                      mouth_right_corner_mask)
    mouth_corner_vid = np.unique(F[mouth_corner_mask])

    eye_left_corner_mask = np.logical_and(
        np.logical_and(V_2d[:, 0] > 0.6113, V_2d[:, 0] < 0.6429),
        np.logical_and(V_2d[:, 1] > 0.6104, V_2d[:, 1] < 0.6384))

    eye_right_corner_mask = np.logical_and(
        np.logical_and(V_2d[:, 0] > 1.0 - 0.6429, V_2d[:, 0] < 1.0 - 0.6113),
        np.logical_and(V_2d[:, 1] > 0.6104, V_2d[:, 1] < 0.6384))

    eye_corner_mask = np.logical_or(eye_left_corner_mask, eye_right_corner_mask)
    eye_corner_vid = np.unique(F[eye_corner_mask])

    scan: trimesh.Trimesh = trimesh.load('data/scan_decimated.obj',
                                         process=False,
                                         maintain_order=True)
    scan_lms_data = json.load(open('data/scan_3d.txt'))
    scan_lms = np.stack(
        [np.array([lm['x'], lm['y'], lm['z']]) for lm in scan_lms_data])

    per_face_verts_scan = torch.from_numpy(
        scan.vertices[scan.faces]).float().cuda()
    face_normals_scan = torch.from_numpy(np.copy(
        scan.face_normals)).float().cuda()

    scan_lms = closest_point_on_triangle(
        torch.from_numpy(scan_lms).float().cuda(), per_face_verts_scan,
        face_normals_scan)[0].detach().cpu().numpy()

    b_f_weight = np.ones(len(lms_fid))
    b_f_weight[np.in1d(lms_fid, np.arange(NF)[mouth_corner_mask])] = 1e-1

    V_weight = np.ones(len(V))
    V_weight[eye_corner_vid] = 1e3

    arap = AsRigidAsPossible(V,
                             F,
                             V_weight=V_weight,
                             b_fid=lms_fid,
                             b_bary_coords=lms_bary_coords,
                             b_f_bounded=False,
                             b_f_weight=b_f_weight)
    V_arap = arap.solve(scan_lms, V)

    # template_lms = (V[F[lms_fid]] * lms_bary_coords[..., None]).sum(1)
    # template_lms_arap = (V_arap[F[lms_fid]] * lms_bary_coords[..., None]).sum(1)

    # ps.init()
    # ps.register_surface_mesh("template", V, F, enabled=False)
    # ps.register_surface_mesh("template_arap", V_arap, F)
    # ps.register_surface_mesh("scan", scan.vertices, scan.faces)
    # ps.register_point_cloud("template_lms",
    #                         template_lms,
    #                         radius=2e-3,
    #                         enabled=False)
    # ps.register_point_cloud("template_lms_arap", template_lms_arap, radius=2e-3)
    # ps.register_point_cloud("scan_lms", scan_lms, radius=2e-3)
    # ps.show()

    # template.vertices = V_arap
    # write_obj("results/lm_arap.obj", template)

    faces = torch.from_numpy(F).long().cuda()
    VF, NI = igl.vertex_triangle_adjacency(F, NV)
    vert_face_adjacency = [
        torch.from_numpy(vf_indices).long().cuda()
        for vf_indices in np.split(VF, NI[1:-1])
    ]
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
    dist_thrs = np.linspace(5e-4, 1e-5, max_iter)
    cos_thrs = np.linspace(0.5, 0.95, max_iter)
    closest_match_weights = np.linspace(1, 1e3, max_iter)

    for i in tqdm(range(max_iter)):
        dist_thr = dist_thrs[i]
        cos_thr = cos_thrs[i]
        b_v_weight = closest_match_weights[i]
        B, BC = get_closest_match(V_arap, dist_thr=dist_thr, cos_thr=cos_thr)

        b_v_weight = b_v_weight * np.ones(len(B))
        v_mask = np.in1d(B, mouth_corner_vid)
        b_v_weight[v_mask] = 1e-2
        v_mask = np.in1d(B, eye_corner_vid)
        b_v_weight[v_mask] = 1e-2

        arap = AsRigidAsPossible(V,
                                 F,
                                 V_weight=V_weight,
                                 b_vid=B,
                                 b_v_bounded=False,
                                 b_v_weight=b_v_weight,
                                 b_fid=lms_fid,
                                 b_bary_coords=lms_bary_coords,
                                 b_f_bounded=False,
                                 b_f_weight=b_f_weight)
        V_arap = arap.solve(np.vstack([BC, scan_lms]), V_arap)

    ps.init()
    ps.register_surface_mesh("template", V, F, enabled=False)
    ps.register_surface_mesh("template_arap", V_arap, F)
    ps.register_surface_mesh("scan", scan.vertices, scan.faces, enabled=False)
    ps.show()

    template.vertices = V_arap
    write_obj("results/lm_nicp_arap_weighted_eye_mouth.obj", template)
