import argparse
import igl
import json
import numpy as np

from mesh_helper import read_obj, write_obj, load_face_landmarks, OBJMesh
from arap import AsRigidAsPossible
import torch
from non_rigid_deformation import closest_point_triangle_match, closest_point_on_triangle
import trimesh
from functools import partial
from tqdm import tqdm
from corase_match_svd import match_correspondence
import copy
from functools import partial

# Debug
import polyscope as ps
from icecream import ic


def load_template(template_path, template_lm_path):
    template = read_obj(template_path)
    lms_data = np.array(json.load(open(template_lm_path)))
    lms_fid = np.int64(lms_data[:, 0])
    lms_uv = np.float64(lms_data[:, 1:])

    lms_bary_coords = np.stack(
        [lms_uv[:, 0], lms_uv[:, 1], 1 - lms_uv[:, 0] - lms_uv[:, 1]], -1)

    template_lms = load_face_landmarks(template, template_lm_path)

    return {
        "template": template,
        "lms_fid": lms_fid,
        "lms_bary_coords": lms_bary_coords,
        "template_lms": template_lms
    }


def solve_deform(template: OBJMesh, lms_fid, lms_bary_coords, template_lms,
                 scan_path, scan_lms_path):
    scan: trimesh.Trimesh = trimesh.load(scan_path,
                                         process=False,
                                         maintain_order=True)
    scan_lms_data = json.load(open(scan_lms_path))
    scan_lms = np.stack(
        [np.array([lm['x'], lm['y'], lm['z']]) for lm in scan_lms_data])

    R, s, t = match_correspondence(template_lms, scan_lms)

    V = s * template.vertices @ R.T + t
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

    per_face_verts_scan = torch.from_numpy(
        scan.vertices[scan.faces]).float().cuda()
    face_normals_scan = torch.from_numpy(np.copy(
        scan.face_normals)).float().cuda()

    scan_lms = closest_point_on_triangle(
        torch.from_numpy(scan_lms).float().cuda(), per_face_verts_scan,
        face_normals_scan)[0].detach().cpu().numpy()

    arap = AsRigidAsPossible(V,
                             F,
                             b_fid=lms_fid,
                             b_bary_coords=lms_bary_coords,
                             b_f_bounded=False)
    V_arap = arap.solve(scan_lms, V)

    lm_dist = np.linalg.norm(scan_lms - (s * template_lms @ R.T + t), axis=1)
    dist_thr = np.median(lm_dist)

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

    def get_closest_match(verts: np.ndarray, dist_thr=dist_thr, cos_thr=0.0):
        valid_mask, pt_matched, dist_closest = closest_match(
            torch.from_numpy(verts).float().cuda(),
            dist_thr=dist_thr,
            cos_thr=cos_thr)
        B = torch.where(valid_mask)[0].detach().cpu().numpy()
        BC = pt_matched.detach().cpu().numpy()
        return B, BC, dist_closest

    max_iter = 20
    dist_thrs = np.linspace(50 * dist_thr, dist_thr, max_iter)
    cos_thrs = np.linspace(0.5, 0.95, max_iter)
    closest_match_weights = np.linspace(1, 1e3, max_iter)
    p_range = np.linspace(10, 0.4, max_iter)
    landmark_weights = np.linspace(1, 1, max_iter)

    # Alternating between rigidity and closest point match
    stiffness_thrs = np.linspace(10, 1, max_iter)
    stiffness_thrs[1::2] *= np.linspace(1e-1, 1e-1, max_iter // 2)
    stiffness_thrs[-1] = 1

    for i in tqdm(range(max_iter)):
        dist_thr = dist_thrs[i]
        cos_thr = cos_thrs[i]
        b_f_weight = landmark_weights[i]
        B, BC, dist_closest = get_closest_match(V_arap,
                                                dist_thr=dist_thr,
                                                cos_thr=cos_thr)

        # ps.init()
        # ps.register_point_cloud('B', V_arap[B])
        # ps.register_point_cloud('BC', BC)
        # ps.show()
        # exit()

        # Robust weight (welsch_weight)
        p = p_range[i]
        base = p * dist_closest.median() / (np.sqrt(2) * 2.3)
        weight = torch.exp(-(dist_closest / base)**2 / np.sqrt(2))

        b_v_weight = weight.detach().cpu().numpy() * closest_match_weights[i]

        b_v_weight /= b_v_weight.max()
        b_v_weight *= stiffness_thrs[i]

        arap = AsRigidAsPossible(V,
                                 F,
                                 b_vid=B,
                                 b_v_bounded=False,
                                 b_v_weight=b_v_weight,
                                 b_fid=lms_fid,
                                 b_bary_coords=lms_bary_coords,
                                 b_f_bounded=False,
                                 b_f_weight=b_f_weight * np.ones(len(lms_fid)))
        V_arap = arap.solve(np.vstack([BC, scan_lms]), V_arap)

        # ps.init()
        # ps.register_surface_mesh('V_arap', V_arap, F)
        # ps.register_surface_mesh('scan', scan.vertices, scan.faces)
        # ps.show()

        # Good enough...
        if i == 2:
            break

    model_matched = copy.deepcopy(template)
    model_matched.vertices = V_arap

    return model_matched


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARAP + Non-rigid ICP.')
    parser.add_argument('--template_pg_path',
                        type=str,
                        default='templates/template_pg.obj')
    parser.add_argument('--template_pg_lm_path',
                        type=str,
                        default='templates/template_pg_lms.txt')
    parser.add_argument('--scan_path', type=str, default='scan_data/scan.ply')
    parser.add_argument('--scan_lms_path',
                        type=str,
                        default='scan_data/scan_3d.txt')
    parser.add_argument('--match_save_path',
                        type=str,
                        default='results/scan_match_arap_nicp.obj')
    parser.add_argument('--remove_interior',
                        action='store_true',
                        help='Remove eye ball and tongue')
    args = parser.parse_args()

    # Template mesh with pre-specified polygon groups
    template_pg_path = args.template_pg_path
    template_pg_lm_path = args.template_pg_lm_path
    scan_path = args.scan_path
    scan_lms_path = args.scan_lms_path
    match_save_path = args.match_save_path
    remove_interior = args.remove_interior

    solve_deform_partial = partial(
        solve_deform, **load_template(template_pg_path, template_pg_lm_path))

    model_save = solve_deform_partial(scan_path=scan_path,
                                      scan_lms_path=scan_lms_path)

    # WARNING: Current implementation is surface deform only.
    # It is prone to self-intersection and can not guarantee the interior correctness
    # Here is the option to remove them
    # FIXME: Implement volume ARAP
    if remove_interior:
        fid = np.arange(len(model_save.faces_quad))
        face_group_ids = np.split(fid, model_save.polygon_groups_quad[1:])

        select_faces_mask = np.ones(len(model_save.faces_quad))
        select_faces_mask[face_group_ids[3]] = 0.
        select_faces_mask[face_group_ids[10]] = 0.
        select_faces_mask[face_group_ids[13]] = 0.
        select_faces_mask = select_faces_mask.astype(bool)

        select_faces = model_save.faces_quad[select_faces_mask]
        select_face_uvs = model_save.face_uvs_idx_quad[select_faces_mask]

        vertex_ids, verts_unique_inverse = np.unique(select_faces,
                                                     return_inverse=True)
        model_save.vertices = model_save.vertices[vertex_ids]

        uv_ids, uv_unique_inverse = np.unique(select_face_uvs,
                                              return_inverse=True)
        model_save.uvs = model_save.uvs[uv_ids]

        model_save.faces_quad = np.arange(
            len(vertex_ids))[verts_unique_inverse].reshape(-1, 4)
        model_save.face_uvs_idx_quad = np.arange(
            len(uv_ids))[uv_unique_inverse].reshape(-1, 4)

    write_obj(match_save_path, model_save)
