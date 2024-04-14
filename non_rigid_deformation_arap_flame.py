import argparse
import igl
import json
import numpy as np

from mesh_helper import OBJMesh, write_obj, OBJMesh
from arap import AsRigidAsPossible
import torch
from non_rigid_deformation import closest_point_triangle_match, closest_point_on_triangle
import trimesh
from functools import partial
from tqdm import tqdm
from corase_match_svd import match_correspondence
import copy
from functools import partial
import pickle

# Debug
import polyscope as ps
from icecream import ic


def load_template(flame_path, flame_mediapipe_lm_path):

    with open(flame_path, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
        V = np.float64(data['v_template'])
        F = np.int64(data['f'])
        VN = igl.per_vertex_normals(V, F)

        template = OBJMesh(V, F, VN)

    lms_data = np.load(flame_mediapipe_lm_path)
    lmk_face_idx = np.int64(lms_data['lmk_face_idx'])
    lmk_b_coords = np.float64(lms_data['lmk_b_coords'])
    landmark_indices = lms_data['landmark_indices']

    template_lms = (template.vertices[template.faces][lmk_face_idx] *
                    lmk_b_coords[..., None]).sum(1)

    return {
        "template": template,
        "lms_fid": lmk_face_idx,
        "lms_bary_coords": lmk_b_coords,
        "template_lms": template_lms,
        "landmark_indices": landmark_indices
    }


def deformation_gradient(V, V_deform, F):
    z = np.array([0., 1., 0.])[None, :]
    FN = igl.per_face_normals(V, F, z)
    FN_deform = igl.per_face_normals(V_deform, F, z)

    per_face_verts = V[F]
    per_face_verts_deform = V_deform[F]

    V_i = np.stack([
        per_face_verts[:, 0] - per_face_verts[:, 2],
        per_face_verts[:, 1] - per_face_verts[:, 2], FN
    ], -1)

    V_j = np.stack([
        per_face_verts_deform[:, 0] - per_face_verts_deform[:, 2],
        per_face_verts_deform[:, 1] - per_face_verts_deform[:, 2], FN_deform
    ], -1)

    # Deformation gradient
    # J @ V_i = V_j
    J = np.einsum('bij,bjk->bik', V_j, np.linalg.inv(V_i))

    return J


def solve_deform(template: OBJMesh, lms_fid, lms_bary_coords, template_lms,
                 landmark_indices, scan_path, scan_lms_path):
    scan: trimesh.Trimesh = trimesh.load(scan_path,
                                         process=False,
                                         maintain_order=True)
    scan_lms_data = json.load(open(scan_lms_path))
    scan_lms = np.stack(
        [np.array([lm['x'], lm['y'], lm['z']]) for lm in scan_lms_data])
    scan_lms = scan_lms[landmark_indices]

    R, s, t = match_correspondence(template_lms, scan_lms)

    V = s * template.vertices @ R.T + t
    F = template.faces

    NV = len(V)
    NF = len(F)

    per_face_verts_scan = torch.from_numpy(
        scan.vertices[scan.faces]).float().cuda()
    face_normals_scan = torch.from_numpy(np.copy(
        scan.face_normals)).float().cuda()

    scan_lms = closest_point_on_triangle(
        torch.from_numpy(scan_lms).float().cuda(), per_face_verts_scan,
        face_normals_scan)[0].detach().cpu().numpy()

    # 3929
    eyeball_center_l_idx = 3929
    eyeball_center_r_idx = 3930

    # 546
    eyeball_l_idx = 3931
    eyeball_r_idx = 3931 + 546

    B_eyeball = np.arange(eyeball_l_idx, eyeball_r_idx + 546)
    BC_eyeball = V[B_eyeball]

    eyeball_l_offset = V[eyeball_l_idx:eyeball_r_idx] - V[eyeball_center_l_idx][
        None, :]
    eyeball_r_offset = V[eyeball_r_idx:] - V[eyeball_center_r_idx][None, :]

    # Ignore isolated eyeballs
    exclude_indices = B_eyeball

    arap = AsRigidAsPossible(V,
                             F,
                             b_vid=B_eyeball,
                             b_fid=lms_fid,
                             b_bary_coords=lms_bary_coords)
    V_arap = arap.solve(np.vstack([BC_eyeball, scan_lms]), V)

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

    for i in tqdm(range(max_iter)):
        dist_thr = dist_thrs[i]
        cos_thr = cos_thrs[i]
        B, BC, dist_closest = get_closest_match(V_arap,
                                                dist_thr=dist_thr,
                                                cos_thr=cos_thr)

        # ps.init()
        # ps.register_point_cloud('B', V_arap[B])
        # ps.register_point_cloud('BC', BC)
        # ps.show()
        # exit()

        # Append eyeball
        B = np.concatenate([B_eyeball, B])
        BC = np.vstack([BC_eyeball, BC])

        arap = AsRigidAsPossible(V,
                                 F,
                                 b_vid=B,
                                 b_fid=lms_fid,
                                 b_bary_coords=lms_bary_coords)
        V_arap = arap.solve(np.vstack([BC, scan_lms]), V_arap)

        # ps.init()
        # ps.register_surface_mesh('V_arap', V_arap, F)
        # ps.register_surface_mesh('scan', scan.vertices, scan.faces)
        # ps.show()

        if i == 2:
            break

    # Fix eyeballs. Need a better approach
    V_arap[eyeball_l_idx:eyeball_r_idx] = eyeball_l_offset + V_arap[
        eyeball_center_l_idx][None, :]
    V_arap[eyeball_r_idx:] = eyeball_r_offset + V_arap[eyeball_center_r_idx][
        None, :]

    model_matched = copy.deepcopy(template)
    model_matched.vertices = V_arap

    return model_matched


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARAP + Non-rigid ICP.')
    parser.add_argument('--flame_path',
                        type=str,
                        default='templates/flame2023.pkl')
    parser.add_argument('--flame_mediapipe_lm_path',
                        type=str,
                        default='templates/mediapipe_landmark_embedding.npz')
    parser.add_argument('--scan_path', type=str, default='scan_data/head1.obj')
    parser.add_argument('--scan_lms_path',
                        type=str,
                        default='scan_data/head1_landmarks.txt')
    parser.add_argument('--match_save_path',
                        type=str,
                        default='results/head1_match_arap_nicp.obj')
    args = parser.parse_args()

    flame_path = args.flame_path
    flame_mediapipe_lm_path = args.flame_mediapipe_lm_path
    scan_path = args.scan_path
    scan_lms_path = args.scan_lms_path
    match_save_path = args.match_save_path

    solve_deform_partial = partial(
        solve_deform, **load_template(flame_path, flame_mediapipe_lm_path))

    model_save = solve_deform_partial(scan_path=scan_path,
                                      scan_lms_path=scan_lms_path)
    write_obj(match_save_path, model_save)
