import argparse
import igl
import json
import numpy as np

from mesh_helper import OBJMesh, write_obj, OBJMesh
from arap import AsRigidAsPossible, SymmetricPointToPlane
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


# Remove unreference vertices and assign new vertex indices
def rm_unref_vertices(V, F):
    V_unique, V_unique_idx, V_unique_idx_inv = np.unique(F.flatten(),
                                                         return_index=True,
                                                         return_inverse=True)
    V_id_new = np.arange(len(V_unique))
    V_map = V_id_new[np.argsort(V_unique_idx)]
    V_map_inv = np.zeros((np.max(V_map) + 1,), dtype=np.int64)
    V_map_inv[V_map] = V_id_new

    F = V_map_inv[V_unique_idx_inv].reshape(F.shape)
    V = V[V_unique][V_map]

    return V, F


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


def solve_deform(template: OBJMesh,
                 lms_fid,
                 lms_bary_coords,
                 template_lms,
                 landmark_indices,
                 scan_path,
                 scan_lms_path,
                 use_symmetry=True):
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

    # Remove disconnected components
    A = igl.adjacency_matrix(F)
    n, C, K = igl.connected_components(A)
    c_idx = np.argmax(K)

    # Remap indices
    V_rm_mask = np.ones(len(V))
    V_rm_mask[C != c_idx] = 0
    F_keep_mask = V_rm_mask[F].sum(1) == 3

    f_map = -np.ones(len(F)).astype(np.int64)
    f_map[F_keep_mask] = np.arange(F_keep_mask.sum())

    V, F = rm_unref_vertices(V, F[F_keep_mask])
    lms_fid = f_map[lms_fid]
    VN = igl.per_vertex_normals(V, F)

    # Use barycentric interpolated vertex normal for lm normals
    template_lm = (lms_bary_coords[..., None] * V[F][lms_fid]).sum(1)
    template_lm_normals = (lms_bary_coords[..., None] * VN[F][lms_fid]).sum(1)

    NV = len(V)
    NF = len(F)

    per_face_verts_scan = torch.from_numpy(
        scan.vertices[scan.faces]).float().cuda()
    face_normals_scan = torch.from_numpy(np.copy(
        scan.face_normals)).float().cuda()

    pt_proj, _, f_indices = closest_point_on_triangle(
        torch.from_numpy(scan_lms).float().cuda(), per_face_verts_scan,
        face_normals_scan)
    scan_lms = pt_proj.detach().cpu().numpy()
    # The scan mesh is sufficiently dense, use its face normals instead
    scan_lms_normals = scan.face_normals[f_indices.detach().cpu().numpy()]

    if use_symmetry:
        sym_p2p = SymmetricPointToPlane(V,
                                        F,
                                        b_fid=lms_fid,
                                        b_bary_coords=lms_bary_coords)
        V_init = sym_p2p.solve(V, template_lm_normals, scan_lms,
                               scan_lms_normals)
    else:
        arap = AsRigidAsPossible(V,
                                 F,
                                 b_fid=lms_fid,
                                 b_bary_coords=lms_bary_coords,
                                 smooth_rotation=True)
        V_init = arap.solve(V, scan_lms)

    lm_dist = np.linalg.norm(scan_lms - (s * template_lms @ R.T + t), axis=1)
    dist_thr_median = np.median(lm_dist)

    faces = torch.from_numpy(F).long().cuda()
    VF, NI = igl.vertex_triangle_adjacency(F, NV)
    vert_face_adjacency = [
        torch.from_numpy(vf_indices).long().cuda()
        for vf_indices in np.split(VF, NI[1:-1])
    ]
    exclude_indices = np.array([])
    exclude_indices = torch.from_numpy(exclude_indices).cuda().long()
    closest_match = partial(closest_point_triangle_match,
                            faces=faces,
                            vert_face_adjacency=vert_face_adjacency,
                            target_per_face_verts=per_face_verts_scan,
                            target_vertex_normals=face_normals_scan,
                            exclude_indices=exclude_indices)

    def get_closest_match(verts: np.ndarray, dist_thr, cos_thr):
        valid_mask, pt_matched, dist_closest, f_indices = closest_match(
            torch.from_numpy(verts).float().cuda(),
            dist_thr=dist_thr,
            cos_thr=cos_thr)
        B = torch.where(valid_mask)[0].detach().cpu().numpy()
        Q = pt_matched.detach().cpu().numpy()
        N_p = VN[valid_mask.detach().cpu().numpy()]
        N_q = scan.face_normals[f_indices.detach().cpu().numpy()]
        return B, N_p, Q, N_q, dist_closest

    max_iter = 10
    dist_thrs = np.linspace(50 * dist_thr_median, 25 * dist_thr_median,
                            max_iter)
    cos_thrs = np.linspace(0.5, 0.75, max_iter)

    V_arap = V_init
    for i in tqdm(range(max_iter)):
        dist_thr = dist_thrs[i]
        cos_thr = cos_thrs[i]
        B, N_p, Q, N_q, _ = get_closest_match(V_arap,
                                              dist_thr=dist_thr,
                                              cos_thr=cos_thr)

        if use_symmetry:
            sym_p2p = SymmetricPointToPlane(V, F, b_vid=B)
            V_arap = sym_p2p.solve(V_arap,
                                   N_p,
                                   Q,
                                   N_q,
                                   robust_weight=True,
                                   w_arap=1,
                                   w_sr=1e-4)
        else:
            arap = AsRigidAsPossible(V,
                                     F,
                                     b_vid=B,
                                     b_fid=lms_fid,
                                     b_bary_coords=lms_bary_coords,
                                     soft_weight=2.0,
                                     soft_exclude=False,
                                     smooth_rotation=True)
            V_arap = arap.solve(V_arap, np.vstack([Q, scan_lms]))

    ps.init()
    ps.register_surface_mesh('V_init', V_arap, F)
    ps.register_surface_mesh('scan', scan.vertices, scan.faces)
    ps.show()
    # exit()

    model_matched = copy.deepcopy(template)
    model_matched.vertices = V_arap
    model_matched.faces = F

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
