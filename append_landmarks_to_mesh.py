import numpy as np
import polyscope as ps
import json
from icecream import ic
import igl
from scipy.spatial import Delaunay

from mesh_helper import read_obj, write_obj

if __name__ == '__main__':
    template = read_obj('results/template_icp_match.obj')
    lms_data = np.array(json.load(open('results/template_icp_match_lms.txt')))
    lms_fid = np.int64(lms_data[:, 0])
    lms_uv = np.float64(lms_data[:, 1:])

    per_landmark_face_verts = template.vertices[template.faces[lms_fid]]
    A = per_landmark_face_verts[:, 0, :]
    B = per_landmark_face_verts[:, 1, :]
    C = per_landmark_face_verts[:, 2, :]
    template_lms = C + (A - C) * lms_uv[:, 0][:, None] + (
        B - C) * lms_uv[:, 1][:, None]

    per_landmark_face_uvs = template.uvs[template.face_uvs_idx[lms_fid]]
    A_uv = per_landmark_face_uvs[:, 0, :]
    B_uv = per_landmark_face_uvs[:, 1, :]
    C_uv = per_landmark_face_uvs[:, 2, :]
    template_lm_uvs = C_uv + (A_uv - C_uv) * lms_uv[:, 0][:, None] + (
        B_uv - C_uv) * lms_uv[:, 1][:, None]

    lms_fid_unique, lms_fid_unique_idx, lms_fid_unique_count = np.unique(
        lms_fid, return_index=True, return_counts=True)

    F_extra = []
    F_uv_extra = []
    duplicate_faces = lms_fid_unique[lms_fid_unique_count > 1]
    for duplicate_face in duplicate_faces:
        duplicate_idx = np.where(lms_fid == duplicate_face)[0]

        tri_verts = template.vertices[template.faces[duplicate_face]]
        tri_uvs = template.uvs[template.face_uvs_idx[duplicate_face]]

        tri_lm_verts = template_lms[duplicate_idx]
        tri_lm_uvs = template_lm_uvs[duplicate_idx]

        points = np.vstack([tri_uvs, tri_lm_uvs])
        tri = Delaunay(points)

        vert_idx_map = np.concatenate([
            template.faces[duplicate_face],
            len(template.vertices) + duplicate_idx
        ])

        uv_idx_map = np.concatenate([
            template.face_uvs_idx[duplicate_face],
            len(template.uvs) + duplicate_idx
        ])

        F_extra.append(vert_idx_map[tri.simplices.flatten()].reshape(-1, 3))
        F_uv_extra.append(uv_idx_map[tri.simplices.flatten()].reshape(-1, 3))

    V_new = np.vstack([template.vertices, template_lms])
    V_new_idx = len(template.vertices) + np.arange(len(template_lms))

    F_keep_mask = np.ones(len(template.faces))
    F_keep_mask[lms_fid_unique] = 0
    F_keep_mask = F_keep_mask.astype(bool)
    F_keep = template.faces[F_keep_mask]

    F_lm_idx = template.faces[lms_fid_unique[lms_fid_unique_count == 1]]
    V_append = V_new_idx[lms_fid_unique_idx[lms_fid_unique_count == 1]]

    F_append = np.vstack([
        np.stack([V_append, F_lm_idx[:, 0], F_lm_idx[:, 1]], -1),
        np.stack([V_append, F_lm_idx[:, 1], F_lm_idx[:, 2]], -1),
        np.stack([V_append, F_lm_idx[:, 2], F_lm_idx[:, 0]], -1)
    ])
    F_new = np.vstack([F_keep, F_append] + F_extra)

    # UV
    UV_new = np.vstack([template.uvs, template_lm_uvs])
    UV_new_idx = len(template.uvs) + np.arange(len(template_lm_uvs))

    F_UV_keep = template.face_uvs_idx[F_keep_mask]
    F_UV_lm_idx = template.face_uvs_idx[lms_fid_unique[lms_fid_unique_count ==
                                                       1]]
    UV_append = UV_new_idx[lms_fid_unique_idx[lms_fid_unique_count == 1]]

    F_UV_new = np.vstack([
        np.stack([UV_append, F_UV_lm_idx[:, 0], F_UV_lm_idx[:, 1]], -1),
        np.stack([UV_append, F_UV_lm_idx[:, 1], F_UV_lm_idx[:, 2]], -1),
        np.stack([UV_append, F_UV_lm_idx[:, 2], F_UV_lm_idx[:, 0]], -1)
    ])
    F_UV_new = np.vstack([F_UV_keep, F_UV_new] + F_uv_extra)

    template.faces_quad = None
    template.face_uvs_idx_quad = None
    template.vertices = V_new
    template.faces = F_new
    template.uvs = UV_new
    template.face_uvs_idx = F_UV_new

    write_obj("results/template_with_lms.obj", template)
