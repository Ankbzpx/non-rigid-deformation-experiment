import numpy as np
import polyscope as ps
import json
from icecream import ic
import trimesh
import igl
import robust_laplacian
import scipy

from mesh_helper import read_obj

if __name__ == '__main__':
    template = read_obj('results/template_icp_match.obj')
    lms_data = np.array(json.load(open('results/template_icp_match_lms.txt')))
    lms_fid = np.int64(lms_data[:, 0])
    lms_uv = np.float64(lms_data[:, 1:])

    per_landmark_face_verts = template.vertices[template.faces[lms_fid]]
    A = per_landmark_face_verts[:, 0, :]
    B = per_landmark_face_verts[:, 1, :]
    C = per_landmark_face_verts[:, 2, :]

    lms = C + (A - C) * lms_uv[:, 0][:, None] + (B - C) * lms_uv[:, 1][:, None]

    per_landmark_face_uv = template.uvs[template.face_uvs_idx[lms_fid]]

    ac = np.linalg.norm(A - C, axis=-1)
    bc = np.linalg.norm(B - C, axis=-1)

    avg_edge_len = igl.avg_edge_length(template.vertices, template.faces)

    thr = 0.1
    thr1 = thr * avg_edge_len / ac
    thr2 = thr * avg_edge_len / bc

    # A
    c1 = np.abs(1 - lms_uv[:, 0]) < thr1
    # B
    c2 = np.abs(1 - lms_uv[:, 1]) < thr2
    # C
    c3 = np.logical_and(
        np.abs(lms_uv[:, 0]) < thr1,
        np.abs(lms_uv[:, 1]) < thr2)

    valid = np.logical_or(np.logical_or(c1, c2), c3)

    lm_vid = np.zeros(len(lms_fid), dtype=np.int64)
    lm_vid[c1] = template.faces[lms_fid][c1, 0]
    lm_vid[c2] = template.faces[lms_fid][c2, 1]
    lm_vid[c3] = template.faces[lms_fid][c3, 2]
    lm_vid = lm_vid[valid]

    scan: trimesh.Trimesh = trimesh.load('data/scan.ply',
                                         process=False,
                                         maintain_order=True)
    scan_lms_data = json.load(open('data/scan_3d.txt'))
    scan_lms = np.stack(
        [np.array([lm['x'], lm['y'], lm['z']]) for lm in scan_lms_data])

    ps.init()
    ps.register_surface_mesh('template', template.vertices, template.faces)
    ps.register_point_cloud('lms', template.vertices[lm_vid])
    ps.register_surface_mesh("scan", scan.vertices, scan.faces)
    ps.register_point_cloud("scan_lms", scan_lms[valid])
    ps.show()

    np.save('results/template_lms_vid.npy', lm_vid)
    np.save('results/scan_lms.npy', scan_lms[valid])
