import numpy as np
import trimesh
import polyscope as ps
import json
from mesh_helper import read_obj, write_obj
from icecream import ic


def match_correspondence(pc_float_lms, pc_fixed_lms):
    P_mu = np.average(pc_float_lms, 0, keepdims=True)
    P_bar = pc_float_lms - P_mu
    Q_mu = np.average(pc_fixed_lms, 0, keepdims=True)
    Q_bar = pc_fixed_lms - Q_mu

    cov = Q_bar.T @ P_bar / len(pc_float_lms)
    U, S, V_T = np.linalg.svd(cov)
    R = U @ V_T

    # Reflection
    E = np.ones(len(cov))
    E[-1] = np.sign(np.linalg.det(R))
    R = R * E

    demo = np.sum(P_bar * P_bar) / len(pc_float_lms)

    s = np.sum(S) / demo
    t = Q_mu - s * P_mu @ R.T

    return R, s, t


if __name__ == '__main__':
    template = read_obj('data/mastermodel_3d.obj')

    # https://docs.r3ds.com/Wrap/Nodes/SelectPoints/SelectPoints.html
    template_lms_data = np.array(json.load(open('data/mastermodel_3d.txt')))
    template_lms_fid = np.int64(template_lms_data[:, 0])
    template_lms_uv = np.float64(template_lms_data[:, 1:])

    per_landmark_face_verts = template.vertices[
        template.faces[template_lms_fid]]
    A = per_landmark_face_verts[:, 0, :]
    B = per_landmark_face_verts[:, 1, :]
    C = per_landmark_face_verts[:, 2, :]

    template_lms = C + (A - C) * template_lms_uv[:, 0][:, None] + (
        B - C) * template_lms_uv[:, 1][:, None]

    scan: trimesh.Trimesh = trimesh.load('data/scan.ply',
                                         process=False,
                                         maintain_order=True)

    scan_lms_data = json.load(open('data/scan_3d.txt'))
    scan_lms = np.stack(
        [np.array([lm['x'], lm['y'], lm['z']]) for lm in scan_lms_data])

    R, s, t = match_correspondence(template_lms, scan_lms)
    template.vertices = s * template.vertices @ R.T + t
    template_lms_matched = s * template_lms @ R.T + t

    write_obj('results/template_icp_match.obj', template)

    ps.init()
    ps.register_surface_mesh("template", template.vertices, template.faces)
    ps.register_surface_mesh("scan", scan.vertices, scan.faces)
    ps.register_point_cloud("template_lms_matched", template_lms_matched)
    ps.register_point_cloud("scan_lms", scan_lms)
    ps.show()
