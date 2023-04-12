import numpy as np
import trimesh
import polyscope as ps
import json
from mesh_helper import read_obj, write_obj, load_face_landmarks
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
    if np.linalg.det(R) < 0:
        E = np.eye(len(cov))
        min_idx = np.argmin(S)
        E[min_idx, min_idx] = -1
        R = U @ E @ V_T

    demo = np.sum(P_bar * P_bar) / len(pc_float_lms)

    s = np.sum(S) / demo
    t = Q_mu - s * P_mu @ R.T

    return R, s, t


if __name__ == '__main__':
    template = read_obj('data/mastermodel_3d.obj')
    template_lms = load_face_landmarks(template, 'data/mastermodel_3d.txt')

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
