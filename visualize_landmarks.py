import numpy as np
import polyscope as ps
import json
from icecream import ic

from mesh_helper import read_obj

if __name__ == '__main__':

    # https://docs.r3ds.com/Wrap/Nodes/SelectPoints/SelectPoints.html
    template_lms_data = np.array(json.load(open('data/mastermodel_3d.txt')))
    template_lms_fid = np.int64(template_lms_data[:, 0])
    template_lms_uv = np.float64(template_lms_data[:, 1:])

    scan_lms_data = json.load(open('data/scan_3d.txt'))
    scan_lms = np.stack(
        [np.array([lm['x'], lm['y'], lm['z']]) for lm in scan_lms_data])

    template = read_obj('data/mastermodel_3d.obj')

    per_landmark_face_verts = template.vertices[
        template.faces[template_lms_fid]]
    A = per_landmark_face_verts[:, 0, :]
    B = per_landmark_face_verts[:, 1, :]
    C = per_landmark_face_verts[:, 2, :]

    template_lms = C + (A - C) * template_lms_uv[:, 0][:, None] + (
        B - C) * template_lms_uv[:, 1][:, None]

    ps.init()
    ps.register_surface_mesh('template', template.vertices, template.faces)
    ps.register_point_cloud('template_lms', template_lms)
    ps.show()

    # scan: trimesh.Trimesh = trimesh.load('data/scan.ply')

    # ps.init()
    # ps.register_surface_mesh('scan', scan.vertices, scan.faces)
    # ps.register_point_cloud('scan_lms', scan_lms)
    # ps.show()
