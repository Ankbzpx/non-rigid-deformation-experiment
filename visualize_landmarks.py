import numpy as np
import polyscope as ps
import json
from icecream import ic
import trimesh

from mesh_helper import read_obj, load_face_landmarks

if __name__ == '__main__':
    template = read_obj('results/template_icp_match.obj')
    template_lms = load_face_landmarks(template,
                                       'results/template_icp_match_lms.txt')

    scan: trimesh.Trimesh = trimesh.load('data/scan.ply',
                                         process=False,
                                         maintain_order=True)
    scan_lms_data = json.load(open('data/scan_3d.txt'))
    scan_lms = np.stack(
        [np.array([lm['x'], lm['y'], lm['z']]) for lm in scan_lms_data])

    ps.init()
    ps.register_surface_mesh('template', template.vertices, template.faces)
    ps.register_point_cloud('template_lms', template_lms)
    ps.register_surface_mesh('scan', scan.vertices, scan.faces)
    ps.register_point_cloud('scan_lms', scan_lms)
    ps.show()
