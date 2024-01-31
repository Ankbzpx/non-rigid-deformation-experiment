import numpy as np
import igl
import torch

import json

from mesh_helper import read_obj, load_face_landmarks
from non_rigid_deformation import closest_point_on_triangle

import polyscope as ps
from icecream import ic

template = read_obj('templates/template_pg.obj')

lms, _ = igl.read_triangle_mesh('templates/lm_map.obj')

FN = igl.per_face_normals(template.vertices, template.faces,
                          np.array([0.0, 1.0, 0.0])[:, None])

lms_proj, _, indices = closest_point_on_triangle(
    torch.from_numpy(lms).float().cuda(),
    torch.from_numpy(template.vertices[template.faces]).float().cuda(),
    torch.from_numpy(FN).float().cuda())

lms_proj = lms_proj.detach().cpu().double().numpy()
indices = indices.detach().cpu().numpy()

per_face_vertices = template.vertices[template.faces[indices]]

A = np.ascontiguousarray(per_face_vertices[:, 0])
B = np.ascontiguousarray(per_face_vertices[:, 1])
C = np.ascontiguousarray(per_face_vertices[:, 2])

lms_bary = igl.barycentric_coordinates_tri(lms_proj, A, B, C)

lm_data = []
for i in range(468):
    lm_data.append([int(indices[i]), lms_bary[i, 0], lms_bary[i, 1]])

json_path = 'templates/template_pg_lms_468.txt'

with open(json_path, 'w') as f:
    json.dump(lm_data, f)

lms_dumped = load_face_landmarks(template, json_path)

ps.init()
ps.register_surface_mesh('template', template.vertices, template.faces)
ps.register_point_cloud('lms', lms)
ps.register_point_cloud('lms_proj', lms_proj)
ps.register_point_cloud('lms_dumped', lms_dumped)
ps.show()
