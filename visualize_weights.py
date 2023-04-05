import numpy as np
import polyscope as ps
from mesh_helper import read_obj, load_face_landmarks
from icecream import ic
import igl

if __name__ == '__main__':
    template = read_obj('results/template_icp_match.obj')
    template_lms = load_face_landmarks(template,
                                       'results/template_icp_match_lms.txt')

    # 10: tongue
    # 152: left eye
    # 1086: right eye
    bbw = np.load('results/bbw.npy')
    bbw = bbw / bbw.sum(1, keepdims=True)
    bi = np.load('results/bi.npy')

    FN = igl.per_face_normals(template.vertices, template.faces,
                              np.array([0., 0., 1.])[None, ...])

    handles = template.vertices[bi]
    deformation = np.zeros_like(template.vertices[bi])

    d = 0.1 * FN[20131]
    deformation[11, :] = d
    deformation[1520, :] = d
    deformation[1558, :] = d
    deformation[616, :] = d

    delta = bbw @ deformation

    ps.init()
    ps.register_surface_mesh('template',
                             template.vertices,
                             template.faces,
                             enabled=False)
    ps.register_surface_mesh('template deform', delta + template.vertices,
                             template.faces)
    ps.register_point_cloud('handles', handles, enabled=False)
    ps.register_point_cloud('handles deform', handles + deformation)
    ps.register_point_cloud('template lms', template_lms, enabled=False)
    ps.show()
