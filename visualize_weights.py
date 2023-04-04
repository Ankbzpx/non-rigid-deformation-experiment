import numpy as np
import polyscope as ps
from mesh_helper import read_obj
from icecream import ic

if __name__ == '__main__':
    template = read_obj('results/template_icp_match.obj')

    # 10: tongue
    # 152: left eye
    # 1086: right eye
    bbw = np.load('results/bbw.npy')
    # bbw = bbw / bbw.sum(1, keepdims=True)

    bi = np.load('results/bi.npy')

    thr = 1e-2

    for i in np.linspace(0, len(template.vertices), 100).astype(np.int64):

        target_vert = template.vertices[i]
        target_control_weights = bbw[i, bbw[i] > thr]
        target_controls = template.vertices[bi][bbw[i] > thr]

        ps.init()
        ps.register_surface_mesh('template', template.vertices, template.faces)
        ps.register_point_cloud('target_vert', target_vert[None, ...])
        pc_ps = ps.register_point_cloud('target_controls', target_controls)
        pc_ps.add_scalar_quantity('weights',
                                  target_control_weights,
                                  enabled=True)
        ps.show()
