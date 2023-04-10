import numpy as np
import polyscope as ps
from mesh_helper import read_obj
from icecream import ic
import potpourri3d as pp3d
import json

if __name__ == '__main__':
    template_with_lm = read_obj('results/template_with_lms.obj')
    lm_verts_count = len(json.load(open('results/template_icp_match_lms.txt')))

    V = template_with_lm.vertices
    F = template_with_lm.faces

    lm_indices = np.arange(len(V))[-lm_verts_count:]

    solver = pp3d.MeshHeatMethodDistanceSolver(V, F)

    geodesic_dists = []
    for i in range(lm_verts_count):
        geodesic_dists.append(solver.compute_distance(lm_indices[i]))
    geodesic_dists = np.stack(geodesic_dists, -1)

    geodesic_weights = geodesic_dists[:len(V) - lm_verts_count]
    np.save('results/geodesic_weights.npy', geodesic_weights)
