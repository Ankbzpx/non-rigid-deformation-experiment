import numpy as np
import polyscope as ps
from mesh_helper import OBJMesh, read_obj, load_face_landmarks
import igl
from icecream import ic
import igl
import json


def get_group_boundary(mesh: OBJMesh, exlude=[]):
    boundary_indices_list = []
    exlude_indices_list = []
    exlude_control_indices = []
    num_groups = len(mesh.polygon_groups)
    for i in range(num_groups):
        mask = np.zeros(len(mesh.faces))
        if i == num_groups - 1:
            mask[mesh.polygon_groups[-1]:] = 1
        else:
            mask[mesh.polygon_groups[i]:mesh.polygon_groups[i + 1]] = 1
        mask = mask.astype(bool)

        masked_faces = mesh.faces[mask]
        boundary_indices = np.unique(igl.boundary_facets(masked_faces))
        boundary_indices_list.append(boundary_indices)

        if i in exlude:
            exlude_control_indices.append(boundary_indices[0])
            # ignore boundary
            exlude_indices_list.append(
                np.setxor1d(np.unique(masked_faces), boundary_indices))

    boundary_indices = np.unique(np.concatenate(boundary_indices_list))
    exlude_control_indices = np.array([
        np.where(boundary_indices == idx) for idx in exlude_control_indices
    ]).flatten()

    return boundary_indices, exlude_indices_list, exlude_control_indices


if __name__ == '__main__':
    template = read_obj('results/template_icp_match.obj')
    template_lms = load_face_landmarks(template,
                                       'results/template_icp_match_lms.txt')

    template_lms_data = np.array(
        json.load(open('results/template_icp_match_lms.txt')))
    template_lms_fid = np.int64(template_lms_data[:, 0])
    template_lms_uv = np.float64(template_lms_data[:, 1:])

    # 3: tongue
    # 10: left eye
    # 13: right eye
    boundary_indices, exlude_indices_list, exlude_control_indices = get_group_boundary(
        template, exlude=[3, 10, 13])

    np.save('results/bi.npy', boundary_indices)

    # boundary condition
    b = np.concatenate([boundary_indices] + exlude_indices_list)
    num_bi = len(boundary_indices)
    bc = np.zeros((len(b), num_bi))
    bc[:num_bi] = np.eye(num_bi)

    # ensure exluded part can only be controlled by exlude_control_indices
    base_num = num_bi
    for i in range(len(exlude_control_indices)):
        num_icrea = len(exlude_indices_list[i])
        bc[base_num:base_num + num_icrea, exlude_control_indices[i]] = 1
        base_num += num_icrea

    bbw = igl.BBW()
    weight = bbw.solve(template.vertices, template.faces, b, bc)

    np.save('results/bbw.npy', weight)
