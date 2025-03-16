import argparse
import igl
import json
import numpy as np

from mesh_helper import read_obj, write_obj, load_face_landmarks, OBJMesh
from arap import AsRigidAsPossible, SymmetricPointToPlane
# from non_rigid_deformation import closest_point_triangle_match, closest_point_on_triangle
import trimesh
from functools import partial
from tqdm import tqdm
from corase_match_svd import match_correspondence
import copy
from functools import partial
import fcpw

# Debug
import polyscope as ps
from icecream import ic


# TODO: GPU query
class ClosestPointQueryHelper:

    def __init__(self, V, F):
        self.scene = fcpw.scene_3D()
        self.scene.set_object_count(1)

        self.scene.set_object_vertices(V, 0)
        self.scene.set_object_triangles(F, 0)

        aggregate_type = fcpw.aggregate_type.bvh_surface_area
        build_vectorized_cpu_bvh = False
        print_stats = False
        reduce_memory_footprint = False
        self.scene.build(aggregate_type, build_vectorized_cpu_bvh, print_stats,
                         reduce_memory_footprint)

    # TODO: Allow vector max_squared_radius to improve performance
    def query(self, query_pos, max_squared_radius=np.inf):
        interactions = fcpw.interaction_3D_list()
        self.scene.find_closest_points(
            query_pos, max_squared_radius * np.ones(len(query_pos)),
            interactions)

        return np.array([i.p for i in interactions
                        ]), np.array([i.primitive_index for i in interactions
                                     ]), np.array([i.d for i in interactions])


def load_template(template_path, template_lm_path):
    template = read_obj(template_path)
    lms_data = np.array(json.load(open(template_lm_path)))
    lms_fid = np.int64(lms_data[:, 0])
    lms_uv = np.float64(lms_data[:, 1:])

    lms_bary_coords = np.stack(
        [lms_uv[:, 0], lms_uv[:, 1], 1 - lms_uv[:, 0] - lms_uv[:, 1]], -1)

    template_lms = load_face_landmarks(template, template_lm_path)

    return {
        "template": template,
        "lms_fid": lms_fid,
        "lms_bary_coords": lms_bary_coords,
        "template_lms": template_lms
    }


def solve_deform(template: OBJMesh,
                 lms_fid,
                 lms_bary_coords,
                 template_lms,
                 scan_path,
                 scan_lms_path,
                 use_symmetry=True):
    scan: trimesh.Trimesh = trimesh.load(scan_path,
                                         process=False,
                                         maintain_order=True)
    scan_lms_data = json.load(open(scan_lms_path))
    scan_lms = np.stack(
        [np.array([lm['x'], lm['y'], lm['z']]) for lm in scan_lms_data])

    R, s, t = match_correspondence(template_lms, scan_lms)

    V = s * template.vertices @ R.T + t
    F = template.faces

    # Use barycentric interpolated vertex normal for lm normals
    template_lm_normals = (
        lms_bary_coords[..., None] *
        template.vertex_normals[template.faces][lms_fid]).sum(1)

    face_groups = np.split(template.faces, template.polygon_groups[1:])

    # 3 Tongue
    # 6 Ears
    # 10 Right Eye
    # 13 Left Eye
    # 27 head
    exclude_indices = np.unique(
        np.concatenate([
            face_groups[3].reshape(-1), face_groups[10].reshape(-1),
            face_groups[13].reshape(-1)
        ]))

    closest_helper = ClosestPointQueryHelper(np.array(scan.vertices),
                                             np.array(scan.faces))
    scan_lms, _, _ = closest_helper.query(scan_lms)

    # if use_symmetry:
    #     sym_p2p = SymmetricPointToPlane(V,
    #                                     F,
    #                                     b_fid=lms_fid,
    #                                     b_bary_coords=lms_bary_coords)
    #     V_init = sym_p2p.solve(V, template_lm_normals, scan_lms,
    #                            scan_lms_normals)
    # else:
    # In this case, it is point to point distance
    arap = AsRigidAsPossible(V,
                             F,
                             b_fid=lms_fid,
                             b_bary_coords=lms_bary_coords,
                             smooth_rotation=True,
                             soft_weight=1,
                             soft_exclude=False)
    V_init = arap.solve(V, scan_lms)

    # ps.init()
    # ps.register_surface_mesh('V_init', V_init, F)
    # ps.register_surface_mesh('scan', scan.vertices, scan.faces)
    # ps.show()
    # exit()

    lm_dist = np.linalg.norm(scan_lms - (s * template_lms @ R.T + t), axis=1)
    dist_thr_median = np.median(lm_dist)

    face_normals_scan = scan.face_normals

    def get_closest_match(V: np.ndarray, dist_thr, cos_thr):
        V_matched, fid, dist = closest_helper.query(V)

        # Assume full vertices here
        VN = igl.per_vertex_normals(V, F)
        VN_matched = face_normals_scan[fid]

        cos = np.einsum('ab,ab->a', VN_matched, VN)
        valid_mask = (dist < dist_thr) & (cos > cos_thr)
        valid_mask[exclude_indices] = False

        B = np.where(valid_mask)[0]
        P = V[valid_mask]
        N_p = VN[valid_mask]
        Q = V_matched[valid_mask]
        N_q = VN_matched[valid_mask]

        # ps.init()
        # ps.register_point_cloud("P", P).add_vector_quantity("N_p",
        #                                                     N_p,
        #                                                     enabled=True)
        # ps.register_point_cloud("Q", Q).add_vector_quantity("N_q",
        #                                                     N_q,
        #                                                     enabled=True)
        # ps.show()

        return B, N_p, Q, N_q

    max_iter = 10
    dist_thrs = np.linspace(50 * dist_thr_median, 25 * dist_thr_median,
                            max_iter)
    cos_thrs = np.linspace(0.5, 0.75, max_iter)

    V_arap = V_init
    for i in tqdm(range(max_iter)):
        dist_thr = dist_thrs[i]
        cos_thr = cos_thrs[i]
        B, N_p, Q, N_q = get_closest_match(V_arap,
                                           dist_thr=dist_thr,
                                           cos_thr=cos_thr)

        if use_symmetry:
            sym_p2p = SymmetricPointToPlane(V, F, b_vid=B)
            V_arap = sym_p2p.solve(V_arap,
                                   N_p,
                                   Q,
                                   N_q,
                                   robust_weight=False,
                                   w_arap=1,
                                   w_sr=1e-4)
        else:
            arap = AsRigidAsPossible(V,
                                     F,
                                     b_vid=B,
                                     b_fid=lms_fid,
                                     b_bary_coords=lms_bary_coords,
                                     soft_weight=2.0,
                                     soft_exclude=False,
                                     smooth_rotation=True)
            V_arap = arap.solve(V_arap, np.vstack([Q, scan_lms]))

        # ps.init()
        # ps.register_surface_mesh('V_arap', V_arap, F)
        # ps.register_surface_mesh('scan', scan.vertices, scan.faces)
        # ps.show()

    model_matched = copy.deepcopy(template)
    model_matched.vertices = V_arap

    return model_matched


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARAP + Non-rigid ICP.')
    parser.add_argument('--template_pg_path',
                        type=str,
                        default='templates/template_pg.obj')
    parser.add_argument('--template_pg_lm_path',
                        type=str,
                        default='templates/template_pg_lms.txt')
    parser.add_argument('--scan_path', type=str, default='scan_data/scan.ply')
    parser.add_argument('--scan_lms_path',
                        type=str,
                        default='scan_data/scan_3d.txt')
    parser.add_argument('--match_save_path',
                        type=str,
                        default='results/scan_match_arap_nicp.obj')
    parser.add_argument('--remove_interior',
                        action='store_true',
                        help='Remove eye ball and tongue')
    args = parser.parse_args()

    # Template mesh with pre-specified polygon groups
    template_pg_path = args.template_pg_path
    template_pg_lm_path = args.template_pg_lm_path
    scan_path = args.scan_path
    scan_lms_path = args.scan_lms_path
    match_save_path = args.match_save_path
    remove_interior = args.remove_interior

    solve_deform_partial = partial(
        solve_deform, **load_template(template_pg_path, template_pg_lm_path))

    model_save = solve_deform_partial(scan_path=scan_path,
                                      scan_lms_path=scan_lms_path)

    # WARNING: Current implementation is surface deform only.
    # It is prone to self-intersection and can not guarantee the interior correctness
    # Here is the option to remove them
    # FIXME: Implement volume ARAP
    if remove_interior:
        fid = np.arange(len(model_save.faces_quad))
        face_group_ids = np.split(fid, model_save.polygon_groups_quad[1:])

        select_faces_mask = np.ones(len(model_save.faces_quad))
        select_faces_mask[face_group_ids[3]] = 0.
        select_faces_mask[face_group_ids[10]] = 0.
        select_faces_mask[face_group_ids[13]] = 0.
        select_faces_mask = select_faces_mask.astype(bool)

        select_faces = model_save.faces_quad[select_faces_mask]
        select_face_uvs = model_save.face_uvs_idx_quad[select_faces_mask]

        vertex_ids, verts_unique_inverse = np.unique(select_faces,
                                                     return_inverse=True)
        model_save.vertices = model_save.vertices[vertex_ids]

        uv_ids, uv_unique_inverse = np.unique(select_face_uvs,
                                              return_inverse=True)
        model_save.uvs = model_save.uvs[uv_ids]

        model_save.faces_quad = np.arange(
            len(vertex_ids))[verts_unique_inverse].reshape(-1, 4)
        model_save.face_uvs_idx_quad = np.arange(
            len(uv_ids))[uv_unique_inverse].reshape(-1, 4)

    write_obj(match_save_path, model_save)
