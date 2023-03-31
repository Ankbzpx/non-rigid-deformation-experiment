from collections import namedtuple
import numpy as np
import os
from PIL import Image
import trimesh

# support vertex color
return_type = namedtuple('return_type', [
    'vertices', 'faces', 'faces_quad', 'uvs', 'face_uvs_idx',
    'face_uvs_idx_quad', 'materials', 'vertex_colors', 'vertex_normals',
    'extras'
])


# TODO: Support multiple materials, face normals
# Modified from https://kaolin.readthedocs.io/en/latest/modules/kaolin.io.obj.html#module-kaolin.io.obj for vertex color support
def read_obj(path, warning=False):
    r"""
    Load obj, support quad mesh, vertex color
    """
    vertices = []
    faces = []
    uvs = []
    vertex_normals = []
    vertex_colors = []
    face_uvs_idx = []
    mtl_path = None
    materials = []
    extras = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.split()
            if len(data) == 0:
                continue
            if data[0] == 'v':
                vertices.append(np.float64(data[1:4]))
                vertex_colors.append(np.float64(data[4:]))
            elif data[0] == 'vt':
                uvs.append(np.float64(data[1:3]))
            elif data[0] == 'vn':
                vertex_normals.append(np.float64(data[1:]))
            elif data[0] == 'f':
                data = [da.split('/') for da in data[1:]]
                faces.append([int(d[0]) for d in data])
                if len(data[1]) > 1 and data[1][1] != '':
                    face_uvs_idx.append([int(d[1]) for d in data])
                else:
                    face_uvs_idx.append([0] * len(data))
            elif data[0] == 'mtllib':
                extras.append(line)
                mtl_path = os.path.join(os.path.dirname(path), data[1])
                if os.path.exists(mtl_path):
                    with open(mtl_path) as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.startswith("map_Kd"):
                                texture_file = line.split(' ')[-1]
                                texture_file = texture_file.strip('\n')
                                texture_file_path = os.path.join(
                                    os.path.dirname(path), texture_file)
                                img = Image.open(texture_file_path)
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                materials.append(img)
                else:
                    if warning:
                        print(
                            f"Failed to load material, {data[1]} doesn't exist")

    vertices = np.stack(vertices).reshape(-1, 3)

    if len(uvs) == 0:
        uvs = None
    else:
        uvs = np.stack(uvs).reshape(-1, 2)

    if len(vertex_colors) == 0:
        vertex_colors = None
    else:
        vertex_colors = np.stack(vertex_colors).reshape(-1, 3)

    if len(face_uvs_idx) == 0:
        face_uvs_idx = None
    else:
        face_uvs_idx = np.int64(face_uvs_idx) - 1

    faces = np.int64(faces) - 1

    faces_quad = None
    face_uvs_idx_quad = None
    face_size = faces.shape[1]
    if face_size == 4:
        faces_quad = faces
        faces = np.vstack([
            np.vstack([[face[0], face[1], face[2]],
                       [face[0], face[2], face[3]]]) for face in faces_quad
        ])
        if face_uvs_idx is not None:
            face_uvs_idx_quad = face_uvs_idx
            face_uvs_idx = np.vstack([
                np.vstack([[face_uv_idx[0], face_uv_idx[1], face_uv_idx[2]],
                           [face_uv_idx[0], face_uv_idx[2], face_uv_idx[3]]])
                for face_uv_idx in face_uvs_idx_quad
            ])

    if len(vertex_normals) == 0 or len(vertex_normals) != len(vertices):
        if warning:
            print("Obj doesn't contain vertex_normals, compute using trimesh")
        vertex_normals = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
            maintain_order=True,
        ).vertex_normals
        assert not np.isnan(vertex_normals).any()
        assert len(vertex_normals) == len(vertices)
    else:
        vertex_normals = np.stack(vertex_normals).reshape(-1, 3)

    return return_type(vertices, faces, faces_quad, uvs, face_uvs_idx,
                       face_uvs_idx_quad, materials, vertex_colors,
                       vertex_normals, extras)


# TODO: Support multiple materials write, face normal
def write_obj(filename, mesh: return_type):
    r"""
    Write obj, support quad mesh
    """
    with open(filename, 'w', encoding='utf-8') as obj_file:
        for extra in mesh.extras:
            if extra.split()[0] == 'mtllib':
                obj_file.write(extra)
        if mesh.vertex_colors is None:
            for v in mesh.vertices:
                obj_file.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        else:
            for v, vc in zip(mesh.vertices, mesh.vertex_colors):
                obj_file.write('v %f %f %f %f %f %f\n' %
                               (v[0], v[1], v[2], vc[0], vc[1], vc[2]))
        for vt in mesh.uvs:
            obj_file.write('vt %f %f\n' % (vt[0], vt[1]))

        if mesh.faces_quad is not None:
            if mesh.face_uvs_idx_quad is None:
                for f in mesh.faces_quad:
                    obj_file.write('f %d %d %d %d\n' %
                                   (f[0] + 1, f[1] + 1, f[2] + 1, f[3] + 1))
            else:
                for f, f_uv in zip(mesh.faces_quad, mesh.face_uvs_idx_quad):
                    obj_file.write(
                        'f %d/%d %d/%d %d/%d %d/%d\n' %
                        (f[0] + 1, f_uv[0] + 1, f[1] + 1, f_uv[1] + 1, f[2] + 1,
                         f_uv[2] + 1, f[3] + 1, f_uv[3] + 1))

        else:
            if mesh.face_uvs_idx is None:
                for f in mesh.faces:
                    obj_file.write('f %d %d %d\n' %
                                   (f[0] + 1, f[1] + 1, f[2] + 1))
            else:
                for f, f_uv in zip(mesh.faces, mesh.face_uvs_idx):
                    obj_file.write('f %d/%d %d/%d %d/%d\n' %
                                   (f[0] + 1, f_uv[0] + 1, f[1] + 1,
                                    f_uv[1] + 1, f[2] + 1, f_uv[2] + 1))
