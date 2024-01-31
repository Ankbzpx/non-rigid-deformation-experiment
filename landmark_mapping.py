import numpy as np
from PIL import Image
import cv2

from mesh_helper import read_obj

# https://github.com/pengHTYX/VisRecon/tree/0f7f343771e9329cbf1cebe6adcd965c11c28e06/vis_fuse_utils
import vis_fuse_utils

import torch
import moderngl
import math
from pyrr import Matrix44

import igl
import mediapipe as mp
from landmark_triangulation import mp_facelandmark_detect_init, refine_triangulation
from non_rigid_deformation import closest_point_on_triangle

import polyscope as ps
from icecream import ic


class PrtRenderTargetUV:

    def __init__(self, ctx: moderngl.Context, width, height):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader='''
                #version 330

                uniform mat4 mvp;

                layout (location = 0) in vec3 in_pos;
                layout (location = 1) in vec2 in_uv;
                layout (location = 2) in float in_sh[9];

                out vec2 uv;
                out float sh[9];

                void main() {
                    gl_Position = mvp * vec4(in_pos, 1.0);
                    uv = in_uv;
                    sh = in_sh;
                }
            ''',
                                fragment_shader='''
                #version 330

                uniform sampler2D texture1;
                uniform float env_sh[27];

                in vec2 uv;
                in float sh[9];

                layout (location = 0) out vec4 light_color;
                layout (location = 1) out vec4 textured_color;
                layout (location = 2) out vec4 albedo_color;

                vec4 gammaCorrection(vec4 vec, float g) {
                    return vec4(pow(vec.x, 1.0/g), pow(vec.y, 1.0/g), pow(vec.z, 1.0/g), vec.w);
                }

                void main() {
                    light_color = vec4(0.0);
                    for (int i = 0; i < 9; i++) {
                        light_color.x += sh[i] * env_sh[i];
                        light_color.y += sh[i] * env_sh[9 + i];
                        light_color.z += sh[i] * env_sh[18 + i];
                    }
                    light_color.w = 1.0;
                    light_color = gammaCorrection(light_color, 2.2);
                    vec4 uv_color = texture(texture1, uv);
                    textured_color = light_color * uv_color;
                    albedo_color = uv_color;
                }
            ''')

        color_attachments = [
            ctx.renderbuffer((width, height), 4, samples=8),
            ctx.renderbuffer((width, height), 4, samples=8),
            ctx.renderbuffer((width, height), 4, samples=8)
        ]
        self.fbo = ctx.framebuffer(color_attachments=color_attachments,
                                   depth_attachment=ctx.depth_renderbuffer(
                                       (width, height), samples=8))
        color_attachments2 = [
            ctx.texture((width, height), 4),
            ctx.texture((width, height), 4),
            ctx.texture((width, height), 4)
        ]
        self.fbo2 = ctx.framebuffer(color_attachments=color_attachments2,
                                    depth_attachment=ctx.depth_renderbuffer(
                                        (width, height)))
        self.vao = None

    def build_vao(self, per_face_vertices, per_face_uv, per_face_prt,
                  texture_image):
        vbo_vert = self.ctx.buffer(per_face_vertices.astype('f4'))
        vbo_uv = self.ctx.buffer(per_face_uv.astype('f4'))
        vbo_prt = self.ctx.buffer(per_face_prt.astype('f4'))

        self.vao = self.ctx.vertex_array(self.prog, [(vbo_vert, '3f', 'in_pos'),
                                                     (vbo_uv, '2f', 'in_uv'),
                                                     (vbo_prt, '9f', 'in_sh')])
        self.texture = self.ctx.texture(texture_image.size, 3,
                                        texture_image.tobytes())
        self.texture.build_mipmaps(max_level=3)

    def render(self, mvp, sh):
        self.prog['mvp'].write(mvp)
        self.prog['env_sh'].write(sh)

        self.fbo.use()
        self.fbo.clear(red=1., green=1., blue=1.)
        self.texture.use()
        self.vao.render()

        self.ctx.copy_framebuffer(self.fbo2, self.fbo)

        light_color_data = self.fbo2.read(components=3, attachment=0)
        image_light_color = Image.frombytes('RGB', self.fbo2.size,
                                            light_color_data).transpose(
                                                Image.Transpose.FLIP_TOP_BOTTOM)

        color_data = self.fbo2.read(components=3, attachment=1)
        image_color = Image.frombytes('RGB', self.fbo2.size,
                                      color_data).transpose(
                                          Image.Transpose.FLIP_TOP_BOTTOM)

        albedo_data = self.fbo2.read(components=3, attachment=2)
        image_albedo = Image.frombytes('RGB', self.fbo2.size,
                                       albedo_data).transpose(
                                           Image.Transpose.FLIP_TOP_BOTTOM)

        return np.array(image_light_color), np.array(image_albedo), np.array(
            image_color)


def normalize_aabb(V, scale=0.95):
    V = np.copy(V)

    V_aabb_max = V.max(0, keepdims=True)
    V_aabb_min = V.min(0, keepdims=True)
    V_center = 0.5 * (V_aabb_max + V_aabb_min)
    V -= V_center
    scale = (V_aabb_max - V_center).max() / scale
    V /= scale

    return V


# Modified from: https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
def cartesian_to_sphere(xyz):
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    theta = np.arctan2(np.sqrt(xy),
                       xyz[:,
                           2])    # for elevation angle defined from Z-axis down
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return phi, theta


# Modified from: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
def fibonacci_sphere(samples):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))    # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2    # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)    # radius at y

        theta = phi * i    # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    xyz = np.array(points)
    phi, theta = cartesian_to_sphere(xyz)

    return xyz, phi, theta


def factratio(N, D):
    if N >= D:
        prod = 1.0
        for i in range(D + 1, N + 1):
            prod *= i
        return prod
    else:
        prod = 1.0
        for i in range(N + 1, D + 1):
            prod *= i
        return 1.0 / prod


def KVal(M, L):
    return math.sqrt(((2 * L + 1) / (4 * math.pi)) * (factratio(L - M, L + M)))


def AssociatedLegendre(M, L, x):
    if M < 0 or M > L or np.max(np.abs(x)) > 1.0:
        return np.zeros_like(x)

    pmm = np.ones_like(x)
    if M > 0:
        somx2 = np.sqrt((1.0 + x) * (1.0 - x))
        fact = 1.0
        for i in range(1, M + 1):
            pmm = -pmm * fact * somx2
            fact = fact + 2

    if L == M:
        return pmm
    else:
        pmmp1 = x * (2 * M + 1) * pmm
        if L == M + 1:
            return pmmp1
        else:
            pll = np.zeros_like(x)
            for i in range(M + 2, L + 1):
                pll = (x * (2 * i - 1) * pmmp1 - (i + M - 1) * pmm) / (i - M)
                pmm = pmmp1
                pmmp1 = pll
            return pll


def SphericalHarmonic(M, L, theta, phi):
    if M > 0:
        return math.sqrt(2.0) * KVal(M, L) * np.cos(
            M * phi) * AssociatedLegendre(M, L, np.cos(theta))
    elif M < 0:
        return math.sqrt(2.0) * KVal(-M, L) * np.sin(
            -M * phi) * AssociatedLegendre(-M, L, np.cos(theta))
    else:
        return KVal(0, L) * AssociatedLegendre(0, L, np.cos(theta))


def getSHCoeffs(order, phi, theta):
    shs = []
    for n in range(0, order + 1):
        for m in range(-n, n + 1):
            s = SphericalHarmonic(m, n, theta, phi)
            shs.append(s)

    return np.stack(shs, 1)


# Adapted from https://github.com/shunsukesaito/PIFu/blob/master/lib/renderer/glm.py
def get_ortho_matrix(left, right, bottom, top, zNear, zFar):
    res = np.identity(4, dtype=np.float32)
    res[0][0] = 2 / (right - left)
    res[1][1] = 2 / (top - bottom)
    res[2][2] = -2 / (zFar - zNear)
    res[3][0] = -(right + left) / (right - left)
    res[3][1] = -(top + bottom) / (top - bottom)
    res[3][2] = -(zFar + zNear) / (zFar - zNear)
    return res.T


def get_persp_matrix(fx, fy, cx, cy, zNear, zFar):
    return np.array([[fx, 0, -cx, 0], [0, fy, -cy, 0],
                     [0, 0, zNear + zFar, zNear * zFar], [0, 0, -1, 0]])


if __name__ == '__main__':
    ps.init()

    width = 512
    height = 512
    fx = 550
    fy = 550
    cx = 0.5 * 512
    cy = 0.5 * 512
    z_near = 0.01
    z_far = 100.0

    T_gl_cv = np.eye(4)
    T_gl_cv[1, 1] = -1
    T_gl_cv[2, 2] = -1

    ortho = get_ortho_matrix(0, width, height, 0, z_near, z_far)
    proj = ortho @ get_persp_matrix(fx, fy, cx, height - cy, z_near, z_far)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    detect_facelandmark = mp_facelandmark_detect_init()

    # https://github.com/shunsukesaito/PIFu/blob/master/env_sh.npy
    sh = np.load('templates/env_sh.npy')[0, ...]
    sh = sh.T.reshape(-1,).astype('f4')

    template = read_obj('templates/template_pg.obj')
    texture = Image.open('templates/FacialSheet_fake_paint.png')
    template.materials.append(texture)

    # Scale should not affect mapping, because landmarks are stored in barycentric coordinate
    template.vertices = normalize_aabb(template.vertices)

    vis_sample_size = 64
    order = 2
    dirs, phi, theta = fibonacci_sphere(vis_sample_size)
    SH = getSHCoeffs(order, phi, theta)

    vis = np.logical_not(
        vis_fuse_utils.sample_occlusion_embree(
            template.vertices, template.faces,
            template.vertices + 1e-3 * template.vertex_normals,
            dirs).astype(bool))
    geo_term = np.clip(np.einsum("ik,jk->ij", template.vertex_normals, dirs), 0,
                       1)
    prt = np.einsum("ij,ij,jk->ik", vis, geo_term,
                    SH) * 4.0 * np.pi / vis_sample_size

    per_face_vertices = template.vertices[template.faces].reshape(-1, 3)
    per_face_uv = template.uvs[template.face_uvs_idx].reshape(-1, 2)
    per_face_prt = prt[template.faces].reshape(-1, 9)
    texture_image = template.materials[0].transpose(
        Image.Transpose.FLIP_TOP_BOTTOM)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.gc_mode = "auto"

    prt_uv_target = PrtRenderTargetUV(ctx, width, height)
    prt_uv_target.build_vao(per_face_vertices, per_face_uv, per_face_prt,
                            texture_image)

    # eye_list = np.array([[0.5, 0.35, -2], [-0.5, 0.35, -2], [0.5, -0.35, -2],
    #                      [-0.5, -0.35, -2], [1, 0.7, -3], [-1, 0.7, -3],
    #                      [1, -0.7, -3], [-1, -0.7, -3], [0.25, -0.7, -1.5],
    #                      [-0.25, -0.7, -1.5], [0.25, 0.7, -1.5],
    #                      [-0.25, 0.7, -1.5]])
    eye_list = np.array([[0.5, 0.35, -2], [-0.5, 0.35, -2], [0.0, 0.35, -2]])

    lms_list = []
    P_list = []
    image_list = []
    T_list = []

    for eye in eye_list:
        # Column major
        mv = np.ascontiguousarray(Matrix44.look_at(eye, [0, 0, 0], [0, 1, 0]).T)
        mvp = (np.ascontiguousarray((proj @ mv @ T_gl_cv).T).astype('f4'))

        image_color = prt_uv_target.render(mvp, sh)[-1]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_color)

        lms = detect_facelandmark(image)

        if len(lms) == 478:
            # IMPORTANT: The eye is in GL coordinate system
            mv_cv = np.ascontiguousarray(
                Matrix44.look_at(T_gl_cv[:3, :3] @ eye, [0, 0, 0], [0, 1, 0]).T)

            P = K @ mv_cv[:3]

            lms_list.append(lms)
            P_list.append(P)
            image_list.append(image_color)
            T_list.append(mv_cv)

    lms_homo = cv2.triangulatePoints(P_list[0], P_list[1], lms_list[0].T,
                                     lms_list[1].T).T
    lms_3d = lms_homo[:, :3] / lms_homo[:, -1][:, None]
    lms_3d_refined = np.float64(refine_triangulation(lms_list, P_list, lms_3d))

    FN = igl.per_face_normals(template.vertices, template.faces,
                              np.array([0.0, 1.0, 0.0])[:, None])

    lms_proj = closest_point_on_triangle(
        torch.from_numpy(lms_3d_refined).float().cuda(),
        torch.from_numpy(template.vertices[template.faces]).float().cuda(),
        torch.from_numpy(FN).float().cuda())[0].detach().cpu().numpy()

    # exit()
    ps.register_surface_mesh('template', template.vertices, template.faces)
    ps.register_point_cloud('lms_3d', lms_3d[:468])
    ps.register_point_cloud('lms_3d_refined', lms_3d_refined[:468])
    ps.register_point_cloud('lms_proj', lms_proj[:468])
    ps.show()
