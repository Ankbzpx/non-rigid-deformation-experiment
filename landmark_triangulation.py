import numpy as np
import cv2
import matplotlib.pyplot as plt
from functools import partial
import os
import open3d as o3d

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from jax import numpy as jnp, vmap, jit
from jaxopt import LevenbergMarquardt

import json
import igl

import polyscope as ps
from icecream import ic


# Landmark in pixel coordinate
def detect_facelandmark(detector, image):
    H, W, _ = image.numpy_view().shape
    face_landmarks = detector.detect(image).face_landmarks

    if len(face_landmarks) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    else:
        # Ignore depth
        return np.stack(
            [np.array([lm.x * W, lm.y * H]) for lm in face_landmarks[0]])


def parse_calibration_data(json_path):
    with open(json_path, 'r') as f:
        calib_data = json.load(f)

        fx = calib_data['fl_x']
        fy = calib_data['fl_y']
        cx = calib_data['cx']
        cy = calib_data['cy']

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        frame_list = calib_data['frames']

        frame_id_list = []
        T_list = []

        for frame in frame_list:
            frame_id_list.append(frame['file_path'])
            T_list.append(np.array(frame['transform_matrix']))

        return K, frame_id_list, T_list


# TODO: Use robust loss
@jit
def reproj_loss(P, lms, lms_3d, ord=1):
    lms_3d_homo = jnp.hstack([lms_3d, np.ones(len(lms_3d))[:, None]])
    lms_proj_homo = lms_3d_homo @ P.T
    lms_proj = lms_proj_homo[:, :2] / lms_proj_homo[:, -1][:, None]
    return vmap(jnp.linalg.norm, in_axes=(0, None))(lms - lms_proj, ord).mean()


if __name__ == '__main__':
    # Model from: https://developers.google.com/mediapipe/solutions/vision/face_landmarker/index#models
    base_options = python.BaseOptions(
        model_asset_path='mediapipe/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    detect_lms = partial(detect_facelandmark, detector=detector)

    capture_root_folder = "scan_data/capture"
    capture_id_list = sorted(os.listdir(capture_root_folder))

    recon_folder = "scan_data/recon"

    for capture_id in capture_id_list:
        ic(capture_id)

        capture_folder = os.path.join(capture_root_folder, capture_id)
        json_path = os.path.join(capture_folder, "transforms_train.json")

        K, frame_id_list, T_list = parse_calibration_data(json_path)

        lms_list = []
        P_list = []
        image_list = []

        for frame_id in frame_id_list[9:21]:
            image_path = os.path.join(capture_folder, f"{frame_id}.png")
            image = mp.Image.create_from_file(image_path)
            lms = detect_lms(image=image)

            if len(lms) == 478:
                # Model matrix
                T = T_list[int(frame_id)]
                # Projection matrix
                P = K @ np.linalg.inv(T)[:3]

                lms_list.append(lms)
                P_list.append(P)
                image_list.append(np.copy(image.numpy_view()))

        # Pick two middle lm detections for triangulation initialization
        init_id_0 = len(lms_list) // 2 - 1
        init_id_1 = len(lms_list) // 2 + 1

        lms_homo = cv2.triangulatePoints(P_list[init_id_0], P_list[init_id_1],
                                         lms_list[init_id_0].T,
                                         lms_list[init_id_1].T).T
        lms_3d = lms_homo[:, :3] / lms_homo[:, -1][:, None]

        # Nonlinear optimization
        lms = jnp.stack(lms_list)
        P = jnp.stack(P_list)
        lms_3d = jnp.array(lms_3d)

        @jit
        def loss_residual(lms_3d):
            return vmap(reproj_loss,
                        in_axes=(0, 0, None))(P, lms, lms_3d.reshape(-1, 3))

        loss = loss_residual(lms_3d).mean()
        ic(loss)

        LM = LevenbergMarquardt(loss_residual)
        lms_3d_refined = LM.run(lms_3d.reshape(-1,)).params.reshape(-1, 3)

        loss_refined = loss_residual(lms_3d_refined).mean()
        ic(loss_refined)

        # Visualization with MVS results
        # recon_path = os.path.join(recon_folder, f"{capture_id}.ply")
        # V, F = igl.read_triangle_mesh(recon_path)

        # pc_gt_o3d = o3d.io.read_point_cloud(
        #     os.path.join(capture_folder, "points3d.ply"))
        # pc_gt = np.asarray(pc_gt_o3d.points)

        # ps.init()
        # ps.register_point_cloud('lms_3d', lms_3d)
        # ps.register_point_cloud('lms_3d_refined', lms_3d_refined)
        # ps.register_point_cloud('pc_gt', pc_gt)
        # ps.register_surface_mesh('recon', V, F)
        # ps.show()

        np.save(os.path.join(capture_folder, 'lms_478.npy'), lms_3d_refined)
