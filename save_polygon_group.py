from mesh_helper import read_obj, write_obj
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
import json
from icecream import ic


def sample_color(img, uv, mode='bilinear'):
    img_height = img.shape[0]
    img_width = img.shape[1]

    # UV coordinates should be (n, 2) float
    uv = np.asanyarray(uv, dtype=np.float64)

    # get texture image pixel positions of UV coordinates
    x = (uv[:, 0] * (img_width - 1))
    y = ((1 - uv[:, 1]) * (img_height - 1))

    # convert to int and wrap to image
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # clip
    x0 = np.clip(x0, 0, img_width - 1)
    x1 = np.clip(x1, 0, img_width - 1)
    y0 = np.clip(y0, 0, img_height - 1)
    y1 = np.clip(y1, 0, img_height - 1)

    # bilinear interpolation
    img = np.asanyarray(img)
    c0 = img[y0, x0]
    c1 = img[y1, x0]
    c2 = img[y0, x1]
    c3 = img[y1, x1]

    if c0.ndim == 1:
        c0 = c0[:, None]
        c1 = c1[:, None]
        c2 = c2[:, None]
        c3 = c3[:, None]

    w0 = ((x1 - x) * (y1 - y))[:, None]
    w1 = ((x1 - x) * (y - y0))[:, None]
    w2 = ((x - x0) * (y1 - y))[:, None]
    w3 = ((x - x0) * (y - y0))[:, None]

    if mode == 'bilinear':
        colors = c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3
    elif mode == 'nearest':
        nearest_idx = np.argmin(np.stack([w0, w1, w2, w3]), axis=0)[None, ...]
        colors = np.stack([c0, c1, c2, c3])
        colors = np.take_along_axis(colors, nearest_idx, axis=0)[0]

    colors = np.array(np.squeeze(colors))
    return colors


if __name__ == '__main__':

    template = read_obj('results/template_icp_match.obj')

    per_face_uv = template.uvs[template.face_uvs_idx]
    barycenter_uv = np.average(per_face_uv, 1)

    facial_sheet = np.array(Image.open('data/FacialSheet_yellow.png')) / 255.
    uv_color = sample_color(facial_sheet, barycenter_uv, mode='nearest')

    unique_color, unique_counts = np.unique(uv_color,
                                            return_counts=True,
                                            axis=0)
    unique_color = unique_color[unique_counts > 10]

    color_idx = np.argmin(cdist(uv_color, unique_color), 1)

    sort_idx = np.argsort(color_idx[::2])
    sort_idx = np.stack([2 * sort_idx, 2 * sort_idx + 1], -1).reshape(-1)

    template_lms_data = np.array(json.load(open('data/mastermodel_3d.txt')))
    template_lms_fid = np.int64(template_lms_data[:, 0])
    template_lms_fid = np.array([
        np.argwhere(sort_idx == lms_fid) for lms_fid in template_lms_fid
    ]).flatten()[:, None]
    template_lms_uv = np.float64(template_lms_data[:, 1:]).astype(object)
    template_lms_data = np.hstack([template_lms_fid, template_lms_uv])

    with open('results/template_icp_match_lms.txt', 'w') as f:
        json.dump(template_lms_data.tolist(), f)

    write_obj('results/template_icp_match.obj', template, color_idx)
