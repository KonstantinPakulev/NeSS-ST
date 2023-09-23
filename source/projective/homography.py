import numpy as np
from scipy.stats import uniform, bernoulli

import torch

from source.projective.utils import to_cartesian, to_homogeneous, grid_sample
from source.utils.common_utils import create_coord_grid, get_rect, get_rect_mask


def warp_h(image_point, h12, shift_scale1, shift_scale2):
    """
    :param image_point: B x N x 2
    :param h12: B x 3 x 3
    :param shift_scale1: B x 4
    :param shift_scale2: B x 4
    """
    b, n = image_point.shape[:2]

    h12 = shift_scale_h(h12, shift_scale1, shift_scale2)

    # Convert grid to homogeneous coordinates
    image_point = to_homogeneous(image_point).permute(0, 2, 1)  # B x 3 x N
    w_image_point = (h12 @ image_point).permute(0, 2, 1)  # B x N x 3
    
    # Convert coordinates from homogeneous to cartesian
    w_image_point = to_cartesian(w_image_point).view(b, -1, 2)

    return w_image_point


"""
Warping functions H
"""


def warp_image_h(image1, h_data,  mode):
    """
    :param image1: B x C x iH x iW
    :param h_data: h_data with orientation 2->1
    :param mode: str
    :return w_image: B x C x H x W
    """
    b, _, h, w = h_data.image_shape2

    grid2 = create_coord_grid(h_data.image_shape2).to(image1.device)
    w_grid2 = warp_h(grid2.view(b, -1, 2), h_data.h2, h_data.shift_scale2, h_data.shift_scale1).view(b, h, w, 2)

    if 'i' in mode and 'm' in mode:
        w_image1 = grid_sample(image1, w_grid2)
        vis_mask2 = get_rect_mask(w_grid2[..., [1, 0]], get_rect(image1.shape)).unsqueeze(1)

        return w_image1, vis_mask2

    elif 'i' in mode:
        w_image1 = grid_sample(image1, w_grid2)

        return w_image1

    elif 'm' in mode:
        vis_mask2 = get_rect_mask(w_grid2[..., [1, 0]], get_rect(image1.shape)).unsqueeze(1)

        return vis_mask2

    else:
        raise NotImplementedError


def warp_points_h(points1, h_data, mode):
    """
    :param points1: B x N x 2, coordinates order is (y, x)
    :param h_data: h_data with orientation 1->2
    :return B x N x 2
    """
    # Because warping operates on x,y coordinates we need to swap them places
    points1 = points1[..., [1, 0]]
    w_points1 = warp_h(points1, h_data.h1, h_data.shift_scale1, h_data.shift_scale2)

    w_points1 = w_points1[..., [1, 0]]

    if 'p' in mode and 'm' in mode:
        w_points_mask1 = get_rect_mask(w_points1, get_rect(h_data.image_shape2))

        return w_points1, w_points_mask1

    elif 'p' in mode:
        return w_points1

    else:
        raise NotImplementedError


"""
Homography transformation
"""


def shift_scale_h(h, shift_scale1, shift_scale2):
    b = h.shape[0]

    t1 = torch.eye(3, device=h.device).repeat(b, 1, 1)
    t1[:, :2, 2] += shift_scale1[:, [1, 0]]

    t1[:, 0, 0] /= shift_scale1[:, 3]
    t1[:, 1, 1] /= shift_scale1[:, 2]

    t2 = torch.eye(3, device=h.device).repeat(b, 1, 1)

    if shift_scale2 is not None:
        t2[:, :2, 2] -= shift_scale2[:, [1, 0]]
        t2[:, :2, 2] *= shift_scale2[:, [3, 2]]

        t2[:, 0, 0] *= shift_scale2[:, 3]
        t2[:, 1, 1] *= shift_scale2[:, 2]

    return t2 @ h @ t1


def sample_homography(patch_size, scale_factor):
    """
    :param patch_shape1: (w,h)
    :param scale_factor: float
    """
    points1 = np.stack([[0., 0.],
                        [0., 1.],
                        [1., 1.],
                        [1., 0.]], axis=0)

    hscale_factor = scale_factor / 2 + 0.5

    outer_crop = 0.5 - 0.5 / hscale_factor

    points2 = np.stack([[outer_crop, outer_crop],
                        [outer_crop, 1 - outer_crop],
                        [1 - outer_crop, 1 - outer_crop],
                        [1 - outer_crop, outer_crop]], axis=0)

    inner_crop = 0.5 / hscale_factor - 0.5 / hscale_factor ** 2

    left_displacement = uniform().rvs() * inner_crop
    right_displacement = uniform().rvs() * -inner_crop

    perspective_displacement = uniform(loc=-inner_crop, scale=inner_crop * 2).rvs()

    points2 += np.array([[left_displacement, perspective_displacement],
                         [left_displacement, -perspective_displacement],
                         [right_displacement, perspective_displacement],
                         [right_displacement, -perspective_displacement]])

    points1 *= np.expand_dims((patch_size, patch_size), axis=0)
    points2 *= np.expand_dims((patch_size, patch_size), axis=0)

    def ax(p, q):
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q):
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(points1[i], points2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.transpose(np.stack([[points2[i][j] for i in range(4) for j in range(2)]], axis=0))

    homography = np.transpose(np.linalg.lstsq(a_mat, p_mat, rcond=None)[0])
    homography = np.reshape(np.concatenate([homography, np.ones([homography.shape[0], 1])], axis=1), [3, 3])

    return homography


"""
Support utils
"""

def sample_crop(left, right):
    sf_sample = uniform().rvs()
    is_right = bernoulli(p=0.5).rvs()

    if is_right:
        return sf_sample * right

    else:
        return  sf_sample * left


# scale_factor = uniform().rvs() * (scale_factor - 1) + 1
# outer_crop = 0.5 - 0.5 / scale_factor
#
# points2 = np.stack([[outer_crop, outer_crop],
#                     [outer_crop, 1 - outer_crop],
#                     [1 - outer_crop, 1 - outer_crop],
#                     [1 - outer_crop, outer_crop]], axis=0)


# margin = (1 - patch_ratio) / 2
# points2 = margin + np.array([[0, 0], [0, patch_ratio], [patch_ratio, patch_ratio], [patch_ratio, 0]])

# perspective_amplitude_x,
# patch_ratio, perspective,

# def crop_homography(h, rect1=None, rect2=None):
#     """
#     :param h: 3 x 3
#     :param rect1: (top, left, bottom, right) for the first image
#     :param rect2: (top, left, bottom, right) for the second image
#     """
#     if rect1 is not None:
#         top1, left1 = rect1[:2]
#
#         t = np.mat([[1, 0, left1],
#                     [0, 1, top1],
#                     [0, 0, 1]], dtype=h.dtype)
#
#         h = h * t
#
#     if rect2 is not None:
#         top2, left2 = rect2[:2]
#
#         t = np.mat([[1, 0, -left2],
#                     [0, 1, -top2],
#                     [0, 0, 1]], dtype=h.dtype)
#
#         h = t * h
#
#     return h
#
#
# def resize_homography(h, scale_factor1=None, scale_factor2=None):
#     """
#     :param h: 3 x 3
#     :param scale_factor1: new_size / size of the first image :type numpy array
#     :param scale_factor2: new_size / size of the second image :type numpy array
#     """
#     if scale_factor1 is not None:
#         if np.ndim(scale_factor1) == 0:
#             wr1 = scale_factor1
#             hr1 = scale_factor1
#
#         else:
#             wr1, hr1 = scale_factor1
#
#         t = np.mat([[1 / wr1, 0, 0],
#                     [0, 1 / hr1, 0],
#                     [0, 0, 1]], dtype=h.dtype)
#
#         h = h * t
#
#     if scale_factor2 is not None:
#         if np.ndim(scale_factor2) == 0:
#             wr2 = scale_factor2
#             hr2 = scale_factor2
#
#         else:
#             wr2, hr2 = scale_factor2
#
#         t = np.mat([[wr2, 0, 0],
#                     [0, hr2, 0],
#                     [0, 0, 1]], dtype=h.dtype)
#
#         h = t * h
#
#     return h
# scaling, rotation, translation
# n_scales, n_angles, scaling_amplitude,
# max_angle

# Random scaling. Sample several scales, check collision with borders, randomly pick a valid one
# if scaling:
#     scales = np.concatenate([[1.], truncnorm(-2, 2, 1, scaling_amplitude / 2).rvs(n_scales)], 0)
#     center = np.mean(points2, axis=0, keepdims=True)
#     scaled = np.expand_dims(points2 - center, axis=0) * np.expand_dims(np.expand_dims(scales, 1), 1) + center
#
#     valid = np.arange(n_scales)  # all scales are valid except scale=1
#
#     idx = valid[np.random.randint(0, valid.shape[0])]
#     points2 = scaled[idx]

# Random translation
# if translation:
#     t_min, t_max = np.amin(points2, axis=0), np.amin(1 - points2, axis=0)
#
#     points2 += np.expand_dims(np.stack([np.random.uniform(-t_min[0], t_max[0]),
#                                         np.random.uniform(-t_min[1], t_max[1])]), axis=0)

# Random rotation. Sample several rotations, check collision with borders, randomly pick a valid one
# if rotation:
#     angles = np.linspace(-max_angle, max_angle, n_angles)
#     angles = np.concatenate([[0.], angles], axis=0)  # in case no rotation is valid
#     center = np.mean(points2, axis=0, keepdims=True)
#     rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles),
#                                    np.sin(angles), np.cos(angles)], axis=1), [-1, 2, 2])
#     rotated = np.matmul(np.tile(np.expand_dims(points2 - center, axis=0), [n_angles + 1, 1, 1]), rot_mat) + center
#     valid = np.arange(n_angles)
#     idx = valid[np.random.randint(0, valid.shape[0])]
#     points2 = rotated[idx]