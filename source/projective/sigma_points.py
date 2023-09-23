from math import sqrt

import torch

from source.projective.utils import to_cartesian, grid_sample
from source.utils.common_utils import get_rect, get_rect_mask
from source.projective.rbt import backproject


def backproject_sigma_points(points_mean, points_cov,
                             depth, intrinsics, extrinsics, shift_scale):
    """
    :param points_mean: B x N x 2; x,y orientation
    :param points_cov: B x N x 2 x 2; x,y orientation
    :param depth: B x N
    :param intrinsics: B x 3 x 3
    :param extrinsics: B x 4 x 4
    :param shift_scale: B x 4
    """
    b, n = points_mean.shape[:2]
    h = sqrt(3)

    points_chol_l = torch.linalg.cholesky(points_cov)

    ex, ey = points_chol_l[..., 0], points_chol_l[..., 1]

    sigma_points = torch.stack([points_mean,
                                points_mean + h * ex,
                                points_mean + h * ey,
                                points_mean - h * ex,
                                points_mean - h * ey], dim=2)

    world_sigma_points = backproject(sigma_points.view(b, n * 5, 2),
                                     depth, intrinsics, extrinsics, shift_scale)

    return world_sigma_points


def project_sigma_points(world_sigma_points, depth, intrinsics, extrinsics, shift_scale):
    b = world_sigma_points.shape[0]

    world_sigma_points = extrinsics @ world_sigma_points.permute(0, 2, 1)  # B x 4 x N
    world_sigma_points = to_cartesian(world_sigma_points, dim=1)  # B x 3 x N

    world_sigma_points_depth = world_sigma_points[:, 2, :].clone().view(b, -1)

    intrinsics = intrinsics.clone()
    intrinsics[:, :2, 2] -= shift_scale[:, [1, 0]]
    intrinsics[:, :2, 2] *= shift_scale[:, [3, 2]]

    intrinsics[:, 0, 0] *= shift_scale[:, 3]
    intrinsics[:, 1, 1] *= shift_scale[:, 2]

    world_sigma_points = intrinsics @ world_sigma_points  # B x 3 x N
    w_sigma_points = to_cartesian(world_sigma_points.permute(0, 2, 1))

    w_sigma_points_depth = grid_sample(depth, w_sigma_points.unsqueeze(1)).view(b, -1)
    w_depth_mask = (w_sigma_points_depth > 0) & (torch.abs(w_sigma_points_depth - world_sigma_points_depth) < 0.05)

    w_sigma_points = w_sigma_points.view(b, -1, 5, 2)
    w_depth_mask = w_depth_mask.view(b, -1, 5).sum(dim=-1) == 5

    weight_m = torch.tensor([1 / 3,
                             1 / 6,
                             1 / 6,
                             1 / 6,
                             1 / 6], device=world_sigma_points.device).view(1, 1, 5, 1)

    w_points_mean = (w_sigma_points * weight_m).sum(dim=2)

    weight_c1 = 1 / 12
    weight_c2 = 1 / 18

    w_sigma_ord1 = w_sigma_points[:, :, 1:3] - w_sigma_points[:, :, 3:5]
    w_sigma_ord2 = w_sigma_points[:, :, 1:3] + w_sigma_points[:, :, 3:5] - 2 * w_sigma_points[:, :, :1]

    w_points_cov = weight_c1 * (w_sigma_ord1.unsqueeze(-1) @ w_sigma_ord1.unsqueeze(-2)).sum(dim=2) + \
                   weight_c2 * (w_sigma_ord2.unsqueeze(-1) @ w_sigma_ord2.unsqueeze(-2)).sum(dim=2)

    return w_points_mean, w_points_cov, w_depth_mask


"""
Sigma points propagation
"""


def warp_gaussian_rbt(points_mean1, points_cov1,
                      scene_data):
    """
    :param points_mean1: B x N x 2, coordinates order is (y, x)
    :param points_cov1: B x N x 2 x 2, coordinates order is (y, x)
    :param scene_data with orientation 1->2
    """
    depth1, intrinsics1, extrinsics1, shift_scale1 = scene_data.depth1, scene_data.intrinsics1, \
                                                     scene_data.extrinsics1, scene_data.shift_scale1
    depth2, intrinsics2, extrinsics2, shift_scale2 = scene_data.depth2, scene_data.intrinsics2, \
                                                     scene_data.extrinsics2, scene_data.shift_scale2

    points_mean1 = torch.flip(points_mean1, [-1]).clone()
    points_cov1 = torch.flip(points_cov1, [-1, -2]).clone()

    world_sigma_points1 = backproject_sigma_points(points_mean1, points_cov1,
                                                   depth1, intrinsics1, extrinsics1, shift_scale1)

    w_points_mean1, w_points_cov1, w_depth_mask1 = project_sigma_points(world_sigma_points1,
                                                                        depth2, intrinsics2, extrinsics2, shift_scale2)

    is_cov_mask1 = (w_points_cov1[:, :, 0, 1] - w_points_cov1[:, :, 1, 0]).abs() < 1e-4

    w_points_mean1 = torch.flip(w_points_mean1, [-1])
    w_points_cov1 = torch.flip(w_points_cov1, [-1, -2])
    w_points_mask1 = get_rect_mask(w_points_mean1, get_rect(depth2.shape)) & \
                     w_depth_mask1 & \
                     is_cov_mask1

    b, n = points_mean1.shape[:2]
    I = torch.eye(2, device=points_mean1.device).view(1, 1, 2, 2)

    w_points_cov1 = w_points_cov1 * w_points_mask1.view(b, n, 1, 1).float() + \
                    I * (~w_points_mask1.view(b, n, 1, 1)).float()

    return w_points_mean1, w_points_cov1, w_points_mask1
