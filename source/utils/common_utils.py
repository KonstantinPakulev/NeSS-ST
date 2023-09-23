import numpy as np

import torch

from torch.nn import functional as F


def normalize_coord(grid, shape, align_corners=False):
    """
    :param grid: B x H x W x 2
    """
    h, w = shape[2:]

    # Make a copy to avoid in-place modification
    norm_grid = grid.clone()

    if align_corners:
        # If norm-grid values are top-left corners of pixels
        norm_grid[:, :, :, 0] = norm_grid[:, :, :, 0] / (w - 1) * 2 - 1
        norm_grid[:, :, :, 1] = norm_grid[:, :, :, 1] / (h - 1) * 2 - 1

    else:
        # If norm-grid values are centers of pixels
        norm_grid[:, :, :, 0] = norm_grid[:, :, :, 0] / w * 2 - 1
        norm_grid[:, :, :, 1] = norm_grid[:, :, :, 1] / h * 2 - 1

    return norm_grid


def create_coord_grid(shape, center=True, scale_factor=1.0):
    """
    :param shape: (b, _, h, w) :type tuple
    :param scale_factor: float
    :param center: bool
    :return B x H x W x 2; x, y orientation of coordinates located in center of pixels :type torch.tensor, float
    """
    b, _, h, w = shape

    grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])

    grid_x = grid_x.float().unsqueeze(-1)
    grid_y = grid_y.float().unsqueeze(-1)
    grid = torch.cat([grid_x, grid_y], dim=-1)  # H x W x 2

    # Each coordinate represents the location of the center of a pixel
    if center:
        grid += 0.5

    grid *= scale_factor

    return grid.unsqueeze(0).repeat(b, 1, 1, 1)


def sample_tensor(t, kp,
                  image_shape,
                  mode='bilinear',
                  align_corners=False):
    """
    :param t: B x C x H x W
    :param kp: B x N x 2
    :param image_shape: (b, c, h, w)
    :param mode: str
    :param align_corners: bool
    :return B x N x C
    """
    kp_grid = normalize_coord(kp[:, :, [1, 0]].unsqueeze(1), image_shape, align_corners)
    kp_t = F.grid_sample(t, kp_grid, mode=mode).squeeze(2).permute(0, 2, 1)

    return kp_t


def get_rect(shape, offset=0):
    """
    :param shape: (b, c, h, w) or B x 3
    :param region_size: int
    :param offset: int
    :return 4 or B x 4
    """
    if torch.is_tensor(shape):
        b = shape.shape[0]

        rect = torch.ones(b, 4) * offset
        rect[:, 2] = shape[:, 1] - 1 - offset
        rect[:, 3] = shape[:, 2] - 1 - offset

        return rect

    else:
        return offset, offset, shape[-2] - 1 - offset, shape[-1] - 1 - offset


def get_rect_mask(points, rect):
    """
    :param points: ... x 2; (y, x) orientation
    :param rect: (y, x, h, w) or B x 4
    :return:
    """
    if torch.is_tensor(rect):
        return (points[..., 0] >= rect[:, None, 0]) & \
               (points[..., 1] >= rect[:, None, 1]) & \
               (points[..., 0] <= rect[:, None, 2]) & \
               (points[..., 1] <= rect[:, None, 3])

    else:
        y, x, h, w = rect

        return points[..., 0].ge(y) & \
               points[..., 1].ge(x) & \
               points[..., 0].le(h) & \
               points[..., 1].le(w)


"""
Kernel functions
"""


def apply_kernel(t, kernel, **params):
    """
    :param t: N x 1 x H x W
    :param kernel: 1 x 1 x ks x ks
    :return: N x 1 x H x W
    """
    t = F.conv2d(t, weight=kernel, padding=kernel.shape[2] // 2, **params)

    return t


def apply_gaussian_filter(score, kernel_size, cov):
    """
    :param score: N x 1 x H x W
    :param kernel_size: kernel size
    :param cov: covariance
    """
    if cov == 0:
        raise NotImplementedError

    else:
        gauss_kernel = get_gaussian_kernel(kernel_size, cov).to(score.device)

        score = apply_kernel(score, gauss_kernel)

        return score


def apply_box_filter(t, kernel_size):
    kernel = torch.ones(1, 1, kernel_size, kernel_size).to(t.device)

    return apply_kernel(t, kernel)


def apply_erosion_filter(t, kernel_size):
    kernel = torch.ones(1, 1, kernel_size, kernel_size).to(t.device)
    return apply_kernel(t, kernel) == (kernel_size**2)


"""
Gradient calculation utils
"""


def get_sobel_kernel(kernel_size, device, transpose=False):
    patch_coord = create_coord_grid((1, 1, kernel_size, kernel_size)) - kernel_size / 2

    kernel = patch_coord[..., 1 if transpose else 0] / (patch_coord ** 2).sum(dim=-1).clamp(min=1e-8)

    return kernel.unsqueeze(0).to(device)


def get_grad_kernels(device):
    kernel = torch.tensor([[0, 0, 0],
                           [-0.5, 0, 0.5],
                           [0, 0, 0]], dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

    return kernel, kernel.permute(0, 1, 3, 2)


def get_hess_kernel(device):
    dxdx_kernel = torch.tensor([[0, 0, 0],
                                [1, -2, 1],
                                [0, 0, 0]], dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

    dydy_kernel = torch.tensor([[0, 1, 0],
                                [0, -2, 0],
                                [0, 1, 0]], dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

    dxdy_kernel = 0.25 * torch.tensor([[1, 0, -1],
                                       [0, 0, 0],
                                       [-1, 0, 1]], dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

    return dxdx_kernel, dydy_kernel, dxdy_kernel


def get_second_moment_matrix(image_gray,
                             sobel_size,
                             window_size, window_cov):
    b, c, h, w = image_gray.shape

    dx_kernel = get_sobel_kernel(sobel_size, image_gray.device)
    dy_kernel = get_sobel_kernel(sobel_size, image_gray.device, transpose=True)

    dx = apply_kernel(image_gray, dx_kernel)
    dx2 = dx * dx

    dy = apply_kernel(image_gray, dy_kernel)
    dy2 = dy * dy

    dxdy = dx * dy

    dI = torch.stack([apply_gaussian_filter(dy2, window_size, window_cov),
                      apply_gaussian_filter(dxdy, window_size, window_cov),
                      apply_gaussian_filter(dxdy, window_size, window_cov),
                      apply_gaussian_filter(dx2, window_size, window_cov)], dim=-1).view(b, c, h, w, 2, 2)

    return dI


"""
Support utils
"""


def get_eigen_values(t):
    """
    :param t: ... x 2 x 2
    :return ... x 2
    """
    tr = t.diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    d = tr ** 2 - 4 * t.det()

    d_mask = (d > 0) | (d == 0 & (tr > 0))

    sqrt_d = torch.sqrt(d * d_mask.float())

    eig_val = torch.stack([(tr + sqrt_d) / 2 * d_mask.float(),
                           (tr - sqrt_d) / 2 * d_mask.float()], dim=-1)

    return eig_val


def get_gaussian_kernel(patch_size, cov):
    patch_coord = create_coord_grid((1, 1, patch_size, patch_size))
    patch_center = torch.tensor([patch_size / 2, patch_size / 2]).view(1, 1, 1, 1, 2)

    diff = patch_coord - patch_center

    ll_pg = torch.exp(-0.5 * (diff.unsqueeze(-2) @ diff.unsqueeze(-1)).squeeze(-1).squeeze(-1) / cov)
    ll_pg = ll_pg / ll_pg.sum(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)

    return ll_pg


def rad2deg(radians):
    return radians * 180 / np.pi


def deg2rad(degrees):
    return degrees / 180 * np.pi


def shoelace_area(points):
    """
    :param points: B x N x K x 2
    """
    return 0.5 * (points[..., 1] * torch.roll(points[..., 0], 1, dims=-1) -
                  torch.roll(points[..., 1], 1, dims=-1) * points[..., 0]).sum(dim=-1).abs()


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


"""
Numpy functions
"""


def normalize(v):
    return v / np.clip(np.linalg.norm(v, axis=-1), a_min=1e-8, a_max=None)[..., None]


def angle_vec(v1, v2, degrees=True):
    """
    :param v1: B x N x C, normalized vector
    :param v2: B x N x C, normalized vector
    :return: B, angle in degrees
    """
    a = np.arccos(np.clip((v1 * v2).sum(-1), a_min=-1.0, a_max=1.0))

    if degrees:
        return rad2deg(a)
    else:
        return a


"""
Legacy code
"""

# if size == 3:
#     kernel = torch.tensor([[-1, 0, 1],
#                            [-2, 0, 2],
#                            [-1, 0, 1]], dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
#
# elif size == 5:
#     kernel = torch.tensor([[-5, -4, 0, 4, 5],
#                            [-8, -10, 0, 10, 8],
#                            [-10, -20, 0, 20, 10],
#                            [-8, -10, 0, 10, 8],
#                            [-5, -4, 0, 4, 5]], dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
#
# else:
#     raise NotImplementedError
#
# if transpose:
#     kernel = kernel.permute(0, 1, 3, 2)
#
# return kernel

# FUND_MAT = 'fund_mat'
# ess_mat = 'ess_mat'
# E_param = 'param_E'
# Rt = 'Rt'

# def clamp_probs(probs):
#     eps = torch.finfo(probs.dtype).eps
#     return probs.clamp(min=eps, max=1 - eps)

# def rotate_a_b_axis_angle(a, b):
#     a = a / np.clip(np.linalg.norm(a), a_min=1e-16, a_max=None)
#     b = b / np.clip(np.linalg.norm(b), a_min=1e-16, a_max=None)
#     rot_axis = np.cross(a, b)
#     #   find a proj onto b
#     a_proj = b * (a.dot(b))
#     a_ort = a - a_proj
#     #   find angle between a and b in [0, np.pi)
#     theta = np.arctan2(np.linalg.norm(a_ort), np.linalg.norm(a_proj))
#     if a.dot(b) < 0:
#         theta = np.pi - theta
#
#     aa = rot_axis / np.clip(np.linalg.norm(rot_axis), a_min=1e-16, a_max=None) * theta
#     return aa


# def compose_gt_transform(intrinsics1, intrinsics2, extrinsics1, extrinsics2, type=F):
#     T12 = extrinsics2 @ extrinsics1.inverse()
#
#     R = T12[:, :3, :3]
#     t = T12[:, :3, 3]
#
#     if type == Rt:
#         return R, t
#
#     elif type == E_param:
#         t = F.normalize(t, dim=-1)
#
#         b = intrinsics1.shape[0]
#         _E_param = torch.zeros(b, 5).to(intrinsics1.device)
#
#         for i in range(b):
#             i_E_param = parametrize_pose(R[i].cpu().numpy(), t[i].cpu().numpy())
#
#             _E_param[i] = torch.tensor(i_E_param).to(intrinsics1.device)
#
#         return _E_param
#
#     else:
#         E = vec2cross(t) @ R
#
#         if type == FUND_MAT:
#             return intrinsics2.inverse().transpose(1, 2) @ E @ intrinsics1.inverse()
#         else:
#             return E





# def change_intrinsics(kp, intrinsics2, intrinsics1):
#     """
#     :param kp: B x N x 2, :type torch.tensor, float
#     :param intrinsics2: B x 3 x 3, initial parameters to set :type torch.tensor, float
#     :param intrinsics1: B x 3 x 3, final intrinsic parameters :type torch.tensor, float
#     """
#     kp_h = to_homogeneous(kp)
#     kp_h = kp_h @ torch.inverse(intrinsics2).transpose(1, 2) @ intrinsics1.transpose(1, 2)
#     return to_cartesian(kp_h)


# def parametrize_pose(R, t):
#     R_param, jac = cv2.Rodrigues(R)
#
#     vec = np.asarray([0, 0, 1])
#     t_param = rotate_a_b_axis_angle(vec, t)
#
#     E_param = np.concatenate([R_param.reshape(-1), t_param[0:2]], axis=0)
#
#     return E_param


# def scatter_and_box_dilate(mask, flat_points, kernel_size):
#     """
#     :param mask: B x 1 x H x W
#     :param flat_points: B x N
#     :param kernel_size: int
#     """
#     b, _, h, w = mask.shape
#
#     mask = mask.view(b, -1).scatter(-1, flat_points, 1.0).view(b, 1, h, w)
#     mask = (box_filter(mask, kernel_size) > 0).float()
#
#     return mask

#
# def icos_mat(tensor1, tensor2):
#     """
#     :param tensor1: B x N1 x C, normalized vector, :type torch.tensor, float
#     :param tensor2: B x N2 x C, normalized vector, :type torch.tensor, float
#     :return inv_cos_sim: B x N1 x N2, :type torch.tensor, float
#     """
#     cos_sim = torch.bmm(tensor1, tensor2.permute(0, 2, 1))
#     return 1.0 - cos_sim
#
#
# def icos_vec(tensor1, tensor2):
#     """
#     :param tensor1: B x N x C, normalized vector, :type torch.tensor, float
#     :param tensor2: B x N x C, normalized vector, :type torch.tensor, float
#     :return inv_cos_sim: B x N, :type torch.tensor, float
#     """
#     cos_sim = torch.sum(tensor1 * tensor2, dim=-1)
#     return 1.0 - cos_sim
#
#
# def l2_vec(tensor1, tensor2):
#     """
#     :param tensor1: B x N x C, :type torch.tensor, float
#     :param tensor2: B x N x C, :type torch.tensor, float
#     :return dist: B x N, :type torch.tensor, float
#     """
#     dist = torch.norm(tensor1 - tensor2, p=2, dim=-1)
#
#     return dist
#
#
# def l2_mat(tensor1, tensor2, return_diff=False):
#     """
#     :param tensor1: B x N1 x C, :type torch.tensor, float
#     :param tensor2: B x N2 x C, :type torch.tensor, float
#     :param return_diff: bool
#     :return dist: B x N1 x N2, :type torch.tensor, float
#     """
#     tensor1 = tensor1.unsqueeze(2).float()
#     tensor2 = tensor2.unsqueeze(1).float()
#
#     diff = tensor1 - tensor2
#     dist = torch.norm(diff, p=2, dim=-1)
#
#     if return_diff:
#         return dist, diff
#
#     else:
#         return dist

# dintensity = torch.stack([dx2,
#                           dxdy,
#                           dxdy,
#                           dy2],dim=-1).view(b, c, h, w, 2, 2)

# dintensity = torch.stack([box_filter(dx2, window_size),
#                           box_filter(dxdy, window_size),
#                           box_filter(dxdy, window_size),
#                           box_filter(dy2, window_size)],dim=-1).view(b, c, h, w, 2, 2)

# sigma = 1.5

# if std:
#     var_coord = (prob_patch.unsqueeze(-1) * coord_patch ** 2).sum(dim=-2) - expected_coord ** 2
#     std_coord = var_coord.clamp(min=1e-8).sqrt().sum(dim=-1)
#
#     return expected_coord, std_coord
#
# else:
#     return expected_coord

# def get_rotmat_z(deg, in_shape, out_shape, device=None):
#     rad = deg2rad(deg)
#
#     R = np.array([[np.cos(rad), -np.sin(rad)],
#                   [np.sin(rad), np.cos(rad)]])
#
#     in_c = np.array([[in_shape[-1] // 2],
#                      [in_shape[-2] // 2]])
#
#     out_c = np.array([[out_shape[-1] // 2],
#                       [out_shape[-2] // 2]])
#
#     t = R @ -out_c + in_c
#
#     return torch.tensor([[R[0, 0], R[0, 1], t[0]],
#                          [R[1, 0], R[1, 1], t[1]],
#                          [0, 0, 1]], dtype=torch.float, device=device).unsqueeze(0).repeat(in_shape[0], 1, 1)

