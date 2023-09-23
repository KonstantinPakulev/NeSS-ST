import numpy as np
import torch
import torch.nn.functional as F

from source.utils.common_utils import normalize_coord


def grid_sample(data, grid):
    """
    :param data: B x C x H_in x W_in
    :param grid: B x H_out x W_out x 2; Grid have have (x,y) coordinates orientation
    :return B x C x H_out x W_out
    """
    norm_grid = normalize_coord(grid, data.shape)

    return F.grid_sample(data, norm_grid, mode='bilinear')


def to_homogeneous(t, dim=-1):
    """
    :param t: Shape B x N x 2 or B x H x W x 3, :type torch.tensor, float
    :param dim: dimension along which to concatenate
    """
    if dim == -1:
        index = len(t.shape) - 1
    else:
        index = dim

    shape = t.shape[:index] + t.shape[index + 1:]
    ones = torch.ones(shape).unsqueeze(dim).float().to(t.device)
    t = torch.cat((t, ones), dim=dim)

    return t


def to_cartesian(t, dim=-1):
    """
    :param t: Shape B x N x 3 or B x H x W x 4, :type torch.tensor, float
    :param dim: dimension along which to normalize
    """
    index = torch.tensor([t.shape[dim] - 1]).to(t.device)
    t = t / torch.index_select(t, dim=dim, index=index).clamp(min=1e-8)

    index = torch.arange(t.shape[dim] - 1).to(t.device)
    t = torch.index_select(t, dim=dim, index=index)

    return t


def to_homogeneous_pose(pose):
    pad = torch.zeros(pose.shape[0], 1, 4)
    pad[:, 0, 3] = 1

    return torch.cat([torch.tensor(pose.astype(np.float32)), pad], dim=1)


def shift_scale_intrinsics_c2w(intrinsics, shift_scale):
    b = intrinsics.shape[0]

    t = torch.eye(3, device=intrinsics.device).repeat(b, 1, 1)
    t[:, :2, 2] += shift_scale[:, [1, 0]]
    
    t[:, 0, 0] /= shift_scale[:, 3]
    t[:, 1, 1] /= shift_scale[:, 2]

    return intrinsics.inverse() @ t


def shift_scale_intrinsics_w2c(intrinsics, shift_scale):
    b = intrinsics.shape[0]

    t = torch.eye(3, device=intrinsics.device).repeat(b, 1, 1)
    t[:, :2, 2] -= shift_scale[:, [1, 0]]
    t[:, :2, 2] *= shift_scale[:, [3, 2]]

    t[:, 0, 0] *= shift_scale[:, 3]
    t[:, 1, 1] *= shift_scale[:, 2]

    return t @ intrinsics


def get_scale_factor(image_shape, score_shape, equal_ratio=True):
    scale_factor = torch.tensor(image_shape[2:], dtype=torch.float) / \
                   torch.tensor(score_shape[2:], dtype=torch.float)

    if equal_ratio:
        if scale_factor[0] == scale_factor[1]:
            return scale_factor[0].item()

        else:
            raise NotImplementedError

    else:
        return scale_factor.unsqueeze(0).unsqueeze(0)


"""
Legacy code
"""


# inv_intrinsics = intrinsics.clone()
    # inv_intrinsics[:, :2, 2] -= shift_scale[:, [1, 0]]
    #
    # inv_intrinsics = inv_intrinsics.inverse()
    # inv_intrinsics[:, 0, 0] /= shift_scale[:, 3]
    # inv_intrinsics[:, 1, 1] /= shift_scale[:, 2]

# def warp_coord_grid_RBT(grid1, depth1, intrinsics1, extrinsics1, shift_scale1, depth2, intrinsics2, extrinsics2,
#                         shift_scale2):
#     """
#     :param grid1: B x H x W x 2
#     :param depth1: B x 1 x H x W
#     :param intrinsics1: B x 3 x 3
#     :param extrinsics1: B x 4 x 4
#     :param shift_scale1: B x 4
#     :param depth2: B x 1 x H x W
#     :param intrinsics2: B x 3 x 3
#     :param extrinsics2: B x 4 x 4
#     :param shift_scale2: B x 4
#     :return: B x H x W x 2, B x 1 x H x W
#     """
#     b, h, w = grid1.shape[:3]
#
#     grid1_depth = grid_sample(depth1, grid1)
#
#     # Prepare intrinsic matrix by accounting for shift and scale of the image
#     c_intrinsics1 = intrinsics1.clone()
#     c_intrinsics1[:, :2, 2] -= shift_scale1[:, [1, 0]]
#
#     c_intrinsics1 = c_intrinsics1.inverse()
#     c_intrinsics1[:, 0, 0] /= shift_scale1[:, 3]
#     c_intrinsics1[:, 1, 1] /= shift_scale1[:, 2]
#
#     # Translate grid cells to their corresponding plane at distance grid_depth from camera
#     grid1_3d = to_homogeneous(grid1).view(b, -1, 3).permute(0, 2, 1)  # B x 3 x H * W
#     grid1_3d = (c_intrinsics1 @ grid1_3d) * grid1_depth.view(b, 1, -1)
#     grid1_3d = to_homogeneous(grid1_3d, dim=1)  # B x 4 x H * W
#
#     # Move 3D points from first camera system to second
#     w_grid1_3d = extrinsics2 @ torch.inverse(extrinsics1) @ grid1_3d
#     w_grid1_3d = to_cartesian(w_grid1_3d, dim=1)  # B x 3 x H * W
#
#     # Warped depth
#     w_grid1_depth = w_grid1_3d[:, 2, :].clone().view(b, 1, h, w)
#
#     # Convert 3D points to their projections on the image plane
#     w_grid1_3d = intrinsics2 @ w_grid1_3d  # B x 3 x H * W
#     w_grid1 = to_cartesian(w_grid1_3d.permute(0, 2, 1)).view(b, h, w, 2)
#
#     w_grid1 = (w_grid1 - shift_scale2[:, None, None, [1, 0]]) * shift_scale2[:, None, None, [3, 2]]
#
#     # Compose occlusion and depth masks
#     w_grid1_depth2 = grid_sample(depth2, w_grid1)
#     depth_mask1 = (w_grid1_depth2 > 0) * (torch.abs(w_grid1_depth - w_grid1_depth2) < 0.05)
#
#     return w_grid1, depth_mask1

# def vis_mask_RBT(depth2, intrinsics2, extrinsics2, shift_scale2, depth1, intrinsics1,
#                  extrinsics1, shift_scale1):
#     grid2 = create_coord_grid(depth2.shape).to(depth2.device)
#
#     w_grid2, depth_mask2 = warp_coord_grid_RBT(grid2, depth2, intrinsics2, extrinsics2, shift_scale2, depth1,
#                                                intrinsics1, extrinsics1, shift_scale1)
#
#     vis_mask2 = get_coord_vis_mask(depth1.shape, w_grid2[..., [1, 0]]).unsqueeze(1) * depth_mask2
#
#     return vis_mask2

# def warp_score_H(score1, image1_shape, image2_shape, H21, modes):
#     scale_factor = get_scale_factor(score1.shape, image1_shape)
#
#     raise NotImplementedError

    # return warp_image_H(score1, image2_shape, H21, modes)



