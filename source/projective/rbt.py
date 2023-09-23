import torch

from source.projective.utils import to_cartesian, to_homogeneous, shift_scale_intrinsics_w2c, \
    shift_scale_intrinsics_c2w, grid_sample
from source.utils.common_utils import create_coord_grid, get_rect, get_rect_mask


def backproject(image_points, depth, intrinsics, extrinsics, shift_scale, return_depth=False):
    """
    :param image_points: B x N x 2
    :param depth: B x 1 x H x W
    :param intrinsics: B x 3 x 3
    :param extrinsics: B x 4 x 4
    :param shift_scale: B x 4
    :param return_depth: bool
    """
    image_points_depth = grid_sample(depth, image_points.unsqueeze(1)).view(image_points.shape[0], 1, -1)

    # Prepare intrinsic matrix by accounting for shift and scale of the image
    intrinsics = shift_scale_intrinsics_c2w(intrinsics, shift_scale)

    # Translate points to their corresponding position at distance grid_depth from camera
    world_points = to_homogeneous(image_points).permute(0, 2, 1)  # B x 3 x N
    world_points = (intrinsics @ world_points) * image_points_depth
    world_points = to_homogeneous(world_points, dim=1)  # B x 4 x N

    # Translate points to world space
    world_points = (torch.inverse(extrinsics) @ world_points).permute(0, 2, 1)  # B x N x 4

    if return_depth:
        return world_points, image_points_depth.squeeze(1)

    else:
        return world_points


def project(world_points, intrinsics, extrinsics, shift_scale,
            depth=None, return_reproj_depth=False):
    """
    :param world_points: B x N x 4
    :param intrinsics: B x 3 x 3
    :param extrinsics: B x 4 x 4
    :param shift_scale: B x 4
    :param depth: B x 1 x H x W
    :param return_reproj_depth: bool
    """
    b = world_points.shape[0]

    # Translate points to local camera coordinate system
    world_points = extrinsics @ world_points.permute(0, 2, 1)  # B x 4 x N
    world_points = to_cartesian(world_points, dim=1)  # B x 3 x N

    # Take depth of 3D points with respect to the image plane
    world_points_depth = world_points[:, 2, :].clone().view(b, -1)

    # Prepare intrinsic matrix by accounting for shift and scale of the image
    intrinsics = shift_scale_intrinsics_w2c(intrinsics, shift_scale)

    # Project points on the image plane
    world_points = intrinsics @ world_points  # B x 3 x N
    proj_points = to_cartesian(world_points.permute(0, 2, 1)).view(b, -1, 2)

    if depth is not None:
        # Verify that the depth estimate is consistent between two images
        image_points_depth = grid_sample(depth, proj_points.unsqueeze(1)).view(b, -1)
        depth_mask = (image_points_depth > 0) & (torch.abs(image_points_depth - world_points_depth) < 0.05)

        return proj_points, depth_mask

    elif return_reproj_depth:
        return proj_points, world_points_depth

    else:
        return proj_points


"""
Warping functions
"""


def warp_image_rbt(image1, rbt_data, mode):
    """
    :param image1: B x C x iH x iW
    :param rbt_data with orientation 2->1
    :param mode: str
    """
    depth1, intrinsics1, extrinsics1, shift_scale1 = rbt_data.depth1, rbt_data.intrinsics1, \
                                                     rbt_data.extrinsics1, rbt_data.shift_scale1
    depth2, intrinsics2, extrinsics2, shift_scale2 = rbt_data.depth2, rbt_data.intrinsics2, \
                                                     rbt_data.extrinsics2, rbt_data.shift_scale2

    grid2 = create_coord_grid(depth2.shape).to(depth2.device)

    b, _, h, w = depth2.shape

    world_grid2 = backproject(grid2.view(b, -1, 2), depth2, intrinsics2, extrinsics2, shift_scale2)  # B x iH * iW x 4
    w_grid2, depth_mask2 = project(world_grid2, intrinsics1, extrinsics1, shift_scale1, depth1)  # B x iH * iW x 2, B x iH * iW

    w_grid2 = w_grid2.view(b, h, w, 2)
    depth_mask2 = depth_mask2.view(b, 1, h, w)

    if 'im' == mode:
        w_image1 = grid_sample(image1, w_grid2) * depth_mask2.float()
        vis_mask2 = get_rect_mask(w_grid2[..., [1, 0]], get_rect(depth1.shape)).unsqueeze(1) & depth_mask2

        return w_image1, vis_mask2

    elif 'i' == mode:
        w_image1 = grid_sample(image1, w_grid2) * depth_mask2.float()

        return w_image1

    elif 'm' == mode:
        vis_mask2 = get_rect_mask(w_grid2[..., [1, 0]], get_rect(depth1.shape)).unsqueeze(1) & depth_mask2

        return vis_mask2

    else:
        raise NotImplementedError


def warp_points_rbt(points1, rbt_data):
    """
    :param points1: B x N x 2, coordinates order is (y, x)
    :param rbt_data with orientation 1->2
    """
    depth1, intrinsics1, extrinsics1, shift_scale1 = rbt_data.depth1, rbt_data.intrinsics1, \
                                                     rbt_data.extrinsics1, rbt_data.shift_scale1
    depth2, intrinsics2, extrinsics2, shift_scale2 = rbt_data.depth2, rbt_data.intrinsics2, \
                                                     rbt_data.extrinsics2, rbt_data.shift_scale2

    #  Because warping operates on x,y coordinates we need to swap h and w dimensions
    points1 = points1[..., [1, 0]].clone()

    world_points = backproject(points1, depth1, intrinsics1, extrinsics1, shift_scale1)
    w_points1, depth_mask1 = project(world_points, intrinsics2, extrinsics2, shift_scale2, depth2)

    w_points1 = w_points1[..., [1, 0]]
    w_point_mask1 = get_rect_mask(w_points1, get_rect(depth2.shape)) & depth_mask1

    return w_points1, w_point_mask1


"""
Miscellaneous
"""


def get_reproj_dist(kp1, nn_kp2, scene_data, return_w_kp1=False):
    """
    :param kp1: B x N x 2
    :param nn_kp2: B x N
    :param scene_data: object
    :param return_w_kp1: bool
    """
    w_kp1, w_kp_mask1 = warp_points_rbt(kp1, scene_data)

    reproj_dist1 = (nn_kp2.float() - w_kp1).norm(dim=-1)

    if return_w_kp1:
        return reproj_dist1, w_kp1, w_kp_mask1

    else:
        return reproj_dist1, w_kp_mask1


def pointcloudify_kp(kp1, depth1, intrinsics1, shift_scale1):
    kp1 = kp1[..., [1, 0]].clone()
    I = torch.eye(4, device=kp1.device).view(1, 4, 4).repeat(kp1.shape[0], 1, 1)

    world_kp1, kp_depth1 = backproject(kp1, depth1, intrinsics1, I, shift_scale1, True)
    world_kp1 = world_kp1[:, :, :3]

    return world_kp1, kp_depth1


def pointcloudify_depth(image1, depth1, intrinsics, shift_scale1, return_color=False):
    b, _, h, w = image1.shape

    grid1 = create_coord_grid(image1.shape).view(b, -1, 2)[:, :, [1, 0]]

    world_grid1, grid_depth1 = pointcloudify_kp(grid1, depth1, intrinsics, shift_scale1)
    grid_depth_mask1 = grid_depth1 > 1e-4

    if return_color:
        # grid_sample()
        # grid_color1 = sample_tensor(image1, grid1, image1.shape)

        raise NotImplementedError("A")

        # return world_grid1, grid_depth_mask1, grid_color1

    else:
        return world_grid1, grid_depth_mask1


def pointcloudify_depth_and_project(image1, depth1, intrinsics1, shift_scale1,
                                    intrinsics2, extrinsics2, shift_scale2):
    world_grid1, _ = pointcloudify_depth(image1, depth1, intrinsics1, shift_scale1)

    proj_grid1, world_grid_depth1 = project(to_homogeneous(world_grid1),
                                            intrinsics2,
                                            extrinsics2,
                                            shift_scale2,
                                            return_reproj_depth=True)

    return proj_grid1, world_grid_depth1


# def warp_score_rbt(score1, image1_shape, scene_data, mode):
#     """
#     :param score1: B x C x iH x iW
#     :param image1_shape: (b, c, h, w)
#     :param scene_data with orientation 2->1
#     :param mode: str
#     """
#     scale_factor = get_scale_factor(score1.shape, image1_shape)
#
#     scene_data.depth1 = F.interpolate(scene_data.depth1, scale_factor=scale_factor, mode='bilinear')
#     scene_data.depth2 = F.interpolate(scene_data.depth2, scale_factor=scale_factor, mode='bilinear')
#
#     scene_data.shift_scale1 = scene_data.shift_scale1.clone()
#     scene_data.shift_scale2 = scene_data.shift_scale2.clone()
#
#     scene_data.shift_scale1[:, 2:] *= scale_factor
#     scene_data.shift_scale2[:, 2:] *= scale_factor
#
#     return warp_image_rbt(score1, scene_data, mode)