import torch
import numpy as np

import source.evaluation.namespace as ns

from source.datasets.base.utils import HDataWrapper
from source.evaluation.utils import get_rel_pose, angle_mat
from source.pose.estimators.utils import recover_relative_pose_from_fund_mat, recover_relative_pose_from_homography
from source.projective.homography import warp_points_h
from source.utils.common_utils import normalize, angle_vec


def relative_pose_error(T12, success,
                        extrinsics1, extrinsics2):
    """
    :param T12: 1 x 3 x 4
    :param success: bool
    :param extrinsics1: 1 x 4 x 4
    :param extrinsics2: 1 x 4 x 4
    """
    if success:
        gt_R, _, gt_t_dir = get_rel_pose(extrinsics1, extrinsics2)

        r_err = angle_mat(T12[:, :3, :3], gt_R)
        t_err = angle_vec(normalize(T12[:, :3, 3]), gt_t_dir)

        return r_err, t_err

    else:
        return np.array([ns.MAX_ANGLE_ERR]), np.array([ns.MAX_ANGLE_ERR])


def relative_pose_error_from_fund_mat(F, inl_mask, success,
                                      kp1, nn_kp2,
                                      intrinsics1, intrinsics2,
                                      extrinsics1, extrinsics2):
    if success:
        T12 = recover_relative_pose_from_fund_mat(F, inl_mask,
                                                  kp1, nn_kp2,
                                                  intrinsics1, intrinsics2)

        return relative_pose_error(T12[None], success,
                                   extrinsics1, extrinsics2)

    else:
        return relative_pose_error(None, success, None, None)


def relative_pose_error_from_homography(T12, inl_mask, success,
                                        kp1, nn_kp2,
                                        H1):
    if success:
        extrinsics2 = np.concatenate([recover_relative_pose_from_homography(H1, inl_mask,
                                                                            kp1, nn_kp2),
                                      np.array([[0, 0, 0, 1]])], axis=0)

        return relative_pose_error(T12, success, np.eye(4)[None], extrinsics2[None])

    else:
        return relative_pose_error(None, success, None, None)


def homography_corners_reprojection_error(H12, success,
                                          image_shape1, shift_scale1, H1):
    if success:
        image_shape1 = torch.tensor(image_shape1, dtype=torch.float)
        shift_scale1 = torch.tensor(shift_scale1)
        H1 = torch.tensor(H1, dtype=torch.float)
        H12 = torch.tensor(H12, dtype=torch.float)[None]

        corners = get_image_corners(image_shape1)

        gt_h_data = HDataWrapper(image_shape1, shift_scale1, H1)
        h_data = HDataWrapper(image_shape1, shift_scale1, H12)

        gt_w_corners = warp_points_h(corners, gt_h_data, 'p')
        w_corners = warp_points_h(corners, h_data, 'p')

        c_reproj_err = (gt_w_corners - w_corners).norm(dim=-1).mean(dim=-1).numpy()

        return c_reproj_err

    else:
        return np.array([image_shape1.shape[-1] * image_shape1.shape[-2]], dtype=np.float)


"""
Support utils
"""


def get_image_corners(shape):
    corners = torch.zeros(shape.shape[0], 4, 2)

    y, x = shape[:, 1] - 1, shape[:, 2] - 1

    corners[:, 1, 1] = x
    corners[:, 2, 0] = y
    corners[:, 2, 1] = x
    corners[:, 3, 0] = y

    return corners + 0.5


"""
Legacy code
"""

# def relative_pose_error_from_qvec_tvec(qvec, tvec, inl_mask,
#                                        extrinsics1, extrinsics2,
#                                        min_num_inliers):
#     T12 = np.concatenate([qvec2rotmat(qvec), tvec.reshape(3, 1)], axis=1)
#
#     print(T12)
#
#     return relative_pose_error(T12[None], inl_mask[None],
#                                extrinsics1, extrinsics2,
#                                min_num_inliers)

# def absolute_pose_error(world_kp1, nn_kp2,
#                         mm_desc_mask1,
#                         intrinsics2,
#                         extrinsics1, extrinsics2,
#                         estimator,
#                         num_repeats=1):
#     b, n = world_kp1.shape[:2]
#
#     gt_R, gt_t, gt_t_dir = get_gt_rel_pose(extrinsics1, extrinsics2)
#
#     rel_pose, inl_mask = estimator.estimate(world_kp1, nn_kp2,
#                                             mm_desc_mask1,
#                                             intrinsics2,
#                                             num_repeats)
#
#
#
#     r_err = np.zeros((b, num_repeats))
#     t_l2_err = np.zeros((b, num_repeats))
#     t_ang_err = np.zeros((b, num_repeats))
#
#     for i in range(num_repeats):
#         r_err[:, i] = angle_mat(rel_pose[:, i, :3, :3], gt_R)
#         t_l2_err[:, i] = np.linalg.norm(rel_pose[:, i, :3, 3] - gt_t, axis=-1)
#         t_ang_err[:, i] = angle_vec(normalize(rel_pose[:, i, :3, 3]), gt_t_dir)
#
#     return r_err, t_l2_err, t_ang_err, inl_mask

# if std:
#     if len(pose_err.shape) == 3:
#         pass
#         acc_stdi = np.sqrt(np.mean(np.mean((thr_mask - acci[None, :, None] ** 2), axis=0), axis=-1))
#
#     else:
#         acc_stdi = np.mean((thr_mask - acci) ** 2)
#         print(acc_stdi)
#         print(np.mean((thr_mask - acci) ** 2))
#         acc_stdi = np.sqrt(np.mean((thr_mask - acci) ** 2))
#
# acc_std.append(acc_stdi)

# if std:
#     return np.array(acc), np.array(acc_std)
#
# else:
#     return np.array(acc)

# if torch.is_tensor(shape):
#     corners = torch.zeros((shape.shape[0], 4, 2))
#
#     y, x = shape[:, 1] - 1, shape[:, 2] - 1
#
#     corners[:, 1, 1] = x
#     corners[:, 2, 0] = y
#     corners[:, 2, 1] = x
#     corners[:, 3, 0] = y
#
# else:
#     b, _, h, w = shape
#
#     corners = torch.zeros(b, 4, 2)
#
#     y, x = h - 1, w - 1
#
#     corners[:, 1, 1] = x
#     corners[:, 2, 0] = y
#     corners[:, 2, 1] = x
#     corners[:, 3, 0] = y
#
# return corners + 0.5
