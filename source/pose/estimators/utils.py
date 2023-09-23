import cv2
import numpy as np

from source.pose.matchers.utils import gather_kp


def revert_shift_scale(kp, shift_scale, change_orientation=True):
    """
    :param kp: B x N x 2; (y,x) orientation; float
    :param shift_scale: B x 4
    :param change_orientation: bool
    """
    # Scale and shift image to its original size
    if len(shift_scale.shape) == 2:
        kp = kp / shift_scale[..., None, [2, 3]] + shift_scale[..., None, [0, 1]]

    else:
        kp = kp / shift_scale[..., [2, 3]] + shift_scale[..., [0, 1]]

    if change_orientation:
        kp = kp[..., [1, 0]]

    return kp

def recover_relative_pose_from_fund_mat(F, inl_mask,
                                        kp1, nn_kp2,
                                        intrinsics1, intrinsics2):
    E = intrinsics2.T @ F @ intrinsics1

    norm_kp1 = to_nic(kp1, intrinsics1)
    norm_nn_kp2 = to_nic(nn_kp2, intrinsics2)

    _, R, t, _ = cv2.recoverPose(E,
                                 norm_kp1[inl_mask],
                                 norm_nn_kp2[inl_mask])

    T = np.zeros((3, 4))
    T[:3, :3] = R
    T[:3, 3] = t.reshape(-1)

    return T


def recover_relative_pose_from_homography(H, inl_mask,
                                          kp1, nn_kp2):
    _, Rs, ts, ns = cv2.decomposeHomographyMat(H, np.eye(3))

    idx = cv2.filterHomographyDecompByVisibleRefpoints(Rs, ns,
                                                       kp1[:, None, :], nn_kp2[:, None, :],
                                                       pointsMask=inl_mask.astype(np.uint8))

    # TODO. What is the correct way?

    if idx is None:
        idx = 0

    else:
        idx = idx[0][0]

    T = np.zeros((3, 4))
    T[:3, :3] = Rs[idx]
    T[:3, 3] = ts[idx].reshape(-1)

    return T


"""
Support utils
"""


def to_nic(kp, intrinsics):
    return to_cartesian(np.transpose(np.linalg.inv(intrinsics) @ np.transpose(to_homogeneous(kp), (1, 0)), (1, 0)))

def to_homogeneous(kp):
    return np.concatenate([kp, np.ones((kp.shape[0], 1))], axis=-1)

def to_cartesian(kp):
    return kp[:, :-1] / kp[:, -1, None]


"""
Legacy code
"""

def prepare_two_view_input(kp1, kp2,
                           mm_desc_mask1, nn_desc_idx1,
                           intrinsics1, intrinsics2,
                           shift_scale1, shift_scale2):
    if shift_scale1 is not None:
        kp1 = revert_shift_scale(kp1, shift_scale1)
        kp2 = revert_shift_scale(kp2, shift_scale2)

    nn_kp2 = gather_kp(kp2, nn_desc_idx1.cpu())

    kp1, nn_kp2 = kp1.numpy(), nn_kp2.numpy()
    mm_desc_mask1 = mm_desc_mask1.cpu().numpy()

    if intrinsics1 is not None:
        intrinsics1, intrinsics2 = intrinsics1.numpy(), intrinsics2.numpy()

    return kp1, nn_kp2, mm_desc_mask1, intrinsics1, intrinsics2


# E = intrinsics2.T @ F @ intrinsics1
#
#           norm_kp1 = to_nic(kp1, intrinsics1)
#           norm_nn_kp2 = to_nic(nn_kp2, intrinsics2)
#
#           _, R, t, _ = cv2.recoverPose(E,
#                                        norm_kp1[_inl_mask],
#                                        norm_nn_kp2[_inl_mask])