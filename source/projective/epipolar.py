import torch

from source.projective.utils import to_homogeneous, shift_scale_intrinsics_c2w, shift_scale_intrinsics_w2c


def get_epipolar_dist(kp1, nn_kp2, rbt_data):
    """
    :param kp1: B x N x 2; (y,x) orientation
    :param nn_kp2: B x N x 2; (y,x) orientation
    :param rbt_data: object
    """
    nn_kp2 = to_homogeneous(torch.flip(nn_kp2, [-1]))
    w_line1 = transform_fund_mat(torch.flip(kp1, [-1]), get_fund_mat(rbt_data))

    return (nn_kp2 * w_line1).sum(dim=-1).abs()

"""
Support utils
"""


def transform_fund_mat(point1, F):
    point_h1 = to_homogeneous(point1)

    w_line1 = point_h1 @ F.transpose(1, 2)
    w_line1 = w_line1 / w_line1[..., :2].norm(dim=-1).unsqueeze(-1).clamp(min=1e-8)

    return w_line1


def get_ess_mat(rbt_data):
    extrinsics1, extrinsics2 = rbt_data.extrinsics1, rbt_data.extrinsics2

    T12 = extrinsics2 @ extrinsics1.inverse()

    R = T12[:, :3, :3]
    t = T12[:, :3, 3]

    E = vec2cross(t) @ R

    return E


def get_fund_mat(rbt_data):
    E = get_ess_mat(rbt_data)

    intrinsics1 = shift_scale_intrinsics_c2w(rbt_data.intrinsics1, rbt_data.shift_scale1)
    intrinsics2 = shift_scale_intrinsics_c2w(rbt_data.intrinsics2, rbt_data.shift_scale2)

    F = intrinsics2.transpose(1, 2) @ E @ intrinsics1

    return F


def vec2cross(vec):
    C = torch.zeros((vec.shape[0], 3, 3), device=vec.device)

    C[:, 0, 1] = -vec[:, 2].squeeze()
    C[:, 0, 2] = vec[:, 1].squeeze()
    C[:, 1, 0] = vec[:, 2].squeeze()
    C[:, 1, 2] = -vec[:, 0].squeeze()
    C[:, 2, 0] = -vec[:, 1].squeeze()
    C[:, 2, 1] = vec[:, 0].squeeze()

    return C
