import torch

import source.models.namespace as m_ns

from source.utils.endpoint_utils import select_kp, localize_kp
from source.utils.common_utils import get_eigen_values, get_second_moment_matrix

from source.models.base.modules.handcrafted import HandCraftedDetectorModule


class ShiDetector(HandCraftedDetectorModule):

    @staticmethod
    def from_config(shi_config):
        return ShiDetector(shi_config.sobel_size,
                           shi_config.window_size, shi_config.window_cov,
                           shi_config.loc)

    def __init__(self, sobel_size,
                 window_size, window_cov,
                 loc):
        super().__init__(loc)
        self.sobel_size = sobel_size
        self.window_size = window_size
        self.window_cov = window_cov

    def get_name(self):
        return m_ns.SHI

    def get_score(self, image_gray):
        return get_shi_score(image_gray,
                             self.sobel_size,
                             self.window_size, self.window_cov)

    def _forward(self, image_gray, eval_params):
        return select_shi_kp(image_gray,
                             eval_params.nms_size, eval_params.topk,
                             self.sobel_size, self.window_size, self.window_cov,
                             self.loc,
                             self.get_border_size())

    def _calculate_border_size(self):
        return max(self.sobel_size, self.window_size) // 2 + self.sobel_size // 2

    def _localize_kp_impl(self, kp, image_gray, score):
        return kp + localize_kp(score, kp)


"""
Support utils
"""


def select_shi_kp(image_gray,
                  nms_size, k,
                  sobel_size,
                  window_size, window_cov,
                  loc,
                  border_size):
    shi_score = get_shi_score(image_gray,
                              sobel_size,
                              window_size, window_cov)

    kp = select_kp(shi_score, nms_size, k, border_size=border_size)

    if loc:
        kp = kp + localize_kp(shi_score, kp)

    return kp


def get_shi_score(image_gray,
                  sobel_size,
                  window_size, window_cov):
    smm = get_second_moment_matrix(image_gray,
                                   sobel_size,
                                   window_size, window_cov)

    shi_score, _ = get_eigen_values(smm).min(dim=-1)

    return shi_score


"""
Legacy code
"""


# from torch.nn.functional import interpolate
# nms, mask_border, flat2grid, create_patch_grid, grid2flat,

# def localize_kp(kp, image_gray, loc_size):
#     b, n = kp.shape[:2]
#     kp_p = sample_tensor_patch(image_gray, kp, loc_size, image_gray.shape).squeeze(-1).\
#         view(b * n, 1, loc_size, loc_size)
#
#     kernel_dx = get_sobel_kernel(loc_size, image_gray.device)
#     kernel_dy = get_sobel_kernel(loc_size, image_gray.device, transpose=True)
#
#     dx = apply_kernel(kp_p, kernel_dx)
#     dy = apply_kernel(kp_p, kernel_dy)
#
#     dxdx = apply_kernel(dx, kernel_dx)
#     dxdy = apply_kernel(dx, kernel_dy)
#     dydx = apply_kernel(dy, kernel_dx)
#     dydy = apply_kernel(dy, kernel_dy)
#
#     D2 = torch.cat([dxdx[:, :, 1, 1], dxdy[:, :, 1, 1],
#                     dydx[:, :, 1, 1], dydy[:, :, 1, 1]], dim=1).view(b, n, 2, 2)
#     D = torch.cat([dx[:, :, 1, 1],
#                    dy[:, :, 1, 1]], dim=1).view(b, n, 2, 1)
#
#     D2_lu = torch.lu(D2)
#
#     x_hat = torch.flip(torch.lu_solve(-D, *D2_lu).squeeze(-1), dims=[-1])
#
#     return kp + x_hat

# ms_s2_shi_score = mask_border(ms_s2_shi_score, border_size, 0)
# s2_w = s2_shi_score.shape[-1]

# s2_kp_value, s2_flat_kp = ms_s2_shi_score.view(b, -1).topk(k, dim=-1)

# s2_kp = (flat2grid(s2_flat_kp, s2_w) + 0.5) * 2
#
# idx = torch.cat([kp_value, s2_kp_value], dim=1).topk(k, dim=-1)[1]

# return torch.cat([kp, s2_kp], dim=-2).gather(-2, idx.unsqueeze(-1).repeat(1, 1, 2))

# max_s2_v, max_s2_idx = torch.cat([max_pool2d(nms_shi_score, 2),
#                                   nms_s2_shi_score], dim=1).max(dim=1)

# max_s2_ch_mask = max_s2_idx == 1
# max_s2_spatial_score = nms(max_s2_v.unsqueeze(1), nms_size)
#
# ms_s2_shi_score = max_s2_ch_mask.float() * max_s2_spatial_score
#
# s2_mask = apply_box_filter((ms_s2_shi_score > 0).float(), nms_size) != 0.0
#
# ms_shi_score = nms_shi_score * interpolate((~s2_mask).float(), scale_factor=2.0, mode='nearest')

# return torch.cat([kp, s2_kp], dim=-2).gather(-2, idx.unsqueeze(-1).repeat(1, 1, 2))

# def select_ms_shi_kp(shi_score, s2_shi_score,
#                      nms_size, k, border_size):
#     nms_shi_score = nms(shi_score, nms_size)
#     nms_s2_shi_score = interpolate(nms(s2_shi_score, nms_size), scale_factor=2, mode='nearest')
#
#     max_v, max_idx = torch.cat([nms_shi_score,
#                                 nms_s2_shi_score], dim=1).max(dim=1)
#
#     ms_shi_score = nms(max_v.unsqueeze(1), nms_size)
#     s2_ch_mask = (max_idx.unsqueeze(1) == 1).float()
#
#     if border_size is None:
#         border_size = nms_size // 2
#
#     ms_shi_score = mask_border(ms_shi_score, border_size, 0)
#
#     b, _, _, w = shi_score.shape
#
#     _, flat_kp = ms_shi_score.view(b, -1).topk(k, dim=-1)
#     kp = flat2grid(flat_kp, w) + 0.5
#
#     kp_pg = create_patch_grid(kp, 3, shi_score.shape)
#     s2_kp_pg_mask = (nms_s2_shi_score > 0).view(b, -1).\
#         gather(-1, grid2flat(kp_pg.long(), w).view(b, -1)).\
#         view(b, -1, 9)
#
#     s2_kp = (kp_pg * s2_kp_pg_mask.unsqueeze(-1)).sum(dim=-2) / 4
#     s2_kp_mask = s2_ch_mask.view(b, -1).gather(-1, flat_kp).unsqueeze(-1)
#
#     kp = kp * (1 - s2_kp_mask) + s2_kp * s2_kp_mask
#
#     return kp
