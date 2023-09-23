import torch

import source.models.namespace as m_ns

from source.utils.endpoint_utils import select_kp, localize_kp
from source.utils.common_utils import get_second_moment_matrix

from source.models.base.modules.handcrafted import HandCraftedDetectorModule


class HarrisDetector(HandCraftedDetectorModule):

    @staticmethod
    def from_config(harris_config):
        return HarrisDetector(harris_config.sobel_size,
                              harris_config.window_size, harris_config.window_cov,
                              harris_config.k,
                              harris_config.loc)

    def __init__(self, sobel_size,
                 window_size, window_cov,
                 k,
                 loc):
        super().__init__(loc)
        self.sobel_size = sobel_size
        self.window_size = window_size
        self.window_cov = window_cov
        self.k = k

    def get_name(self):
        return m_ns.HARRIS

    def get_score(self, image_gray):
        return get_harris_score(image_gray,
                                self.sobel_size,
                                self.window_size, self.window_cov,
                                self.k)

    def _forward(self, image_gray, eval_params):
        return select_harris_kp(image_gray,
                                eval_params.nms_size, eval_params.topk, eval_params.score_thresh,
                                self.sobel_size, self.window_size, self.window_cov,
                                self.k,
                                self.loc,
                                self.get_border_size())

    def _calculate_border_size(self):
        return max(self.sobel_size, self.window_size) // 2 + self.sobel_size // 2

    def _localize_kp_impl(self, kp, image_gray, score):
        return kp + localize_kp(score, kp)


"""
Support utils
"""


def select_harris_kp(image_gray,
                     nms_size, k, score_thresh,
                     sobel_size,
                     window_size, window_cov,
                     harris_k,
                     loc,
                     border_size):
    harris_score = get_harris_score(image_gray,
                                    sobel_size,
                                    window_size, window_cov,
                                    harris_k)

    kp = select_kp(harris_score,
                   nms_size, k,
                   score_thresh=score_thresh,
                   border_size=border_size)

    if loc:
        kp = kp + localize_kp(harris_score, kp)

    return kp


def get_harris_score(image_gray,
                     sobel_size,
                     window_size, window_cov,
                     k):
    smm = get_second_moment_matrix(image_gray,
                                   sobel_size,
                                   window_size, window_cov)

    harris_score = (smm.det() - k * smm.diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)**2).exp()

    return harris_score
