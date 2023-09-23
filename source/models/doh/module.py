import source.models.namespace as m_ns

from source.utils.endpoint_utils import select_kp, localize_kp
from source.utils.common_utils import apply_kernel, get_hess_kernel, apply_gaussian_filter

from source.models.base.modules.handcrafted import HandCraftedDetectorModule


class DeterminantOfHessianDetector(HandCraftedDetectorModule):

    @staticmethod
    def from_config(doh_config):
        return DeterminantOfHessianDetector(doh_config.gauss_size, doh_config.gauss_cov,
                                            doh_config.loc)

    def __init__(self, gauss_size, gauss_cov, loc):
        super().__init__(loc)
        self.gauss_size = gauss_size
        self.gauss_cov = gauss_cov

    def get_name(self):
        return m_ns.DOH

    def get_score(self, image_gray):
        return get_doh_score(image_gray, self.gauss_size, self.gauss_cov)

    def _forward(self, image_gray, eval_params):
        return select_doh_kp(image_gray,
                             eval_params.nms_size, eval_params.topk,
                             self.gauss_size, self.gauss_cov,
                             self.loc,
                             self.get_border_size())

    def _calculate_border_size(self):
        return self.gauss_size // 2 + 1

    def _localize_kp_impl(self, kp, image_gray, score):
        return kp + localize_kp(score, kp)


"""
Support utils
"""


def select_doh_kp(image_gray,
                  nms_size, k,
                  gauss_size, gauss_cov,
                  loc,
                  border_size):
    doh_score = get_doh_score(image_gray, gauss_size, gauss_cov)

    kp = select_kp(doh_score, nms_size, k, border_size=border_size)

    if loc:
        kp = kp + localize_kp(doh_score, kp)

    return kp


def get_doh_score(image_gray, gauss_size, gauss_cov):
    image_gray = apply_gaussian_filter(image_gray, gauss_size, gauss_cov)

    dxdx_kernel, dydy_kernel, dxdy_kernel = get_hess_kernel(image_gray.device)

    dxdx = apply_kernel(image_gray, dxdx_kernel)
    dydy = apply_kernel(image_gray, dydy_kernel)
    dxdy = apply_kernel(image_gray, dxdy_kernel)

    return dxdx * dydy - dxdy ** 2
