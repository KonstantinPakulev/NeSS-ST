import source.models.namespace as m_ns

from source.utils.endpoint_utils import select_kp, localize_kp
from source.utils.common_utils import apply_kernel, get_hess_kernel, apply_gaussian_filter

from source.models.base.modules.handcrafted import HandCraftedDetectorModule


class LaplacianOfGaussianDetector(HandCraftedDetectorModule):

    @staticmethod
    def from_config(log_config):
        return LaplacianOfGaussianDetector(log_config.gauss_size, log_config.gauss_cov,
                                           log_config.loc)

    def __init__(self, gauss_size, gauss_cov, loc):
        super().__init__(loc)
        self.gauss_size = gauss_size
        self.gauss_cov = gauss_cov

    def get_name(self):
        return m_ns.LOG

    def _forward(self, image_gray, eval_params):
        return select_log_kp(image_gray,
                             eval_params.nms_size, eval_params.topk,
                             self.gauss_size, self.gauss_cov,
                             self.loc,
                             self.get_border_size())

    def get_score(self, image_gray):
        return get_log_score(image_gray, self.gauss_size, self.gauss_cov)

    def _calculate_border_size(self):
        return self.gauss_size // 2 + 1

    def _localize_kp_impl(self, kp, image_gray, score):
        return kp + localize_kp(score, kp)


"""
Support utils
"""


def select_log_kp(image_gray,
                  nms_size, k,
                  gauss_size, gauss_cov,
                  loc,
                  border_size):
    log_score = get_log_score(image_gray, gauss_size, gauss_cov)

    kp = select_kp(log_score, nms_size, k, border_size=border_size)

    if loc:
        kp = kp + localize_kp(log_score, kp)

    return kp


def get_log_score(image_gray, gauss_size, gauss_cov):
    image_gray = apply_gaussian_filter(image_gray, gauss_size, gauss_cov)

    dxdx_kernel, dydy_kernel, _ = get_hess_kernel(image_gray.device)

    dxdx = apply_kernel(image_gray, dxdx_kernel)
    dydy = apply_kernel(image_gray, dydy_kernel)

    return dxdx + dydy
