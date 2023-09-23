from source.utils.endpoint_utils import nms, get_resized_image, flat2grid, mask_border


class NeSSDetector:

    @staticmethod
    def from_config(base_detector):
        return NeSSDetector(base_detector)

    def __init__(self, base_detector):
        self.base_detector = base_detector

    def __call__(self, image_gray, ness, eval_params):
        nms_size = eval_params.nms_size
        score_thresh = eval_params.score_thresh
        k = eval_params.topk

        base_score = self.base_detector.get_score(image_gray)

        nms_base_score_mask = nms(base_score, nms_size) > score_thresh
        nms_exp_ness_score = (-ness).exp() * nms_base_score_mask.float()
        nms_exp_ness_score = mask_border(nms_exp_ness_score, self.base_detector.get_border_size(), 0)

        b, _, _, w = base_score.shape

        kp_score, flat_kp = nms_exp_ness_score.view(b, -1).topk(k, dim=-1)
        kp = flat2grid(flat_kp, w) + 0.5

        kp_base_score = base_score.view(b, -1).gather(-1, flat_kp)

        kp = self.base_detector.localize_kp(kp, image_gray, base_score)

        return kp, kp_score, kp_base_score

"""
Legacy code
"""

# SCALES = 'scales'
# if self.scales is not None:
#     return self._detect_multi_scale(image_gray, ness,
#                                     nms_size, score_thresh, k)
#
# else:
#     return self._detect_single_scale(image_gray, ness,
#                                      nms_size, score_thresh, k)

# def _detect_single_scale(self, image_gray, ness,
#                          nms_size, score_thresh, k):


# def _detect_multi_scale(self, image_gray, ness,
#                         nms_size, shi_thresh, k):
#     pass

# from source.utils.endpoint_utils import sample_tensor, nms, mask_border, flat2grid, grid2flat, localize_kp, nms_3d, \
#     get_resized_image