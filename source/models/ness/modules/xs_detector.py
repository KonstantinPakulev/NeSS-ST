import torch

import source.models.ness.criteria.namespace as c_ns

from source.utils.endpoint_utils import nms, mask_border
from source.models.ness.criteria.homography import generate_homographies
from source.models.ness.criteria.keypoint_scores import get_stability_score, get_repeatability_score
from source.models.ness.modules.utils import pad2topk


class XSDetector:

    @staticmethod
    def from_config(base_detector, module_config):
        if c_ns.SS in module_config:
            xs_config = module_config.ss
            criterion = c_ns.SS

        elif c_ns.RS in module_config:
            xs_config = module_config.rs
            criterion = c_ns.RS

        return XSDetector(base_detector,
                          xs_config.scale_factor,
                          xs_config.num_samples,
                          xs_config.block_size,
                          criterion)

    def __init__(self, base_detector,
                 scale_factor, num_samples, block_size,
                 criterion):
        self.base_detector = base_detector

        self.scale_factor = scale_factor
        self.num_samples = num_samples
        self.block_size = block_size

        self.criterion = criterion

    def __call__(self, image_gray, eval_params, device):
        assert image_gray.shape[0] == 1

        nms_size = eval_params.nms_size
        score_thresh = eval_params.score_thresh
        topk = eval_params.topk

        patch_size = nms_size + self.base_detector.get_border_size() * 2
        scaled_patch_size = int(patch_size * self.scale_factor) // 2 * 2 + 1
        border_size = scaled_patch_size + int(4 * self.scale_factor)

        score_map = self.base_detector.get_score(image_gray)

        nms_score = nms(score_map, nms_size) > score_thresh
        nms_score = mask_border(nms_score.float(), border_size)

        nz_kp = nms_score.nonzero()[:, 2:].float().unsqueeze(0) + 0.5

        h_data = generate_homographies(scaled_patch_size,
                                       self.scale_factor, self.num_samples,
                                       device)

        nz_kp_xs = []

        for i in range(0, nz_kp.shape[1], self.block_size):
            nz_kpi = nz_kp[:, i:i + self.block_size]
            nz_kpi = pad2topk(nz_kpi, nz_kpi.shape[1], self.block_size)

            if self.criterion == c_ns.SS:
                nz_kp_xsi = get_stability_score(image_gray, nz_kpi, h_data,
                                                scaled_patch_size, patch_size,
                                                self.num_samples,
                                                self.block_size, nms_size,
                                                self.base_detector)

            elif self.criterion == c_ns.RS:
                nz_kp_xsi = get_repeatability_score(image_gray, nz_kpi, h_data,
                                                    scaled_patch_size, patch_size,
                                                    self.num_samples,
                                                    self.block_size, nms_size,
                                                    self.base_detector)

            else:
                raise ValueError(self.criterion)

            nz_kp_xs.append(nz_kp_xsi)

        nz_kp_xs = torch.cat(nz_kp_xs, dim=1)[:, :nz_kp.shape[1]]

        kp_xs, kp_xs_idx = torch.topk(nz_kp_xs, min(topk, nz_kp.shape[1]), -1, largest=False)
        kp = nz_kp.gather(-2, kp_xs_idx.unsqueeze(-1).repeat(1, 1, 2))

        return kp, kp_xs
