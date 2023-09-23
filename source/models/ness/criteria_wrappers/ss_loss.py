import torch
from omegaconf import OmegaConf
from copy import deepcopy

import source.models.ness.criteria.namespace as cu_ns
import source.models.ness.modules_wrappers.namespace as n_ns
import source.datasets.base.utils as du

from source.core.criterion import CriterionWrapper

from source.models.ness.criteria.mse_loss import mse_loss
from source.models.ness.criteria.homography import generate_homographies
from source.models.ness.criteria.keypoint_scores import get_stability_score

from source.utils.endpoint_utils import sample_tensor


class SSLoss(CriterionWrapper):

    def __init__(self, model_mode_wrapper, config):
        self.ness_detector = deepcopy(model_mode_wrapper.model_wrapper.modules_wrappers[0].detector)

        self.scale_factor = config.scale_factor
        self.num_samples = config.num_samples

        self.config = OmegaConf.merge(*[config, config.get(self.ness_detector.base_detector.get_name(), {})])

        self.patch_size = self.config.nms_size + self.ness_detector.base_detector.get_border_size() * 2
        self.scaled_patch_size = int(self.patch_size * self.scale_factor) // 2 * 2 + 1

        self.ness_detector.base_detector.set_border_size(self.scaled_patch_size + int(4 * self.scale_factor))
        self.ness_detector.base_detector.set_loc(False)

    def forward(self, engine, device, batch, bundle, endpoint):
        image_gray = batch[du.IMAGE_GRAY1].to(device)
        ness = bundle[n_ns.NESS1].to(device)

        kp, kp_score, kp_base_score = self.ness_detector(image_gray, ness.detach(), self.config)

        kp_mask = kp_base_score > self.config.score_thresh

        h_data = generate_homographies(self.scaled_patch_size,
                                       self.scale_factor, self.num_samples,
                                       device)

        kp_ss = get_stability_score(image_gray, kp, h_data,
                                    self.scaled_patch_size, self.patch_size,
                                    self.num_samples,
                                    self.config.topk, self.config.nms_size,
                                    self.ness_detector.base_detector)

        kp_ness = sample_tensor(ness, kp, ness.shape).squeeze(-1)

        loss = mse_loss(kp_ness, kp_ss, kp_mask)

        endpoint[cu_ns.MSE_LOSS] = loss

        return loss

    def get(self):
        return [cu_ns.MSE_LOSS]


"""
Legacy code
"""
