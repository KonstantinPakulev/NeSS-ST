import torch

import source.evaluation.classical.namespace as c_ns
import source.utils.endpoint_utils as eu
import source.evaluation.namespace as ns

from source.core.evaluation import PairMetricTransformer
from source.datasets.base.utils import HDataWrapper
from source.pose.matchers.factory import instantiate_matcher
from source.evaluation.classical.metrics import repeatability_score, mean_matching_accuracy
from source.evaluation.utils import get_sweep_range, get_kp_desc_and_kp_desc_mask


class RepTransformer(PairMetricTransformer):

    def __init__(self, entity_id, metric_config, model_mode_eval_params, device):
        super().__init__(entity_id, metric_config)
        self.px_thresh = get_sweep_range(metric_config.px_thresh)

    def on_iteration_completed(self, engine, batch, endpoint):
        it_values = super().on_iteration_completed(engine, batch, endpoint)

        kp1, kp2 = batch[eu.KP1][..., [1, 0]], batch[eu.KP2][..., [1, 0]]

        h_data = HDataWrapper().init_from_batch(batch, torch.device('cpu'))

        rep = repeatability_score(kp1, kp2,
                                  h_data,
                                  self.px_thresh)

        it_values[c_ns.REP] = rep

        return it_values


class MMATransformer(PairMetricTransformer):

    def __init__(self, entity_id, metric_config, model_mode_eval_params, device):
        super().__init__(entity_id, metric_config)
        self.matcher = instantiate_matcher(model_mode_eval_params, device)

        self.px_thresh = get_sweep_range(metric_config.px_thresh)

    def on_iteration_completed(self, engine, batch, endpoint):
        it_values = super().on_iteration_completed(engine, batch, endpoint)

        kp1, kp2 = batch[eu.KP1][..., [1, 0]], batch[eu.KP2][..., [1, 0]]
        kp_desc1, kp_desc2, kp_desc_mask1, kp_desc_mask2 = get_kp_desc_and_kp_desc_mask(batch)

        h_data = HDataWrapper().init_from_batch(batch, torch.device('cpu'))

        mma = mean_matching_accuracy(kp1, kp2,
                                     kp_desc1, kp_desc2,
                                     kp_desc_mask1, kp_desc_mask2,
                                     self.matcher,
                                     h_data,
                                     self.px_thresh)

        it_values[c_ns.MMA] = mma

        return it_values
