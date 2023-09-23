import os
import numpy as np

import source.core.evaluation as evu
import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu
import source.core.namespace as c_ns

from source.core.evaluation import BaseTransformer
from source.pose.estimators.utils import revert_shift_scale


class FeaturesTransformer(BaseTransformer):

    def __init__(self, metric_config, model_mode_eval_params):
        self.output_keys = metric_config.get(evu.OUTPUT_KEYS, []) if metric_config is not None else []
        self.topk = model_mode_eval_params.topk

    def on_iteration_completed(self, engine, batch, endpoint):
        kp1 = endpoint[eu.KP1]
        shift_scale1 = batch[du.SHIFT_SCALE1]

        r_kp1 = revert_shift_scale(kp1, shift_scale1).numpy()[0]
        kp_desc1 = endpoint[eu.KP_DESC1].numpy()[0]

        n = kp1.shape[1]
        image_name_no_ext = os.path.splitext(batch[du.IMAGE_NAME1][0].replace('/', '_'))[0]

        engine.state.metrics[du.IMAGE_NAME] = image_name_no_ext
        engine.state.metrics[eu.KP] = pad2topk(r_kp1, n, self.topk)
        engine.state.metrics[eu.KP_DESC] = pad2topk(kp_desc1, n, self.topk)

        for ok in self.output_keys:
            if ok == du.IMAGE_SHAPE:
                engine.state.metrics[du.IMAGE_SHAPE] = list(batch[du.IMAGE1].shape[1:])

            else:
                engine.state.metrics[ok] = pad2topk(batch[f"{ok}1"], n, self.topk)


class TimeTransformer(BaseTransformer):

    def on_iteration_completed(self, engine, batch, endpoint):
        it_values = {}

        for k, v in endpoint.items():
            if c_ns.INFERENCE_TIME in k:
                it_values[k] = v

            elif c_ns.PROCESS_TIME in k:
                it_values[k] = v

        return it_values

    def on_epoch_completed(self, engine, values):
        for key, value in values.items():
            engine.state.metrics[key] = value


"""
Support utils
"""


def pad2topk(t, n, topk):
    offset = topk - n

    if t.shape[0] != n or offset == 0:
        return t

    else:
        if len(t.shape) == 2:
            return np.concatenate((t, np.zeros((offset, t.shape[-1]))), axis=0)

        elif len(t.shape) == 1:
            return np.concatenate((t, np.zeros((offset,))), axis=0)


"""
Legacy code
"""

# kp_score1 = endpoint[eu.KP_SCORE1].numpy()[0] if eu.KP_SCORE1 in endpoint else None
# kp_shi_score1 = endpoint[deu.KP_SHI_SCORE1].numpy()[0] if deu.KP_SHI_SCORE1 in endpoint else None
#
# if r_kp1.shape[0] < self.topk:
#     offset =
#     r_kp1 =
#     kp_desc1 = np.concatenate((kp_desc1, np.zeros((offset, kp_desc1.shape[-1]))), axis=0)
#
#     if kp_score1 is not None:
#         kp_score1 = np.concatenate((kp_score1, np.zeros((offset,))), axis=0)
#
#     if kp_shi_score1 is not None:
#         kp_shi_score1 = np.concatenate((kp_shi_score1, np.zeros(offset,)), axis=0)
#
#
# if kp_score1 is not None:
#     engine.state.metrics[eu.KP_SCORE1] = kp_score1
#
# if kp_shi_score1 is not None:
#     engine.state.metrics[deu.KP_SHI_SCORE1] = kp_shi_score1

