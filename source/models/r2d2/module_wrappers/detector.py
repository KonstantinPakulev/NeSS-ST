import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu
import source.models.r2d2.module_wrappers.utils as r2d2_u

from source.core.module import ModuleWrapper

from source.models.r2d2.module import Quad_L2Net_ConfCFS
from source.models.r2d2.module_wrappers.utils import select_kp


class DetectorWrapper(ModuleWrapper):

    def __init__(self, experiment_config):
        super().__init__(experiment_config)
        self.r2d2 = Quad_L2Net_ConfCFS()

    def _get_forward_base_key(self):
        return du.IMAGE

    def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
        imagei = batch[f"{du.IMAGE}{i}"].to(device)

        scorei, conf_scorei, _ = self.r2d2(imagei)

        bundle[f"{eu.SCORE}{i}"] = scorei
        bundle[f"{r2d2_u.CONF_SCORE}{i}"] = conf_scorei

    def _get_process_base_key(self):
        return eu.SCORE

    def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
        nms_size = eval_params.nms_size
        k = eval_params.topk
        score_thresh = eval_params.score_thresh
        conf_thresh = eval_params.conf_thresh

        scorei, conf_scorei = inference_bundle[f"{eu.SCORE}{i}"].to(device), inference_bundle[f"{r2d2_u.CONF_SCORE}{i}"].to(device)

        kp_scorei, kpi = select_kp(scorei, conf_scorei,
                                   nms_size, score_thresh, conf_thresh,
                                   k,
                                   return_score=True)

        endpoint[f"{eu.KP_SCORE}{i}"] = kp_scorei
        endpoint[f"{eu.KP}{i}"] = kpi

    def get(self):
        return self.r2d2


"""
Support utils
"""

# kp_scale_idx_key = f"{eu.KP_SCALE_IDX}{i}"
# kp_scale_idxi = torch.ones_like(kp_scorei, dtype=torch.long) * scale_idx
# endpoint[kp_scale_idx_key] = kp_scale_idxi
# endpoint[kp_scale_idx_key] = torch.cat([endpoint[kp_scale_idx_key], kp_scale_idxi], dim=1)