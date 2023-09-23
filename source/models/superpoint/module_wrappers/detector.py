import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu

from source.core.module import ModuleWrapper
from source.models.superpoint.modules.detector import SuperPointDetector

from source.utils.endpoint_utils import select_kp
from source.projective.utils import get_scale_factor


class DetectorWrapper(ModuleWrapper):

    def __init__(self, experiment_config):
        super().__init__(experiment_config)

        self.detector = SuperPointDetector()

    def _get_forward_base_key(self):
        return eu.X

    def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
        xi = bundle[f"{eu.X}{i}"]

        scorei = self.detector(xi)

        bundle[f"{eu.SCORE}{i}"] = scorei

    def _get_process_base_key(self):
        return eu.SCORE

    def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
        nms_size = eval_params.nms_size
        k = eval_params.topk
        score_thresh = eval_params.score_thresh

        imagei = batch[f"{du.IMAGE}{i}"]
        scorei = inference_bundle[f"{eu.SCORE}{i}"].to(device)

        scale_factori = get_scale_factor(imagei.shape, scorei.shape)

        kp_scorei, kpi = select_kp(scorei,
                                   nms_size, k,
                                   score_thresh=score_thresh, scale_factor=scale_factori,
                                   return_score=True)

        endpoint[f"{eu.KP_SCORE}{i}"] = kp_scorei
        endpoint[f"{eu.KP}{i}"] = kpi

    def get(self):
        return self.detector