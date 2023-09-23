import source.utils.model_utils as mu
import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu

from source.core.module import ModuleWrapper

from source.models.rekd.module import REKD
from source.utils.endpoint_utils import select_kp


class DetectorWrapper(ModuleWrapper):

    def __init__(self, module_config, experiment_config):
        super().__init__(experiment_config)
        self.rekd = REKD.from_config(module_config)

    def _get_forward_base_key(self):
        return du.IMAGE_GRAY

    def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
        image_grayi = batch[f"{du.IMAGE_GRAY}{i}"].to(device)

        scorei, _ = self.rekd.forward(image_grayi)

        bundle[f"{eu.SCORE}{i}"] = scorei

    def _get_process_base_key(self):
        return eu.SCORE

    def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
        nms_size = eval_params.nms_size
        k = eval_params.topk

        scorei = inference_bundle[f"{eu.SCORE}{i}"].to(device)

        kp_scorei, kpi = select_kp(scorei,
                                   nms_size, k,
                                   border_size=nms_size,
                                   return_score=True)

        endpoint[f"{eu.KP_SCORE}{i}"] = kp_scorei
        endpoint[f"{eu.KP}{i}"] = kpi

    def get(self):
        return self.rekd
