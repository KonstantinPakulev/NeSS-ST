import source.utils.model_utils as mu
import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu
import source.models.namespace as m_ns

from source.core.module import ModuleWrapper

from source.models.harris.module import HarrisDetector


class DetectorWrapper(ModuleWrapper):

    def __init__(self, module_config, experiment_config):
        super().__init__(experiment_config)
        self.harris = HarrisDetector.from_config(module_config)

    def _get_forward_base_key(self):
        return du.IMAGE_GRAY

    def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
        pass

    def _get_process_base_key(self):
        return None

    def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
        image_gray = batch[f"{du.IMAGE_GRAY}{i}"].to(device)

        kp = self.harris(image_gray, eval_params)

        endpoint[f"{eu.KP}{i}"] = kp

    def _process_branches(self, engine, device, num_branches, batch, endpoint, eval_params):
        pass

    def get(self):
        return None
