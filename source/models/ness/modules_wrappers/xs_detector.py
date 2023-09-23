import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu
import source.models.ness.criteria.namespace as c_ns

from source.core.module import ModuleWrapper, get_ith_key_input

from source.models.ness.modules.base_detector import create_base_detector
from source.models.ness.modules.xs_detector import XSDetector


class XSDetectorWrapper(ModuleWrapper):

    def __init__(self, module_config, experiment_config):
        super().__init__(experiment_config)
        self.base_detector = create_base_detector(module_config)

        self.xs_detector = XSDetector.from_config(self.base_detector, module_config)

    def _get_forward_base_key(self):
        return du.IMAGE

    def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
        pass

    def _get_process_base_key(self):
        return None

    def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
        image_gray = get_ith_key_input(du.IMAGE_GRAY, i, batch, None, endpoint, device)

        kp, kp_score = self.xs_detector(image_gray, eval_params, device)

        endpoint[f"{eu.KP}{i}"] = kp
        endpoint[f"{eu.KP_SCORE}{i}"] = kp_score

    def get(self):
        return None
