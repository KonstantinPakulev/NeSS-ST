import source.utils.model_utils as mu
import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu
import source.models.namespace as m_ns

from source.core.module import ModuleWrapper

from source.models.shi.module import ShiDetector


class DetectorWrapper(ModuleWrapper):

    def __init__(self, module_config, experiment_config):
        super().__init__(experiment_config)
        self.shi = ShiDetector.from_config(module_config)

    def _get_forward_base_key(self):
        return du.IMAGE_GRAY

    def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
        pass

    def _get_process_base_key(self):
        return None

    def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
        image_gray = batch[f"{du.IMAGE_GRAY}{i}"].to(device)

        kp = self.shi(image_gray, eval_params)

        endpoint[f"{eu.KP}{i}"] = kp

    def _process_branches(self, engine, device, num_branches, batch, endpoint, eval_params):
        pass

    def get(self):
        return None


"""
Legacy code
"""

# image_grayi = batch[f"{du.IMAGE_GRAY}{i}"].to(device)

# if self.loc_size is not None:
#     kpi = localize_kp(kpi, image_grayi, self.loc_size)
# border_size = 2 * (self.sobel_size // 2) + self.window_size // 2
#
#       if self.ms:
#           s2_shi_scorei = endpoint[f"{eu.S2_SHI_SCORE}{i}"].to(device)
#
#           kpi = select_ms_shi_kp(shi_scorei, s2_shi_scorei,
#                                  nms_size, k, border_size=border_size)
#
#       else:
#           kpi = select_kp(shi_scorei, nms_size, k, border_size=border_size)
# if self.ms:
#     s2_shi_scorei = get_shi_score(interpolate(image_grayi, scale_factor=0.5),
#                                   self.sobel_size, self.window_size, self.window_cov)
#
#     bundle[f"{eu.S2_SHI_SCORE}{i}"] = s2_shi_scorei
