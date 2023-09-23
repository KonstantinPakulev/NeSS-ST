from torch.nn.functional import normalize

import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu

from source.core.module import ModuleWrapper

from source.models.disk.module import DISK

from source.utils.common_utils import sample_tensor


class DescriptorWrapper(ModuleWrapper):

    def __init__(self, experiment_config):
        super().__init__(experiment_config)
        self.disk = DISK()

    def _get_forward_base_key(self):
        return du.IMAGE

    def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
        imagei = batch[f"{du.IMAGE}{i}"].to(device)

        desci, _ = self.disk._split(self.disk.unet(imagei))

        bundle[f"{eu.DESC}{i}"] = desci

    def _get_process_base_key(self):
        return eu.DESC

    def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
        kpi = endpoint[f"{eu.KP}{i}"].to(device)
        desci = inference_bundle[f"{eu.DESC}{i}"].to(device)

        kp_desci = sample_tensor(desci, kpi, batch[f'{du.IMAGE}{i}'].shape)
        kp_desci = normalize(kp_desci, dim=-1)

        endpoint[f"{eu.KP_DESC}{i}"] = kp_desci

    def get(self):
        return self.disk.unet


"""
Legacy code
"""


# desci = normalize(desci, dim=1)

# keypointsi = [Keypoints(k.long(), torch.zeros(k.shape[0]).to(k.device))
#               for k in kpi[..., [1, 0]]]
# featuresi = [kp.merge_with_descriptors(desci[i]) for i, kp in enumerate(keypointsi)]
#
# kp_desci = torch.stack([f.desc for f in featuresi])