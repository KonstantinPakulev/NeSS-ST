import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu

from source.core.module import ModuleWrapper
from source.models.superpoint.modules.descriptor import SuperPointDescriptor

from source.utils.common_utils import sample_tensor


class DescriptorWrapper(ModuleWrapper):

    def __init__(self, experiment_config):
        super().__init__(experiment_config)

        self.descriptor = SuperPointDescriptor()

    def _get_forward_base_key(self):
        return eu.X

    def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
        xi = bundle[f"{eu.X}{i}"]

        desci = self.descriptor(xi)

        bundle[f"{eu.DESC}{i}"] = desci

    def _get_process_base_key(self):
        return eu.DESC

    def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
        kpi = endpoint[f"{eu.KP}{i}"].to(device)
        desci = inference_bundle[f"{eu.DESC}{i}"].to(device)

        kp_desci = sample_tensor(desci, kpi, batch[f'{du.IMAGE}{i}'].shape)

        endpoint[f"{eu.KP_DESC}{i}"] = kp_desci

    def get(self):
        return self.descriptor