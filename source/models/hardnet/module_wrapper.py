import source.utils.model_utils as mu
import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu

from source.core.module import ModuleWrapper

from source.models.hardnet.module import HardNetPS
from source.utils.endpoint_utils import sample_tensor_patch


class DescriptorWrapper(ModuleWrapper):

    def __init__(self, module_config, experiment_config):
        super().__init__(experiment_config)
        self.hard_net = HardNetPS()

        self.patch_size = module_config.patch_size

    def _get_forward_base_key(self):
        return du.IMAGE_GRAY

    def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
        pass

    def _get_process_base_key(self):
        return eu.KP

    def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
        image_grayi = batch[f'{du.IMAGE_GRAY}{i}'].to(device)
        kpi = endpoint[f"{eu.KP}{i}"].to(device)

        b, n = kpi.shape[:2]

        kp_patchi = sample_tensor_patch(image_grayi, kpi, self.patch_size, image_grayi.shape).squeeze(-1). \
            view(-1, 1, self.patch_size, self.patch_size)

        kp_desci = self.hard_net(kp_patchi).view(b, n, -1)

        endpoint[f'{eu.KP_DESC}{i}'] = kp_desci

    def get(self):
        return self.hard_net


"""
Legacy code
"""

# pg, pg_mask = create_patch_grid(kpi, self.patch_size, image_grayi.shape, mode='pm')
# flat_pg = grid2flat(pg.long(), image_grayi.shape[-1])
#
# kp_patchi = gather_tensor_patch(image_grayi, flat_pg * pg_mask.long()).view(b * n, 1, self.patch_size, self.patch_size)
