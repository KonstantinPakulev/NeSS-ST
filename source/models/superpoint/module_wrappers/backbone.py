import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu

from source.core.module import ModuleWrapper
from source.models.superpoint.modules.backbone import SuperPointBackbone


class BackboneWrapper(ModuleWrapper):

    def __init__(self, experiment_config):
        super().__init__(experiment_config)
        self.backbone = SuperPointBackbone()

    def _get_forward_base_key(self):
        return du.IMAGE

    def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
        image_key = f"{du.IMAGE_GRAY}{i}"

        imagei = batch[image_key].to(device)

        xi = self.backbone(imagei)

        bundle[f"{eu.X}{i}"] = xi

    def _get_process_base_key(self):
        return None

    def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
        pass

    def get(self):
        return self.backbone
