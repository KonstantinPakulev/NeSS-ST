import torch

import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu

from source.core.module import ModuleWrapper

from source.models.disk.module import DISK


class DetectorDescriptorWrapper(ModuleWrapper):

    def __init__(self, experiment_config):
        super().__init__(experiment_config)
        self.disk = DISK()

    def _get_forward_base_key(self):
        return du.IMAGE

    def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
        imagei = batch[f"{du.IMAGE}{i}"].to(device)

        desci, scorei = self.disk._split(self.disk.unet(imagei))

        bundle[f"{eu.DESC}{i}"] = desci
        bundle[f"{eu.SCORE}{i}"] = scorei

    def _get_process_base_key(self):
        return eu.DESC

    def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
        nms_size = eval_params.nms_size
        k = eval_params.topk

        desci = inference_bundle[f"{eu.DESC}{i}"].to(device)
        scorei = inference_bundle[f"{eu.SCORE}{i}"].to(device)

        keypointsi = self.disk.detector.nms(scorei, window_size=nms_size, cutoff=0., n=k)
        featuresi = [kp.merge_with_descriptors(desci[i]) for i, kp in enumerate(keypointsi)]

        kpi = torch.stack([f.kp[..., [1, 0]] for f in featuresi])
        kp_desci = torch.stack([f.desc for f in featuresi])

        endpoint[f"{eu.KP}{i}"] = kpi
        endpoint[f"{eu.KP_DESC}{i}"] = kp_desci

    def get(self):
        return self.disk.unet
