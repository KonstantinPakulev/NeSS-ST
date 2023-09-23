import torch

import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu

from source.core.module import ModuleWrapper

from source.models.disk.module import DISK


class DetectorWrapper(ModuleWrapper):

    def __init__(self, experiment_config):
        super().__init__(experiment_config)
        self.disk = DISK()

    def _get_forward_base_key(self):
        return du.IMAGE

    def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
        imagei = batch[f"{du.IMAGE}{i}"].to(device)

        _, scorei = self.disk._split(self.disk.unet(imagei))

        bundle[f"{eu.SCORE}{i}"] = scorei

    def _get_process_base_key(self):
        return eu.SCORE

    def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
        nms_size = eval_params.nms_size
        k = eval_params.topk

        scorei = inference_bundle[f"{eu.SCORE}{i}"].to(device)

        keypoints = self.disk.detector.nms(scorei, window_size=nms_size, cutoff=0., n=k)

        kp_scorei = torch.stack([i.logp for i in keypoints])
        kpi = torch.stack([i.xys[..., [1, 0]] for i in keypoints]).float() + 0.5

        endpoint[f"{eu.KP_SCORE}{i}"] = kp_scorei
        endpoint[f"{eu.KP}{i}"] = kpi

    def get(self):
        return self.disk.unet