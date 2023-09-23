import numpy as np
import torch

from skimage.feature import ORB

import source.datasets.base.utils as du
import source.utils.model_utils as mu
import source.utils.endpoint_utils as eu

from source.core.module import ModuleWrapper


def create_orb_modules_wrappers(model_config):
    modules_wrappers = []

    for key, value in model_config.modules.items():
        if key == mu.DETECTOR_DESCRIPTOR:
            modules_wrappers.append(DetectorDescriptorWrapper())

        else:
            raise NotImplementedError(key)

    return modules_wrappers


class DetectorDescriptorWrapper(ModuleWrapper):

    def _get_forward_base_key(self):
        return du.IMAGE_GRAY

    def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
        pass

    def _get_process_base_key(self):
        return None

    def _process_branch(self, engine, device, i, batch, endpoint, eval_params):
        k = eval_params.topk

        image_grayi = batch[f"{du.IMAGE_GRAY}{i}"]

        if image_grayi.shape[0] > 1:
            raise NotImplementedError

        orb = ORB(n_keypoints=k)
        orb.detect_and_extract(image_grayi.numpy()[0, 0].astype(np.float64))

        endpoint[f"{eu.KP}{i}"] = torch.tensor(orb.keypoints[None, ...], dtype=torch.float)
        endpoint[f"{eu.KP_DESC}{i}"] = torch.tensor(orb.descriptors[None, ...], dtype=torch.bool)

    def get(self):
        return None
