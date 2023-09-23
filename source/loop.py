import torch

from torch.nn.functional import interpolate
from copy import deepcopy

import source.datasets.base.utils as du
import source.core.namespace as ns

from source.core.loop import Loop, copy_dict2dict
from source.core.model import get_num_branches

from source.models.model import get_input_size_divisor
from source.utils.endpoint_utils import get_divisor_crop_rect


class SummertimeLoop(Loop):

    def _loop_iteration(self, engine, batch):
        if self.loop_mode == ns.EVAL and self.model_mode_wrapper.has_multi_scaling():
            scales = self.model_mode_wrapper.get_scales()

            endpoint = {}

            for idx, scale in enumerate(scales):
                scale_batch = {}

                input_size_divisor = get_input_size_divisor(self.model_mode_wrapper.model_wrapper.models_configs)

                idx = torch.tensor(idx, device=self.device)

                resize_and_divisor_crop(du.IMAGE, batch, scale, idx, input_size_divisor, scale_batch, self.device)
                resize_and_divisor_crop(du.IMAGE_GRAY, batch, scale, idx, input_size_divisor, scale_batch, self.device)

                scale_inference_bundle = {}
                scale_endpoint = {}

                with torch.no_grad():
                    if self.model_mode_wrapper.has_model():
                        self.model_mode_wrapper(engine, self.device, scale_batch, scale_inference_bundle, scale_endpoint)

                    self.model_mode_wrapper.process_at_scale(engine, self.device, scale_batch, scale_inference_bundle, scale_endpoint, endpoint)

                if self.engine.state.return_output:
                    for k, v in scale_batch.items():
                        endpoint[f"{k}_{scale:.4f}"] = v

                    copy_dict2dict(scale_inference_bundle, scale_endpoint)

                    for k, v in scale_endpoint.items():
                        endpoint[f"{k}_{scale:.4f}"] = v

            self.model_mode_wrapper.process_multi_scale(engine, self.device, batch, endpoint)

            return endpoint

        else:
            return super()._loop_iteration(engine, batch)


"""
Support utils
"""


def resize_and_divisor_crop(base_key, batch, scale, scale_idx, input_size_divisor, scale_batch, device):
    if scale == 1.0:
        for i in range(1, get_num_branches(base_key, batch.keys()) + 1):
            image_keyi = f"{base_key}{i}"

            scale_batch[image_keyi] = deepcopy(batch[image_keyi])
            scale_batch[f"{du.SHIFT_SCALE}{i}"] = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device).unsqueeze(0)
            scale_batch[f"{du.SCALE_IDX}{i}"] = scale_idx
    else:
        for i in range(1, get_num_branches(base_key, batch.keys()) + 1):
            image_keyi = f"{base_key}{i}"
            imagei = deepcopy(batch[image_keyi])

            resized_image = interpolate(imagei.to(device), scale_factor=scale, mode='bilinear')

            initial_shape = torch.tensor(imagei.shape[2:])
            resized_shape = torch.tensor(resized_image.shape[2:])

            rect = get_divisor_crop_rect(resized_shape, input_size_divisor)

            scale_batch[image_keyi] = resized_image[:, :, rect[0]:rect[0] + rect[2], rect[1]:rect[1] + rect[3]]
            scale_batch[f"{du.SHIFT_SCALE}{i}"] = torch.cat([rect[:2], resized_shape / initial_shape], dim=0).to(device).unsqueeze(0)
            scale_batch[f"{du.SCALE_IDX}{i}"] = scale_idx

