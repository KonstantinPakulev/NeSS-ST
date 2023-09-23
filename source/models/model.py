from math import gcd
from functools import reduce
from abc import ABC, abstractmethod

import torch

import source.utils.endpoint_utils as eu
import source.models.namespace as m_ns

from source.core.model import ModelWrapper, ModelWrapperFactory, ModelModeWrapper
from source.evaluation.bindings import bind_ignite_transformers

from source.models.doh.wrapper_factory import DoHWrapperFactory
from source.models.log.wrapper_factory import LoGWrapperFactory
from source.models.shi.wrapper_factory import ShiWrapperFactory
from source.models.harris.wrapper_factory import HarrisWrapperFactory
from source.models.sift.wrapper_factory import SIFTWrapperFactory
from source.models.superpoint.wrapper_factory import SuperPointWrapperFactory
from source.models.r2d2.wrapper_factory import R2D2WrapperFactory
from source.models.keynet.wrapper_factory import KeyNetWrapperFactory
from source.models.disk.wrapper_factory import DISKWrapperFactory
from source.models.rekd.wrapper_factory import REKDWrapperFactory
from source.models.ness.wrappers_factory import NeXSWrapperFactory, XSWrapperFactory
from source.models.hardnet.wrapper_factory import HardNetWrapperFactory


SCALES = 'scales'


class SummertimeModelWrapper(ModelWrapper):

    def process_at_scale(self, engine, device, scale_batch, scale_inference_bundle, scale_endpoint, eval_params, endpoint):
        for idx in self.process_modules_idx:
            self.modules_wrappers[idx].process_at_scale(engine, device,
                                                        scale_batch, scale_inference_bundle, scale_endpoint, eval_params,
                                                        endpoint)

    def process_multi_scale(self, engine, device, batch, endpoint, eval_params):
        ms_bundle = {}

        for idx in self.process_modules_idx:
            self.modules_wrappers[idx].process_multi_scale(engine, device, batch, ms_bundle, endpoint, eval_params)


class SummertimeModelWrapperFactory(ModelWrapperFactory):

    def _create_modules_wrappers(self, model_name,
                                 model_config, models_config,
                                 experiment_config):
        if model_name == m_ns.SHI:
            return ShiWrapperFactory().create(model_config, experiment_config)

        elif model_name == m_ns.HARRIS:
            return HarrisWrapperFactory().create(model_config, experiment_config)

        elif model_name == m_ns.DOH:
            return DoHWrapperFactory().create(model_config, experiment_config)

        elif model_name == m_ns.LOG:
            return LoGWrapperFactory().create(model_config, experiment_config)

        elif model_name == m_ns.SIFT:
            return SIFTWrapperFactory().create(model_config, experiment_config)

        elif model_name == m_ns.SUPERPOINT:
            return SuperPointWrapperFactory().create(model_config, experiment_config)

        elif model_name == m_ns.R2D2:
            return R2D2WrapperFactory().create(model_config, experiment_config)

        elif model_name == m_ns.KEYNET:
            return KeyNetWrapperFactory().create(model_config, experiment_config)

        elif model_name == m_ns.DISK:
            return DISKWrapperFactory().create(model_config, experiment_config)

        elif model_name == m_ns.REKD:
            return REKDWrapperFactory().create(model_config, experiment_config)

        elif model_name in [m_ns.SHINESS,
                            m_ns.DOHNESS,
                            m_ns.LOGNESS,
                            m_ns.SHINERS]:
            return NeXSWrapperFactory().create(model_config, experiment_config)

        elif model_name in [m_ns.SHISS,
                            m_ns.SHIRS]:
            return XSWrapperFactory().create(model_config, experiment_config)

        elif model_name == m_ns.HARDNET:
            return HardNetWrapperFactory().create(model_config, experiment_config)

        else:
            raise NotImplementedError

    def _instantiate_model_wrapper(self, model, modules_wrappers, process_modules_idx,
                                   models_config, models_configs, base_model_eval_params, modules_configs):
        return SummertimeModelWrapper(model, modules_wrappers, process_modules_idx,
                                      models_config, models_configs, base_model_eval_params, modules_configs)


class SummertimeModelModeWrapper(ModelModeWrapper):

    def process_at_scale(self, engine, device, scale_batch, scale_inference_bundle, scale_endpoint, endpoint):
        self.model_wrapper.process_at_scale(engine, device,
                                            scale_batch, scale_inference_bundle, scale_endpoint, self.model_mode_eval_params,
                                            endpoint)

    def process_multi_scale(self, engine, device, batch, endpoint):
        self.model_wrapper.process_multi_scale(engine, device, batch, endpoint, self.model_mode_eval_params)

    def has_multi_scaling(self):
        return self.has_model() and (SCALES in self.model_wrapper.models_config)

    def get_scales(self):
        return [1.0] + list(self.model_wrapper.models_config.scales)

    def attach(self, engine, device):
        bind_ignite_transformers(engine, device,
                                 self.dataset_mode_config, self.dataset_mode_eval_config,
                                 self.model_mode_eval_params,
                                 self.config)


"""
Support utils
"""


def get_input_size_divisor(models_configs):
    input_size_divisor = [model_config.input_size_divisor for model_config in models_configs.values()
                          if m_ns.INPUT_SIZE_DIVISOR in model_config]

    if len(input_size_divisor) == 0:
        return 1

    else:
        return reduce(lambda a, b: a * b // gcd(a, b), input_size_divisor)


"""
Legacy code
"""

# elif model_name == m_ns.CAPS:
#     return create_caps_modules_wrappers(model_config)
# from source.models.caps.model_wrappers import create_caps_modules_wrappers
# from source.models.orb.model_wrapper import create_orb_modules_wrappers
# elif model_name == m_ns.ORB:
#     return create_orb_modules_wrappers(model_config)


# def divisor_crop(image, input_size_divisor):
#     if image.shape[2] % input_size_divisor != 0:
#         new_height = (image.shape[2] // input_size_divisor) * input_size_divisor
#         offset_h = int(round((image.shape[2] - new_height) / 2.))
#     else:
#         offset_h = 0
#         new_height = image.shape[2]
#
#     if image.shape[3] % input_size_divisor != 0:
#         new_width = (image.shape[3] // input_size_divisor) * input_size_divisor
#         offset_w = int(round((image.shape[3] - new_width) / 2.))
#     else:
#         offset_w = 0
#         new_width = image.shape[3]
#
#     return image[:, :, offset_h:new_height, offset_w:new_width]

# def _prepare_ms_batch(self, batch, scale_factor):
#     batch = copy.deepcopy(batch)
#
#     for key, value in batch.items():
#         if key in [du.IMAGE1, du.IMAGE2, du.C_IMAGE1, du.C_IMAGE2]:
#             s_value = interpolate(value, mode='bilinear', scale_factor=scale_factor)
#
#             batch[key] = divisor_crop(s_value, 2)
#
#     return batch