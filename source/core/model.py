import re
import torch
from collections import ChainMap
from omegaconf import OmegaConf
from abc import ABC, abstractmethod
import itertools

from ignite.engine.engine import Events

import torch.nn as nn

import source.core.namespace as ns


PROCESS = 'process'

FORWARD_MODULES = 'forward_modules'
FREEZE = 'freeze'
SAVE = 'save'
REMAP = 'remap'


# TODO. Not model wrapper but more like a container or a sequnce of wrappers. Or model?

class ModelWrapper:

    def __init__(self, model, modules_wrappers,
                 process_modules_idx,
                 models_config, models_configs, model_eval_params, modules_configs):
        self.model = model
        self.modules_wrappers = modules_wrappers

        self.process_modules_idx = process_modules_idx

        self.models_config = models_config
        self.models_configs = models_configs
        self.model_eval_params = model_eval_params
        self.modules_configs = modules_configs

    def __call__(self, engine, device, batch, inference_bundle, endpoint):
        for m_w in self.modules_wrappers:
            m_w(engine, device, batch, inference_bundle, endpoint)

    def process(self, engine, device, batch, inference_bundle, endpoint, eval_params):
        for idx in self.process_modules_idx:
            self.modules_wrappers[idx].process(engine, device, batch, inference_bundle, endpoint, eval_params)

    def load_state_dict(self, state_dicts):
        ordered_state_dicts = []

        for model_name, state_dict in state_dicts.items():
            if self.models_config[model_name].checkpoint.get(REMAP, False):
                idx_mapping = {'.'.join(key.split('.')[1:]): key.split('.')[0]
                               for key in self.model.state_dict()}

                ordered_state_dict = {}

                for key, value in state_dict.items():
                    if key in idx_mapping:
                        ordered_state_dict[idx_mapping[key] + '.' + key] = value

            else:
                idx_mapping = {module.default_idx: module.idx
                               for module in self.models_config[model_name].modules.values() if module.default_idx != -1}

                ordered_state_dict = {}

                for key, value in state_dict.items():
                    idx = int(key.split('.')[0])

                    if idx in idx_mapping:
                        new_idx = idx_mapping[idx]
                        new_key = key.replace(f'{idx}', f'{new_idx}', 1)

                        ordered_state_dict[new_key] = value

            ordered_state_dicts.append(ordered_state_dict)

        merged_state_dict = dict(ChainMap(*ordered_state_dicts))

        strict = False not in [ns.CHECKPOINT in model_config and
                               True not in [module.get(FREEZE, False) for module in model_config.modules.values()]
                               for model_config in self.models_configs.values()]

        self.model.load_state_dict(merged_state_dict, strict=strict)

    def state_dict(self):
        save_modules = [mc.idx for mc in self.modules_configs if mc.get(SAVE, True)]

        if len(save_modules) == len(self.modules_configs):
            return self.model.state_dict()

        else:
            idx_mapping = {idx: str(i) for i, idx in enumerate(save_modules)}

            state_dict = {}

            for key, value in self.model.state_dict().items():
                splits = key.split('.')
                idx = int(splits[0])

                if idx in save_modules:
                    new_key_idx = idx_mapping[idx]
                    new_key = '.'.join([new_key_idx] + splits[1:])

                    state_dict[new_key] = value

            return state_dict


class ModelWrapperFactory:

    def create_from_config(self, models_config,
                           experiment_config):
        models_configs = get_models_configs(models_config)
        base_model_eval_params = get_base_model_eval_params(models_configs)
        modules_configs = get_modules_configs(models_configs)

        modules_wrappers = []

        for key, model_config in models_configs.items():
            modules_wrappers.extend(self._create_modules_wrappers(key[1],
                                                                  model_config, models_config,
                                                                  experiment_config))

        modules_idx = [mc.idx for mc in modules_configs]
        modules_wrappers = [modules_wrappers[i] for i in modules_idx]

        model = nn.ModuleList([mr.get() for mr in modules_wrappers])

        freezed_modules = [mc.idx for mc in modules_configs if mc.get(FREEZE, False)]

        for name, param in model.named_parameters():
            if int(name.split('.')[0]) in freezed_modules:
                param.requires_grad = False

        process_modules_idx = sorted([(mc.process_idx, mc.idx) for mc in modules_configs],
                                     key=lambda x: x[0])
        process_modules_idx = [i[1] for i in process_modules_idx]

        return ModelWrapper(model, modules_wrappers, process_modules_idx,
                            models_config, models_configs, base_model_eval_params, modules_configs)

    @abstractmethod
    def _create_modules_wrappers(self, model_name,
                                 model_config, models_configs,
                                 experiment_config):
        ...


class ModelModeWrapper:

    def __init__(self, model_wrapper, cfg_mode,
                 dataset_mode_config, dataset_mode_eval_config,
                 model_mode_eval_params,
                 config):
        f_modules_idxs = None

        for value in model_wrapper.models_configs.values():
            if cfg_mode in value and FORWARD_MODULES in value[cfg_mode]:
                if f_modules_idxs is None:
                    f_modules_idxs = []

                f_modules = value[cfg_mode].forward_modules

                if f_modules is not None:
                    f_modules_idxs.extend([value.modules[fm].idx for fm in f_modules])

        if f_modules_idxs is None:
            self.model_wrapper = model_wrapper

        elif len(f_modules_idxs) == 0:
            self.model_wrapper = None

        else:
            f_modules_idxs = sorted(f_modules_idxs)

            modules_wrappers = [model_wrapper.modules_wrappers[idx] for idx in f_modules_idxs]

            model = nn.ModuleList([model_wrapper.model[idx] for idx in f_modules_idxs])

            process_modules_idx = [idx for idx in model_wrapper.process_modules_idx if idx in f_modules_idxs]

            self.model_wrapper = ModelWrapper(model, modules_wrappers, process_modules_idx,
                                              model_wrapper.models_config, model_wrapper.models_configs,
                                              model_wrapper.model_eval_params, model_wrapper.modules_configs)

        self.dataset_mode_config = dataset_mode_config
        self.dataset_mode_eval_config = dataset_mode_eval_config
        self.model_mode_eval_params = model_mode_eval_params
        self.config = config

    def __call__(self, engine, device, batch, inference_bundle, endpoint):
        self.model_wrapper(engine, device, batch, inference_bundle, endpoint)

    def process(self, engine, device, batch, inference_bundle, endpoint):
        self.model_wrapper.process(engine, device, batch, inference_bundle, endpoint, self.model_mode_eval_params)

    def has_model(self):
        return self.model_wrapper is not None

    def do_process(self):
        return self.dataset_mode_config.get(PROCESS, True) and self.has_model()

    def train(self):
        self.model_wrapper.model.train()

    def eval(self):
        self.model_wrapper.model.eval()

    def to(self, device):
        self.model_wrapper.model.to(device)

    def cpu(self):
        self.model_wrapper.model.cpu()

    def parameters(self):
        yield from self.model_wrapper.model.parameters()

    def attach(self, engine, device):
        pass


"""
Support utils
"""


def get_modules_configs(models_configs):
    return list(itertools.chain.from_iterable([list(model_config.modules.values())
                                               for model_config in models_configs.values()]))


def get_models_configs(models_config):
    models_configs = {}

    for model_name in models_config.model_name.split('_'):
        if '#' in model_name:
            base_model_name = model_name.split('#')[0]

        else:
            base_model_name = model_name

        models_configs[(model_name, base_model_name)] = models_config[base_model_name]

    return models_configs


def get_base_model_eval_params(models_configs):
    base_models_eval_params = [model_config.eval_params for model_config in models_configs.values()
                               if ns.EVAL_PARAMS in model_config]

    if len(base_models_eval_params) == 0:
        return {}

    elif len(base_models_eval_params) == 1:
        return base_models_eval_params[0]

    else:
        return OmegaConf.merge(*base_models_eval_params)


def get_num_branches(base_key, keys):
    if base_key is not None:
        exp = re.compile(f"{base_key}\d$")

        return len([key for key in keys if exp.match(key)])

    else:
        return 0


"""
Legacy code
"""

# if '-' in model_name:
#     base_model_name = model_name.split('-')[0]
#
# el

# def is_on_device(self, device):
#     if sum(1 for _ in self.model_wrapper.model.parameters()) != 0:
#         return next(self.model_wrapper.model.parameters()).device == device
#
#     else:
#         return True


# MULTI_SCALE = 'multi_scale'
# def _prepare_ms_batch(self, batch, scale_factor):
#     raise NotImplementedError

# class ModuleWrapperMS(ModuleWrapper, ABC):
#
#     def __init__(self, device, forward_mode, multi_scale):
#         super().__init__(device, forward_mode, True)
#         self.multi_scale = multi_scale
#
#     def process(self, engine, batch, endpoint, eval_params):
#         if self.forward_mode == SINGLE:
#             if self.multi_scale is not None:
#                 self.single_process_ms(engine, batch, endpoint, eval_params)
#
#             else:
#                 self.single_process(engine, batch, endpoint, eval_params)
#
#         elif self.forward_mode == PAIR:
#             if self.multi_scale is not None:
#                 self.pair_process_ms(engine, batch, endpoint, eval_params)
#
#             else:
#                 self.pair_process(engine, batch, endpoint, eval_params)
#
#         else:
#             raise NotImplementedError
#
#     @abstractmethod
#     def single_process_ms(self, engine, batch, endpoint, eval_params):
#         ...
#
#     @abstractmethod
#     def pair_process_ms(self, engine, batch, endpoint, eval_params):
#         ...

 # if has_ms:
        #     scale_factors = self.models_config.multi_scale.scale_factors
        #
        #     for i, scale_factor in enumerate(scale_factors):
        #         batch_i = self._prepare_ms_batch(batch, scale_factor)
        #
        #         for m_w in self.modules_wrappers:
        #             if m_w.supports_ms:
        #                 m_w(engine, batch_i, endpoint, bundle, i + 1)
