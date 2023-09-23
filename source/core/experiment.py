import os
import time
import shutil
from abc import ABC, abstractmethod
from omegaconf import OmegaConf
from collections import ChainMap

import torch

from hydra.utils import get_original_cwd

from ignite.handlers import ModelCheckpoint
from ignite.engine.events import Events, CallableEventWithFilter

import source.core.namespace as ns

from source.core.loop import Loop

# TODO. The same structure in core as in source


"""
Checkpoint and config variables
"""

ID = 'id'

LOG_DIR = 'log_dir'

CHECKPOINT_SAVER = 'checkpoint_saver'
OPTIMIZER_MODE = 'optimizer_mode'

LOAD_OPTIMIZER = 'load_optimizer'

CWD = 'cwd'

CLEAN_LOG_DIR = 'clean_log_dir'
DETECT_ANOMALY = 'detect_anomaly'
RETURN_OUTPUT = 'return_output'

MODE_ITER = 'mode_iter'
MODE_EPOCH = 'mode_epoch'

EVAL_PARAMS_RESERVED_KEYS = ['estimator', 'tuner']

MODEL = 'model'
OPTIMIZER = 'optimizer'
CRITERIA = 'criteria'
EVALUATION = 'evaluation'


class Experiment(ABC):

    def __init__(self, config):
        self.id = config.experiment.get(ID, str(time.time()))

        self.config = config

        self.device = init_device(self.config.experiment.device)

        clear_log_dir = self.config.experiment.get(CLEAN_LOG_DIR, False)

        if clear_log_dir:
            clean_directory(os.getcwd())

        self.model_wrapper = self._create_model_wrapper()

        self._load_models_checkpoints()

        self.main_loop, cfg_main_mode = self._create_main_loop()
        self.sub_loops = self._create_sub_loops(cfg_main_mode)

        self._bind_checkpoints()

    def run(self):
        num_epochs = self.config.experiment.num_epochs
        return_output = self.config.experiment.get(RETURN_OUTPUT, False)

        self.main_loop.engine.logger.info(f"The experiment is starting with id {self.id}")

        if self.config.experiment.get(DETECT_ANOMALY, False):
            with torch.autograd.detect_anomaly():
                output = self.main_loop.run(num_epochs, return_output)
        else:
            output = self.main_loop.run(num_epochs, return_output)

        return output

    @abstractmethod
    def _create_model_wrapper(self):
        ...

    def _create_main_loop(self):
        cfg_mode, loop_mode = get_cfg_and_loop_modes(self.config.experiment.modes[0])

        print("\n")
        print("Model name: ", self.config.models.model_name)
        print("\n")

        main_loop, dataset_mode_config, dataset_mode_eval_config = self._create_loop(cfg_mode, loop_mode, None)

        self.bind_logging(main_loop.engine, main_loop.engine, cfg_mode, dataset_mode_config, dataset_mode_eval_config)

        return main_loop, cfg_mode

    def _create_sub_loops(self, cfg_main_mode):
        sub_loops = {}

        for sub_mode in self.config.experiment.modes[1:]:
            cfg_mode, loop_mode = get_cfg_and_loop_modes(sub_mode)

            loops_so_far = {cfg_main_mode: self.main_loop, **sub_loops}

            sub_loop, dataset_mode_config, dataset_mode_eval_config = \
                self._create_loop(cfg_mode, loop_mode, loops_so_far)

            if MODE_ITER in dataset_mode_eval_config:
                raise NotImplementedError

            elif MODE_EPOCH in dataset_mode_eval_config:
                self.main_loop.engine.add_event_handler(Events.EPOCH_COMPLETED(every=dataset_mode_eval_config.mode_epoch),
                                                        UnloaderHandler(sub_loop, loops_so_far))

            else:
                raise ValueError

            self.bind_logging(sub_loop.engine, self.main_loop.engine, cfg_mode,
                              dataset_mode_config, dataset_mode_eval_config)

            sub_loops[cfg_mode] = sub_loop

        return sub_loops

    def _create_loop(self, cfg_mode, loop_mode,
                     loops):
        dataset_name = get_dataset_name_by_config_mode(self.config.datasets, cfg_mode)
        dataset_config = self.config.datasets[dataset_name]

        dataset_mode_config = dataset_config[cfg_mode]
        loader_mode_config = dataset_config.loader[cfg_mode]

        dataset_mode_eval_config = None

        if EVALUATION in dataset_config:
            dataset_mode_eval_config = dataset_config.evaluation.get(cfg_mode)

        dataset = self._create_dataset(dataset_name, dataset_mode_config, loops)
        loader = self._create_loader(dataset, loader_mode_config)

        model_mode_eval_params = get_model_mode_eval_params(self.model_wrapper.model_eval_params,
                                                            dataset_config, cfg_mode,
                                                            self.config.models.model_name)

        print(f"Eval params of mode '{cfg_mode}':")
        print(OmegaConf.to_yaml(model_mode_eval_params))

        model_mode_wrapper = self._create_model_mode_wrapper(cfg_mode,
                                                             dataset_mode_config, dataset_mode_eval_config,
                                                             model_mode_eval_params)

        criterion_mode_config = None

        if CRITERIA in self.config:
            criterion_mode_config = self.config.criteria.get(cfg_mode)

        optimizer_mode_config = None

        if OPTIMIZER in self.config:
            optimizer_mode_config = self.config.optimizer.get(cfg_mode)

        criterion_chain = self._create_criterion_chain(model_mode_wrapper,
                                                       criterion_mode_config, dataset_mode_eval_config)
        optimizer_wrapper = self._create_optimizer_wrapper(model_mode_wrapper, optimizer_mode_config)

        if loop_mode == ns.TRAIN:
            self._load_optimizer_checkpoint(optimizer_wrapper, cfg_mode)

        loop = self._instantiate_loop(self.device, loop_mode,
                                      model_mode_wrapper, criterion_chain, optimizer_wrapper,
                                      dataset, loader)

        return loop, dataset_mode_config, dataset_mode_eval_config

    @abstractmethod
    def _instantiate_loop(self, device, loop_mode,
                          model_mode_wrapper, criterion_chain, optimizer,
                          dataset, loader):
        ...

    @abstractmethod
    def _create_dataset(self, dataset_name, dataset_mode_config, loops):
        ...

    @abstractmethod
    def _create_loader(self, dataset, loader_mode_config):
        ...

    @abstractmethod
    def _create_model_mode_wrapper(self, cfg_mode,
                                   dataset_mode_config, dataset_mode_eval_config,
                                   model_mode_eval_params):
        ...

    @abstractmethod
    def _create_criterion_chain(self, model_mode_wrapper,
                                criterion_mode_config, dataset_mode_eval_config):
        ...

    @abstractmethod
    def _create_optimizer_wrapper(self, model_mode_wrapper, optimizer_config):
        ...

    @abstractmethod
    def bind_logging(self, data_engine, state_engine, cfg_mode,
                     dataset_mode_config, dataset_mode_eval_config):
        ...

    def _load_models_checkpoints(self):
        state_dicts = {}

        for key, value in self.model_wrapper.models_configs.items():
            if ns.CHECKPOINT in value:
                state_dicts[key[1]] = torch.load(get_checkpoint(self.config.experiment.get(CWD),
                                                                value.checkpoint.rel_path,
                                                                value.checkpoint.name,
                                                                MODEL), map_location='cpu')

                print(f"{MODEL.capitalize()} checkpoint {value.checkpoint.name} is loaded")

        if len(state_dicts) != 0:
            self.model_wrapper.load_state_dict(state_dicts)

    def _load_optimizer_checkpoint(self, optimizer, cfg_mode):
        if OPTIMIZER in self.config and \
                cfg_mode in self.config.optimizer:
            for key, value in self.model_wrapper.models_configs.items():
                if ns.CHECKPOINT in value and \
                        LOAD_OPTIMIZER in value.checkpoint and \
                        value.checkpoint.load_optimizer:
                    state_dict = torch.load(get_checkpoint(self.config.experiment.get(CWD),
                                                           value.checkpoint.rel_path,
                                                           value.checkpoint.name,
                                                           OPTIMIZER), map_location='cpu')

                    optimizer.load_state_dict(state_dict)

                    print(f"{OPTIMIZER.capitalize()} checkpoint {value.checkpoint.name} is loaded")

    def _bind_checkpoints(self):
        if CHECKPOINT_SAVER in self.config.experiment:
            checkpoint_saver_cfg = self.config.experiment.checkpoint_saver

            score_name = checkpoint_saver_cfg.score_name
            score_names = checkpoint_saver_cfg.score_name.split('-')

            def score_function(engine):
                total = 0

                for sn in score_names:
                    total += engine.state.metrics[sn]

                total /= len(score_names)

                return total

            bind_mode = checkpoint_saver_cfg.bind_mode

            if bind_mode in self.sub_loops:
                model_checkpoint_saver = ModelCheckpoint(os.path.join(os.getcwd(), 'checkpoints'), '',
                                                         score_function=score_function,
                                                         score_name=score_name,
                                                         n_saved=checkpoint_saver_cfg.num_saved,
                                                         require_empty=False)

                self.sub_loops[bind_mode].engine.add_event_handler(Events.EPOCH_COMPLETED,
                                                                   model_checkpoint_saver,
                                                                   {MODEL: self.model_wrapper})

                if OPTIMIZER_MODE in checkpoint_saver_cfg:
                    if checkpoint_saver_cfg.optimizer_mode == self.config.experiment.modes[0]:
                        optimizer = self.main_loop.optimizer_wrapper.optimizer

                    else:
                        optimizer = self.sub_loops[checkpoint_saver_cfg.optimizer_mode].optimizer_wrapper.optimizer

                    if optimizer is None:
                        raise ValueError(f"Optimizer in mode {checkpoint_saver_cfg.optimizer_mode} wasn't found")

                    optimizer_checkpoint_saver = ModelCheckpoint(os.path.join(os.getcwd(), 'checkpoints'), '',
                                                                 score_function=score_function,
                                                                 score_name=score_name,
                                                                 n_saved=checkpoint_saver_cfg.num_saved,
                                                                 require_empty=False)

                    self.sub_loops[bind_mode].engine.add_event_handler(Events.EPOCH_COMPLETED,
                                                                       optimizer_checkpoint_saver,
                                                                       {OPTIMIZER: optimizer})

            else:
                raise ValueError(f"No {bind_mode} loop found to bind checkpoint saver")


"""
Support utils
"""


def clean_directory(path):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)

        if os.path.isdir(file_path):
            shutil.rmtree(file_path)

        else:
            os.remove(file_path)


class UnloaderHandler:

    def __init__(self, loop, loops):
        self.loop = loop
        self.loops = loops

    def __call__(self, engine):
        for key, value in self.loops.items():
            if value.is_loaded:
                value.unload_model()

        self.loop.run(1)
        self.loop.unload_model()


def get_checkpoint(cwd, rel_path, name, prefix):
    if cwd is None:
        cwd = get_original_cwd()

    return os.path.join(cwd, 'runs', rel_path, 'checkpoints', f"{prefix}_{name}.pt")


def init_device(device):
    if device == 'cpu':
        return torch.device('cpu')

    elif torch.cuda.is_available():
        return torch.device(f'cuda:{device}')

    else:
        raise ValueError(f"CUDA is not available. No such device: {device}")


def get_cfg_and_loop_modes(mode):
    if '-' in mode:
        cfg_mode, loop_mode = mode.split('-')

        return cfg_mode, loop_mode

    else:
        return mode, mode


def get_dataset_name_by_config_mode(datasets_config, cfg_mode):
    for key, value in datasets_config.items():
        if OmegaConf.is_dict(value) and cfg_mode in value:
            return key


def get_model_mode_eval_params(model_eval_params,
                               dataset_config, cfg_mode,
                               model_name):
    dataset_model_mode_eval_params = {}

    if EVALUATION in dataset_config:
        evaluation_config = dataset_config.evaluation[cfg_mode]
        
        if ns.EVAL_PARAMS in evaluation_config:
            eval_params = evaluation_config.eval_params

            dataset_mode_common_eval_params = {k: v for k, v in eval_params.items()
                                               if not OmegaConf.is_dict(v) or k in EVAL_PARAMS_RESERVED_KEYS}

            models_names = model_name.split('_')

            if model_name in eval_params:
                dataset_model_mode_eval_params = eval_params[model_name]

            else:
                base_models_names = [i.split('#')[0] for i in models_names]
                dataset_model_mode_eval_params = eval_params.get('_'.join(base_models_names), {})

            dataset_model_mode_eval_params = OmegaConf.merge(*[dataset_mode_common_eval_params,
                                                               dataset_model_mode_eval_params])

    return OmegaConf.merge(*[model_eval_params,
                             dataset_model_mode_eval_params])


def get_datasets_configs(config):
    return OmegaConf.create(dict(ChainMap(*[config.datasets[config.datasets[get_cfg_and_loop_modes(mode)[0]].dataset_name]
                                            for mode in config.experiment.modes])))


"""
Legacy code
"""

# dataset_base_models_mode_eval_params = [eval_params[bmn]
#                                         for bmn in base_models_names
#                                         if bmn in eval_params]
#
# if len(dataset_base_models_mode_eval_params) != 0:
#     dataset_base_models_mode_eval_params = OmegaConf.merge(*dataset_base_models_mode_eval_params)
#
# else:
#     dataset_base_models_mode_eval_params = {}

# dataset_model_mode_eval_params = OmegaConf.merge(*[dataset_base_models_mode_eval_params,
#                                                    dataset_model_mode_eval_params])

# .split('-')[0]
# if log_dir is not None and log_dir in eval_params:
#     dataset_model_mode_eval_params = eval_params[log_dir]

# if not os.path.exists(checkpoints_dir):
    #     if '-' in model_name:
    #         ver_splits = model_name.split('-')
    #
    #         for i in range(len(ver_splits) - 1, 0, -1):
    #             checkpoints_dir = get_checkpoints_dir(cwd, is_train, '-'.join(ver_splits[:i]))
    #
    #             if os.path.exists(checkpoints_dir):
    #                 break

    # elif '#' in model_name:
    #     checkpoints_dir = get_checkpoints_dir(cwd, is_train, '-'.join(ver_splits[:i]))

# if is_train:
#     return os.path.join(cwd, 'runs/train', model_name, 'checkpoints')
#
# else:
#     return os.path.join(cwd, 'runs/models', model_name, 'checkpoints')

# class CheckpointSaver(ModelCheckpoint):
#
#     def __call__(self, engine, to_save):
#         if len(to_save) == 0:
#             raise RuntimeError("No objects to checkpoint found.")
#
#         self._iteration += 1
#
#         if self._score_function is not None:
#             priority = self._score_function(engine)
#
#         else:
#             priority = self._iteration
#             if (self._iteration % self._save_interval) != 0:
#                 return
#
#         if (len(self._saved) < self._n_saved) or (self._saved[0][0] < priority):
#             saved_objs = []
#
#             suffix = ""
#             if self._score_name is not None:
#                 suffix = "_{}={:.7}".format(self._score_name, abs(priority))
#
#             for name, obj in to_save.items():
#                 fname = '{}_{}_{}{}.pt'.format(self._fname_prefix, name, self._iteration, suffix)
#                 path = os.path.join(self._dirname, fname)
#
#                 self._save(obj=obj, path=path)
#                 saved_objs.append(path)
#
#             self._saved.append((priority, saved_objs))
#             self._saved.sort(key=lambda item: item[0])
#
#         if len(self._saved) > self._n_saved:
#             _, paths = self._saved.pop(0)
#             for p in paths:
#                 os.remove(p)

# @self.main_loop.engine.on(Events.ITERATION_COMPLETED(every=dataset_mode_eval_config.mode_iter))
# def on_mode_iter_event(engine):
#     print('called MODE EPOCH', sub_mode)

# UnloaderHandler(sub_loop, loops_so_far)(engine)


