import os

from source.core.experiment import Experiment
from source.core.criterion import CriterionChain
from source.core.optimizer import OptimizerWrapper

from source.loop import SummertimeLoop
from source.models.model import SummertimeModelWrapperFactory, SummertimeModelModeWrapper

from source.datasets.dataset import create_dataset, create_loader
from source.evaluation.bindings import bind_ignite_logging

from source.models.ness.wrappers_factory import create_shiness_criteria


class SummertimeExperiment(Experiment):

    def _create_model_wrapper(self):
        return SummertimeModelWrapperFactory().create_from_config(self.config.models,
                                                                  self.config.experiment)

    def _instantiate_loop(self, device, loop_mode,
                          model_mode_wrapper, criterion_chain, optimizer_wrapper,
                          dataset, loader):
        return SummertimeLoop(device, loop_mode,
                              model_mode_wrapper, criterion_chain, optimizer_wrapper,
                              dataset, loader)

    def _create_dataset(self, dataset_name, dataset_mode_config, loops):
        return create_dataset(dataset_name, dataset_mode_config, self.model_wrapper.models_configs, loops, self.config)

    def _create_loader(self, dataset, loader_mode_config):
        return create_loader(dataset, loader_mode_config)

    def _create_model_mode_wrapper(self, cfg_mode, dataset_mode_config, dataset_mode_eval_config, model_mode_eval_params):
        return SummertimeModelModeWrapper(self.model_wrapper, cfg_mode,
                                          dataset_mode_config, dataset_mode_eval_config,
                                          model_mode_eval_params,
                                          self.config)

    def _create_criterion_chain(self, model_mode_wrapper,
                                criterion_mode_config, dataset_mode_eval_config):
        if criterion_mode_config is not None:
            criteria_wrappers = create_shiness_criteria(model_mode_wrapper, criterion_mode_config)

            return CriterionChain(criteria_wrappers, dataset_mode_eval_config)

        else:
            return None

    def _create_optimizer_wrapper(self, model_mode_wrapper, optimizer_mode_config):
        if optimizer_mode_config is not None:
            return OptimizerWrapper(model_mode_wrapper.parameters(), optimizer_mode_config)

        else:
            return None

    def bind_logging(self, data_engine, state_engine, cfg_mode, dataset_mode_config, dataset_mode_eval_config):
        bind_ignite_logging(data_engine, state_engine, cfg_mode,
                            dataset_mode_config, dataset_mode_eval_config,
                            self.config)
