import os
import ruamel.yaml

from omegaconf import OmegaConf
from pathlib import Path

import source.pipeline.base.utils as pbu

from source.pipeline.base.utils import get_eval_params_config_path


class AblationManager:

    def __init__(self, ablation,
                 criterion_prompt=None, model_prompt=None):
        self.ablation = ablation
        self.criterion_prompt = criterion_prompt
        self.model_prompt = model_prompt

        self.formatted_tag_patterns = []
        self.formatted_log_patterns = []

        if self.has_ablation():
            if self.has_criterion_ablation():
                self.ablation_config, self.criterion_name = get_criterion_ablation_config(self.criterion_prompt, self.ablation)
                self.model_name = None

            elif self.has_eval_params_ablation():
                self.ablation_config, self.model_name = get_eval_params_ablation_config(self.model_prompt, self.ablation)

            else:
                raise ValueError("Either criterion_prompt or model_prompt has to be provided")

            for sweep_value in self.get_sweep_values():
                tag_pattern, log_pattern = get_formatted_patterns(self.ablation_config, sweep_value)

                self.formatted_tag_patterns.append(tag_pattern)
                self.formatted_log_patterns.append(log_pattern)

    def has_ablation(self):
        return self.ablation is not None

    def get_ablation(self):
        return self.ablation

    def has_criterion_ablation(self):
        return self.criterion_prompt is not None

    def has_eval_params_ablation(self):
        return self.model_prompt is not None

    def get_criterion_name(self):
        return self.criterion_name

    def get_model_name(self):
        return self.model_name

    def get_sweep_param(self):
        return self.ablation_config.sweep_param

    def get_sweep_values(self):
        return self.ablation_config.sweep_values

    def get_formatted_tag_patterns(self):
        return self.formatted_tag_patterns

    def get_formatted_log_patterns(self):
        return self.formatted_log_patterns

    def create_configs(self, path_manager):
        if self.has_ablation():
            if path_manager.get_mode() == pbu.TRAIN:
                for ablation_model_name, train_log_dir, fht_log_dir, ablation_model_path in \
                        zip(path_manager.get_models_names(),
                            path_manager.get_train_log_dirs(),
                            path_manager.get_fht_log_dirs(),
                            path_manager.get_model_paths()):
                    ablation_configs_path = get_ablation_configs_path(ablation_model_path)

                    create_train_config(path_manager.get_model_rel_path(),
                                        ablation_model_name,
                                        train_log_dir,
                                        ablation_configs_path,
                                        path_manager.check)

                    create_features_htune_test_config(path_manager.get_model_rel_path(),
                                                      ablation_model_name,
                                                      fht_log_dir,
                                                      ablation_configs_path,
                                                      path_manager.check)

                    create_custom_config(path_manager.get_model_rel_path(),
                                         ablation_model_name,
                                         ablation_configs_path,
                                         path_manager.check)

            elif path_manager.get_mode() == pbu.FEATURES_HTUNE_TEST and \
                    self.has_eval_params_ablation():
                for ablation_model_name, fht_log_dir, ablation_model_path, sweep_value in \
                        zip(path_manager.get_models_names(),
                            path_manager.get_fht_log_dirs(),
                            path_manager.get_model_paths(),
                            self.get_sweep_values()):
                    ablation_configs_path = get_ablation_configs_path(ablation_model_path)

                    create_features_htune_test_config(path_manager.get_model_rel_path(),
                                                      ablation_model_name,
                                                      fht_log_dir,
                                                      ablation_configs_path,
                                                      path_manager.check,
                                                      self.model_name,
                                                      self.get_sweep_param(),
                                                      sweep_value)

                    create_custom_config(path_manager.get_model_rel_path(),
                                         ablation_model_name,
                                         ablation_configs_path,
                                         path_manager.check,
                                         self.model_name,
                                         self.get_sweep_param(),
                                         sweep_value)

    def create_eval_params_file(self, path_manager):
        if self.has_ablation() and \
                (path_manager.get_mode() == pbu.TRAIN or
                 (path_manager.get_mode() == pbu.FEATURES_HTUNE_TEST
                  and self.has_eval_params_ablation())):
            ablation_eval_params_config_path = path_manager.get_eval_params_configs_paths()[0]
            eval_params_config_path = get_eval_params_config_path(path_manager.get_eval_params_dirs_rel_paths()[0])

            eval_params_config = OmegaConf.load(eval_params_config_path)

            if os.path.exists(ablation_eval_params_config_path):
                ablation_eval_params_config = OmegaConf.load(ablation_eval_params_config_path)

            else:
                ablation_eval_params_config = OmegaConf.create({'eval_params': {}})
                ablation_eval_params_config.eval_params.estimator = eval_params_config.eval_params.estimator
                ablation_eval_params_config.eval_params.tuner = eval_params_config.eval_params.tuner

            for ablation_model_name in path_manager.get_models_names():
                ablation_eval_params_config.eval_params[ablation_model_name] = eval_params_config.eval_params[path_manager.get_model_name()]

            OmegaConf.save(ablation_eval_params_config, ablation_eval_params_config_path)


"""
Support utils
"""

def get_criterion_ablation_config(criterion_prompt, ablation):
    criterion_config, criterion_rel_path = load_config(criterion_prompt)
    criterion_name = criterion_rel_path.split('/')[1]

    ablation_config = criterion_config.train[criterion_name].ablations[ablation]

    return ablation_config, criterion_name

def get_eval_params_ablation_config(model_prompt, ablation):
    model_config, model_rel_path = load_config(model_prompt)
    model_name = model_rel_path.split('/')[1].split('_')[0]

    ablation_config = model_config.models[model_name].eval_params.ablations[ablation]

    return ablation_config, model_name

def load_config(prompt):
    rel_path = prompt.replace('+', '').replace('=', '/')

    config_path = os.path.join('config', rel_path) + '.yaml'
    config = OmegaConf.load(config_path)

    return config, rel_path

def get_formatted_patterns(ablation_config, sweep_value):
    return ablation_config.tag_pattern.format(sweep_value).replace('.', '-'),\
           ablation_config.log_pattern.format(sweep_value).replace('.', '-')

def get_ablation_configs_path(ablation_model_path):
    ablation_configs_path = os.path.join('config', ablation_model_path)

    if not os.path.exists(ablation_configs_path):
        os.makedirs(ablation_configs_path)

    return ablation_configs_path

def create_train_config(model_rel_path, ablation_model_name, train_log_dir,
                        ablation_configs_path,
                        check):
    config_path = Path(ablation_configs_path, 'train.yaml')

    if not config_path.exists() or not check:
        train_config_str = """
                    # @package _global_

                    defaults:
                      - /{0}: train

                    models:
                      model_name: '{1}'
                      log_dir: '{2}'
                    """.format(model_rel_path, ablation_model_name, train_log_dir)

        yaml = ruamel.yaml.YAML()
        yaml.dump(yaml.load(train_config_str), config_path)

def create_features_htune_test_config(model_rel_path, ablation_model_name, fht_log_dir,
                                      ablation_configs_path,
                                      check,
                                      model_name=None, sweep_param=None, sweep_value=None):
    config_path = Path(ablation_configs_path, 'features_htune_test.yaml')

    if not config_path.exists() or not check:
        if model_name is None:
            features_htune_test_config_str = """
                  # @package _global_
                  
                  defaults:
                    - /{0}: features_htune_test
                    
                  models:
                    model_name: '{1}'
                    log_dir: '{2}'
                  """.format(model_rel_path, ablation_model_name, fht_log_dir)

        else:
            features_htune_test_config_str = """
                  # @package _global_
                  
                  defaults:
                    - /{0}: features_htune_test
                
                  models:
                    model_name: '{1}'
                    log_dir: '{2}'
                                    
                    {3}:
                      eval_params:
                        {4}: {5}
                  """.format(model_rel_path, ablation_model_name, fht_log_dir,
                             model_name, sweep_param, sweep_value)

        yaml = ruamel.yaml.YAML()
        yaml.dump(yaml.load(features_htune_test_config_str), config_path)


def create_custom_config(model_rel_path, ablation_model_name,
                         ablation_configs_path,
                         check,
                         model_name=None, sweep_param=None, sweep_value=None):
    config_path = Path(ablation_configs_path, 'custom.yaml')

    if not config_path.exists() or not check:
        if model_name is None:
            custom_config_str = """
                   # @package _global_
                   
                   defaults:
                     - /{0}: custom
        
                   models:
                     model_name: '{1}'
                   """.format(model_rel_path, ablation_model_name)

        else:
            custom_config_str = """
                  # @package _global_
                  
                  defaults:
                    - /{0}: custom
    
                  models:
                    model_name: '{1}'
                              
                  {2}:
                    eval_params:
                      {3}: {4}
                  """.format(model_rel_path, ablation_model_name,
                             model_name, sweep_param, sweep_value)

        yaml = ruamel.yaml.YAML()
        yaml.dump(yaml.load(custom_config_str), config_path)
