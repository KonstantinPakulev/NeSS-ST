import os

from omegaconf import OmegaConf

import source.pipeline.base.utils as pbu
import source.evaluation.namespace as eva_ns

from source.evaluation.logging import get_htune_eval_log_path
from source.pipeline.base.utils import replace_num_features, get_eval_params_config_path


class PathManager:

    def __init__(self, model_prompt, dataset_prompt,
                 check, num_features, import_params,
                 ablation_manager):
        self.ablation_manager = ablation_manager
        self.check = check

        self.model_rel_path, self.mode = get_model_rel_path(model_prompt)
        self.eval_params_file = get_eval_params_file(self.ablation_manager.get_ablation())

        eval_params_dir_rel_path, self.base_splits = get_eval_params_dir_rel_path(dataset_prompt,
                                                                                  self.check, import_params)

        if num_features is None:
            self.eval_params_dirs_rel_paths = [eval_params_dir_rel_path]

        else:
            self.eval_params_dirs_rel_paths = replace_num_features(eval_params_dir_rel_path, num_features)

        model_name = get_model_name(self.model_rel_path)

        if self.ablation_manager.has_ablation():
            self.model_name = model_name

            model_base_names = get_model_base_names(model_name)

            self.models_names = []
            self.fht_log_dirs = []

            for ftp, flp in zip(self.ablation_manager.get_formatted_tag_patterns(),
                                self.ablation_manager.get_formatted_log_patterns()):
                self.models_names.append(get_ablation_model_name(model_base_names, ftp))
                self.fht_log_dirs.append(get_fht_log_dir(model_name,
                                                         self.ablation_manager.get_ablation(), flp))

            if self.mode == pbu.TRAIN:
                self.model_base_name = model_base_names[0]
                self.train_log_dirs = []

                for flp in self.ablation_manager.get_formatted_log_patterns():
                    self.train_log_dirs.append(get_train_log_dir(model_base_names,
                                                                 self.ablation_manager.get_ablation(), flp))

        else:
            self.models_names = [model_name]
            self.fht_log_dirs = [model_name]

            if self.mode == pbu.TRAIN:
                self.model_base_name = get_model_base_names(model_name)[0]
                self.train_log_dirs = [self.model_base_name]

    def get_dataset_name(self):
        return self.base_splits[0]

    def get_eval_tag(self):
        if self.check:
            return eva_ns.CHECK_EVAL_TAG

        else:
            return f"#{self.base_splits[1]}"

    def get_evaluation_task(self):
        return self.base_splits[2]

    def get_backend(self):
        return '/'.join([self.base_splits[3],
                         self.base_splits[4].split('_')[1]])

    def get_model_rel_path(self):
        return self.model_rel_path

    def get_mode(self):
        return self.mode

    def get_model_name(self):
        return self.model_name

    def get_model_base_name(self):
        return self.model_base_name

    def get_train_log_dirs(self):
        return self.train_log_dirs

    def get_fht_log_dirs(self):
        return self.fht_log_dirs

    def get_model_paths(self):
        if self.ablation_manager.has_ablation():
            return [os.path.join(self.model_rel_path, self.ablation_manager.get_ablation(), flp)
                    for flp in self.ablation_manager.get_formatted_log_patterns()]
        else:
            return [self.model_rel_path]

    def get_eval_params_dirs_rel_paths(self):
        return self.eval_params_dirs_rel_paths

    def get_eval_params_file(self, extension=False):
        if extension:
            return f'{self.eval_params_file}.yaml'

        else:
            return self.eval_params_file

    def get_eval_params_configs_paths(self):
        eval_params_file = self.get_eval_params_file(True)
        return [os.path.join('config', dir, eval_params_file) for dir in self.eval_params_dirs_rel_paths]

    def get_htune_eval_log_paths(self, htune_param, num_features):
        htune_dir = os.path.join('runs', 'htune', self.get_dataset_name())
        backend = self.get_backend()
        eval_tag = self.get_eval_tag()

        if num_features is None:
            htune_eval_log_pathsi = []

            for fht_log_dir in self.fht_log_dirs:
                htune_eval_log_pathsi.append(get_htune_eval_log_path(htune_dir,
                                                                     self.get_evaluation_task(),
                                                                     fht_log_dir,
                                                                     backend,
                                                                     htune_param, eval_tag))

            return [htune_eval_log_pathsi]

        else:
            htune_eval_log_paths = []

            for nf in num_features:
                htune_eval_log_pathsi = []

                for fht_log_dir in self.fht_log_dirs:
                    htune_eval_log_pathsi.append(get_htune_eval_log_path(htune_dir,
                                                                         self.get_evaluation_task(),
                                                                         fht_log_dir,
                                                                         replace_backend(backend, nf),
                                                                         htune_param, eval_tag))

                htune_eval_log_paths.append(htune_eval_log_pathsi)

            return htune_eval_log_paths

    def get_models_names(self):
        return self.models_names


"""
Support utils
"""

def get_model_rel_path(model_prompt):
    model_rel_path, mode = model_prompt.replace('+', '').split('=')
    return model_rel_path, mode


def get_model_name(model_rel_path):
    return model_rel_path.split('/')[1]


def get_ablation_model_name(model_base_names, formatted_tag_pattern):
    return '_'.join([f"{model_base_names[0]}#{formatted_tag_pattern}"] + model_base_names[1:])


def get_model_base_names(model_name):
    return model_name.split('_')


def get_train_log_dir(model_base_names, ablation, formatted_log_pattern):
    return os.path.join(model_base_names[0], ablation, formatted_log_pattern)


def get_fht_log_dir(model_name, ablation, formatted_log_pattern):
    return os.path.join(model_name, ablation, formatted_log_pattern)


def get_eval_params_file(ablation):
    return pbu.METHODS if ablation is None else pbu.ABLATION_METHODS


def get_eval_params_dir_rel_path(dataset_prompt, check, import_params):
    dataset_rel_path_splits = dataset_prompt.replace('+', '').replace('=', '/').split('/')

    base_splits = [dataset_rel_path_splits[i] for i in [1, 3, 5, 6, 7]]

    if import_params is not None:
        dataset_rel_path_splits[1] = import_params

    del dataset_rel_path_splits[4]
    dataset_rel_path_splits[2] = 'evaluation/params'

    if check:
        dataset_rel_path_splits.insert(4, pbu.CHECK)

    eval_params_dir_rel_path = '/'.join(dataset_rel_path_splits)

    if not os.path.exists(get_eval_params_config_path(eval_params_dir_rel_path)):
        eval_params_dir_rel_path = '/'.join(dataset_rel_path_splits[:-2])

        if not os.path.exists(get_eval_params_config_path(eval_params_dir_rel_path)):
            raise ValueError()

    return eval_params_dir_rel_path, base_splits


def replace_backend(backend, nf):
    splits = backend.split('/')[:-1]

    return '/'.join(splits + [nf])
