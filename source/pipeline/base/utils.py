import os
import ruamel.yaml

from omegaconf import OmegaConf
from pathlib import Path

import source.pose.estimators.namespace as est_ns
import source.evaluation.namespace as eva_ns

from source.evaluation.logging import read_htune_eval_log
from source.evaluation.utils import get_best_threshold


MODEL_PROMPT = 'model_prompt'
DATASET_PROMPT = 'dataset_prompt'
VAL_DATASET_PROMPT = 'val_dataset_prompt'
CRITERION_PROMPT = 'criterion_prompt'
OPTIMIZER_PROMPT = 'optimizer_prompt'
CHECK = 'check'

FEATURES = '--features'
HTUNE_LOWE_RATIO = '--htune_lowe_ratio'
HTUNE_INL_THRESH = '--htune_inl_thresh'
TEST = '--test'
CRITERION = '--criterion'
ABLATION = '--ablation'
NUM_FEATURES = '--num_features'
IMPORT_PARAMS = '--import_params'

FEATURES_SHORT = '-f'
HTUNE_LOWE_RATIO_SHORT = '-lr'
HTUNE_INL_THRESH_SHORT = '-it'
TEST_SHORT = '-t'
CRITERION_SHORT = '-c'
ABLATION_SHORT = '-a'
NUM_FEATURES_SHORT = '-nf'
IMPORT_PARAMS_SHORT = '-ip'

NUM = 'num'

METHODS = 'methods'
ABLATION_METHODS = 'ablation_methods'

TRAIN = 'train'
FEATURES_HTUNE_TEST = 'features_htune_test'

"""
Configs
"""

def add_checkpoint_to_configs(train_log_dir, model_rel_path, path_manager):
    checkpoints_rel_path = os.path.join('train', train_log_dir)
    checkpoints_path = os.path.join('runs', checkpoints_rel_path, 'checkpoints')

    checkpoints_names = [i for i in os.listdir(checkpoints_path) if i.startswith('model')]
    checkpoints_scores = [float(i.split('=')[1].split('.pt')[0]) for i in checkpoints_names]

    best_score_idx = max(range(len(checkpoints_scores)), key=checkpoints_scores.__getitem__)
    checkpoint_name = checkpoints_names[best_score_idx].split("model_")[1].split(".pt")[0]

    model_base_name = path_manager.get_model_base_name()

    add_checkpoint_to_config(os.path.join('config', model_rel_path, 'features_htune_test.yaml'),
                             model_base_name,
                             checkpoints_rel_path, checkpoint_name,
                             path_manager.check)

    add_checkpoint_to_config(os.path.join('config', model_rel_path, 'custom.yaml'),
                             model_base_name,
                             checkpoints_rel_path, checkpoint_name,
                             path_manager.check)

def update_lowe_ratio(htune_eval_log_path, eval_params_config_path, model_name):
    mAA, thresh = read_htune_eval_log(htune_eval_log_path)

    eval_params_config = OmegaConf.load(eval_params_config_path)
    eval_params_config.eval_params[model_name].matcher.lowe_ratio = thresh[mAA.argmax()]

    OmegaConf.save(eval_params_config, eval_params_config_path)

def update_inl_thresh(htune_eval_log_path, eval_params_config_path, model_name):
    mAA, thresh = read_htune_eval_log(htune_eval_log_path)

    eval_params_config = OmegaConf.load(eval_params_config_path)
    eval_params_config.eval_params[model_name].estimator.inl_thresh = thresh[mAA.argmax()]

    OmegaConf.save(eval_params_config, eval_params_config_path)

"""
Prompts
"""

def get_model_prompts(model_prompt, path_manager):
    if path_manager.ablation_manager.has_ablation():
        return [f"+{amp}={path_manager.get_mode()}"
                for amp in path_manager.get_model_paths()]

    else:
        return [model_prompt]

def get_train_dataset_prompt(dataset_prompt, check):
    return "=".join(get_dataset_prompts(dataset_prompt.replace('=', '/'), check, None, -1)[0].rsplit("/", 1))

def get_htune_dataset_prompts(dataset_prompt, check, num_features):
    return get_dataset_prompts(dataset_prompt, check, num_features, -3)

def get_test_dataset_prompts(dataset_prompt, check, num_features):
    return get_dataset_prompts(dataset_prompt, check, num_features, -2)

def get_eval_params_override(path_manager, mode):
    return f"datasets.{path_manager.get_dataset_name()}.evaluation.{mode}"

def get_overridden_eval_params_prompts(eval_params_override, path_manager):
    eval_params_dirs_rel_paths = path_manager.get_eval_params_dirs_rel_paths()
    eval_params_file = path_manager.get_eval_params_file()

    return [f"+{dir}@{eval_params_override}={eval_params_file}" for dir in eval_params_dirs_rel_paths]

def get_sweep_param_overrides(ablation_manager):
    if ablation_manager.has_ablation():
        if ablation_manager.has_criterion_ablation():
            return [f"criteria.train.{ablation_manager.get_criterion_name()}.{ablation_manager.get_sweep_param()}={v}"
                    for v in ablation_manager.get_sweep_values()]

        elif ablation_manager.has_eval_params_ablation():
            return [f"models.{ablation_manager.get_model_name()}.eval_params.{ablation_manager.get_sweep_param()}={v}"
                    for v in ablation_manager.get_sweep_values()]

    else:
        return None

def get_train_experiment_prompt(check):
    if check:
        return '+experiment/check=train'

    else:
        return '+experiment=train'

def get_eval_tag_prompt(path_manager):
    return f"+experiment.eval_tag='{path_manager.get_eval_tag()}'"

"""
Support utils
"""

def str2bool(v):
    if v == 'True':
        return True
    elif v == 'False':
        return False
    else:
        raise ValueError()

def add_checkpoint_to_config(config_path,
                             base_model_name,
                             checkpoint_rel_path, checkpoint_name,
                             check):
    yaml = ruamel.yaml.YAML()

    config_path = Path(config_path)
    config = yaml.load(config_path)

    if not 'checkpoint' in config['models'][base_model_name] or not check:
        config['models'][base_model_name] = {'checkpoint':
                                                 {'rel_path': checkpoint_rel_path,
                                                  'name': checkpoint_name}}

        yaml.dump(config, config_path)

def get_dataset_prompts(dataset_prompt, check, num_features, at):
    if check:
        splits = dataset_prompt.split('/')
        splits.insert(at, CHECK)
        dataset_prompt = '/'.join(splits)

    if num_features is not None:
        splits = dataset_prompt.split('=')
        if NUM in splits[1]:
            return ['='.join([splits[0], p]) for p in replace_num_features(splits[1], num_features)]
        else:
            return ['='.join([p, splits[1]]) for p in replace_num_features(splits[0], num_features)]

    else:
        return [dataset_prompt]

def replace_num_features(path, num_features):
    paths = []
    splits = path.split('_')

    for nf in num_features:
        splits_c = splits.copy()
        splits_c[-1] = nf

        paths.append('_'.join(splits_c))

    return paths

def get_eval_params_config_path(eval_params_dir_rel_path):
    return os.path.join('config', eval_params_dir_rel_path, f"{METHODS}.yaml")
