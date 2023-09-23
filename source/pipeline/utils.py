# import os
# import ruamel.yaml
#
# from pathlib import Path
# from omegaconf import OmegaConf
#
# from source.evaluation import namespace as eva_ns
# from source.evaluation.logging import get_htune_eval_log_path, read_htune_eval_log
# from source.evaluation.utils import get_best_threshold
# from source.pose.estimators import namespace as est_ns
#
# MODEL_PROMPT = 'model_prompt'
# ABLATION = 'ablation'
#
# DATASET_PROMPT = 'dataset_prompt'
# DATASET_LOWE_RATIO_PROMPT = 'dataset_lowe_ratio_prompt'
# DATASET_INL_THRESH_PROMPT = 'dataset_inl_thresh_prompt'
#
# EVAL_PARAMS_PROMPT = 'eval_params_prompt'
# EVAL_PARAMS_OVERRIDE = 'eval_params_override'
# NUM_FEATURES = 'num_features'
# OPTIMIZER_PROMPT = 'optimizer_prompt'
# EVAL_TAG = 'eval_tag'
# MODE = '--mode'
# CRITERION_PROMPT = '--criterion_prompt'
#
# MODE_SHORT = '-m'
# CRITERION_PROMPT_SHORT = '-c'
#
# METHODS = 'methods'
# ABLATION_METHODS = 'ablation_methods'
# CHECK_METHODS = 'check_methods'
#
#
# def get_formatted_patterns(ablation_config, sweep_value):
#     return ablation_config.tag_pattern.format(sweep_value).replace('.', '-'),\
#            ablation_config.log_pattern.format(sweep_value).replace('.', '-')
#
#
# def get_model_names(model_rel_path, f_tag_pattern):
#     model_name, base_model_names = get_base_model_names(model_rel_path)
#
#     abl_model_name = '_'.join([f"{base_model_names[0]}#{f_tag_pattern}"] + base_model_names[1:])
#
#     return model_name, abl_model_name, base_model_names
#
# """
# Prompt management
# """
#
# def get_train_experiment_prompt(dataset_prompt):
#     if 'check' in dataset_prompt:
#         return '+experiment/check=train'
#
#     else:
#         return '+experiment=train'
#
#
# def get_ablation_model_prompt(model_prompt, ablation, f_log_pattern):
#     model_rel_path, mode = get_model_rel_path(model_prompt)
#
#     return f"+{os.path.join(model_rel_path, ablation, f_log_pattern)}={mode}", model_rel_path
#
#
# def get_overridden_eval_params_prompt(eval_params_prompt, eval_params_override,
#                                       eval_params_file):
#     eval_params_rel_path = eval_params_prompt.replace('+', '').replace('=', '/')
#
#     eval_params_rel_path_splits = eval_params_rel_path.split('/')
#     overridden_eval_params_rel_path = '/'.join(eval_params_rel_path_splits[:-1])
#
#     return f"+{overridden_eval_params_rel_path}@{eval_params_override}={eval_params_file}", overridden_eval_params_rel_path
#
#
# def get_n_features_overridden_eval_params_prompt(over_eval_params_prompt, num_features):
#     over_eval_params_splits = over_eval_params_prompt.split('@')
#     eval_params_splits = over_eval_params_splits[0].split('/')
#     eval_params_splits[-1] = f"num_{num_features}"
#
#     return f"{'/'.join(eval_params_splits)}@{over_eval_params_splits[1]}"
#
#
# def replace_num_features(prompt, num_features):
#     prompt_splits = prompt.split('=')
#     p_head_splits = prompt_splits[0].split('_')
#
#     p_head_splits[-1] = num_features
#
#     return '='.join(['_'.join(p_head_splits), *prompt_splits[1:]])
#
# """
# Path management
# """
#
# def get_train_log_dir(base_model_names, ablation, f_log_pattern):
#     return os.path.join(base_model_names[0], ablation, f_log_pattern)
#
#
# def get_fht_log_dir(base_model_names, ablation, f_log_pattern):
#     return os.path.join('_'.join(base_model_names), ablation, f_log_pattern)
#
#
# def prompt2config_path(prompt):
#     rel_path = prompt.replace('+', '').replace('=', '/')
#     return os.path.join('config', rel_path) + '.yaml'
#
#
# def get_eval_params_file(eval_params_file, dataset_prompt):
#     if 'check' in dataset_prompt:
#         return CHECK_METHODS
#
#     else:
#         return eval_params_file
#
#
# def get_eval_params_config_path(eval_params_rel_path, eval_params_file):
#     return os.path.join('config', eval_params_rel_path, f'{eval_params_file}.yaml')
#
#
# def get_checkpoints_rel_path(train_log_dir):
#     return os.path.join('train', train_log_dir)
#
#
# def get_checkpoints_dir(train_log_dir):
#     return os.path.join('runs', get_checkpoints_rel_path(train_log_dir), 'checkpoints')
#
#
# def get_model_rel_path(model_prompt):
#     model_rel_path, mode = model_prompt.replace('+', '').split('=')
#     return model_rel_path, mode
#
#
# def get_features_htune_test_config_path(model_rel_path):
#     return os.path.join(model_rel_path, 'features_htune_test.yaml')
#
#
# def get_custom_config_path(abl_model_rel_path):
#     return os.path.join(abl_model_rel_path, 'custom.yaml')
#
#
# def get_base_model_names(model_rel_path):
#     model_name = model_rel_path.split('/')[1]
#     base_model_names = model_name.split('_')
#
#     return model_name, base_model_names
#
# """
# Config management
# """
#
# def get_ablation_config(criterion_prompt, ablation):
#     criterion_rel_path = criterion_prompt.replace('+', '').replace('=', '/')
#     criterion_name = criterion_rel_path.split('/')[1]
#
#     criterion_config_path = os.path.join('config', criterion_rel_path) + '.yaml'
#     criterion_config = OmegaConf.load(criterion_config_path)
#
#     ablation_config = criterion_config.train[criterion_name].ablations[ablation]
#
#     return ablation_config, criterion_name
#
#
# def add_checkpoint_to_configs(train_log_dir, model_rel_path, base_model_name1):
#     checkpoints_rel_path = get_checkpoints_rel_path(train_log_dir)
#     checkpoints_dir = get_checkpoints_dir(train_log_dir)
#
#     checkpoints_names = [i for i in os.listdir(checkpoints_dir) if i.startswith('model')]
#     checkpoints_scores = [float(i.split('=')[1].split('.pt')[0]) for i in checkpoints_names]
#
#     best_score_idx = max(range(len(checkpoints_scores)), key=checkpoints_scores.__getitem__)
#     checkpoint_name = checkpoints_names[best_score_idx].split("model_")[1].split(".pt")[0]
#
#     add_checkpoint_to_config(get_features_htune_test_config_path(model_rel_path),
#                              base_model_name1,
#                              checkpoints_rel_path, checkpoint_name)
#
#     add_checkpoint_to_config(get_custom_config_path(model_rel_path),
#                              base_model_name1,
#                              checkpoints_rel_path, checkpoint_name)
#
#
# def add_checkpoint_to_config(config_path,
#                              base_model_name,
#                              checkpoint_rel_path, checkpoint_name):
#     yaml = ruamel.yaml.YAML()
#
#     config_path = Path(config_path)
#     config = yaml.load(config_path)
#
#     config['models'][base_model_name] = {'checkpoint':
#                                           {'rel_path': checkpoint_rel_path,
#                                            'name': checkpoint_name}}
#
#     yaml.dump(config, config_path)
#
#
# def update_lowe_ratio(dataset_config, fht_log_dir, eval_tag,
#                       eval_params_config_path, model_name):
#     htune_eval_log_path = get_htune_eval_log_path(os.path.join('runs', 'htune', dataset_config.dataset_name),
#                                                   dataset_config.evaluation_task, fht_log_dir, dataset_config.backend,
#                                                   eva_ns.LOWE_RATIO, eval_tag)
#
#     r_mAA, t_mAA, thresh = read_htune_eval_log(htune_eval_log_path)
#
#     eval_params_config = OmegaConf.load(eval_params_config_path)
#
#     eval_params_config.eval_params[model_name].matcher.lowe_ratio = thresh[get_best_threshold(r_mAA, t_mAA)]
#
#     OmegaConf.save(eval_params_config, eval_params_config_path)
#
#
# def update_inl_thresh(dataset_config, fht_log_dir, eval_tag,
#                       eval_params_config_path, model_name):
#     htune_eval_log_path = get_htune_eval_log_path(os.path.join('runs', 'htune', dataset_config.dataset_name),
#                                                   dataset_config.evaluation_task, fht_log_dir, dataset_config.backend,
#                                                   eva_ns.INL_THRESH, eval_tag)
#
#     r_mAA, t_mAA, thresh = read_htune_eval_log(htune_eval_log_path)
#
#     eval_params_config = OmegaConf.load(eval_params_config_path)
#
#     estimator_name = eval_params_config.eval_params.estimator.name
#
#     if estimator_name in est_ns.TWO_VIEW_ESTIMATORS:
#         eval_params_config.eval_params[model_name].estimator.inl_thresh = thresh[get_best_threshold(r_mAA, t_mAA)]
#
#     else:
#         raise NotImplementedError(f"No such estimator {estimator_name}")
#
#     OmegaConf.save(eval_params_config, eval_params_config_path)
#
# # TODO. FIlter functions in this file; some are not used at all
#
#
