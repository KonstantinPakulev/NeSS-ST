import argparse
import subprocess

import source.pipeline.base.utils as su
import source.evaluation.namespace as eva_ns

from source.pipeline.base.ablation_manager import AblationManager
from source.pipeline.base.path_manager import PathManager
from source.pipeline.base.utils import get_model_prompts, get_htune_dataset_prompts, get_eval_params_override, \
    get_overridden_eval_params_prompts, get_eval_tag_prompt, update_lowe_ratio, update_inl_thresh, str2bool


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(su.MODEL_PROMPT)
    parser.add_argument(su.DATASET_PROMPT)
    parser.add_argument(su.CHECK, type=str2bool)

    parser.add_argument(su.FEATURES,
                        su.FEATURES_SHORT,
                        action='store_true')
    parser.add_argument(su.HTUNE_LOWE_RATIO,
                        su.HTUNE_LOWE_RATIO_SHORT,
                        action='store_true')
    parser.add_argument(su.HTUNE_INL_THRESH,
                        su.HTUNE_INL_THRESH_SHORT,
                        action='store_true')
    parser.add_argument(su.ABLATION, su.ABLATION_SHORT)
    parser.add_argument(su.CRITERION, su.CRITERION_SHORT)
    parser.add_argument(su.NUM_FEATURES,
                        su.NUM_FEATURES_SHORT,
                        nargs="*")

    args = parser.parse_args()

    model_prompt = args.model_prompt
    dataset_prompt = args.dataset_prompt
    check = args.check

    features = args.features
    htune_lowe_ratio = args.htune_lowe_ratio
    htune_inl_thresh = args.htune_inl_thresh
    all = not features and not htune_lowe_ratio and not htune_inl_thresh
    ablation = args.ablation
    criterion = args.criterion
    num_features = args.num_features

    ablation_manager = AblationManager(ablation,
                                       criterion_prompt=criterion,
                                       model_prompt=model_prompt)
    path_manager = PathManager(model_prompt, dataset_prompt,
                               check, num_features, None,
                               ablation_manager)

    dataset_lowe_ratio_prompt = f"{dataset_prompt}={eva_ns.LOWE_RATIO}"

    model_prompts = get_model_prompts(model_prompt, path_manager)
    dataset_lowe_ratio_prompts = get_htune_dataset_prompts(dataset_lowe_ratio_prompt, check, num_features)
    eval_params_override = get_eval_params_override(path_manager, eva_ns.HTUNE)
    overidden_eval_params_prompts = get_overridden_eval_params_prompts(eval_params_override, path_manager)
    eval_tag_prompt = get_eval_tag_prompt(path_manager)

    if features or all:
        ablation_manager.create_configs(path_manager)
        ablation_manager.create_eval_params_file(path_manager)

        for _model_prompt in model_prompts:
            command = ['python3', 'run.py']
            command += [_model_prompt]
            command += [dataset_lowe_ratio_prompts[0]]
            command += [overidden_eval_params_prompts[0]]
            command += ['+experiment=features']
            command += [eval_tag_prompt]
            command += ['--config-name=htune']

            subprocess.run(command)

    if htune_lowe_ratio or htune_inl_thresh or all:
        models_names = path_manager.get_models_names()
        eval_params_configs_paths = path_manager.get_eval_params_configs_paths()

        dataset_inl_thresh_prompt = f"{dataset_prompt}={eva_ns.INL_THRESH}"
        dataset_inl_thresh_prompts = get_htune_dataset_prompts(dataset_inl_thresh_prompt,
                                                               check, num_features)

        lr_htune_eval_log_paths = path_manager.get_htune_eval_log_paths(eva_ns.LOWE_RATIO, num_features)
        it_htune_eval_log_paths = path_manager.get_htune_eval_log_paths(eva_ns.INL_THRESH, num_features)

        for dataset_lowe_ratio_prompt, dataset_inl_thresh_prompt, overidden_eval_params_prompt, \
            lr_htune_eval_log_pathsi, it_htune_eval_log_pathsi,\
            eval_params_config_path in \
                zip(dataset_lowe_ratio_prompts, dataset_inl_thresh_prompts, overidden_eval_params_prompts,
                    lr_htune_eval_log_paths, it_htune_eval_log_paths,
                    eval_params_configs_paths):
                for _model_prompt,\
                    lr_htune_eval_log_path, it_htune_eval_log_path,\
                    model_name in zip(model_prompts,
                                      lr_htune_eval_log_pathsi, it_htune_eval_log_pathsi,
                                      models_names):
                    if htune_lowe_ratio or all:
                        command = ['python3', 'run.py']
                        command += [_model_prompt]
                        command += [dataset_lowe_ratio_prompt]
                        command += [overidden_eval_params_prompt]
                        command += ['+experiment=htune']
                        command += [eval_tag_prompt]
                        command += ['--config-name=htune']

                        subprocess.run(command)

                        update_lowe_ratio(lr_htune_eval_log_path, eval_params_config_path, model_name)

                    if htune_inl_thresh or all:
                        command = ['python3', 'run.py']
                        command += [_model_prompt]
                        command += [dataset_inl_thresh_prompt]
                        command += [overidden_eval_params_prompt]
                        command += ['+experiment=htune']
                        command += [eval_tag_prompt]
                        command += ['--config-name=htune']

                        subprocess.run(command)

                        update_inl_thresh(it_htune_eval_log_path, eval_params_config_path, model_name)
