import argparse
import os
import yaml
import subprocess
import ruamel.yaml

import source.pipeline.base.utils as su
import source.evaluation.namespace as eva_ns

from source.pipeline.base.ablation_manager import AblationManager
from source.pipeline.base.path_manager import PathManager

from source.pipeline.base.utils import get_model_prompts, get_train_dataset_prompt, get_test_dataset_prompts, \
    get_eval_params_override, get_overridden_eval_params_prompts, get_sweep_param_overrides, get_train_experiment_prompt, \
    add_checkpoint_to_configs, str2bool


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(su.MODEL_PROMPT)
    parser.add_argument(su.DATASET_PROMPT)
    parser.add_argument(su.CRITERION_PROMPT)
    parser.add_argument(su.OPTIMIZER_PROMPT)
    parser.add_argument(su.VAL_DATASET_PROMPT)
    parser.add_argument(su.CHECK, type=str2bool)

    parser.add_argument(su.ABLATION, su.ABLATION_SHORT)

    args = parser.parse_args()

    model_prompt = args.model_prompt
    dataset_prompt = args.dataset_prompt
    criterion_prompt = args.criterion_prompt
    optimizer_prompt = args.optimizer_prompt
    val_dataset_prompt = args.val_dataset_prompt
    check = args.check

    ablation = args.ablation

    ablation_manager = AblationManager(ablation, criterion_prompt=criterion_prompt)
    path_manager = PathManager(model_prompt, val_dataset_prompt,
                               check, None, None,
                               ablation_manager)

    ablation_manager.create_configs(path_manager)
    ablation_manager.create_eval_params_file(path_manager)

    model_prompts = get_model_prompts(model_prompt, path_manager)
    dataset_prompt = get_train_dataset_prompt(dataset_prompt, path_manager.check)
    sweep_param_overrides = get_sweep_param_overrides(ablation_manager)
    val_dataset_prompt = get_test_dataset_prompts(val_dataset_prompt, path_manager.check, None)[0]
    eval_params_override = get_eval_params_override(path_manager, eva_ns.TEST)
    overidden_eval_params_prompt = get_overridden_eval_params_prompts(eval_params_override, path_manager)[0]

    train_log_dirs = path_manager.get_train_log_dirs()
    model_paths = path_manager.get_model_paths()

    for i, (_model_prompt, train_log_dir, model_path) in enumerate(zip(model_prompts, train_log_dirs, model_paths)):
        command = ['python3', 'run.py']
        command += [_model_prompt]
        command += [dataset_prompt]
        command += [criterion_prompt]

        if sweep_param_overrides is not None:
            command += [sweep_param_overrides[i]]

        command += [optimizer_prompt]
        command += [val_dataset_prompt]
        command += [overidden_eval_params_prompt]
        command += [get_train_experiment_prompt(path_manager.check)]
        command += ['experiment.clean_log_dir=True']
        command += ['--config-name=train']

        subprocess.run(command)

        add_checkpoint_to_configs(train_log_dir, model_path, path_manager)
