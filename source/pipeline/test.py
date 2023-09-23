import argparse
import subprocess

import source.pipeline.base.utils as su
import source.evaluation.namespace as eva_ns

from source.pipeline.base.ablation_manager import AblationManager
from source.pipeline.base.path_manager import PathManager
from source.pipeline.base.utils import get_model_prompts, get_test_dataset_prompts, get_eval_params_override, \
    get_overridden_eval_params_prompts, get_eval_tag_prompt, str2bool


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(su.MODEL_PROMPT)
    parser.add_argument(su.DATASET_PROMPT)
    parser.add_argument(su.CHECK, type=str2bool)

    parser.add_argument(su.FEATURES,
                        su.FEATURES_SHORT,
                        action='store_true')
    parser.add_argument(su.TEST,
                        su.TEST_SHORT,
                        action='store_true')
    parser.add_argument(su.ABLATION, su.ABLATION_SHORT)
    parser.add_argument(su.CRITERION, su.CRITERION_SHORT)
    parser.add_argument(su.NUM_FEATURES,
                        su.NUM_FEATURES_SHORT,
                        nargs="*")
    parser.add_argument(su.IMPORT_PARAMS, su.IMPORT_PARAMS_SHORT)

    args = parser.parse_args()

    model_prompt = args.model_prompt
    dataset_prompt = args.dataset_prompt
    check = args.check

    features = args.features
    test = args.test
    all = not features and not test
    ablation = args.ablation
    criterion = args.criterion
    num_features = args.num_features
    import_params = args.import_params

    ablation_manager = AblationManager(ablation,
                                       criterion_prompt=criterion,
                                       model_prompt=model_prompt)
    path_manager = PathManager(model_prompt, dataset_prompt,
                               check, num_features, import_params,
                               ablation_manager)

    model_prompts = get_model_prompts(model_prompt, path_manager)
    dataset_prompts = get_test_dataset_prompts(dataset_prompt, path_manager.check, num_features)
    eval_params_override = get_eval_params_override(path_manager, eva_ns.TEST)
    overidden_eval_params_prompts = get_overridden_eval_params_prompts(eval_params_override, path_manager)
    eval_tag_prompt = get_eval_tag_prompt(path_manager)

    if features or all:
        for _model_prompt in model_prompts:
            command = ['python3', 'run.py']
            command += [_model_prompt]
            command += [dataset_prompts[0]]
            command += [overidden_eval_params_prompts[0]]
            command += ['+experiment=features']
            command += [eval_tag_prompt]
            command += ['--config-name=test']

            subprocess.run(command)

    if test or all:
        for dataset_prompt, overidden_eval_params_prompt in \
                zip(dataset_prompts, overidden_eval_params_prompts):
            for _model_prompt in model_prompts:
                command = ['python3', 'run.py']
                command += [_model_prompt]
                command += [dataset_prompt]
                command += [overidden_eval_params_prompt]
                command += ['+experiment=test']
                command += [eval_tag_prompt]
                command += ['--config-name=test']

                subprocess.run(command)
