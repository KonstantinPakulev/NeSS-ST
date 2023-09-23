import os
import pandas as pd
import numpy as np

from source.datasets.base import utils as du
from source.evaluation import namespace as eva_ns, logging as lg


def load_classical_metric_eval(test_dir, evaluation_task,
                               methods_list):
    metric_v_list, metric_v_illum_list, metric_v_viewpoint_list = [], [], []

    for methods_dict in methods_list:
        metric_v, metric_v_illum, metric_v_viewpoint = [], [], []

        backend = methods_dict[eva_ns.BACKEND]
        metric_name = backend.split('/')[0]

        for k in methods_dict[eva_ns.METHODS].keys():
            eval_log_file_name = lg.EVAL_LOG_FILE.format(methods_dict.get(eva_ns.EVAL_TAG, eva_ns.DEFAULT_EVAL_TAG))
            eval_log_path = os.path.join(test_dir, evaluation_task, k, backend, eval_log_file_name)

            if not os.path.exists(eval_log_path):
                print(f"No {eval_log_path}. Skipping")
                metric_v.append(None)
                metric_v_illum.append(None)
                metric_v_viewpoint.append(None)
                continue

            eval_log = pd.read_csv(eval_log_path, index_col=[0])

            illum_mask = eval_log[du.SCENE_NAME].apply(lambda x: x[0]) == 'i'

            metric_vi = eval_log.filter(like=metric_name, axis=1)
            metric_vi = metric_vi.to_numpy().mean(axis=0)

            metric_v_illumi = eval_log[illum_mask].filter(like=metric_name, axis=1)
            metric_v_illumi = metric_v_illumi.to_numpy().mean(axis=0)

            metic_v_viewpointi = eval_log[~illum_mask].filter(like=metric_name, axis=1)
            metic_v_viewpointi = metic_v_viewpointi.to_numpy().mean(axis=0)

            metric_v.append(metric_vi)
            metric_v_illum.append(metric_v_illumi)
            metric_v_viewpoint.append(metic_v_viewpointi)

        metric_v_list.append(np.array(metric_v))
        metric_v_illum_list.append(np.array(metric_v_illum))
        metric_v_viewpoint_list.append(np.array(metric_v_viewpoint))

    return metric_v_list, metric_v_illum_list, metric_v_viewpoint_list
