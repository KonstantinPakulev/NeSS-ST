import os
import pandas as pd
import numpy as np

from source.datasets.base import utils as du

from source.evaluation import namespace as eva_ns, logging as lg
from source.evaluation.logging import get_htune_eval_log_path, get_test_eval_log_path, read_htune_eval_log
from source.evaluation.metrics import accuracy


def load_rel_pose_eval(test_dir, evaluation_task,
                       methods_list,
                       r_err_thresh=10, t_err_thresh=10):
    r_acc_list, t_acc_list, num_inl_list = [], [], []

    for methods_dict in methods_list:
        r_acc, t_acc, num_inl = [], [], []

        backend = methods_dict[eva_ns.BACKEND]

        for k, v in methods_dict[eva_ns.METHODS].items():
            eval_log_path = get_test_eval_log_path(test_dir, evaluation_task, k, backend,
                                                   methods_dict.get(eva_ns.EVAL_TAG, eva_ns.DEFAULT_EVAL_TAG),
                                                   rel_path=v.get(eva_ns.REL_PATH))
            
            if not os.path.exists(eval_log_path):
                print(f"No {eval_log_path}. Skipping")
                r_acc.append(None)
                t_acc.append(None)
                num_inl.append(None)
                continue

            eval_log = pd.read_csv(eval_log_path, index_col=[0])

            r_err = eval_log.filter(like=eva_ns.R_ERR, axis=1)
            t_err = eval_log.filter(like=eva_ns.T_ERR, axis=1)

            r_acci = accuracy(r_err.to_numpy(), r_err_thresh).mean(axis=-1)
            t_acci = accuracy(t_err.to_numpy(), t_err_thresh).mean(axis=-1)

            num_inli = eval_log.filter(like=eva_ns.NUM_INL, axis=1).to_numpy().mean()

            r_acc.append(r_acci)
            t_acc.append(t_acci)
            num_inl.append(num_inli)

        r_acc_list.append(np.array(r_acc))
        t_acc_list.append(np.array(t_acc))
        num_inl_list.append(np.array(num_inl))

    return r_acc_list, t_acc_list, num_inl_list


def load_cat_rel_pose_eval(test_dir, evaluation_task,
                           methods_list,
                           cats,
                           cat_name=du.SCENE_NAME,
                           r_err_thresh=10, t_err_thresh=10):
    r_mAA_list, t_mAA_list = [], []

    for methods_dict in methods_list:
        r_mAA, t_mAA = [], []

        backend = methods_dict[eva_ns.BACKEND]

        for k, v in methods_dict[eva_ns.METHODS].items():
            eval_log_path = get_test_eval_log_path(test_dir, evaluation_task, k, backend,
                                                   methods_dict.get(eva_ns.EVAL_TAG, eva_ns.DEFAULT_EVAL_TAG),
                                                   rel_path=v.get(eva_ns.REL_PATH))

            if not os.path.exists(eval_log_path):
                print(f"No {eval_log_path}. Skipping")
                r_mAA.append([None] * len(cats))
                t_mAA.append([None] * len(cats))
                continue

            eval_log = pd.read_csv(eval_log_path, index_col=[0])

            cat_r_mAA, cat_t_mAA = [], []

            for cat in cats:
                cat_eval_log = eval_log[eval_log[cat_name].isin(cat)]

                cat_r_err = cat_eval_log.filter(like=eva_ns.R_ERR, axis=1)
                cat_t_err = cat_eval_log.filter(like=eva_ns.T_ERR, axis=1)

                cat_r_acci = accuracy(cat_r_err.to_numpy(), r_err_thresh)
                cat_t_acci = accuracy(cat_t_err.to_numpy(), t_err_thresh)

                cat_r_mAA.append(np.mean(cat_r_acci))
                cat_t_mAA.append(np.mean(cat_t_acci))

            r_mAA.append(cat_r_mAA)
            t_mAA.append(cat_t_mAA)

        r_mAA_list.append(np.array(r_mAA))
        t_mAA_list.append(np.array(t_mAA))

    return r_mAA_list, t_mAA_list


def load_hpatches_rel_pose_eval(test_dir, evaluation_task,
                                methods_list,
                                hcr_err_thresh=5):
    hcr_acc_list, hcr_acc_illum_list, hcr_acc_viewpoint_list = [], [], []

    for methods_dict in methods_list:
        hcr_acc, hcr_acc_illum, hcr_acc_viewpoint = [], [], []

        backend = methods_dict[eva_ns.BACKEND]

        for k, v in methods_dict[eva_ns.METHODS].items():
            eval_log_path = get_test_eval_log_path(test_dir, evaluation_task, k, backend,
                                                   methods_dict.get(eva_ns.EVAL_TAG, eva_ns.DEFAULT_EVAL_TAG),
                                                   rel_path=v.get(eva_ns.REL_PATH))

            if not os.path.exists(eval_log_path):
                print(f"No {eval_log_path}. Skipping")
                hcr_acc.append(None)
                hcr_acc_illum.append(None)
                hcr_acc_viewpoint.append(None)
                continue

            eval_log = pd.read_csv(eval_log_path, index_col=[0])

            illum_mask = eval_log[du.IMAGE_NAME1].apply(lambda x: x.split('/')[0][0]) == 'i'

            hcr_err = eval_log.filter(like=eva_ns.HCR_ERR, axis=1)

            hcr_acci = accuracy(hcr_err.to_numpy(), hcr_err_thresh).mean(axis=-1)
            hcr_acc_illumi = accuracy(hcr_err[illum_mask].to_numpy(), hcr_err_thresh).mean(axis=-1)
            hcr_acc_viewpointi = accuracy(hcr_err[~illum_mask].to_numpy(), hcr_err_thresh).mean(axis=-1)

            hcr_acc.append(hcr_acci)
            hcr_acc_illum.append(hcr_acc_illumi)
            hcr_acc_viewpoint.append(hcr_acc_viewpointi)

        hcr_acc_list.append(np.array(hcr_acc))
        hcr_acc_illum_list.append(np.array(hcr_acc_illum))
        hcr_acc_viewpoint_list.append(np.array(hcr_acc_viewpoint))

    return hcr_acc_list, hcr_acc_illum_list, hcr_acc_viewpoint_list


def load_rel_pose_htune(htune_dir, evaluation_task,
                        methods_list,
                        htune_param):
    mAA_list, thresh_list = [], []

    for methods_dict in methods_list:
        mAA, thresh = [], None

        backend = methods_dict[eva_ns.BACKEND]

        for k, v in methods_dict[eva_ns.METHODS].items():
            eval_log_path = get_htune_eval_log_path(htune_dir, evaluation_task, k, backend,
                                                    htune_param,
                                                    methods_dict.get(eva_ns.EVAL_TAG, eva_ns.DEFAULT_EVAL_TAG),
                                                    rel_path=v.get(eva_ns.REL_PATH))

            if not os.path.exists(eval_log_path):
                print(f"No {eval_log_path}. Skipping")
                mAA.append(None)
                continue

            mAAi, thresh = read_htune_eval_log(eval_log_path)

            mAA.append(mAAi)

        mAA_list.append(np.array(mAA))
        thresh_list.append(np.array(thresh))

    return mAA_list, thresh_list
