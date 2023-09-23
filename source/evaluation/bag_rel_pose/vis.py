import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby

import source.datasets.base.utils as du
import source.evaluation.logging as lg
import source.evaluation.namespace as eva_ns

from source.evaluation.vis import BasePlotter
from source.evaluation.bag_rel_pose.metrics import bag_grouped_mAA, bag_size_grouped_mAA
from source.utils.vis_utils import autolabel


class BagRelPosemAAPlotter(BasePlotter):

    def __init__(self, methods_list,
                 r_mAA_list, t_mAA_list,
                 vis_indices):
        super().__init__(methods_list)
        self.r_mAA_list = r_mAA_list
        self.t_mAA_list = t_mAA_list
        self.vis_indices = vis_indices

    def _get_figure_and_axes(self):
        return plt.subplots(1, 2, figsize=(14, 6), dpi=100)

    def _plot_impl(self, fig, axes, idx, save):
        axes[0].set_ylabel("Rotation mAA", fontsize=17.0)
        axes[1].set_ylabel("Translation mAA", fontsize=17.0)

        i = 0

        label_font_size = self.plot_params_list[idx][eva_ns.BAG_mAA][eva_ns.FONT_SIZE]

        for j, (alias, color, r_mAAj) in enumerate(zip(self.aliases_list[idx], self.colors_list[idx],
                                                       self.r_mAA_list[idx])):
            if r_mAAj is not None and \
                    (self.vis_indices is None or j in self.vis_indices):
                barj = axes[0].bar(i * 0.25, r_mAAj, color=color, width=0.25, alpha=0.7, label=alias)
                autolabel(axes[0], barj, font_size=label_font_size)

                i += 1

        i = 0

        for j, (alias, color, t_mAAj) in enumerate(zip(self.aliases_list[idx], self.colors_list[idx],
                                                       self.t_mAA_list[idx])):
            if t_mAAj is not None and \
                    (self.vis_indices is None or j in self.vis_indices):
                barj = axes[1].bar(i * 0.25, t_mAAj, color=color, width=0.25, alpha=0.7, label=alias)
                autolabel(axes[1], barj, font_size=label_font_size)

                i += 1

        for ax in axes:
            ax.margins(0.12)

    def _get_name(self):
        return eva_ns.BAG_mAA


class BagSizeBagRelPosemAAPlotter(BasePlotter):

    def __init__(self, methods_list,
                 bag_size_r_mAA_list, bag_size_t_mAA_list,
                 vis_indices):
        super().__init__(methods_list)
        self.bag_size_r_mAA_list = bag_size_r_mAA_list
        self.bag_size_t_mAA_list = bag_size_t_mAA_list
        self.vis_indices = vis_indices

    def _get_figure_and_axes(self):
        return plt.subplots(2, 1, figsize=(16, 9), dpi=300)

    def _plot_impl(self, fig, axes, idx, save):
        not_none_r_mAAi = next((r_mAAi for r_mAAi in self.bag_size_r_mAA_list[idx] if r_mAAi is not None), None)

        if not_none_r_mAAi is None:
            return

        num_methods = len(self.bag_size_r_mAA_list[idx]) if self.vis_indices is None else len(self.vis_indices)
        inum_methods = 0.74 / num_methods

        i = 0
        x = np.arange(len(not_none_r_mAAi))

        label_font_size = self.plot_params_list[idx][self._get_name()][eva_ns.FONT_SIZE]

        for j, (alias, color, r_mAAj) in enumerate(zip(self.aliases_list[idx], self.colors_list[idx],
                                                       self.bag_size_r_mAA_list[idx])):
            if r_mAAj is not None and \
                (self.vis_indices is None or j in self.vis_indices):
                barj = axes[0].bar(x + (i - (num_methods - 1) / 2) * inum_methods,
                                   list(r_mAAj.values()),
                                   label=alias,
                                   width=inum_methods, color=color)
                autolabel(axes[0], barj, font_size=label_font_size)

                i += 1

        i = 0

        for j, (alias, color, t_mAAj) in enumerate(zip(self.aliases_list[idx], self.colors_list[idx],
                                                       self.bag_size_t_mAA_list[idx])):
            if t_mAAj is not None and \
                    (self.vis_indices is None or j in self.vis_indices):
                barj = axes[1].bar(x + (i - (num_methods - 1) / 2) * inum_methods,
                                   list(t_mAAj.values()),
                                   label=alias,
                                   width=inum_methods, color=color)
                autolabel(axes[1], barj, font_size=label_font_size)

                i += 1

        for axi in axes:
            axi.set_xticks(x)
            axi.set_xticklabels([f"Bag size {i}" for i in list(not_none_r_mAAi.keys())])

        axes[0].set_ylabel("Rotation mAA", fontsize=17.0)
        axes[1].set_ylabel("Translation mAA", fontsize=17.0)

        for ax in axes:
            ax.margins(0.12)

        fig.tight_layout()

    def _get_name(self):
        return eva_ns.BAG_SIZE_BAG_mAA


"""
Loading functions
"""


def load_bag_rel_pose_eval(test_dir, evaluation_task,
                           methods_list,
                           r_err_thresh=10, t_err_thresh=10):
    r_mAA_list, t_mAA_list = [], []
    bag_size_r_mAA_list, bag_size_t_mAA_list = [], []

    for methods_dict in methods_list:
        r_mAA, t_mAA = [], []
        bag_size_r_mAA, bag_size_t_mAA = [], []

        backend = methods_dict[eva_ns.BACKEND]
        import_evaluation_task = methods_dict.get(eva_ns.IMPORT_EVALUATION_TASK)

        for k in methods_dict[eva_ns.METHODS].keys():
            eval_log_file_name = lg.EVAL_LOG_FILE.format(methods_dict.get(eva_ns.EVAL_TAG, eva_ns.DEFAULT_EVAL_TAG))
            eval_log_path = os.path.join(test_dir, evaluation_task, k, backend, eval_log_file_name)

            if not os.path.exists(eval_log_path):
                print(f"No {eval_log_path}. Skipping")
                r_mAA.append(None)
                t_mAA.append(None)
                bag_size_r_mAA.append(None)
                bag_size_t_mAA.append(None)
                continue

            eval_log = pd.read_csv(eval_log_path, index_col=[0])

            if import_evaluation_task is not None:
                import_eval_log_file_name = lg.EVAL_LOG_FILE.format(methods_dict.get(eva_ns.IMPORT_EVAL_TAG, eva_ns.DEFAULT_EVAL_TAG))
                import_backend = methods_dict[eva_ns.IMPORT_BACKEND]
                import_eval_log = pd.read_csv(os.path.join(test_dir, import_evaluation_task,
                                                           k, import_backend, import_eval_log_file_name), index_col=[0])

                eval_log = eval_log.drop(labels=['r_err', 't_err'], axis=1)
                eval_log = pd.merge(eval_log, import_eval_log,
                                    how='left',
                                    on=[du.SCENE_NAME, du.IMAGE_NAME1, du.IMAGE_NAME2])

            bag_groups_r_mAA = bag_grouped_mAA(eval_log[du.BAG_ID], eval_log[eva_ns.R_ERR], r_err_thresh)
            bag_groups_t_mAA = bag_grouped_mAA(eval_log[du.BAG_ID], eval_log[eva_ns.T_ERR], t_err_thresh)

            bag_size_groups_r_mAAi = dict(sorted(bag_size_grouped_mAA(bag_groups_r_mAA).items(), key=lambda x: int(x[0])))
            bag_size_groups_t_mAAi = dict(sorted(bag_size_grouped_mAA(bag_groups_t_mAA).items(), key=lambda x: int(x[0])))

            r_mAAi = np.array(list(bag_size_groups_r_mAAi.values())).mean(axis=0)
            t_mAAi = np.array(list(bag_size_groups_t_mAAi.values())).mean(axis=0)

            r_mAA.append(r_mAAi)
            t_mAA.append(t_mAAi)

            bag_size_r_mAA.append(bag_size_groups_r_mAAi)
            bag_size_t_mAA.append(bag_size_groups_t_mAAi)

        r_mAA_list.append(r_mAA)
        t_mAA_list.append(t_mAA)

        bag_size_r_mAA_list.append(bag_size_r_mAA)
        bag_size_t_mAA_list.append(bag_size_t_mAA)

    return r_mAA_list, t_mAA_list, bag_size_r_mAA_list, bag_size_t_mAA_list



