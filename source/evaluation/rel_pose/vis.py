import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator

import source.datasets.base.utils as du
import source.evaluation.logging as lg
import source.evaluation.rel_pose.transformers as trs
import source.evaluation.namespace as eva_ns

from source.evaluation.metrics import accuracy
from source.evaluation.logging import get_htune_eval_log_path, get_test_eval_log_path
from source.evaluation.vis import BasePlotter, LinePlotter
from source.evaluation.utils import get_best_threshold
from source.utils.vis_utils import autolabel


"""
Plotting functions
"""


class RelPoseAccuracyPlotter(LinePlotter):

    def __init__(self, methods_list,
                 r_acc_list, t_acc_list,
                 vis_indices):
        super().__init__(methods_list)
        self.r_acc_list = r_acc_list
        self.t_acc_list = t_acc_list
        self.vis_indices = vis_indices

    def _get_figure_and_axes(self):
        return plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    def _plot_impl(self, fig, axes, idx, save):
        for axi in axes:
            axi.xaxis.set_minor_locator(AutoMinorLocator())

            axi.tick_params(which='both', width=2)
            axi.tick_params(which='major', length=7)
            axi.tick_params(which='minor', length=4, color='r')

            axi.set_xlabel("Threshold [degrees]", fontsize=15.0)

            axi.grid()

        axes[0].set_ylabel("Rotation accuracy (%)", fontsize=17.0)
        axes[1].set_ylabel("Translation accuracy (%)", fontsize=17.0)

        r_angles = np.linspace(1, len(self.r_acc_list[idx][0]), num=len(self.r_acc_list[idx][0]))
        t_angles = np.linspace(1, len(self.t_acc_list[idx][0]), num=len(self.t_acc_list[idx][0]))

        for j, (alias, color, line_style, r_acci) in enumerate(zip(self.aliases_list[idx],
                                                                   self.colors_list[idx], self.line_styles_list[idx],
                                                                   self.r_acc_list[idx])):
            if r_acci is not None and \
                    (self.vis_indices is None or j in self.vis_indices):
                labeli = f"{alias}:{r_acci.mean():.3f} mAA" if not save else alias
                axes[0].plot(r_angles, r_acci * 100,
                             linewidth=2, label=labeli,
                             color=color, linestyle=line_style)

        for j, (alias, color, line_style, t_acci) in enumerate(zip(self.aliases_list[idx],
                                                                   self.colors_list[idx], self.line_styles_list[idx],
                                                                   self.t_acc_list[idx])):
            if t_acci is not None and \
                    (self.vis_indices is None or j in self.vis_indices):
                labeli = f"{alias}:{t_acci.mean():.3f} mAA" if not save else alias
                axes[1].plot(t_angles, t_acci * 100,
                             linewidth=2, label=labeli,
                             color=color, linestyle=line_style)

    def _get_name(self):
        return eva_ns.R_T_ACC


class RelPosemAAPlotter(BasePlotter):

    def __init__(self, methods_list,
                 r_acc_list, t_acc_list,
                 vis_indices):
        super().__init__(methods_list)
        self.r_acc_list = r_acc_list
        self.t_acc_list = t_acc_list
        self.vis_indices = vis_indices

    def _get_figure_and_axes(self):
        return plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    def _plot_impl(self, fig, axes, idx, save):
        axes[0].set_ylabel("Rotation mAA", fontsize=17.0)
        axes[1].set_ylabel("Translation mAA", fontsize=17.0)

        axes[0].margins(0.12)
        axes[1].margins(0.12)

        i = 0

        for j, (alias, color, r_acci) in enumerate(zip(self.aliases_list[idx],
                                                       self.colors_list[idx],
                                                       self.r_acc_list[idx])):
            if r_acci is not None and \
                    (self.vis_indices is None or j in self.vis_indices):
                barj = axes[0].bar(i * 0.25, r_acci.mean(), color=color, width=0.25, alpha=0.7, label=alias)

                autolabel(axes[0], barj,
                          font_size=self.plot_params_list[idx][self._get_name()][eva_ns.FONT_SIZE],
                          precision=3)

                i += 1

        i = 0

        for j, (alias, color, t_acci) in enumerate(zip(self.aliases_list[idx],
                                                       self.colors_list[idx],
                                                       self.t_acc_list[idx])):
            if t_acci is not None and \
                    (self.vis_indices is None or j in self.vis_indices):
                barj = axes[1].bar(i * 0.25, t_acci.mean(), color=color, width=0.25, alpha=0.7, label=alias)

                autolabel(axes[1], barj,
                          font_size=self.plot_params_list[idx][self._get_name()][eva_ns.FONT_SIZE],
                          precision=3)

                i += 1

    def _get_name(self):
        return eva_ns.R_T_MAA


class RelPoseNumInlPlotter(BasePlotter):

    def __init__(self, methods_list,
                 num_inl_list,
                 vis_indices):
        super().__init__(methods_list)
        self.num_inl_list = num_inl_list
        self.vis_indices = vis_indices

    def _get_figure_and_axes(self):
        return plt.subplots(1, 1, figsize=(8, 6), dpi=100)

    def _plot_impl(self, fig, axes, idx, save):
        axes.set_ylabel("Number of inliers", fontsize=17.0)

        i = 0

        for j, (alias, color, num_inli) in enumerate(zip(self.aliases_list[idx], self.colors_list[idx],
                                                         self.num_inl_list[idx])):
            if num_inli is not None and \
                    (self.vis_indices is None or j in self.vis_indices):
                barj = axes.bar(i * 0.25, num_inli, color=color, width=0.25, alpha=0.7, label=alias)
                
                autolabel(axes, barj,
                          font_size=self.plot_params_list[idx][self._get_name()][eva_ns.FONT_SIZE],
                          precision=1)

                i += 1

        axes.margins(0.12)

    def _get_name(self):
        return eva_ns.NUM_INL


class CatRelPosemAAPlotter(BasePlotter):

    def __init__(self, methods_list, cat_names,
                 r_mAA_list, t_mAA_list,
                 vis_indices):
        super().__init__(methods_list)
        self.cat_names = cat_names
        self.r_mAA_list = r_mAA_list
        self.t_mAA_list = t_mAA_list
        self.vis_indices = vis_indices

    def _get_figure_and_axes(self):
        return plt.subplots(2, 1, figsize=(16, 9), dpi=300)

    def _plot_impl(self, fig, axes, idx, save):
        num_methods = self.r_mAA_list[idx].shape[0] if self.vis_indices is None else len(self.vis_indices)
        inum_methods = 0.74 / num_methods

        i = 0

        x = np.arange(self.r_mAA_list[idx].shape[1])

        label_font_size = self.plot_params_list[idx][self._get_name()][eva_ns.FONT_SIZE]

        for j, (alias, color, r_mAAj) in enumerate(zip(self.aliases_list[idx], self.colors_list[idx],
                                                       self.r_mAA_list[idx])):
            if r_mAAj[0] is not None and \
                    (self.vis_indices is None or j in self.vis_indices):
                barj = axes[0].bar(x + (i - (num_methods - 1) / 2) * inum_methods,
                                   r_mAAj,
                                   label=alias,
                                   width=inum_methods, color=color)
                autolabel(axes[0], barj, font_size=label_font_size)

                i += 1

        i = 0

        for j, (alias, color, t_mAAj) in enumerate(zip(self.aliases_list[idx], self.colors_list[idx],
                                                       self.t_mAA_list[idx])):
            if t_mAAj[0] is not None and \
                    (self.vis_indices is None or j in self.vis_indices):
                barj = axes[1].bar(x + (i - (num_methods - 1) / 2) * inum_methods,
                                   t_mAAj,
                                   label=alias,
                                   width=inum_methods, color=color)
                autolabel(axes[1], barj, font_size=label_font_size)

                i += 1

        for axi in axes:
            axi.set_xticks(x)
            axi.set_xticklabels(self.cat_names)

        axes[0].set_ylabel("Rotation mAA", fontsize=17.0)
        axes[1].set_ylabel("Translation mAA", fontsize=17.0)

        fig.tight_layout()

    def _get_name(self):
        return eva_ns.CAT_ABL


class HCRAccuracyPlotter(LinePlotter):

    def __init__(self, methods_list,
                 hcr_acc_list, hcr_acc_illum_list, hcr_acc_viewpoint_list):
        super().__init__(methods_list)
        self.hcr_acc_list = hcr_acc_list
        self.hcr_acc_illum_list = hcr_acc_illum_list
        self.hcr_acc_viewpoint_list = hcr_acc_viewpoint_list

    def _get_figure_and_axes(self):
        return plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)

    def _plot_impl(self, fig, axes, idx, save):
        for axi in axes:
            axi.xaxis.set_minor_locator(AutoMinorLocator())

            axi.tick_params(which='both', width=2)
            axi.tick_params(which='major', length=7)
            axi.tick_params(which='minor', length=4, color='r')

            axi.set_xlabel("Threshold [px]", fontsize=15.0)
            axi.set_ylabel("Accuracy (%)", fontsize=17.0)

            axi.grid()

        axes[0].set_title("Overall", fontsize=20)
        axes[1].set_title("Illumination", fontsize=20)
        axes[2].set_title("Viewpoint", fontsize=20)

        px_thresh = np.linspace(1, len(self.hcr_acc_list[idx][0]), num=len(self.hcr_acc_list[idx][0]))

        for alias, color, line_style, hcr_acc in zip(self.aliases_list[idx],
                                                     self.colors_list[idx], self.line_styles_list[idx],
                                                     self.hcr_acc_list[idx]):
            label = f"{alias}:{hcr_acc.mean():.3f} mAA" if not save else alias
            axes[0].plot(px_thresh, hcr_acc * 100,
                         linewidth=2, label=label,
                         color=color, linestyle=line_style)

        for alias, color, line_style, hcr_acc_illum in zip(self.aliases_list[idx],
                                                           self.colors_list[idx], self.line_styles_list[idx],
                                                           self.hcr_acc_illum_list[idx]):
            label = f"{alias}:{hcr_acc_illum.mean():.3f} mAA" if not save else alias
            axes[1].plot(px_thresh, hcr_acc_illum * 100,
                       linewidth=2, label=label,
                       color=color, linestyle=line_style)

        for alias, color, line_style, hcr_acc_viewpoint in zip(self.aliases_list[idx],
                                                               self.colors_list[idx], self.line_styles_list[idx],
                                                               self.hcr_acc_viewpoint_list[idx]):
            label = f"{alias}:{hcr_acc_viewpoint.mean():.3f} mAA" if not save else alias
            axes[2].plot(px_thresh, hcr_acc_viewpoint * 100,
                         linewidth=2, label=label,
                         color=color, linestyle=line_style)

        fig.tight_layout()

    def _get_name(self):
        return eva_ns.HCR_ACC


class HTuneRelPosemAAPlotter(LinePlotter):

    def __init__(self, methods_list,
                 htune_param,
                 mAA_list, thresh_list,
                 vis_indices=None):
        super().__init__(methods_list)
        self.htune_param = htune_param
        self.mAA_list = mAA_list
        self.thresh_list = thresh_list
        self.vis_indices = vis_indices

    def _is_r_mAA(self):
        return next(i for i in self.mAA_list[0] if i is not None).shape[0] == 2

    def _get_figure_and_axes(self):
        if self._is_r_mAA():
            return plt.subplots(1, 2, figsize=(16, 6), dpi=300)

        else:
            return plt.subplots(1, 1, figsize=(7, 5), dpi=300)

    def _plot_impl(self, fig, axes, idx, save):
        if self._is_r_mAA():
            self._plot_r_mAA(fig, axes, idx, save)

        else:
            self._plot_hcr_mAA(fig, axes, idx, save)

    def _plot_r_mAA(self, fig, axes, idx, save):
        for axi in axes:
            axi.xaxis.set_minor_locator(AutoMinorLocator())

            axi.tick_params(which='both', width=2)
            axi.tick_params(which='major', length=7)
            axi.tick_params(which='minor', length=4, color='r')

            axi.set_xlabel(eva_ns.HTUNE2NAME[self.htune_param], fontsize=15.0)

            axi.grid()

        axes[0].set_ylabel("Rotation mAA", fontsize=17.0)
        axes[1].set_ylabel("Translation mAA", fontsize=17.0)

        for j, (alias, color, line_style, mAAi) in enumerate(zip(self.aliases_list[idx],
                                                                 self.colors_list[idx], self.line_styles_list[idx],
                                                                 self.mAA_list[idx])):
            if mAAi is not None and \
                    (self.vis_indices is None or j in self.vis_indices):
                thresh = self.thresh_list[idx]

                max_idx = mAAi.argmax(axis=1)
                max_value_thr = thresh[max_idx]
                max_value = mAAi[np.arange(2), max_idx]

                label = f"{alias}: {max_value[0]:.3f} mAA; " f"{max_value_thr[0]:.2f}" \
                    if not save else f"{alias}: {max_value_thr[0]:.2f}"

                axes[0].plot(thresh, mAAi[0],
                             linewidth=2, label=label,
                             color=color, linestyle=line_style)
                axes[0].scatter(max_value_thr[0], max_value[0],
                                color=color, marker="*")

                label = f"{alias}: {max_value[0]:.3f} mAA; {max_value_thr[0]:.2f}" \
                    if not save else f"{alias}: {max_value_thr[0]:.2f}"

                axes[1].plot(thresh, mAAi[1],
                             linewidth=2, label=label,
                             color=color, linestyle=line_style)
                axes[1].scatter(max_value_thr[1], max_value[1],
                                color=color, marker="*")

    def _plot_hcr_mAA(self, fig, axes, idx, save):
        axes.xaxis.set_minor_locator(AutoMinorLocator())

        axes.tick_params(which='both', width=2)
        axes.tick_params(which='major', length=7)
        axes.tick_params(which='minor', length=4, color='r')

        axes.set_xlabel(eva_ns.HTUNE2NAME[self.htune_param], fontsize=15.0)
        axes.set_ylabel("mAA", fontsize=17.0)

        axes.grid()

        for alias, color, line_style, mAA in zip(self.aliases_list[idx],
                                                 self.colors_list[idx], self.line_styles_list[idx],
                                                 self.mAA_list[idx]):
            thresh = self.thresh_list[idx]

            max_idx = mAA.argmax()
            max_value = mAA[0, max_idx]
            max_value_thr = thresh[max_idx]

            label = f"{alias}: {max_value:.3f} mAA; {max_value_thr:.2f}" if not save else f"{alias}: {max_value_thr:.2f}"

            axes.plot(thresh, mAA[0],
                      linewidth=2, label=label,
                      color=color, linestyle=line_style)
            axes.scatter(max_value_thr, max_value, color=color, marker="*")

        fig.tight_layout()

    def _get_name(self):
        if self.htune_param in [eva_ns.LOWE_RATIO, eva_ns.INL_THRESH]:
            return self.htune_param

        else:
            raise ValueErro(self.htune_param)


"""
Threshold selection
"""


def print_best_threshold(methods_list, htune_param,
                         mAA_list, thresh_list):
    aliases_list = [[ms[eva_ns.ALIAS] for ms in msl[eva_ns.METHODS].values()] for msl in methods_list]

    for idx, (aliases, mAA, thresh) in enumerate(zip(aliases_list, mAA_list, thresh_list)):
        print(methods_list[idx][eva_ns.PLOT_PARAMS][eva_ns.PLOT_TITLE])

        for alias, mAAi in zip(aliases, mAA):
            if mAAi is not None:
                best_thresh_idx = mAAi.sum(axis=0).argmax()

                print('\t', f"{alias}: {eva_ns.HTUNE2NAME[htune_param]} is {thresh[best_thresh_idx]}.")
                if mAAi.shape[0] == 2:
                    print('\t', f"R: {mAAi[0, best_thresh_idx]:.3f} mAA, "
                                f"T: {mAAi[1, best_thresh_idx]:.3f} mAA", end='\n\n')

                else:
                    print('\t', f"HCR: {mAAi[0, best_thresh_idx]:.3f} mAA", end='\n\n')


"""
Legacy code
"""

# import matplotlib.colors as mcolors
# import matplotlib.cm as cm
# from scipy.ndimage.filters import minimum_filter

# if legend_loc is not None:
#     ax[0].legend()
#     ax[1].legend()
#
# else:
# ax[0].legend()

# def plot_projected_pcd(image, proj_grid1, world_grid_depth1,
#                        batch_idx=0,
#                        fig_size=(18, 18), alpha=0.3):
#     """
#     :param image: B x C x H x W
#     :param proj_grid1: B x N x 2
#     :param world_grid_depth1: B x N
#     :param batch_idx: int
#     :param fig_size: tuple
#     :param alpha: float
#     """
#     image = image[batch_idx].permute(1, 2, 0).cpu().numpy()
#     proj_grid = proj_grid1[batch_idx].cpu().numpy()
#     world_grid_depth1 = world_grid_depth1[batch_idx].cpu().numpy()
#
#     h, w = image.shape[:2]
#
#     proj_grid = np.round(proj_grid).astype(np.int)
#
#     proj_mask = (proj_grid[:, 0] >= 0) &\
#                 (proj_grid[:, 0] < w) & \
#                 (proj_grid[:, 1] >= 0) & \
#                 (proj_grid[:, 1] < h) & \
#                 (world_grid_depth1 != 0)
#
#     proj_grid = proj_grid[proj_mask]
#     depth = world_grid_depth1[proj_mask]
#
#     max_value = depth.max() * 2
#
#     proj_pcd_image = np.ones((h, w)) * max_value
#     proj_pcd_image[proj_grid[:, 1], proj_grid[:, 0]] = depth
#     proj_pcd_image = minimum_filter(proj_pcd_image, footprint=np.ones((5, 5)))
#     proj_pcd_image[proj_pcd_image == max_value] = 0.0
#
#     ppi = proj_pcd_image.reshape(-1)
#     normalize = mcolors.Normalize(vmin=np.min(ppi[ppi != 0]),
#                                   vmax=np.max(ppi[ppi != 0]))
#     s_map = cm.ScalarMappable(norm=normalize, cmap=cm.viridis)
#
#     proj_pcd_image = s_map.to_rgba(proj_pcd_image) * (1 - np.expand_dims(proj_pcd_image == 0.0, -1))
#
#     plt.figure(figsize=fig_size)
#     plt.imshow(image)
#     plt.imshow(proj_pcd_image, alpha=alpha)

# ax[1].errorbar(t_angles, t_acc_i * 100, yerr=np.array(t_acc_std_i) * 100,
#                linewidth=2, label=f"{name}:{t_acc_i.mean():.3f} mAA",
#                color=color, linestyle=line_style, ecolor=ecolor)

# if fig_name is not None:
#     if not os.path.exists('megadepth_eval'):
#         os.mkdir('megadepth_eval')
#
#     fig.savefig(f"megadepth_eval/{fig_name}.pdf", bbox_inches='tight')

# ax[0].errorbar(r_angles, r_acci * 100, yerr=np.array(r_acc_stdi) * 100,
#                        linewidth=2, label=f"{name}:{r_acci.mean():.3f} mAA",
#                        color=color, linestyle=line_style, ecolor=ecolor)


# def load_cat_pose_eval(test_dir, megadepth_root,
#                        methods_config,
#                        gt_r_cat, eval_type='pose_2k'):
#     r_acc, t_acc = [], []
#     r_acc_std, t_acc_std = [], []
#     num_inl = []
#
#     for k in methods_config.keys():
#         eval_log = pd.read_csv(os.path.join(test_dir, f'{eval_type}/{k}', lg.EVAL_LOG_FILE))
#         eval_log = append_gt_r(megadepth_root, eval_log)
#
#         cat_r_acc, cat_t_acc = [], []
#         cat_r_acc_std, cat_t_acc_std = [], []
#         cat_num_inl = []
#
#         for i, cat in enumerate(gt_r_cat[:-1]):
#             mask = eval_log[GT_R] >= cat
#
#             if i + 1 != len(gt_r_cat):
#                 mask &= eval_log[GT_R] < gt_r_cat[i + 1]
#
#             cat_eval_log = eval_log[mask]
#
#             cat_r_err = cat_eval_log.filter(like=R_ERR, axis=1)
#             cat_t_err = cat_eval_log.filter(like=T_ERR, axis=1)
#
#             cat_r_acci, cat_r_acc_stdi = pose_accuracy(cat_r_err.to_numpy(), 10, True)
#             cat_t_acci, cat_t_acc_stdi = pose_accuracy(cat_t_err.to_numpy(), 10, True)
#
#             cat_num_inli = cat_eval_log.filter(like=NUM_INL, axis=1).to_numpy().mean()
#
#             cat_r_acc.append(cat_r_acci)
#             cat_t_acc.append(cat_t_acci)
#
#             cat_r_acc_std.append(cat_r_acc_stdi)
#             cat_t_acc_std.append(cat_t_acc_stdi)
#
#             cat_num_inl.append(cat_num_inli)
#
#         r_acc.append(cat_r_acc)
#         t_acc.append(cat_t_acc)
#
#         r_acc_std.append(cat_r_acc_std)
#         t_acc_std.append(cat_t_acc_std)
#
#         num_inl.append(cat_num_inl)
#
#     return r_acc, t_acc, r_acc_std, t_acc_std, num_inl


# ax.set_xticks()
#
# for ax in zip(row_ax, px_threshs):
#     ax.set_xticks(pos, metrics)
#
#     for j, (m_v, mm_v) in enumerate(zip(methods.values(), methods_metrics.values())):
#
#     ax.set_title(f"Metrics {px_thresh} px; {category[1]}")
#
#     ax.set_xticks(pos + len(metrics) * 0.25 / len(methods.values()))
#     ax.set_xticklabels(metrics.values())