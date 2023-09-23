import cv2
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator

import source.evaluation.classical.namespace
import source.evaluation.logging as lg
import source.datasets.base.utils as du
import source.evaluation.namespace
import source.evaluation.namespace as eva_ns

from source.evaluation.vis import LinePlotter
from source.utils.vis_utils import torch2cv


"""
Plotting functions
"""

class ClassicalPlotter(LinePlotter):

    def __init__(self, methods_list,
                 metric_v_list, metric_v_illum_list, metric_v_viewpoint_list,
                 vis_indices):
        super().__init__(methods_list)
        self.metric_v_list = metric_v_list
        self.metric_v_illum_list = metric_v_illum_list
        self.metric_v_viewpoint_list = metric_v_viewpoint_list
        self.vis_indices = vis_indices

    def _get_figure_and_axes(self):
        return plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)

    def _plot_impl(self, fig, axes, idx, save):
        for axi in axes:
            axi.xaxis.set_minor_locator(AutoMinorLocator())

            axi.tick_params(which='both', width=2)
            axi.tick_params(which='major', length=7)
            axi.tick_params(which='minor', length=4, color='r')

            axi.set_xlabel("Threshold [px]", fontsize=15.0)
            axi.set_ylabel(self.plot_params_list[idx][eva_ns.Y_AXIS_LABEL], fontsize=17.0)

            axi.grid()

        axes[0].set_title("Overall", fontsize=20)
        axes[1].set_title("Illumination", fontsize=20)
        axes[2].set_title("Viewpoint", fontsize=20)

        px_thresh = np.linspace(1, len(self.metric_v_list[idx][0]), num=len(self.metric_v_list[idx][0]))

        for alias, color, line_style, metric_v in zip(self.aliases_list[idx],
                                                      self.colors_list[idx], self.line_styles_list[idx],
                                                      self.metric_v_list[idx]):
            if metric_v is not None:
                axes[0].plot(px_thresh, metric_v,
                             linewidth=2, label=alias,
                             color=color, linestyle=line_style)

        for alias, color, line_style, metric_v_illum in zip(self.aliases_list[idx],
                                                       self.colors_list[idx], self.line_styles_list[idx],
                                                       self.metric_v_illum_list[idx]):
            if metric_v_illum is not None:
                axes[1].plot(px_thresh, metric_v_illum,
                             linewidth=2, label=alias,
                             color=color, linestyle=line_style)

        for alias, color, line_style, metric_v_viewpoint in zip(self.aliases_list[idx],
                                                           self.colors_list[idx], self.line_styles_list[idx],
                                                           self.metric_v_viewpoint_list[idx]):
            if metric_v_viewpoint is not None:
                axes[2].plot(px_thresh, metric_v_viewpoint,
                             linewidth=2, label=alias,
                             color=color, linestyle=line_style)

        fig.tight_layout()

    def _get_name(self):
        return None


"""
Drawing functions
"""


def draw_cv_keypoints(image, kp,
                      batch_id=0,
                      draw_mask=None, as_pixels=False,
                      normalize=False, color=(0, 255, 0)):
    """
    :param image: B x C x H x W, :type torch.tensor
    :param kp: B x N x 2
    :param batch_id: int
    :param draw_mask: B x N
    :param normalize: bool
    :param color: tuple (r, g, b)
    """
    cv_image = torch2cv(image[batch_id], normalize=normalize)

    if draw_mask is not None:
        kp = kp[batch_id][draw_mask[batch_id]]
    else:
        kp = kp[batch_id]

    if not as_pixels:
        cv_kp = to_cv2_keypoint(kp)
        return cv2.drawKeypoints(cv_image, cv_kp, None, color=color)

    else:
        if torch.is_tensor(kp):
            kp = kp.long().cpu().numpy()

        cv_image[kp[:, 0], kp[:, 1], :] = np.array(color)

        return cv_image


def draw_cv_matches(image1, image2,
                    kp1, kp2, matches, match_mask=None,
                    batch_id=0,
                    match_color=(0, 255, 0), single_point_color=(255, 0, 0),
                    num_match_sample=None):
    """
    :param image1: B x C x H x W, :type torch.tensor
    :param image2: B x C x H x W, :type torch.tensor
    :param kp1: B x N x 2, :type torch.int64
    :param kp2: B x N x 2, :type torch.int64
    :param matches: B x N, :type torch.bool
    :param match_mask: B x N, :type torch.bool
    :param batch_id: int
    :param match_color: (r, g, b) :type tuple
    :param single_point_color: (r, g, b) :type tuple
    :param num_match_sample: int
    """
    cv_image1 = torch2cv(image1[batch_id])
    cv_image2 = torch2cv(image2[batch_id])

    cv_kp1 = to_cv2_keypoint(kp1[batch_id])
    cv_kp2 = to_cv2_keypoint(kp2[batch_id])

    matches = to_cv2_dmatch(kp1.shape[1], matches[batch_id])

    if match_mask is not None:
        flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS

        match_mask = match_mask[batch_id]

        if num_match_sample is not None:
            s_match_mask = torch.zeros_like(match_mask)
            idx = torch.arange(match_mask.shape[0])

            sample_idx = torch.randperm(match_mask.sum())[:num_match_sample]
            s_match_mask[idx[match_mask][sample_idx]] = True

            match_mask = s_match_mask

        if torch.is_tensor(match_mask):
            match_mask = match_mask.long().detach().cpu().numpy()

        else:
            match_mask = match_mask.astype(np.long)

        match_mask = match_mask.tolist()

    else:
        flags = None

    return cv2.drawMatches(cv_image1, cv_kp1, cv_image2, cv_kp2,
                           matches, None,
                           matchColor=match_color,
                           singlePointColor=single_point_color,
                           matchesMask=match_mask, flags=flags)


def draw_cv_epipolar_lines(image2, kp2, w_line1,
                           batch_idx=0,
                           kp_color=(0, 255, 0),
                           draw_mask=None):
    """
    :param image2: B x 1 x H x W
    :param kp2: B x N x 2
    :param w_line1: B x N x 3
    :param batch_idx: int
    :param kp_color: tuple (r,g,b)
    :param draw_mask B x N
    """
    w = image2.shape[-1]

    cv_image2 = cv2.cvtColor(torch2cv(image2[batch_idx]), cv2.COLOR_BGR2RGB)

    kp2 = kp2[batch_idx]
    w_line1 = w_line1[batch_idx]

    if draw_mask is not None:
        kp2 = kp2[draw_mask[batch_idx]]
        w_line1 = w_line1[draw_mask[batch_idx]]

    cv_kp_match1 = to_cv2_keypoint(kp2)
    cv_line2 = w_line1.cpu().numpy()

    for linei in cv_line2:
        color = tuple(np.random.randint(0, 255, 3).tolist())

        x0, y0 = map(int, [0, -linei[2] / linei[1]])
        x1, y1 = map(int, [w, -(linei[2] + linei[0] * w) / linei[1]])

        cv_image2 = cv2.line(cv_image2, (x0, y0), (x1, y1), color, 1, lineType=cv2.LINE_AA)

    cv_image2 = cv2.drawKeypoints(cv_image2, cv_kp_match1, None, color=kp_color)
    cv_image2 = cv2.cvtColor(cv_image2, cv2.COLOR_RGB2BGR)

    return cv_image2


"""
Support utils
"""


def to_cv2_keypoint(kp):
    """
    :param kp: N x 2
    """
    if torch.is_tensor(kp):
        kp = kp.detach().cpu().numpy()

    kp = list(map(lambda x: cv2.KeyPoint(x[1], x[0], 0), kp))

    return kp


def to_cv2_dmatch(num_kp, matches):
    """
    :param num_kp: int
    :param matches: N
    """
    matches = matches.detach().cpu().numpy()
    return list(map(lambda x: cv2.DMatch(x[0], x[1], 0, 0), zip(np.arange(0, num_kp), matches)))


"""
Legacy code
"""

# """
# Printing functions
# """
#
#
# def print_mma(names, mma, mma_illum, mma_viewpoint, thresholds):
#     for name, mmai, mma_illumi, mma_viewpointi in zip(names, mma, mma_illum, mma_viewpoint):
#         print('\t', f"{name}:")
#         for t in thresholds:
#             print('\t', f"Overall MMA@{t}px: {mmai[int(t) - 1]:.3f}")
#             print('\t', f"Illumination MMA@{t}px: {mma_illumi[int(t) - 1]:.3f}")
#             print('\t', f"Viewpoint MMA@{t}px: {mma_viewpointi[int(t) - 1]:.3f}")
#             print()

# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# import source.evaluation.logging as lg
#
# from source.evaluation.standard.transformers import MMA, NUM_GT_MATCH, NUM_GT_MM
# from source.evaluation.utils import append_gt_r, GT_R
# from source.utils.vis_utils import autolabel

# def plot_mma(names,
#              mma, num_gt_match, num_gt_mm,
#              colors,
#              px_thresh):
#     fig, ax= plt.subplots(1, 3,
#                           figsize=(16, 4))
#
#     ax[0].set_ylabel("MMA", fontsize=17.0)
#     ax[1].set_ylabel("Num. GT matches (correct)", fontsize=12.0)
#     ax[2].set_ylabel("Num. GT mutual matches", fontsize=12.0)
#
#     width = 0.5
#
#     for i, (name, mmai, color) in enumerate(zip(names, mma, colors)):
#         for j, thr in enumerate(px_thresh):
#             namej = name if j == 0 else None
#             barj = ax[0].bar(width + i * width,
#                              mmai[j],
#                              color=color, width=width, alpha=1.0 * (len(px_thresh) - j) / len(px_thresh),
#                              label=namej)
#
#             autolabel(ax[0], barj, label=f'{thr}px')
#
#     for i, (name, num_gt_matchi, color) in enumerate(zip(names, num_gt_match, colors)):
#         for j, thr in enumerate(px_thresh):
#             namej = name if j == 0 else None
#             barj = ax[1].bar(width + i * width,
#                              num_gt_matchi[j],
#                              color=color, width=width, alpha=1.0 * (len(px_thresh) - j) / len(px_thresh),
#                              label=namej)
#
#             autolabel(ax[1], barj, label=f'{thr}px')
#
#     for i, (name, num_gt_mmi, color) in enumerate(zip(names, num_gt_mm, colors)):
#         bari = ax[2].bar(width + i * width,
#                          num_gt_mmi,
#                          color=color, width=width,
#                          label=name)
#
#         autolabel(ax[2], bari)
#
#     for axi in ax:
#         axi.grid()
#         axi.legend(loc='lower right')
#
#         axi.margins(0.1 * len(names))

# def load_metrics_eval(test_dir, megadepth_root,
#                       methods_config,
#                       gt_r_cat, eval_type='metrics_2k'):
#     mma = []
#     num_gt_match = []
#     num_gt_mm = []
#
#     for k in methods_config.output_keys():
#         eval_log = pd.read_csv(os.path.join(test_dir, f'{eval_type}/{k}', lg.EVAL_LOG_FILE))
#         eval_log = append_gt_r(megadepth_root, eval_log)
#
#         cat_mma = []
#         cat_num_gt_match = []
#         cat_num_gt_mm = []
#
#         for i, cat in enumerate(gt_r_cat[:-1]):
#             mask = eval_log[GT_R] >= cat
#
#             if i + 1 != len(gt_r_cat):
#                 mask &= eval_log[GT_R] < gt_r_cat[i + 1]
#
#             cat_eval_log = eval_log[mask]
#
#             cat_mmai = cat_eval_log.filter(like=MMA, axis=1).to_numpy().mean(axis=0)
#             cat_num_gt_matchi = cat_eval_log.filter(like=NUM_GT_MATCH, axis=1).to_numpy().mean(axis=0)
#             cat_num_gt_mmi = cat_eval_log[[NUM_GT_MM]].to_numpy().mean(axis=0)
#
#             cat_mma.append(cat_mmai)
#             cat_num_gt_match.append(cat_num_gt_matchi)
#             cat_num_gt_mm.append(cat_num_gt_mmi)
#
#         mma.append(cat_mma)
#         num_gt_match.append(cat_num_gt_match)
#         num_gt_mm.append(cat_num_gt_mm)
#
#     return mma, num_gt_match, num_gt_mm

# for i, (row_ax, metrics) in enumerate(zip(axes, metrics_groups)):
#     pos = np.arange(len(metrics))
#
#     for ax, px_thresh in zip(row_ax, px_threshs):
#         ax.set_xticks(pos, metrics)
#
#         for j, (m_v, mm_v) in enumerate(zip(methods.values(), methods_metrics.values())):
#             barj = ax.bar(pos + j * 0.25, mm_v[i][f'metrics_{px_thresh}'],
#                           color=m_v['color'], width=0.25, alpha=0.6, label=m_v['alias'])
#
#             autolabel(ax, barj)
#
#         ax.set_title(f"Metrics {px_thresh} px; {category[1]}")
#
#         ax.set_xticks(pos + len(metrics) * 0.25 / len(methods.values()))
#         ax.set_xticklabels(metrics.values())
#
#         ax.legend()
#
# plt.tight_layout()
# plt.show()

# for i, (row_ax, metrics) in enumerate(zip(axes, metrics_groups)):
#     pos = np.arange(len(metrics))
#
#     for ax, px_thresh in zip(row_ax, px_threshs):
#         ax.set_xticks(pos, metrics)
#
#         for j, (m_v, mm_v) in enumerate(zip(methods.values(), methods_metrics.values())):
#             barj = ax.bar(pos + j * 0.25, mm_v[i][f'metrics_{px_thresh}'],
#                           color=m_v['color'], width=0.25, alpha=0.6, label=m_v['alias'])
#
#             autolabel(ax, barj)
#
#         ax.set_title(f"Metrics {px_thresh} px; {category[1]}")
#
#         ax.set_xticks(pos + len(metrics) * 0.25 / len(methods.values()))
#         ax.set_xticklabels(metrics.values())
#
#         ax.legend()
#
# plt.tight_layout()
# plt.show()
