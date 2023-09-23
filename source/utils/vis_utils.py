import cv2
import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from matplotlib import transforms

import torch

from source.utils.endpoint_utils import grid2flat


"""
Matplotlib plotting functions
"""


def plot_figures(figures, nrows=1, ncols=1, size=(18, 18), return_axes=False):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=size)
    for ind, title in zip(range(len(figures)), figures):
        if nrows * ncols != 1:
            axes.ravel()[ind].imshow(figures[title], cmap='gray')
            axes.ravel()[ind].set_title(title)
            axes.ravel()[ind].set_axis_off()
        else:
            axes.imshow(figures[title], cmap='gray')
            axes.set_title(title)
            axes.set_axis_off()

    plt.tight_layout()  # optional

    if return_axes:
        return fig, axes


def draw_cv_covariance(ax, mean, cov, ec='purple', label=None):
    pearson = cov[0, 1] / torch.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, fc='none', ec=ec, ls='--')

    scale_x = torch.sqrt(cov[0, 0])
    scale_y = torch.sqrt(cov[1, 1])

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)

    ax.add_patch(ellipse)

    if label is not None:
        ellipse.set(clip_box=ax.bbox, edgecolor=ec, label=label)

    ax.legend()


def draw_cv_mean_and_covariance(mean, cov, pts,
                                mean_label='mean',
                                cov_label='cov',
                                pts_label='points',
                                return_axis=False):
    fig, ax = plt.subplots(figsize=(6, 6))

    plt.scatter(mean[0], mean[1], marker='o', color='purple', label=mean_label)
    draw_cv_covariance(ax, mean, cov, ec='purple', label=cov_label)
    plt.scatter(pts[:, 0], pts[:, 1], marker='x', color='red', label=pts_label)

    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)

    major_ticks = np.arange(-2.5, 3.5, 1)

    plt.xticks(major_ticks)
    plt.yticks(major_ticks)

    plt.grid(which='major')
    plt.legend()

    ax.set_aspect('equal', 'box')

    if return_axis:
        return ax


def draw_crosshair(axis, position, color='g', markersize=10, markeredgewidth=1.5):
    """
    :param axis: Axis object
    :param position: (y, x)
    :param color: color
    """
    if len(position) == 1:
        axis.plot(position[0], position[0], color=color, marker='x',
                  markersize=markersize, markeredgewidth=markeredgewidth)

    elif len(position) == 2:
        axis.plot(position[1], position[0], color=color, marker='x',
                  markersize=markersize, markeredgewidth=markeredgewidth)

    else:
        raise NotImplementedError()


"""
Torch score helper functions
"""


def scatter_score(shape, kp, kp_mask=None, kp_weight=None):
    """
    :param shape: (b, c, h, w)
    :param kp: B x N x 2
    :param kp_mask: B x N
    :param kp_weight: B x N
    """
    b, _, h, w = shape

    if kp_mask is None:
        kp_mask = torch.ones(b, kp.shape[1], dtype=torch.bool).to(kp.device)

    if kp_weight is None:
        kp_weight = 1.0

    score = torch.zeros((b, 1, h, w)).to(kp.device)
    score = score.view(b, -1).\
        scatter(-1,
                grid2flat(kp.long(), w) * kp_mask.long(),
                kp_weight * kp_mask.float()).\
        view(b, 1, h, w)

    return score


def scatter_score_flat(shape, flat_kp, kp_mask, kp_weight):
    b, _, h, w = shape

    score = torch.zeros(shape).to(flat_kp.device)
    score = score.view(b, -1).scatter(-1,
                                      flat_kp * kp_mask.long(),
                                      kp_weight * kp_mask.float()).view(b, 1, h, w)

    return score


"""
Open3D visualization
"""


def to_open3d_pcd(world_kp1, kp_color1=None):
    if torch.is_tensor(world_kp1):
        world_kp1 = world_kp1.numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_kp1)

    if kp_color1 is not None:
        if torch.is_tensor(kp_color1):
            kp_color1 = kp_color1.numpy()

        pcd.colors_list = o3d.utility.Vector3dVector(kp_color1)

    return pcd


"""
Support functions
"""


def autolabel(ax, rects, label=None, font_size=None, precision=3):
    for rect in rects:
        height = rect.get_height()

        ax.annotate(('{:.' + str(precision) + 'f}').format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=font_size)

        if label is not None:
            ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, -12),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=font_size)


def torch2cv(img, normalize=False, to_rgb=False):
    """
    :param img: C x H x W
    :param normalize: normalize image
    :param to_rgb: convert image to rgb from grayscale
    """
    if normalize:
        img = (img - img.min()) / (img.max() - img.min())

    img = img.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)

    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 1:
        img = img[:, :, 0]

    return img


def cv2torch(img):
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    return img


def mix_colors(color1, color2, mix=0):
    color1 = np.array(mpl.colors.to_rgb(color1))
    color2 = np.array(mpl.colors.to_rgb(color2))

    return mpl.colors.to_hex((1 - mix) * color1 + mix * color2)


"""
Legacy code
"""


# def draw_confidence_ellipse(ax,
#                             mean, cov,
#                             facecolor='none', edgecolor='red', **kwargs):
#     pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
#     ell_radius_x = np.sqrt(1 + pearson)
#     ell_radius_y = np.sqrt(1 - pearson)
#
#     ellipse = Ellipse((0, 0),
#                       width=ell_radius_x * 2, height=ell_radius_y * 2,
#                       facecolor=facecolor, edgecolor=edgecolor,
#                       **kwargs)
#
#     scale_x = np.sqrt(cov[0, 0])
#     scale_y = np.sqrt(cov[1, 1])
#
#     transf = transforms.Affine2D() \
#         .rotate_deg(45) \
#         .scale(scale_x, scale_y) \
#         .translate(mean[0], mean[1])
#
#     ellipse.set_transform(transf + ax.transData)
#
#     ax.add_patch(ellipse)
#     ax.scatter(mean[0], mean[1], c=edgecolor, transform=ax.transData, marker='x', clip_on=False)



# def draw_cv_covariances(cv_image, mean, cov,
#                         batch_idx=0,
#                         color=(255, 0, 0)):
#     """
#     :param cv_image: H x W x 3, :type np.array
#     :param mean: B x N x 2, :type torch.tensor
#     :param cov: B x N x 2 x 2, :type torch.tensor
#     :param batch_idx: int
#     :param color: tuple (r,g,b)
#     """
#     mean = mean[batch_idx]
#     cov = cov[batch_idx]
#
#     for m, c in zip(mean.cpu().numpy(), cov.cpu().numpy()):
#         cv_image = cv2.ellipse(np.ascontiguousarray(cv_image),
#                                m.astype(np.int32)[::-1], (2, 2), 0,
#                                0, 360,
#                                color)
#
#     return cv_image

# def draw_confidence_ellipse(mean, cov, ax, facecolor='none', **kwargs):
#     pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
#     ell_radius_x = np.sqrt(1 + pearson)
#     ell_radius_y = np.sqrt(1 - pearson)
#
#     ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
#
#     scale_x = np.sqrt(cov[0, 0])
#     scale_y = np.sqrt(cov[1, 1])
#
#     transf = transforms.Affine2D() \
#         .rotate_deg(45) \
#         .scale(scale_x, scale_y) \
#         .translate(mean[0], mean[1])
#
#     ellipse.set_transform(transf + ax.transData)
#     return ax.add_patch(ellipse)


# def draw_cv_grid(cv_image, grid_size, grid_color=(25, 25, 25)):
#     """
#     :param cv_image: H x W x C, :type numpy.uint8
#     :param grid_size: int
#     :param grid_color: (r, g, b) :type tuple
#     """
#     h, w = cv_image.shape[:2]
#
#     x, y = np.arange(0, w, step=grid_size), np.arange(0, h, step=grid_size)
#
#     for i in x:
#         cv_image = cv2.line(cv_image, (i, 0), (i, h - 1), grid_color, thickness=1)
#
#     for i in y:
#         cv_image = cv2.line(cv_image, (0, i), (w - 1, i), grid_color, thickness=1)
#
#     return cv_image


# def draw_neigh_mask(image2, w_desc_grid1, neigh_mask_ids, desc_shape, grid_size, w_desc_id):
#     w_desc_point = w_desc_grid1[None, w_desc_id].cpu().numpy()
#     w_desc_neigh_id = neigh_mask_ids[w_desc_id]
#
#     wc = desc_shape[-1]
#     w_desc_neigh_points = flat2grid(w_desc_neigh_id, wc).cpu().numpy() * grid_size
#
#     cv_image = draw_cv_keypoints(image2, w_desc_point, (255, 0, 0))
#     cv_image = draw_cv_keypoints(image2, w_desc_neigh_points, (0, 255, 0))
#
#     return cv_image
# def plot_reproj_error_hist(nn_kp_values, matches_mask, batch_id=0):
#     """
#     :param nn_kp_values: B x N
#     :param matches_mask: B x N
#     :param batch_id: int
#     """
#     reproj_errors = nn_kp_values[batch_id][matches_mask[batch_id]].cpu().numpy()
#
#     plt.hist(reproj_errors)
#     plt.xlabel('Re-proj. error', fontsize=14.0)
#     plt.ylabel('Num. matches', fontsize=14.0)


# def plot_mean_matching_accuracy(eval_results):
#     fig, axes = plt.subplots(1, len(eval_results), figsize=(7 * len(eval_results), 6))
#
#     if not isinstance(axes, np.ndarray):
#         axes = [axes]
#
#     axes[0].set_ylabel("MMA", fontsize=21.0)
#
#     for i, (ax, (dataset_name, eval_summary)) in enumerate(zip(axes, eval_results)):
#         xticks = np.arange(1, 11)
#
#         for key, value in eval_summary.items():
#             ax.plot(xticks, value, linewidth=3, label=key)
#
#         ax.set_title(dataset_name, fontsize=23.0)
#         ax.set_xlabel('Threshold [px]', fontsize=21.0)
#
#         ax.set_xlim([1, 10])
#         ax.set_ylim([0, 1])
#         ax.set_xticks(xticks)
#         ax.tick_params(axis='both', which='major', labelsize=20)
#
#         ax.grid()
#         ax.legend()
#
#
# def plot_precision_recall_curve(precision, recall):
#     """
#     :param precision: N
#     :param recall: N
#     """
#     _, ax = plt.subplots(1, 1, figsize=(5, 5))
#
#     ax.plot(precision, recall)
#
#     ax.set_xlabel('precision', fontsize=25.0)
#     ax.set_ylabel('recall', fontsize=25.0)
#
#     ax.grid()
#     ax.tick_params(axis='both', which='major', labelsize=20)

# def plot_state_obs_conf_interval(image,
#                                  state_hypo_mean, state_hypo_cov,
#                                  obs_mean, obs_cov,
#                                  obs_factor_adj,
#                                  state_hypo_mask=None,
#                                  batch_idx=0, state_only=False, as_kp=False,
#                                  size=(18, 18), state_hypo_edge_color='mediumseagreen'):
#     cv_image = torch2cv(image[batch_idx])
#
#     fig, ax = plt.subplots(ncols=1, nrows=1, figsize=size)
#
#     if not state_only:
#         for obs_fpair_idxi in obs_factor_adj[batch_idx][1]:
#             line_starti = obs_mean[batch_idx][obs_fpair_idxi[0]].numpy()
#             line_endi = state_hypo_mean[batch_idx][obs_fpair_idxi[1]].numpy()
#
#             linei = np.stack([line_starti, line_endi], axis=0)
#
#             mpl_line = Line2D(linei[:, 0], linei[:, 1], color='khaki', alpha=0.25)
#
#             ax.add_line(mpl_line)
#
#         for obs_meani, obs_covi in zip(obs_mean[batch_idx].numpy(), obs_cov[batch_idx].numpy()):
#             draw_confidence_ellipse(obs_meani, obs_covi, ax, edgecolor='royalblue')
#
#     for i, (state_hypo_meani, state_hypo_covi) in enumerate(zip(state_hypo_mean[batch_idx].numpy(),
#                                                                 state_hypo_cov[batch_idx].numpy())):
#         if state_hypo_mask is not None and not state_hypo_mask[batch_idx][i]:
#             break
#
#         if as_kp:
#             draw_confidence_ellipse(state_hypo_meani[[1, 0]], state_hypo_covi[[1, 0], :][:, [1, 0]],
#                                     ax, edgecolor=state_hypo_edge_color)
#
#         else:
#             draw_confidence_ellipse(state_hypo_meani, state_hypo_covi, ax, edgecolor=state_hypo_edge_color)
#
#     ax.imshow(cv_image)
