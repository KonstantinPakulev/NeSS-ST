import torch


def mse_loss(kp_nexs, kp_xs, kp_mask):
    num_kp = kp_mask.sum(dim=-1).float()
    batch_mask = (num_kp != 0).float()

    loss = 0.5 * (kp_xs - kp_nexs) ** 2
    loss = (loss * kp_mask.float()).sum(dim=-1) / num_kp.clamp(min=1e-8)
    loss = (loss * batch_mask).sum(dim=-1) / batch_mask.sum(dim=-1).clamp(min=1e-8)

    return loss

"""
Legacy code
"""

# from torch.nn.functional import binary_cross_entropy_with_logits
#
# def bce_loss(kp_log_conf_score, w_kp_rep_mask, w_kp_mask, pos_weight):
#     pos_mask = w_kp_rep_mask & w_kp_mask
#
#     num_kp = w_kp_mask.float().sum(dim=-1)
#
#     batch_mask = (num_kp != 0).float()
#
#     loss = binary_cross_entropy_with_logits(kp_log_conf_score, pos_mask.float(), reduction='none', pos_weight=pos_weight)
#     loss = (loss * w_kp_mask.float()).sum(dim=-1) / num_kp.clamp(min=1e-8)
#     loss = (loss * batch_mask).sum(dim=-1) / batch_mask.sum(dim=-1).clamp(min=1e-8)
#
#     return loss


# def mse_hist_loss(kp_u_est, ha_kp_u_est, ha_kp_weight, w_kp_mask, min_u_est, max_u_est):
#     w_kp_mask &= (ha_kp_weight != 0)
#
#     num_kp = w_kp_mask.sum(dim=-1).float()
#     batch_mask = (num_kp != 0).float()
#
#     loss = 0.5 * ha_kp_weight * (ha_kp_u_est - kp_u_est.clamp(min=min_u_est, max=max_u_est)) ** 2
#     loss = (loss * w_kp_mask.float()).sum(dim=-1) / num_kp.clamp(min=1e-8)
#     loss = (loss * batch_mask).sum(dim=-1) / batch_mask.sum(dim=-1).clamp(min=1e-8)
#
#     return loss

# import torch
#
#
# def mse_loss(kp_var, gt_ha_kp_var, kp_score, num_samples):
#     gt_ha_kp_var = gt_ha_kp_var.clamp(min=1 / num_samples)
#
#     kp_score_mask = (kp_score > 0).float()
#     num_kp = kp_score_mask.sum(dim=-1)
#
#     batch_mask = (num_kp != 0).float()
#
#     loss = 0.5 * (gt_ha_kp_var - kp_var) ** 2
#     loss = (loss * kp_score_mask.float()).sum(dim=-1) / num_kp.clamp(min=1e-8)
#     loss = (loss * batch_mask).sum(dim=-1) / batch_mask.sum(dim=-1).clamp(min=1e-8)
#
#     return loss.mean()

# import torch
#
#
# def mse_loss(kp_var, gt_ha_kp_var, kp_value, gt_ha_kp_mask,
#              lambda_thresh, num_samples):
#     var_thresh = 1 / num_samples
#     gt_ha_kp_var = gt_ha_kp_var.clamp(min=var_thresh)
#
#     gt_ha_kp_mask = ((kp_value > lambda_thresh) & gt_ha_kp_mask).float()
#     gt_ha_kp_mask_sum = gt_ha_kp_mask.sum(dim=-1)
#
#     batch_mask = (gt_ha_kp_mask_sum != 0).float()
#
#     loss = 0.5 * (gt_ha_kp_var - kp_var) ** 2
#     loss = (loss * gt_ha_kp_mask.float()).sum(dim=-1) / gt_ha_kp_mask_sum.clamp(min=1e-8)
#     loss = (loss * batch_mask).sum(dim=-1) / batch_mask.sum(dim=-1).clamp(min=1e-8)
#
#     return loss.mean()

# def mse_log_loss(ha_kp_log_var, ha_kp_cov_eigv, ha_kp_value,
#                  lambda_thresh, num_samples, use_bias):
#     if use_bias:
#         ha_kp_log_eigv = (ha_kp_cov_eigv.clamp(min=1 / num_samples) / 2).log()
#
#     else:
#         ha_kp_log_eigv = ha_kp_cov_eigv.clamp(min=1 / num_samples).log()
#
#     ha_kp_value_mask = (ha_kp_value > lambda_thresh).float()
#     ha_kp_value_mask_sum = ha_kp_value_mask.sum(dim=-1)
#
#     batch_mask = (ha_kp_value_mask_sum != 0).float()
#
#     loss = 0.5 * (ha_kp_log_var - ha_kp_log_eigv) ** 2
#     loss = (loss * ha_kp_value_mask.float()).sum(dim=-1) / ha_kp_value_mask_sum.clamp(min=1e-8)
#     loss = (loss * batch_mask).sum(dim=-1) / batch_mask.sum(dim=-1).clamp(min=1e-8)
#
#     return loss.mean()
#
#
# def mse_symmetric_loss(ha_kp_log_var, ha_kp_cov_eigv, ha_kp_value,
#                        lambda_thresh, num_samples, cov_mean):
#     cov_thresh = 1 / num_samples
#
#     ha_kp_cov_eigv = ha_kp_cov_eigv.clamp(min=cov_thresh, max=cov_mean * 2 - cov_thresh) / cov_mean
#     ha_kp_log_cov_eigv = ha_kp_cov_eigv.log() - (cov_mean - ha_kp_cov_eigv).log()
#
#     ha_kp_value_mask = (ha_kp_value > lambda_thresh).float()
#     ha_kp_value_mask_sum = ha_kp_value_mask.sum(dim=-1)
#
#     batch_mask = (ha_kp_value_mask_sum != 0).float()
#
#     loss = 0.5 * (ha_kp_log_cov_eigv - ha_kp_log_var) ** 2
#     loss = (loss * ha_kp_value_mask.float()).sum(dim=-1) / ha_kp_value_mask_sum.clamp(min=1e-8)
#     loss = (loss * batch_mask).sum(dim=-1) / batch_mask.sum(dim=-1).clamp(min=1e-8)
#
#     return loss.mean()
#
#
# def kld_loss(ha_kp_log_var, ha_kp_cov_eigv, ha_kp_value,
#              lambda_thresh, num_samples):
#     ha_kp_log_eigv = ha_kp_cov_eigv.clamp(min=1 / num_samples).log()
#
#     ha_kp_value_mask = (ha_kp_value > lambda_thresh).float()
#     ha_kp_value_mask_sum = ha_kp_value_mask.sum(dim=-1)
#
#     batch_mask = (ha_kp_value_mask_sum != 0).float()
#
#     loss = 0.5 * (ha_kp_log_eigv - ha_kp_log_var - 1 + torch.exp(ha_kp_log_var - ha_kp_log_eigv))
#     loss = (loss * ha_kp_value_mask.float()).sum(dim=-1) / ha_kp_value_mask_sum.clamp(min=1e-8)
#     loss = (loss * batch_mask).sum(dim=-1) / batch_mask.sum(dim=-1).clamp(min=1e-8)
#
#     return loss.mean()

# from math import sqrt

# from source.projective.rbt import warp_image_rbt
# from source.utils.endpoint_utils import flat2grid, sample_tensor_patch
# from source.utils.common_utils import sample_tensor

# from source.models.shi.model import get_shi_score

# from source.models.devl.utils.criterion_utils import get_kp_scale_mask

# def err_loss(kp_err1, gt_kp_err1, gt_kp_mask1):
#     num_kp = gt_kp_mask1.float().sum(dim=-1)
#     batch_mask = num_kp != 0
#
#     e_loss = ((kp_err1 - gt_kp_err1) ** 2 * gt_kp_mask1.float()).sum(dim=-1) / num_kp.clamp(min=1e-8)
#     e_loss = (e_loss * batch_mask.float()) / batch_mask.float().sum(dim=-1).clamp(min=1e-8)
#
#     return e_loss

# def get_err_loss_sws_gt(kp1, image_gray2,
#                         scene_data,
#                         sobel_size, window_size, window_cov,
#                         nms_size, max_sq_scale):
#     b, n = kp1.shape[:2]
#
#     ps_border = (max(sobel_size, window_size) // 2 + sobel_size // 2)
#     patch_size = nms_size + ps_border
#
#     l_p_b = ps_border // 2
#     r_p_b = patch_size - l_p_b
#
#     w_image_gray2, vis_mask1 = warp_image_rbt(image_gray2, scene_data, 'im')
#     shi_w_score2 = get_shi_score(w_image_gray2, sobel_size, window_size, window_cov)
#
#     kp_sws_p1, kp_pg1, _ = sample_tensor_patch(shi_w_score2, kp1, patch_size, shi_w_score2.shape, True)
#
#     kp_sws_p1 = kp_sws_p1.squeeze(-1).view(b, n, patch_size, patch_size)[:, :, l_p_b:r_p_b, l_p_b:r_p_b]
#     kp_p_vis_mask1 = sample_tensor_patch(vis_mask1.float(), kp_pg1, patch_size, vis_mask1.shape). \
#         squeeze(-1).prod(-1).bool()
#
#     kp_pg1 = kp_pg1.view(b, n, patch_size, patch_size, 2)[:, :, l_p_b:r_p_b, l_p_b:r_p_b].reshape(b, n, -1, 2)
#     kp_scale_mask1 = get_kp_scale_mask(kp_pg1, scene_data, nms_size, max_sq_scale)
#
#     gt_kp_idx1 = kp_sws_p1.reshape(b, n, -1).argmax(dim=-1)
#     gt_kp_err1 = (flat2grid(gt_kp_idx1, nms_size) - nms_size // 2).float().norm(dim=-1)
#     gt_kp_mask1 = kp_p_vis_mask1 & kp_scale_mask1
#
#     return gt_kp_err1, gt_kp_mask1
#
#
# def get_kp_err(kp, err):
#     return sample_tensor(err, kp, err.shape).squeeze(-1)

# def get_err_loss_wss_gt(kp1, shi_score2,
#                         scene_data,
#                         nms_size):
#     b, n = kp1.shape[:2]
#
#     w_shi_score2, vis_mask1 = warp_score_rbt(shi_score2, shi_score2.shape, scene_data, mode='im')
#
#     kp_wss_p1, kp_pg1, _ = sample_tensor_patch(w_shi_score2, kp1, nms_size, w_shi_score2.shape, True)
#
#     kp_p_vis_mask1 = sample_tensor_patch(vis_mask1.float(), kp_pg1, nms_size, vis_mask1.shape). \
#         squeeze(-1).prod(-1).bool()
#
#     kp_cov1 = torch.eye(2, device=kp1.device).view(1, 1, 2, 2).repeat(b, n, 1, 1)
#     w_kp_cov1, w_kp_mask1 = warp_gaussian_rbt(kp1, kp_cov1, scene_data)[1:]
#
#     w_kp_cov_eig_v1 = get_eigen_values(w_kp_cov1)
#     w_kp_scale_mask1 = (w_kp_cov_eig_v1 <= 2).prod(dim=-1) & (w_kp_cov_eig_v1 >= 0.5).prod(dim=-1)
#
#     gt_kp_diff = (flat2grid(kp_wss_p1.reshape(b, n, -1).argmax(dim=-1), nms_size) - nms_size // 2).float()
#     gt_kp_err1 = gt_kp_diff.norm(dim=-1)
#     gt_kp_mask1 = (kp_p_vis_mask1 & w_kp_mask1 & w_kp_scale_mask1).float()
#
#     return gt_kp_err1, gt_kp_mask1

# def err_loss2(kp_err1, gt_kp_err1, gt_kp_weight1, gt_kp_mask1):
#     total_weight = (gt_kp_weight1 * gt_kp_mask1.float()).sum(dim=-1)
#     batch_mask = total_weight != 0
#
#     e_loss = 0.5 * ((kp_err1 - gt_kp_err1) ** 2 * gt_kp_weight1 * gt_kp_mask1.float()).sum(dim=-1) / total_weight.clamp(min=1e-8)
#     e_loss = (e_loss * batch_mask.float()).sum(dim=-1) / batch_mask.float().sum(dim=-1).clamp(min=1e-8)
#
#     return e_loss

# gt_kp_weight1 = (1 / w_kp_cov1.det()).clamp(min=1e-8).sqrt()

# def get_kp_w_cov(kp, scene_data):
#     b, n = kp.shape[:2]
#
#     angles = torch.linspace(0, 315 / 180 * math.pi, steps=8)
#     x, y = torch.cos(angles), torch.sin(angles)
#
#     kp = kp.unsqueeze(-2)
#
#     kp_fan = torch.cat([kp,
#                          kp + torch.stack([y, x], dim=-1).unsqueeze(0).unsqueeze(0).to(kp.device)],
#                         dim=-2).view(b, -1, 2)
#
#     w_kp_fan, w_kp_fan_mask = warp_points_rbt(kp_fan, scene_data)
#
#     w_kp_fan = w_kp_fan.view(b, n, -1, 2)
#     w_kp_diff = (w_kp_fan[:, :, 1:, :] - w_kp_fan[:, :, :1, :]).unsqueeze(-1)
#     kp_w_cov = (w_kp_diff @ w_kp_diff.permute(0, 1, 2, 4, 3)).sum(dim=-3) / 8
#
#     w_kp_mask = w_kp_fan_mask.view(b, n, -1).sum(dim=-1) == 9
#
#     return kp_fan, w_kp_fan

    # return kp_w_cov, w_kp_mask

# label = grid2flat(w_kp2 - kp1 + nms_size // 2, nms_size).long()

# cat = Categorical(logits=kp_pg_cat_logits1)
# cat_flat_s1 = cat.sample()
# cat_kp1 = flat2grid(cat_flat_s1, nms_size) - nms_size // 2 + kp1

# log_pxz = torch.log(clamp_probs(((cat_kp1 == w_kp2).sum(dim=-1) == 2).float()))
# log_pz = torch.log(torch.tensor(1 / nms_size ** 2, device=kp1.device))
# log_qz = cat.log_prob(cat_flat_s1)

# cost = (log_pxz + log_pz - log_qz).detach()

# def err_delta_loss(kp_err_score1, loc_kp1, loc_w_kp2, delta_mask1, kp_mask1, err_lambda):
#     delta_num_kp = delta_mask1.float().sum(dim=-1)
#     delta_batch_mask = delta_num_kp != 0
#
#     l2 = (loc_kp1 - loc_w_kp2).norm(dim=-1)
#
#     d_loss = (l2 * delta_mask1.float()).sum(dim=-1) / delta_num_kp.clamp(min=1e-8)
#     d_loss = (d_loss * delta_batch_mask.float()) / delta_batch_mask.float().sum(dim=-1).clamp(min=1e-8)
#
#     err_num_kp = kp_mask1.float().sum(dim=-1)
#     err_batch_mask = err_num_kp != 0
#
#     e_loss = ((kp_err_score1 - l2.detach()) ** 2 * kp_mask1.float()).sum(dim=-1) / err_num_kp.clamp(min=1e-8)
#     e_loss = err_lambda * (e_loss * err_batch_mask.float()) / err_batch_mask.float().sum(dim=-1).clamp(min=1e-8)
#
#     loss = d_loss + e_loss
#
#     return loss, d_loss, e_loss
