import numpy as np
import torch

from source.pose.matchers.utils import gather_kp
from source.projective.homography import warp_points_h

"""
Metrics
"""

def repeatability_score(kp1, kp2,
                        h_data,
                        px_thresh):
    w_kp1, w_kp_mask1 = warp_points_h(kp1, h_data, mode='pm')

    dist = torch.cdist(w_kp1, kp2)
    nn_kp2 = gather_kp(kp2, dist.argmin(dim=-1))

    w_nn_kp_dist1 = (nn_kp2 - w_kp1).norm(dim=-1)

    num_thresh = len(px_thresh)
    b = kp1.shape[0]

    rep = np.zeros((b, num_thresh))

    for i, thr in enumerate(px_thresh):
        match_maski = w_kp_mask1 & w_nn_kp_dist1.le(thr)

        rep[:, i] = (match_maski.sum(dim=-1) / w_kp_mask1.sum(dim=-1).clamp(min=1e-8)).cpu().numpy()

    return rep


def mean_matching_accuracy(kp1, kp2,
                           kp_desc1, kp_desc2,
                           kp_desc_mask1, kp_desc_mask2,
                           matcher, h_data,
                           px_thresh):
    mm_desc_mask1, nn_desc_idx1 = matcher.match(kp_desc1, kp_desc2,
                                                kp_desc_mask1, kp_desc_mask2)

    w_kp1, w_kp_mask1 = warp_points_h(kp1, h_data, mode='pm')
    nn_kp2 = gather_kp(kp2, nn_desc_idx1.cpu())

    mm_w_kp_mask1 = mm_desc_mask1.cpu() & w_kp_mask1

    w_nn_kp_dist1 = (nn_kp2 - w_kp1).norm(dim=-1)

    num_thresh = len(px_thresh)
    b = kp1.shape[0]

    mma = np.zeros((b, num_thresh))

    for i, thr in enumerate(px_thresh):
        match_maski = mm_w_kp_mask1 & w_nn_kp_dist1.le(thr)

        mma[:, i] = (match_maski.sum(dim=-1) / mm_w_kp_mask1.sum(dim=-1).clamp(min=1e-8)).cpu().numpy()

    return mma


"""
Legacy code
"""


# def match_score(kp1, kp2,
#                 w_kp1, w_kp2,
#                 w_kp_mask1, w_kp_mask2,
#                 kp_desc1, kp_desc2,
#                 px_thresh,
#                 matcher):
#     """
#     :param kp1: B x N x 2; keypoints on the first image
#     :param kp2: B x N x 2; keypoints on the second image
#     :param w_kp1: B x N x 2; keypoints on the first image projected to the second
#     :param w_kp2: B x N x 2; keypoints on the second image projected to the first
#     :param w_kp_mask1: B x N; keypoints on the first image which are visible on the second
#     :param w_kp_mask2: B x N; keypoints on the second image which are visible on the first
#     :param kp_desc1: B x N x C; descriptors for keypoints on the first image
#     :param kp_desc2: B x N x C; descriptors for keypoints on the second image
#     :param px_thresh: P; list
#     :param matcher: Matcher object
#     """
#     mm_desc_mask1, nn_desc_idx1 = matcher.match(kp_desc1, kp_desc2)
#
#     valid_match_mask1, kp_dist1 = get_valid_match_mask(nn_desc_idx1,
#                                                        kp1, kp2,
#                                                        w_kp1, w_kp2,
#                                                        w_kp_mask1, w_kp_mask2,
#                                                        px_thresh, True)
#
#     num_vis_match = torch.cat([w_kp_mask1.sum(dim=-1).unsqueeze(-1),
#                                w_kp_mask2.sum(dim=-1).unsqueeze(-1)], dim=-1).min(dim=-1)[0].float().clamp(min=1e-8)
#
#     num_thresh = len(px_thresh)
#     b, n = kp1.shape[:2]
#
#     m_score = torch.zeros(num_thresh, b)
#     match_mask = torch.zeros(num_thresh, b, n)
#
#     for i, thr in enumerate(px_thresh):
#         if i != num_thresh - 1:
#             mm_match_maski = mm_desc_mask1 * kp_dist1.le(thr)
#         else:
#             mm_match_maski = mm_desc_mask1 & valid_match_mask1
#
#         m_score[i] = mm_match_maski.sum(dim=-1) / num_vis_match
#         match_mask[i] = mm_match_maski
#
#     return m_score, num_vis_match, match_mask


# def repeatability_score(kp1, kp2, w_kp1, w_kp2, w_kp_mask1, w_kp_mask2, px_thresh):
#     """
#     :param kp1: B x N x 2; keypoints on the first image
#     :param kp2: B x N x 2; keypoints on the second image
#     :param w_kp1: B x N x 2; keypoints on the first image projected to the second
#     :param w_vis_kp1_mask: B x N; keypoints on the first image which are visible on the second
#     :param w_vis_kp2_mask: B x N; keypoints on the second image which are visible on the first
#     :param px_thresh: P; torch.tensor
#     :return P or P x B, P x B, B x N, B x N, P x B x N
#     """
#     # Use the largest px threshold to determine matches
#     gt_matches_mask, nn_kp_values, nn_kp_ids = get_gt_matches(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask,
#                                                               px_thresh[-1], return_reproj=True)
#
#     # Select minimum number of visible points for each scene
#     num_vis_kp = get_num_vis_gt_matches(w_vis_kp1_mask, w_vis_kp2_mask)
#
#     num_thresh = len(px_thresh)
#     b, n = kp1.shape[:2]
#
#     rep_scores = torch.zeros(num_thresh, b)
#     num_matches = torch.zeros(num_thresh, b)
#     match_mask = torch.zeros(num_thresh, b, n)
#
#     # Filter matches by lower thresholds
#     for i, thresh in enumerate(px_thresh):
#         if i != num_thresh - 1:
#             i_gt_matches_mask = nn_kp_values.le(thresh) * gt_matches_mask
#         else:
#             i_gt_matches_mask = gt_matches_mask
#
#         i_num_matches = i_gt_matches_mask.sum(dim=-1).float()
#
#         rep_scores[i] = i_num_matches / num_vis_kp
#         num_matches[i] = i_num_matches
#         match_mask[i] = i_gt_matches_mask
#
#     return rep_scores, num_matches, num_vis_kp, nn_kp_ids, match_mask
#
#
# def desc_sim_ratio(kp1_desc, kp2_desc, kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask, px_thresh, sim_measure):
#     desc_dist = calculate_descriptor_distance(kp1_desc, kp2_desc, sim_measure)
#
#     nn_desc_value1, nn_desc_ids1 = desc_dist.topk(dim=-1, k=2, largest=False)
#     nn_desc_value2, nn_desc_ids2 = desc_dist.topk(dim=-2, k=2, largest=False)
#
#     mutual_desc_matches_mask = get_cyclic_match_mask(nn_desc_ids1[..., 0], nn_desc_ids2[:, 0, :])
#
#     v_mutual_desc_matches_mask = verify_mutual_desc_matches(nn_desc_ids1[..., 0], kp1, kp2, w_kp1, w_kp2,
#                                                             w_vis_kp1_mask, w_vis_kp2_mask, px_thresh)
#
#     mutual_desc_matches_mask = mutual_desc_matches_mask * v_mutual_desc_matches_mask
#
#     lowe_ratio1 = nn_desc_value1[..., 0] / nn_desc_value1[..., 1]
#     lowe_ratio2 = nn_desc_value2[:, 0, :] / nn_desc_value2[:, 1, :]
#
#     nn_lowe_ratio2 = torch.gather(lowe_ratio2, -1, nn_desc_ids1[..., 0])
#
#     lowe_ratio = torch.cat([lowe_ratio1.unsqueeze(-1), nn_lowe_ratio2.unsqueeze(-1)], dim=-1).mean(dim=-1)
#
#     correct_lowe_ratio = (lowe_ratio * mutual_desc_matches_mask.float()).sum(dim=-1) / mutual_desc_matches_mask.float().sum(dim=-1).clamp(min=1e-8)
#     incorrect_lowe_ratio = (lowe_ratio * (~mutual_desc_matches_mask).float()).sum(dim=-1) / (~mutual_desc_matches_mask).float().sum(dim=-1).clamp(min=1e-8)
#
#     return correct_lowe_ratio, incorrect_lowe_ratio




# def pose_mAA(pose_accuracy):
#     pass
#
# def pose_mAA(pose_err, pose_thresh, max_angle=180):
#     angles = np.linspace(1, max_angle, num=max_angle)
#     precision = [np.sum(pose_err < a) / len(pose_err) for a in angles]
#
#     mAP = {thresh: np.mean(precision[:thresh]) for thresh in pose_thresh}
#
#     return mAP, precision


# Legacy code

# def relative_param_pose_error(kp1, kp2, kp1_desc, kp2_desc, shift_scale1, shift_scale2,
#                               intrinsics1, intrinsics2, extrinsics1, extrinsics2, px_thresh, sim_measure, lowe_ratio):
#     mutual_desc_matches_mask, nn_desc_ids = get_mutual_desc_matches(kp1_desc, kp2_desc, sim_measure, lowe_ratio)
#
#     r_kp1 = revert_data_transform(kp1, shift_scale1)
#     r_kp2 = revert_data_transform(kp2, shift_scale2)
#
#     nn_r_kp2 = select_kp(r_kp2, nn_desc_ids)
#     nn_r_i1_kp2 = change_intrinsics(nn_r_kp2, intrinsics2, intrinsics1)
#
#     gt_E_param = compose_gt_transform(intrinsics1, intrinsics2, extrinsics1, extrinsics2, E_param)
#
#     num_thresh = len(px_thresh)
#     b = kp1.shape[0]
#
#     R_param_err = torch.zeros(num_thresh, b)
#     t_param_err = torch.zeros(num_thresh, b)
#
#     success_mask = torch.zeros(num_thresh, b, dtype=torch.bool)
#
#     for i, thresh in enumerate(px_thresh):
#         i_est_E_param, i_success_mask = prepare_param_rel_pose(r_kp1, nn_r_i1_kp2, mutual_desc_matches_mask,
#                                                                intrinsics1, intrinsics2, thresh)
#
#         R_param_err[i] = (i_est_E_param[:, :3] - gt_E_param[:, :3]).norm(dim=-1)
#         t_param_err[i] = (i_est_E_param[:, 3:] - gt_E_param[:, 3:]).norm(dim=-1)
#
#         success_mask[i] = i_success_mask
#
#     return R_param_err, t_param_err, success_mask

# , prepare_param_rel_pose

# def estimate_rel_poses_opencv(kp1, kp2, kp1_desc, kp2_desc,
#                               intrinsics1, intrinsics2, shift_scale1, shift_scale2,
#                               px_thresh, dd_measure,
#                               detailed=False):
#     i_intrinsics1 = intrinsics1.inverse()
#     i_intrinsics2 = intrinsics2.inverse()
#
#     m_desc_matches_mask, nn_desc_ids1 = get_mutual_desc_matches(kp1_desc, kp2_desc, dd_measure, 0.9)
#
#     o_kp1 = revert_data_transform(kp1, shift_scale1)
#     o_kp2 = revert_data_transform(kp2, shift_scale2)
#     nn_o_kp2 = select_kp(o_kp2, nn_desc_ids1)
#
#     est_rel_pose = torch.zeros(len(px_thresh), kp1.shape[0], 3, 4).to(kp1.device)
#
#     for i, p_th in enumerate(px_thresh):
#         for b in range(kp1.shape[0]):
#             b_gt_matches_mask = m_desc_matches_mask[b]
#
#             if b_gt_matches_mask.sum() < 8:
#                 continue
#
#             cv_kp1 = o_kp1[b][b_gt_matches_mask].cpu().numpy()
#             cv_nn_kp2 = nn_o_kp2[b][b_gt_matches_mask].cpu().numpy()
#
#             cv_intrinsics1 = intrinsics1[b].cpu().numpy()
#             cv_intrinsics2 = intrinsics2[b].cpu().numpy()
#
#             cv_i_intrinsics1 = i_intrinsics1[b].cpu().numpy()
#             cv_i_intrinsics2 = i_intrinsics2[b].cpu().numpy()
#
#             E_est_init = estimate_ess_mat_opencv(cv_kp1, cv_nn_kp2, cv_intrinsics1, cv_intrinsics2)
#             opt_res = least_squares(loss_fun, E_est_init, jac=loss_fun_jac,
#                                     args=(cv_kp1, cv_nn_kp2, cv_i_intrinsics1, cv_i_intrinsics2), method='lm')
#
#             if opt_res.success:
#                 R, _ = cv2.Rodrigues(opt_res.x[:3].reshape(-1))
#
#                 ab = np.append(opt_res.x[3:], 0)
#                 R_z, _ = cv2.Rodrigues(ab.reshape(-1))
#
#                 z_0 = np.array([0, 0, 1])
#                 t = R_z @ z_0
#
#                 est_rel_pose[i][b][:3,:3] = torch.tensor(R).to(kp1.device)
#                 est_rel_pose[i][b][:3, 3] = normalize(torch.tensor(-t).to(kp1.device), dim=-1)
#
#     if detailed:
#         return est_rel_pose, m_desc_matches_mask, nn_desc_ids1
#     else:
#         return est_rel_pose

#
# def estimate_rel_poses_sift_opengv(sift_kp1, sift_kp2, intrinsics1, intrinsics2, px_thresh):
#     est_rel_pose = torch.zeros(len(px_thresh), 3, 4)
#     est_inl_mask = torch.zeros(len(px_thresh), sift_kp1.shape[0], dtype=torch.bool)
#
#     for i, p_th in enumerate(px_thresh):
#         T, b_inliers = \
#             relative_pose_opengv(sift_kp1, sift_kp2, intrinsics1, intrinsics2, p_th)
#
#         est_rel_pose[i] = torch.tensor(T)
#         est_rel_pose[i][:3, 3] = normalize(est_rel_pose[i][:3, 3], dim=-1)
#
#         est_inl_mask[i] = torch.tensor(b_inliers)
#
#     return est_rel_pose, est_inl_mask


# def nn_mAP(w_kp1, kp2, wv_kp1_mask, wv_kp2_mask, pixel_thresholds, kp1_desc, kp2_desc, desc_dist_measure,
#            detailed=False):
#     """
#     :param w_kp1: B x N x 2; keypoints on the first image projected to the second
#     :param kp2: B x N x 2; keypoints on the second image
#     :param wv_kp1_mask: B x N; keypoints on the first image which are visible on the second
#     :param wv_kp2_mask: B x N; keypoints on the second image which are visible on the first
#     :param pixel_thresholds: P :type torch.float
#     :param kp1_desc: B x N x C; descriptors for keypoints on the first image
#     :param kp2_desc: B x N x C; descriptors for keypoints on the second image
#     :param desc_dist_measure: measure of descriptor distance. Can be L2-norm or similarity measure
#     :param detailed: return detailed information :type bool
#     """
#     b, n = wv_kp1_mask.shape
#
#     # Calculate pairwise distance/similarity measure
#     if desc_dist_measure is DescriptorDistance.INV_COS_SIM:
#         desc_sim = smooth_inv_cos_sim_mat(kp1_desc, kp2_desc)
#     else:
#         desc_sim = calculate_distance_matrix(kp1_desc, kp2_desc)
#
#     nn_desc_values, nn_desc_ids = desc_sim.min(dim=-1)
#
#     # Remove duplicate matches in each scene
#     unique_match_mask = calculate_unique_match_mask(nn_desc_values, nn_desc_ids)
#
#     # Calculate pairwise keypoints distances
#     kp_dist = calculate_distance_matrix(w_kp1, kp2)
#     kp_dist = mask_non_visible_pairs(kp_dist, wv_kp1_mask, wv_kp2_mask)
#
#     # Retrieve correspondent keypoints
#     nn_kp_values = torch.gather(kp_dist, -1, nn_desc_ids.view(b, n, 1)).view(b, n)
#
#     if detailed:
#         nn_mAP_scores = torch.zeros(pixel_thresholds.shape[0], w_kp1.shape[0])
#         precisions = torch.zeros(pixel_thresholds.shape[0], w_kp1.shape[0], w_kp1.shape[1])
#         recalls = torch.zeros(pixel_thresholds.shape[0], w_kp1.shape[0], w_kp1.shape[1])
#     else:
#         nn_mAP_scores = torch.zeros_like(pixel_thresholds)
#
#     for i, thresh in enumerate(pixel_thresholds):
#         # Threshold correspondences
#         t_match_mask = nn_kp_values.le(thresh) * unique_match_mask
#
#         # Calculate number of matches for each scene
#         t_matches = t_match_mask.sum(dim=-1).float()
#
#         # Calculate tp and fp
#         tp = torch.cumsum(t_match_mask == True, dim=-1).float()
#         fp = torch.cumsum((t_match_mask == False) * wv_kp1_mask, dim=-1).float()
#
#         precision = tp / (tp + fp).clamp(min=1e-8)
#         recall = tp / t_matches.view(-1, 1).clamp(min=1e-8)
#
#         if detailed:
#             nn_mAP_scores[i] = torch.trapz(precision, recall)
#             precisions[i] = precision.sort(dim=-1)[0]
#             recalls[i] = recall.sort(dim=-1)[0]
#         else:
#             nn_mAP_scores[i] = torch.trapz(precision, recall).mean()
#
#     if detailed:
#         return nn_mAP_scores, precisions, recalls
#     else:
#         return nn_mAP_scores
# def epipolar_match_score(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask,
#                          kp1_desc, kp2_desc, shift_scale1, shift_scale2,
#                          intrinsics1, intrinsics2, extrinsics1, extrinsics2,
#                          px_thresh, sim_measure, detailed=False):
#     mutual_desc_matches_mask, nn_desc_ids = get_mutual_desc_matches(kp1_desc, kp2_desc, sim_measure, None)
#
#     # Verify matches by using the largest pixel threshold
#     v_mutual_desc_matches_mask, nn_kp_values = verify_mutual_desc_matches(nn_desc_ids, kp1, kp2, w_kp1, w_kp2,
#                                                                           w_vis_kp1_mask, w_vis_kp2_mask, px_thresh[-1],
#                                                                           return_reproj=True)
#
#     # Select minimum number of visible points for each scene
#     num_vis_gt_matches = get_num_vis_gt_matches(w_vis_kp1_mask, w_vis_kp2_mask)
#
#     num_thresh = len(px_thresh)
#     b, n = kp1.shape[:2]
#
#     o_kp1 = revert_data_transform(kp1, shift_scale1)
#     o_kp2 = revert_data_transform(kp2, shift_scale2)
#
#     nn_o_kp2 = select_kp(o_kp2, nn_desc_ids)
#
#     F = compose_gt_transform(intrinsics1, intrinsics2, extrinsics1, extrinsics2)
#
#     ep_dist = epipolar_distance(o_kp1, nn_o_kp2, F)
#
#     if detailed:
#         em_scores = torch.zeros(num_thresh, b)
#         num_matches = torch.zeros(num_thresh, b)
#         match_mask = torch.zeros(num_thresh, b, n)
#     else:
#         em_scores = torch.zeros(num_thresh)
#
#     for i, thresh in enumerate(px_thresh):
#         if i != num_thresh - 1:
#             i_mutual_matches_mask = mutual_desc_matches_mask * nn_kp_values.le(thresh) * v_mutual_desc_matches_mask * \
#                                     ep_dist.le(thresh)
#         else:
#             i_mutual_matches_mask = mutual_desc_matches_mask * v_mutual_desc_matches_mask * ep_dist.le(thresh)
#
#         i_num_matches = i_mutual_matches_mask.sum(dim=-1).float()
#
#         if detailed:
#             em_scores[i] = i_num_matches / num_vis_gt_matches
#             num_matches[i] = i_num_matches
#             match_mask[i] = i_mutual_matches_mask
#         else:
#             em_scores[i] = (i_num_matches / num_vis_gt_matches).mean()
#
#     if detailed:
#         return em_scores, num_matches, num_vis_gt_matches, nn_desc_ids, match_mask
#     else:
#         return em_scores
