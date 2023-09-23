import torch


def gather_kp(kp, idx):
    return torch.gather(kp, 1, idx.unsqueeze(-1).repeat(1, 1, 2))


def get_valid_match_mask(nn_desc_idx,
                         kp1, kp2,
                         w_kp1, w_kp2,
                         w_kp_mask1, w_kp_mask2,
                         px_thresh,
                         return_dist=False):
    dist12 = torch.cdist(w_kp1, kp2)
    dist12 = mask_dist_dim(dist12, w_kp_mask1, w_kp_mask2)

    dist21 = torch.cdist(kp1, w_kp2)
    dist21 = mask_dist_dim(dist21, w_kp_mask1, w_kp_mask2)

    nn_kp_dist1 = torch.gather(dist12, -1, nn_desc_idx.unsqueeze(-1))
    nn_kp_dist2 = torch.gather(dist21, -1, nn_desc_idx.unsqueeze(-1))

    nn_kp_dist = torch.cat([nn_kp_dist1, nn_kp_dist2], dim=-1).min(dim=-1)[0]

    valid_match_mask = nn_kp_dist.le(px_thresh)

    if return_dist:
        return valid_match_mask, nn_kp_dist

    else:
        return valid_match_mask


def calculate_kp_distance(w_kp1, kp2, w_kp_mask1, w_kp_mask2, return_diff=False):
    if return_diff:
        kp_diff = w_kp1.unsqueeze(2).float() - kp2.unsqueeze(1).float()
        kp_dist = torch.norm(kp_diff, p=2, dim=-1)

        kp_dist = mask_dist_dim(kp_dist, w_kp_mask1, w_kp_mask2)

        return kp_dist, kp_diff

    else:
        kp_dist = torch.cdist(w_kp1, kp2)
        kp_dist = mask_dist_dim(kp_dist, w_kp_mask1, w_kp_mask2)

        return kp_dist


def mask_dist_dim(dist, dim_mask1=None, dim_mask2=None, mask_by_max=True):
    """
    :param dist: B x N1 x N2
    :param dim_mask1: B x N1
    :param dim_mask2: B x N2
    :param mask_by_max: bool
    """
    b, n1, n2 = dist.shape

    mask_dist = (dist.max() - dist.min()) * 2

    if not mask_by_max:
        mask_dist = -mask_dist

    if dim_mask1 is not None:
        dist += (1 - dim_mask1.float().view(b, n1, 1)) * mask_dist

    if dim_mask2 is not None:
        dist += (1 - dim_mask2.float().view(b, 1, n2)) * mask_dist

    return dist


"""
Legacy code
"""


# def get_gt_matches(kp1, kp2, w_kp1, w_kp2, vis_w_kp1_mask, vis_w_kp2_mask, px_thresh, return_reproj=False):
#     """
#     :param kp1: B x N x 2
#     :param kp2: B x N x 2
#     :param w_kp1: B x N x 2
#     :param w_kp2: B x N x 2
#     :param vis_w_kp1_mask: B x N
#     :param vis_w_kp2_mask: B x N
#     :param px_thresh: int
#     :param return_reproj: bool
#     """
#     mutual_gt_matches_mask1, nn_kp_values1, nn_kp_ids1 = \
#         get_mutual_gt_matches(w_kp1, kp2, vis_w_kp1_mask, vis_w_kp2_mask, px_thresh)
#
#     mutual_gt_matches_mask2, nn_kp_values2, nn_kp_ids2 = \
#         get_mutual_gt_matches(kp1, w_kp2, vis_w_kp1_mask, vis_w_kp2_mask, px_thresh)
#
#     mutual_gt_matches_mask = mutual_gt_matches_mask1 * mutual_gt_matches_mask2
#
#     nn_kp_values, nn_kp_values_ids = torch.cat([nn_kp_values1.unsqueeze(-1),
#                                                 nn_kp_values2.unsqueeze(-1)], dim=-1).min(dim=-1)
#
#     nn_kp_ids = torch.cat([nn_kp_ids1.unsqueeze(-1),
#                            nn_kp_ids2.unsqueeze(-1)], dim=-1)
#
#     nn_kp_ids = torch.gather(nn_kp_ids, dim=-1, index=nn_kp_values_ids.unsqueeze(-1)).squeeze(-1)
#
#     if return_reproj:
#         return mutual_gt_matches_mask, nn_kp_values, nn_kp_ids
#
#     else:
#         return mutual_gt_matches_mask, nn_kp_ids


# def get_mutual_gt_matches(w_kp1, kp2, vis_w_kp1_mask, vis_w_kp2_mask, px_thresh):
#     kp_dist1 = calculate_kp_distance(w_kp1, kp2, vis_w_kp1_mask, vis_w_kp2_mask)
#
#     nn_kp_values1, nn_kp_ids1 = kp_dist1.min(dim=-1)
#     _, nn_kp_ids2 = kp_dist1.min(dim=-2)
#
#     mutual_gt_matches_mask = get_cyclic_match_mask(nn_kp_ids1, nn_kp_ids2) * nn_kp_values1.le(px_thresh)
#
#     return mutual_gt_matches_mask, nn_kp_values1, nn_kp_ids1


# def get_cosine_dist(kp_desc1, kp_desc2):
#     """
#     :param kp_desc1: B x N1 x C, normalized
#     :param kp_desc2: B x N2 x C, normalized
#     """
#     b, n1, c = kp_desc1.shape
#     n2 = kp_desc2.shape[1]
#
#     l2_dist = torch.cdist(kp_desc1, kp_desc2).view(b, n1, n2)
#
#     cosine_dist = 1.0 - 0.5 * l2_dist ** 2
#
#     return cosine_dist

# def mask_dist(dist, mask, mask_by_max=True):
#     mask_dist = (dist.max() - dist.min()) * 2
#
#     if not mask_by_max:
#         mask_dist = -mask_dist
#
#     dist += mask.float() * mask_dist
#
#     return dist
#
# def get_num_vis_gt_matches(w_vis_kp1_mask, w_vis_kp2_mask):
#     v1 = w_vis_kp1_mask.sum(dim=-1).unsqueeze(-1)
#     v2 = w_vis_kp2_mask.sum(dim=-1).unsqueeze(-1)
#     num_vis_gt_matches = torch.cat([v1, v2], dim=-1).min(dim=-1)[0].float().clamp(min=1e-8)
#
#     return num_vis_gt_matches
#
#

#
#



# Legacy code

# def get_mutual_desc_matches_v2(kp1, kp2, desc1, desc2, kp1_desc, kp2_desc, grid_size, dd_measure, lowe_ratio=None):
#     desc_dist = calculate_descriptor_distance(kp1_desc, kp2_desc, dd_measure)
#
#     nn_desc_value1, nn_desc_ids1 = desc_dist.topk(dim=-1, k=2, largest=False)
#     nn_desc_value2, nn_desc_ids2 = desc_dist.topk(dim=-2, k=2, largest=False)
#
#     mutual_desc_matches_mask = get_mutual_matches(nn_desc_ids1[..., 0], nn_desc_ids2[:, 0, :])
#
#     # Create Lowe ratio test masks
#     lowe_ratio_mask1 = nn_desc_value1[..., 0] < nn_desc_value1[..., 1] * lowe_ratio
#     lowe_ratio_mask2 = nn_desc_value2[:, 0, :] < nn_desc_value2[:, 1, :] * lowe_ratio
#
#     nn_lowe_ratio_mask2 = torch.gather(lowe_ratio_mask2, -1, nn_desc_ids1[..., 0])
#
#     mutual_desc_matches_mask *= lowe_ratio_mask1 * nn_lowe_ratio_mask2
#
#     # Gather neighbouring descriptors and stack them with keypoints descriptors
#
#     neigh_desc1 = sample_neigh_desc(desc1, kp1, grid_size)
#
#     neigh_desc2 = sample_neigh_desc(desc2, kp2, grid_size)
#     nn_neigh_desc2 = torch.gather(neigh_desc2, 2, nn_desc_ids1[..., 0].unsqueeze(1).unsqueeze(-1).repeat(1, 8, 1, 64))
#
#     second_mask = inv_cos_sim_vec(neigh_desc1.view(-1, 512 * 8, 64), nn_neigh_desc2.view(-1, 512 * 8, 64)) < 0.25
#     second_mask = second_mask.view(-1, 8, 512).sum(dim=1) > 5
#
#     # print(mutual_desc_matches_mask.sum(dim=-1))
#     # print(second_mask.sum(dim=-1))
#
#     return mutual_desc_matches_mask * second_mask, nn_desc_ids1[..., 0]

# def get_nn_desc_idx(kp1_desc, kp2_desc, lowe_ratio=None):
#     desc_dist = torch.cdist(kp1_desc, kp2_desc)
#
#     if lowe_ratio is not None:
#         nn_desc_value1, nn_desc_idx1 = desc_dist.topk(dim=-1, k=2, largest=False)
#
#         desc_ratio1 = nn_desc_value1[..., 0] / nn_desc_value1[..., 1]
#
#         nn_desc_mask = desc_ratio1 < lowe_ratio
#
#         return nn_desc_mask, nn_desc_idx1[..., 0]
#
#     else:
#         nn_desc_idx = desc_dist.min(dim=-1)[1]
#         return torch.ones_like(nn_desc_idx), nn_desc_idx
#
