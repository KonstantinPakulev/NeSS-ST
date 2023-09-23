import torch

from source.pose.matchers.base import BaseMatcher


class L2Matcher(BaseMatcher):

    def __init__(self, lowe_ratio, device):
        self.lowe_ratio = lowe_ratio
        self.device = device

    def match(self, kp_desc1, kp_desc2,
              kp_desc_mask1=None, kp_desc_mask2=None):
        desc_dist = torch.cdist(kp_desc1.to(self.device), kp_desc2.to(self.device))

        if self.lowe_ratio is not None:
            nn_desc_value1, nn_desc_idx1 = desc_dist.topk(dim=-1, k=2, largest=False)
            nn_desc_value2, nn_desc_idx2 = desc_dist.topk(dim=-2, k=2, largest=False)

            cyc_match_mask1 = get_cyclic_match_mask(nn_desc_idx1[..., 0], nn_desc_idx2[:, 0, :])

            desc_ratio1 = nn_desc_value1[..., 0] / nn_desc_value1[..., 1]
            desc_ratio2 = nn_desc_value2[:, 0, :] / nn_desc_value2[:, 1, :]

            nn_desc_ratio2 = torch.gather(desc_ratio2, -1, nn_desc_idx1[..., 0])

            mm_desc_mask1 = cyc_match_mask1 & (desc_ratio1 < self.lowe_ratio) & (nn_desc_ratio2 < self.lowe_ratio)

            nn_desc_idx1 = nn_desc_idx1[..., 0]

        else:
            nn_desc_idx1 = desc_dist.min(dim=-1)[1]
            nn_desc_idx2 = desc_dist.min(dim=-2)[1]

            mm_desc_mask1 = get_cyclic_match_mask(nn_desc_idx1, nn_desc_idx2)

        if kp_desc_mask1 is not None:
            mm_desc_mask1 &= kp_desc_mask1.to(self.device)

        if kp_desc_mask2 is not None:
            mm_desc_mask1 &= torch.gather(kp_desc_mask2.float().to(self.device), -1, nn_desc_idx1).bool()

        return mm_desc_mask1, nn_desc_idx1


"""
Support utils
"""


def get_cyclic_match_mask(nn_idx1, nn_idx2):
    idx = torch.arange(0, nn_idx1.shape[1]).repeat(nn_idx1.shape[0], 1).to(nn_idx1.device)

    nn_idx = torch.gather(nn_idx2, -1, nn_idx1)

    return idx == nn_idx