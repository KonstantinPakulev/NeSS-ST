import torch
import numpy as np
from skimage.feature import match_descriptors

from source.pose.matchers.base import BaseMatcher


class HammingMatcher(BaseMatcher):

    def __init__(self, lowe_ratio):
        self.lowe_ratio = lowe_ratio

    def match(self, kp_desc1, kp_desc2):
        if kp_desc1.shape[0] > 1:
            raise NotImplementedError

        matches = match_descriptors(kp_desc1[0].numpy(), kp_desc2[0].numpy(), cross_check=True, max_ratio=self.lowe_ratio)

        mm_desc_mask1 = np.zeros((kp_desc1.shape[1]))
        mm_desc_mask1[matches[:, 0]] = 1.0
        mm_desc_mask1 = torch.tensor(mm_desc_mask1, dtype=torch.bool).unsqueeze(0)

        nn_desc_idx1 = np.zeros((kp_desc1.shape[1]))
        nn_desc_idx1[matches[:, 0]] = matches[:, 1]
        nn_desc_idx1 = torch.tensor(nn_desc_idx1, dtype=torch.long).unsqueeze(0)

        return mm_desc_mask1, nn_desc_idx1