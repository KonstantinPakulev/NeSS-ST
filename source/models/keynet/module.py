import torch
import torch.nn as nn
import torch.nn.functional as F

from source.models.keynet.utils.model_utils import feature_extractor, custom_pyrdown


class KeyNet(nn.Module):

    @staticmethod
    def from_config(config):
        return KeyNet(config.num_filters, config.num_levels, config.kernel_size)

    def __init__(self, num_filters, num_levels, kernel_size):
        super(KeyNet, self).__init__()
        self.num_levels = num_levels
        padding = kernel_size // 2

        self.feature_extractor = feature_extractor()
        self.last_conv = nn.Sequential(nn.Conv2d(in_channels=num_filters*self.num_levels,
                                                 out_channels=1, kernel_size=kernel_size, padding=padding),
                                       nn.ReLU(inplace=True))

    def forward(self, x):
        """
        x - input image
        """
        shape_im = x.shape
        for i in range(self.num_levels):
            if i == 0:
                feats = self.feature_extractor(x)
            else:
                x = custom_pyrdown(x, factor=1.2)
                feats_i = self.feature_extractor(x)
                feats_i = F.interpolate(feats_i, size=(shape_im[2], shape_im[3]), mode='bilinear')
                feats = torch.cat([feats, feats_i], dim=1)

        scores = self.last_conv(feats)
        return scores
