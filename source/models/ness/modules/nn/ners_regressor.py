import torch.nn as nn

from source.models.ness.modules.nn.backbones import UNet


class NeRSRegressor(nn.Module):

    @staticmethod
    def from_config(module_config):
        ners_config = module_config.ners
        unet = UNet.from_config(ners_config)
        return NeRSRegressor(unet)

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, image):
        features = self.unet(image)

        return features[-1].sigmoid()
