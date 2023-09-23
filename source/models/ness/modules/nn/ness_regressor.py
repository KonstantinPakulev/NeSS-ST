import torch.nn as nn

from source.models.ness.modules.nn.backbones import UNet


class NeSSRegressor(nn.Module):
    
    @staticmethod
    def from_config(module_config):
        ness_config = module_config.ness
        unet = UNet.from_config(ness_config)
        return NeSSRegressor(unet, ness_config.min_ness, ness_config.max_ness)

    def __init__(self, unet, min_ness, max_ness):
        super().__init__()
        self.unet = unet

        self.min_ness = min_ness
        self.max_ness = max_ness

    def forward(self, image):
        features = self.unet(image)

        return features[-1].clamp(min=self.min_ness, max=self.max_ness)
