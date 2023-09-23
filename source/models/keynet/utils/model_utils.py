import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from kornia.filters import filter2d


class feature_extractor(nn.Module):
    '''
        It loads both, the handcrafted and learnable blocks
    '''
    def __init__(self):
        super(feature_extractor, self).__init__()

        self.hc_block = handcrafted_block()
        self.lb_block = learnable_block()

    def forward(self, x):
        x_hc = self.hc_block(x)
        x_lb = self.lb_block(x_hc)
        return x_lb


class handcrafted_block(nn.Module):
    '''
        It defines the handcrafted filters within the Key.Net handcrafted block
    '''
    def __init__(self):
        super(handcrafted_block, self).__init__()

    def forward(self, x):

        sobel = kornia.filters.spatial_gradient(x)
        dx, dy = sobel[:, :, 0, :, :], sobel[:, :, 1, :, :]

        sobel_dx = kornia.filters.spatial_gradient(dx)
        dxx, dxy = sobel_dx[:, :, 0, :, :], sobel_dx[:, :, 1, :, :]

        sobel_dy = kornia.filters.spatial_gradient(dy)
        dyy = sobel_dy[:, :, 1, :, :]

        hc_feats = torch.cat([dx, dy, dx**2., dy**2., dx*dy, dxy, dxy**2., dxx, dyy, dxx*dyy], dim=1)

        return hc_feats


class learnable_block(nn.Module):
    '''
        It defines the learnable blocks within the Key.Net
    '''
    def __init__(self, in_channels=10):
        super(learnable_block, self).__init__()

        self.conv0 = conv_blck(in_channels)
        self.conv1 = conv_blck()
        self.conv2 = conv_blck()

    def forward(self, x):
        x = self.conv2(self.conv1(self.conv0(x)))
        return x


def conv_blck(in_channels=8, out_channels=8, kernel_size=5,
              stride=1, padding=2, dilation=1):
    '''
    Default learnable convolutional block.
    '''
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))


def custom_pyrdown(input: torch.Tensor, factor: float = 2., border_type: str = 'reflect', align_corners: bool = False) -> torch.Tensor:
    r"""Blurs a tensor and downsamples it.
    Args:
        input (tensor): the tensor to be downsampled.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail.
    Return:
        torch.Tensor: the downsampled tensor.
    Examples:
        >>> input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        >>> pyrdown(input, align_corners=True)
        tensor([[[[ 3.7500,  5.2500],
                  [ 9.7500, 11.2500]]]])
    """
    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")
    kernel: torch.Tensor = _get_pyramid_gaussian_kernel()
    b, c, height, width = input.shape
    # blur image
    x_blur: torch.Tensor = filter2d(input, kernel, border_type)

    # downsample.
    out: torch.Tensor = F.interpolate(x_blur, size=(int(height // factor), int(width // factor)), mode='bilinear',
                                      align_corners=align_corners)
    return out


# Utility from Kornia: https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/transform/pyramid.html
def _get_pyramid_gaussian_kernel() -> torch.Tensor:
    """Utility function that return a pre-computed gaussian kernel."""
    return torch.tensor([[
        [1., 4., 6., 4., 1.],
        [4., 16., 24., 16., 4.],
        [6., 24., 36., 24., 6.],
        [4., 16., 24., 16., 4.],
        [1., 4., 6., 4., 1.]
    ]]) / 256.