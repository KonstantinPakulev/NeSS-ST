import argparse
import torch
import os

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage import io

import torchvision.transforms.functional as tv_F


"""
UNet
"""


class ConvLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, bias=True,
                 batch_norm=True, relu=True):
        conv = []

        if batch_norm:
            conv += [nn.BatchNorm2d(out_channels)]

        if relu:
            conv += [nn.ReLU()]

        padding = (kernel_size - 1) // 2

        conv = [nn.Conv2d(in_channels, out_channels,
                          kernel_size=kernel_size, padding=padding, bias=bias)]

        super().__init__(*conv)


class DownLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels,
                 is_first=False):
        super().__init__()

        if is_first:
            conv = [ConvLayer(in_channels, out_channels, batch_norm=False, relu=False),
                    ConvLayer(out_channels, out_channels)]

        else:
            conv = [nn.BatchNorm2d(in_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    ConvLayer(in_channels, out_channels),
                    ConvLayer(out_channels, out_channels)]

        super().__init__(*conv)


class UpLayer(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(nn.BatchNorm2d(in_channels),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(in_channels, in_channels,
                                                        kernel_size=2, stride=2))

        cat_channels = in_channels + skip_channels

        self.conv = nn.Sequential(ConvLayer(cat_channels, cat_channels),
                                  ConvLayer(cat_channels, out_channels))

    def forward(self, x, x_skip):
        x_up = self.up_conv(x)
        x_cat = torch.cat([x_up, x_skip], dim=1)

        return self.conv(x_cat)


class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.down_layers = nn.ModuleList([DownLayer(3, 16, True),
                                          DownLayer(16, 32),
                                          DownLayer(32, 64),
                                          DownLayer(64, 64),
                                          DownLayer(64, 64)])
        self.up_layers = nn.ModuleList([UpLayer(64, 64, 64),
                                        UpLayer(64, 64, 64),
                                        UpLayer(64, 32, 64),
                                        UpLayer(64, 16, 1)])

    def __call__(self, image):
        features = [image]

        for layer in self.down_layers:
            features.append(layer(features[-1]))

        skip_features = features[-2:0:-1]

        for layer, s_f in zip(self.up_layers, skip_features):
            features.append(layer(features[-1], s_f))

        return features


"""
Shi-Tomasi detector
"""

def create_coord_grid(shape, center=True, scale_factor=1.0):
    """
    :param shape: (b, _, h, w) :type tuple
    :param scale_factor: float
    :param center: bool
    :return B x H x W x 2; x, y orientation of coordinates located in center of pixels :type torch.tensor, float
    """
    b, _, h, w = shape

    grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])

    grid_x = grid_x.float().unsqueeze(-1)
    grid_y = grid_y.float().unsqueeze(-1)
    grid = torch.cat([grid_x, grid_y], dim=-1)  # H x W x 2

    # Each coordinate represents the location of the center of a pixel
    if center:
        grid += 0.5

    grid *= scale_factor

    return grid.unsqueeze(0).repeat(b, 1, 1, 1)


def apply_gaussian_filter(t, size, cov):
    patch_coord = create_coord_grid((1, 1, size, size))
    patch_center = torch.tensor([size / 2, size / 2]).view(1, 1, 1, 1, 2)

    diff = patch_coord - patch_center

    ll_pg = torch.exp(-0.5 * (diff.unsqueeze(-2) @ diff.unsqueeze(-1)).squeeze(-1).squeeze(-1) / cov)
    gauss_kernel = ll_pg / ll_pg.sum(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)

    t = F.conv2d(t, weight=gauss_kernel, padding=gauss_kernel.shape[2] // 2)

    return t


def get_eigen_values(t):
    """
    :param t: ... x 2 x 2
    :return ... x 2
    """
    tr = t.diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    d = tr ** 2 - 4 * t.det()

    d_mask = (d > 0) | (d == 0 & (tr > 0))

    sqrt_d = torch.sqrt(d * d_mask.float())

    eig_val = torch.stack([(tr + sqrt_d) / 2 * d_mask.float(),
                           (tr - sqrt_d) / 2 * d_mask.float()], dim=-1)

    return eig_val


def normalize_coord(grid, shape, align_corners=False):
    """
    :param grid: B x H x W x 2
    """
    h, w = shape[2:]

    # Make a copy to avoid in-place modification
    norm_grid = grid.clone()

    if align_corners:
        # If norm-grid values are top-left corners of pixels
        norm_grid[:, :, :, 0] = norm_grid[:, :, :, 0] / (w - 1) * 2 - 1
        norm_grid[:, :, :, 1] = norm_grid[:, :, :, 1] / (h - 1) * 2 - 1

    else:
        # If norm-grid values are centers of pixels
        norm_grid[:, :, :, 0] = norm_grid[:, :, :, 0] / w * 2 - 1
        norm_grid[:, :, :, 1] = norm_grid[:, :, :, 1] / h * 2 - 1

    return norm_grid


def sample_tensor(t, kp,
                  image_shape,
                  mode='bilinear',
                  align_corners=False):
    """
    :param t: B x C x H x W
    :param kp: B x N x 2
    :param image_shape: (b, c, h, w)
    :param mode: str
    :param align_corners: bool
    :return B x N x C
    """
    kp_grid = normalize_coord(kp[:, :, [1, 0]].unsqueeze(1), image_shape, align_corners)
    kp_t = F.grid_sample(t, kp_grid, mode=mode).squeeze(2).permute(0, 2, 1)

    return kp_t


class ShiTomasi:

    def __init__(self, window_size=3, window_cov=2):
        self.sobel_size = 3
        self.window_size = window_size
        self.window_cov = window_cov

    def __call__(self, image_gray):
        b, c, h, w = image_gray.shape
        device = image_gray.device

        dx_kernel = torch.tensor([[-0.5, 0, 0.5],
                                  [-1, 0, 1],
                                  [-0.5, 0, 0.5]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        dy_kernel = dx_kernel.permute(0, 1, 3, 2)

        dx = F.conv2d(image_gray, weight=dx_kernel, padding=dx_kernel.shape[2] // 2)
        dy = F.conv2d(image_gray, weight=dy_kernel, padding=dy_kernel.shape[2] // 2)
        dx2 = dx * dx
        dy2 = dy * dy
        dxdy = dx * dy

        smm = torch.stack([apply_gaussian_filter(dy2, self.window_size, self.window_cov),
                          apply_gaussian_filter(dxdy, self.window_size, self.window_cov),
                          apply_gaussian_filter(dxdy, self.window_size, self.window_cov),
                          apply_gaussian_filter(dx2, self.window_size, self.window_cov)], dim=-1).view(b, c, h, w, 2, 2)

        shi_score, _ = get_eigen_values(smm).min(dim=-1)

        return shi_score

    def localize_kp(self, kp, score):
        dx_kernel = torch.tensor([[0, 0, 0],
                                  [-0.5, 0, 0.5],
                                  [0, 0, 0]], device=kp.device, dtype=torch.float32).view(1, 1, 3, 3)
        dy_kernel = dx_kernel.permute(0, 1, 3, 2)

        dxdx_kernel = torch.tensor([[0, 0, 0],
                                    [1, -2, 1],
                                    [0, 0, 0]], device=kp.device, dtype=torch.float32).view(1, 1, 3, 3)

        dydy_kernel = torch.tensor([[0, 1, 0],
                                    [0, -2, 0],
                                    [0, 1, 0]], device=kp.device, dtype=torch.float32).view(1, 1, 3, 3)

        dxdy_kernel = 0.25 * torch.tensor([[1, 0, -1],
                                           [0, 0, 0],
                                           [-1, 0, 1]], device=kp.device, dtype=torch.float32).view(1, 1, 3, 3)

        dx = F.conv2d(score, weight=dx_kernel, padding=dx_kernel.shape[2] // 2)
        dy = F.conv2d(score, weight=dy_kernel, padding=dy_kernel.shape[2] // 2)

        dxdx = F.conv2d(score, weight=dxdx_kernel, padding=dxdx_kernel.shape[2] // 2)
        dydy = F.conv2d(score, weight=dydy_kernel, padding=dydy_kernel.shape[2] // 2)
        dxdy = F.conv2d(score, weight=dxdy_kernel, padding=dxdy_kernel.shape[2] // 2)

        kp_dx = sample_tensor(dx, kp, score.shape)
        kp_dy = sample_tensor(dy, kp, score.shape)

        kp_dxdx = sample_tensor(dxdx, kp, score.shape)
        kp_dydy = sample_tensor(dydy, kp, score.shape)
        kp_dxdy = sample_tensor(dxdy, kp, score.shape)

        kp_det = (kp_dxdx * kp_dydy - kp_dxdy ** 2).clamp(min=1e-8)
        kp_det_mask = kp_det > 1e-3

        kp_ihess_00 = kp_dxdx / kp_det
        kp_ihess_01 = -kp_dxdy / kp_det
        kp_ihess_11 = kp_dydy / kp_det

        kp_loc = torch.stack([-(kp_ihess_00 * kp_dy + kp_ihess_01 * kp_dx),
                              -(kp_ihess_01 * kp_dy + kp_ihess_11 * kp_dx)], dim=-1)

        return kp + kp_loc.squeeze(-2) * kp_det_mask.float()

"""
Loading functions
"""


def get_divisor_crop_rect(shape, size_divisor):
    if shape[0] % size_divisor != 0:
        new_height = (shape[0] // size_divisor) * size_divisor
        offset_h = torch.round((shape[0] - new_height) / 2.).long()
    else:
        offset_h = 0
        new_height = shape[0]

    if shape[1] % size_divisor != 0:
        new_width = (shape[1] // size_divisor) * size_divisor
        offset_w = torch.round((shape[1] - new_width) / 2.).long()
    else:
        offset_w = 0
        new_width = shape[1]

    rect = torch.tensor([offset_h, offset_w, new_height, new_width])

    return rect


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    for old_key in list(checkpoint.keys()):
        checkpoint[old_key[2:]] = checkpoint[old_key]
        del checkpoint[old_key]

    return checkpoint


def load_image(image_path):
    image = io.imread(image_path)

    image = tv_F.to_pil_image(image)
    image_gray = tv_F.to_grayscale(image)

    image = tv_F.to_tensor(image)
    image_gray = tv_F.to_tensor(image_gray)

    rect = get_divisor_crop_rect(torch.tensor(image.shape)[1:], 16)
    image = image[:, rect[0]:rect[0] + rect[2], rect[1]:rect[1] + rect[3]].unsqueeze(0)
    image_gray = image_gray[:, rect[0]:rect[0] + rect[2], rect[1]:rect[1] + rect[3]].unsqueeze(0)

    return image, image_gray


"""
NeSS-ST
"""


def mask_border(score, border, mask_value=0.0):
    """
    :param score: ... x H x W
    :param border: int
    :param mask_value: any type
    """
    masked_score = score.clone()
    masked_score[..., :border, :] = mask_value
    masked_score[..., :, :border] = mask_value
    masked_score[..., -border:, :] = mask_value
    masked_score[..., :, -border:] = mask_value

    return masked_score


def flat2grid(flat_ids, w):
    """
    :param flat_ids: ... x N tensor of indices taken from flattened tensor of shape ... x H x W
    :param w: Last dimension (W) of tensor from which indices were taken
    :return: ... x N x 2 tensor of coordinates in input tensor ... x H x W
    """
    y = flat_ids // w
    x = flat_ids - y * w

    y = y.unsqueeze(-1)
    x = x.unsqueeze(-1)

    return torch.cat((y, x), dim=-1)


def nms(score, nms_size, return_mask=False):
    """
    :param score: B x 1 x H x W
    :param nms_size: odd int
    :param return_mask: bool
    :return B x 1 x H x W
    """
    b, _, h, w = score.shape

    idx = F.max_pool2d(score,
                       kernel_size=nms_size,
                       stride=1,
                       padding=nms_size // 2,
                       return_indices=True)[1]

    coord = torch.arange(h * w, dtype=torch.float, device=score.device).view(1, 1, h, w).repeat(b, 1, 1, 1)

    nms_mask = idx == coord

    if return_mask:
        return (score > 0) & nms_mask

    else:
        return score * nms_mask.float()


class NeSSST(nn.Module):

    def __init__(self, min_ness=0, max_ness=32):
        super().__init__()
        self.unet = UNet()
        self.shi_tomasi = ShiTomasi()

        self.min_ness = min_ness
        self.max_ness = max_ness

    def __call__(self, image, image_gray, nms_size, k):
        """
        :param image: B x 3 x H x W; torch.tensor
        """
        ness = self.unet(image)[-1].clamp(min=self.min_ness, max=self.max_ness)
        shi_score = self.shi_tomasi(image_gray)

        sobel_size = self.shi_tomasi.sobel_size
        window_size = self.shi_tomasi.window_size

        border_size = max(sobel_size, window_size) // 2 + sobel_size // 2

        nms_shi_score_mask = nms(shi_score, nms_size, return_mask=True).float()
        nms_exp_ness_score = (-ness).exp() * nms_shi_score_mask.float()
        nms_exp_ness_score = mask_border(nms_exp_ness_score, border_size, 0)

        b, _, _, w = shi_score.shape

        flat_kp = nms_exp_ness_score.view(b, -1).topk(k, dim=-1)[1]
        kp = flat2grid(flat_kp, w) + 0.5
        kp = self.shi_tomasi.localize_kp(kp, shi_score)

        return kp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path')
    parser.add_argument('image_path')
    parser.add_argument('num_features', type=int)

    parser.add_argument('--output_path', '-O', default='output/nessst_kp.npy')
    parser.add_argument('--device', '-d', default='cpu')

    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    image_path = args.image_path
    num_features = args.num_features

    output_path = args.output_path
    device = torch.device(args.device)

    checkpoint = load_checkpoint(checkpoint_path)

    nessst = NeSSST()
    nessst.load_state_dict(checkpoint)
    nessst = nessst.eval()
    nessst = nessst.to(device)

    image, image_gray = load_image(image_path)

    kp = nessst(image.to(device), image_gray.to(device), 5, num_features)[..., [1, 0]]

    dir_path, file_name = os.path.split(output_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    np.save(output_path, kp.squeeze().detach().numpy())
