import torch

"""
Modules blocks
"""

BACKBONE = 'backbone'
DETECTOR = 'detector'
DESCRIPTOR = 'descriptor'
DETECTOR_DESCRIPTOR = 'detector_descriptor'


def depth_to_space(x, grid_size):
    b, cc, hc, wc = x.shape
    sq_grid_size = grid_size**2

    x = x.permute(0, 2, 3, 1)

    c = int(cc / sq_grid_size)
    h = int(hc * grid_size)
    w = int(wc * grid_size)

    x = x.reshape(b, hc, wc, sq_grid_size, c)
    x = x.split(grid_size, 3)
    x = [xi.reshape(b, hc, w, c) for xi in x]

    x = torch.stack(x, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(b, h, w, c)

    x = x.permute(0, 3, 1, 2)

    return x


def space_to_depth(x, grid_size):
    b, c, h, w = x.shape

    hc = h // grid_size
    wc = w // grid_size

    x = x.view(b, c, hc, grid_size, wc, grid_size)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # B x grid_size x grid_size x C x HC x WC

    return x.view(b, c * (grid_size ** 2), hc, wc)
