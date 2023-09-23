import torch

from source.datasets.base import utils as du

from source.datasets.base.utils import HDataWrapper
from source.projective.homography import sample_homography
from source.projective.warp import warp_points
from source.utils.endpoint_utils import create_patch_grid, sample_tensor_patch, mask_border, flat2grid


def generate_homographies(patch_size, scale_factor, num_samples, device):
    h = []

    for _ in range(num_samples):
        hi = sample_homography(patch_size, scale_factor)

        h.append(torch.tensor(hi, device=device, dtype=torch.float32).unsqueeze(0))

    h_batch = {du.H1: [],
               du.H2: [],
               du.SHIFT_SCALE1: [],
               du.SHIFT_SCALE2: [],
               du.IMAGE_SHAPE1: torch.Size((num_samples, *[1, patch_size, patch_size])),
               du.IMAGE_SHAPE2: torch.Size((num_samples, *[1, patch_size, patch_size]))}

    for hi in h:
        h_batch[du.H1].append(hi)
        h_batch[du.H2].append(hi.inverse())
        h_batch[du.SHIFT_SCALE1].append(torch.tensor([0, 0, 1, 1]).unsqueeze(0))
        h_batch[du.SHIFT_SCALE2].append(torch.tensor([0, 0, 1, 1]).unsqueeze(0))

    h_batch[du.H1] = torch.cat(h_batch[du.H1], dim=0).view(num_samples, 3, 3)
    h_batch[du.H2] = torch.cat(h_batch[du.H2], dim=0).view(num_samples, 3, 3)

    h_batch[du.SHIFT_SCALE1] = torch.cat(h_batch[du.SHIFT_SCALE1], dim=0).view(num_samples, 4)
    h_batch[du.SHIFT_SCALE2] = torch.cat(h_batch[du.SHIFT_SCALE2], dim=0).view(num_samples, 4)

    h_data = HDataWrapper().init_from_batch(h_batch, device)

    return h_data


def get_ww_patch_kp(image_gray, kp, h_data,
                    scaled_patch_size, patch_size, num_samples, k,
                    nms_size,
                    base_detector):
    b = image_gray.shape[0]
    bk = b * k
    kn = k * num_samples
    bkn = bk * num_samples
    ext_patch_size = patch_size + 2
    nms_border = (patch_size - nms_size) // 2

    center = torch.tensor([scaled_patch_size // 2 + 0.5], device=kp.device). \
        view(1, 1, 1). \
        repeat(num_samples, 1, 2)
    w_center = warp_points(center, h_data)[0]
    w_center_nn = w_center.long() + 0.5

    w_center_nn_pg = create_patch_grid(w_center_nn.
                                       permute(1, 0, 2),
                                       ext_patch_size, image_gray.shape). \
        permute(1, 0, 2, 3). \
        view(num_samples, -1, 2)

    ww_center_nn_pg = (warp_points(w_center_nn_pg, h_data.swap())[0].view(num_samples, 1, -1, 2)
                       - center.unsqueeze(-2)
                       + kp.view(1, bk, 1, 2)). \
        view(num_samples, b, k, -1, 2). \
        permute(1, 2, 0, 3, 4). \
        reshape(b, kn, -1, 2)

    w_ig_patch = sample_tensor_patch(image_gray, ww_center_nn_pg, ext_patch_size, image_gray.shape).squeeze(-1)

    del w_center_nn_pg, ww_center_nn_pg

    ig_patch_w_center = (torch.tensor([ext_patch_size // 2 + 0.5], device=kp.device).
                         view(1, 1, 1).
                         repeat(num_samples, 1, 2)
                         + (w_center - w_center_nn)). \
        view(1, 1, num_samples, 1, 2). \
        repeat(b, k, 1, 1, 1). \
        view(bkn, 1, 2)

    w_ig_patch = sample_tensor_patch(w_ig_patch.view(bkn, 1, ext_patch_size, ext_patch_size),
                                     ig_patch_w_center,
                                     patch_size, (bkn, 1, ext_patch_size, ext_patch_size)). \
        view(bkn, 1, patch_size, patch_size)

    del ig_patch_w_center

    w_ss_patch = base_detector.get_score(w_ig_patch).view(bkn, patch_size, patch_size)
    w_ss_patch = mask_border(w_ss_patch, nms_border, -1).view(bkn, -1)

    w_patch_kp = (flat2grid(w_ss_patch.argmax(dim=-1), patch_size).unsqueeze(-2) \
                  - patch_size // 2 \
                  + w_center.repeat(bk, 1, 1)).\
        view(b, k, num_samples, 2).\
        permute(2, 0, 1, 3).\
        reshape(num_samples, -1, 2)

    ww_patch_kp = (warp_points(w_patch_kp, h_data.swap())[0] - center).view(num_samples, b, k, 2)

    return ww_patch_kp
