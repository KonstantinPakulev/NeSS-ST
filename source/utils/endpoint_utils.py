import torch
import torch.nn.functional as F

from source.utils.common_utils import create_coord_grid, sample_tensor, get_rect, get_rect_mask, apply_box_filter, \
    get_grad_kernels, get_hess_kernel, apply_kernel

"""
Common keys
"""
X = 'x'

DESC = 'desc'
SCORE = 'score'
KP = 'kp'
KP_SCORE = 'kp_score'
KP_DESC = 'kp_desc'
KP_SCALE_IDX = 'kp_scale_idx'

DESC1 = f'{DESC}1'
DESC2 = f'{DESC}2'

SCORE1 = f'{SCORE}1'
SCORE2 = f'{SCORE}2'

KP1 = f'{KP}1'
KP2 = f'{KP}2'

KP_SCORE1 = f'{KP_SCORE}1'
KP_SCORE2 = f'{KP_SCORE}2'

KP_DESC1 = f'{KP_DESC}1'
KP_DESC2 = f'{KP_DESC}2'

KP_SCALE_IDX1 = f'{KP_SCALE_IDX}1'
KP_SCALE_IDX2 = f'{KP_SCALE_IDX}2'

W_KP1 = 'w_kp1'
W_KP2 = 'w_kp2'

W_KP_MASK1 = 'w_kp1_mask'
W_KP_MASK2 = 'w_kp2_mask'


"""
Endpoint processing
"""


def select_kp_flat(score, nms_size, k,
                   score_thresh=None,
                   border_size=None):
    """
    :param score: B x 1 x H x W :type torch.float
    :param nms_size :type float
    :param k :type int
    :param score_thresh: float
    :param border_size: int
    :param border_value: int
    """
    nms_score = nms(score, nms_size)

    if border_size is not None:
        nms_score = mask_border(nms_score, border_size)

    if k == -1:
        raise NotImplementedError()

    else:
        kp_score, flat_kp = torch.topk(nms_score.view(nms_score.shape[0], -1), k)

    if score_thresh is not None:
        assert score.shape[0] == 1

        kp_score_mask = (kp_score > score_thresh).squeeze()
        kp_score, flat_kp = kp_score[:, kp_score_mask], flat_kp[:, kp_score_mask]

    return kp_score, flat_kp


def select_kp(score, nms_size, k,
              score_thresh=None,
              border_size=None,
              scale_factor=1.0, center=True, return_score=False):
    """
    :param score: B x 1 x H x W :type torch.float
    :param nms_size :type float
    :param k :type int
    :param score_thresh: float
    :param border_size: int
    :param scale_factor: float
    :param center: bool
    :param return_score: bool
    :return B x N x 2 :type torch.long
    """
    kp_score, flat_kp = select_kp_flat(score,
                                       nms_size, k,
                                       score_thresh,
                                       border_size)
    kp = flat2grid(flat_kp, score.shape[-1]).float()

    if center:
        kp = kp + 0.5

    if scale_factor != 1.0:
        kp = kp * scale_factor

    if return_score:
        return kp_score, kp

    else:
        return kp


def sample_tensor_patch(t, kp, patch_size,
                        image_shape,
                        mode='bilinear',
                        align_corners=False,
                        return_grid=False):
    """
    :param t: B x C x H x W
    :param kp: B x N x 2 or B x N x patch_size^2 x 2
    :param patch_size: int
    :param image_shape: (b, c, h, w)
    :param mode: str
    :param align_corners: bool
    :param return_grid: bool
    :return B x N x patch_size^2 x C
    """
    b, c = t.shape[:2]
    n = kp.shape[1]
    sq_patch_size = patch_size ** 2

    if len(kp.shape) == 4:
        kp_patch = sample_tensor(t, kp.view(b, n * sq_patch_size, 2), image_shape,
                                 mode,
                                 align_corners).view(b, n, sq_patch_size, c)

        return kp_patch

    else:
        if return_grid:
            kp_pg, kp_pg_mask = create_patch_grid(kp, patch_size, t.shape, mode='pm')

            kp_patch = sample_tensor(t, kp_pg.view(b, n * sq_patch_size, 2), image_shape,
                                     mode,
                                     align_corners).view(b, n, sq_patch_size, c)

            return kp_patch, kp_pg, kp_pg_mask

        else:
            kp_pg = create_patch_grid(kp, patch_size, t.shape, mode='p')

            kp_patch = sample_tensor(t, kp_pg.view(b, n * sq_patch_size, 2), image_shape,
                                     mode,
                                     align_corners).view(b, n, sq_patch_size, c)

            return kp_patch


def gather_tensor_patch(t, pg):
    """
    :param t: B x C x H x W
    :param pg: B x N x patch_size ** 2
    :return B x N x patch_size ** 2
    """
    b, _, h, w = t.shape
    sq_patch_size = pg.shape[-1]

    return t.view(b, -1).gather(-1, pg.view(b, -1)).view(b, -1, sq_patch_size)


def localize_kp(score, kp):
    dx_kernel, dy_kernel = get_grad_kernels(score.device)

    dxdx_kernel, dydy_kernel, dxdy_kernel = get_hess_kernel(score.device)

    dx = apply_kernel(score, dx_kernel)
    dy = apply_kernel(score, dy_kernel)

    dxdx = apply_kernel(score, dxdx_kernel)
    dydy = apply_kernel(score, dydy_kernel)
    dxdy = apply_kernel(score, dxdy_kernel)

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

    return kp_loc.squeeze(-2) * kp_det_mask.float()


def get_resized_image(image, scales, input_size_divisor=None):
    initial_shape = torch.tensor(image.shape[2:])

    resized_image = [image]
    resized_shift_scale = [torch.tensor([0.0, 0.0, 1.0, 1.0], device=image.device)]
    cropped_shape = [None]

    for s in scales:
        resized_image_grayi = F.interpolate(image, scale_factor=s, mode='bilinear')
        resized_shapei = torch.tensor(resized_image_grayi.shape[2:])
        cropped_shapei = None

        shift_scalei = torch.tensor([0.0, 0.0, *(resized_shapei / initial_shape)], device=image.device)

        if input_size_divisor is not None:
            rect = get_divisor_crop_rect(resized_shapei, input_size_divisor)

            resized_image_grayi = resized_image_grayi[:, :, rect[0]:rect[0] + rect[2], rect[1]:rect[1] + rect[3]]
            shift_scalei[:2] = rect[:2].to(image.device)
            cropped_shapei = rect[2:]

        resized_image.append(resized_image_grayi)
        resized_shift_scale.append(shift_scalei)
        cropped_shape.append(cropped_shapei)

    return resized_image, resized_shift_scale, cropped_shape


"""
Support utils
"""


def create_patch_grid(patch_coord, patch_size, shape,
                      center=False, dtype=torch.float,
                      mode='p', scale_factor=1.0):
    """
    :param patch_coord: B x N x 2; in image scale, default orientation is (y, x)
    :param patch_size: int; in patch scale
    :param shape: (b, c, h, w)
    :param center: bool
    :param dtype: grid dtype
    :param mode: str
    :param scale_factor: float
    :return B x N * patch_size**2 x 2 or B x N * patch_size**2, B x N * patch_size**2
    """
    # b, _, _, _ = shape
    b, n = patch_coord.shape[:2]

    patch_grid = create_coord_grid((b * n, 1, patch_size, patch_size),
                                   center=center,
                                   scale_factor=scale_factor).to(patch_coord.device).flip([-1])
    patch_center = torch.tensor([patch_size // 2, patch_size // 2]).to(patch_coord.device)

    if dtype == torch.float:
        patch_center = patch_center.float()
        patch_grid = patch_grid.view(b, n, -1, 2) - patch_center.view(1, 1, 1, 2) * scale_factor

    elif dtype == torch.long:
        patch_grid = patch_grid.long()
        patch_coord = patch_coord.long()
        patch_grid = patch_grid.view(b, n, -1, 2) - (patch_center.view(1, 1, 1, 2) * scale_factor).long()

    else:
        raise NotImplementedError

    patch_grid = patch_grid + patch_coord.unsqueeze(-2)

    if mode == 'pm':
        patch_grid_mask = get_rect_mask(patch_grid, get_rect(shape))

        return patch_grid, patch_grid_mask

    elif mode == 'p':
        return patch_grid

    else:
        raise NotImplementedError


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


def nms_3d(score, nms_size, num_channels=3, return_mask=False):
    """
    :param score: B x C x H x W
    :param nms_size: odd int
    :param return_mask: bool
    :return B x 1 x H x W
    """
    b, c, h, w = score.shape
    h_nms_size = nms_size // 2

    idx = F.max_pool3d(score,
                       kernel_size=(num_channels, nms_size, nms_size),
                       stride=1,
                       padding=(num_channels // 2, h_nms_size, h_nms_size),
                       return_indices=True)[1]

    coord = torch.arange(c * h * w, dtype=torch.float, device=score.device).\
        view(1, c, h, w).repeat(b, 1, 1, 1)

    nms_mask = idx == coord

    if return_mask:
        return (score > 0) & nms_mask

    else:
        return score * nms_mask.float()


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


def clamp_points(points, rect):
    """
    :param points: ... x 2; (y, x) orientation
    :param rect: (y, x, h, w)
    :return:
    """
    c_points = torch.zeros_like(points).to(points.device)

    y, x, h, w = rect

    c_points[..., 0] = points[..., 0].clamp(min=y, max=h)
    c_points[..., 1] = points[..., 1].clamp(min=x, max=w)

    return c_points


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


def grid2flat(ids, w):
    """
    :param ids: ... x 2, tensor of indices :type torch.long
    :param w: last dimension (W) of tensor of indices :type long
    """
    return w * ids[..., 0] + ids[..., 1]


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


"""
Legacy code
"""


def iterative_nms(score, nms_size):
    prev_nms_mask = nms(score, nms_size, True)
    prev_nms_box_mask = apply_box_filter(prev_nms_mask.float(), nms_size) > 0

    while True:
        score = score * (~prev_nms_box_mask).float()
        nms_mask = nms(score, nms_size, True) & (~prev_nms_box_mask)

        if nms_mask.sum() == 0:
            break

        nms_box_mask = apply_box_filter(nms_mask.float(), nms_size) > 0

        prev_nms_mask = nms_mask | prev_nms_mask
        prev_nms_box_mask = nms_box_mask | prev_nms_box_mask

    return prev_nms_mask


# LOG_SSCORE = 'log_sscore'
# LOG_SSCORE1 = f'{LOG_SSCORE}1'
# LOG_SSCORE2 = f'{LOG_SSCORE}2'

# def scatter_unique(kp, kp_mask, nms_size, shape):
#     b, c, h, w = shape
#
#     flat_kp = grid2flat(kp, w)
#
#     corr_mask = torch.zeros(shape, device=kp.device).view(b, -1)
#     corr_mask = corr_mask.scatter(-1,
#                                   flat_kp * kp_mask.long(),
#                                   kp_mask.float())
#     corr_mask = nms(corr_mask.view(b, 1, h, w), nms_size).view(b, -1)
#
#     return corr_mask.gather(-1, flat_kp * kp_mask.long()).bool()


# def spatial_soft_nms(log_score, soft_nms_kernel_size):
#     """
#     :param log_score: B x 1 x H x W
#     :param soft_nms_kernel_size: int
#     :param soft_strength: float
#     """
#     padding = soft_nms_kernel_size // 2
#
#     min_log_score = F.max_pool2d(-log_score, kernel_size=soft_nms_kernel_size, padding=padding, stride=1)  # B x 1 x H x W
#
#     exp = torch.exp(log_score + min_log_score)
#     weight = torch.ones((1, 1, soft_nms_kernel_size, soft_nms_kernel_size)).to(log_score.device)
#     sum_exp = F.conv2d(exp, weight=weight, padding=padding).clamp(min=1e-8)
#
#     return exp / sum_exp

#     patch_grid = clamp_points(patch_grid, get_rect(shape))
    #
    #     patch_grid = grid2flat(patch_grid, w)

# def masked_batch2list(batch, mask):
#     """
#     :param batch: B x N x ...
#     :param mask: B x N
#     """
#     l = []
#
#     for b, m in zip(batch, mask):
#         l.append(b[m])
#
#     return l

#     assert b == 1
#
#     kp_value_thresh_mask = (kp_value >= score_thresh).squeeze()
#
#     kp_value, flat_kp = kp_value[:, kp_value_thresh_mask], flat_kp[:, kp_value_thresh_mask]
# assert score.shape[0] == 1 and score_thresh is not None
#
# thresh_mask = score_nms >= score_thresh
# score_thresh = score_nms * thresh_mask.float()
#
# flat_kp = score_thresh.view(-1).nonzero().squeeze().unsqueeze(0)
# kp_value = score.view(-1)[flat_kp.squeeze(0)]

# def nms(score, nms_kernel_size, return_score=True):
#     """
#     :param score: B x 1 x H x W
#     :param nms_kernel_size: int
#     :param return_score: bool
#     :return B x 1 x H x W
#     """
#     b, c, h, w = score.shape
#     padding = nms_kernel_size // 2
#
#     max_score = F.max_pool2d(score, kernel_size=nms_kernel_size, stride=1, padding=padding)
#     max_score_mask = max_score == score
#
#     # To avoid cases when multiple ground-truths are available
#     p_max_score_mask = max_score_mask.float() + \
#                        max_score_mask.float() * (torch.randperm(h * w, dtype=torch.float, device=score.device).
#                                                  view(1, c, h, w).
#                                                  repeat(b, 1, 1, 1) + 1)
#
#     max_score_mask = max_score_mask & (p_max_score_mask == F.max_pool2d(p_max_score_mask,
#                                                                         kernel_size=nms_kernel_size,
#                                                                         stride=1, padding=padding))
#
#     if return_score:
#         return score * max_score_mask.float()
#
#     else:
#         return max_score_mask

