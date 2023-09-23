import torch
import numpy as np
from PIL import Image

import source.datasets.base.utils as du

from source.datasets.base.transforms import RandomCrop, CropBase, BaseTransformWrapper, ImageDepthCropWrapper, \
    CalibToTensorWrapper,ImageDepthTFactory, ImageDepthCalibFeaturesTFactory, CROP_RECT


TRAIN = 'train'
TEST = 'test'

DEPTH_CROP = 'depth_crop'


def get_megadepth_transforms(dataset_config, input_size_divisor):
    transforms_config = dataset_config.transforms

    if transforms_config.base == TRAIN:
        return MegaDepthTrainTFactory(transforms_config, input_size_divisor).create()

    elif transforms_config.base == TEST:
        return ImageDepthCalibFeaturesTFactory(transforms_config, input_size_divisor).create()

    else:
        raise ValueError


class MegaDepthTrainTFactory(ImageDepthTFactory):

    def _crop(self, value):
        if value.type == DEPTH_CROP:
            return [MegaDepthCropWrapper(DepthCrop())]

        else:
            return super()._crop(value)

    def _to_tensor(self):
        return super()._to_tensor() + [CalibToTensorWrapper()]


"""
Transforms
"""

class MegaDepthCropWrapper(ImageDepthCropWrapper):

    def _get_crop_reference(self, i):
        if isinstance(self.crop, DepthCrop):
            return f"{du.DEPTH}{i}"

        else:
            return super()._get_crop_reference(i)


"""
Transform units
"""

class DepthCrop(CropBase):

    def get_rect(self, depth):
        depth = np.array(depth)

        column_mask = depth.sum(axis=-1) > 0
        row_mask = depth.sum(axis=-2) > 0

        h_loc = column_mask.nonzero()[0]
        w_loc = row_mask.nonzero()[0]

        offset_h = h_loc[0]
        new_height = h_loc[-1] - offset_h

        offset_w = w_loc[0]
        new_width = w_loc[-1] - offset_w

        rect = (offset_h, offset_w, new_height, new_width)

        return rect


"""
Legacy code
"""

# HA_CROP = 'ha_crop'
# class HACropWrapper(ImageDepthCropWrapper):
#
#     def _transform_branch(self, item, bundle, i):
#         item, bundle = super()._transform_branch(item, bundle, i)
#
#         recti = bundle[f"{CROP_RECT}{i}"]
#         ha_kpi = item[f"{du.HA_KP}{i}"]
#
#         ha_kp_maski = (ha_kpi[:, 0] >= recti[0]) & (ha_kpi[:, 0] < (recti[0] + recti[2])) & \
#                       (ha_kpi[:, 1] >= recti[1]) & (ha_kpi[:, 1] < (recti[1] + recti[3]))
#
#         item[f"{du.HA_KP}{i}"] = ha_kpi - np.array(recti[:2])[None]
#         item[f"{du.HA_KP_MASK}{i}"] = ha_kp_maski
#
#         return item, bundle


# class HAToTensorWrapper(BaseTransformWrapper):
#
#     def __init__(self):
#         pass
#
#     def _base_key(self):
#         return du.HA_KP
#
#     def _transform_branch(self, item, bundle, i):
#         item[f"{du.HA_KP_VALUE}{i}"] = torch.from_numpy(item[f"{du.HA_KP_VALUE}{i}"].astype(np.float32, copy=False))
#         item[f"{du.HA_KP}{i}"] = torch.from_numpy(item[f"{du.HA_KP}{i}"].astype(np.float32, copy=False))
#         item[f"{du.HA_KP_COV_EIGV}{i}"] = torch.from_numpy(item[f"{du.HA_KP_COV_EIGV}{i}"].astype(np.float32, copy=False))
#         item[f"{du.HA_KP_MASK}{i}"] = torch.from_numpy(item[f"{du.HA_KP_MASK}{i}"].astype(np.bool, copy=False))
#
#         return item, bundle

# if value.type == HA_CROP:
#     return [HACropWrapper(RandomCrop((value.crop_size, value.crop_size)))]
#
# eli

# from source.projective.homography import sample_homography





# class MegaDepthTransformsFactory(IDCFeaturesTransformsFactory):
#
#     def _crop_wrapper(self, crop):
#         return MegaDepthCropWrapper(crop)



# class MegaDepthWarpH(BaseTransform):
#
#     def _base_key(self):
#         return du.IMAGE
#
#     def _transform_branches(self, item, bundle):
#         h1 = sample_homography(item[du.IMAGE1].size[::-1],
#                                 True, False, False, False,
#                                 5, 25, 0.1,
#                                 0, 0.3, 1.0, np.pi / 2)
#
#         item[du.IMAGE2] = item[du.IMAGE1].transform(item[du.IMAGE1].size,
#                                                     Image.PERSPECTIVE, h1.reshape(-1),
#                                                     Image.BILINEAR)
#
#         item[du.H1] = np.linalg.inv(h1)
#         item[du.H2] = h1
#
#         return item

# elif transforms_config.base == VAL:
#     return get_val_transforms(transforms_config, input_channels)

# def get_val_transforms(transforms_config, input_channels):
#     size = (transforms_config.height, transforms_config.width)
#
#     item_transforms = [ImageDepthToPILImage()]
#
#     pre_fit = transforms_config.get(du.PRE_FIT)
#
#     if pre_fit is not None:
#         for pre_key, pre_value in pre_fit.items():
#             if pre_key == 'grayscale':
#                 if 1 not in input_channels:
#                     item_transforms += [ImageToGrayScale()]
#
#             else:
#                 raise ValueError
#
#     if 1 in input_channels:
#         item_transforms += [ImageToGrayScale()]
#
#     item_transforms += [ImageDepthResize(AspectResize(size, True)),
#                         MegaDepthCrop(CentralCrop(size, True)),
#                         ImageDepthToTensor(),
#                         CalibToTensor()]
#
#     return item_transforms

# if 'warp_h' not in pre_fit.keys():
#     item_transforms += []
#
# else:
#     item_transforms += [HToTensor()]

# item_transforms += []

# elif pre_key == 'warp_h':
#     item_transforms += [MegaDepthWarpH()]
#
# else:
#     raise ValueError

# item[f"{du.HA_KP_VIS}{i}"] = torch.from_numpy(item[f"{du.HA_KP_VIS}{i}"].astype(np.float32, copy=False))

# shi_kp_key = f"{du.SHI_KP}{i}"
# shi_kp_grad_key = f"{du.SHI_KP_GRAD}{i}"
# shi_kp_count_key = f"{du.SHI_KP_COUNT}{i}"

# if shi_kp_key in item:
#     item[shi_kp_key] = torch.from_numpy(item[shi_kp_key].astype(np.float32, copy=False))
#     item[shi_kp_grad_key] = torch.from_numpy(item[shi_kp_grad_key].astype(np.float32, copy=False))
#     item[shi_kp_count_key] = torch.from_numpy(item[shi_kp_count_key].astype(np.float32, copy=False))

# def clamp_shi_kp(shi_kp, shi_kp_count, rect, border=3):
#     shift = np.array(rect[:2])[None, :]
#
#     shi_kp = shi_kp - shift
#
#     shi_kp_mask = (shi_kp[:, 0] >= border) & \
#                   (shi_kp[:, 1] >= border) & \
#                   (shi_kp[:, 0] < (rect[2] - border)) & \
#                   (shi_kp[:, 1] < (rect[3] - border))
#
#     shi_kp_count = shi_kp_count * shi_kp_mask.astype(np.int)
#
#     return shi_kp, shi_kp_count

# shi_kp_key = f"{du.SHI_KP}{i}"
# shi_kp_count_key = f"{du.SHI_KP_COUNT}{i}"
#
# if shi_kp_key in item and not isinstance(self.crop, MegaDepthSharedAreaCrop):
#     shi_kp, shi_kp_count = clamp_shi_kp(item[shi_kp_key],
#                                         item[shi_kp_count_key],
#                                         self.last_rect)
#
#     item[shi_kp_key] = shi_kp
#     item[shi_kp_count_key] = shi_kp_count
#
# return item

# class MegaDepthToPILImage(ImageDepthToPILImage):
#
#     def _transform_branch(self, item, i):
#         item = super()._transform_branch(item, i)
#
#         return item

# class MegaDepthPairScale(ImageTransform):
#
#     def __init__(self, scale_factors):
#         self.scale_factors = scale_factors
#
#     def _transform_branches(self, item):
#         scale_factor = np.random.choice(self.scale_factors)
#
#         in_size, _ = FactorResize(scale_factor).get_size_scale(item[du.IMAGE1])
#
#         image2 = F.resize(item[du.IMAGE1], in_size)
#
#         out_size, _ = FactorResize(1.0 / scale_factor).get_size_scale(image2)
#
#         image2 = F.resize(image2, out_size)
#
#         item[du.IMAGE2] = image2
#         item[du.IMAGE_NAME2] = item[du.IMAGE_NAME1] + "_resized"
#
#         return item

# for post_key, post_value in post_fit.items():
#     if post_key == 'pair_scale':
#         item_transforms += [MegaDepthPairScale(post_value.scale_factors)]
#
#     else:
#         raise ValueError

# class MegaDepthMergeWarps(MegaDepthTransform):
#
#     def _transform_branches(self, item):
#         image1 = item[du.IMAGE1].unsqueeze(0)
#         image2 = item[du.IMAGE2].unsqueeze(0)
#
#         scene_data = SceneDataWrapper(item, torch.device('cpu')).unsqueeze()
#
#         w_image2, vis_mask1 = warp_image_rbt(image2, scene_data, 'im')
#         w_image1, vis_mask2 = warp_image_rbt(image1, scene_data.swap(), 'im')
#
#         idx = np.random.randint(0, 2)
#
#         if idx == 1:
#             image1 = image2 * (~vis_mask2).float() + w_image1
#             item[du.IMAGE1] = image1.squeeze(0)
#             item[du.VIS_MASK1] = vis_mask2.squeeze(0)
#
#         else:
#             image2 = image1 * (~vis_mask1).float() + w_image2
#             item[du.IMAGE2] = image2.squeeze(0)
#             item[du.VIS_MASK1] = vis_mask1.squeeze(0)
#
#         item_keys = list(item.keys())
#
#         for key in item_keys:
#             if any(map(key.__contains__, [du.DEPTH, du.EXTRINSICS, du.INTRINSICS, du.SHIFT_SCALE])):
#                 del item[key]
#
#         return item

# class CopyColorJitter:
#
#     def __init__(self, brightness, saturation):
#         self.brightness = brightness
#         self.saturation = saturation
#
#     def __call__(self, item):
#         item[du.IMAGE_NAME2] = item[f"{du.IMAGE_NAME1}"] + "_copy"
#
#         image_copy = F.adjust_brightness(item[du.IMAGE1],
#                                          float(torch.empty(1).uniform_(1.0 - self.brightness, 1.0 + self.brightness)))
        # image_copy = F.adjust_saturation(image_copy,
        #                                  float(torch.empty(1).uniform_(1.0 - self.saturation, 1.0 + self.saturation)))

        # item[du.IMAGE2] = image_copy

        # return item


# elif add_key == 'copy_color_jitter':
#     item_transforms += [CopyColorJitter(add_value.brightness,
#                                         add_value.saturation)]

# def get_megadepth_test_transforms(dataset_config, input_channels, input_size_divisor):


# if du.MEMORY_BANK_DATA in item:
#     item[du.MEMORY_BANK_DATA].to_tensor_()
#
# if du.C_IMAGE1 in item:
#     item[du.C_IMAGE1] = F.to_tensor(item[du.C_IMAGE1])

# if du.MEMORY_BANK_DATA in item and \
#         (isinstance(self.crop, du.CentralCrop) or isinstance(self.crop, du.RandomCrop)):
#     item[du.MEMORY_BANK_DATA].crop_(rect1, self.nms_kernel_size, self.kr_size)
#
# if du.C_IMAGE1 in item:
#     item[du.C_IMAGE1] = F.crop(item[du.C_IMAGE1], *rect1)

# nms_kernel_size = None, kr_size = None
# self.nms_kernel_size = nms_kernel_size
# self.kr_size = kr_size

# import math
# import cv2
# from source.utils.projectivity_utils import sample_homography, resize_homography, crop_homography

# def aggregate_sift_kp(sift_kp, image_shape):
#     _, h, w = image_shape
#
#     flat_vis_sift_kp = grid2flat(clamp_points(torch.round(sift_kp).long(), get_clamp_rect(image_shape, 0)), w)
#
#     cum_score = torch.zeros((1, h * w))
#     cum_score = cum_score.scatter_add(-1, flat_vis_sift_kp, torch.ones_like(flat_vis_sift_kp).float()).view(1, 1, h,w).clamp(min=1e-8)
#
#     kr_size = 3
#     nms_kernel_size = 13
#     top_k = 512
#
#     border_mask_size = 3 * (kr_size // 2) + nms_kernel_size // 2
#     a_sift_kp_value, flat_a_sift_kp = select_kr_flat(cum_score, nms_kernel_size, kr_size, top_k, border_mask_size)
#
#     cum_kp = torch.zeros((1, h * w, 2))
#     cum_kp = cum_kp.scatter_add(-2, flat_vis_sift_kp.unsqueeze(-1).repeat(1, 1, 2), sift_kp).permute(0, 2, 1).view(1, 2, h, w)
#
#     kp_kernel = torch.ones(1, 1, kr_size, kr_size)
#
#     cum_kp_y = apply_kernel(cum_kp[:, 0, None, :, :], kp_kernel)
#     cum_kp_x = apply_kernel(cum_kp[:, 1, None, :, :], kp_kernel)
#     cum_kp = torch.cat([cum_kp_y, cum_kp_x], dim=1)
#
#     cum_count = torch.zeros((1, h * w))
#     cum_count = cum_count.scatter_add(-1, flat_vis_sift_kp, torch.ones_like(flat_vis_sift_kp).float()).view(1, 1, h, w)
#
#     cum_count = apply_kernel(cum_count, kp_kernel).clamp(min=1e-8)
#
#     cum_mean_kp = (cum_kp / cum_count).permute(0, 2, 3, 1).view(1, -1, 2)
#
#     loc_sift_kp = cum_mean_kp.gather(dim=-2, index=flat_a_sift_kp.unsqueeze(-1).repeat(1, 1, 2)).view(-1, 2)
#
#     return loc_sift_kp

# item[du.SIFT_KP1] = torch.from_numpy(item[du.SIFT_KP1].astype(np.float32, copy=False))
# item[du.SIFT_KP2] = torch.from_numpy(item[du.SIFT_KP2].astype(np.float32, copy=False))
#
# item[du.SIFT_KP1] = aggregate_sift_kp(item[du.SIFT_KP1].unsqueeze(0), item[du.IMAGE1].shape)
# item[du.SIFT_KP2] = aggregate_sift_kp(item[du.SIFT_KP2].unsqueeze(0), item[du.IMAGE2].shape)
# from .source.models.joint_net.utils.endpoint_utils import select_kr_flat, apply_kernel

# def get_megadepth_ha_transforms(dataset_config):
#     num_samples = dataset_config[du.NUM_SAMPLES]
#     size = (dataset_config[du.HEIGHT], dataset_config[du.WIDTH])
#
#     item_transforms = [MegaDepthHAGenHomography(num_samples),
#                        MegaDepthHAToPILImage(),
#                        MegaDepthHAResize(du.AspectResize(size, False)),
#                        MegaDepthHAToTensor()]
#
#     return item_transforms
#
#
# def get_megadepth_labeling_transforms(dataset_config, input_channels, input_size_divisor):
#     size = (dataset_config[du.HEIGHT], dataset_config[du.WIDTH])
#
#     item_transforms = [MegaDepthToPILImage()]
#
#     if 1 in input_channels:
#         item_transforms += [MegaDepthToGrayScale(input_channels)]
#
#     item_transforms += [MegaDepthCrop(MegaDepthSharedAreaCrop()),
#                         MegaDepthResize(du.AspectResize(size, True)),
#                         MegaDepthCrop(du.DivisorCrop(input_size_divisor)),
#                         MegaDepthToTensor()]
#
#     return item_transforms

# """
# MegaDepthHA transforms
# """
#
#
# class MegaDepthHAToPILImage:
#
#     def __call__(self, item):
#         image1, image2 = item[du.IMAGE1], item[du.IMAGE2]
#
#         item[du.IMAGE1] = F.to_pil_image(image1)
#         item[du.IMAGE2] = [F.to_pil_image(im) for im in image2]
#
#         return item
#
#
# class MegaDepthHAToTensor:
#
#     def __call__(self, item):
#         image1, image2 = item[du.IMAGE1], item[du.IMAGE2]
#         h12, h21 = item[du.H12], item[du.H21]
#
#         item[du.IMAGE1] = F.to_tensor(image1)
#         item[du.IMAGE2] = torch.cat([F.to_tensor(im) for im in image2])
#
#         item[du.H12] = torch.from_numpy(h12.astype(np.float32, copy=False))
#         item[du.H21] = torch.from_numpy(h21.astype(np.float32, copy=False))
#
#         return item
#
#
# class MegaDepthHAResize:
#
#     def __init__(self, resize):
#         self.resize = resize
#
#     def __call__(self, item):
#         image1, image2 = item[du.IMAGE1], item[du.IMAGE2]
#         h12, h21 = item[du.H12], item[du.H21]
#
#         size1, scale1 = self.resize.get_size_scale(image1)
#
#         for i, (image2i, h12i, h21i) in enumerate(zip(image2, h12, h21)):
#             size2i, scale2i = self.resize.get_size_scale(image2i)
#
#             image2[i] = F.resize(image2i, size2i)
#
#             h12[i] = resize_homography(h12i, scale_factor1=scale1, scale_factor2=scale2i)
#             h21[i] = resize_homography(h21i, scale_factor1=scale2i, scale_factor2=scale1)
#
#         item[du.IMAGE1] = F.resize(image1, size1)
#         item[du.IMAGE2] = image2
#
#         item[du.H12] = h12
#         item[du.H21] = h21
#
#         return item
#
#
# class MegaDepthHACrop:
#
#     def __init__(self, crop):
#         self.crop = crop
#
#     def __call__(self, item):
#         image1, image2 = item[du.IMAGE1], item[du.IMAGE2]
#         h12, h21 = item[du.H12], item[du.H21]
#
#         rect1 = self.crop.get_rect(image1)
#
#         for i, (image2i, h12i, h21i) in enumerate(zip(image2, h12, h21)):
#             rect2i = self.crop.get_rect(image2i)
#
#             image2[i] = F.crop(image2i, *rect2i)
#
#             h12[i] = crop_homography(h12i, rect1, rect2i)
#             h21[i] = crop_homography(h21i, rect2i, rect1)
#
#         item[du.IMAGE1] = F.crop(image1, *rect1)
#         item[du.IMAGE2] = image2
#
#         item[du.H12] = h12
#         item[du.H21] = h21
#
#         return item
#
#
# class MegaDepthHAGenHomography:
#
#     def __init__(self, num_samples, perspective=True, scaling=True, rotation=True, translation=True,
#                  n_scales=5, n_angles=5, scaling_amplitude=0.2,
#                  perspective_amplitude_x=0.3, perspective_amplitude_y=0.2,
#                  patch_ratio=0.85, max_angle=math.pi / 16):
#         self.num_samples = num_samples
#
#         self.perspective = perspective
#         self.scaling = scaling
#         self.rotation = rotation
#         self.translation = translation
#         self.n_scales = n_scales
#         self.n_angles = n_angles
#         self.scaling_amplitude = scaling_amplitude
#         self.perspective_amplitude_x = perspective_amplitude_x
#         self.perspective_amplitude_y = perspective_amplitude_y
#         self.patch_ratio = patch_ratio
#         self.max_angle = max_angle
#
#     def __call__(self, item):
#         image1 = item[du.IMAGE1]
#         shape = image1.shape[1::-1]
#
#         h12 = np.zeros((self.num_samples, 3, 3))
#         h21 = np.zeros((self.num_samples, 3, 3))
#
#         image2 = np.zeros((self.num_samples,) + image1.shape, dtype=np.uint8)
#
#         for i in range(self.num_samples):
#             h12[i] = np.asmatrix(sample_homography(shape, self.perspective, self.scaling, self.rotation, self.translation,
#                                                    self.n_scales, self.n_angles, self.scaling_amplitude,
#                                                    self.perspective_amplitude_x, self.perspective_amplitude_y,
#                                                    self.patch_ratio, self.max_angle))
#             h21[i] = np.linalg.inv(h12[i])
#
#             image2[i] = cv2.warpPerspective(image1, h12[i], shape)
#
#         item[du.H12] = h12
#         item[du.H21] = h21
#
#         item[du.IMAGE2] = image2
#
#         return item
# if du.EXT_PRIOR_SCORE1 in item:
#     item[du.EXT_PRIOR_SCORE1] = F.to_pil_image(item[du.EXT_PRIOR_SCORE1])
#     item[du.EXT_PRIOR_SCORE2] = F.to_pil_image(item[du.EXT_PRIOR_SCORE2])
#
#
#     item[du.C_IMAGE2] = F.to_tensor(item[du.C_IMAGE2])
#
# if du.SIFT_KP1 in item:
#     raise NotImplementedError
#
# if du.EXT_PRIOR_SCORE1 in item:
#     item[du.EXT_PRIOR_SCORE1] = F.to_tensor(item[du.EXT_PRIOR_SCORE1])
#     item[du.EXT_PRIOR_SCORE2] = F.to_tensor(item[du.EXT_PRIOR_SCORE2])
# if du.SIFT_KP1 in item:
#     item[du.SIFT_KP1] = item[du.SIFT_KP1] - np.array(rect1[:2]).reshape(1, -1)
#     item[du.SIFT_KP2] = item[du.SIFT_KP2] - np.array(rect2[:2]).reshape(1, -1)
#
# if du.EXT_PRIOR_SCORE1 in item and isinstance(self.crop, du.RandomCrop):
#     item[du.EXT_PRIOR_SCORE1] = F.crop(item[du.EXT_PRIOR_SCORE1], *rect1)
#     item[du.EXT_PRIOR_SCORE2] = F.crop(item[du.EXT_PRIOR_SCORE2], *rect2)
# if du.SIFT_KP1 in item:
#     item[du.SIFT_KP1] = item[du.SIFT_KP1] * scale1
#     item[du.SIFT_KP2] = item[du.SIFT_KP2] * scale2

# if du.VIS_SCORE1 in item:
#     item[du.VIS_SCORE1] = F.to_pil_image(item[du.VIS_SCORE1].astype(np.int32), mode='I')

# if du.VIS_SCORE1 in item:
#     item[du.VIS_SCORE1] = torch.from_numpy(np.array(item[du.VIS_SCORE1]).astype(np.int32, copy=False)).unsqueeze(0)

# if du.VIS_SCORE1 in item and not isinstance(self.crop, MegaDepthSharedAreaCrop):
#     item[du.VIS_SCORE1] = F.crop(item[du.VIS_SCORE1], *rect1)