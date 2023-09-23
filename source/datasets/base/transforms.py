from abc import ABC, abstractmethod

import torch
import numpy as np
from numpy import random as random
from torchvision.transforms import functional as F

from source.datasets.base import utils as du

from source.core.model import get_num_branches
from source.utils import endpoint_utils as eu

GRAYSCALE = 'grayscale'
RESIZE = 'resize'
CROP = 'crop'

DIVISOR_CROP = 'divisor_crop'
RANDOM_CROP = 'random_crop'
CENTRAL_CROP = 'central_crop'

CROP_RECT = 'crop_rect'


class ImageDepthTFactory:

    def __init__(self, config, input_size_divisor):
        self.config = config
        self.input_size_divisor = input_size_divisor

    def _to_pil(self):
        return [ImageDepthToPILImageWrapper()]

    def _grayscale(self):
        return [ImageToGrayScaleWrapper()]

    def _resize(self, value):
        if du.HEIGHT in value and du.WIDTH in value:
            size = (value.height, value.width)

            if size != (-1, -1):
                return [ImageDepthResizeWrapper(AspectResize(size, False))]

        return []

    def _crop(self, value):
        if value.type == DIVISOR_CROP:
            return [ImageDepthCropWrapper(DivisorCrop(self.input_size_divisor))]

        elif value.type == RANDOM_CROP:
            return [ImageDepthCropWrapper(RandomCrop((value.height, value.width)))]

        elif value.type == CENTRAL_CROP:
            return [ImageDepthCropWrapper(CentralCrop((value.height, value.width), False))]

        else:
            return []

    def _to_tensor(self):
        return [ImageDepthToTensorWrapper()]

    def _prepare_transform(self, key, value):
        if key == GRAYSCALE:
            return self._grayscale()

        if RESIZE in key:
            return self._resize(value)

        if CROP in key:
            return self._crop(value)

        return []

    def create(self):
        item_transforms = []

        if self.config is not None:
            item_transforms += self._to_pil()

            for key, value in self.config.items():
                item_transforms += self._prepare_transform(key, value)

        item_transforms += self._to_tensor()

        return flatten(item_transforms)


class ImageDepthCalibFeaturesTFactory(ImageDepthTFactory):

    def _to_tensor(self):
        return super()._to_tensor() + [CalibToTensorWrapper()] + [FeaturesToTensorWrapper()]


class ImageHCalibTFactory(ImageDepthTFactory):

    def _to_tensor(self):
        return super()._to_tensor() + [HToTensorWrapper()] + [FeaturesToTensorWrapper()]


"""
Transforms
"""


class BaseTransformWrapper:

    def __call__(self, item):
        num_branches = get_num_branches(self._base_key(), item.keys())

        if num_branches != 0:
            bundle = {}

            for i in range(1, num_branches + 1):
                item, bundle = self._transform_branch(item, bundle, i)

            item = self._transform_branches(item, bundle)

        return item

    @abstractmethod
    def _base_key(self):
        ...

    def _transform_branch(self, item, bundle, i):
        return item, bundle

    def _transform_branches(self, item, bundle):
        return item


class ImageTransformWrapper(BaseTransformWrapper, ABC):

    def _base_key(self):
        return du.IMAGE


class ImageDepthToPILImageWrapper(ImageTransformWrapper):

    def _transform_branch(self, item, bundle, i):
        image_key = f"{du.IMAGE}{i}"
        depth_key = f"{du.DEPTH}{i}"

        item[image_key] = F.to_pil_image(item[image_key])

        if depth_key in item:
            item[depth_key] = F.to_pil_image(item[depth_key], mode='F')

        return item, bundle


class ImageToGrayScaleWrapper(ImageTransformWrapper):

    def _transform_branch(self, item, bundle, i):
        item[f"{du.IMAGE_GRAY}{i}"] = F.to_grayscale(item[f"{du.IMAGE}{i}"])

        return item, bundle


class ImageDepthCropWrapper(ImageTransformWrapper):

    def __init__(self, crop):
        self.crop = crop

    def _get_crop_reference(self, i):
        return f"{du.IMAGE}{i}"

    def _transform_branch(self, item, bundle, i):
        image_key = f"{du.IMAGE}{i}"
        image_gray_key = f"{du.IMAGE_GRAY}{i}"
        depth_key = f"{du.DEPTH}{i}"
        shift_scale_key = f"{du.SHIFT_SCALE}{i}"

        recti = self.crop.get_rect(item[self._get_crop_reference(i)])

        item[image_key] = F.crop(item[image_key], *recti)

        if image_gray_key in item:
            item[image_gray_key] = F.crop(item[image_gray_key], *recti)

        if depth_key in item:
            item[depth_key] = F.crop(item[depth_key], *recti)

        item = create_shift_scale_if_not_exists(item, i)
        item[shift_scale_key] = crop_shift_scale(item[shift_scale_key], recti)

        bundle[f"{CROP_RECT}{i}"] = recti

        return item, bundle


class ImageDepthResizeWrapper(ImageTransformWrapper):

    def __init__(self, resize):
        self.resize = resize

    def _transform_branch(self, item, bundle, i):
        image_key = f"{du.IMAGE}{i}"
        image_gray_key = f"{du.IMAGE_GRAY}{i}"
        depth_key = f"{du.DEPTH}{i}"
        shift_scale_key = f"{du.SHIFT_SCALE}{i}"

        sizei, scalei = self.resize.get_size_scale(item[image_key])

        item[image_key] = F.resize(item[image_key], sizei)

        if image_gray_key in item:
            item[image_gray_key] = F.resize(item[image_gray_key], sizei)

        if depth_key in item:
            item[depth_key] = F.resize(item[depth_key], sizei)

        item = create_shift_scale_if_not_exists(item, i)
        item[shift_scale_key] = resize_shift_scale(item[shift_scale_key], scalei)

        return item, bundle


class ImageDepthToTensorWrapper(ImageTransformWrapper):

    def _transform_branch(self, item, bundle, i):
        image_key = f"{du.IMAGE}{i}"
        image_gray_key = f"{du.IMAGE_GRAY}{i}"
        depth_key = f"{du.DEPTH}{i}"
        shift_scale_key = f"{du.SHIFT_SCALE}{i}"

        item[image_key] = F.to_tensor(item[image_key])

        if image_gray_key in item:
            item[image_gray_key] = F.to_tensor(item[image_gray_key])

        if depth_key in item:
            item[depth_key] = F.to_tensor(np.array(item[depth_key]))

        if shift_scale_key in item:
            item[shift_scale_key] = torch.from_numpy(item[shift_scale_key].astype(np.float32, copy=False))

        return item, bundle


class CalibToTensorWrapper(BaseTransformWrapper):

    def _base_key(self):
        return du.EXTRINSICS

    def _transform_branch(self, item, bundle, i):
        extrinsics_key = f"{du.EXTRINSICS}{i}"
        intrinsics_key = f"{du.INTRINSICS}{i}"
        shift_scale_key = f"{du.SHIFT_SCALE}{i}"

        item[extrinsics_key] = torch.from_numpy(item[extrinsics_key].astype(np.float32, copy=False))
        item[intrinsics_key] = torch.from_numpy(item[intrinsics_key].astype(np.float32, copy=False))

        if shift_scale_key not in item:
            item = create_shift_scale_if_not_exists(item, i)
            item[shift_scale_key] = torch.from_numpy(item[shift_scale_key].astype(np.float32, copy=False))

        return item, bundle


class HToTensorWrapper(BaseTransformWrapper):

    def _base_key(self):
        return du.H

    def _transform_branch(self, item, bundle, i):
        h_key = f"{du.H}{i}"
        shift_scale_key = f"{du.SHIFT_SCALE}{i}"

        item[h_key] = torch.from_numpy(item[h_key].astype(np.float32, copy=False))

        if shift_scale_key not in item:
            item = create_shift_scale_if_not_exists(item, i)
            item[shift_scale_key] = torch.from_numpy(item[shift_scale_key].astype(np.float32, copy=False))

        return item, bundle


class FeaturesToTensorWrapper(BaseTransformWrapper):

    def _base_key(self):
        return eu.KP

    def _transform_branch(self, item, bundle, i):
        kp_key = f"{eu.KP}{i}"
        kp_desc_key = f"{eu.KP_DESC}{i}"

        # TODO. Need keys from config?

        item[kp_key] = torch.from_numpy(item[kp_key].astype(np.float32, copy=False))
        item[kp_desc_key] = torch.from_numpy(item[kp_desc_key].astype(np.float32, copy=False))

        return item, bundle


"""
Crop transform units
"""


class CropBase(ABC):

    @abstractmethod
    def get_rect(self, image):
        ...


class CentralCrop(CropBase):

    def __init__(self, size, is_train):
        """
        :param size: (h, w)
        """
        self.size = size
        self.is_train = is_train

    def get_rect(self, image):
        if image.size[0] > image.size[1] or self.is_train:
            new_height = self.size[0]
            new_width = self.size[1]
        else:
            new_height = self.size[1]
            new_width = self.size[0]

        offset_h = int(round((image.size[1] - new_height) / 2.))
        offset_w = int(round((image.size[0] - new_width) / 2.))

        rect = (offset_h, offset_w, new_height, new_width)

        return rect


class RandomCrop(CropBase):

    def __init__(self, size):
        """
        :param size: (h, w)
        """
        self.size = size

    def get_rect(self, image):
        offset_h = random.randint(0, image.size[1] - self.size[0] + 1)
        offset_w = random.randint(0, image.size[0] - self.size[1] + 1)

        rect = (offset_h, offset_w, self.size[0], self.size[1])

        return rect


class DivisorCrop(CropBase):

    def __init__(self, size_divisor):
        self.size_divisor = size_divisor

    def get_rect(self, image):
        if image.size[1] % self.size_divisor != 0:
            new_height = (image.size[1] // self.size_divisor) * self.size_divisor
            offset_h = int(round((image.size[1] - new_height) / 2.))
        else:
            offset_h = 0
            new_height = image.size[1]

        if image.size[0] % self.size_divisor != 0:
            new_width = (image.size[0] // self.size_divisor) * self.size_divisor
            offset_w = int(round((image.size[0] - new_width) / 2.))
        else:
            offset_w = 0
            new_width = image.size[0]

        rect = (offset_h, offset_w, new_height, new_width)

        return rect


"""
Resize transform units
"""


class ResizeBase(ABC):

    @abstractmethod
    def get_size_scale(self, image):
        """
        :param image: PILImage. Note that it's size argument returns (w, h)
        """
        ...


class Resize(ResizeBase):

    def __init__(self, size):
        """
        :param size: (h, w)
        """
        self.size = size

    def get_size_scale(self, image):
        scale = np.array((self.size[0] / image.size[1], self.size[1] / image.size[0]))

        return self.size, scale


class FactorResize(ResizeBase):

    def __init__(self, resize_factor):
        self.resize_factor = resize_factor

    def get_size_scale(self, image):
        size = (np.array(image.size[::-1]) * self.resize_factor).astype(np.int).tolist()

        return size, self.resize_factor


class AspectResize(ResizeBase):

    def __init__(self, size, is_train):
        """
        :param size: (h, w)
        """
        self.size = size
        self.is_train = is_train

    def get_size_scale(self, image):
        if self.is_train:
            # During training (and validation) images in a batch can only have horizontal orientation, so
            # if orientations do not align we transform it in a way to capture more vertical area of the input image
            if image.size[0] > image.size[1]:
                new_ar = float(self.size[1]) / float(self.size[0])
                ar = float(image.size[0]) / float(image.size[1])

                if ar >= new_ar:
                    new_size = min(self.size)
                    scale_factor = np.array(new_size / image.size[1])

                else:
                    scale_factor = np.array(self.size[1] / image.size[0])
                    new_size = (int(image.size[1] * scale_factor), self.size[1])

            else:
                new_size = max(self.size)
                scale_factor = np.array(new_size / min(image.size))

        else:
            # During test there is only one image in a batch, so the restriction above is removed
            new_size = self.size[0]
            scale_factor = np.array(self.size[0] / min(image.size))

        return new_size, scale_factor


"""
Support utils
"""


def create_shift_scale_if_not_exists(item, i):
    shift_scale_key = f"{du.SHIFT_SCALE}{i}"

    if shift_scale_key not in item:
        item[shift_scale_key] = np.array([0., 0., 1., 1.])

    return item


def crop_shift_scale(shift_scale, rect):
    shift_scale[:2] += np.array(rect[:2]) / shift_scale[2:]
    return shift_scale


def resize_shift_scale(shift_scale, scale_factor):
    shift_scale[2:] *= scale_factor
    return shift_scale


def flatten(l):
    f_l = []

    for i in l:
        if isinstance(i, list):
            f_l.extend(i)

        else:
            f_l.append(i)

    return f_l
