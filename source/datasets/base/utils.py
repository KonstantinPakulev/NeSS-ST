import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import torch

from torch.utils.data.sampler import Sampler

"""
Datasets
"""

HPATCHES = 'hpatches'
MEGADEPTH = 'megadepth'
AACHEN = 'aachen'
IMC_PT = 'imcpt'
SAMSUNG_OFFICE = 'samsung_office'
MROB_LAB = 'mrob_lab'
SCANNET = 'scannet'


"""
Config keys
"""
DATASET_ROOT = 'dataset_root'

ENTITY_ID = 'entity_id'

CSV_PATH = 'csv_path'

HEIGHT = 'height'
WIDTH = 'width'

PRE_FIT = 'pre_fit'
POST_FIT = 'post_fit'

NUM_SAMPLES = 'num_samples'
SHUFFLE = 'shuffle'
START_FROM = 'start_from'
NUM_WORKERS = 'num_workers'
SAMPLER = 'sampler'
COLLATE = 'collate'

SUBSET_SAMPLER = 'subset'
START_SEQ_SAMPLER = 'start_seq'


"""
Batch keys
"""
INDEX = 'index'

SCENE_NAME = 'scene_name'
BAG_ID = 'bag_id'
FRAME_INTERVAL = 'frame_interval'

IMAGE_NAME1 = 'image_name1'
IMAGE_NAME2 = 'image_name2'

IMAGE1 = 'image1'
IMAGE2 = 'image2'

IMAGE_GRAY1 = 'image_gray1'
IMAGE_GRAY2 = 'image_gray2'

VIS_MASK1 = 'vis_mask1'
VIS_MASK2 = 'vis_mask2'

DEPTH1 = 'depth1'
DEPTH2 = 'depth2'

EXTRINSICS1 = 'extrinsics1'
EXTRINSICS2 = 'extrinsics2'

INTRINSICS1 = 'intrinsics1'
INTRINSICS2 = 'intrinsics2'

IMAGE_SHAPE1 = 'image_shape1'
IMAGE_SHAPE2 = 'image_shape2'

SHIFT_SCALE1 = 'shift_scale1'
SHIFT_SCALE2 = 'shift_scale2'

H1 = 'h1'
H2 = 'h2'

POSE1 = 'pose1'
POSE2 = 'pose2'

HA_KP1 = 'ha_kp1'
HA_KP_VALUE1 = 'ha_kp_value1'
HA_KP_COV_EIGV1 = 'ha_kp_cov_eigv1'
HA_KP_MASK1 = 'ha_kp_mask1'


"""
Annotations and external data keys
"""

IMAGE = 'image'
IMAGE_GRAY = 'image_gray'
IMAGE_NAME = 'image_name'
DEPTH = 'depth'
SHIFT_SCALE = 'shift_scale'
CALIB = 'calib'
FEATURES = 'features'
H = 'h'

CALIB1 = 'calib1'
CALIB2 = 'calib2'

EXTRINSICS = 'extrinsics'
INTRINSICS = 'intrinsics'

IMAGE_SHAPE = 'image_shape'

SCALE_IDX = 'scale_idx'

POSE = 'pose'

HA_GT = 'ha_gt'

HA_KP = 'ha_kp'
HA_KP_VALUE = 'ha_kp_value'
HA_KP_COV_EIGV = 'ha_kp_cov_eigv'
HA_KP_MASK = 'ha_kp_mask'


"""
Data wrappers
"""


def instantiate_data_wrapper(batch, device):
    if H1 in batch:
        return HDataWrapper().init_from_batch(batch, device)

    else:
        return RBTDataWrapper().init_from_batch(batch, device)


class BaseDataWrapper(ABC):

    @abstractmethod
    def init_from_batch(self, batch, device):
        ...

    @abstractmethod
    def swap(self):
        ...


class RBTDataWrapper(BaseDataWrapper):

    def __init__(self):
        self.depth1 = None
        self.extrinsics1 = None
        self.intrinsics1 = None
        self.shift_scale1 = None

        self.depth2 = None
        self.extrinsics2 = None
        self.intrinsics2 = None
        self.shift_scale2 = None

    def init_from_batch(self, batch, device):
        if DEPTH1 in batch:
            self.depth1 = batch[DEPTH1].to(device)

        self.extrinsics1 = batch[EXTRINSICS1].to(device)
        self.intrinsics1 = batch[INTRINSICS1].to(device)
        self.shift_scale1 = batch[SHIFT_SCALE1].to(device)

        if DEPTH2 in batch:
            self.depth2 = batch[DEPTH2].to(device)

        if EXTRINSICS2 in batch:
            self.extrinsics2 = batch[EXTRINSICS2].to(device)
            self.intrinsics2 = batch[INTRINSICS2].to(device)
            self.shift_scale2 = batch[SHIFT_SCALE2].to(device)

        return self

    def init_from_batches(self, batch1, batch2, device):
        if DEPTH1 in batch1:
            self.depth1 = batch1[DEPTH1].to(device)

        if EXTRINSICS1 in batch1:
            self.extrinsics1 = batch1[EXTRINSICS1].to(device)
            self.intrinsics1 = batch1[INTRINSICS1].to(device)
            self.shift_scale1 = batch1[SHIFT_SCALE1].to(device)

        if DEPTH1 in batch2:
            self.depth2 = batch2[DEPTH1].to(device)

        if EXTRINSICS1 in batch2:
            self.extrinsics2 = batch2[EXTRINSICS1].to(device)
            self.intrinsics2 = batch2[INTRINSICS1].to(device)
            self.shift_scale2 = batch2[SHIFT_SCALE1].to(device)

        return self

    def swap(self):
        if self.extrinsics2 is None:
            raise ValueError("extrinsics2 cannot be None")

        batch = {DEPTH1: self.depth2,
                 DEPTH2: self.depth1,

                 EXTRINSICS1: self.extrinsics2,
                 EXTRINSICS2: self.extrinsics1,

                 INTRINSICS1: self.intrinsics2,
                 INTRINSICS2: self.intrinsics1,

                 SHIFT_SCALE1: self.shift_scale2,
                 SHIFT_SCALE2: self.shift_scale1}

        return RBTDataWrapper().init_from_batch(batch, self.depth1.device)


class HDataWrapper(BaseDataWrapper):

    def __init__(self, image_shape1=None, shift_scale=None, h1=None):
        self.image_shape1 = image_shape1
        self.shift_scale1 = shift_scale
        self.h1 = h1

        self.image_shape2 = None
        self.shift_scale2 = None
        self.h2 = None

    def init_from_batch(self, batch, device):
        if IMAGE1 in batch:
            self.image_shape1 = batch[IMAGE1].shape
        else:
            self.image_shape1 = batch[IMAGE_SHAPE1]

        self.shift_scale1 = batch[SHIFT_SCALE1].to(device)
        self.h1 = batch[H1].to(device)

        if IMAGE2 in batch:
            self.image_shape2 = batch[IMAGE2].shape
        else:
            self.image_shape2 = batch[IMAGE_SHAPE2]

        self.shift_scale2 = batch[SHIFT_SCALE2].to(device)
        self.h2 = batch[H2].to(device)

        return self

    def swap(self):
        batch = {IMAGE_SHAPE1: self.image_shape2,
                 IMAGE_SHAPE2: self.image_shape1,

                 H1: self.h2,
                 H2: self.h1,

                 SHIFT_SCALE1: self.shift_scale2,
                 SHIFT_SCALE2: self.shift_scale1}

        return HDataWrapper().init_from_batch(batch, self.h1.device)


"""
Samplers
"""


# TODO. Replace with standard torch implementation
class SubsetSampler(Sampler):

    def __init__(self, data_source, num_samples, shuffle):
        super().__init__(data_source)
        self.data_source = data_source

        self.num_samples = num_samples
        self.shuffle = shuffle

    def __iter__(self):
        indices = torch.arange(len(self.data_source)).tolist()

        if self.shuffle:
            np.random.shuffle(indices)

        if self.num_samples != -1:
            indices = indices[:self.num_samples]

        return iter(indices)

    def __len__(self):
        if self.num_samples == -1:
            return len(self.data_source)

        else:
            return self.num_samples


class StartSeqSampler(Sampler):

    def __init__(self, data_source, num_samples, start_from):
        super().__init__(data_source)
        self.data_source = data_source

        self.num_samples = num_samples
        self.start_from = start_from

    def __iter__(self):
        indices = torch.arange(len(self.data_source)).tolist()[self.start_from:]

        if self.num_samples != -1:
            indices = indices[:self.num_samples]

        return iter(indices)

    def __len__(self):
        if self.num_samples == -1:
            return len(self.data_source) - self.start_from

        else:
            return min(self.num_samples, len(self.data_source) - self.start_from)


"""
Annotations utils
"""


def from_pairs_annotations_csv(dataset_path, pairs_filename, filename):
    ann_pairs = pd.read_csv(os.path.join(dataset_path, pairs_filename), index_col=[0])

    ann = from_pairs_annotations(ann_pairs)
    ann.to_csv(os.path.join(dataset_path, filename))


def from_pairs_annotations(ann_pairs):
    columns = []
    rename_columns = {}

    if SCENE_NAME in ann_pairs.columns:
        columns.append(ann_pairs[SCENE_NAME].append(ann_pairs[SCENE_NAME]).reset_index(drop=True))
        rename_columns[len(rename_columns)] = SCENE_NAME

    if FRAME_INTERVAL in ann_pairs.columns:
        columns.append(ann_pairs[FRAME_INTERVAL].append(ann_pairs[FRAME_INTERVAL]).reset_index(drop=True))
        rename_columns[len(rename_columns)] = FRAME_INTERVAL

    columns.append(pd.Series(mix_lists(ann_pairs[IMAGE1], ann_pairs[IMAGE2])))
    rename_columns[len(rename_columns)] = IMAGE1

    if DEPTH1 in ann_pairs.columns:
        columns.append(pd.Series(mix_lists(ann_pairs[DEPTH1], ann_pairs[DEPTH2])))
        rename_columns[len(rename_columns)] = DEPTH1

    if CALIB1 in ann_pairs.columns:
        columns.append(pd.Series(mix_lists(ann_pairs[CALIB1], ann_pairs[CALIB2])))
        rename_columns[len(rename_columns)] = CALIB1

    ann = pd.concat(columns, ignore_index=True, axis=1).\
        rename(columns=rename_columns).\
        drop_duplicates(subset=IMAGE1).\
        reset_index(drop=True)

    return ann


def mix_lists(a, b):
    return [i for p in zip(a, b) for i in p]


"""
Legacy code
"""

# self.depth1 = batch[DEPTH1].to(device)
# self.extrinsics1 = batch[EXTRINSICS1].to(device)
# self.intrinsics1 = batch[INTRINSICS1].to(device)
# self.shift_scale1 = batch[SHIFT_SCALE1].to(device)
#
# if DEPTH2 in batch:
#     self.depth2 = batch[DEPTH2].to(device)
#     self.extrinsics2 = batch[EXTRINSICS2].to(device)
#     self.intrinsics2 = batch[INTRINSICS2].to(device)
#     self.shift_scale2 = batch[SHIFT_SCALE2].to(device)

# class SceneDataWrapper:
#
#     def join(self):
#         if self.depth2 is not None:
#             depth = torch.cat([self.depth1, self.depth2], dim=0)
#             extrinsics = torch.cat([self.extrinsics1, self.extrinsics2], dim=0)
#             intrinsics = torch.cat([self.intrinsics1, self.intrinsics2], dim=0)
#             shift_scale = torch.cat([self.shift_scale1, self.shift_scale2], dim=0)
#
#             return JSceneDataWrapper(depth, extrinsics, intrinsics, shift_scale)
#
#         else:
#             return JSceneDataWrapper(self.depth1, self.extrinsics1, self.intrinsics1, self.shift_scale1)
#
#
# class JSceneDataWrapper:
#
#     def __init__(self, depth, extrinsics, intrinsics, shift_scale):
#         self.depth = depth
#         self.extrinsics = extrinsics
#         self.intrinsics = intrinsics
#         self.shift_scale = shift_scale

# def __init__(self, depth1, extrinsics1, intrinsics1, shift_scale1,
#              depth2=None, extrinsics2=None, intrinsics2=None, shift_scale2=None):
#     self.depth1 = depth1
#     self.extrinsics1 = extrinsics1
#     self.intrinsics1 = intrinsics1
#     self.shift_scale1 = shift_scale1
#
#     self.depth2 = depth2
#     self.extrinsics2 = extrinsics2
#     self.intrinsics2 = intrinsics2
#     self.shift_scale2 = shift_scale2
#
# def unsqueeze(self):
#     if self.depth2 is None:
#         raise NotImplementedError
#
#     batch = {DEPTH1: self.depth1.unsqueeze(0),
#              DEPTH2: self.depth2.unsqueeze(0),
#
#              EXTRINSICS1: self.extrinsics1.unsqueeze(0),
#              EXTRINSICS2: self.extrinsics2.unsqueeze(0),
#
#              INTRINSICS1: self.intrinsics1.unsqueeze(0),
#              INTRINSICS2: self.intrinsics2.unsqueeze(0),
#
#              SHIFT_SCALE1: self.shift_scale1.unsqueeze(0),
#              SHIFT_SCALE2: self.shift_scale2.unsqueeze(0)}
#
#     return SceneDataWrapper.from_batch(batch, self.depth1.device)

# @staticmethod
# def from_batches(batch1, batch2, device):
#     depth1 = batch1[DEPTH1].to(device)
#     extrinsics1 = batch1[EXTRINSICS1].to(device)
#     intrinsics1 = batch1[INTRINSICS1].to(device)
#     shift_scale1 = batch1[SHIFT_SCALE1].to(device)
#
#     depth2 = batch2[DEPTH1].to(device)
#     extrinsics2 = batch2[EXTRINSICS1].to(device)
#     intrinsics2 = batch2[INTRINSICS1].to(device)
#     shift_scale2 = batch2[SHIFT_SCALE1].to(device)
#
#     return SceneDataWrapper(depth1, extrinsics1, intrinsics1, shift_scale1,
#                             depth2, extrinsics2, intrinsics2, shift_scale2)

# SCENE_SAMPLER = 'scene'
# BATCH_SIZE = 'batch_size'
# EXPERIMENT_ID = 'experiment_id'
# MEMORY_BANK_DATA = 'memory_bank_data'

# MEMORY_BANK_COLLATE = 'memory_bank'

# SCENE_INFO_ROOT = 'scene_info_root'

# class CompositeBatch:
#
#     def __init__(self, h, r3, device):
#         self._h = h
#         self._r3 = r3
#
#         self.device = device
#
#         self.joint_index = len(self._h[IMAGE1]) if IMAGE1 in self._h else 0
#
#         self._is_homo = IMAGE1 in self._h
#         self._is_r3 = IMAGE1 in self._r3
#
#     @property
#     def is_h(self):
#         return self._is_homo
#
#     @property
#     def is_r3(self):
#         return self._is_r3
#
#     def get_homo(self, key):
#         return self._h[key].to(self.device)
#
#     def get_r3(self, key):
#         return self._r3[key].to(self.device)
#
#     def get(self, key):
#         joint_tensor = self.join(self._h.get(key), self._r3.get(key))
#         return joint_tensor.to(self.device) \
#             if joint_tensor is not None and not isinstance(joint_tensor, list) else joint_tensor
#
#     def split_h(self, tensor):
#         return tensor[:self.joint_index]
#
#     def split_r3(self, tensor):
#         return tensor[self.joint_index:]
#
#     def join(self, tensor1, tensor2):
#         joint_tensor = None
#
#         if self.is_h:
#             joint_tensor = tensor1
#
#         if self.is_r3:
#             if joint_tensor is not None:
#                 joint_tensor = torch.cat([joint_tensor, tensor2])
#             else:
#                 joint_tensor = tensor2
#
#         return joint_tensor


# class TwoDatasetsCollate:
#
#     def __init__(self, device):
#         self.device = device
#
#     def __call__(self, batch):
#         batch_homo = []
#         batch_r3 = []
#
#         for elem in batch:
#             (batch_homo if H12 in elem.keys() else batch_r3).append(elem)
#
#         t_batch_homo = default_collate(batch_homo) if len(batch_homo) != 0 else {}
#         t_batch_r3 = default_collate(batch_r3) if len(batch_r3) != 0 else {}
#
#         return CompositeBatch(t_batch_homo, t_batch_r3, self.device)
# class ColorJitter(object):
#
#     def __init__(self, brightness=0.1, contrast=0.1):
#         self.brightness = brightness
#         self.contrast = contrast
#
#     def __call__(self, item):
#         brightness_factor = random.uniform(max(1 - self.brightness, 0), 1 + self.brightness)
#         contrast_factor = random.uniform(max(1 - self.contrast, 0), 1 + self.contrast)
#
#         transforms = [Lambda(lambda image: F.adjust_brightness(image, brightness_factor)),
#                       Lambda(lambda image: F.adjust_contrast(image, contrast_factor))]
#         random.shuffle(transforms)
#         transforms = Compose(transforms)
#
#         item[d.IMAGE1] = transforms(item[d.IMAGE1])
#         item[d.IMAGE2] = transforms(item[d.IMAGE2])
#
#         return item
#
#
# class Normalize(object):
#
#     def __init__(self, mean, std):
#         self.mean = np.array(mean)
#         self.std = np.array(std)
#
#     def __call__(self, item):
#         item[d.IMAGE1] = item[d.IMAGE1] / 255.0
#         item[d.IMAGE2] = item[d.IMAGE2] / 255.0
#
#         item[d.IMAGE1] = (item[d.IMAGE1] - self.mean.reshape([1, 1, 3])) / self.std.reshape([1, 1, 3])
#         item[d.IMAGE2] = (item[d.IMAGE2] - self.mean.reshape([1, 1, 3])) / self.std.reshape([1, 1, 3])
#
#         return item
# from torchvision.transforms.transforms import Lambda, Compose

# MEGADEPTH_HA = 'megadepth_ha'

# CSV_WARP_PATH = 'csv_warp_path'
# TO_GRAYSCALE = 'to_grayscale'
# RESIZE = 'resize'

# SIFT_KP1 = 'sift_kp1'
# SIFT_KP2 = 'sift_kp2'

# INDEX = 'index'
