import os
import h5py
import numpy as np
import pandas as pd

import source.datasets.base.utils as du

from source.datasets.base.dataset import ImageDepthCalibFeaturesAnnotationsDataset, OPTIONS_CONFIG


class MegaDepthDataset(ImageDepthCalibFeaturesAnnotationsDataset):

    @staticmethod
    def from_config(dataset_config, backend, item_transforms):
        root_path = os.path.split(os.path.split(dataset_config.csv_path)[0])[0]
        annotations = pd.read_csv(dataset_config.csv_path, index_col=[0])

        return MegaDepthDataset(root_path,
                                annotations,
                                dataset_config.data_options, dataset_config.get(OPTIONS_CONFIG),
                                backend,
                                item_transforms)

    def _init_item(self, row):
        item = super()._init_item(row)

        item[du.SCENE_NAME] = row[du.SCENE_NAME]

        return item

    def _load_depth(self, depth_path):
        with h5py.File(depth_path, 'r') as file:
            depth = np.array(file['/depth'])
            return depth

    def _load_calib(self, calib_path):
        with open(calib_path, 'rb') as file:
            calib = np.load(file, allow_pickle=True).item()

            return calib[du.EXTRINSICS], calib[du.INTRINSICS]


"""
Legacy code
"""

# def _process_data_options(self, row, item, i):
#     if du.HA_GT in self.data_options:
#         ha_kp_value, ha_kp, ha_kp_cov_eigv = self._load_ha_gt(item[du.SCENE_NAME],
#                                                               item[f"{du.IMAGE_NAME}{i}"])
#
#         item[f"{du.HA_KP_VALUE}{i}"] = ha_kp_value
#         item[f"{du.HA_KP}{i}"] = ha_kp
#         item[f"{du.HA_KP_COV_EIGV}{i}"] = ha_kp_cov_eigv

# def _load_ha_gt(self, scene_name, image_name):
#     file_name = os.path.splitext(image_name)[0]
#
#     ha_ground_truth_dir = os.path.join(self.root_path, 'Undistorted_SfM',
#                                        str.zfill(scene_name, 4), 'ha_ground_truth')
#     ha_ground_truth_path = os.path.join(ha_ground_truth_dir, f"{file_name}.npy")
#
#     with open(ha_ground_truth_path, 'rb') as file:
#         ha_ground_truth =  np.load(file, allow_pickle=True).item()
#
#         ha_kp_value = ha_ground_truth[du.HA_KP_VALUE]
#         ha_kp = ha_ground_truth[du.HA_KP]
#         ha_kp_cov_eigv = ha_ground_truth[du.HA_KP_COV_EIGV]
#
#         return ha_kp_value, ha_kp, ha_kp_cov_eigv


# if du.SHI_STATS in self.data_options:
#     shi_stats = load_shi_stats(image_path)
#
#     item[f"{du.SHI_KP}{i}"] = shi_stats[du.SHI_KP]
#     item[f"{du.SHI_KP_GRAD}{i}"] = shi_stats[du.SHI_KP_GRAD]
#     item[f"{du.SHI_KP_COUNT}{i}"] = shi_stats[du.SHI_KP_COUNT]

# from source.datasets.megadepth.memory_bank_utils import load_memory_bank_data

# class MegaDepthWarpDataset(Dataset):
#
#     @staticmethod
#     def from_config(dataset_config, item_transforms):
#         return MegaDepthWarpDataset(dataset_config[du.DATASET_ROOT],
#                                     dataset_config[du.CSV_WARP_PATH],
#                                     transforms.Compose(item_transforms),
#                                     dataset_config[duS])
#
#     def __init__(self, dataset_root, csv_path, item_transforms=None, sources=False):
#         self.dataset_root = dataset_root
#         self.annotations = pd.read_csv(csv_path, index_col=[0])
#         self.item_transforms = item_transforms
#         self.sources = sources
#
#     def __len__(self):
#         return len(self.annotations)
#
#     def __getitem__(self, index):
#         iloc = self.annotations.iloc[index]
#
#         image1_name = iloc[du.IMAGE1].split("/")[-1]
#         image2_name = image1_name + '_warp'
#
#         image1 = io.imread(iloc[du.IMAGE1])
#
#         item = {du.SCENE_NAME: iloc[du.SCENE_NAME],
#                 du.IMAGE1_NAME: image1_name,
#                 du.IMAGE2_NAME: image2_name,
#                 du.IMAGE1: image1}
#
#         if self.sources:
#             item[du.S_IMAGE1] = image1.copy()
#
#         if self.item_transforms is not None:
#             item = self.item_transforms(item)
#
#         return item

# class MegaDepthHADataset(Dataset):
#
#     @staticmethod
#     def from_config(dataset_config, item_transforms):
#         return MegaDepthHADataset(dataset_config[du.CSV_PATH],
#                                   item_transforms=transforms.Compose(item_transforms))
#
#     def __init__(self, csv_path, item_transforms=None):
#         self.annotations = pd.read_csv(csv_path, index_col=[0])
#         self.item_transforms = item_transforms
#
#     def __len__(self):
#         return len(self.annotations)
#
#     def __getitem__(self, index):
#         row = self.annotations.iloc[index]
#
#         scene_name = str(row[du.SCENE_NAME])
#
#         image1_name = row[du.IMAGE1].split("/")[-1]
#
#         id1 = str(row[du.ID1])
#
#         image1 = io.imread(row[du.IMAGE1])
#
#         item = {du.SCENE_NAME: scene_name,
#                 du.IMAGE_NAME1: image1_name,
#                 du.ID1: id1,
#                 du.IMAGE1: image1}
#
#         if self.item_transforms is not None:
#             item = self.item_transforms(item)
#
#         return item


# class MegaDepthSingleDataset(Dataset):
#
#     def __init__(self, csv_path, item_transforms=None):
#         self.annotations = pd.read_csv(csv_path, index_col=[0])
#         self.item_transforms = item_transforms
#
#     def __len__(self):
#         return len(self.annotations)
#
#     def __getitem__(self, index):
#         row = self.annotations.iloc[index]
#
#         index = row[du.INDEX]
#         scene_name = str(row[du.SCENE_NAME])
#         image_path1 = row[du.IMAGE1]
#         image_name1 = os.path.basename(image_path1)
#
#         image1 = io.imread(row[du.IMAGE1])
#
#         scene_data = load_scene_data(row[du.SCENE_DATA1])
#
#         depth1 = load_depth(row[du.DEPTH1])
#         extrinsics1 = scene_data[du.EXTRINSICS]
#         intrinsics1 = scene_data[du.INTRINSICS]
#         shift_scale1 = np.array([0., 0., 1., 1.])
#
#         item = {du.INDEX: index,
#                 du.SCENE_NAME: scene_name,
#                 du.IMAGE_NAME1: image_name1,
#                 du.IMAGE1: image1,
#                 du.DEPTH1: depth1,
#                 du.EXTRINSICS1: extrinsics1,
#                 du.INTRINSICS1: intrinsics1,
#                 du.SHIFT_SCALE1: shift_scale1}
#
#         if self.item_transforms is not None:
#             item = self.item_transforms(item)
#
#         return item


# class MegaDepthPairDataset(Dataset):
#
#     @staticmethod
#     def from_config(dataset_config, item_transforms):
#         return MegaDepthPairDataset(dataset_config.csv_path, item_transforms=transforms.Compose(item_transforms))
#
#     def __init__(self, csv_path, item_transforms=None):
#         self.annotations = pd.read_csv(csv_path, index_col=[0])
#         self.item_transforms = item_transforms
#
#     def __len__(self):
#         return len(self.annotations)
#
#     def __getitem__(self, index):
#
#
#         return item
