import os
import h5py
import numpy as np
import pandas as pd

import source.datasets.base.utils as du

from source.datasets.base.dataset import ImageDepthCalibFeaturesAnnotationsDataset, OPTIONS_CONFIG


class IMCPTDataset(ImageDepthCalibFeaturesAnnotationsDataset):

    @staticmethod
    def from_config(dataset_config, backend, item_transforms):
        root_path = os.path.split(dataset_config.csv_path)[0]
        annotations = pd.read_csv(dataset_config.csv_path, index_col=[0])

        return IMCPTDataset(root_path,
                            annotations,
                            dataset_config.data_options, dataset_config.get(OPTIONS_CONFIG),
                            backend,
                            item_transforms)

    def _init_item(self, row):
        item = super()._init_item(row)

        item[du.SCENE_NAME] = row[du.SCENE_NAME]

        if du.BAG_ID in row.index:
            item[du.BAG_ID] = row[du.BAG_ID]

        return item

    def _load_depth(self, depth_path):
        with h5py.File(depth_path, 'r') as file:
            depth = np.array(file['/depth']).astype(np.float32)
            return depth

    def _load_calib(self, calib_path):
        with h5py.File(calib_path, 'r') as file:
            R, t = np.array(file['R']), np.array(file['T'])
            extrinsics = np.zeros((4, 4))
            extrinsics[:3, :3] = R
            extrinsics[:3, 3] = t
            extrinsics[3, 3] = 1

            intrinsics = np.array(file['K'])

            return extrinsics, intrinsics
