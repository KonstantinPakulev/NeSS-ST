import os
import pandas as pd
import numpy as np

import source.datasets.base.utils as du

from source.datasets.base.dataset import ImageDepthCalibFeaturesAnnotationsDataset, OPTIONS_CONFIG


class ScanNetDataset(ImageDepthCalibFeaturesAnnotationsDataset):
    
    @staticmethod
    def from_config(dataset_config, backend, item_transforms):
        root_path = os.path.split(dataset_config.csv_path)[0]
        annotations = pd.read_csv(dataset_config.csv_path, index_col=[0])

        return ScanNetDataset(root_path,
                              annotations,
                              dataset_config.data_options, dataset_config.get(OPTIONS_CONFIG),
                              backend,
                              item_transforms)

    def _init_item(self, row):
        item = super()._init_item(row)

        item[du.FRAME_INTERVAL] = str(row[du.FRAME_INTERVAL])

        return item

    def _load_depth(self, depth_path):
        raise NotImplementedError

    def _load_calib(self, extrinsics_path):
        with open(extrinsics_path) as f:
            extrinsics = [i.strip().split(' ') for i in f.readlines()]
            extrinsics = np.linalg.inv(np.array(extrinsics, dtype=np.float32))

        intrinsics_path = os.path.join(os.path.split(os.path.split(extrinsics_path)[0])[0],
                                       'intrinsic', 'intrinsic_color.txt')

        with open(intrinsics_path) as f:
            intrinsics = [i.strip().split(' ') for i in f.readlines()]
            intrinsics = np.array(intrinsics, dtype=np.float32)[:3, :3]

        return extrinsics, intrinsics
