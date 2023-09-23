import os
import pandas as pd
import numpy as np

import source.datasets.base.utils as du

from source.datasets.base.dataset import ImageDepthCalibFeaturesAnnotationsDataset, OPTIONS_CONFIG


class AachenDataset(ImageDepthCalibFeaturesAnnotationsDataset):

    @staticmethod
    def from_config(dataset_config, backend, item_transforms):
        root_path = os.path.split(dataset_config.csv_path)[0]
        annotations = pd.read_csv(dataset_config.csv_path, index_col=[0])

        return AachenDataset(root_path,
                             annotations,
                             dataset_config.data_options, dataset_config.get(OPTIONS_CONFIG),
                             backend,
                             item_transforms)

    def _load_depth(self, depth_path):
        raise NotImplementedException()

    def _load_calib(self, calib_path):
        with open(calib_path, 'rb') as file:
            calib = np.load(file, allow_pickle=True).item()

            return calib[du.EXTRINSICS], calib[du.INTRINSICS]
