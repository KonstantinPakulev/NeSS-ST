import os
import numpy as np
import pandas as pd
from skimage import io

import source.datasets.base.utils as du

from source.datasets.base.dataset import ImageHCalibAnnotationsDataset, OPTIONS_CONFIG


class HPatchesDataset(ImageHCalibAnnotationsDataset):

    @staticmethod
    def from_config(dataset_config, backend, item_transforms):
        root_path = os.path.split(dataset_config.csv_path)[0]
        annotations = pd.read_csv(dataset_config.csv_path, index_col=[0])

        return HPatchesDataset(root_path,
                               annotations,
                               dataset_config.data_options, dataset_config.get(OPTIONS_CONFIG),
                               backend,
                               item_transforms)

    def _init_item(self, row):
        item = super()._init_item(row)

        item[du.SCENE_NAME] = row[du.SCENE_NAME]

        return item

    def _load_h(self, h_path, i):
        if i == 1:
            return np.asmatrix(np.loadtxt(h_path)).astype(np.float)

        else:
            return np.asmatrix(np.loadtxt(h_path)).astype(np.float).I
