import os
import h5py
import numpy as np
from abc import ABC, abstractmethod
from skimage import io

from torch.utils.data import Dataset
from torchvision import transforms

import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu

from source.core.model import get_num_branches
from source.evaluation.logging import get_features_dir_path

OPTIONS_CONFIG = 'options_config'
INPUT_KEYS = 'input_keys'
NUM_FEATURES = 'num_features'


class ImageFeaturesAnnotationsDataset(Dataset, ABC):

    def __init__(self, root_path,
                 annotations,
                 data_options, options_config,
                 backend,
                 item_transforms):
        self.root_path = root_path
        self.annotations = annotations
        self.data_options = data_options
        self.options_config = options_config
        self.has_backend = backend is not None
        self.item_transforms = transforms.Compose(item_transforms)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]

        item = self._init_item(row)

        for i in range(1, get_num_branches(du.IMAGE, row.keys()) + 1):
            item = self._add_data_options_to_item(row, i, item)

        item = self.item_transforms(item)

        return item

    def _init_item(self, row):
        item = {}

        return item

    def _add_data_options_to_item(self, row, i, item):
        image_key = f"{du.IMAGE}{i}"
        image_path = os.path.join(self.root_path, row[image_key])

        item[f"{du.IMAGE_NAME}{i}"] = row[image_key]

        if du.IMAGE in self.data_options:
            item[image_key] = self._load_image(image_path)

        if du.FEATURES in self.data_options:
            image_name_no_ext = os.path.splitext(item[f"{du.IMAGE_NAME}{i}"].replace('/', '_'))[0]
            features_path = os.path.join(get_features_dir_path(self.has_backend), f"{image_name_no_ext}.h5py")

            input_keys = []
            num_features = None

            if self.options_config is not None:
                if du.FEATURES in self.options_config:
                    input_keys = self.options_config.features.get(INPUT_KEYS, [])
                    num_features = self.options_config.features.get(NUM_FEATURES)

            features = self._load_features(features_path, input_keys, num_features)

            for f_name in [eu.KP, eu.KP_DESC] + input_keys:
                item[f"{f_name}{i}"] = features[f_name]

        return item

    def __len__(self):
        return len(self.annotations)

    def _load_image(self, image_path):
        return io.imread(image_path)

    def _load_features(self, features_path, features_input_keys, num_features):
        with h5py.File(features_path, 'r') as file:
            features = {eu.KP: np.array(file[f'/{eu.KP}']),
                        eu.KP_DESC: np.array(file[f'/{eu.KP_DESC}'])}

            if num_features is not None:
                features[eu.KP] = features[eu.KP][:num_features]
                features[eu.KP_DESC] = features[eu.KP_DESC][:num_features]

            for fik in features_input_keys:
                features[fik] = np.array(file[f'/{fik}'])

            return features


class ImageDepthCalibFeaturesAnnotationsDataset(ImageFeaturesAnnotationsDataset, ABC):

    def _add_data_options_to_item(self, row, i, item):
        item = super()._add_data_options_to_item(row, i, item)

        if du.DEPTH in self.data_options:
            depth_key = f"{du.DEPTH}{i}"
            depth_path = os.path.join(self.root_path, row[depth_key])

            item[depth_key] = self._load_depth(depth_path)

        if du.CALIB in self.data_options:
            calib_key = f"{du.CALIB}{i}"
            calib_path = os.path.join(self.root_path, row[calib_key])

            extrinsics, intrinsics = self._load_calib(calib_path)

            item[f"{du.EXTRINSICS}{i}"] = extrinsics
            item[f"{du.INTRINSICS}{i}"] = intrinsics

        return item

    @abstractmethod
    def _load_depth(self, depth_path):
        ...

    @abstractmethod
    def _load_calib(self, calib_path):
        ...


class ImageHCalibAnnotationsDataset(ImageFeaturesAnnotationsDataset):

    def _add_data_options_to_item(self, row, i, item):
        item = super()._add_data_options_to_item(row, i, item)

        if du.H in self.data_options:
            h_key = f"{du.H}{i}"
            h_path = os.path.join(self.root_path, row[h_key])

            item[h_key] = self._load_h(h_path, i)

        return item

    @abstractmethod
    def _load_h(self, h_path, i):
        ...


"""
Legacy code
"""

# scene_name = str(row[du.SCENE_NAME])
# du.SCENE_NAME: scene_name