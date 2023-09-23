import os
from skimage import io
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

import source.datasets.base.utils as du


class MRobLabDataset(Dataset):

    @staticmethod
    def from_config(dataset_config, item_transforms):
        return MRobLabDataset(dataset_config.root_path,
                              transforms.Compose(item_transforms))

    def __init__(self, root_path, item_transforms):
        self.root_path = root_path

        file_names = os.listdir(os.path.join(root_path, 'data/map/color_pr'))
        self.file_names = [os.path.splitext(i)[0] for i in file_names]
        self.file_names.sort()

        self.item_transforms = item_transforms

    def __getitem__(self, index):
        filename = self.file_names[index]

        image1 = io.imread(os.path.join(self.root_path, 'data/map/color_pr', f'{filename}.png'))
        depth1 = load_depth(os.path.join(self.root_path, 'data/map/depth_pr', f'{filename}.png'))
        image2 = io.imread(os.path.join(self.root_path, 'data/query', f'{filename}.png'))

        item = {du.IMAGE1: image1,
                du.DEPTH1: depth1,
                du.IMAGE2: image2}

        item = self.item_transforms(item)

        return item

    def __len__(self):
        return len(self.file_names)


"""
Support utils
"""


def load_depth(path):
    depth = io.imread(path) / 5000
    return depth.astype(np.float32)