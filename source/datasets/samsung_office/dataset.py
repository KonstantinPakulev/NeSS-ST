import os
import numpy as np
from skimage import io
from scipy.spatial.transform import Rotation

from torch.utils.data import Dataset
from torchvision import transforms

import source.datasets.base.utils as du

REFERENCE_TRAJ = 'reference_traj'
INTRINSICS_FILE = 'camera_params.npy'


class SamsungOfficeDataset(Dataset):

    @staticmethod
    def from_config(dataset_config, item_transforms):
        associations = read_associations(os.path.join(dataset_config.root, dataset_config.associations))

        reference_traj = None

        if REFERENCE_TRAJ in dataset_config:
            reference_traj = read_reference_traj(os.path.join(dataset_config.root, dataset_config.reference_traj))

        return SamsungOfficeDataset(dataset_config.root,
                                    associations, reference_traj,
                                    transforms.Compose(item_transforms))

    def __init__(self, root,
                 associations, reference_traj,
                 item_transforms=None):
        self.root = root
        self.associations = associations
        self.reference_traj = reference_traj
        self.item_transforms = item_transforms

    def __getitem__(self, index):
        row = self.associations[index]

        item = {}

        for i in range(1, len(row) // 2 + 1):
            image_path, depth_path = row[(i - 1) * 2], row[(i - 1) * 2 + 1]
            image_name = os.path.basename(image_path)

            item[f"{du.IMAGE_NAME}{i}"] = image_name
            item[f"{du.IMAGE}{i}"] = io.imread(os.path.join(self.root, image_path))
            item[f"{du.DEPTH}{i}"] = load_depth(os.path.join(self.root, depth_path))

            if self.reference_traj is not None:
                item[f"{du.POSE}{i}"] = get_pose_by_image_name(image_name, self.reference_traj)

        item[du.INTRINSICS] = load_intrinsics(os.path.join(self.root, INTRINSICS_FILE))

        if self.item_transforms is not None:
            item = self.item_transforms(item)

        return item

    def __len__(self):
        return len(self.associations)


"""
Support utils
"""


def read_associations(path):
    with open(path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    associations = []

    for i in lines:
        associations.append(i.split(' ')[1::2])

    return associations


def load_depth(path):
    depth = io.imread(path) / 5000
    return depth.astype(np.float32)


def load_intrinsics(path):
    return np.load(path).astype(np.float32)


def read_reference_traj(traj_path):
    rows = []

    with open(traj_path, 'r') as file:
        for line in file:
            rows.append([float(i) for i in line.strip().split(' ')])

    reference_traj = []

    for i, rowi in enumerate(rows):
        t = rowi[1:4]
        R = Rotation.from_quat(rowi[4:8])

        Ti = np.eye(4)
        Ti[:3, :3] = R.as_matrix()
        Ti[:3, 3] = t

        reference_traj.append((str(rowi[0]), Ti))

    return reference_traj


def get_pose_by_image_name(image_name, reference_traj):
    for i in reference_traj:
        if i[0] in image_name:
            return i[1]

    return None