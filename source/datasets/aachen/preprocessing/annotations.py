import os
import shutil
import pandas as pd
import numpy as np

import source.datasets.base.utils as du

from source.utils.common_utils import qvec2rotmat


V1_0 = '1.0'
V1_1 = '1.1'

REL_IMAGES_PATH = 'images/images_upright'


def create_test_pairs_annotations(dataset_path, version):
    match_pairs_path = get_matches_list_file_path(dataset_path, version)

    with open(match_pairs_path) as file:
        matches = [line.rstrip() for line in file]

    annotations = pd.DataFrame(columns=[du.IMAGE1, du.IMAGE2])

    for m in matches:
        img_path1, img_path2 = m.split(' ')

        annotations = annotations.append({du.IMAGE1: os.path.join(REL_IMAGES_PATH, img_path1),
                                          du.IMAGE2: os.path.join(REL_IMAGES_PATH, img_path2)}, ignore_index=True)

    annotations.to_csv(os.path.join(dataset_path, f'test_pairs_{version}.csv'))


def create_val_pairs(dataset_path, num_samples=2500):
    image_names = [i[0] for i in get_images_iterator(get_extrinsics_path(dataset_path, V1_0))]

    test_pairs = pd.read_csv(os.path.join(dataset_path, 'test_pairs_v1_0.csv'), index_col=[0])
    test_pairs[du.CALIB1] = test_pairs[du.IMAGE1].apply(lambda x: os.path.relpath(x, REL_IMAGES_PATH))
    test_pairs[du.CALIB2] = test_pairs[du.IMAGE2].apply(lambda x: os.path.relpath(x, REL_IMAGES_PATH))

    in_mask = test_pairs[du.CALIB1].isin(image_names) & test_pairs[du.CALIB2].isin(image_names)
    val_pairs = test_pairs[in_mask].sample(n=num_samples)

    val_pairs[du.CALIB1] = val_pairs[du.CALIB1].apply(lambda x: os.path.join('calib', os.path.splitext(x)[0] + '.npy'))
    val_pairs[du.CALIB2] = val_pairs[du.CALIB2].apply(lambda x: os.path.join('calib', os.path.splitext(x)[0] + '.npy'))

    val_pairs.to_csv(os.path.join(dataset_path, f'val_pairs.csv'))


def create_calibration(dataset_root, version):
    cameras = {}

    cameras_iterator = get_cameras_iterator(get_intrinsics_path(dataset_root, version))

    for i in cameras_iterator:
        image_name, _, _, _, params = i

        K = np.array([[params[0], 0, params[1]],
                      [0, params[0], params[2]],
                      [0, 0, 1]])

        cameras[image_name] = K

    calibration_dir_path = os.path.join(dataset_root, 'calib')

    if os.path.exists(calibration_dir_path):
        shutil.rmtree(calibration_dir_path)

    os.mkdir(calibration_dir_path)

    images_iterator = get_images_iterator(get_extrinsics_path(dataset_root, version))

    for i in images_iterator:
        image_name, qvec, tvec = i

        T = np.zeros((4, 4))
        T[:3, :3] = qvec2rotmat(qvec)
        T[:3, 3] = tvec
        T[3, 3] = 1

        head_path, tail_path = os.path.split(image_name)

        calibration_dir_pathi = os.path.join(calibration_dir_path, head_path)

        if not os.path.exists(calibration_dir_pathi):
            os.makedirs(calibration_dir_pathi)

        calibration_pathi = os.path.join(calibration_dir_pathi, os.path.splitext(tail_path)[0])

        np.save(calibration_pathi, {du.EXTRINSICS: T,
                                    du.INTRINSICS: cameras[image_name]})


"""
Support utils
"""


def get_cameras_iterator(intrinsics_path):
    with open(intrinsics_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            line_items = line.strip('\n').split(' ')

            image_name = line_items[0]
            camera_model_name = line_items[1]
            width = int(line_items[2])
            height = int(line_items[3])
            params = np.array(line_items[4:], dtype=np.float32)

            yield image_name, camera_model_name, width, height, params


def get_images_iterator(extrinsics_path):
    with open(extrinsics_path, "r") as f:
        lines = f.readlines()

        num_cameras = int(lines[2])
        lines = lines[3: 3 + num_cameras]

        for line in lines:
            line_items = line.strip('\n').split(' ')

            image_name = line_items[0]
            qw, qx, qy, qz, cx, cy, cz = [float(param) for param in line_items[2: -2]]

            qvec = np.array([qw, qx, qy, qz])
            c = np.array([cx, cy, cz])
            tvec = -np.matmul(qvec2rotmat(qvec), c)

            yield image_name, qvec, tvec


def get_intrinsics_path(dataset_path, version):
    if version == V1_0:
        return os.path.join(dataset_path, '3D-models', 'aachen_v1_0/database_intrinsics.txt')

    elif version == V1_1:
        return os.path.join(dataset_path, '3D-models', 'aachen_v1_1/database_intrinsics.txt')

    else:
        raise ValueError(version)


def get_extrinsics_path(dataset_path, version):
    if version == V1_0:
        return os.path.join(dataset_path, '3D-models', 'aachen_v1_0/aachen_cvpr2018_db.nvm')

    elif version == V1_1:
        return os.path.join(dataset_path, '3D-models', 'aachen_v1_1/aachen.nvm')

    else:
        raise ValueError(version)


def get_matches_list_file_path(dataset_path, version):
    if version == V1_0:
        return os.path.join(dataset_path, 'image_pairs_to_match_v1_0.txt')

    elif version == V1_1:
        return os.path.join(dataset_path, 'image_pairs_to_match_v1_1.txt')

    else:
        raise ValueError(version)
