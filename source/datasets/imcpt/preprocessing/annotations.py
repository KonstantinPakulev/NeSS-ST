import os
import itertools
import h5py
import numpy as np
import pandas as pd
import deepdish as dd

from numpy.random import choice

import source.datasets.base.utils as du
import source.utils.common_utils

from source.pose.estimators.colmap.utils import read_bin_images, read_bin_cameras

BAG_FILENAME = "{:d}bag_{:03d}.txt"


def create_val_pairs(root_path, scenes):
    annotations_dict = {du.SCENE_NAME: [],
                        du.IMAGE1: [],
                        du.IMAGE2: [],

                        du.DEPTH1: [],
                        du.DEPTH2: [],

                        du.CALIB1: [],
                        du.CALIB2: []}

    for scene in scenes:
        pairs = np.load(os.path.join(get_scene_path(root_path, scene), 'new-vis-pairs', 'keys-th-0.1.npy'))

        for p in pairs:
            p1, p2 = p.split('-')

            annotations_dict[du.SCENE_NAME].append(scene)

            annotations_dict[du.IMAGE1].append(os.path.join(scene, 'set_100', 'images', f"{p1}.jpg"))
            annotations_dict[du.IMAGE2].append(os.path.join(scene, 'set_100', 'images', f"{p2}.jpg"))

            annotations_dict[du.DEPTH1].append(os.path.join(scene, 'set_100', 'depth_maps', f"{p1}.h5"))
            annotations_dict[du.DEPTH2].append(os.path.join(scene, 'set_100', 'depth_maps', f"{p2}.h5"))

            annotations_dict[du.CALIB1].append(os.path.join(scene, 'set_100', 'calibration', f"calibration_{p1}.h5"))
            annotations_dict[du.CALIB2].append(os.path.join(scene, 'set_100', 'calibration', f"calibration_{p2}.h5"))

    annotations = pd.DataFrame(data=annotations_dict)
    annotations.to_csv(os.path.join(root_path, f'val_pairs.csv'))


def create_val_sfm_pairs(root_path, scenes, min_num_vis_kp=100):
    annotations_dict = {du.SCENE_NAME: [],
                        du.BAG_ID: [],

                        du.IMAGE1: [],
                        du.IMAGE2: [],

                        du.DEPTH1: [],
                        du.DEPTH2: [],

                        du.CALIB1: [],
                        du.CALIB2: []}

    for scene in scenes:
        scene_path = get_scene_path(root_path, scene)
        scene_bags_path = get_scene_bags_path(scene_path)
        scene_bag_pairs_path = get_scene_bag_pairs_path(scene_path)

        with open(os.path.join(scene_path, 'images.txt')) as file:
            image_paths = [line.rstrip() for line in file]

        with open(os.path.join(scene_path, 'visibility.txt')) as file:
            visibility_paths = [line.rstrip() for line in file]

        for bag_filename in [b for b in os.listdir(scene_bags_path) if not b.startswith('.')]:
            with open(os.path.join(scene_bags_path, bag_filename)) as file:
                bag_image_paths = [line.rstrip() for line in file]

            bag_image_idx = []

            for bip in bag_image_paths:
                for i, ip in enumerate(image_paths):
                    if bip == ip:
                        bag_image_idx.append(i)

            bag_image_vis = []

            for bii in bag_image_idx:
                bag_image_visi = np.loadtxt(os.path.join(scene_path, visibility_paths[bii])).\
                    flatten().astype('float32')[bag_image_idx]

                bag_image_vis.append(bag_image_visi)

            bag_image_pairs = []

            for p_idx1, p_idx2 in itertools.product(range(len(bag_image_paths)), range(len(bag_image_paths))):
                if p_idx1 != p_idx2:
                    if bag_image_vis[p_idx1][p_idx2] > min_num_vis_kp:
                        p1 = os.path.splitext(os.path.basename(bag_image_paths[p_idx1]))[0]
                        p2 = os.path.splitext(os.path.basename(bag_image_paths[p_idx2]))[0]

                        annotations_dict[du.SCENE_NAME].append(scene)
                        annotations_dict[du.BAG_ID].append(os.path.splitext(bag_filename)[0])

                        annotations_dict[du.IMAGE1].append(os.path.join(scene, 'set_100', 'images', f"{p1}.jpg"))
                        annotations_dict[du.IMAGE2].append(os.path.join(scene, 'set_100', 'images', f"{p2}.jpg"))

                        annotations_dict[du.DEPTH1].append(os.path.join(scene, 'set_100', 'depth_maps', f"{p1}.h5"))
                        annotations_dict[du.DEPTH2].append(os.path.join(scene, 'set_100', 'depth_maps', f"{p2}.h5"))

                        annotations_dict[du.CALIB1].append(os.path.join(scene, 'set_100', 'calibration', f"calibration_{p1}.h5"))
                        annotations_dict[du.CALIB2].append(os.path.join(scene, 'set_100', 'calibration', f"calibration_{p2}.h5"))

                        bag_image_pairs.append(f"{p1}.jpg {p2}.jpg")

            if not os.path.exists(scene_bag_pairs_path):
                os.mkdir(scene_bag_pairs_path)

            with open(os.path.join(scene_bag_pairs_path, bag_filename), 'w') as file:
                for bip in bag_image_pairs:
                    file.write(f"{bip}\n")

    annotations = pd.DataFrame(data=annotations_dict)
    annotations.to_csv(os.path.join(root_path, f'val_sfm_pairs.csv'))


def create_test_calibration(root_path, scenes):
    for scene in scenes:
        scene_path = get_scene_path(root_path, scene)
        scene_sparse_path = get_scene_sparse_path(scene_path)

        cameras = read_bin_cameras(scene_sparse_path)
        images = read_bin_images(scene_sparse_path)

        test_image_names = os.listdir(get_imw2020_scene_path(root_path, scene))

        scene_calib_dir = os.path.join(get_scene_path(root_path, scene), 'calibration')

        if not os.path.exists(scene_calib_dir):
            os.mkdir(scene_calib_dir)

        for k, v in images.items():
            if v.name in test_image_names:
                R = source.utils.common_utils.qvec2rotmat()
                T = v.tvec

                cam_params = cameras[k].params
                K = np.array([[cam_params[0], 0, cam_params[2]],
                              [0, cam_params[1], cam_params[3]],
                              [0, 0, 1]])

                name = v.name.split(".")[0]
                file_path = os.path.join(scene_calib_dir, f"{name}.h5")

                if os.path.exists(file_path):
                    os.remove(file_path)

                with h5py.File(file_path, 'w') as file:
                    file.create_dataset("R", data=R)
                    file.create_dataset("T", data=T)
                    file.create_dataset("K", data=K)


def create_test_pairs(root_path, scenes, vis_thresh=0.1):
    annotations_dict = {du.SCENE_NAME: [],

                        du.IMAGE1: [],
                        du.IMAGE2: [],

                        du.DEPTH1: [],
                        du.DEPTH2: [],

                        du.CALIB1: [],
                        du.CALIB2: []}

    for scene in scenes:
        scene_path = get_scene_path(root_path, scene)
        images = read_bin_images(get_scene_sparse_path(scene_path))

        pair_data = dd.io.load(os.path.join(get_scene_stereo_path(scene_path), 'pairs-dilation-0.00-fixed2.h5'))
        test_image_names = os.listdir(get_imw2020_scene_path(root_path, scene))

        for p_key in pair_data.output_keys():
            p = pair_data[p_key]
            image1, image2 = images[p_key[0]], images[p_key[1]]
            if p[2] > vis_thresh and p[3] > vis_thresh and \
                    image1.name in test_image_names and \
                    image2.name in test_image_names:
                annotations_dict[du.SCENE_NAME].append(scene)

                annotations_dict[du.IMAGE1].append(os.path.join(scene, 'dense', 'images', image1.name))
                annotations_dict[du.IMAGE2].append(os.path.join(scene, 'dense', 'images', image2.name))

                name1 = image1.name.split(".")[0]
                name2 = image2.name.split(".")[0]

                annotations_dict[du.DEPTH1].append(os.path.join(scene, 'dense', 'stereo',
                                                                'depth_maps_clean_300_th_0.10', f"{name1}.h5"))
                annotations_dict[du.DEPTH2].append(os.path.join(scene, 'dense', 'stereo',
                                                                'depth_maps_clean_300_th_0.10', f"{name2}.h5"))

                annotations_dict[du.CALIB1].append(os.path.join(scene, 'dense', 'calibration', f"{name1}.h5"))
                annotations_dict[du.CALIB2].append(os.path.join(scene, 'dense', 'calibration', f"{name2}.h5"))

    annotations = pd.DataFrame(data=annotations_dict)
    annotations.to_csv(os.path.join(root_path, f'test_pairs.csv'))


def create_test_sfm_pairs(root_path, scenes,
                          bag_groups=[(3, 100),
                                      (5, 100),
                                      (10, 50),
                                      (25, 25)],
                          min_num_vis_kp=100):
    annotations_dict = {du.SCENE_NAME: [],
                        du.BAG_ID: [],

                        du.IMAGE1: [],
                        du.IMAGE2: [],

                        du.DEPTH1: [],
                        du.DEPTH2: [],

                        du.CALIB1: [],
                        du.CALIB2: []}

    for scene in scenes:
        scene_path = get_scene_path(root_path, scene)
        images = read_bin_images(get_scene_sparse_path(scene_path))

        pair_data = dd.io.load(os.path.join(get_scene_stereo_path(scene_path), 'pairs-dilation-0.00-fixed2.h5'))
        test_image_names = os.listdir(get_imw2020_scene_path(root_path, scene))

        test_image_ids = [k for k, v in images.items() if v.name in test_image_names]

        scene_bags_path = get_scene_bags_path(scene_path)
        scene_bag_pairs_path = get_scene_bag_pairs_path(scene_path)

        if not os.path.exists(scene_bags_path):
            os.mkdir(scene_bags_path)

        if not os.path.exists(scene_bag_pairs_path):
            os.mkdir(scene_bag_pairs_path)

        for bag_group in bag_groups:
            num_samples = bag_group[0]
            for i in range(bag_group[1]):
                done = False
                while not done:
                    bag_ids = choice(test_image_ids, size=num_samples, replace=False)
                    bag_id_pairs = []

                    for p_idx1, p_idx2 in itertools.product(range(len(bag_ids)), range(len(bag_ids))):
                        if p_idx1 != p_idx2:
                            pairij = (bag_ids[p_idx1], bag_ids[p_idx2])
                            pair_dataij = pair_data.get(pairij)
                            if pair_dataij is not None and pair_dataij[-1] > min_num_vis_kp:
                                bag_id_pairs.append(pairij)

                    done = len(bag_id_pairs) >= num_samples

                bag_filename = BAG_FILENAME.format(num_samples, i)

                with open(os.path.join(scene_bags_path, bag_filename), 'w') as f:
                    for bag_id in bag_ids:
                        f.write(f"images/{images[bag_id].name}\n")

                with open(os.path.join(scene_bag_pairs_path, bag_filename), 'w') as f:
                    for bag_id_pair in bag_id_pairs:
                        f.write(f"{images[bag_id_pair[0]].name} {images[bag_id_pair[1]].name}\n")

                for bag_id_pair in bag_id_pairs:
                    annotations_dict[du.SCENE_NAME].append(scene)
                    annotations_dict[du.BAG_ID].append(os.path.splitext(bag_filename)[0])

                    img_n1 = images[bag_id_pair[0]].name
                    img_n2 = images[bag_id_pair[1]].name

                    annotations_dict[du.IMAGE1].append(os.path.join(scene, 'dense', 'images', img_n1))
                    annotations_dict[du.IMAGE2].append(os.path.join(scene, 'dense', 'images', img_n2))

                    name1 = os.path.splitext(img_n1)[0]
                    name2 = os.path.splitext(img_n2)[0]

                    annotations_dict[du.DEPTH1].append(os.path.join(scene, 'dense', 'stereo',
                                                                    'depth_maps_clean_300_th_0.10', f"{name1}.h5"))
                    annotations_dict[du.DEPTH2].append(os.path.join(scene, 'dense', 'stereo',
                                                                    'depth_maps_clean_300_th_0.10', f"{name2}.h5"))

                    annotations_dict[du.CALIB1].append(os.path.join(scene, 'dense', 'calibration', f"{name1}.h5"))
                    annotations_dict[du.CALIB2].append(os.path.join(scene, 'dense', 'calibration', f"{name2}.h5"))

    annotations = pd.DataFrame(data=annotations_dict)
    annotations.to_csv(os.path.join(root_path, f'test_sfm_pairs.csv'))


"""
Support utils
"""


def get_scene_path(root_path, scene_name):
    scene_path = os.path.join(root_path, scene_name, 'set_100')

    if os.path.exists(scene_path):
        return scene_path

    else:
        return os.path.join(root_path, scene_name, 'dense')


def get_imw2020_scene_path(root_path, scene_name):
    return os.path.join(root_path, 'imw-2020-test', scene_name)


def get_scene_bags_path(scene_path):
    return os.path.join(scene_path, 'sub_set')


def get_scene_bag_pairs_path(scene_path):
    return os.path.join(scene_path, 'sub_set_pairs')


def get_scene_sparse_path(scene_path):
    return os.path.join(scene_path, 'sparse')


def get_scene_stereo_path(scene_path):
    return os.path.join(scene_path, 'stereo')
