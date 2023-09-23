import os
import numpy as np
import pandas as pd
from numpy.random import choice

import source.datasets.base.utils as du


def create_prefix_pairs(root_path, prefix, num_samples, name=None):
    scenes_path = os.path.join(root_path, f"scannetv2_{prefix}.txt")

    with open(scenes_path) as f:
        scenes = [i.strip() for i in f.readlines()]

    scans_dir = 'scans_test' if prefix == 'test' else 'scans'
    scans_path = os.path.join(root_path, scans_dir)

    intervals = [10, 30, 60]

    annotations_dict = {du.SCENE_NAME: [],

                        du.FRAME_INTERVAL: [],

                        du.IMAGE1: [],
                        du.IMAGE2: [],

                        du.DEPTH1: [],
                        du.DEPTH2: [],

                        du.CALIB1: [],
                        du.CALIB2: []}

    for s in scenes:
        scene_color_path = os.path.join(scans_path, s, 'color')
        names = [n.split('.')[0] for n in sorted(os.listdir(scene_color_path), key=filename2int)]

        selected_names1 = []
        selected_names2 = []
        selected_frame_interval = []

        for i in intervals:
            selected_names1i = list(choice(names[:-i], (num_samples,), replace=False))
            selected_names2i = [str(int(j) + i) for j in selected_names1i]

            selected_names1.extend(selected_names1i)
            selected_names2.extend(selected_names2i)

            selected_frame_interval.extend([i] * num_samples)

            names = sorted(list(set(names) - set(selected_names1i)), key=lambda x: int(x))

        image_path1, image_path2 = [], []
        depth_path1, depth_path2 = [], []
        calib_path1, calib_path2 = [], []
        frame_interval = []

        for n1, n2, fi in zip(selected_names1, selected_names2, selected_frame_interval):
            calib_path1i = os.path.join(scans_path, s, 'pose', f'{n1}.txt')
            calib_path2i = os.path.join(scans_path, s, 'pose', f'{n2}.txt')

            if is_extrinsics_valid(calib_path1i) and is_extrinsics_valid(calib_path2i):
                image_path1i = os.path.join(scans_dir, s, 'color', f'{n1}.jpg')
                image_path2i = os.path.join(scans_dir, s, 'color', f'{n2}.jpg')

                depth_path1i = os.path.join(scans_dir, s, 'depth', f'{n1}.png')
                depth_path2i = os.path.join(scans_dir, s, 'depth', f'{n2}.png')

                image_path1.append(image_path1i)
                image_path2.append(image_path2i)

                depth_path1.append(depth_path1i)
                depth_path2.append(depth_path2i)

                calib_path1.append(os.path.join(scans_dir, s, 'pose', f'{n1}.txt'))
                calib_path2.append(os.path.join(scans_dir, s, 'pose', f'{n2}.txt'))

                frame_interval.append(fi)

        annotations_dict[du.SCENE_NAME].extend([s] * len(image_path1))

        annotations_dict[du.FRAME_INTERVAL].extend(frame_interval)

        annotations_dict[du.IMAGE1].extend(image_path1)
        annotations_dict[du.IMAGE2].extend(image_path2)

        annotations_dict[du.DEPTH1].extend(depth_path1)
        annotations_dict[du.DEPTH2].extend(depth_path2)

        annotations_dict[du.CALIB1].extend(calib_path1)
        annotations_dict[du.CALIB2].extend(calib_path2)

    annotations = pd.DataFrame(data=annotations_dict)
    annotations.to_csv(os.path.join(root_path, f'{prefix}_pairs.csv' if name is None else f'{name}_pairs.csv'))


"""
Support utils
"""

def is_extrinsics_valid(calib_path):
    with open(calib_path) as f:
        extrinsics = [i.strip().split(' ') for i in f.readlines()]
        extrinsics = np.array(extrinsics, dtype=np.float32)

        return (np.isnan(extrinsics).sum() == 0) and (np.isinf(extrinsics).sum() == 0)


def filename2int(s):
    name, _ = s.split('.')
    return int(name)
