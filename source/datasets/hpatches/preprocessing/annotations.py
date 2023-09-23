import os
import pandas as pd

import source.datasets.base.utils as du


def create_pairs_annotations(dataset_root, file_name, include_scenes=None, exclude_scenes=None):
    annotations = pd.DataFrame(columns=[du.SCENE_NAME, du.IMAGE1, du.IMAGE2, du.H1, du.H2])

    for scene_namei in next(os.walk(dataset_root))[1]:
        if include_scenes is not None and \
                scene_namei not in include_scenes:
            continue

        elif exclude_scenes is not None and scene_namei in exclude_scenes:
            continue

        for j in range(2, 7):
            annotations = annotations.append({du.SCENE_NAME: scene_namei,
                                              du.IMAGE1: f'{scene_namei}/1.ppm',
                                              du.IMAGE2: f'{scene_namei}/{j}.ppm',
                                              du.H1: f'{scene_namei}/H_1_{j}',
                                              du.H2: f'{scene_namei}/H_1_{j}'}, ignore_index=True)

    annotations.to_csv(os.path.join(dataset_root, f'{file_name}_pairs.csv'))


def create_annotations(dataset_root, file_name, include_scenes=None, exclude_scenes=None):
    annotations = pd.DataFrame(columns=[du.SCENE_NAME, du.IMAGE1])

    for scene_namei in next(os.walk(dataset_root))[1]:
        if include_scenes is not None and \
                scene_namei not in include_scenes:
            continue

        elif exclude_scenes is not None and scene_namei in exclude_scenes:
            continue

        annotations = annotations.append({du.SCENE_NAME: scene_namei,
                                          du.IMAGE1: f'{scene_namei}/1.ppm'}, ignore_index=True)

        for j in range(2, 7):
            annotations = annotations.append({du.SCENE_NAME: scene_namei,
                                              du.IMAGE1: f'{scene_namei}/{j}.ppm'}, ignore_index=True)

    annotations.to_csv(os.path.join(dataset_root, f'{file_name}.csv'))
