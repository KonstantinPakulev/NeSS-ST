import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

import source.datasets.base.utils as du


"""
MegaDepth annotations utils
"""


def create_dataset_annotations(dataset_root, scene_info_root, name='annotations'):
    annotations_dict = {du.SCENE_NAME: [],

                        du.IMAGE1: [],
                        du.IMAGE2: [],

                        du.DEPTH1: [],
                        du.DEPTH2: [],

                        du.CALIB1: [],
                        du.CALIB2: []}

    min_overlap_ratio = 0.5
    max_overlap_ratio = 1.0
    max_scale_ratio = np.inf

    scene_files = os.listdir(scene_info_root)
    scene_files = [name for name in scene_files if name.endswith(".npz")]

    for scene_file in tqdm(scene_files):
        scene_path = os.path.join(scene_info_root, scene_file)
        scene_info = np.load(scene_path, allow_pickle=True)

        # Square matrix of overlap between two images
        overlap_matrix = scene_info['overlap_matrix']
        # Square matrix of depth ratio between two estimated depths
        scale_ratio_matrix = scene_info['scale_ratio_matrix']

        image_paths = scene_info['image_paths']
        depth_paths = scene_info['depth_paths']

        extrinsics = scene_info['poses']
        intrinsics = scene_info['intrinsics']

        valid = np.logical_and(np.logical_and(overlap_matrix >= min_overlap_ratio,
                                              overlap_matrix <= max_overlap_ratio),
                               scale_ratio_matrix <= max_scale_ratio)
        # Pairs of overlapping images
        pairs = np.vstack(np.where(valid)).transpose(1, 0)

        scene_name = scene_file.split('.')[0]
        rel_scene_data_path = os.path.join('Undistorted_SfM', scene_name, 'data')
        scene_data_path = os.path.join(dataset_root, rel_scene_data_path)

        if not os.path.exists(scene_data_path):
            os.mkdir(scene_data_path)

        for pair in pairs:
            # Pair of matching images
            idx1, idx2 = pair

            file_name1 = f"{image_paths[idx1].split('/')[-1].split('.')[0]}.npy"
            file_name2 = f"{image_paths[idx2].split('/')[-1].split('.')[0]}.npy"

            np.save(os.path.join(scene_data_path, file_name1), {du.EXTRINSICS: extrinsics[idx1],
                                                                du.INTRINSICS: intrinsics[idx1]})
            np.save(os.path.join(scene_data_path, file_name2), {du.EXTRINSICS: extrinsics[idx2],
                                                                du.INTRINSICS: intrinsics[idx2]})

            annotations_dict[du.SCENE_NAME].append(scene_name)

            annotations_dict[du.IMAGE1].append(image_paths[idx1])
            annotations_dict[du.IMAGE2].append(image_paths[idx2])

            annotations_dict[du.DEPTH1].append(depth_paths[idx1])
            annotations_dict[du.DEPTH2].append(depth_paths[idx2])

            annotations_dict[du.CALIB1].append(os.path.join(rel_scene_data_path, file_name1))
            annotations_dict[du.CALIB2].append(os.path.join(rel_scene_data_path, file_name2))

    annotations = pd.DataFrame(data=annotations_dict)
    annotations.to_csv(os.path.join(scene_info_root, f'{name}.csv'))


def create_train_pairs_disk(scene_info_root):
    annotations = pd.read_csv(os.path.join(scene_info_root, 'annotations.csv'), index_col=[0])

    with open(os.path.join(scene_info_root, 'disk_dataset.json')) as file:
        train_disk_set = json.load(file)

        mask = pd.Series(np.zeros((len(annotations[du.IMAGE1])), dtype=np.bool))

        for k in train_disk_set.keys():
            k_image_names = train_disk_set[k]['images']
            k_mask = (annotations[du.SCENE_NAME] == int(k)) & \
                     annotations[du.IMAGE1].transform(lambda x: x.split('/')[-1]).isin(k_image_names) & \
                     annotations[du.IMAGE2].transform(lambda x: x.split('/')[-1]).isin(k_image_names)

            mask |= k_mask

        annotations.loc[mask].reset_index(drop=True).to_csv(os.path.join(scene_info_root, f'train_disk_pairs.csv'))


def create_prefix_pairs(scene_info_root, prefix, scenes, num_samples=200):
    annotations = pd.read_csv(os.path.join(scene_info_root, 'annotations.csv'), index_col=[0])

    scenes_pairs = annotations[annotations[du.SCENE_NAME].isin(scenes)]
    scenes_pairs = scenes_pairs.groupby(du.SCENE_NAME, group_keys=False).apply(pd.DataFrame.sample, n=num_samples)

    scenes_pairs.reset_index(drop=True).to_csv(os.path.join(scene_info_root, f'{prefix}_pairs.csv'))


"""
Legacy code
"""

# def select_annotations_by_scene(scene_info_root, mode, scenes, is_in=True, reset_index=True):
#     annotations = pd.read_csv(os.path.join(scene_info_root, 'annotations.csv'), index_col=[0])
#
#     if is_in:
#         mode_annotations = annotations.loc[annotations[du.SCENE_NAME].isin(scenes)]
#     else:
#         mode_annotations = annotations.loc[~annotations[du.SCENE_NAME].isin(scenes)]
#
#     if reset_index:
#         mode_annotations = mode_annotations.reset_index()
#
#     mode_annotations.to_csv(os.path.join(scene_info_root, f'{mode}.csv'))
#
#
# def sample_annotations_by_scene(scene_info_root, mode, scenes, num_samples, reset_index=True):
#     annotations = pd.read_csv(os.path.join(scene_info_root, 'annotations.csv'), index_col=[0])
#
#     mode_annotations = annotations.loc[annotations[du.SCENE_NAME].isin(scenes)]
#     mode_annotations = mode_annotations.groupby(du.SCENE_NAME, group_keys=False).apply(pd.DataFrame.sample, n=num_samples)
#
#     if reset_index:
#         mode_annotations = mode_annotations.reset_index()
#
#     mode_annotations.to_csv(os.path.join(scene_info_root, f'{mode}.csv'))
#
#


# import torch

#
# import source.core.model as m

#
#
# from source.datasets.megadepth.megadepth_dataset import MegaDepthDataset
# from source.datasets.megadepth.megadepth_utils import SceneDataWrapper
#
# from source.utils.projectivity_utils import warp_image_rbt

# def filter_by_co_visibility(csv_path, device, co_vis_ratio=0.2):
#     size = (240, 320)
#
#     item_transforms = transforms.Compose([mt.MegaDepthToPILImage(),
#                                           mt.MegaDepthCrop(mt.MegaDepthSharedAreaCrop()),
#                                           mt.MegaDepthResize(du.AspectResize(size, True)),
#                                           mt.MegaDepthCrop(du.CentralCrop(size, True)),
#                                           mt.MegaDepthToTensor()])
#
#     datasets = MegaDepthPairDataset(csv_path, item_transforms)
#
#     batch_size = 16
#
#     data_loader = DataLoader(datasets, batch_size=batch_size, num_workers=8)
#
#     overlap_mask = torch.zeros(datasets.__len__(), dtype=torch.bool)
#
#     for i, batch in tqdm(enumerate(data_loader)):
#         image1, image2 = batch[du.IMAGE1].to(device), batch[du.IMAGE2].to(device)
#
#         scene_data = SceneDataWrapper(batch, m.PAIR, device)
#
#         b = image1.shape[0]
#
#         vis_mask1 = warp_image_rbt(image2, scene_data, 'm')
#         vis_mask2 = warp_image_rbt(image1, scene_data.swap(), 'm')
#
#         total1 = vis_mask1.shape[2] * vis_mask1.shape[3]
#         total2 = vis_mask2.shape[2] * vis_mask2.shape[3]
#
#         vis_ratio1 = vis_mask1.view(b, -1).sum(dim=-1).float() / total1
#         vis_ratio2 = vis_mask2.view(b, -1).sum(dim=-1).float() / total2
#
#         vis_ratio = torch.min(vis_ratio1, vis_ratio2)
#
#         start = i * batch_size
#         end = min(start + batch_size, datasets.__len__())
#
#         overlap_mask[start:end] = vis_ratio.cpu() >= co_vis_ratio
#
#     print(f"Total number of pairs: {datasets.__len__()}, Number of filtered pairs: {overlap_mask.sum().item()}")
#
#     annotations = pd.read_csv(csv_path, index_col=[0])
#     annotations = annotations[overlap_mask.numpy().tolist()].reset_index(drop=True)
#
#     annotations.to_csv(csv_path)


# def create_visibility_scores(dataset_root, csv_path, device):
#     size = (240, 320)
#
#     item_transforms = transforms.Compose([mt.MegaDepthToPILImage(),
#                                           mt.MegaDepthCrop(mt.MegaDepthSharedAreaCrop()),
#                                           mt.MegaDepthResize(du.AspectResize(size, True)),
#                                           mt.MegaDepthToTensor()])
#
#     datasets = MegaDepthPairDataset(csv_path, item_transforms)
#
#     data_loader = DataLoader(datasets, batch_size=1)
#
#     visited = {}
#
#     for batch in tqdm(data_loader):
#         image1, image2 = batch[du.IMAGE1].to(device), batch[du.IMAGE2].to(device)
#
#         scene_data_wrapper = SceneDataWrapper(batch, m.PAIR, device)
#
#         vis_mask1 = warp_image_RBT(image2, scene_data_wrapper, 'm')
#         vis_mask2 = warp_image_RBT(image1, scene_data_wrapper.swap(), 'm')
#
#         scene_name = batch[du.SCENE_NAME]
#         image_name1, image_name2 = batch[du.IMAGE_NAME1], batch[du.IMAGE_NAME2]
#
#         base_scene_data_path = os.path.join(dataset_root, 'Undistorted_SfM', scene_name[0].zfill(4), "data",)
#
#         for image_name, vis_mask in zip([image_name1[0], image_name2[0]],
#                                         [vis_mask1, vis_mask2]):
#             scene_data_path = os.path.join(base_scene_data_path,
#                                            image_name.split('.')[0] + '.npy')
#
#             with open(scene_data_path, 'rb') as r_file:
#                 scene_data = np.load(r_file, allow_pickle=True).item()
#
#             image_key = f'{scene_name}/{image_name}'
#
#             if image_key not in visited:
#                 scene_data[du.VIS_SCORE] = np.zeros((vis_mask.shape[2], vis_mask.shape[3]), dtype=np.int)
#
#                 visited[image_key] = True
#
#             scene_data[du.VIS_SCORE] += vis_mask.long().cpu().numpy()[0, 0]
#
#             with open(scene_data_path, 'wb') as w_file:
#                 np.save(w_file, scene_data)


#         vis_diff = (vis_ratio1 - vis_ratio2).abs()
#         overlap_mask[start:end] = vis_diff.cpu() < 0.1
# def create_annotations_by_log(log, annotations_path, log_path, file_name, reset_index=True):
#     file_path = "/".join(log_path.split("/")[:-1])
#     file_path = os.path.join(file_path, f"{file_name}.csv")
#
#     annotations = select_from_annotations(annotations_path, log)
#
#     if reset_index:
#         annotations = annotations.reset_index()
#
#     annotations.to_csv(file_path)


# def create_ha_annotations(csv_path):
#     ha_annotations = pd.read_csv(csv_path, index_col=[0]).drop(du.DEPTH1, axis=1).drop(du.DEPTH2, axis=1).\
#         drop(du.SCENE_DATA1, axis=1).drop(du.SCENE_DATA2, axis=1)
#
#     ha_annotations1 = ha_annotations.drop_duplicates(du.IMAGE1).drop(du.IMAGE2, axis=1).drop(du.ID2, axis=1)
#     ha_annotations2 = ha_annotations.drop_duplicates(du.IMAGE2).drop(du.IMAGE1, axis=1).drop(du.ID1, axis=1).\
#         rename(columns={du.IMAGE2: du.IMAGE1, du.ID2: du.ID1})
#     ha_annotations = pd.concat([ha_annotations1, ha_annotations2], sort=False).drop_duplicates(du.IMAGE1)
#
#     split = csv_path.split('/')
#     csv_name = split[-1].split('.')[-2]
#     scene_info_root = '/'.join(split[:-1])
#
#     ha_annotations.reset_index(drop=True).to_csv(os.path.join(scene_info_root, f'{csv_name}_ha.csv'))


# """
# Support utils
# """


# def visualize_log_row(annotations_row):
#     image1_path, image2_path = annotations_row[du.IMAGE1].item(), annotations_row[du.IMAGE2].item()
#
#     plot_figures({'image1': io.imread(image1_path),
#                   'image2': io.imread(image2_path)}, 1, 2)


# def select_from_annotations(annotations_path, log):
#     annotations = pd.read_csv(annotations_path, index_col=[0])
#
#     feature = annotations[du.ID1].astype(str) + annotations[du.ID2].astype(str)
#     selection = (log[du.ID1].astype(str) + log[du.ID2].astype(str)).tolist()
#
#     annotations = annotations[feature.isin(selection)]
#
#     return annotations

# sift_kp = scene_info['points3D_id_to_2D']
        # sift_kp1, sift_kp2 = np.array(list(sift_kp[idx1].values())), np.array(list(sift_kp[idx2].values()))