import numpy as np

import source.utils.endpoint_utils as eu
import source.evaluation.namespace as eva_ns

from source.utils.common_utils import normalize, rad2deg


def get_rel_pose(extrinsic1, extrinsic2):
    """
    :param extrinsic1: B x 4 x 4, :type np.float
    :param extrinsic2: B x 4 x 4, :type np.float
    :return: (B x 3 x 4) :type np.float
    """
    T = extrinsic2 @ np.linalg.inv(extrinsic1)

    return T[:, :3, :3], T[:, :3, 3], normalize(T[:, :3, 3])


def angle_mat(R1, R2):
    """
    :param R1: B x 3 x 3
    :param R2: B x 3 x 3
    :return: B, angle in degrees
    """
    R_d = R1.transpose((0, 2, 1)) @ R2

    angles = np.zeros((R1.shape[0],))
    for i, i_R_d in enumerate(R_d):
        c = (np.trace(i_R_d) - 1) / 2
        angles[i] = rad2deg(np.arccos(np.clip(c, a_min=-1.0, a_max=1.0)))

    return angles


def get_sweep_range(d):
    start = d.start
    end = d.end
    step = d.step
    num = round((end - start) / step) + 1

    return np.around(np.linspace(start, end, num), 2)


def get_kp_desc_and_kp_desc_mask(batch):
    kp1, kp2 = batch[eu.KP1], batch[eu.KP2]
    kp_desc1, kp_desc2 = batch[eu.KP_DESC1], batch[eu.KP_DESC2]
    kp_desc_mask1, kp_desc_mask2 = (kp1 != eva_ns.INVALID_KP).prod(-1).bool(), (kp2 != eva_ns.INVALID_KP).prod(-1).bool()

    return kp_desc1, kp_desc2, kp_desc_mask1, kp_desc_mask2


def get_best_threshold(mAA):
    best_thresh_idx = (r_mAA + t_mAA).argmax()
    return best_thresh_idx


"""
Legacy code
"""

# GT_R = 'gt_r'

# import os
# import numpy as np
# import pandas as pd
# import source.datasets.utils as du
# from source.evaluation.rel_pose.metrics import angle_mat

#
# def append_gt_r(dataset_root, eval_summary):
#     on = [du.SCENE_NAME, du.IMAGE_NAME1, du.IMAGE_NAME2]
#
#     gt_r_summary = []
#
#     for i, row in eval_summary[on].iterrows():
#         scene_name, image1_name, image2_name = row[du.SCENE_NAME], row[du.IMAGE_NAME1], row[du.IMAGE_NAME2]
#
#         base_scene = os.path.join(dataset_root, "Undistorted_SfM", str(scene_name).zfill(4), 'data')
#
#         image1_path = os.path.join(base_scene, image1_name).split('.')[0]
#         image2_path = os.path.join(base_scene, image2_name).split('.')[0]
#
#         scene_data1 = np.load(image1_path + '.npy', allow_pickle=True).item()
#         scene_data2 = np.load(image2_path + '.npy', allow_pickle=True).item()
#
#         extrinsics1 = np.array(scene_data1[du.EXTRINSICS], dtype=np.float)
#         extrinsics2 = np.array(scene_data2[du.EXTRINSICS], dtype=np.float)
#
#         gt_r = angle_mat(extrinsics1[None, :3, :3], extrinsics2[None, :3, :3]).item()
#
#         gt_r_summary.append([scene_name, image1_name, image2_name, gt_r])
#
#     gt_r_summary = pd.DataFrame(gt_r_summary, columns=[du.SCENE_NAME, du.IMAGE_NAME1, du.IMAGE_NAME2, GT_R])
#
#     return eval_summary.merge(gt_r_summary, on=on)
