import os
import numpy as np

from itertools import groupby

import source.evaluation.namespace as ns

from source.evaluation.metrics import accuracy
from source.evaluation.utils import get_rel_pose, angle_mat
from source.pose.estimators.colmap.utils import read_bin_images
from source.utils.common_utils import angle_vec


def relative_pose_error(model_path,
                        image_name1, image_name2,
                        extrinsics1, extrinsics2,
                        image_name2id):
    b = len(image_name1)

    model_path = os.path.join(model_path, '0')

    if os.path.exists(model_path):
        images = read_bin_images(model_path)

        est_extrinsics1, est_extrinsics2 = np.tile(np.eye(4), (b, 1, 1)), np.tile(np.eye(4), (b, 1, 1))
        est_mask = np.zeros((b), dtype=np.bool)

        for i, (i_n1, i_n2) in enumerate(zip(image_name1, image_name2)):
            image_id1, image_id2 = image_name2id[i_n1], image_name2id[i_n2]

            im1i, im2i = images.get(image_id1), images.get(image_id2)

            if im1i is not None and im2i is not None:
                est_extrinsics1[i, :3, :3] = im1i.qvec2rotmat()
                est_extrinsics1[i, :3, 3] = im1i.tvec

                est_extrinsics2[i, :3, :3] = im2i.qvec2rotmat()
                est_extrinsics2[i, :3, 3] = im2i.tvec

                est_mask[i] = True

        est_R, _, est_t_dir = get_rel_pose(est_extrinsics1, est_extrinsics2)
        gt_R, _, gt_t_dir = get_rel_pose(extrinsics1, extrinsics2)

        r_err = angle_mat(est_R, gt_R)
        t_err = angle_vec(est_t_dir, gt_t_dir)

        r_err[~est_mask] = ns.MAX_ANGLE_ERR
        t_err[~est_mask] = ns.MAX_ANGLE_ERR

    else:
        r_err = np.ones((b)) * ns.MAX_ANGLE_ERR
        t_err = np.ones((b)) * ns.MAX_ANGLE_ERR

    return r_err, t_err


def bag_grouped_mAA(bag_id, metric_value, err_thresh):
    bag_group_mAA = {}

    for k, v in groupby(zip(bag_id, metric_value), lambda x: x[0]):
        mv = np.stack([i[1] for i in list(v)])

        bag_group_mAA[k] = accuracy(mv, err_thresh).mean(axis=0)

    bag_group_mAA = dict(sorted(bag_group_mAA.items()))

    return bag_group_mAA


def bag_size_grouped_mAA(bag_group_mAA):
    return {k: np.stack([i[1] for i in list(v)]).mean(axis=0)
            for k, v in groupby(bag_group_mAA.items(), lambda x: x[0].split("bag")[0])}
