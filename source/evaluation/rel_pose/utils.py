import os
import numpy as np

from PIL import Image

import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu
import source.evaluation.namespace as eva_ns
import source.pose.estimators.namespace as est_ns

from source.core.evaluation import AsyncRequest
from source.evaluation.rel_pose.metrics import relative_pose_error, relative_pose_error_from_fund_mat,\
    relative_pose_error_from_homography, homography_corners_reprojection_error


class PairData:

    def __init__(self, dataset_path,
                 image_name1, image_name2,
                 rec_path):
        self.image_names = [image_name1, image_name2]
        self.image_sizes = [Image.open(os.path.join(dataset_path, img_n)).size for img_n in self.image_names]

        self.image_pair_path = os.path.join(rec_path, 'pair.txt')

        with open(self.image_pair_path, 'w') as f:
            f.write(f"{image_name1} {image_name2}\n")


class RelPoseRequest(AsyncRequest):

    def __init__(self, entity_id, output_keys, estimator):
        super().__init__(entity_id, output_keys)
        self.estimator = estimator

        self.nn_kp2 = None
        self.mm_desc_mask1 = None

    def update(self, batch, nn_kp2, mm_desc_mask1, i):
        super().update_state(batch, i, i + 1)

        self.nn_kp2 = nn_kp2[i].numpy()
        self.mm_desc_mask1 = mm_desc_mask1[i].cpu().numpy()

    def _get_state_keys(self):
        state_keys = super()._get_state_keys()
        state_keys.update([eu.KP1,
                           du.INTRINSICS1, du.INTRINSICS2,
                           du.EXTRINSICS1, du.EXTRINSICS2,
                           du.H1, du.IMAGE_SHAPE1, du.SHIFT_SCALE1])

        return state_keys

    def _process_request(self):
        if self.estimator.name in [est_ns.F_PYDEGENSAC,
                                   est_ns.E_PYOPENGV]:
            T12, inl_mask, sucess = self.estimator.estimate(self.state[eu.KP1][0], self.nn_kp2,
                                                            self.mm_desc_mask1,
                                                            self.state[du.INTRINSICS1][0], self.state[du.INTRINSICS2][0])

            r_err, t_err = relative_pose_error(T12[None], sucess,
                                               self.state[du.EXTRINSICS1], self.state[du.EXTRINSICS2])

            values = {eva_ns.R_ERR: r_err,
                      eva_ns.T_ERR: t_err}

        elif self.estimator.name in [est_ns.H_PYDEGENSAC,
                                     est_ns.H_OPENCV]:
            if next(filter(lambda  i: i in self.output_keys, [eva_ns.HCR_mAA,
                                                              eva_ns.HCR_ERR]), False):
                H12, inl_mask, success = self.estimator.estimate(self.state[eu.KP1][0], self.nn_kp2,
                                                                 self.mm_desc_mask1, False)

                hcr_err = homography_corners_reprojection_error(H12, success,
                                                                self.state[du.IMAGE_SHAPE1],
                                                                self.state[du.SHIFT_SCALE1],
                                                                self.state[du.H1])

                values = {eva_ns.HCR_ERR: hcr_err}

            elif next(filter(lambda i: i in self.output_keys, [eva_ns.R_ERR,
                                                               eva_ns.T_ERR,
                                                               eva_ns.R_mAA,
                                                               eva_ns.T_mAA]), False):
                T12, inl_mask, success = self.estimator.estimate(self.state[eu.KP1][0], self.nn_kp2,
                                                                 self.mm_desc_mask1, True)

                r_err, t_err = relative_pose_error_from_homography(T12, inl_mask, success,
                                                                   self.state[eu.KP1][0], self.nn_kp2,
                                                                   self.state[du.H1][0])

                values = {eva_ns.R_ERR: r_err,
                          eva_ns.T_ERR: t_err}

        if eva_ns.NUM_INL in self.output_keys:
            num_inl = np.array(inl_mask.sum(axis=-1))

            values[eva_ns.NUM_INL] = num_inl

        return values


class COLMAPRelPoseRequest(AsyncRequest):

    def __init__(self, entity_id, output_keys, estimator, pair_data):
        super().__init__(entity_id, output_keys)
        self.estimator = estimator
        self.pair_data = pair_data

        self.nn_kp2 = None
        self.mm_desc_mask1 = None
        self.nn_desc_idx1 = None

    def update(self, batch, nn_kp2, mm_desc_mask1, nn_desc_idx1, i):
        i_inc = i + 1
        super().update_state(batch, i, i_inc)

        self.nn_kp2 = nn_kp2[i].numpy()
        self.mm_desc_mask1 = mm_desc_mask1[i:i_inc].cpu().numpy()
        self.nn_desc_idx1 = nn_desc_idx1[i:i_inc].cpu().numpy()

    def _get_state_keys(self):
        state_keys = super()._get_state_keys()
        state_keys.update([du.IMAGE_NAME1, du.IMAGE_NAME2,
                           eu.KP1, eu.KP2,
                           du.INTRINSICS1, du.INTRINSICS2,
                           du.EXTRINSICS1, du.EXTRINSICS2])

        return state_keys

    def _process_request(self):
        self.estimator.create_new_database()
        self.estimator.initialize_images_and_cameras(self.pair_data.image_names, self.pair_data.image_sizes)

        kp1, kp2 = self.state[eu.KP1], self.state[eu.KP2]

        self.estimator.import_keypoints_and_matches(self.state[du.IMAGE_NAME1], self.state[du.IMAGE_NAME2],
                                                    kp1, kp2,
                                                    self.mm_desc_mask1,
                                                    self.nn_desc_idx1)

        self.estimator.run_matches_importer(self.pair_data.image_pair_path)

        F, inl_mask = self.estimator.get_fund_mat_and_inliers_mask(self.state[du.IMAGE_NAME1],
                                                                   self.state[du.IMAGE_NAME2],
                                                                   kp1.shape[1])

        r_err, t_err = relative_pose_error_from_fund_mat(F[0], inl_mask[0],
                                                         kp1[0], self.nn_kp2,
                                                         self.state[du.INTRINSICS1][0], self.state[du.INTRINSICS2][0],
                                                         self.state[du.EXTRINSICS1], self.state[du.EXTRINSICS2],
                                                         self.estimator.two_view_eval_params.min_num_inliers)

        self.estimator.delete_reconstruction()

        values = {eva_ns.R_ERR: r_err,
                  eva_ns.T_ERR: t_err}

        if eva_ns.NUM_INL in self.output_keys:
            num_inl = np.array(inl_mask.sum(axis=-1))

            values[eva_ns.NUM_INL] = num_inl

        return values


# TODO. Find optimal hparam here


"""
Legacy code
"""

# qvec, tvec, inl_mask = self.estimator.get_qvec_tvec_and_inlier_mask(self.state[du.IMAGE_NAME1],
#                                                                     self.state[du.IMAGE_NAME2],
#                                                                     kp1.shape[1])
#
# r_err, t_err = relative_pose_error_from_qvec_tvec(qvec[0], tvec[0], inl_mask,
#                                                   self.state[du.EXTRINSICS1], self.state[du.EXTRINSICS2],
#                                                   self.estimator.two_view_eval_params.min_num_inliers)