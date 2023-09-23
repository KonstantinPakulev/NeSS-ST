import cv2
import pydegensac
import numpy as np

import source.pose.estimators.namespace as est_ns

from source.pose.estimators.base.rel_pose import RelPoseEstimator
from source.pose.estimators.utils import recover_relative_pose_from_homography

class HomographyEstimator(RelPoseEstimator):

    @staticmethod
    def from_config(model_mode_eval_params):
        estimator_eval_params = model_mode_eval_params.estimator
        return HomographyEstimator(estimator_eval_params.inl_thresh,
                                   estimator_eval_params.confidence,
                                   estimator_eval_params.num_ransac_iter,
                                   estimator_eval_params.min_num_inliers,
                                   estimator_eval_params.name)

    def estimate(self, kp1, nn_kp2,
                 mm_desc_mask1,
                 recover_pose=True):
        """
        :param kp1: N x 2
        :param nn_kp2: N x 2
        :param mm_desc_mask1: N
        :param intrinsics1: 3 x 3
        :param intrinsics2: 3 x 3
        :return: 3 x 4 or 3 x 3, N; returns T or H camera1->camera2
        """
        if recover_pose:
            T12 = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)

        else:
            H = np.eye(3)

        inl_mask = np.zeros((kp1.shape[0]), dtype=np.bool)

        if mm_desc_mask1.sum() < self.min_num_inliers:
            if recover_pose:
                return T12, inl_mask, False

            else:
                return H, inl_mask, False

        else:
            kp1, nn_kp2 = kp1[mm_desc_mask1], nn_kp2[mm_desc_mask1]

            if self.name == est_ns.H_PYDEGENSAC:
                _H, _inl_mask = pydegensac.findHomography(kp1, nn_kp2,
                                                          self.inl_thresh,
                                                          self.confidence,
                                                          self.num_ransac_iter)

            elif self.name == est_ns.H_OPENCV:
                _H, _inl_mask = cv2.findHomography(kp1, nn_kp2, cv2.RANSAC,
                                                   self.inl_thresh,
                                                   None,
                                                   self.num_ransac_iter,
                                                   self.confidence)
                _inl_mask = _inl_mask[:, 0]

            if recover_pose:
                if _inl_mask.sum() < self.min_num_inliers:
                    return T12, inl_mask, False

                else:
                    T12 = recover_relative_pose_from_homography(_H, _inl_mask,
                                                                kp1, nn_kp2)
                    inl_mask[mm_desc_mask1] = _inl_mask

                    return T12, inl_mask, True

            else:
                if _inl_mask.sum() < self.min_num_inliers:
                    return H, inl_mask, False

                else:
                    H = _H
                    inl_mask[mm_desc_mask1] = _inl_mask

                    return H, inl_mask, True
