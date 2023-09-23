import cv2
import pydegensac
import numpy as np

from source.pose.estimators.base.rel_pose import RelPoseEstimator
from source.pose.estimators.utils import recover_relative_pose_from_fund_mat


class FundMatEstimator(RelPoseEstimator):

    @staticmethod
    def from_config(model_mode_eval_params):
        estimator_eval_params = model_mode_eval_params.estimator
        return FundMatEstimator(estimator_eval_params.inl_thresh,
                                estimator_eval_params.confidence,
                                estimator_eval_params.num_ransac_iter,
                                estimator_eval_params.min_num_inliers,
                                estimator_eval_params.name)

    def estimate(self, kp1, nn_kp2,
                 mm_desc_mask1,
                 intrinsics1, intrinsics2,
                 recover_pose=True):
        """
        :param kp1: N x 2
        :param nn_kp2: N x 2
        :param mm_desc_mask1: N
        :param intrinsics1: 3 x 3
        :param intrinsics2: 3 x 3
        :param recover_pose: bool
        :return: 3 x 4 or 3 x 3, N; returns T or F camera1->camera2
        """
        if recover_pose:
            T12 = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)

        else:
            F = np.eye(3)

        inl_mask = np.zeros((kp1.shape[0]), dtype=np.bool)

        if mm_desc_mask1.sum() < 8:
            if recover_pose:
                return T12, inl_mask, False

            else:
                return F, inl_mask, False

        else:
            kp1, nn_kp2 = kp1[mm_desc_mask1], nn_kp2[mm_desc_mask1]

            _F, _inl_mask = pydegensac.findFundamentalMatrix(kp1, nn_kp2,
                                                             self.inl_thresh,
                                                             self.confidence,
                                                             self.num_ransac_iter)

            if recover_pose:
                if _inl_mask.sum() < self.min_num_inliers:
                    return T12, inl_mask, False

                else:
                    T12 = recover_relative_pose_from_fund_mat(_F, _inl_mask,
                                                              kp1, nn_kp2,
                                                              intrinsics1, intrinsics2)
                    inl_mask[mm_desc_mask1] = _inl_mask

                    return T12, inl_mask, True

            else:
                if _inl_mask.sum() < self.min_num_inliers:
                    return F, inl_mask, False

                else:
                    F = _F
                    inl_mask[mm_desc_mask1] = _inl_mask

                    return F, inl_mask
