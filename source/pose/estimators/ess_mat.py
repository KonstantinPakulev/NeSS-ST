import pyopengv
import numpy as np

from source.pose.estimators.base.rel_pose import RelPoseEstimator
from source.pose.estimators.utils import to_nic
from source.utils.common_utils import normalize, angle_vec


class EssMatEstimator(RelPoseEstimator):

    @staticmethod
    def from_config(model_mode_eval_params):
        estimator_eval_params = model_mode_eval_params.estimator
        return EssMatEstimator(estimator_eval_params.inl_thresh,
                               estimator_eval_params.confidence,
                               estimator_eval_params.num_ransac_iter,
                               estimator_eval_params.min_num_inliers,
                               estimator_eval_params.name)

    def estimate(self, kp1, nn_kp2,
                 mm_desc_mask1,
                 intrinsics1, intrinsics2):
        """
        :param kp1: N x 2
        :param nn_kp2: N x 2
        :param mm_desc_mask1: N
        :param intrinsics1: 3 x 3
        :param intrinsics2: 3 x 3
        :return: 3 x 4, N; returns T camera1->camera2
        """
        T12 = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
        inl_mask = np.zeros((kp1.shape[0]), dtype=np.bool)

        if mm_desc_mask1.sum() < 8:
            return T12, inl_mask, False

        else:
            kp1, nn_kp2 = kp1[mm_desc_mask1], nn_kp2[mm_desc_mask1]

            bear1 = to_bearing(to_nic(kp1, intrinsics1))
            nn_bear2 = to_bearing(to_nic(nn_kp2, intrinsics2))

            angle_thresh1x = np.arctan2(self.inl_thresh, intrinsics1[0, 0])
            angle_thresh1y = np.arctan2(self.inl_thresh, intrinsics1[1, 1])

            angle_thresh2x = np.arctan2(self.inl_thresh, intrinsics2[0, 0])
            angle_thresh2y = np.arctan2(self.inl_thresh, intrinsics2[1, 1])

            angle_thresh = np.minimum(np.minimum(angle_thresh1x, angle_thresh1y),
                                      np.minimum(angle_thresh2x, angle_thresh2y))

            T21 = pyopengv.relative_pose_ransac(bear1, nn_bear2, "STEWENIUS",
                                                1.0 - np.cos(angle_thresh),
                                                self.num_ransac_iter, self.confidence)

            _inl_mask = compute_bearing_inliers_mask(bear1, nn_bear2, T21, angle_thresh)

            if _inl_mask.sum() < self.min_num_inliers:
                return T12, inl_mask, False

            T21 = pyopengv.relative_pose_optimize_nonlinear(bear1[_inl_mask], nn_bear2[_inl_mask],
                                                            T21[:, 3], T21[:3, :3])

            _inl_mask = compute_bearing_inliers_mask(bear1, nn_bear2, T21, angle_thresh)

            T12[:3, :3] = T21[:3, :3].T
            T12[:3, 3:] = -(T21[:3, :3].T @ T21[:, 3, None])

            inl_mask[mm_desc_mask1] = _inl_mask

            return T12, inl_mask, True


"""
Support utils
"""

def to_bearing(norm_kp):
    norm_kp = to_homogeneous(norm_kp)
    return normalize(norm_kp)

def to_homogeneous(kp):
    return np.concatenate([kp, np.ones((kp.shape[0], 1))], axis=-1)

def compute_bearing_inliers_mask(bear1, nn_bear2, T21, angle_thresh):
    R, t = T21[:3, :3], T21[:, 3]

    tr_bear1 = pyopengv.triangulation_triangulate(bear1, nn_bear2, t, R)
    w_tr_bear1 = (R.T @ (tr_bear1 - t[None, :]).T).T

    inl_mask1 = angle_vec(normalize(tr_bear1), bear1, False) < angle_thresh
    inl_mask2 = angle_vec(normalize(w_tr_bear1), nn_bear2, False) < angle_thresh

    return inl_mask1 & inl_mask2
