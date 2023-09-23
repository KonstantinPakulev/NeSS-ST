from abc import ABC, abstractmethod


class RelPoseEstimator(ABC):

    def __init__(self,
                 inl_thresh, confidence, num_ransac_iter,
                 min_num_inliers,
                 name):
        self.inl_thresh = inl_thresh
        self.confidence = confidence
        self.num_ransac_iter = num_ransac_iter
        self.min_num_inliers = min_num_inliers
        self.name = name
