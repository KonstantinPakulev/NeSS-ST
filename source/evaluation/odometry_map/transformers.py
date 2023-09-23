import torch

import source.pose.matcher as mr

import source.pose.matchers.namespace
from source.core.evaluation import PairMetricTransformer

from source.pose.matcher import L2Matcher, HammingMatcher
from deprecated.abs_pose import AbsPoseEstimatorG2O

# from source.evaluation.odometry_map.utils import pointcloudify_depth

ODOMETRY_MAP = 'odometry_map'

POSE = 'pose'
PCD = 'pcd'
PCD_COLOR = 'pcd_color'


class OdometryMapTransformer(PairMetricTransformer):

    def __init__(self, entity_id, metric_config, model_mode_eval_params, device):
        super().__init__(entity_id, metric_config)
        self.device = device

        lowe_ratio = model_mode_eval_params.get(mr.LOWE_RATIO)

        if metric_config.matcher == source.pose.matchers.namespace.L2:
            self.matcher = L2Matcher(lowe_ratio, device)

        elif metric_config.matcher == source.pose.matchers.namespace.HAMMING:
            self.matcher = HammingMatcher(lowe_ratio)

        else:
            raise ValueError(f"Unknown matcher: {metric_config.matcher}")

        self.estimator = AbsPoseEstimatorG2O()

        self.prev_kp = None
        self.prev_kp_desc = None
        self.prev_depth = None
        self.prev_shift_scale = None
        self.prev_intrinsics = None

        self.curr_extrinsics = torch.eye(4).reshape(1, 4, 4)

    def _transform_impl(self, engine, batch, endpoint):
        pass

        # kp1 = endpoint[eu.KP1]
        # kp_desc1 = endpoint[eu.KP_DESC1]
        #
        # image1 = batch[du.IMAGE1]
        # depth1 = batch[du.DEPTH1]
        # shift_scale1 = batch[du.SHIFT_SCALE1]
        # intrinsics = batch[du.INTRINSICS]
        #
        # if self.prev_kp is not None:
        #     extrinsics, _ = self.estimator.match_and_estimate(self.prev_kp, kp1,
        #                                                       self.prev_kp_desc, kp_desc1,
        #                                                       self.prev_depth,
        #                                                       self.prev_shift_scale, shift_scale1,
        #                                                       self.prev_intrinsics, intrinsics,
        #                                                       self.matcher)
        #
        #     self.curr_extrinsics = to_homogeneous_pose(extrinsics) @ self.curr_extrinsics
        #
        # curr_pose = self.curr_extrinsics.inverse()
        #
        # world_grid1, grid_depth_mask1, grid_color1 = pointcloudify_depth(image1, depth1, intrinsics, shift_scale1, True)
        #
        # world_grid1 = to_cartesian((curr_pose @ to_homogeneous(world_grid1).permute(0, 2, 1)).permute(0, 2, 1))
        #
        # world_grid1 = world_grid1[0, grid_depth_mask1[0]]
        # grid_color1 = grid_color1[0, grid_depth_mask1[0]]
        #
        # transformed = {POSE: curr_pose.numpy(),
        #                PCD: world_grid1.numpy(),
        #                PCD_COLOR: grid_color1.numpy()}
        #
        # self.prev_kp = kp1
        # self.prev_kp_desc = kp_desc1
        # self.prev_depth = depth1
        # self.prev_shift_scale = shift_scale1
        # self.prev_intrinsics = intrinsics
        #
        # return transformed
