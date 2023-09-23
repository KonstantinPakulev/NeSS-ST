import os
import numpy as np
import copy

from PIL import Image

import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu
import source.evaluation.namespace as eva_ns
import source.pose.estimators.namespace as est_ns

from source.core.evaluation import AsyncPairMetricTransformer
from source.pose.estimators.factory import instantiate_estimator
from source.pose.matchers.factory import instantiate_matcher
from source.pose.matchers.utils import gather_kp
from source.evaluation.metrics import accuracy
from source.evaluation.rel_pose.utils import PairData, RelPoseRequest, COLMAPRelPoseRequest
from source.evaluation.utils import get_sweep_range, get_kp_desc_and_kp_desc_mask


REC_ID = "{}_{}_{}"
HTUNE_REC_ID = "{}_{}_{}_{}"


"""
Metrics transformers
"""


class PairAsyncPairMetricTransformer(AsyncPairMetricTransformer):

    def __init__(self, dataset_mode_config, metric_config,
                 model_mode_eval_params,
                 device):
        super().__init__(dataset_mode_config.get(du.ENTITY_ID), metric_config)
        self.dataset_mode_config = dataset_mode_config
        self.model_mode_eval_params = model_mode_eval_params
        self.device = device

    def _process_pair_request(self, batch,
                              nn_kp2, mm_desc_mask1, nn_desc_idx1,
                              model_mode_eval_params,
                              i):
        estimator = instantiate_estimator(model_mode_eval_params)

        request = RelPoseRequest(self.entity_id, self.output_keys,
                                 estimator)
        request.update(batch, nn_kp2, mm_desc_mask1, i)

        self._submit_request(request)


class RelPoseTransformer(PairAsyncPairMetricTransformer):

    def __init__(self, dataset_mode_config, metric_config, model_mode_eval_params, device):
        super().__init__(dataset_mode_config, metric_config, model_mode_eval_params, device)
        self.matcher = instantiate_matcher(self.model_mode_eval_params, device)

    def on_iteration_completed(self, engine, batch, endpoint):
        kp_desc1, kp_desc2, kp_desc_mask1, kp_desc_mask2 = get_kp_desc_and_kp_desc_mask(batch)
        kp2 = batch[eu.KP2]

        mm_desc_mask1, nn_desc_idx1 = self.matcher.match(kp_desc1, kp_desc2,
                                                         kp_desc_mask1, kp_desc_mask2)

        nn_kp2 = gather_kp(kp2.to(self.device), nn_desc_idx1).cpu()

        for i in range(kp2.shape[0]):
            try:
                self._process_pair_request(batch, nn_kp2, mm_desc_mask1, nn_desc_idx1,
                                           self.model_mode_eval_params, i)

            except Exception as e:
                print(e)

        try:
            return super().on_iteration_completed(engine, batch, endpoint)
        
        except Exception as e:
            print(e)

    def _reduce(self, engine, metric_name, values):
        if metric_name == eva_ns.R_mAA:
            acc = accuracy(values[eva_ns.R_ERR], self.metric_config[eva_ns.R_ERR_THRESH])

        elif metric_name == eva_ns.T_mAA:
            acc = accuracy(values[eva_ns.T_ERR], self.metric_config[eva_ns.T_ERR_THRESH])

        elif metric_name == eva_ns.HCR_mAA:
            acc = accuracy(values[eva_ns.HCR_ERR], self.metric_config[eva_ns.HCR_ERR_THRESH])

        else:
            raise NotImplementedError

        mAA = acc.mean()

        return {metric_name: mAA}


class HTuneRelPoseLoweRatioTransformer(PairAsyncPairMetricTransformer):

    def __init__(self, dataset_mode_config, metric_config, model_mode_eval_params, device):
        super().__init__(dataset_mode_config, metric_config, model_mode_eval_params, device)
        self.lowe_ratio_range = get_sweep_range(model_mode_eval_params.tuner.matcher.lowe_ratio)

    def on_iteration_completed(self, engine, batch, endpoint):
        kp_desc1, kp_desc2, kp_desc_mask1, kp_desc_mask2 = get_kp_desc_and_kp_desc_mask(batch)
        kp2 = batch[eu.KP2]

        lr_sweep_outputs = []

        for lr in self.lowe_ratio_range:
            model_mode_eval_paramsi = copy.deepcopy(self.model_mode_eval_params)
            model_mode_eval_paramsi.matcher.lowe_ratio = float(lr)
            model_mode_eval_paramsi.estimator.id = f"{lr:.3f}"

            matcher = instantiate_matcher(model_mode_eval_paramsi, self.device)

            mm_desc_mask1, nn_desc_idx1 = matcher.match(kp_desc1, kp_desc2,
                                                        kp_desc_mask1, kp_desc_mask2)

            nn_kp2 = gather_kp(kp2.to(self.device), nn_desc_idx1).cpu()

            lr_sweep_outputs.append((nn_kp2, mm_desc_mask1.cpu(), nn_desc_idx1.cpu(),
                                     model_mode_eval_paramsi))

        for i in range(kp2.shape[0]):
            for lr_sweep_output in lr_sweep_outputs:
                self._process_pair_request(batch,
                                           lr_sweep_output[0], lr_sweep_output[1], lr_sweep_output[2],
                                           lr_sweep_output[3], i)

        return super().on_iteration_completed(engine, batch, endpoint)

    def _reduce(self, engine, metric_name, values):
        range_len = len(self.lowe_ratio_range)

        if metric_name == eva_ns.R_mAA:
            acc = accuracy(values[eva_ns.R_ERR].reshape(-1, range_len),
                           self.metric_config[eva_ns.R_ERR_THRESH])

        elif metric_name == eva_ns.T_mAA:
            acc = accuracy(values[eva_ns.T_ERR].reshape(-1, range_len),
                           self.metric_config[eva_ns.T_ERR_THRESH])

        elif metric_name == eva_ns.HCR_mAA:
            acc = accuracy(values[eva_ns.HCR_ERR].reshape(-1, range_len),
                           self.metric_config[eva_ns.HCR_ERR_THRESH])

        else:
            raise NotImplementedError(metric_name)

        mAA = acc.mean(axis=0)

        return {metric_name: mAA,
                eva_ns.LOWE_RATIO: self.lowe_ratio_range}


class HTuneRelPoseInlThreshTransformer(PairAsyncPairMetricTransformer):

    def __init__(self, dataset_mode_config, metric_config, model_mode_eval_params, device):
        super().__init__(dataset_mode_config, metric_config, model_mode_eval_params, device)
        self.matcher = instantiate_matcher(self.model_mode_eval_params, self.device)

        if self.model_mode_eval_params.estimator.name in est_ns.TWO_VIEW_ESTIMATORS:
            self.inl_thresh_range = get_sweep_range(self.model_mode_eval_params.tuner.estimator.inl_thresh)

        else:
            self.inl_thresh_range = get_sweep_range(self.model_mode_eval_params.tuner.estimator.two_view.inl_thresh)

    def on_iteration_completed(self, engine, batch, endpoint):
        kp_desc1, kp_desc2, kp_desc_mask1, kp_desc_mask2 = get_kp_desc_and_kp_desc_mask(batch)
        kp2 = batch[eu.KP2]

        mm_desc_mask1, nn_desc_idx1 = self.matcher.match(kp_desc1, kp_desc2,
                                                         kp_desc_mask1, kp_desc_mask2)

        nn_kp2 = gather_kp(kp2.to(self.device), nn_desc_idx1).cpu()

        for i in range(kp2.shape[0]):
            for it in self.inl_thresh_range:
                model_mode_eval_paramsi = copy.deepcopy(self.model_mode_eval_params)
                model_mode_eval_paramsi.estimator.id = f"{it:.3f}"

                if self.model_mode_eval_params.estimator.name in est_ns.TWO_VIEW_ESTIMATORS:
                    model_mode_eval_paramsi.estimator.inl_thresh = float(it)

                else:
                    model_mode_eval_paramsi.estimator.two_view.inl_thresh = float(it)

                self._process_pair_request(batch, nn_kp2, mm_desc_mask1, nn_desc_idx1,
                                           model_mode_eval_paramsi, i)

        return super().on_iteration_completed(engine, batch, endpoint)

    def _reduce(self, engine, metric_name, values):
        range_len = len(self.inl_thresh_range)

        if metric_name == eva_ns.R_mAA:
            acc = accuracy(values[eva_ns.R_ERR].reshape(-1, range_len),
                           self.metric_config[eva_ns.R_ERR_THRESH])

        elif metric_name == eva_ns.T_mAA:
            acc = accuracy(values[eva_ns.T_ERR].reshape(-1, range_len),
                           self.metric_config[eva_ns.T_ERR_THRESH])

        elif metric_name == eva_ns.HCR_mAA:
            acc = accuracy(values[eva_ns.HCR_ERR].reshape(-1, range_len),
                           self.metric_config[eva_ns.HCR_ERR_THRESH])

        else:
            raise NotImplementedError(metric_name)

        mAA = acc.mean(axis=0)

        return {metric_name: mAA,
                eva_ns.INL_THRESH: self.inl_thresh_range}


"""
Legacy code
"""

# r_err, t_err = [], []
#
# estimator = instantiate_rel_pose_estimator_from_config(self.model_mode_eval_params, None)
#
# kp1 = kp1.numpy()
#
# for lr in self.lowe_ratio_range:
#     matcher = instantiate_matcher(self.model_mode_eval_params.matcher, lr, self.device)
#
#     mm_desc_mask1, nn_desc_idx1 = matcher.match(kp_desc1, kp_desc2,
#                                                 kp_desc_mask1, kp_desc_mask2)
#
#     nn_kp2 = gather_kp(kp2, nn_desc_idx1.cpu()).numpy()
#     mm_desc_mask1 = mm_desc_mask1.cpu().numpy()
#
#     r_erri, t_erri, _ = relative_pose_error(kp1, nn_kp2,
#                                             mm_desc_mask1,
#                                             intrinsics1, intrinsics2,
#                                             extrinsics1, extrinsics2,
#                                             estimator)
#
#     r_err.append(r_erri)
#     t_err.append(t_erri)
#
# it_values[eva_ns.R_ERR] = np.stack(r_err, axis=1)
# it_values[eva_ns.T_ERR] = np.stack(t_err, axis=1)
#
# return it_values

# scene_namei = batch[du.SCENE_NAME][i]
# image_name1i, image_name2i = batch[du.IMAGE_NAME1][i], batch[du.IMAGE_NAME2][i]
#
# image_name_no_ext1i = os.path.splitext(os.path.split(image_name1i)[1])[0]
# image_name_no_ext2i = os.path.splitext(os.path.split(image_name2i)[1])[0]
#
# id = model_mode_eval_params.estimator.get(eva_ns.ID)
# rec_id = HTUNE_REC_ID.format(scene_namei, image_name_no_ext1i, image_name_no_ext2i, id) \
#     if id is not None else REC_ID.format(scene_namei, image_name_no_ext1i, image_name_no_ext2i)
#
# estimator = COLMAPEstimator.from_config(rec_id, model_mode_eval_params)
#
# pair_data = PairData(self.dataset_mode_config.dataset_path,
#                      image_name1i, image_name2i,
#                      estimator.rec_path)
#
# request = COLMAPRelPoseRequest(self.entity_id, self.output_keys, estimator, pair_data)
# request.update(batch, nn_kp2, mm_desc_mask1, nn_desc_idx1, i)
#
# self._submit_request(request)
