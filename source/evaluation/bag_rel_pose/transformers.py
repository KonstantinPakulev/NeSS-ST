import numpy as np
import copy

from source.datasets.base import utils as du
import source.utils.endpoint_utils as eu
import source.evaluation.namespace as eva_ns
import source.evaluation.bag_rel_pose.namespace as brp_ns
import source.pose.estimators.namespace as est_ns

from source.core.evaluation import AsyncPairMetricTransformer, submit_request_impl
from source.pose.matchers.factory import instantiate_matcher
from source.pose.matchers.utils import gather_kp
from source.pose.estimators.fund_mat import FundMatEstimator
from source.pose.estimators.colmap.estimator import COLMAPEstimator
from source.evaluation.metrics import accuracy
from source.evaluation.bag_rel_pose.utils import BagData, FRequest, BagRelPoseRequest
from source.evaluation.utils import get_kp_desc_and_kp_desc_mask, get_sweep_range


REC_ID = "{}_{}"
HTUNE_REC_ID = "{}_{}_{}"


class BagAsyncPairMetricTransformer(AsyncPairMetricTransformer):

    def __init__(self, dataset_mode_config, metric_config, model_mode_eval_params, device):
        super().__init__(dataset_mode_config.get(du.ENTITY_ID), metric_config)
        self.dataset_mode_config = dataset_mode_config
        self.model_mode_eval_params = model_mode_eval_params
        self.device = device

        self.pending_request_dict = {}

    def _get_splits_and_bag_data(self, batch):
        scene_name = batch[du.SCENE_NAME]
        bag_id = batch[du.BAG_ID]

        rec_ids2submit = [rec_id for rec_id in self.pending_request_dict.keys()
                          if REC_ID.format(scene_name[0], bag_id[0]) not in rec_id]

        for rec_id in rec_ids2submit:
            request = self.pending_request_dict[rec_id]

            self._submit_request(request)

            del self.pending_request_dict[rec_id]

        splits = get_splits(bag_id)
        bag_data = {}

        for start in splits[:-1]:
            scene_namei = scene_name[start]
            bag_idi = bag_id[start]

            if start == 0 and len([rec_id for rec_id in self.pending_request_dict.keys()
                                   if REC_ID.format(scene_namei, bag_idi) in rec_id]) != 0:
                continue

            bag_data[bag_idi] = BagData(self.dataset_mode_config.dataset_path, scene_namei, bag_idi)

        return splits, bag_data

    def _process_pair_requests(self, batch, bundle, two_view_eval_params):
        kp2 = batch[eu.KP2]

        estimator_name = two_view_eval_params.estimator.name

        if estimator_name == est_ns.F_PYDEGENSAC:
            estimator = FundMatEstimator.from_config(two_view_eval_params)

        else:
            raise ValueError(f'No such estimator {estimator_name}')

        nn_kp2 = gather_kp(kp2.to(self.device), bundle[brp_ns.NN_DESC_IDX1]).cpu()

        bundle[brp_ns.NN_KP2] = nn_kp2

        async_results = []

        for i in range(kp2.shape[0]):
            request = FRequest(estimator)
            request.update(batch, bundle, i)

            async_results.append(self.pool.apply_async(submit_request_impl, (request,)))

        F = []
        inl_mask = []

        for ar in async_results:
            resulti = ar.get()

            F.append(resulti[brp_ns.F_ESTIMATE])
            inl_mask.append(resulti[brp_ns.INL_MASK])

        F = np.stack(F)
        inl_mask =np.stack(inl_mask)

        bundle[brp_ns.F_ESTIMATE] = F
        bundle[brp_ns.INL_MASK] = inl_mask

        del bundle[brp_ns.NN_KP2]

    def _process_bag_request(self, batch, bundle,
                             rec_id, bag_data, model_mode_eval_params,
                             start, end):
        bag_id = batch[du.BAG_ID]
        bag_idi = bag_id[start]

        if start == 0:
            if rec_id in self.pending_request_dict:
                request = self.pending_request_dict[rec_id]
                request.update(batch, bundle, start, end)

                if end < len(bag_id):
                    self._submit_request(request)

                    del self.pending_request_dict[rec_id]

            else:
                request = BagRelPoseRequest(self.entity_id,
                                            self.output_keys,
                                            COLMAPEstimator.from_config(rec_id, model_mode_eval_params),
                                            bag_data[bag_idi])
                request.update(batch, bundle, start, end)

                if end < len(bag_id):
                    self._submit_request(request)

                else:
                    self.pending_request_dict[rec_id] = request

        elif end == len(bag_id):
            request = BagRelPoseRequest(self.entity_id,
                                        self.output_keys,
                                        COLMAPEstimator.from_config(rec_id, model_mode_eval_params),
                                        bag_data[bag_idi])
            request.update(batch, bundle, start, end)

            self.pending_request_dict[rec_id] = request

        else:
            request = BagRelPoseRequest(self.entity_id,
                                        self.output_keys,
                                        COLMAPEstimator.from_config(rec_id, model_mode_eval_params),
                                        bag_data[bag_idi])
            request.update(batch, bundle, start, end)

            self._submit_request(request)

    def on_before_epoch_completed(self, engine):
        for v in self.pending_request_dict.values():
            self._submit_request(v)

        self.pending_request_dict.clear()

        return super().on_before_epoch_completed(engine)


class BagRelPoseTransformer(BagAsyncPairMetricTransformer):

    def __init__(self, dataset_mode_config, metric_config, model_mode_eval_params, device):
        super().__init__(dataset_mode_config, metric_config, model_mode_eval_params, device)

        self.matcher = instantiate_matcher(self.model_mode_eval_params, self.device)

    def on_iteration_completed(self, engine, batch, endpoint):
        scene_name = batch[du.SCENE_NAME]
        bag_id = batch[du.BAG_ID]
        kp_desc1, kp_desc2, kp_desc_mask1, kp_desc_mask2 = get_kp_desc_and_kp_desc_mask(batch)

        splits, bag_data = self._get_splits_and_bag_data(batch)

        mm_desc_mask1, nn_desc_idx1 = self.matcher.match(kp_desc1, kp_desc2,
                                                         kp_desc_mask1, kp_desc_mask2)

        two_view_eval_params = self.model_mode_eval_params.estimator.two_view

        bundle = {brp_ns.MM_DESC_MASK1: mm_desc_mask1,
                  brp_ns.NN_DESC_IDX1: nn_desc_idx1}

        if eva_ns.ESTIMATOR in two_view_eval_params:
            self._process_pair_requests(batch, bundle, two_view_eval_params)

        for start, end in zip(splits[:-1], splits[1:]):
            rec_id = REC_ID.format(scene_name[start], bag_id[start])

            self._process_bag_request(batch, bundle,
                                      rec_id, bag_data, self.model_mode_eval_params,
                                      start, end)

        return super().on_iteration_completed(engine, batch, endpoint)


class HTuneBagRelPoseAbsPoseInlThreshTransformer(BagAsyncPairMetricTransformer):

    def __init__(self, dataset_mode_config, metric_config, model_mode_eval_params, device):
        super().__init__(dataset_mode_config, metric_config, model_mode_eval_params, device)
        self.matcher = instantiate_matcher(self.model_mode_eval_params, self.device)

        self.inl_thresh_range = get_sweep_range(self.model_mode_eval_params.estimator.abs_pose.htune_inl_thresh)

        self.r_err_thresh = metric_config[eva_ns.R_ERR_THRESH]
        self.t_err_thresh = metric_config[eva_ns.T_ERR_THRESH]

    def on_iteration_completed(self, engine, batch, endpoint):
        scene_name = batch[du.SCENE_NAME]
        bag_id = batch[du.BAG_ID]
        kp_desc1, kp_desc2, kp_desc_mask1, kp_desc_mask2 = get_kp_desc_and_kp_desc_mask(batch)

        splits, bag_data = self._get_splits_and_bag_data(batch)

        mm_desc_mask1, nn_desc_idx1 = self.matcher.match(kp_desc1, kp_desc2,
                                                         kp_desc_mask1, kp_desc_mask2)

        bundle = {brp_ns.MM_DESC_MASK1: mm_desc_mask1,
                  brp_ns.NN_DESC_IDX1: nn_desc_idx1}

        two_view_eval_params = self.model_mode_eval_params.estimator.two_view

        if eva_ns.ESTIMATOR in two_view_eval_params:
            self._process_pair_requests(batch, bundle, two_view_eval_params)

        for start, end in zip(splits[:-1], splits[1:]):
            for it in self.inl_thresh_range:
                idi = f"{it:.3f}"

                # TODO. check if "it" is properly used

                model_mode_eval_paramsi = copy.deepcopy(self.model_mode_eval_params)
                model_mode_eval_paramsi.estimator.id = idi
                model_mode_eval_paramsi.estimator.abs_pose.inl_thresh = float(it)

                rec_id = HTUNE_REC_ID.format(scene_name[start], bag_id[start], idi)

                self._process_bag_request(batch, bundle,
                                          rec_id, bag_data, model_mode_eval_paramsi,
                                          start, end)

        return super().on_iteration_completed(engine, batch, endpoint)

    def _reduce(self, engine, metric_name, values):
        range_len = len(self.inl_thresh_range)

        if metric_name == eva_ns.R_mAA:
            acc = accuracy(values[eva_ns.R_ERR].reshape(-1, range_len), self.r_err_thresh)

        elif metric_name == eva_ns.T_mAA:
            acc = accuracy(values[eva_ns.T_ERR].reshape(-1, range_len), self.t_err_thresh)

        else:
            raise NotImplementedError

        mAA = acc.mean(axis=0)

        return {metric_name: mAA,
                eva_ns.INL_THRESH: self.inl_thresh_range}


"""
Support utils
"""


def get_splits(bag_id):
    splits = []
    prev_bag_id = None

    for i, bag_idi in enumerate(bag_id):
        if prev_bag_id is None or prev_bag_id != bag_idi:
            splits.append(i)

        prev_bag_id = bag_idi

    splits.append(len(bag_id))

    return splits


"""
Legacy code
"""

# import torch
# print('mem before: ', torch.cuda.mem_get_info(0))
# print('mem before:', torch.cuda.memory_summary(0))
# print('mem after: ', torch.cuda.mem_get_info(0))
# print('mem after:', torch.cuda.memory_summary(0))

# class HTuneBagSfMLoweRatioTransformer(BagAsyncPairMetricTransformer):
#
#     def __init__(self, metric_config, model_mode_eval_params, device):
#         super().__init__(None, metric_config)
#         self.device = device
#
#         self.model_mode_eval_params = model_mode_eval_params
#
#         self.lowe_ratio_range = get_sweep_range(model_mode_eval_params.htune_lowe_ratio)
#
#         self.r_err_thresh = metric_config[ns.R_ERR_THRESH]
#         self.t_err_thresh = metric_config[ns.T_ERR_THRESH]
#
#     def on_iteration_completed(self, engine, batch, endpoint):
#         scene_name = batch[du.SCENE_NAME]
#         bag_id = batch[du.BAG_ID]
#
#         kp1, kp2 = batch[eu.KP1], batch[eu.KP2]
#         kp_desc1, kp_desc2 = batch[eu.KP_DESC1], batch[eu.KP_DESC2]
#         kp_desc_mask1, kp_desc_mask2 = (kp1 != ns.INVALID_KP).prod(-1).bool(), (kp2 != ns.INVALID_KP).prod(-1).bool()
#
#         splits, bag_data = self._get_splits_and_bag_data(batch)
#
#         lr_mask_and_matches = []
#
#         for lr in self.lowe_ratio_range:
#             matcher = instantiate_matcher(self.model_mode_eval_params, self.device)
#
#             mm_desc_mask1, nn_desc_idx1 = matcher.match(kp_desc1, kp_desc2,
#                                                         kp_desc_mask1, kp_desc_mask2)
#
#             lr_mask_and_matches.append((mm_desc_mask1.cpu(), nn_desc_idx1.cpu()))
#
#         for start, end in zip(splits[:-1], splits[1:]):
#             for i, lr in enumerate(self.lowe_ratio_range):
#                 rec_id = f"{scene_name[start]}_{bag_id[start]}_{lr:.3f}"
#
#                 self._process_bag_request(rec_id, bag_data,
#                                           batch,
#                                           lr_mask_and_matches[i][0], lr_mask_and_matches[i][1],
#                                           start, end)
#
#         return super().on_iteration_completed(engine, batch, endpoint)
#
#     def _reduce(self, engine, metric_name, values):
#         num_thresh = len(self.lowe_ratio_range)
#
#         bag_id = values[du.BAG_ID].reshape(-1, num_thresh)[:, 0]
#
#         if metric_name == ns.R_mAA:
#             metric_value = values[ns.R_ERR].reshape(-1, num_thresh)
#             bag_groups_mAA = bag_grouped_mAA(bag_id, metric_value, self.r_err_thresh)
#
#         elif metric_name == ns.T_mAA:
#             metric_value = values[ns.T_ERR].reshape(-1, num_thresh)
#             bag_groups_mAA = bag_grouped_mAA(bag_id, metric_value, self.t_err_thresh)
#
#         else:
#             raise NotImplementedError
#
#         mAA = np.stack(list(bag_size_grouped_mAA(bag_groups_mAA).values())).mean(axis=0)
#
#         return {metric_name: mAA,
#                 ns.LOWE_RATIO: self.lowe_ratio_range}

