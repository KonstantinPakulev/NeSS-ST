import os

import source.core.criterion as cu
import source.core.evaluation as evu
import source.core.namespace as c_ns
import source.datasets.base.utils as du
import source.evaluation.namespace as eva_ns
import source.pose.estimators.namespace as est_ns

from ignite.contrib.handlers.custom_events import Events

from source.core.experiment import init_device
from source.core.evaluation import TransformerHandler
from source.evaluation.transformers import FeaturesTransformer, TimeTransformer
from source.evaluation.classical.transformers import RepTransformer, MMATransformer
from source.evaluation.rel_pose.transformers import RelPoseTransformer, HTuneRelPoseLoweRatioTransformer, \
    HTuneRelPoseInlThreshTransformer
from source.evaluation.bag_rel_pose.transformers import BagRelPoseTransformer, HTuneBagRelPoseAbsPoseInlThreshTransformer
from source.evaluation.visual_localization.transformers import AachenVisualLocalizationTransformer
from source.evaluation.logging import create_features_dir, save_features_as_h5py, \
    log_to_tensorboard, save_eval_as_csv, save_htune_as_csv, save_visual_localization_eval

METRICS = 'metrics'
DEVICE = 'device'


def bind_ignite_transformers(engine, device,
                             dataset_mode_config, dataset_mode_eval_config,
                             model_mode_eval_params,
                             config):
    if dataset_mode_eval_config is None:
        return

    metrics_config = dataset_mode_eval_config.get(METRICS)

    if metrics_config is None:
        return

    if DEVICE in dataset_mode_eval_config:
        eval_device = init_device(dataset_mode_eval_config.device)

    else:
        eval_device = device

    entity_id = dataset_mode_config.get(du.ENTITY_ID)

    metric_log_iter = dataset_mode_eval_config.get(c_ns.METRIC_LOG_ITER)

    for metric_name, v in metrics_config.items():
        if metric_name == eva_ns.FEATURES:
            transformer = FeaturesTransformer(v, model_mode_eval_params)

        elif metric_name == eva_ns.REL_POSE:
            estimator_name = model_mode_eval_params.estimator.name

            if estimator_name in [est_ns.F_PYDEGENSAC,
                                  est_ns.E_PYOPENGV,
                                  est_ns.COLMAP,
                                  est_ns.H_PYDEGENSAC,
                                  est_ns.H_OPENCV]:
                transformer = RelPoseTransformer(dataset_mode_config, v, model_mode_eval_params, eval_device)

            else:
                raise ValueError(f"No such estimator: {estimator_name}")

        elif metric_name == eva_ns.BAG_REL_POSE:
            estimator_name = model_mode_eval_params.estimator.name
            
            if estimator_name == est_ns.COLMAP:
                transformer = BagRelPoseTransformer(dataset_mode_config, v, model_mode_eval_params, eval_device)

            else:
                raise ValueError(f"No such estimator: {estimator_name}")

        elif metric_name == eva_ns.VISUAL_LOCALIZATION:
            estimator_name = model_mode_eval_params.estimator.name

            if estimator_name == est_ns.COLMAP:
                dataset_name = config.datasets.dataset_name

                if dataset_name == du.AACHEN:
                    transformer = AachenVisualLocalizationTransformer(dataset_mode_config, model_mode_eval_params, eval_device)

                else:
                    raise ValueError(f"No such dataset: {dataset_name}")

            else:
                raise ValueError(f"No such estimator: {estimator_name}")

        elif metric_name == eva_ns.HTUNE:
            evaluation_task = config.datasets.evaluation_task
            param = v.param

            if evaluation_task == eva_ns.REL_POSE:
                if param == eva_ns.LOWE_RATIO:
                    transformer = HTuneRelPoseLoweRatioTransformer(dataset_mode_config, v,
                                                                   model_mode_eval_params, device)

                elif param == eva_ns.INL_THRESH:
                    transformer = HTuneRelPoseInlThreshTransformer(dataset_mode_config, v,
                                                                   model_mode_eval_params, device)

                else:
                    raise ValueError(param)

            elif evaluation_task == eva_ns.BAG_REL_POSE:
                if param == eva_ns.INL_THRESH:
                    transformer = HTuneBagRelPoseAbsPoseInlThreshTransformer(dataset_mode_config, v,
                                                                             model_mode_eval_params, device)

                else:
                    raise ValueError(param)

            else:
                raise ValueError(f"Unknown problem: {evaluation_task}")

        elif metric_name == eva_ns.MMA:
            transformer = MMATransformer(entity_id, v, model_mode_eval_params, eval_device)

        elif metric_name == eva_ns.REP:
            transformer = RepTransformer(entity_id, v, model_mode_eval_params, eval_device)

        elif metric_name == eva_ns.TIME:
            transformer = TimeTransformer()

        else:
            raise ValueError(f"No such metric: {metric_name}")

        TransformerHandler(transformer, metric_log_iter).attach(engine, [metric_name])


def bind_ignite_logging(data_engine, state_engine, cfg_mode,
                        dataset_mode_config, dataset_mode_eval_config,
                        config):
    if dataset_mode_eval_config is None:
        return

    loss_log_iter = dataset_mode_eval_config.get(c_ns.LOSS_LOG_ITER)

    if loss_log_iter is not None:
        log_to_tensorboard(state_engine, data_engine,
                           cfg_mode, loss_log_iter, [c_ns.LOSS, c_ns.IND])

    metrics_config = dataset_mode_eval_config.get(METRICS)

    if metrics_config is None:
        return

    metric_log_iter = dataset_mode_eval_config.get(c_ns.METRIC_LOG_ITER)

    for metric_name, v in metrics_config.items():
        if metric_name == eva_ns.FEATURES:
            @data_engine.on(Events.EPOCH_STARTED)
            def on_epoch_started(engine):
                create_features_dir(config.datasets.get(eva_ns.BACKEND))

            @data_engine.on(Events.ITERATION_COMPLETED)
            def on_iteration_completed(engine):
                save_features_as_h5py(engine.state.metrics,
                                      config.datasets.get(eva_ns.BACKEND),
                                      v.get(evu.OUTPUT_KEYS, []) if v is not None else [])

        elif metric_name in [eva_ns.REL_POSE,
                             eva_ns.BAG_REL_POSE,
                             eva_ns.REP,
                             eva_ns.MMA]:
            if v.reduce:
                if metric_log_iter != -1:
                    raise ValueError(metric_log_iter)

                log_to_tensorboard(state_engine, data_engine,
                                   cfg_mode, metric_log_iter, v.output_keys)

            else:
                @data_engine.on(Events.EPOCH_COMPLETED)
                def on_epoch_completed(engine):
                    save_eval_as_csv(dataset_mode_config.entity_id,
                                     v.output_keys,
                                     engine.state.metrics,
                                     config.experiment.get(eva_ns.EVAL_TAG, eva_ns.DEFAULT_EVAL_TAG))

        elif metric_name == eva_ns.VISUAL_LOCALIZATION:
            @data_engine.on(Events.EPOCH_COMPLETED)
            def on_epoch_completed(engine):
                save_visual_localization_eval(engine.state.metrics,
                                              config.datasets.dataset_name,
                                              dataset_mode_config,
                                              config.models.model_name,
                                              config.experiment.get(eva_ns.EVAL_TAG, eva_ns.DEFAULT_EVAL_TAG))

        elif metric_name in [eva_ns.HTUNE]:
            if metric_log_iter != -1:
                raise ValueError(metric_log_iter)

            if v.reduce:
                @data_engine.on(Events.EPOCH_COMPLETED)
                def on_epoch_completed(engine):
                    save_htune_as_csv(v.output_keys,
                                      engine.state.metrics,
                                      v.param,
                                      config.experiment.get(eva_ns.EVAL_TAG, eva_ns.DEFAULT_EVAL_TAG))

            else:
                raise NotImplementedError("No implementation")


"""
Legacy code
"""

# from source.evaluation.homography.transformers import HomographyTransformer, HTuneHomographyLoweRatioTransformer, \
#     HTuneHomographyInlThreshTransformer

# elif metric_name == SFM_LOC:
#     transformer = SfMLocalizationTransformer(v, model_mode_eval_params, device)

# elif estimator_name == est_ns.H_OPENCV:
#     raise NotImplementedError()
# transformer = HomographyTransformer(entity_id, v, model_mode_eval_params, device)

# elif estimator_name == est_ns.COLMAP:
# transformer = HTuneBagSfMLoweRatioTransformer(v, model_mode_eval_params, device)

# elif estimator_name == est_ns.H_OPENCV:
#     raise NotImplementedError()
# transformer = HTuneHomographyLoweRatioTransformer(entity_id, v, model_mode_eval_params, device)

# elif metric_name == c_trs.MMA or \
#      metric_name == rp_trs.H:

# elif estimator_name == est_ns.H_OPENCV:
#     raise NotImplementedError()
# transformer = HTuneHomographyInlThreshTransformer(entity_id, v, model_mode_eval_params, device)
#
#
# @data_engine.on(Events.EPOCH_COMPLETED)
# def on_epoch_completed(engine):
#     save_eval_as_csv(os.getcwd(), engine.state.metrics,
#                      dataset_mode_config.entity_id,
#                      v.output_keys)

# metric_name == rp_trs.HTUNE_SHI_THRESH or \
# metric_name == rp_trs.HTUNE_VAR_THRESH:
# save_map_as_npy
# from source.evaluation.odometry_map.transformers import ODOMETRY_MAP, OdometryMapTransformer

# elif metric_name == rp_trs.HTUNE_VAR_THRESH:
#     transformer = HTuneVarThreshTransformer(entity_id, v, model_mode_eval_params, device)
#
# elif metric_name == rp_trs.HTUNE_SHI_THRESH:
#     transformer = HTuneShiThreshTransformer(entity_id, v, model_mode_eval_params, device)

#
# elif metric_name == ODOMETRY_MAP:
#     if metric_log_iter != -1:
#         raise ValueError(metric_log_iter)
#
#     @data_engine.on(Events.EPOCH_COMPLETED)
#     def on_epoch_completed(engine):
#         save_map_as_npy(os.getcwd(), engine.state.metrics, metric_config.output_keys)
#
# else:
#     raise ValueError(f"No such metric: {metric_name}")


# elif metric_name == MMA:
#     transformer = MMATransformer(entity_id, metric_config, model_mode_eval_params, eval_device)

# if num_branches == 2:
#     warp_points_endpoint(batch, endpoint, device)

# if metric_name == meu.REP:
#     raise NotImplementedError
    # px_thresh = metric.px_thresh
    # transformer = meu.RepTransformer(detailed, entity_id, outputs, eval_device, px_thresh)

# elif metric_name == meu.MS:
#     raise NotImplementedError
    # px_thresh = metric.px_thresh
    # transformer = meu.MSTransformer(detailed, entity_id, outputs, eval_device, px_thresh)

# elif metric_name == meu.MMA:
#     raise NotImplementedError
    # px_thresh = metric.px_thresh
    # transformer = meu.MMATransformer(detailed, entity_id, outputs, eval_device, px_thresh)

# EVALUATION = 'evaluation'

# GPU = 'gpu'

# NMS_KERNEL_SIZE = 'nms_kernel_size'
# KR_SIZE = 'kr_size'
# SCORE_THRESH = 'score_thresh'
# SAL_THRESH = 'sal_thresh'
# CONF_THRESH = 'conf_thresh'
# SIM_MEASURE = 'sim_measure'
# LOWE_RATIO = 'lowe_ratio'
# INL_THRESH = 'inl_thresh'

# TOPK = "topk"

# PX_THRESH = 'px_thresh'
# R_ACC_THRESH = 'r_acc_thresh'
# T_ACC_THRESH = 't_acc_thresh'

# METRICS = 'metrics'
# NAME = 'name'
# OUTPUTS = 'outputs'
# NUM_REPEATS = 'num_repeats'
# THRESHOLD = 'threshold'
# DETAILED = 'detailed'

# LOGS = 'logs'

# elif log == lu.AACHEN_INFERENCE:
#
#
# @data_engine.on(Events.ITERATION_COMPLETED)
# def on_iteration_completed(engine):
#     raise NotImplementedError
#     # save_aachen_inference(engine, dataset_mode_config.dataset_root, config.models.model_name)
#
# elif log == lu.HPATCHES_INFERENCE:
#
#
# @data_engine.on(Events.ITERATION_COMPLETED)
# def on_iteration_completed(engine):
#     save_hpatches_inference(engine, dataset_mode_config.dataset_root, config.models.model_name)
#
# elif log == lu.IMB_INFERENCE:
# logger = IMBLogger(os.getcwd(), dataset_mode_config.sub_dataset)
#
# logger.attach_default(data_engine)

# def bind_pre_process_metrics_ignite(engine, evaluation_config):
    # metrics = evaluation_config.get(METRICS)
    #
    # if metrics is None:
    #     return

    # metric_log_iter = evaluation_config.get(METRIC_LOG_ITER)

    # for metric in metrics:
    #     metric_name = metric[NAME]
    #
    #     if metric_name == meu.FORWARD_TIME:
    #         @engine.on(Events.ITERATION_STARTED)
    #         def on_iteration_started(engine):
    #             engine.state.bundle = {}
    #             start_time_measurement(engine.state.bundle)
    #
    #         @engine.on(Events.ITERATION_COMPLETED)
    #         def on_iteration_completed(engine):
    #             end_time_measurement(engine.state.bundle, engine.state.output[1])
    #
    #         transformer = lambda output: output[1][meu.FORWARD_TIME]
    #
    #         AveragePeriodicMetric(transformer, metric_log_iter).attach(engine, metric_name)

# def attach_desc_analyse(engine, device, modes, forward_mode, model_config, dataset_config, metric_config):
#     if modes == lp.ANALYZE:
#         px_thresh = metric_config[meu.PX_THRESH]
#         sim_measure = model_config[mu.SIM_MEASURE]
#
#         DetailedMetric(SimRatioTransformer(device, px_thresh, sim_measure), 1).attach(engine, meu.MS)

# from .source.evaluation.metrics_utils import start_time_measurement, end_time_measurement

# elif metric_name == rp_trs.ODOMETRY_MAP:
#     transformer = OdometryMapTransformer(entity_id, v, model_mode_eval_params, eval_device)

