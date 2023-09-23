import os
import warnings
import numbers
import shutil
import h5py
import numpy as np
import pandas as pd

import source.datasets.aachen.preprocessing.annotations as aa_an
import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu
import source.evaluation.namespace as eva_ns

from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.contrib.handlers.custom_events import Events, CustomPeriodicEvent

EVAL_LOG_FILE = 'eval_log{}.csv'
HTUNE_LOG_FILE = 'htune_{}_log{}.csv'

AACHEN_EVAL_FILE_V1_0 = "Aachen_eval_{}{}.txt"
AACHEN_EVAL_FILE_V1_1 = "Aachen_v1_1_eval_{}{}.txt"


def create_features_dir(backend):
    features_dir_path = get_features_dir_path(backend is not None)

    if os.path.exists(features_dir_path):
        shutil.rmtree(features_dir_path)

    os.mkdir(features_dir_path)


def save_features_as_h5py(metrics, backend, output_keys):
    filepath = os.path.join(get_features_dir_path(backend is not None), f"{metrics[du.IMAGE_NAME]}.h5py")

    with h5py.File(filepath, 'w') as file:
        for key in [eu.KP, eu.KP_DESC] + output_keys:
            value = metrics[key]

            file.create_dataset(key, data=value)


def log_to_tensorboard(state_engine, data_engine,
                       cfg_mode, log_iter, output_keys):
    if not hasattr(state_engine, 'tb_logger'):
        setattr(state_engine, 'tb_logger', TensorboardLogger(log_dir=os.getcwd()))

        @state_engine.on(Events.COMPLETED)
        def on_completed(engine):
            state_engine.tb_logger.close()

    state_engine.tb_logger.attach(data_engine,
                                  MetricHandler(cfg_mode, state_engine, output_keys),
                                  event_name=create_tensorboard_event(data_engine, log_iter))


def save_eval_as_csv(entity_id, output_keys, metrics, eval_tag):
    keys_to_save = {}

    for key in entity_id + output_keys:
        value = metrics[key]

        if not isinstance(value, list) and len(value.shape) == 2:
            for i in range(value.shape[1]):
                keys_to_save[f"{key}_{i}"] = value[:, i]

        else:
            keys_to_save[key] = value

    eval_file = EVAL_LOG_FILE.format(eval_tag)
    pd.DataFrame.from_dict(keys_to_save).to_csv(os.path.join(os.getcwd(), eval_file))


def save_htune_as_csv(output_keys, metrics, htune_param, eval_tag):
    keys_to_save = {}

    for key in output_keys:
        value = metrics[key]

        for ti, vi in zip(metrics[htune_param], value):
            keys_to_save[f"{key}_{ti}"] = [vi]

    eval_file = HTUNE_LOG_FILE.format(htune_param, eval_tag)
    pd.DataFrame.from_dict(keys_to_save).to_csv(os.path.join(os.getcwd(), eval_file))


def save_visual_localization_eval(metrics, dataset_name, dataset_mode_config,
                                  model_name, eval_tag):
    if dataset_name == du.AACHEN:
        version = dataset_mode_config.version

        if version == aa_an.V1_0:
            eval_file = AACHEN_EVAL_FILE_V1_0.format(model_name, eval_tag)

        elif version == aa_an.V1_1:
            eval_file = AACHEN_EVAL_FILE_V1_1.format(model_name, eval_tag)

        else:
            raise ValueError(f"No such Aachen version: {version}")

        query_extrinsics = metrics[eva_ns.AACHEN_REC_ID.format(version)]

        with open(os.path.join(os.getcwd(), eval_file), 'w') as f:
            for k, v in query_extrinsics.items():
                f.write('%s %s\n' % (k, v))


"""
Support utils
"""


def create_tensorboard_event(data_engine, log_iter):
    if log_iter == -1:
        return Events.EPOCH_COMPLETED

    else:
        event = CustomPeriodicEvent(n_iterations=log_iter)

        event.attach(data_engine)

        return event._periodic_event_completed


class MetricHandler:

    def __init__(self, tag, state_engine, output_keys):
        self.tag = tag
        self.state_engine = state_engine
        self.output_keys = output_keys

    def __call__(self, engine, logger, event_name):
        metrics = self._get_metrics_data(engine)

        for key, value in metrics.items():
            if isinstance(value, numbers.Number) or\
                    isinstance(value, np.ndarray) and value.ndim == 0:
                logger.writer.add_scalar("{}/{}".format(self.tag, key),
                                         value,
                                         self.state_engine.state.iteration)

            elif isinstance(value, np.ndarray) and value.ndim == 1:
                for i, v in enumerate(value):
                    logger.writer.add_scalar("{}/{}/{}".format(self.tag, key, i),
                                             v,
                                             self.state_engine.state.iteration)

            else:
                warnings.warn("TensorboardLogger output_handler can not log "
                              "metrics value type {}".format(type(value)))

    def _get_metrics_data(self, engine):
        metrics = {}

        for key, value in engine.state.metrics.items():
            if next((ok for ok in self.output_keys if ok in key), None) is not None:
                metrics[key] = value

        return metrics


"""
Path utils
"""

def get_test_eval_log_path(test_dir, evaluation_task, method, backend,
                           eval_tag, rel_path=None):
    test_log_file_name = EVAL_LOG_FILE.format(eval_tag)

    if rel_path is None:
        return os.path.join(test_dir, evaluation_task,
                            method, backend, test_log_file_name)

    else:
        return os.path.join(test_dir, evaluation_task,
                            method, rel_path, backend, test_log_file_name)


def get_htune_eval_log_path(htune_dir, evaluation_task, method, backend,
                            htune_param, eval_tag, rel_path=None):
    htune_log_file_name = HTUNE_LOG_FILE.format(htune_param, eval_tag)

    if rel_path is None:
        return os.path.join(htune_dir, evaluation_task,
                            method, backend, htune_log_file_name)

    else:
        return os.path.join(htune_dir, evaluation_task,
                            method, rel_path, backend, htune_log_file_name)


def read_htune_eval_log(eval_log_path):
    htune_log = pd.read_csv(eval_log_path, index_col=[0])

    if eva_ns.HCR_mAA in htune_log.columns[0]:
        hcr_mAA = htune_log.filter(like=eva_ns.HCR_mAA, axis=1)

        thresh = [float(k.replace(f"{eva_ns.HCR_mAA}_", '')) for k in hcr_mAA.keys()]

        return np.array([hcr_mAA.to_numpy()[0]]), thresh

    elif eva_ns.R_mAA in htune_log.columns[0]:
        r_mAA = htune_log.filter(like=eva_ns.R_mAA, axis=1)
        t_mAA = htune_log.filter(like=eva_ns.T_mAA, axis=1)

        thresh = [float(k.replace(f"{eva_ns.R_mAA}_", '')) for k in r_mAA.keys()]

        return np.array([r_mAA.to_numpy()[0], t_mAA.to_numpy()[0]]), thresh


def get_features_dir_path(has_backend):
    if has_backend:
        return os.path.join(os.getcwd(), '..', '..', du.FEATURES)

    else:
        return os.path.join(os.getcwd(), du.FEATURES)

"""
Legacy code
"""

# if create_if_not_exists and \
#         not os.path.exists(features_dir_path):
#     os.mkdir(features_dir_path)

# query_image_list_path = os.path.join(dataset_path, ns.AACHEN_QUERIES_REL_PATH)
#
# with open(query_image_list_path) as f:
#     raw_queries = f.readlines()
#
# query_names = set()
#
# for raw_query in raw_queries:
#     raw_query = raw_query.strip('\n').split(' ')
#     query_name = raw_query[0]
#     query_names.add(query_name)
#
# with open(os.path.join(os.getcwd(), ns.FINAL_TXT_MODEL_DIR, ns.IMAGES_FILE)) as f:
#     raw_extrinsics = f.readlines()
#
# if version == ns.V1_0:
#     eval_file = AACHEN_EVAL_FILE_V1_0.format(model_name, eval_tag)
#
# elif version == ns.V1_1:
#     eval_file = AACHEN_EVAL_FILE_V1_1.format(model_name, eval_tag)
#
# else:
#     raise ValueError(version)
#
#
# for extrinsics in raw_extrinsics[4:: 2]:
#     extrinsics = extrinsics.strip('\n').split(' ')
#
#     image_name = extrinsics[-1]
#
#     if image_name in query_names:
#         f.write('%s %s\n' % (image_name.split('/')[-1], ' '.join(extrinsics[1: -2])))
#
# f.close()

# def save_map_as_npy(log_dir, metrics, output_keys):
#     for key in output_keys:
#         value = metrics[key]
#
#         file_path = os.path.join(log_dir, f"{key}.npy")
#         np.save(file_path, value)

# for e_id in entity_id:
#     metrics_to_save[e_id] = metrics[e_id]
#
# for metric_name in metrics_names:
#     metrics_to_save[metric_name] = metrics[metric_name]
#
# results = {}
#
# for k, v in metrics_to_save.items():

# SUMMARY_CSV = 'summary_csv'
# AACHEN_INFERENCE = 'aachen_inference'
# HPATCHES_INFERENCE = 'hpatches_inference'
# IMB_INFERENCE = 'imb_inference'

#
# def save_aachen_inference(data_engine, aachen_dataset_root, model_name):
#     batch, endpoint = data_engine.state.batch, data_engine.state.output
#
#     scene_name, image_name = batch.get(du.SCENE_NAME)[0], batch.get(du.IMAGE_NAME1)[0]
#
#     kp = endpoint[eu.KP1].cpu()
#     shift_scale = batch.get(du.SHIFT_SCALE1)
#
#     kp = revert_shift_scale(kp, shift_scale).squeeze().numpy()
#     kp_desc = endpoint[eu.KP_DESC1].cpu().squeeze().numpy()
#
#     file_path = os.path.join(aachen_dataset_root, scene_name, f"{image_name}.{model_name}.npz")
#     r_file_path = os.path.join(aachen_dataset_root, scene_name, f"{image_name}.{model_name}")
#
#     np.savez(file_path, keypoints=kp, descriptors=kp_desc)
#     os.rename(file_path, r_file_path)
#
#
# def save_hpatches_inference(data_engine, hpatches_dataset_root, model_name):
#     batch, endpoint = data_engine.state.batch, data_engine.state.output
#
#     scene_name = batch.get(du.SCENE_NAME)[0]
#     image1_name, image2_name = batch.get(du.IMAGE_NAME1)[0], batch.get(du.IMAGE_NAME2)[0]
#
#     kp1, kp2 = endpoint[eu.KP1], endpoint[eu.KP2]
#     shift_scale1, shift_scale2 = batch.get(du.SHIFT_SCALE1), batch.get(du.SHIFT_SCALE2)
#
#     kp1, kp2 = revert_shift_scale(kp1, shift_scale1).squeeze().numpy(), revert_shift_scale(kp2, shift_scale2).squeeze().numpy()
#     kp_desc1, kp_desc2 = endpoint[eu.KP_DESC1].cpu().squeeze().numpy(), endpoint[eu.KP_DESC2].cpu().squeeze().numpy()
#
#     file_path1 = os.path.join(hpatches_dataset_root, scene_name, f"{image1_name}.{model_name}.npz")
#     file_path2 = os.path.join(hpatches_dataset_root, scene_name, f"{image2_name}.{model_name}.npz")
#
#     r_file_path1 = os.path.join(hpatches_dataset_root, scene_name, f"{image1_name}.{model_name}")
#     r_file_path2 = os.path.join(hpatches_dataset_root, scene_name, f"{image2_name}.{model_name}")
#
#     np.savez(file_path1, keypoints=kp1, descriptors=kp_desc1)
#     np.savez(file_path2, keypoints=kp2, descriptors=kp_desc2)
#
#     os.rename(file_path1, r_file_path1)
#     os.rename(file_path2, r_file_path2)

# def print_metric(metric_name, image1_name, image2_name, metric, num_matches, num_vis_matches, px_thresh):
#     for b in range(len(image1_name)):
#         pair_name = image1_name[b] + " and " + image2_name[b]
#         print(f"Pair: {pair_name}")
#         print("-" * 66)
#
#         for i, thresh in enumerate(px_thresh):
#             print("\t" + f"Threshold {thresh} px")
#             print("\t" + "-" * 18)
#             if num_matches is None:
#                 print("\t" * 2 + f"{metric_name}: {metric[i][b]:.4}")
#             else:
#                 print(
#                     "\t" * 2 + f"{metric_name}: {metric[i][b]:.4} ({num_matches[i][b]}/{num_vis_matches[b]})")
#
#             print()
#
#         print()

# class IMBLogger(BaseLogger):
#
#     def __init__(self, log_dir, sub_dataset):
#         self.sub_dataset_path = os.path.join(log_dir, sub_dataset)
#
#         self.scene_kp = None
#         self.scene_desc = None
#
#     def attach_default(self, engine):
#         imb_handler = IMBHandler()
#
#         self.attach(engine, imb_handler, Events.EPOCH_STARTED)
#         self.attach(engine, imb_handler, Events.ITERATION_COMPLETED)
#         self.attach(engine, imb_handler, Events.EPOCH_COMPLETED)
#
#     def init_log(self):
#         if os.path.exists(self.sub_dataset_path):
#             shutil.rmtree(self.sub_dataset_path)
#
#         os.mkdir(self.sub_dataset_path)
#
#     def write(self, scene_name, image_name, shift_scale, kp, kp_desc, is_new):
#         if is_new:
#             self.close()
#
#             scene_path = os.path.join(self.sub_dataset_path, scene_name)
#             os.mkdir(scene_path)
#
#             self.scene_kp = h5py.File(os.path.join(scene_path, "keypoints.h5"), 'w')
#             self.scene_desc = h5py.File(os.path.join(scene_path, "descriptors.h5"), 'w')
#
#         key = os.path.splitext(image_name)[0]
#
#         kp = kp.cpu()
#         kp_desc = kp_desc.cpu().numpy()
#
#         kp = revert_shift_scale(kp, shift_scale).numpy()
#
#         self.scene_kp[key] = kp
#         self.scene_desc[key] = kp_desc
#
#     def close(self):
#         if self.scene_kp is not None:
#             self.scene_kp.close()
#             self.scene_desc.close()
#
#
# class IMBHandler:
#
#     def __init__(self):
#         self._prev_scene = None
#
#     def __call__(self, engine, logger, event_name):
#         if event_name == Events.EPOCH_STARTED:
#             logger.init_log()
#
#         elif event_name == Events.ITERATION_COMPLETED:
#             batch = engine.state.batch
#             endpoint = engine.state.output
#
#             scene_name, image_name, shift_scale = batch[du.SCENE_NAME], \
#                                                   batch[du.IMAGE_NAME1],\
#                                                   batch[du.SHIFT_SCALE1]
#
#             kp, kp_desc = endpoint[eu.KP1], endpoint[eu.KP_DESC1]
#
#             for scene_namei, image_namei, shift_scalei, kpi, kp_desci in\
#                 zip(scene_name, image_name, shift_scale, kp, kp_desc):
#
#                 is_new = self._prev_scene is None or scene_namei != self._prev_scene
#
#                 logger.write(scene_namei, image_namei, shift_scalei, kpi, kp_desci, is_new)
#
#                 self._prev_scene = scene_namei
#
#         elif event_name == Events.EPOCH_COMPLETED:
#             logger.close()


# def print_summary(metrics, detailed_metrics_names):
#     print("Evaluation summary")
#     print("-" * 18)
#
#     metrics_names = [k for k in list(metrics.keys()) if k not in detailed_metrics_names]
#
#     for metric_name in metrics_names:
#         print("\t" + f"{metric_name}: {metrics[metric_name]:.3f}")
# mode_dir = os.path.join(log_dir, mode)
#
# if not os.path.exists(mode_dir):
#     os.mkdir(mode_dir)
#
# dataset_dir = os.path.join(mode_dir, dataset_name)
#
# if not os.path.exists(dataset_dir):
#     os.mkdir(dataset_dir)
#
# option_dir = os.path.join(dataset_dir, option)
#
# if os.path.exists(option_dir):
#     shutil.rmtree(option_dir)
#
# os.mkdir(option_dir)

# def save_analysis_log(log_dir, log, models_config):
#     checkpoint_name = models_config[exp.CHECKPOINT_NAME]
#     log[0].to_csv(os.path.join(log_dir, f"{checkpoint_name[0]}_analysis_log.csv"))

# if metric_config[wr.MEASURE_TIME]:
#     print("\t" + f"Forward time: {data_engine.state.metrics[wr.FORWARD_TIME]}")

# """
# Evaluation saving/loading functions
# """

#
# def save(data, log_dir, checkpoint_name):
#     file_path = os.path.join(log_dir, f"{checkpoint_name}.pkl")
#
#     with open(file_path, 'wb') as file:
#         pickle.dump(data, file)



# def plot_scores(writer, state_engine, data_engine, keys, normalize=False):
#     batch, endpoint = data_engine.state.output
#
#     image1_name, image2_name = batch.get(du.IMAGE1_NAME), batch.get(du.IMAGE2_NAME)
#
#     s1, s2 = endpoint[keys[0]], endpoint[keys[1]]
#
#     if normalize:
#         s1 = s1 / s1.max()
#         s2 = s2 / s2.max()
#
#     image1_name = image1_name[0]
#     image2_name = image2_name[0]
#
#     s = make_grid(torch.cat((s1, s2), dim=0))
#
#     writer.add_image(f"{keys[0]} and {keys[1]} of {image1_name} and {image2_name}", s, state_engine.state.epoch)
#
#
# def plot_kp_matches(writer, state_engine, data_engine, px_thresh):
#     batch, endpoint = data_engine.state.output
#
#     image1, image2 = batch.get(du.IMAGE1), batch.get(du.IMAGE2)
#     image1_name, image2_name = batch.get(du.IMAGE1_NAME), batch.get(du.IMAGE2_NAME)
#     kp1, kp2 = endpoint[eu.KP1], endpoint[eu.KP2]
#     w_kp1, w_kp2 = endpoint[eu.W_KP1], endpoint[eu.W_KP2]
#     w_vis_kp1_mask, w_vis_kp2_mask = endpoint[eu.W_KP1_MASK], endpoint[eu.W_KP2_MASK]
#
#     image1_name = image1_name[0]
#     image2_name = image2_name[0]
#
#     _, _, _, nn_kp_ids, match_mask = repeatability_score(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask,
#                                                          px_thresh, True)
#
#     cv_keypoints_matches = draw_cv_matches(image1, image2, kp1, kp2, nn_kp_ids, match_mask[0])
#
#     writer.add_image(f"{image1_name} and {image2_name} keypoints matches", cv_keypoints_matches,
#                      state_engine.state.epoch, dataformats='HWC')
#
#
# def plot_desc_matches(writer, state_engine, data_engine, px_thresh, sim_measure):
#     batch, endpoint = data_engine.state.output
#
#     image1, image2 = batch.get(du.IMAGE1), batch.get(du.IMAGE2)
#     image1_name, image2_name = batch.get(du.IMAGE1_NAME), batch.get(du.IMAGE2_NAME)
#     kp1, kp2 = endpoint[eu.KP1], endpoint[eu.KP2]
#     w_kp1, w_kp2 = endpoint[eu.W_KP1], endpoint[eu.W_KP2]
#     w_vis_kp1_mask, w_vis_kp2_mask = endpoint[eu.W_KP1_MASK], endpoint[eu.W_KP2_MASK]
#     kp1_desc, kp2_desc = endpoint[eu.KP1_DESC], endpoint[eu.KP2_DESC]
#
#     image1_name = image1_name[0]
#     image2_name = image2_name[0]
#
#     _, _, _, nn_desc_ids, match_mask = match_score(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask,
#                                                    kp1_desc, kp2_desc, px_thresh, sim_measure, detailed=True)
#
#     cv_desc_matches = draw_cv_matches(image1, image2, kp1, kp2, nn_desc_ids, match_mask[0])
#
#     writer.add_image(f"{image1_name} and {image2_name} descriptor matches", cv_desc_matches, state_engine.state.epoch,
#                      dataformats='HWC')

# import torch
# from torchvision.utils import make_grid
#
# from .source.utils.vis_utils import draw_cv_matches
# from .source.evaluation.metric import repeatability_score, match_score

# def save_super_point_inference(output):
#     batch, endpoint = output
#
#     image_name = batch.get(du.IMAGE1_NAME) + batch.get(du.IMAGE2_NAME)
#     score = [endpoint[eu.SCORE1], endpoint[eu.SCORE2]]
#
#     for im_path, s in zip(image_name, score):
#         splits = im_path.split('/')
#         score_dir = os.path.join('/'.join(splits[:-2]), 'scores')
#
#         if not os.path.exists(score_dir):
#             os.mkdir(score_dir)
#
#         score_path = os.path.join(score_dir, splits[-1].split('.')[0])
#
#         np.save(score_path, s.permute(0, 2, 3, 1).cpu().numpy()[0])

# def join_detailed_metrics(data_engine, metrics_names):
#     detailed_metrics = data_engine.state.metrics[metrics_names[0]]
#
#     return detailed_metrics
# class TBLossWriter:
#
#     def __init__(self, mode):
#         self.mode = mode
#
#     def __call__(self, data_engine, state_engine):
#         for key, value in data_engine.state.metrics.items():
#             if meu.LOSS in key and value is not None:
#                 state_engine.state.writer.add_scalar(f"{self.mode}/{key}", value, state_engine.state.iteration)
#
#
# class TBMetricWriter:
#
#     def __init__(self, mode, metrics_names):
#         self.mode = mode
#         self.metrics_names = metrics_names
#
#     def __call__(self, data_engine, state_engine):
#         for metric_name in self.metrics_names:
#             value = data_engine.state.metrics[metric_name]
#
#             if value is not None:
#                 state_engine.state.writer.add_scalar(f"{self.mode}/{metric_name}", value, state_engine.state.iteration)


        # if self.metrics_summary is not None:
        #     for name in self.metrics_summary:
        #         found = False
        #
        #         for metric_name in engine.state.metrics:
        #             if name in metric_name:
        #                 metrics[metric_name] = engine.state.metrics[metric_name]
        #                 found = True
        #
        #         if not found:
        #             warnings.warn("Provided metric name '{}' is missing "
        #                           "in engine's state metrics: {}".format(name, list(engine.state.metrics.keys())))
        #             continue
        #
        # elif

        # if value is contained in metrics_summary