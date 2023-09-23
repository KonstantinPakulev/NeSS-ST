import numpy as np
import cv2
import torch

import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu

from source.core.module import ModuleWrapper


class DetectorWrapper(ModuleWrapper):

    def __init__(self, module_config, experiment_config):
        super().__init__(experiment_config)
        self.sift = cv2.SIFT_create(0,
                                    module_config.nOctaveLayers,
                                    module_config.contrastThreshold,
                                    module_config.edgeThreshold,
                                    module_config.sigma)

    def _get_forward_base_key(self):
        return du.IMAGE_GRAY

    def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
        pass

    def _get_process_base_key(self):
        return None

    def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
        k = eval_params.topk

        image_grayi = batch[f"{du.IMAGE_GRAY}{i}"]

        if image_grayi.shape[0] > 1:
            raise NotImplementedError

        cv_kpi = self.sift.detect((image_grayi.numpy()[0, 0] * 255).astype(np.uint8), None)

        kp_scorei = np.array([j.response for j in cv_kpi])
        kp_idxi = np.argsort(kp_scorei)[::-1]

        kp_scorei = kp_scorei[kp_idxi]
        kpi = np.array([cv_kpj.pt for cv_kpj in cv_kpi])[kp_idxi]

        kp_unique_idxi = np.array([idx for idx, kpj in enumerate(kpi) if kpj not in kpi[:idx]])

        kp_scorei = kp_scorei[kp_unique_idxi]
        kpi = kpi[kp_unique_idxi]

        kp_scorei = kp_scorei[None, :k]
        kpi = kpi[None, :k, [1, 0]] + 0.5

        endpoint[f"{eu.KP_SCORE}{i}"] = torch.tensor(kp_scorei, dtype=torch.float)
        endpoint[f"{eu.KP}{i}"] = torch.tensor(kpi, dtype=torch.float)

    def get(self):
        return None


"""
Support utils
"""

# def get_scale_from_octave(octave):
#     octave = octave & 255
#     if octave >= 128:
#         octave = octave | -128
#     scale = 1 / float(1 << octave) if octave >= 0 else float(1 << -octave)
#     return scale

# kp_scalei = np.array([get_scale_from_octave(cv_kpj.octave) for cv_kpj in cv_kpi])[kp_idxi]
# kp_scalei = kp_scalei[kp_unique_idxi]

# def _process_branch_multi_scale(self, engine, device, i, batch, ms_bundle, endpoint, eval_params):
#     kp_score_key = f"{eu.KP_SCORE}{i}"
#     kp_key = f"{eu.KP}{i}"
#
#     kp_scorei = torch.cat([sj for sj in endpoint[kp_score_key] if sj is not None], dim=1)
#     kpi = torch.cat([kpj for kpj in endpoint[kp_key] if kpj is not None], dim=1)
#
#     kp_scorei, kp_score_idxi = torch.sort(kp_scorei, descending=True, dim=-1)
#
#     ms_bundle[f'{KP_SCORE_IDX}{i}'] = kp_score_idxi
#
#     endpoint[kp_score_key] = kp_scorei
#     endpoint[kp_key] = kpi.gather(-2, kp_score_idxi.unsqueeze(-1).repeat(1, 1, 2))

# def _process_branch_at_scale(self, engine, device, i, scale_batch, scale_inference_bundle, scale_endpoint, eval_params, endpoint):
#     scale_idxi = scale_batch[f"{du.SCALE_IDX}{i}"]
#
#     kp_score_key = f"{eu.KP_SCORE}{i}"
#     kp_key = f"{eu.KP}{i}"
#
#     if scale_idxi == 0:
#         self._process_branch(engine, device, i, scale_batch, scale_inference_bundle, scale_endpoint, eval_params)
#
#         kp_scorei = scale_endpoint[kp_score_key]
#         kpi = scale_endpoint[kp_key]
#         kp_scalei = scale_endpoint[f"{KP_SCALE}{i}"]
#
#         scale_ordered_kp_scorei = []
#         scale_ordered_kpi = []
#
#         for s in [1.0] + self.extra_scales:
#             kp_scale_maskj = kp_scalei == s
#
#             if kp_scale_maskj.sum() != 0:
#                 scale_ordered_kp_scorei.append(kp_scorei[:, kp_scale_maskj[0]])
#                 scale_ordered_kpi.append(kpi[:, kp_scale_maskj[0]])
#
#             else:
#                 scale_ordered_kp_scorei.append(None)
#                 scale_ordered_kpi.append(None)
#
#         endpoint[kp_score_key] = scale_ordered_kp_scorei
#         endpoint[kp_key] = scale_ordered_kpi
#
#     scale_kp_scorei = endpoint[kp_score_key][scale_idxi]
#
#     if scale_kp_scorei is not None:
#         scale_endpoint[kp_score_key] = scale_kp_scorei
#         scale_endpoint[kp_key] = endpoint[kp_key][scale_idxi]

# if self.extra_scales is not None:
#     kp_scale_maski = kp_scalei >= min(self.extra_scales)
#
#     kp_scorei = kp_scorei[kp_scale_maski][None]
#     kpi = kpi[kp_scale_maski][None, :k, [1, 0]] + 0.5
#
#     endpoint[f"{KP_SCALE}{i}"] = torch.tensor(np.minimum(kp_scalei[kp_scale_maski], 1.0)[None], dtype=torch.float)
#
# else: