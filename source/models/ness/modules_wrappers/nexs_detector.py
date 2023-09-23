import source.datasets.base.utils as du
import source.models.ness.modules_wrappers.namespace as n_ns
import source.utils.endpoint_utils as eu

from source.core.module import ModuleWrapper, get_ith_key_input

from source.models.ness.modules.base_detector import create_base_detector
from source.models.ness.modules.nn.ness_regressor import NeSSRegressor
from source.models.ness.modules.nn.ners_regressor import NeRSRegressor
from source.models.ness.modules.ness_detector import NeSSDetector
from source.models.ness.modules.ners_detector import NeRSDetector


class NeXSDetectorWrapper(ModuleWrapper):

    def __init__(self, module_config, experiment_config):
        super().__init__(experiment_config)
        self.base_detector = create_base_detector(module_config)

        if n_ns.NESS in module_config:
            self.regressor = NeSSRegressor.from_config(module_config)
            self.detector = NeSSDetector(self.base_detector)
            self.process_base_key = n_ns.NESS

        elif n_ns.NERS in module_config:
            self.regressor = NeRSRegressor.from_config(module_config)
            self.detector = NeRSDetector(self.base_detector)
            self.process_base_key = n_ns.NERS

    def _get_forward_base_key(self):
        return du.IMAGE

    def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
        image = get_ith_key_input(du.IMAGE, i, batch, bundle, endpoint, device)

        nexs = self.regressor(image)

        bundle[f"{self.process_base_key}{i}"] = nexs

    def _get_process_base_key(self):
        return self.process_base_key

    def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
        image_gray = get_ith_key_input(du.IMAGE_GRAY, i, batch, None, endpoint, device)

        nexs = inference_bundle[f"{self.process_base_key}{i}"].to(device)

        nexs_output = self.detector(image_gray, nexs, eval_params)

        endpoint[f"{eu.KP}{i}"] = nexs_output[0]
        endpoint[f"{eu.KP_SCORE}{i}"] = nexs_output[1]
        endpoint[f"{n_ns.KP_BASE_SCORE}{i}"] = nexs_output[2]

    def get(self):
        return self.regressor


"""
Legacy code
"""

# resized_image_gray, resized_shift_scale, cropped_shape = \
#     get_resized_image(image_gray, self.scales, self.input_size_divisor)
#
# ms_kp_score = []
# ms_kp = []
# ms_kp_scale_idx = []

# for j, (resized_image_grayi,
#         scalei,
#         resized_shift_scalei,
#         cropped_shapei) in enumerate(zip(resized_image_gray,
#                                          [1.0] + self.scales,
#                                          resized_shift_scale,
#                                          cropped_shape)):
#     kp_scorej, kpj = select_shi_ness_kp_at_scale(resized_image_grayi, ness,
#                                                  scalei, resized_shift_scalei, cropped_shapei,
#                                                  self.sobel_size, self.window_size, self.window_cov,
#                                                  self.loc,
#                                                  nms_size, shi_thresh,
#                                                  self.min_ness, self.max_ness,
#                                                  k)
#
#     ms_kp_score.append(kp_scorej)
#     ms_kp.append(kpj)
#     ms_kp_scale_idx.append(torch.ones_like(kp_scorej, dtype=torch.long) * j)
#
# ms_kp_score = torch.cat(ms_kp_score, dim=1)
# ms_kp = torch.cat(ms_kp, dim=1)
# ms_kp_scale_idx = torch.cat(ms_kp_scale_idx, dim=1)
#
# kp_nms_3d_mask = get_kp_nms_3d_mask(image_gray.shape, [1.0] + self.scales,
#                                     ms_kp, ms_kp_score, ms_kp_scale_idx,
#                                     nms_size)
#
# k = min(k, ms_kp_score.shape[1])
#
# kp_score, idx = torch.topk(ms_kp_score * kp_nms_3d_mask.float(), k, dim=-1)
#
# endpoint[f"{eu.KP_SCORE}{i}"] = kp_score
# endpoint[f"{eu.KP}{i}"] = ms_kp.gather(-2, idx.unsqueeze(-1).repeat(1, 1, 2))

# if self.scales is not None:
#     resized_image, _ = get_resized_image(image, self.scales, self.input_size_divisor)
#
#     ness = []
#
#     for resized_imagei in resized_image:
#         ness.append(self.detector(resized_imagei))
#
#     bundle[f"{deu.NESS}{i}"] = ness
#
# else:


# U_EST_THRESH = 'u_est_thresh'

# select_shi_u_est_kp, select_u_est_kp

# self._process_branch(engine, device, i, scale_batch, scale_inference_bundle, scale_endpoint, eval_params)
#
# kp_score_key = f"{eu.KP_SCORE}{i}"
# kp_key = f"{eu.KP}{i}"
#
# shift_scalei = scale_batch[f"{du.SHIFT_SCALE}{i}"]
#
# kp_scorei = scale_endpoint[kp_score_key]
#
# if self.ms_score_correction:
#     kp_scorei = kp_scorei ** ((1 / shift_scalei[..., 2:].mean(dim=-1).unsqueeze(-1)) ** 2)
#
# kpi = revert_shift_scale(scale_endpoint[kp_key], shift_scalei, change_orientation=False)
#
# if kp_key not in endpoint:
#     endpoint[kp_score_key] = kp_scorei
#     endpoint[kp_key] = kpi
#
# else:
#     endpoint[kp_score_key] = torch.cat([endpoint[kp_score_key], kp_scorei], dim=1)
#     endpoint[kp_key] = torch.cat([endpoint[kp_key], kpi], dim=1)
#
# if self.ms_nms_type == NMS_3D:
#     kp_scale_idx_key = f"{eu.KP_SCALE_IDX}{i}"
#
#     scale_idx = scale_batch[f"{du.SCALE_IDX}{i}"]
#
#     kp_scale_idxi = torch.ones_like(kp_scorei, dtype=torch.long) * scale_idx
#
#     if kp_scale_idx_key not in endpoint:
#         endpoint[kp_scale_idx_key] = kp_scale_idxi
#
#     else:
#         endpoint[kp_scale_idx_key] = torch.cat([endpoint[kp_scale_idx_key], kp_scale_idxi], dim=1)
# if self.ms_nms_type == NMS_3D:
#     nms_size = eval_params.nms_size
#
#     kp_score_key = f"{eu.KP_SCORE}{i}"
#
#     kp_scorei = endpoint[kp_score_key]
#     kp_nms_3d_maski = get_kp_nms_3d_mask(batch[f"{du.IMAGE}{i}"].shape, 1 + len(self.extra_scales),
#                                          endpoint[f"{eu.KP}{i}"], kp_scorei, endpoint[f"{eu.KP_SCALE_IDX}{i}"],
#                                          nms_size)
#
#     kp_scorei = kp_scorei * kp_nms_3d_maski.float()
#
#     endpoint[kp_score_key] = kp_scorei
#
# super()._process_branch_multi_scale(engine, device, i, batch, ms_bundle, endpoint, eval_params)

# self.extra_scales = models_config.get(SCALES, None)
# self.ms_nms_type = module_config.get(MS_NMS_TYPE, NAIVE)
# self.ms_score_correction = module_config.get(MS_SCORE_CORRECTION, False)

# from omegaconf import OmegaConf
#
# import source.datasets.base.utils as du
# import source.utils.endpoint_utils as eu
# import source.models.shiness.utils.endpoint_utils as deu
#
# import torch
# import torch.nn.functional as F
#
# from source.module import DetectorModuleWrapper, KP_SCORE_IDX, SCALES
#
# from source.models.shiness.model.unet import DeVLUNet
# from source.models.shiness.utils.endpoint_utils import get_ith_key_input, select_shi_u_est_kp, get_kp_nms_3d_mask
# from source.pose.utils import revert_shift_scale
#
#
# VAR_THRESH = 'var_thresh'
#
# MS_NMS_TYPE = 'ms_nms_type'
# MS_SCORE_CORRECTION = 'ms_score_correction'
#
# NAIVE = 'naive'
# NMS_3D = 'nms_3d'
#
#
# class ShiUEstWrapper(DetectorModuleWrapper):
#
#     def __init__(self, module_config, models_config):
#         self.detector = DeVLUNet(module_config)
#
#         self.sobel_size = module_config.shi.sobel_size
#         self.window_size = module_config.shi.window_size
#         self.window_cov = module_config.shi.window_cov
#         self.loc = module_config.shi.loc
#
#         self.min_var = module_config.min_var
#         self.max_var = module_config.max_var
#
#         self.extra_scales = models_config.get(SCALES, None)
#         self.ms_nms_type = module_config.get(MS_NMS_TYPE, NAIVE)
#         self.ms_score_correction = module_config.get(MS_SCORE_CORRECTION, False)
#
#     def _get_forward_base_key(self):
#         return du.IMAGE
#
#     def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
#         imagei = get_ith_key_input(du.IMAGE, i, batch, bundle, endpoint, device)
#
#         vari = self.detector(imagei)
#
#         bundle[f"{deu.VAR}{i}"] = vari
#
#     def _get_process_base_key(self):
#         return deu.VAR
#
#     def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
#         nms_size = eval_params.nms_size
#         shi_thresh = eval_params.shi_thresh
#         var_thresh = eval_params.get(VAR_THRESH)
#         k = eval_params.topk
#
#         image_grayi = get_ith_key_input(du.IMAGE_GRAY, i, batch, None, endpoint, device)
#         vari = inference_bundle[f"{deu.VAR}{i}"].to(device)
#
#         kp_scorei, shi_kp_scorei, kpi = select_shi_u_est_kp(image_grayi, vari,
#                                                             self.sobel_size, self.window_size, self.window_cov, self.loc,
#                                                             nms_size, shi_thresh,
#                                                             self.min_var, self.max_var,
#                                                             var_thresh, k)
#
#         endpoint[f"{eu.KP_SCORE}{i}"] = kp_scorei
#         endpoint[f"{deu.KP_SHI_SCORE}{i}"] = shi_kp_scorei
#         endpoint[f"{eu.KP}{i}"] = kpi
#
#     def _process_branch_at_scale(self, engine, device, i, scale_batch, scale_inference_bundle, scale_endpoint, eval_params, endpoint):
#         self._process_branch(engine, device, i, scale_batch, scale_inference_bundle, scale_endpoint, eval_params)
#
#         kp_score_key = f"{eu.KP_SCORE}{i}"
#         shi_kp_score_key = f"{deu.KP_SHI_SCORE}{i}"
#         kp_key = f"{eu.KP}{i}"
#
#         shift_scalei = scale_batch[f"{du.SHIFT_SCALE}{i}"]
#
#         kp_scorei = scale_endpoint[kp_score_key]
#
#         if self.ms_score_correction:
#             kp_scorei = kp_scorei ** ((1 / shift_scalei[..., 2:].mean(dim=-1).unsqueeze(-1)) ** 2)
#
#         kpi = revert_shift_scale(scale_endpoint[kp_key], shift_scalei, change_orientation=False)
#
#         if kp_key not in endpoint:
#             endpoint[kp_score_key] = kp_scorei
#             endpoint[shi_kp_score_key] = scale_endpoint[shi_kp_score_key]
#             endpoint[kp_key] = kpi
#
#         else:
#             endpoint[kp_score_key] = torch.cat([endpoint[kp_score_key], kp_scorei], dim=1)
#             endpoint[shi_kp_score_key] = torch.cat([endpoint[shi_kp_score_key], scale_endpoint[shi_kp_score_key]], dim=1)
#             endpoint[kp_key] = torch.cat([endpoint[kp_key], kpi], dim=1)
#
#         if self.ms_nms_type == NMS_3D:
#             kp_scale_idx_key = f"{eu.KP_SCALE_IDX}{i}"
#
#             scale_idx = scale_batch[f"{du.SCALE_IDX}{i}"]
#
#             kp_scale_idxi = torch.ones_like(kp_scorei, dtype=torch.long) * scale_idx
#
#             if kp_scale_idx_key not in endpoint:
#                 endpoint[kp_scale_idx_key] = kp_scale_idxi
#
#             else:
#                 endpoint[kp_scale_idx_key] = torch.cat([endpoint[kp_scale_idx_key], kp_scale_idxi], dim=1)
#
#     def _process_branch_multi_scale(self, engine, device, i, batch, ms_bundle, endpoint, eval_params):
#         if self.ms_nms_type == NMS_3D:
#             nms_size = eval_params.nms_size
#
#             kp_score_key = f"{eu.KP_SCORE}{i}"
#
#             kp_scorei = endpoint[kp_score_key]
#             kp_nms_3d_maski = get_kp_nms_3d_mask(batch[f"{du.IMAGE}{i}"].shape, 1 + len(self.extra_scales),
#                                                  endpoint[f"{eu.KP}{i}"], kp_scorei, endpoint[f"{eu.KP_SCALE_IDX}{i}"],
#                                                  nms_size)
#
#             kp_scorei = kp_scorei * kp_nms_3d_maski.float()
#
#             endpoint[kp_score_key] = kp_scorei
#
#         super()._process_branch_multi_scale(engine, device, i, batch, ms_bundle, endpoint, eval_params)
#
#         shi_kp_score_key = f"{deu.KP_SHI_SCORE}{i}"
#
#         kp_score_idxi = ms_bundle[f'{KP_SCORE_IDX}{i}']
#
#         endpoint[shi_kp_score_key] = endpoint[shi_kp_score_key].gather(-1, kp_score_idxi)
#
#     def get(self):
#         return self.detector

# import source.datasets.base.utils as du
# import source.utils.endpoint_utils as eu
# import source.models.shiness.utils.endpoint_utils as deu
#
# from source.module import DetectorModuleWrapper
#
# from source.models.shiness.model.unet import DeVLUNet
# from source.models.shiness.utils.endpoint_utils import get_ith_key_input, select_doh_u_est_kp
#
#
# class DoHUEstWrapper(DetectorModuleWrapper):
#
#     def __init__(self, module_config, models_config):
#         self.detector = DeVLUNet(module_config)
#
#         self.sobel_size = module_config.doh.sobel_size
#         self.loc = module_config.doh.loc
#
#         self.min_var = module_config.min_var
#         self.max_var = module_config.max_var
#
#     def _get_forward_base_key(self):
#         return du.IMAGE
#
#     def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
#         imagei = get_ith_key_input(du.IMAGE, i, batch, bundle, endpoint, device)
#
#         vari = self.detector(imagei)
#
#         bundle[f"{deu.VAR}{i}"] = vari
#
#     def _get_process_base_key(self):
#         return deu.VAR
#
#     def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
#         nms_size = eval_params.nms_size
#         k = eval_params.topk
#
#         image_grayi = get_ith_key_input(du.IMAGE_GRAY, i, batch, None, endpoint, device)
#         vari = inference_bundle[f"{deu.VAR}{i}"].to(device)
#
#         kpi = select_doh_u_est_kp(image_grayi, vari,
#                                   self.sobel_size, self.loc,
#                                   nms_size,
#                                   self.min_var, self.max_var,
#                                   k)[2]
#
#         endpoint[f"{eu.KP}{i}"] = kpi
#
#     def _process_branch_at_scale(self, engine, device, i, scale_batch, scale_inference_bundle, scale_endpoint, eval_params, endpoint):
#         pass
#
#     def _process_branch_multi_scale(self, engine, device, i, batch, ms_bundle, endpoint, eval_params):
#         pass
#
#     def get(self):
#         return self.detector
# import source.datasets.base.utils as du
# import source.utils.endpoint_utils as eu
# import source.models.shiness.utils.endpoint_utils as deu
#
# from source.module import DetectorModuleWrapper, KP_SCORE_IDX, SCALES
#
# from source.models.shiness.model.unet import DeVLUNet
# from source.models.shiness.model_wrappers.shi_u_est import MS_NMS_TYPE, MS_SCORE_CORRECTION, NMS_3D, NAIVE
# from source.models.shiness.utils.endpoint_utils import get_ith_key_input, select_u_est_kp, get_kp_nms_3d_mask
# from source.models.shi.model import get_shi_score
#
#
# class UEstWrapper(DetectorModuleWrapper):
#
#     def __init__(self, module_config, models_config):
#         self.detector = DeVLUNet(module_config)
#
#         self.sobel_size = module_config.shi.sobel_size
#         self.window_size = module_config.shi.window_size
#         self.window_cov = module_config.shi.window_cov
#
#         self.u_est_key = module_config.u_est_key
#
#         self.min_u_est = module_config.min_u_est
#         self.max_u_est = module_config.max_u_est
#
#         self.extra_scales = models_config.get(SCALES, None)
#         self.ms_nms_type = module_config.get(MS_NMS_TYPE, NAIVE)
#         self.ms_score_correction = module_config.get(MS_SCORE_CORRECTION, False)
#
#     def _get_forward_base_key(self):
#         return du.IMAGE
#
#     def _forward_branch(self, engine, device, i, batch, bundle, endpoint):
#         imagei = get_ith_key_input(du.IMAGE, i, batch, bundle, endpoint, device)
#
#         u_est = self.detector(imagei)
#
#         bundle[f"{self.u_est_key}{i}"] = u_est
#
#     def _get_process_base_key(self):
#         return self.u_est_key
#
#     def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
#         nms_size = eval_params.nms_size
#         shi_thresh = eval_params.shi_thresh
#         k = eval_params.topk
#
#         image_grayi = get_ith_key_input(du.IMAGE_GRAY, i, batch, None, endpoint, device)
#         u_esti = inference_bundle[f"{self.u_est_key}{i}"].to(device)
#
#         shi_scorei = get_shi_score(image_grayi, self.sobel_size, self.window_size, self.window_cov)
#
#         kp_scorei, kpi = select_u_est_kp(u_esti,
#                                          nms_size,
#                                          self.min_u_est, self.max_u_est,
#                                          k,
#                                          u_est_mask=shi_scorei > shi_thresh)
#
#         endpoint[f"{eu.KP_SCORE}{i}"] = kp_scorei
#         endpoint[f"{eu.KP}{i}"] = kpi
#
#     def _process_branch_at_scale(self, engine, device, i, scale_batch, scale_inference_bundle, scale_endpoint,
#                                  eval_params, endpoint):
#         self._process_branch(engine, device, i, scale_batch, scale_inference_bundle, scale_endpoint, eval_params)
#
#         kp_score_key = f"{eu.KP_SCORE}{i}"
#         kp_key = f"{eu.KP}{i}"
#
#         shift_scalei = scale_batch[f"{du.SHIFT_SCALE}{i}"]
#
#         kp_scorei = scale_endpoint[kp_score_key]
#
#         if self.ms_score_correction:
#             kp_scorei = kp_scorei ** ((1 / shift_scalei[..., 2:].mean(dim=-1).unsqueeze(-1)) ** 2)
#
#         kpi = revert_shift_scale(scale_endpoint[kp_key], shift_scalei, change_orientation=False)
#
#         if kp_key not in endpoint:
#             endpoint[kp_score_key] = kp_scorei
#             endpoint[kp_key] = kpi
#
#         else:
#             endpoint[kp_score_key] = torch.cat([endpoint[kp_score_key], kp_scorei], dim=1)
#             endpoint[kp_key] = torch.cat([endpoint[kp_key], kpi], dim=1)
#
#         if self.ms_nms_type == NMS_3D:
#             kp_scale_idx_key = f"{eu.KP_SCALE_IDX}{i}"
#
#             scale_idx = scale_batch[f"{du.SCALE_IDX}{i}"]
#
#             kp_scale_idxi = torch.ones_like(kp_scorei, dtype=torch.long) * scale_idx
#
#             if kp_scale_idx_key not in endpoint:
#                 endpoint[kp_scale_idx_key] = kp_scale_idxi
#
#             else:
#                 endpoint[kp_scale_idx_key] = torch.cat([endpoint[kp_scale_idx_key], kp_scale_idxi], dim=1)
#
#     def _process_branch_multi_scale(self, engine, device, i, batch, ms_bundle, endpoint, eval_params):
#         if self.ms_nms_type == NMS_3D:
#             nms_size = eval_params.nms_size
#
#             kp_score_key = f"{eu.KP_SCORE}{i}"
#
#             kp_scorei = endpoint[kp_score_key]
#             kp_nms_3d_maski = get_kp_nms_3d_mask(batch[f"{du.IMAGE}{i}"].shape, 1 + len(self.extra_scales),
#                                                  endpoint[f"{eu.KP}{i}"], kp_scorei, endpoint[f"{eu.KP_SCALE_IDX}{i}"],
#                                                  nms_size)
#
#             kp_scorei = kp_scorei * kp_nms_3d_maski.float()
#
#             endpoint[kp_score_key] = kp_scorei
#
#         super()._process_branch_multi_scale(engine, device, i, batch, ms_bundle, endpoint, eval_params)
#
#     def get(self):
#         return self.detector
#
#
# """
# Legacy code
# """
#
# # from source.models.devl.model_wrappers.shi_u_est_conf import U_EST_THRESH