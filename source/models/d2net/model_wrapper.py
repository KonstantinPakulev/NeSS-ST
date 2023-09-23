import source.datasets.base.utils as du
import source.models.joint_net.utils.endpoint_utils as eu
import source.models.joint_net.utils.model_utils as mu
import source.evaluation.classical.metrics as meu

from source.core.wrappers import ModuleWrapper

from source.models.d2net.models import D2Net

from source.models.joint_net.utils.endpoint_utils import select_kp, sample_descriptors, sample_loc
from source.projective.projectivity_utils import get_scale_factor
from source.utils.common_utils import clamp_points, get_clamp_rect


class D2NetContainer(ModuleWrapper):

    def __init__(self, device, mode, model_config, dataset_config, metric_config):
        d2_net_wrapper = D2NetWrapper(device, mode, model_config, metric_config)
        super().__init__(device, mode, [d2_net_wrapper], model_config, dataset_config, metric_config)

    def load_state_dict(self, state_dict, strict=True):
        self.model.__getitem__(0).d2_net.load_checkpoints(state_dict, strict)


class D2NetWrapper(ModuleWrapper):

    def __init__(self, device, mode, model_config, metric_config):
        super().__init__(device, mode, metric_config)
        self.nms_kernel_size = model_config[mu.NMS_KERNEL_SIZE]
        self.top_k = metric_config[mode][meu.TOP_K]

        self.d2_net = D2Net()

    def forward(self, engine, batch, endpoint, bundle):
        if self.deploy_mode:
            image1 = batch[du.IMAGE1]

            score1, desc1, loc1 = self.d2_net(image1.to(self.device))

            scale_factor1 = get_scale_factor(image1.shape, score1.shape, False).to(self.device)

            kp1 = select_kp(score1, self.nms_kernel_size, self.top_k, scale_factor1)

            kp1_loc = sample_loc(loc1, kp1, image1.shape)

            kp1 = clamp_points(kp1 + kp1_loc, get_clamp_rect(image1.shape, 0))

            kp1_desc = sample_descriptors(desc1, kp1, image1.shape)

            endpoint[eu.KP1] = kp1
            endpoint[eu.KP_DESC1] = kp1_desc

        else:
            image1, image2 = batch[du.IMAGE1], batch[du.IMAGE2]

            score1, desc1, loc1 = self.d2_net(image1.to(self.device))
            score2, desc2, loc2 = self.d2_net(image2.to(self.device))

            scale_factor1 = get_scale_factor(image1.shape, score1.shape, False).to(self.device)
            scale_factor2 = get_scale_factor(image2.shape, score2.shape, False).to(self.device)

            kp1 = select_kp(score1, self.nms_kernel_size, self.top_k, scale_factor1)
            kp2 = select_kp(score2, self.nms_kernel_size, self.top_k, scale_factor2)

            kp1_loc = sample_loc(loc1, kp1, image1.shape)
            kp2_loc = sample_loc(loc2, kp2, image2.shape)

            kp1 = clamp_points(kp1 + kp1_loc, get_clamp_rect(image1.shape, 0))
            kp2 = clamp_points(kp2 + kp2_loc, get_clamp_rect(image2.shape, 0))

            # w_kp1, vis_w_kp1_mask, w_kp2, vis_w_kp2_mask = warp_points(kp1, kp2, image1.shape, image2.shape, batch)

            kp1_desc = sample_descriptors(desc1, kp1, image1.shape)
            kp2_desc = sample_descriptors(desc2, kp2, image2.shape)

            endpoint[eu.SCORE1] = score1
            endpoint[eu.SCORE2] = score2

            endpoint[eu.KP1] = kp1
            endpoint[eu.KP2] = kp2

            # endpoint[eu.W_KP1] = w_kp1
            # endpoint[eu.W_KP2] = w_kp2

            # endpoint[eu.W_KP1_MASK] = vis_w_kp1_mask
            # endpoint[eu.W_KP2_MASK] = vis_w_kp2_mask

            endpoint[eu.DESC1] = desc1
            endpoint[eu.DESC2] = desc2

            endpoint[eu.KP_DESC1] = kp1_desc
            endpoint[eu.KP_DESC2] = kp2_desc

        return endpoint, bundle
