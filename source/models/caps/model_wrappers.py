import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu

from source.models.caps.models import CAPSNet

from source.core.module import ModuleWrapper

FULL = 'full'


def create_caps_modules_wrappers(model_config):
    modules_wrappers = []

    for key, value in model_config.modules.items():
        if key == FULL:
            modules_wrappers.append(CAPSWrapper(device))

        else:
            raise NotImplementedError

    return modules_wrappers


class CAPSWrapper(ModuleWrapper):

    def __init__(self, device):
        super().__init__(device)
        self.caps = CAPSNet(device)
        raise NotImplementedError()

    def single_forward(self, engine, batch, endpoint, bundle):
        if du.C_IMAGE1 in batch:
            image1 = batch[du.C_IMAGE1]

        else:
            image1 = batch[du.IMAGE1]

        image1 = image1.to(self.device)

        xc1, xf1 = self.caps.net(image1)

        endpoint[eu.DESC1] = (xc1, xf1)

    def pair_forward(self, engine, batch, endpoint, bundle):
        if du.C_IMAGE1 in batch:
            image1, image2 = batch[du.C_IMAGE1], batch[du.C_IMAGE2]

        else:
            image1, image2 = batch[du.IMAGE1], batch[du.IMAGE2]

        image1, image2 = image1.to(self.device), image2.to(self.device)

        xc1, xf1 = self.caps.net(image1)
        xc2, xf2 = self.caps.net(image2)

        endpoint[eu.DESC1] = (xc1, xf1)
        endpoint[eu.DESC2] = (xc2, xf2)

    def single_process(self, engine, batch, endpoint, evaluation_config):
        image1 = batch[du.IMAGE1]

        xc1, xf1 = endpoint[eu.DESC1]

        kp1 = endpoint[eu.KP1]

        kp1_desc_c, kp1_desc_f = self.caps.extract_features(xc1, xf1, image1, kp1[:, :, [1, 0]])

        kp1_desc = kp1_desc_f

        # kp1_desc = torch.cat((kp1_desc_c, kp1_desc_f), -1)

        endpoint[eu.KP_DESC1] = kp1_desc

    def pair_process(self, engine, batch, endpoint, evaluation_config):
        image1, image2 = batch[du.IMAGE1], batch[du.IMAGE2]

        kp1, kp2 = endpoint[eu.KP1], endpoint[eu.KP2]

        xc1, xf1 = endpoint[eu.DESC1]
        xc2, xf2 = endpoint[eu.DESC2]

        kp1_desc_c, kp1_desc_f = self.caps.extract_features(xc1, xf1, image1, kp1[:, :, [1, 0]])
        kp2_desc_c, kp2_desc_f = self.caps.extract_features(xc2, xf2, image2, kp2[:, :, [1, 0]])

        kp1_desc = kp1_desc_f
        kp2_desc = kp2_desc_f

        # kp1_desc = torch.cat((kp1_desc_c, kp1_desc_f), -1)
        # kp2_desc = torch.cat((kp2_desc_c, kp2_desc_f), -1)

        endpoint[eu.KP_DESC1] = kp1_desc
        endpoint[eu.KP_DESC2] = kp2_desc

    def get(self):
        return self.caps
