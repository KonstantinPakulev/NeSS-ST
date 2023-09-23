from abc import ABC, abstractmethod


class HandCraftedDetectorModule(ABC):

    def __init__(self, loc):
        self.loc = loc
        self.border_size = None

    def __call__(self, image_gray, eval_params):
        """
        :param image: B x 1 x H x W
        """
        return self._forward(image_gray, eval_params)

    @abstractmethod
    def get_name(self):
        ...

    @abstractmethod
    def get_score(self, image_gray):
        ...

    def set_loc(self, loc):
        self.loc = loc

    def localize_kp(self, kp, image_gray, score):
        if self.loc:
            return self._localize_kp_impl(kp, image_gray, score)

        else:
            return kp

    def set_border_size(self, border_size):
        self.border_size = border_size

    def get_border_size(self):
        return self._calculate_border_size() if self.border_size is None else self.border_size

    @abstractmethod
    def _forward(self, image_gray, eval_params):
        """
        :param image_gray: B x 1 x H x W
        """
        ...

    @abstractmethod
    def _calculate_border_size(self):
        ...

    @abstractmethod
    def _localize_kp_impl(self, kp, image_gray, score):
        ...
