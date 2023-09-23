from source.models.base.wrapper_factory import WrapperFactory

from source.models.r2d2.module_wrappers.detector_descriptor import DetectorDescriptorWrapper
from source.models.r2d2.module_wrappers.detector import DetectorWrapper


class R2D2WrapperFactory(WrapperFactory):

    def _create_detector_wrapper(self, module_config, model_config,
                                 experiment_config):
        return DetectorWrapper(experiment_config)

    def _create_detector_descriptor_wrapper(self, module_config, model_config,
                                            experiment_config):
        return DetectorDescriptorWrapper(experiment_config)
