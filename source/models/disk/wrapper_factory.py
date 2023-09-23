import source.utils.model_utils as mu

from source.models.base.wrapper_factory import WrapperFactory

from source.models.disk.module_wrappers.detector_descriptor import DetectorDescriptorWrapper
from source.models.disk.module_wrappers.descriptor import DescriptorWrapper
from source.models.disk.module_wrappers.detector import DetectorWrapper


class DISKWrapperFactory(WrapperFactory):

    def _create_detector_wrapper(self, module_config, model_config,
                                 experiment_config):
        return DetectorWrapper(experiment_config)

    def _create_descriptor_wrapper(self, module_config, model_config,
                                   experiment_config):
        return DescriptorWrapper(experiment_config)

    def _create_detector_descriptor_wrapper(self, module_config, model_config,
                                            experiment_config):
        return DetectorDescriptorWrapper(experiment_config)

