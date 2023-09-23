from source.models.base.wrapper_factory import WrapperFactory

from source.models.superpoint.module_wrappers.backbone import BackboneWrapper
from source.models.superpoint.module_wrappers.detector import DetectorWrapper
from source.models.superpoint.module_wrappers.descriptor import DescriptorWrapper


class SuperPointWrapperFactory(WrapperFactory):

    def _create_backbone_wrapper(self, module_config, model_config,
                                 experiment_config):
        return BackboneWrapper(experiment_config)

    def _create_detector_wrapper(self, module_config, model_config,
                                 experiment_config):
        return DetectorWrapper(experiment_config)

    def _create_descriptor_wrapper(self, module_config, model_config,
                                   experiment_config):
        return DescriptorWrapper(experiment_config)

