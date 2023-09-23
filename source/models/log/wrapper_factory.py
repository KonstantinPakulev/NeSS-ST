from source.models.base.wrapper_factory import WrapperFactory
from source.models.log.module_wrapper import DetectorWrapper

class LoGWrapperFactory(WrapperFactory):

    def _create_detector_wrapper(self, module_config, model_config,
                                 experiment_config):
        return DetectorWrapper(module_config, experiment_config)
