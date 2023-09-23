from source.models.base.wrapper_factory import WrapperFactory
from source.models.harris.module_wrapper import DetectorWrapper


class HarrisWrapperFactory(WrapperFactory):

    def _create_detector_wrapper(self, module_config, model_config, experiment_config):
        return DetectorWrapper(module_config, experiment_config)