from source.models.base.wrapper_factory import WrapperFactory
from source.models.hardnet.module_wrapper import DescriptorWrapper


class HardNetWrapperFactory(WrapperFactory):

    def _create_descriptor_wrapper(self, module_config, model_config,
                                 experiment_config):
        return DescriptorWrapper(module_config, experiment_config)
