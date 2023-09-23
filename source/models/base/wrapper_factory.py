import source.utils.model_utils as mu


class WrapperFactory:

    def create(self, model_config,
               experiment_config):
        modules_wrappers = []

        for key, value in model_config.modules.items():
            if key == mu.BACKBONE:
                wrapper = self._create_backbone_wrapper(value, model_config, experiment_config)
                if wrapper is not None:
                    modules_wrappers.append(wrapper)

            elif key == mu.DETECTOR:
                wrapper = self._create_detector_wrapper(value, model_config, experiment_config)
                if wrapper is not None:
                    modules_wrappers.append(wrapper)

            elif key == mu.DESCRIPTOR:
                wrapper = self._create_descriptor_wrapper(value, model_config, experiment_config)
                if wrapper is not None:
                    modules_wrappers.append(wrapper)

            elif key == mu.DETECTOR_DESCRIPTOR:
                wrapper = self._create_detector_descriptor_wrapper(value, model_config, experiment_config)
                if wrapper is not None:
                    modules_wrappers.append(wrapper)

            else:
                raise NotImplementedError(key)

        return modules_wrappers

    def _create_backbone_wrapper(self, module_config, model_config,
                                 experiment_config):
        return None

    def _create_detector_wrapper(self, module_config, model_config,
                                 experiment_config):
        return None

    def _create_descriptor_wrapper(self, module_config, model_config,
                                   experiment_config):
        return None

    def _create_detector_descriptor_wrapper(self, module_config, model_config,
                                            experiment_config):
        return None