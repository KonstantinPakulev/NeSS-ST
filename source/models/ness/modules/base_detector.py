import source.models.namespace as m_ns

from source.models.base.modules.handcrafted import HandCraftedDetectorModule
from source.models.shi.module import ShiDetector
from source.models.doh.module import DeterminantOfHessianDetector
from source.models.log.module import LaplacianOfGaussianDetector


def create_base_detector(module_config):
    if m_ns.SHI in module_config:
        return ShiDetector.from_config(module_config.shi)

    elif m_ns.DOH in module_config:
        return DeterminantOfHessianDetector.from_config(module_config.doh)

    elif m_ns.LOG in module_config:
        return LaplacianOfGaussianDetector.from_config(module_config.log)

    else:
        raise ValueError("No base detector found")
