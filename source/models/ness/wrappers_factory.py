import source.models.ness.criteria.namespace as cu_ns

from source.models.base.wrapper_factory import WrapperFactory

from source.models.ness.modules_wrappers.nexs_detector import NeXSDetectorWrapper
from source.models.ness.modules_wrappers.xs_detector import XSDetectorWrapper
from source.models.ness.criteria_wrappers.ss_loss import SSLoss
from source.models.ness.criteria_wrappers.rs_loss import RSLoss


class NeXSWrapperFactory(WrapperFactory):

    def _create_detector_wrapper(self, module_config, model_config,
                                 experiment_config):
        return NeXSDetectorWrapper(module_config, experiment_config)


class XSWrapperFactory(WrapperFactory):

    def _create_detector_wrapper(self, module_config, model_config,
                                 experiment_config):
        return XSDetectorWrapper(module_config, experiment_config)


def create_shiness_criteria(model_mode_wrapper,
                            criteria_configs):
    criteria_wrappers = []

    for key, value in criteria_configs.items():
        if key == cu_ns.SS_LOSS:
            cw = SSLoss(model_mode_wrapper, value)

        elif key == cu_ns.RS_LOSS:
            cw = RSLoss(model_mode_wrapper, value)

        else:
            raise NotImplementedError

        criteria_wrappers.append(cw)

    return criteria_wrappers


"""
Legacy code
"""


# elif key == cu_ns.ONLINE_LSS_LOSS:
#     cw = OnlineLSSLoss(value)
#
# elif key == cu_ns.ONLINE_SS_LSS_LOSS:
#     cw = OnlineSSLSSLoss(value)
#
# elif key == cu_ns.ONLINE_MS_LSS_LOSS:
#     cw = OnlineMSLSSLoss(value)

# from source.models.ness.criteria_wrappers.online_lss_loss import OnlineLSSLoss
# from source.models.ness.criteria_wrappers.online_ss_lss_loss import OnlineSSLSSLoss
# from source.models.ness.criteria_wrappers.online_ms_lss_loss import OnlineMSLSSLoss

# from source.models.shiness.model_wrappers.shi_u_est_conf import ShiUEstConfWrapper
# from source.models.shiness.model_wrappers.doh_u_est import DoHUEstWrapper
# from source.models.shiness.model_wrappers.u_est import UEstWrapper
# SHI_U_EST_CONF = 'shi_u_est_conf'

# else:
# raise NotImplementedError(value.wrapper)

# if value.wrapper == SHI_U_EST:
#     modules_wrappers.append(ShiUEstWrapper(value, models_config))
#
# elif value.wrapper == SHI_U_EST_CONF:
#     modules_wrappers.append(ShiUEstConfWrapper(value, models_config))
#
# elif value.wrapper == DOH_U_EST:
#     modules_wrappers.append(DoHUEstWrapper(value, models_config))
#
# elif value.wrapper == U_EST:
#     modules_wrappers.append(UEstWrapper(value, models_config))

# SHI_U_EST = 'shi_u_est'
# DOH_U_EST = 'doh_u_est'
# U_EST = 'u_est'

# if key == cu.HA_LOSS:
#     cw = HALoss(cfg_mode, value)

# from source.models.devl.criteria_wrappers.ha_loss import HALoss

# SHI_CAT = 'shi_cat'
# SHI_CAT_DENSE = 'shi_cat_dense'

# from deprecated.cat_loss import CatLoss
# from deprecated.cat_ha_loss import CatHALoss

# elif key == cu.CAT_LOSS:
#     cw = CatLoss(cfg_mode, value)
#
# elif key == cu.CAT_HA_LOSS:
#     cw = CatHALoss(cfg_mode, value)

# from source.models.devl.criteria_wrappers.gauss_loss import ErrLoss
# from source.models.devl.criteria_wrappers.gauss_ha_loss import ErrStatsLoss
# from source.models.devl.criteria_wrappers.loc_loss import LocLoss

# elif key == cu.LOC_LOSS:
#     cw = LocLoss()

# elif key == cu.ERR_STATS_LOSS:
# cw = ErrStatsLoss(cfg_mode, value)

# elif key == cu.CAT_STATS_LOSS:
#     cw = CatStatsLoss(cfg_mode, value)
# from source.models.devl.criteria_wrappers.cat_stats_loss import CatStatsLoss

#
# elif value.wrapper == SHI_CAT:
#     modules_wrappers.append(ShiCatWrapper(value))
#
# elif value.wrapper == SHI_CAT_DENSE:
#     modules_wrappers.append(ShiCatDenseWrapper(value))