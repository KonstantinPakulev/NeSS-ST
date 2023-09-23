import source.pose.estimators.namespace as est_ns

from source.pose.estimators.fund_mat import FundMatEstimator
from source.pose.estimators.ess_mat import EssMatEstimator
from source.pose.estimators.homography import HomographyEstimator
from source.pose.estimators.colmap.estimator import COLMAPEstimator


def instantiate_estimator(model_mode_eval_params):
    estimator_name = model_mode_eval_params.estimator.name

    if estimator_name == est_ns.F_PYDEGENSAC:
        return  FundMatEstimator.from_config(model_mode_eval_params)

    elif estimator_name == est_ns.E_PYOPENGV:
        return  EssMatEstimator.from_config(model_mode_eval_params)

    elif estimator_name in [est_ns.H_OPENCV,
                            est_ns.H_PYDEGENSAC]:
        return  HomographyEstimator.from_config(model_mode_eval_params)
