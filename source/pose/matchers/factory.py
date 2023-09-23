import source.pose.matchers.namespace as m_ns

from source.pose.matchers.l2 import L2Matcher
from source.pose.matchers.hamming import HammingMatcher


def instantiate_matcher(model_mode_eval_params, device):
    matcher_eval_params = model_mode_eval_params.matcher
    matcher_name = matcher_eval_params.name

    if matcher_name == m_ns.L2:
        return L2Matcher(matcher_eval_params.lowe_ratio, device)

    elif matcher_name == m_ns.HAMMING:
        return HammingMatcher(matcher_eval_params.lowe_ratio)

    else:
        raise ValueError(f"{matcher} - unknown matcher")
