import torch

from source.utils.endpoint_utils import nms, flat2grid

CONF_SCORE = 'conf_score'


def select_kp(score, conf_score,
              nms_size,  score_thresh, conf_thresh,
              k,
              return_score=False):
    nms_score = nms(score, nms_size)

    if score_thresh is not None:
        nms_score = nms_score * (score >= score_thresh).float()

    if conf_thresh is not None:
        nms_score = nms_score * (conf_score >= conf_thresh).float()

    nms_score = nms_score * conf_score

    if k == -1:
        raise NotImplementedError()

    else:
        kp_score, flat_kp = torch.topk(nms_score.view(nms_score.shape[0], -1), k)

    kp_score_mask = (kp_score >= 1e-8).squeeze()
    flat_kp = flat_kp[:, kp_score_mask]

    kp = flat2grid(flat_kp, score.shape[-1]).float() + 0.5

    if return_score:
        kp_score = kp_score[:, kp_score_mask]

        return kp_score, kp

    else:
        return kp
