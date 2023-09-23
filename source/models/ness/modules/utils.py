import torch


def pad2topk(t, n, topk):
    offset = topk - n

    if offset == 0:
        return t

    else:
        return torch.cat([t, torch.zeros(t.shape[0], offset, t.shape[2], device=t.device)], dim=1)