import numpy as np


def accuracy(err, max_threshold):
    """
    :param err: B or B x N
    :param max_threshold: int
    :param std: bool
    """
    angles = np.linspace(1, max_threshold, num=max_threshold)

    acc = []

    for a in angles:
        thr_mask = (err <= a).astype(np.float32)

        acci = np.mean(thr_mask, axis=0)
        acc.append(acci)

    return np.array(acc)