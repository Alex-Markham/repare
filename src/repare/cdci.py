"""Adapted from https://github.com/baosws/CDCI/blob/main/CDCI.py"""

from collections import Counter

import numpy as np

EPSILON = 1e-8


def cond_dist(x, y, max_dev=3):
    vmax = 2 * max_dev
    vmin = -2 * max_dev

    x = (x - x.mean()) / (x.std() + EPSILON)
    t = x[np.abs(x) < max_dev]
    x = (x - t.mean()) / (t.std() + EPSILON)
    xd = np.round(x * 2)
    xd[xd > vmax] = vmax
    xd[xd < vmin] = vmin

    x_count = Counter(xd)
    vrange = range(vmin, vmax + 1)

    pyx = []
    for x in x_count:
        if x_count[x] > 12:
            yx = y[xd == x]
            yx = (yx - np.mean(yx)) / (np.std(yx) + EPSILON)
            yx = np.round(yx * 2)
            yx[yx > vmax] = vmax
            yx[yx < vmin] = vmin
            count_yx = Counter(yx)
            pyx_x = np.array([count_yx[i] for i in vrange], dtype=np.float64)
            pyx_x = pyx_x / pyx_x.sum()
            pyx.append(pyx_x)
    return pyx


def CKL(A, B, **kargs):
    """Causal score via Kullback-Leibler divergence"""
    pyx = cond_dist(A, B, **kargs)
    if len(pyx) == 0:
        return 0
    pyx = np.array(pyx)  # axis 0: x; axis 1: y
    mean_y = pyx.mean(axis=0)
    return (pyx * np.log((pyx + EPSILON) / (mean_y + EPSILON))).sum(axis=1).mean()


def CKM(A, B, **kargs):
    """Causal score via Kolmogorov metric"""
    pyx = cond_dist(A, B, **kargs)
    if len(pyx) == 0:
        return 0
    pyx = np.array(pyx)  # axis 0: x; axis 1: y
    mean_y = pyx.mean(axis=0).cumsum()
    pyx = pyx.cumsum(axis=1)

    return np.abs(pyx - mean_y).max(axis=1).mean()


def CHD(A, B, **kargs):
    """Causal score via Hellinger Distance"""
    pyx = cond_dist(A, B, **kargs)
    if len(pyx) == 0:
        return 0
    pyx = np.array(pyx)  # axis 0: x; axis 1: y
    mean_y = pyx.mean(axis=0)
    return (((pyx**0.5 - mean_y**0.5) ** 2).sum(axis=1) ** 0.5).mean()


def CCS(A, B, **kargs):
    """Causal score via Chi-Squared distance"""
    pyx = cond_dist(A, B, **kargs)
    if len(pyx) == 0:
        return 0
    pyx = np.array(pyx)  # axis 0: x; axis 1: y
    mean_y = pyx.mean(axis=0)
    return ((pyx - mean_y) ** 2 / (mean_y + EPSILON)).sum(axis=1).mean()


def CTV(A, B, **kargs):
    """Causal score via Total Variation distance"""
    pyx = cond_dist(A, B, **kargs)
    if len(pyx) == 0:
        return 0
    pyx = np.array(pyx)  # axis 0: x; axis 1: y
    mean_y = pyx.mean(axis=0)
    return 0.5 * np.abs(pyx - mean_y).sum(axis=1).mean()


def causal_score(variant, A, B, **kargs):
    variant = eval(variant)
    return variant(B, A, **kargs) - variant(A, B, **kargs)
