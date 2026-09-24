"""Calibration and split-conformal helpers shared by the evaluation scripts."""
import random

import numpy as np
import torch


def ece(confidences, correct, n_bins=10):
    """Expected calibration error with equal-width bins; bin 0 is [0, 1/n],
    the rest are (lo, hi]. Bin edges come from float32 torch.linspace to
    match the numbers reported earlier in this repo exactly."""
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    edges = torch.linspace(0, 1, n_bins + 1).tolist()
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (conf > lo) & (conf <= hi)
        if i == 0:
            in_bin |= conf == lo
        if in_bin.any():
            total += in_bin.mean() * abs(corr[in_bin].mean() - conf[in_bin].mean())
    return float(total)


def split_val(rows, frac_a=0.5, seed=123):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_a = int(len(rows) * frac_a)
    return rows[:n_a], rows[n_a:]


def conformal_quantile(scores, alpha):
    """Finite-sample-corrected split conformal quantile: the
    ceil((n+1)(1-alpha))/n empirical quantile of the calibration scores,
    clipped to the max score when that level exceeds 1."""
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    if q_level >= 1.0:
        return float(np.max(scores))
    return float(np.quantile(scores, q_level, method="higher"))


def aps_score(probs, true_labels):
    """Non-randomized APS nonconformity score (Romano, Sesia & Candes 2020):
    cumulative sorted probability mass up to and including the true class."""
    order = np.argsort(-probs, axis=1)
    cumsum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
    ranks = np.argmax(order == np.asarray(true_labels)[:, None], axis=1)
    return cumsum[np.arange(len(ranks)), ranks]


def aps_prediction_set_sizes_and_coverage(probs, true_labels, qhat):
    """APS prediction sets: classes in descending probability up to and
    including the first whose cumulative mass reaches qhat."""
    order = np.argsort(-probs, axis=1)
    cumsum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
    k = np.minimum((cumsum < qhat).sum(axis=1), probs.shape[1] - 1)
    true_rank = np.argmax(order == np.asarray(true_labels)[:, None], axis=1)
    return k + 1, true_rank <= k
