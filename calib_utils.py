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


def _fit_inverse_temperature(loss_of_scaled_logits):
    """1/T minimizing loss(logits / T); the loss is convex in 1/T, and LBFGS uses a line search."""
    inv_t = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([inv_t], lr=1.0, max_iter=100, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        loss = loss_of_scaled_logits(inv_t)
        loss.backward()
        return loss

    opt.step(closure)
    return 1.0 / inv_t.item()


def fit_temperature(logits, labels):
    """Temperature T minimizing cross-entropy of logits / T (Guo et al. 2017).

    Until 2026-11-12 this ran LBFGS on T with a fixed step of 0.01 from
    T=1.5, which stopped short of the minimum; see FINDINGS.md section 10.
    """
    logits = logits.detach().float()
    return _fit_inverse_temperature(lambda inv_t: torch.nn.functional.cross_entropy(logits * inv_t, labels))


def fit_temperature_binary(logits, labels):
    """Temperature T minimizing binary cross-entropy of logits / T, for a single-logit head."""
    logits, labels = logits.detach().float(), labels.float()
    return _fit_inverse_temperature(
        lambda inv_t: torch.nn.functional.binary_cross_entropy_with_logits(logits * inv_t, labels))


def paired_bootstrap_diff(a, b, n_boot=2000, seed=0):
    """Mean of a - b with a 95% percentile interval, resampling examples jointly (paired)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)
