import numpy as np
import pytest

from calib_utils import (aps_prediction_set_sizes_and_coverage, aps_score, conformal_quantile,
                         ece, split_val)


def test_ece_is_zero_when_confidence_matches_accuracy():
    assert ece([0.75] * 4 + [0.25] * 4, [1, 1, 1, 0] + [1, 0, 0, 0]) == pytest.approx(0.0)


def test_ece_is_one_when_certain_and_always_wrong():
    assert ece([1.0] * 5, [0] * 5) == pytest.approx(1.0)


def test_ece_puts_zero_confidence_in_the_first_bin():
    assert ece([0.0, 0.0], [0, 0]) == pytest.approx(0.0)


def test_conformal_quantile_clips_to_max_when_calibration_set_is_tiny():
    assert conformal_quantile(np.array([0.1, 0.5, 0.3]), alpha=0.1) == 0.5


def test_conformal_quantile_gives_the_promised_coverage():
    rng = np.random.default_rng(0)
    n, alpha, trials = 99, 0.1, 3000
    covered = []
    for _ in range(trials):
        cal, test = rng.normal(size=n), rng.normal(size=200)
        covered.append(np.mean(test <= conformal_quantile(cal, alpha)))
    # Split conformal guarantees 1-alpha <= coverage <= 1-alpha + 1/(n+1) on average.
    assert 1 - alpha - 0.005 <= np.mean(covered) <= 1 - alpha + 1 / (n + 1) + 0.005


def test_aps_score_is_mass_up_to_and_including_the_true_class():
    probs = np.array([[0.1, 0.6, 0.3]])
    assert aps_score(probs, [1])[0] == pytest.approx(0.6)
    assert aps_score(probs, [2])[0] == pytest.approx(0.9)
    assert aps_score(probs, [0])[0] == pytest.approx(1.0)


def test_aps_sets_always_contain_the_top_class_and_grow_with_qhat():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(10), size=50)
    top = probs.argmax(1)
    small, covered = aps_prediction_set_sizes_and_coverage(probs, top, qhat=0.0)
    assert np.all(small == 1) and np.all(covered)
    big, _ = aps_prediction_set_sizes_and_coverage(probs, top, qhat=1.0)
    assert np.all(big == 10)
    mid, _ = aps_prediction_set_sizes_and_coverage(probs, top, qhat=0.7)
    assert np.all((small <= mid) & (mid <= big))


def test_aps_reaches_target_coverage_when_labels_follow_the_probabilities():
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.ones(20) * 0.5, size=6000)
    labels = np.array([rng.choice(20, p=p) for p in probs])
    qhat = conformal_quantile(aps_score(probs[:3000], labels[:3000]), alpha=0.1)
    _, covered = aps_prediction_set_sizes_and_coverage(probs[3000:], labels[3000:], qhat)
    assert covered.mean() >= 0.88


def test_split_val_is_deterministic_disjoint_and_complete():
    a1, b1 = split_val(range(101), seed=5)
    a2, b2 = split_val(range(101), seed=5)
    assert (a1, b1) == (a2, b2)
    assert len(a1) == 50 and sorted(a1 + b1) == list(range(101))
