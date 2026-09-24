"""
Does the Choice head's conformal coverage guarantee survive typo noise?

Split conformal's 90% guarantee assumes calibration and test examples are
exchangeable (drawn the same way). Calibrating on clean text and then
serving typo'd text breaks that. This script measures what happens, for
the baseline and the typo-augmented model, using the same protocol as
conformal.py (temperature refit on val_a, APS calibrated on val_b, both
BANKING77 validation halves; test n=3080):

  calibrated on clean val_b  -> evaluated on clean / typo'd test
  calibrated on typo'd val_b -> evaluated on typo'd test (the obvious fix:
                                calibrate on data that looks like deployment)

Typo'd text uses heldout_noise_test.perturbed (fixed seeds).

Usage: python3 conformal_typo.py
"""
import numpy as np
import torch

from calib_utils import (aps_prediction_set_sizes_and_coverage, aps_score, conformal_quantile,
                         fit_temperature, split_val)
from heldout_noise_test import encode_batch, load, perturbed
from train_multitask import data, device

ALPHA = 0.10
NOISES = ("swap2", "keyboard2", "mixed3")


def logits_for(model, texts):
    ids, mask = encode_batch(texts)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            pooled = model.encode(ids[i:i + 256].to(device), mask[i:i + 256].to(device))
            out.append(model.choice_head(pooled).cpu())
    return torch.cat(out)


def probs(logits, T):
    return torch.softmax(logits / T, dim=-1).numpy()


if __name__ == "__main__":
    val_a, val_b = split_val(data["choice"]["val"])
    test = data["choice"]["test"]
    ya = torch.tensor([r["label"] for r in val_a])
    yb = np.array([r["label"] for r in val_b])
    yt = np.array([r["label"] for r in test])
    texts_b, texts_t = [r["text"] for r in val_b], [r["text"] for r in test]

    print(f"APS conformal, target coverage {1 - ALPHA:.0%}; val_a n={len(val_a)}, "
          f"val_b n={len(val_b)}, test n={len(test)}\n")
    print(f"{'model':<10}{'calibrated on':<15}{'test text':<11}{'coverage':>9}{'avg set':>9}{'median':>8}")
    for name, path in (("baseline", "jev_clone_multitask_calibrated.pt"),
                       ("augmented", "jev_clone_multitask_augmented_calibrated.pt")):
        model = load(path)
        T = fit_temperature(logits_for(model, [r["text"] for r in val_a]), ya)
        qhat_clean = conformal_quantile(aps_score(probs(logits_for(model, texts_b), T), yb), ALPHA)

        def report(calib, test_name, qhat, test_texts):
            sizes, covered = aps_prediction_set_sizes_and_coverage(
                probs(logits_for(model, test_texts), T), yt, qhat)
            print(f"{name:<10}{calib:<15}{test_name:<11}{covered.mean():>9.4f}"
                  f"{sizes.mean():>9.2f}{np.median(sizes):>8.0f}")

        report("clean", "clean", qhat_clean, texts_t)
        for noise in NOISES:
            report("clean", noise, qhat_clean, perturbed(texts_t, noise))
        for noise in NOISES:
            qhat_noisy = conformal_quantile(
                aps_score(probs(logits_for(model, perturbed(texts_b, noise)), T), yb), ALPHA)
            report(f"{noise}", noise, qhat_noisy, perturbed(texts_t, noise))
        print()
