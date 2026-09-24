"""
95% bootstrap confidence intervals for the headline test-set numbers.

Every metric in this repo is measured on one fixed test split (BANKING77's
is the dataset's official test set), so the remaining uncertainty is test-
set sampling: how much would the number move on a different draw of test
examples of the same size? This script resamples each test set with
replacement (B=2000, percentile intervals) using the existing checkpoints;
nothing is retrained. Differences between two models or two decision
rules are bootstrapped PAIRED (same resampled examples for both), which is
what makes a small difference measurable.

Usage: python3 bootstrap_ci.py  (needs the multitask, augmented and CLINC150
checkpoints produced by the other scripts)
"""
import numpy as np
import torch

import clinc150_oos_threshold as clinc
from calib_utils import ece
from heldout_noise_test import encode_batch, load, perturbed
from train_multitask import data, device

B = 2000
rng = np.random.default_rng(0)


def ci(stat, n):
    """Point estimate on the full set plus the 2.5/97.5 bootstrap percentiles."""
    idx = rng.integers(0, n, size=(B, n))
    boots = np.array([stat(i) for i in idx])
    return stat(np.arange(n)), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def rows_out(label, n, est):
    point, lo, hi = est
    print(f"  {label:<46}{point:>9.4f}   [{lo:.4f}, {hi:.4f}]   n={n}")


def predict(model, texts, fn):
    ids, mask = encode_batch(texts)
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            outs.append(fn(model, ids[i:i + 256].to(device), mask[i:i + 256].to(device)))
    return [torch.cat([o[k] for o in outs]).cpu().numpy() for k in range(len(outs[0]))]


def noul_fn(m, ids, mask):
    is_spam, conf, _ = m.predict(ids, mask)
    return is_spam.float(), conf


def choice_fn(m, ids, mask):
    cls, conf, _ = m.predict_choice(ids, mask)
    return cls, conf


def score_fn(m, ids, mask):
    return (m.predict_score(ids, mask),)


if __name__ == "__main__":
    base = load("jev_clone_multitask_calibrated.pt")
    aug = load("jev_clone_multitask_augmented_calibrated.pt")

    print(f"95% bootstrap intervals (B={B}, percentile), point estimate on the full test set\n")

    noul = data["noul"]["test"]
    y = np.array([r["label"] for r in noul])
    pred, conf = predict(base, [r["text"] for r in noul], noul_fn)
    correct = (pred == y).astype(float)
    print("Noul (SMS spam, calibrated baseline):")
    rows_out("accuracy", len(y), ci(lambda i: correct[i].mean(), len(y)))
    rows_out("ECE", len(y), ci(lambda i: ece(conf[i], correct[i]), len(y)))

    choice = data["choice"]["test"]
    texts = [r["text"] for r in choice]
    yc = np.array([r["label"] for r in choice])
    n = len(yc)
    cls_b, conf_b = predict(base, texts, choice_fn)
    cls_a, conf_a = predict(aug, texts, choice_fn)
    cor_b, cor_a = (cls_b == yc).astype(float), (cls_a == yc).astype(float)
    print("\nChoice (BANKING77, calibrated):")
    rows_out("baseline accuracy", n, ci(lambda i: cor_b[i].mean(), n))
    rows_out("baseline ECE", n, ci(lambda i: ece(conf_b[i], cor_b[i]), n))
    rows_out("augmented - baseline accuracy (clean text)", n, ci(lambda i: (cor_a[i] - cor_b[i]).mean(), n))

    for noise in ("swap2", "keyboard2", "mixed3"):
        noisy = perturbed(texts, noise)
        nb, _ = predict(base, noisy, choice_fn)
        na, _ = predict(aug, noisy, choice_fn)
        nb, na = (nb == yc).astype(float), (na == yc).astype(float)
        print(f"  -- {noise} typos --")
        rows_out(f"baseline accuracy drop, clean - {noise}", n, ci(lambda i: (cor_b[i] - nb[i]).mean(), n))
        rows_out(f"augmented - baseline accuracy on {noise}", n, ci(lambda i: (na[i] - nb[i]).mean(), n))

    score = data["score"]["test"]
    ys = np.array([r["label"] for r in score])
    (ps,) = predict(base, [r["text"] for r in score], score_fn)
    print("\nScore (Amazon Fine Food Reviews, baseline):")
    rows_out("MAE", len(ys), ci(lambda i: np.abs(ps[i] - ys[i]).mean(), len(ys)))
    rows_out("Pearson r", len(ys), ci(lambda i: np.corrcoef(ps[i], ys[i])[0, 1], len(ys)))

    val_p, val_y = clinc.probs_and_labels(clinc.data["val"])
    test_p, test_y = clinc.probs_and_labels(clinc.data["test"])
    tau = clinc.choose_tau(val_p, val_y)
    arg, thr = clinc.predict(test_p, 0.0), clinc.predict(test_p, tau)
    ins, oos = test_y != clinc.OOS, test_y == clinc.OOS
    hit_arg, hit_thr = (arg == test_y), (thr == test_y)
    nt = len(test_y)

    def within(mask, hits):
        return lambda i: hits[i][mask[i]].mean()

    print(f"\nCLINC150 out-of-scope threshold (tau={tau:.2f}, chosen on validation):")
    rows_out("oos recall, argmax only", int(oos.sum()), ci(within(oos, hit_arg), nt))
    rows_out("oos recall, with threshold", int(oos.sum()), ci(within(oos, hit_thr), nt))
    rows_out("in-scope accuracy, argmax only", int(ins.sum()), ci(within(ins, hit_arg), nt))
    rows_out("in-scope accuracy, with threshold", int(ins.sum()), ci(within(ins, hit_thr), nt))
    rows_out("in-scope accuracy change from threshold", int(ins.sum()),
             ci(lambda i: hit_thr[i][ins[i]].mean() - hit_arg[i][ins[i]].mean(), nt))
