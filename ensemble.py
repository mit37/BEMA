"""
Ensembles for accuracy: the three tuned n-gram BEMA seeds averaged, and
that average mixed with the TF-IDF linear baseline.

Each member's Choice probabilities are temperature-calibrated on
validation; the ensemble averages probabilities. The linear model's
mixing weight w (0 = n-gram only, 1 = linear only) is chosen on Choice
validation accuracy, and the ensemble's own temperature is then refitted
on validation. Noul averages probabilities, Score averages predictions,
with the same w. Test is scored once.

Needs the checkpoints from:
  python3 train_ngram.py --d-model 256 --dropout 0.4 --sparse-lr 3e-3 --epochs 8 --tag _tuned
Usage: python3 ensemble.py
"""
import argparse
import pickle

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge

from calib_utils import ece, fit_temperature
from heldout_noise_test import perturbed
from linear_baseline import acc, fit_best, neg_mae, split
from ngram_model import NgramEncoder, ngram_features
from train_multitask import num_choice_classes
from train_ngram import NOISES, SEEDS, predict

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_ngram(cfg, seed):
    model = NgramEncoder(d_model=cfg.d_model, num_choice_classes=num_choice_classes, enable_score=True)
    model.load_state_dict(torch.load(f"bema_ngram{cfg.tag}_seed{seed}.pt"))
    return model.eval()


def fit_linear():
    """The linear baseline, with its regularization chosen on validation exactly as in linear_baseline.py."""
    grids = {"noul": (lambda c: LogisticRegression(C=c, max_iter=3000, class_weight="balanced"), (1, 10, 100), acc),
             "choice": (lambda c: LogisticRegression(C=c, max_iter=3000), (5, 20, 50), acc),
             "score": (lambda a: Ridge(alpha=a), (0.3, 1, 3), neg_mae)}
    return {task: fit_best(task, *g)[:2] for task, g in grids.items()}


def linear_outputs(lin, task, texts):
    vec, model = lin[task]
    X = vec.transform(texts)
    if task == "score":
        return torch.tensor(np.clip(model.predict(X), 0, 1)).float()
    p = torch.tensor(model.predict_proba(X)).float()
    return p[:, 1] if task == "noul" else p


def ngram_outputs(models, temps, task, texts):
    feats = [ngram_features(t, models[0].buckets) for t in texts]
    outs = [predict(m, feats)[task] for m in models]
    if task == "noul":
        return torch.stack([torch.sigmoid(o) for o in outs]).mean(0)
    if task == "score":
        return torch.stack(outs).mean(0)
    return torch.stack([torch.softmax(o / T, -1) for o, T in zip(outs, temps)]).mean(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--tag", default="_tuned")
    cfg = ap.parse_args()

    models = [load_ngram(cfg, s) for s in SEEDS]
    lin = fit_linear()
    xv, yv = split("choice", "val")
    temps = [fit_temperature(predict(m, [ngram_features(t, m.buckets) for t in xv])["choice"], torch.tensor(yv))
             for m in models]
    lin_T = fit_temperature(torch.log(linear_outputs(lin, "choice", xv).clamp_min(1e-12)), torch.tensor(yv))

    def choice_probs(texts, w):
        p_lin = torch.softmax(torch.log(linear_outputs(lin, "choice", texts).clamp_min(1e-12)) / lin_T, -1)
        return (1 - w) * ngram_outputs(models, temps, "choice", texts) + w * p_lin

    val_acc = {w: (choice_probs(xv, w).argmax(-1).numpy() == yv).mean() for w in WEIGHTS}
    for w in WEIGHTS:
        print(f"  w={w:.2f}  Choice val accuracy {val_acc[w]:.4f}")
    w = max(WEIGHTS, key=lambda k: (val_acc[k], -abs(k - 0.5)))
    T = fit_temperature(torch.log(choice_probs(xv, w).clamp_min(1e-12)), torch.tensor(yv))
    print(f"Chosen on validation: w={w}, ensemble temperature {T:.3f}")

    results = {"w": w}
    for name, ww in (("ngram3", 0.0), ("mix", w)):
        xn, yn = split("noul", "test")
        pn = (1 - ww) * ngram_outputs(models, temps, "noul", xn) + ww * linear_outputs(lin, "noul", xn)
        results[f"{name}_noul_acc"] = ((pn > 0.5).numpy() == yn).mean()
        xs, ys = split("score", "test")
        ps = ((1 - ww) * ngram_outputs(models, temps, "score", xs) + ww * linear_outputs(lin, "score", xs)).numpy()
        results[f"{name}_score_mae"] = np.abs(ps - ys).mean()
        results[f"{name}_score_r"] = np.corrcoef(ps, ys)[0, 1]
        xt, yt = split("choice", "test")
        Tn = T if name == "mix" else fit_temperature(
            torch.log(choice_probs(xv, 0.0).clamp_min(1e-12)), torch.tensor(yv))
        for noise in ("clean",) + NOISES:
            t = xt if noise == "clean" else perturbed(xt, noise)
            probs = torch.softmax(torch.log(choice_probs(t, ww).clamp_min(1e-12)) / Tn, -1)
            conf, pred = probs.max(-1)
            correct = (pred.numpy() == yt).astype(float)
            results[f"{name}_{noise}_acc"] = correct.mean()
            results[f"{name}_{noise}_overconf"] = conf.mean().item() - correct.mean()
            results[f"{name}_{noise}_ece"] = ece(conf.tolist(), correct.tolist())
        print(f"\n{name} (w={ww}) test: Noul {results[f'{name}_noul_acc']:.4f} (n={len(yn)}), "
              f"Score MAE {results[f'{name}_score_mae']:.4f} r {results[f'{name}_score_r']:.4f} (n={len(ys)}), "
              f"Choice n={len(yt)}:")
        for noise in ("clean",) + NOISES:
            print(f"  {noise:<10} acc {results[f'{name}_{noise}_acc']:.4f}  "
                  f"overconf {results[f'{name}_{noise}_overconf']:+.4f}  ECE {results[f'{name}_{noise}_ece']:.4f}")
    with open("ensemble_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved ensemble_results.pkl")
