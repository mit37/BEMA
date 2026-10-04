"""
Classical baseline: TF-IDF features (word 1-2 grams + character 2-5 grams)
with a linear model, on the same splits as the multitask model.

  Noul   (spam/ham):   logistic regression
  Choice (BANKING77):  multinomial logistic regression
  Score  (ratings):    ridge regression, predictions clipped to [0, 1]

The regularization strength is chosen on each task's validation split;
test is scored once. For Choice, confidence is calibrated with a
temperature fitted on validation, and typo robustness is measured with
the same perturbations as heldout_noise_test.py.

Usage: python3 linear_baseline.py
"""
import pickle

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_union

from calib_utils import ece, fit_temperature
from heldout_noise_test import perturbed
from train_multitask import data

NOISES = ("swap2", "keyboard2", "mixed3")


def split(task, name):
    rows = data[task][name]
    return [r["text"] for r in rows], np.array([r["label"] for r in rows])


def features():
    return make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2))


def fit_best(task, make_model, grid, score):
    (xtr, ytr), (xva, yva) = split(task, "train"), split(task, "val")
    vec = features()
    Xtr, Xva = vec.fit_transform(xtr), vec.transform(xva)
    fitted = [(c, make_model(c).fit(Xtr, ytr)) for c in grid]
    c, model = max(fitted, key=lambda cm: score(cm[1], Xva, yva))
    return vec, model, c


def acc(model, X, y):
    return (model.predict(X) == y).mean()


def neg_mae(model, X, y):
    return -np.abs(np.clip(model.predict(X), 0, 1) - y).mean()


if __name__ == "__main__":
    results = {}

    vec, model, c = fit_best("noul", lambda c: LogisticRegression(C=c, max_iter=3000, class_weight="balanced"),
                             (1, 10, 100), acc)
    xte, yte = split("noul", "test")
    results["noul_acc"] = acc(model, vec.transform(xte), yte)
    print(f"Noul   (C={c}): test accuracy {results['noul_acc']:.4f}  (n={len(yte)})")

    vec, model, c = fit_best("choice", lambda c: LogisticRegression(C=c, max_iter=3000), (5, 20, 50), acc)
    xva, yva = split("choice", "val")
    xte, yte = split("choice", "test")
    log_p = lambda texts: torch.tensor(np.log(np.clip(model.predict_proba(vec.transform(texts)), 1e-12, 1)))
    T = fit_temperature(log_p(xva).float(), torch.tensor(yva))
    for name in ("clean",) + NOISES:
        probs = torch.softmax(log_p(xte if name == "clean" else perturbed(xte, name)).float() / T, -1)
        conf, pred = probs.max(-1)
        correct = (pred.numpy() == yte).astype(float)
        results[f"{name}_acc"] = correct.mean()
        results[f"{name}_overconf"] = conf.mean().item() - correct.mean()
        results[f"{name}_ece"] = ece(conf.tolist(), correct.tolist())
    print(f"Choice (C={c}, T={T:.3f}): test accuracy {results['clean_acc']:.4f}, "
          f"calibrated ECE {results['clean_ece']:.4f}  (n={len(yte)})")
    for name in NOISES:
        print(f"  {name:<10} accuracy {results[f'{name}_acc']:.4f}  "
              f"drop {results['clean_acc'] - results[f'{name}_acc']:.4f}  "
              f"overconfidence {results[f'{name}_overconf']:+.4f}  ECE {results[f'{name}_ece']:.4f}")

    vec, model, c = fit_best("score", lambda a: Ridge(alpha=a), (0.3, 1, 3), neg_mae)
    xte, yte = split("score", "test")
    pred = np.clip(model.predict(vec.transform(xte)), 0, 1)
    results["score_mae"] = np.abs(pred - yte).mean()
    results["score_r"] = np.corrcoef(pred, yte)[0, 1]
    print(f"Score  (alpha={c}): test MAE {results['score_mae']:.4f}, Pearson r {results['score_r']:.4f}  (n={len(yte)})")

    with open("linear_baseline_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved linear_baseline_results.pkl")
