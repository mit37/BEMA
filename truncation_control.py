"""
Control for an input-length confound in FINDINGS.md section 9.

The transformer reads only the first 64 BPE tokens of each input
(prepare_multitask.MAX_LEN); the TF-IDF and n-gram models read the whole
text. This refits the TF-IDF linear baseline (same features, grids and
validation selection as linear_baseline.py) on text cut to exactly what
the transformer sees, the first 64 tokens of the shared tokenizer, and
compares it with the full-text fit. It also reports how many inputs the
cut affects in each task.

Usage: python3 truncation_control.py
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from heldout_noise_test import max_len, tokenizer
from linear_baseline import features
from train_multitask import data

TASKS = {
    "noul": (lambda c: LogisticRegression(C=c, max_iter=3000, class_weight="balanced"), (1, 10, 100)),
    "choice": (lambda c: LogisticRegression(C=c, max_iter=3000), (5, 20, 50)),
    "score": (lambda a: Ridge(alpha=a), (0.3, 1, 3)),
}


def truncate(text):
    """The prefix of `text` covered by its first max_len tokens."""
    enc = tokenizer.encode(text)
    return text if len(enc.ids) <= max_len else text[:enc.offsets[max_len - 1][1]]


def predict(task, model, X):
    return np.clip(model.predict(X), 0, 1) if task == "score" else model.predict(X)


def val_score(task, y, p):
    return -np.abs(p - y).mean() if task == "score" else (p == y).mean()


def fit_and_test(task, view):
    make, grid = TASKS[task]
    xs = {s: [view(r["text"]) for r in data[task][s]] for s in ("train", "val", "test")}
    ys = {s: np.array([r["label"] for r in data[task][s]]) for s in ("train", "val", "test")}
    vec = features()
    Xtr = vec.fit_transform(xs["train"])
    fitted = [(c, make(c).fit(Xtr, ys["train"])) for c in grid]
    c, model = max(fitted, key=lambda cm: val_score(task, ys["val"], predict(task, cm[1], vec.transform(xs["val"]))))
    p, y = predict(task, model, vec.transform(xs["test"])), ys["test"]
    if task == "score":
        return {"c": c, "mae": np.abs(p - y).mean(), "r": np.corrcoef(p, y)[0, 1], "n": len(y)}
    return {"c": c, "acc": (p == y).mean(), "n": len(y)}


if __name__ == "__main__":
    results = {}
    for task in TASKS:
        cut = {s: np.mean([len(tokenizer.encode(r["text"]).ids) > max_len for r in data[task][s]])
               for s in ("train", "test")}
        full, short = fit_and_test(task, lambda t: t), fit_and_test(task, truncate)
        results[task] = {"share_truncated": cut, "full": full, "first_64_tokens": short}
        print(f"\n{task}: inputs longer than {max_len} tokens: train {cut['train']:.1%}, test {cut['test']:.1%} "
              f"(test n={full['n']})")
        for name, r in (("full text", full), (f"first {max_len} tokens", short)):
            metric = (f"MAE {r['mae']:.4f}  r {r['r']:.4f}" if task == "score" else f"accuracy {r['acc']:.4f}")
            print(f"  linear, {name:<16} (C/alpha={r['c']}): {metric}")
    with open("truncation_control_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nSaved truncation_control_results.pkl")
