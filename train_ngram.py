"""
Train and evaluate BEMA with the n-gram trunk (ngram_model.py), using the
same data, losses, checkpoint selection and evaluation as the transformer
runs, so the results are directly comparable:

  - joint multitask training, one batch per task per step, combined
    validation score for checkpoint selection (as train_multitask.py);
  - test accuracy / calibrated ECE / Score MAE and r (as multiseed_sweep.py);
  - typo robustness with the same perturbations (as heldout_noise_test.py).

Seeds 42, 123, 2024. The embedding table is sparse (SparseAdam); the
heads use AdamW.

Usage: python3 train_ngram.py
"""
import math
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn

from calib_utils import ece, fit_temperature
from heldout_noise_test import perturbed
from ngram_model import NgramEncoder, collate, ngram_features
from train_multitask import data, num_choice_classes

SEEDS = [42, 123, 2024]
EPOCHS = 12
BATCH = 32
NOISES = ("swap2", "keyboard2", "mixed3")


def featurized(task, split, buckets):
    rows = data[task][split]
    return [ngram_features(r["text"], buckets) for r in rows], [r["label"] for r in rows]


def batches(feats, labels, gen):
    order = torch.randperm(len(feats), generator=gen).tolist()
    while True:
        for i in range(0, len(order) - BATCH + 1, BATCH):
            idx = order[i:i + BATCH]
            yield collate([feats[j] for j in idx]), torch.tensor([labels[j] for j in idx])
        order = torch.randperm(len(feats), generator=gen).tolist()


def predict(model, feats, chunk=512):
    model.eval()
    out = {"noul": [], "choice": [], "score": []}
    with torch.no_grad():
        for i in range(0, len(feats), chunk):
            h = model.encode(*collate(feats[i:i + chunk]))
            out["noul"].append(model.noul_head(h).squeeze(-1))
            out["choice"].append(model.choice_head(h))
            out["score"].append(torch.sigmoid(model.score_head(h).squeeze(-1)))
    return {k: torch.cat(v) for k, v in out.items()}


def train(seed, path):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = NgramEncoder(num_choice_classes=num_choice_classes, enable_score=True)
    B = model.buckets
    tr = {t: featurized(t, "train", B) for t in ("noul", "choice", "score")}
    va = {t: featurized(t, "val", B) for t in ("noul", "choice", "score")}
    its = {t: batches(*tr[t], gen) for t in tr}

    n_pos = sum(tr["noul"][1])
    noul_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(tr["noul"][1]) - n_pos) / n_pos]))
    sparse_opt = torch.optim.SparseAdam(list(model.bag.parameters()), lr=3e-3)
    dense = [p for n, p in model.named_parameters() if not n.startswith("bag.") and p.requires_grad]
    dense_opt = torch.optim.AdamW(dense, lr=1e-3, weight_decay=1e-4)
    steps = math.ceil(len(tr["choice"][0]) / BATCH)

    best, best_state = -1.0, None
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for _ in range(steps):
            (xn, yn), (xc, yc), (xs, ys) = next(its["noul"]), next(its["choice"]), next(its["score"])
            loss = (noul_loss(model.noul_head(model.encode(*xn)).squeeze(-1), yn.float())
                    + nn.functional.cross_entropy(model.choice_head(model.encode(*xc)), yc)
                    + nn.functional.mse_loss(torch.sigmoid(model.score_head(model.encode(*xs)).squeeze(-1)), ys.float()))
            sparse_opt.zero_grad()
            dense_opt.zero_grad()
            loss.backward()
            sparse_opt.step()
            dense_opt.step()
            total += loss.item()
        n_acc = ((predict(model, va["noul"][0])["noul"] > 0).float().numpy() == np.array(va["noul"][1])).mean()
        c_acc = (predict(model, va["choice"][0])["choice"].argmax(-1).numpy() == np.array(va["choice"][1])).mean()
        s_mae = np.abs(predict(model, va["score"][0])["score"].numpy() - np.array(va["score"][1])).mean()
        print(f"[seed {seed}] epoch {epoch + 1:2d} loss {total / steps:.4f}  noul_val {n_acc:.4f}  "
              f"choice_val {c_acc:.4f}  score_val_mae {s_mae:.4f}", flush=True)
        if n_acc + c_acc + (1 - s_mae) > best:
            best = n_acc + c_acc + (1 - s_mae)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), path)
    return model


def evaluate(model):
    B = model.buckets
    r = {}
    xn, yn = featurized("noul", "test", B)
    r["noul_acc"] = ((predict(model, xn)["noul"] > 0).float().numpy() == np.array(yn)).mean()
    xs, ys = featurized("score", "test", B)
    ps = predict(model, xs)["score"].numpy()
    r["score_mae"] = np.abs(ps - np.array(ys)).mean()
    r["score_r"] = np.corrcoef(ps, ys)[0, 1]

    xv, yv = featurized("choice", "val", B)
    T = fit_temperature(predict(model, xv)["choice"], torch.tensor(yv))
    texts = [row["text"] for row in data["choice"]["test"]]
    y = np.array([row["label"] for row in data["choice"]["test"]])
    for name in ("clean",) + NOISES:
        t = texts if name == "clean" else perturbed(texts, name)
        probs = torch.softmax(predict(model, [ngram_features(s, B) for s in t])["choice"] / T, -1)
        conf, pred = probs.max(-1)
        correct = (pred.numpy() == y).astype(float)
        r[f"{name}_acc"] = correct.mean()
        r[f"{name}_overconf"] = conf.mean().item() - correct.mean()
        r[f"{name}_ece"] = ece(conf.tolist(), correct.tolist())
    for name in NOISES:
        r[f"{name}_drop"] = r["clean_acc"] - r[f"{name}_acc"]
    return r


if __name__ == "__main__":
    random.seed(0)
    results = {}
    for seed in SEEDS:
        path = f"bema_ngram_seed{seed}.pt"
        if os.path.exists(path):
            model = NgramEncoder(num_choice_classes=num_choice_classes, enable_score=True)
            model.load_state_dict(torch.load(path))
        else:
            model = train(seed, path)
        results[seed] = evaluate(model)
        print(f"[seed {seed}] " + " ".join(f"{k}={v:.4f}" for k, v in results[seed].items()), flush=True)

    keys = ["noul_acc", "clean_acc", "clean_ece", "score_mae", "score_r"]
    keys += [f"{n}_{k}" for n in NOISES for k in ("acc", "drop", "overconf")]
    print(f"\nTest metrics, mean +/- sample std over seeds {SEEDS}:")
    for k in keys:
        v = np.array([results[s][k] for s in SEEDS])
        print(f"  {k:<18}{v.mean():.4f} +/- {v.std(ddof=1):.4f}")
    with open("ngram_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved ngram_results.pkl")
