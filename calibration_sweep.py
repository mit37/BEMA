"""
Phase 4: automated calibration-tuning loop.

For the Noul (spam) and Choice (banking77) classification heads of the
trained multi-task model, sweep three calibration methods -- temperature
scaling (Guo et al. 2017), Platt scaling (logistic regression on the
logit), and isotonic regression -- each fit on the VALIDATION split and
measured on the held-out TEST split, and log every result to
results_log.csv so progress is inspectable without re-running everything.

This does NOT retrain the encoder; it only re-fits the calibration layer
on top of a fixed trained model, which is cheap enough to sweep multiple
methods and (if re-run later) multiple checkpoints.

Usage: python3 train_multitask.py && python3 calibration_sweep.py
"""
import csv
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from model import JevCloneEncoder
from train_multitask import TaskDataset, data, vocab_size, max_len, num_choice_classes, device

RESULTS_PATH = "results_log.csv"

model = JevCloneEncoder(
    vocab_size=vocab_size, max_len=max_len,
    num_choice_classes=num_choice_classes, enable_score=True,
).to(device)
model.load_state_dict(torch.load("jev_clone_multitask.pt", map_location=device))
model.eval()


def ece_from_conf_correct(confidences, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    confidences = np.asarray(confidences)
    correct = np.asarray(correct)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        total += (mask.sum() / len(confidences)) * abs(bin_acc - bin_conf)
    return total


def collect_noul_logits(loader):
    logits, labels = [], []
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask = ids.to(device), mask.to(device)
            logits.append(model.forward(ids, mask).cpu())
            labels.append(label)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def collect_choice_logits(loader):
    logits, labels = [], []
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask = ids.to(device), mask.to(device)
            pooled = model.encode(ids, mask)
            logits.append(model.choice_head(pooled).cpu())
            labels.append(label)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature_binary(val_logits, val_labels):
    T = nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=200)
    vl = torch.tensor(val_logits, dtype=torch.float)
    vy = torch.tensor(val_labels, dtype=torch.float)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(vl / T, vy)
        loss.backward()
        return loss

    opt.step(closure)
    return T.item()


def fit_temperature_multiclass(val_logits, val_labels):
    T = nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=200)
    vl = torch.tensor(val_logits, dtype=torch.float)
    vy = torch.tensor(val_labels, dtype=torch.long)
    ce = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = ce(vl / T, vy)
        loss.backward()
        return loss

    opt.step(closure)
    return T.item()


def evaluate_binary(logits, labels, get_conf):
    """get_conf(logits) -> calibrated confidence for the predicted class."""
    probs = get_conf(logits)
    preds = (probs > 0.5).astype(int)
    conf = np.where(preds == 1, probs, 1 - probs)
    correct = (preds == labels).astype(int)
    acc = correct.mean()
    return acc, ece_from_conf_correct(conf, correct)


def evaluate_multiclass(logits, labels, get_preds_conf):
    """get_preds_conf(logits) -> (preds, conf). Kept as two explicit arrays,
    not a recalibrated probability matrix whose argmax is taken -- isotonic
    regression recalibrates the CONFIDENCE value only and must never be
    allowed to silently change which class was predicted (see calibration_sweep.py
    git history: an earlier version overwrote one class's probability in
    place and re-took argmax over the doctored array, which could flip the
    recorded prediction to a class the model never actually chose)."""
    preds, conf = get_preds_conf(logits)
    correct = (preds == labels).astype(int)
    acc = correct.mean()
    return acc, ece_from_conf_correct(conf, correct)


def sweep_task(task_name, val_logits, val_labels, test_logits, test_labels, multiclass):
    results = []

    if not multiclass:
        raw_acc, raw_ece = evaluate_binary(test_logits, test_labels, sigmoid)
        results.append(("raw", raw_acc, raw_ece))

        T = fit_temperature_binary(val_logits, val_labels)
        acc, ece = evaluate_binary(test_logits, test_labels, lambda l: sigmoid(l / T))
        results.append((f"temperature(T={T:.3f})", acc, ece))

        platt = LogisticRegression()
        platt.fit(val_logits.reshape(-1, 1), val_labels)
        acc, ece = evaluate_binary(
            test_logits, test_labels,
            lambda l: platt.predict_proba(l.reshape(-1, 1))[:, 1],
        )
        results.append(("platt", acc, ece))

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(sigmoid(val_logits), val_labels)
        acc, ece = evaluate_binary(test_logits, test_labels, lambda l: iso.predict(sigmoid(l)))
        results.append(("isotonic", acc, ece))
    else:
        def raw_preds_conf(l):
            p = softmax(l)
            return p.argmax(axis=-1), p.max(axis=-1)

        raw_acc, raw_ece = evaluate_multiclass(test_logits, test_labels, raw_preds_conf)
        results.append(("raw", raw_acc, raw_ece))

        T = fit_temperature_multiclass(val_logits, val_labels)

        def temp_preds_conf(l):
            p = softmax(l / T)
            return p.argmax(axis=-1), p.max(axis=-1)

        acc, ece = evaluate_multiclass(test_logits, test_labels, temp_preds_conf)
        results.append((f"temperature(T={T:.3f})", acc, ece))

        # Isotonic regression on the max-softmax confidence only (1-D),
        # a common simplification for multi-class ECE calibration. This
        # recalibrates the CONFIDENCE VALUE ONLY -- the prediction (argmax)
        # always comes from the raw, uncalibrated softmax and is never
        # touched by the isotonic fit, so isotonic scaling cannot change
        # which class is reported as predicted.
        val_probs = softmax(val_logits)
        val_conf = val_probs.max(axis=-1)
        val_correct = (val_probs.argmax(axis=-1) == val_labels).astype(int)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_conf, val_correct)

        def iso_preds_conf(l):
            p = softmax(l)
            preds = p.argmax(axis=-1)  # prediction: always from raw softmax
            conf = iso.predict(p.max(axis=-1))  # confidence: isotonic-recalibrated
            return preds, conf

        acc, ece = evaluate_multiclass(test_logits, test_labels, iso_preds_conf)
        results.append(("isotonic(max-conf)", acc, ece))

    return results


def log_results(rows):
    file_exists = os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp_utc", "task", "method", "accuracy", "ece", "n_test"])
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    noul_val_loader = DataLoader(TaskDataset(data["noul"]["val"], torch.float), batch_size=64)
    noul_test_loader = DataLoader(TaskDataset(data["noul"]["test"], torch.float), batch_size=64)
    choice_val_loader = DataLoader(TaskDataset(data["choice"]["val"], torch.long), batch_size=64)
    choice_test_loader = DataLoader(TaskDataset(data["choice"]["test"], torch.long), batch_size=64)

    noul_val_logits, noul_val_labels = collect_noul_logits(noul_val_loader)
    noul_test_logits, noul_test_labels = collect_noul_logits(noul_test_loader)
    choice_val_logits, choice_val_labels = collect_choice_logits(choice_val_loader)
    choice_test_logits, choice_test_labels = collect_choice_logits(choice_test_loader)

    ts = datetime.now(timezone.utc).isoformat()
    all_rows = []

    print("\n=== Noul (spam) calibration sweep ===")
    noul_results = sweep_task("noul", noul_val_logits, noul_val_labels, noul_test_logits, noul_test_labels, multiclass=False)
    for method, acc, ece in noul_results:
        print(f"{method:25s} acc={acc:.4f}  ece={ece:.4f}")
        all_rows.append([ts, "noul", method, f"{acc:.4f}", f"{ece:.4f}", len(noul_test_labels)])

    print("\n=== Choice (banking77) calibration sweep ===")
    choice_results = sweep_task("choice", choice_val_logits, choice_val_labels, choice_test_logits, choice_test_labels, multiclass=True)
    for method, acc, ece in choice_results:
        print(f"{method:25s} acc={acc:.4f}  ece={ece:.4f}")
        all_rows.append([ts, "choice", method, f"{acc:.4f}", f"{ece:.4f}", len(choice_test_labels)])

    log_results(all_rows)
    print(f"\nAppended {len(all_rows)} rows to {RESULTS_PATH}")
