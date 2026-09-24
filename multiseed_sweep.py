"""
Workstream E: mean pooling (baseline) vs learned attention pooling on the
multitask model, across several training seeds. Closes two gaps FINDINGS.md
names: seed variance had only been measured for Noul (single-task), and
the pooling comparison was a single seed per variant.

The data split is held FIXED (data/prepared_multitask.pkl); only the
training seed (weight init + batch order) varies. That isolates the
effect of the architecture from the effect of which examples landed in
test. Seed 42 reuses jev_clone_multitask.pt / jev_clone_attnpool.pt when
present; any missing checkpoint is trained here with train_multitask's
recipe (seed 42 mean pooling reproduces train_multitask.py exactly).

Usage: python3 multiseed_sweep.py
"""
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from calib_utils import ece as ece_score
from model import JevCloneEncoder
from train_multitask import (TaskDataset, data, device, eval_choice, eval_noul,
                             max_len, num_choice_classes, train_model, vocab_size)

SEEDS = [42, 123, 2024]
EXISTING = {("mean", 42): "jev_clone_multitask.pt", ("attn", 42): "jev_clone_attnpool.pt"}


def factory(pooling):
    return lambda: JevCloneEncoder(vocab_size=vocab_size, max_len=max_len,
                                   num_choice_classes=num_choice_classes, enable_score=True,
                                   pooling=pooling)


FACTORIES = {"mean": factory("mean"), "attn": factory("attn")}


def loader(task, split, dtype):
    return DataLoader(TaskDataset(data[task][split], dtype), batch_size=64)


def choice_logits(model, split):
    model.eval()
    out, labels = [], []
    with torch.no_grad():
        for ids, mask, label in loader("choice", split, torch.long):
            out.append(model.choice_head(model.encode(ids.to(device), mask.to(device))).cpu())
            labels.append(label)
    return torch.cat(out), torch.cat(labels)


def calibrated_choice_ece(model):
    val_logits, val_labels = choice_logits(model, "val")
    test_logits, test_labels = choice_logits(model, "test")
    T = nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=200)
    ce = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = ce(val_logits / T, val_labels)
        loss.backward()
        return loss

    opt.step(closure)
    probs = torch.softmax(test_logits / T.item(), dim=-1)
    conf, pred = probs.max(dim=-1)
    return ece_score(conf.tolist(), (pred == test_labels).float().tolist())


def score_metrics(model):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for ids, mask, label in loader("score", "test", torch.float):
            preds.append(torch.sigmoid(model.score_head(model.encode(ids.to(device), mask.to(device))).squeeze(-1)).cpu())
            labels.append(label)
    p, y = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return float(np.abs(p - y).mean()), float(np.corrcoef(p, y)[0, 1])


def evaluate(model):
    mae, r = score_metrics(model)
    return {
        "noul_acc": eval_noul(model, loader("noul", "test", torch.float)),
        "choice_acc": eval_choice(model, loader("choice", "test", torch.long)),
        "choice_ece_cal": calibrated_choice_ece(model),
        "score_mae": mae,
        "score_r": r,
    }


def get_model(variant, seed):
    path = EXISTING.get((variant, seed), f"jev_clone_{variant}pool_seed{seed}.pt")
    if os.path.exists(path):
        model = FACTORIES[variant]().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"[{variant} seed={seed}] loaded {path}")
        return model
    print(f"[{variant} seed={seed}] training -> {path}")
    return train_model(seed=seed, model_factory=FACTORIES[variant], save_path=path)


def summarize(values):
    a = np.array(values)
    return a.mean(), a.std(ddof=1)


if __name__ == "__main__":
    results = {}
    for seed in SEEDS:
        for variant in ("mean", "attn"):
            results[(variant, seed)] = evaluate(get_model(variant, seed))
            print(f"[{variant} seed={seed}] {results[(variant, seed)]}")

    metrics = ["noul_acc", "choice_acc", "choice_ece_cal", "score_mae", "score_r"]
    print("\n" + "=" * 78)
    print(f"Test-set metrics over training seeds {SEEDS} (fixed data split); "
          f"mean +/- sample std (n={len(SEEDS)})")
    print("=" * 78)
    print(f"{'Metric':<18}{'Mean-pool':>20}{'Attn-pool':>20}{'Paired diff (attn-mean)':>26}")
    for m in metrics:
        mm, ms = summarize([results[("mean", s)][m] for s in SEEDS])
        am, as_ = summarize([results[("attn", s)][m] for s in SEEDS])
        dm, ds = summarize([results[("attn", s)][m] - results[("mean", s)][m] for s in SEEDS])
        print(f"{m:<18}{mm:>12.4f} +/- {ms:.4f}{am:>12.4f} +/- {as_:.4f}{dm:>+16.4f} +/- {ds:.4f}")

    with open("multiseed_results.pkl", "wb") as f:
        pickle.dump({"seeds": SEEDS, "results": results}, f)
    print("\nSaved multiseed_results.pkl")
