"""
Phase 4 follow-up: FINDINGS.md section 3 flagged that no run in this repo
had measured uncertainty on its accuracy/ECE numbers -- every number was
from a single train/val/test split and a single training run. This script
closes that gap for the single-task Noul (spam) pipeline: re-run data
splitting + training + calibration across multiple random seeds, and log
the empirical spread (not a single point estimate) to results_log.csv.

This directly answers "how much should I trust any single ECE number
quoted elsewhere in this repo" with a real measured range instead of an
assumption.
"""
import csv
import os
import random
import re
import pickle
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer

from model import JevCloneEncoder

RESULTS_PATH = "results_log.csv"
SEEDS = [42, 123, 2024]
MAX_LEN = 48
VOCAB_SIZE = 4000


def build_split(seed):
    random.seed(seed)
    rows = []
    with open("data/spam.csv", encoding="latin-1") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            if len(r) < 2:
                continue
            label, text = r[0], r[1]
            if label not in ("ham", "spam"):
                continue
            rows.append((text, 1 if label == "spam" else 0))
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train = rows[:n_train]
    val = rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:]

    corpus_path = f"data/_seed{seed}_corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for text, _ in train:
            f.write(text.replace("\n", " ") + "\n")
    tok = ByteLevelBPETokenizer()
    tok.train(files=[corpus_path], vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=["<pad>", "<unk>"])
    os.remove(corpus_path)
    pad_id = tok.token_to_id("<pad>")

    def encode(text):
        ids = tok.encode(text).ids[:MAX_LEN]
        mask = [1] * len(ids)
        pad_n = MAX_LEN - len(ids)
        return ids + [pad_id] * pad_n, mask + [0] * pad_n

    def build(rows):
        out = []
        for text, label in rows:
            ids, mask = encode(text)
            out.append({"ids": ids, "mask": mask, "label": label})
        return out

    return {
        "train": build(train), "val": build(val), "test": build(test),
        "vocab_size": tok.get_vocab_size(),
    }


class SpamDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return (
            torch.tensor(r["ids"], dtype=torch.long),
            torch.tensor(r["mask"], dtype=torch.long),
            torch.tensor(r["label"], dtype=torch.float),
        )


def ece_score(confidences, correct, n_bins=10):
    bins = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = bins[i].item(), bins[i + 1].item()
        idxs = [j for j, c in enumerate(confidences) if lo < c <= hi or (i == 0 and c == lo)]
        if not idxs:
            continue
        bin_conf = sum(confidences[j] for j in idxs) / len(idxs)
        bin_acc = sum(correct[j] for j in idxs) / len(idxs)
        total += (len(idxs) / len(confidences)) * abs(bin_acc - bin_conf)
    return total


def run_seed(seed):
    torch.manual_seed(seed)
    data = build_split(seed)
    device = "cpu"

    train_loader = DataLoader(SpamDataset(data["train"]), batch_size=32, shuffle=True)
    val_loader = DataLoader(SpamDataset(data["val"]), batch_size=64)
    test_loader = DataLoader(SpamDataset(data["test"]), batch_size=64)

    model = JevCloneEncoder(vocab_size=data["vocab_size"], max_len=MAX_LEN).to(device)
    n_pos = sum(r["label"] for r in data["train"])
    n_neg = len(data["train"]) - n_pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_val_acc, best_state = 0, None
    for epoch in range(15):
        model.train()
        for ids, mask, label in train_loader:
            optimizer.zero_grad()
            loss = criterion(model.forward(ids, mask), label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for ids, mask, label in val_loader:
                pred = (torch.sigmoid(model.forward(ids, mask)) > 0.5).float()
                correct += (pred == label).sum().item()
                n += label.size(0)
        val_acc = correct / n
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    # Raw ECE on test
    model.eval()
    val_logits, val_labels = [], []
    with torch.no_grad():
        for ids, mask, label in val_loader:
            val_logits.append(model.forward(ids, mask))
            val_labels.append(label)
    val_logits, val_labels = torch.cat(val_logits), torch.cat(val_labels)

    T = nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=200)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(val_logits / T, val_labels)
        loss.backward()
        return loss

    opt.step(closure)

    def evaluate(calibrated):
        confs, correct_list = [], []
        correct, n = 0, 0
        with torch.no_grad():
            for ids, mask, label in test_loader:
                logit = model.forward(ids, mask)
                if calibrated:
                    logit = logit / T
                prob = torch.sigmoid(logit)
                pred = (prob > 0.5).float()
                conf = torch.where(pred == 1, prob, 1 - prob)
                confs.extend(conf.tolist())
                correct_list.extend((pred == label).tolist())
                correct += (pred == label).sum().item()
                n += label.size(0)
        return correct / n, ece_score(confs, correct_list)

    raw_acc, raw_ece = evaluate(calibrated=False)
    cal_acc, cal_ece = evaluate(calibrated=True)
    return raw_acc, raw_ece, cal_acc, cal_ece, T.item(), len(data["test"])


if __name__ == "__main__":
    results = []
    for seed in SEEDS:
        print(f"\n=== seed={seed} ===")
        raw_acc, raw_ece, cal_acc, cal_ece, T, n_test = run_seed(seed)
        print(f"raw:  acc={raw_acc:.4f} ece={raw_ece:.4f}")
        print(f"cal:  acc={cal_acc:.4f} ece={cal_ece:.4f}  (T={T:.3f}, n_test={n_test})")
        results.append((seed, raw_acc, raw_ece, cal_acc, cal_ece, T, n_test))

    raw_accs = [r[1] for r in results]
    raw_eces = [r[2] for r in results]
    cal_accs = [r[3] for r in results]
    cal_eces = [r[4] for r in results]

    def stats(xs):
        m = sum(xs) / len(xs)
        # Sample variance (Bessel's correction, divide by n-1): the mean is
        # itself estimated from these same n points, so dividing by n would
        # be the biased population-variance estimator and understate std
        # for small n (about 18% low at n=3).
        denom = len(xs) - 1 if len(xs) > 1 else 1
        var = sum((x - m) ** 2 for x in xs) / denom
        return m, var ** 0.5, min(xs), max(xs)

    print("\n" + "=" * 70)
    print(f"SUMMARY across {len(SEEDS)} seeds {SEEDS} (single-task Noul/spam pipeline)")
    print("=" * 70)
    for name, xs in [("raw_acc", raw_accs), ("raw_ece", raw_eces), ("cal_acc", cal_accs), ("cal_ece", cal_eces)]:
        m, sd, lo, hi = stats(xs)
        print(f"{name:10s} mean={m:.4f}  std={sd:.4f}  range=[{lo:.4f}, {hi:.4f}]")

    ts = datetime.now(timezone.utc).isoformat()
    file_exists = os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp_utc", "task", "method", "accuracy", "ece", "n_test"])
        for seed, raw_acc, raw_ece, cal_acc, cal_ece, T, n_test in results:
            writer.writerow([ts, "noul_seed_sweep", f"raw(seed={seed})", f"{raw_acc:.4f}", f"{raw_ece:.4f}", n_test])
            writer.writerow([ts, "noul_seed_sweep", f"temperature(seed={seed},T={T:.3f})", f"{cal_acc:.4f}", f"{cal_ece:.4f}", n_test])
    print(f"\nAppended {2*len(SEEDS)} rows to {RESULTS_PATH}")
