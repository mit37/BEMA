"""
Workstream B: typo-noise data augmentation, to test whether it fixes the
calibration gap found in FINDINGS.md sec. 5 -- ordinary typos (2
adjacent-character swaps) collapse Choice accuracy 81.9%->51.2% while
confidence barely drops (0.791->0.612).

Approach: augment TRAINING data only. val/test splits are copied
byte-for-byte from the existing data/prepared_multitask.pkl (same
tokenizer, same encoding) so results are directly comparable to the
pre-fix baseline -- this experiment changes ONLY what the model was
trained on, nothing about how it's measured. For a fraction of TRAIN
rows in each task (Noul, Choice, Score), a typo-perturbed duplicate is
added alongside the original clean row, using the exact same
perturbation mechanism as adversarial_stress_test.py's Test 3 (2
adjacent-character swaps), re-encoded with the SAME already-trained
tokenizer (not retrained -- isolates the effect to "more typo-diverse
training data," not "different tokenizer vocabulary").

Usage: python3 augment_and_retrain.py
Produces: jev_clone_multitask_augmented.pt,
          jev_clone_multitask_augmented_calibrated.pt,
          augment_results.pkl
"""
import copy
import math
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from model import JevCloneEncoder

SEED = 42
AUGMENT_FRAC = 0.5  # fraction of train rows per task that get a typo'd duplicate added
torch.manual_seed(SEED)
random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("data/prepared_multitask.pkl", "rb") as f:
    base_data = pickle.load(f)

vocab_size = base_data["vocab_size"]
max_len = base_data["max_len"]
num_choice_classes = base_data["num_choice_classes"]
pad_id = base_data["pad_id"]
tokenizer = Tokenizer.from_file("data/tokenizer_multitask.json")


def perturb_typo(text, n_typos=2):
    """Identical mechanism to adversarial_stress_test.py's Test 3."""
    chars = list(text)
    for _ in range(n_typos):
        if len(chars) < 3:
            break
        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def encode(text):
    ids = tokenizer.encode(text).ids[:max_len]
    mask = [1] * len(ids)
    pad_n = max_len - len(ids)
    return ids + [pad_id] * pad_n, mask + [0] * pad_n


def augment_train_split(rows, frac):
    augmented = list(rows)  # keep all originals
    n_to_augment = int(len(rows) * frac)
    chosen = random.sample(rows, n_to_augment)
    for r in chosen:
        typo_text = perturb_typo(r["text"])
        ids, mask = encode(typo_text)
        augmented.append({"ids": ids, "mask": mask, "label": r["label"], "text": typo_text})
    return augmented


print("Building augmented training data (val/test copied unchanged from the baseline)...")
aug_data = {
    "noul": {
        "train": augment_train_split(base_data["noul"]["train"], AUGMENT_FRAC),
        "val": base_data["noul"]["val"],
        "test": base_data["noul"]["test"],
    },
    "choice": {
        "train": augment_train_split(base_data["choice"]["train"], AUGMENT_FRAC),
        "val": base_data["choice"]["val"],
        "test": base_data["choice"]["test"],
    },
    "score": {
        "train": augment_train_split(base_data["score"]["train"], AUGMENT_FRAC),
        "val": base_data["score"]["val"],
        "test": base_data["score"]["test"],
    },
    "num_choice_classes": num_choice_classes,
    "choice_categories": base_data["choice_categories"],
    "vocab_size": vocab_size,
    "max_len": max_len,
    "pad_id": pad_id,
}
for task in ["noul", "choice", "score"]:
    print(f"  {task}: train {len(base_data[task]['train'])} -> {len(aug_data[task]['train'])} "
          f"(+{len(aug_data[task]['train']) - len(base_data[task]['train'])} typo'd duplicates), "
          f"val {len(aug_data[task]['val'])} (unchanged), test {len(aug_data[task]['test'])} (unchanged)")


class TaskDataset(Dataset):
    def __init__(self, rows, label_dtype):
        self.rows = rows
        self.label_dtype = label_dtype

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return (
            torch.tensor(r["ids"], dtype=torch.long),
            torch.tensor(r["mask"], dtype=torch.long),
            torch.tensor(r["label"], dtype=self.label_dtype),
        )


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def eval_noul(model, loader):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask = ids.to(device), mask.to(device)
            pred = (torch.sigmoid(model.forward(ids, mask)) > 0.5).float().cpu()
            correct += (pred == label).sum().item()
            n += label.size(0)
    return correct / n


def eval_choice(model, loader):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask = ids.to(device), mask.to(device)
            pooled = model.encode(ids, mask)
            pred = model.choice_head(pooled).argmax(dim=-1).cpu()
            correct += (pred == label).sum().item()
            n += label.size(0)
    return correct / n


def eval_score_mae(model, loader):
    model.eval()
    total_abs_err, n = 0.0, 0
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask = ids.to(device), mask.to(device)
            pooled = model.encode(ids, mask)
            pred = torch.sigmoid(model.score_head(pooled).squeeze(-1)).cpu()
            total_abs_err += (pred - label).abs().sum().item()
            n += label.size(0)
    return total_abs_err / n


print("\nTraining on augmented data (same architecture/hyperparameters as train_multitask.py)...")

model = JevCloneEncoder(
    vocab_size=vocab_size, max_len=max_len,
    num_choice_classes=num_choice_classes, enable_score=True,
).to(device)

BATCH = 32
noul_loader = DataLoader(TaskDataset(aug_data["noul"]["train"], torch.float), batch_size=BATCH, shuffle=True, drop_last=True)
choice_loader = DataLoader(TaskDataset(aug_data["choice"]["train"], torch.long), batch_size=BATCH, shuffle=True, drop_last=True)
score_loader = DataLoader(TaskDataset(aug_data["score"]["train"], torch.float), batch_size=BATCH, shuffle=True, drop_last=True)
noul_iter, choice_iter, score_iter = infinite(noul_loader), infinite(choice_loader), infinite(score_loader)

steps_per_epoch = math.ceil(len(aug_data["choice"]["train"]) / BATCH)
EPOCHS = 12

n_pos = sum(r["label"] for r in aug_data["noul"]["train"])
n_neg = len(aug_data["noul"]["train"]) - n_pos
noul_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos]).to(device))
choice_criterion = nn.CrossEntropyLoss()
score_criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

noul_val_loader = DataLoader(TaskDataset(aug_data["noul"]["val"], torch.float), batch_size=64)
choice_val_loader = DataLoader(TaskDataset(aug_data["choice"]["val"], torch.long), batch_size=64)
score_val_loader = DataLoader(TaskDataset(aug_data["score"]["val"], torch.float), batch_size=64)

best_score_sum = -1
best_state = None

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for step in range(steps_per_epoch):
        ids_n, mask_n, label_n = next(noul_iter)
        ids_c, mask_c, label_c = next(choice_iter)
        ids_s, mask_s, label_s = next(score_iter)
        optimizer.zero_grad()

        pooled_n = model.encode(ids_n.to(device), mask_n.to(device))
        loss_n = noul_criterion(model.noul_head(pooled_n).squeeze(-1), label_n.to(device))
        pooled_c = model.encode(ids_c.to(device), mask_c.to(device))
        loss_c = choice_criterion(model.choice_head(pooled_c), label_c.to(device))
        pooled_s = model.encode(ids_s.to(device), mask_s.to(device))
        pred_s = torch.sigmoid(model.score_head(pooled_s).squeeze(-1))
        loss_s = score_criterion(pred_s, label_s.to(device))

        loss = loss_n + loss_c + loss_s
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    noul_acc = eval_noul(model, noul_val_loader)
    choice_acc = eval_choice(model, choice_val_loader)
    score_mae = eval_score_mae(model, score_val_loader)
    print(f"Epoch {epoch+1:2d}  loss={total_loss/steps_per_epoch:.4f}  "
          f"noul_val_acc={noul_acc:.4f}  choice_val_acc={choice_acc:.4f}  score_val_mae={score_mae:.4f}")

    combined = noul_acc + choice_acc + (1 - score_mae)
    if combined > best_score_sum:
        best_score_sum = combined
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
torch.save(model.state_dict(), "jev_clone_multitask_augmented.pt")
print(f"\nSaved jev_clone_multitask_augmented.pt (best combined val score={best_score_sum:.4f})")

# ---------------------------------------------------------------------------
# Calibrate (temperature scaling for Noul/Choice), same recipe as
# calibrate_multitask.py, on the SAME val split.
# ---------------------------------------------------------------------------
print("\nCalibrating (temperature scaling)...")


def fit_temperature_binary(loader):
    logits, labels = [], []
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask = ids.to(device), mask.to(device)
            logits.append(model.forward(ids, mask).cpu())
            labels.append(label)
    logits, labels = torch.cat(logits), torch.cat(labels)
    T = nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=200)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return T.item()


def fit_temperature_multiclass(loader):
    logits, labels = [], []
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask = ids.to(device), mask.to(device)
            pooled = model.encode(ids, mask)
            logits.append(model.choice_head(pooled).cpu())
            labels.append(label)
    logits, labels = torch.cat(logits), torch.cat(labels)
    T = nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=200)
    ce = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = ce(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return T.item()


model.temperature.data = torch.tensor([fit_temperature_binary(noul_val_loader)])
model.choice_temperature.data = torch.tensor([fit_temperature_multiclass(choice_val_loader)])
print(f"Noul temperature: {model.temperature.item():.4f}   Choice temperature: {model.choice_temperature.item():.4f}")
torch.save(model.state_dict(), "jev_clone_multitask_augmented_calibrated.pt")
print("Saved jev_clone_multitask_augmented_calibrated.pt")

# ---------------------------------------------------------------------------
# Evaluate: clean-test accuracy (robustness/accuracy tradeoff check) AND
# typo-perturbed accuracy/confidence (the actual fix check), same
# methodology as adversarial_stress_test.py's Test 3, on the SAME
# (unaugmented) test split for a direct, apples-to-apples comparison to
# the pre-fix baseline numbers already in FINDINGS.md.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("EVALUATION: clean-test accuracy (tradeoff check) + typo robustness (the actual fix check)")
print("=" * 70)

choice_test_rows = base_data["choice"]["test"]  # untouched, same as baseline eval

# Fix the typo perturbations ONCE so the baseline (pre-fix) model and the
# augmented model are evaluated on the EXACT SAME perturbed text -- a
# controlled comparison, not two different random draws of noise.
random.seed(SEED)
typo_texts = [perturb_typo(r["text"]) for r in choice_test_rows]
typo_encoded = [encode(t) for t in typo_texts]


def eval_choice_typo_robustness(eval_model):
    eval_model.eval()
    with torch.no_grad():
        base_correct, base_confs = 0, []
        typo_correct, typo_confs = 0, []
        for r, (ids2, mask2) in zip(choice_test_rows, typo_encoded):
            ids_t = torch.tensor([r["ids"]], dtype=torch.long).to(device)
            mask_t = torch.tensor([r["mask"]], dtype=torch.long).to(device)
            pooled = eval_model.encode(ids_t, mask_t)
            logits = eval_model.choice_head(pooled) / eval_model.choice_temperature
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            base_correct += int(pred.item() == r["label"])
            base_confs.append(conf.item())

            ids2_t = torch.tensor([ids2], dtype=torch.long).to(device)
            mask2_t = torch.tensor([mask2], dtype=torch.long).to(device)
            pooled2 = eval_model.encode(ids2_t, mask2_t)
            logits2 = eval_model.choice_head(pooled2) / eval_model.choice_temperature
            probs2 = torch.softmax(logits2, dim=-1)
            conf2, pred2 = probs2.max(dim=-1)
            typo_correct += int(pred2.item() == r["label"])
            typo_confs.append(conf2.item())
    n = len(choice_test_rows)
    return {
        "clean_acc": base_correct / n, "clean_conf": sum(base_confs) / n,
        "typo_acc": typo_correct / n, "typo_conf": sum(typo_confs) / n,
    }


aug_results = eval_choice_typo_robustness(model)
n_choice = len(choice_test_rows)
print(f"\nAUGMENTED MODEL, Choice head, n={n_choice}:")
print(f"  Clean queries    accuracy={aug_results['clean_acc']:.4f}  mean_confidence={aug_results['clean_conf']:.4f}")
print(f"  With typos       accuracy={aug_results['typo_acc']:.4f}  mean_confidence={aug_results['typo_conf']:.4f}")
print(f"  Accuracy drop under typos: {aug_results['clean_acc'] - aug_results['typo_acc']:.4f}")
print(f"  Confidence drop under typos: {aug_results['clean_conf'] - aug_results['typo_conf']:.4f}")

# Dynamically recompute the pre-fix baseline from the existing checkpoint
# (not hardcoded) -- avoids the exact staleness risk flagged in an earlier
# review of this repo (fix_score_head.py's original hardcoded-baseline bug).
print("\nLoading pre-fix baseline model for a live (not hardcoded) comparison...")
baseline_model = JevCloneEncoder(
    vocab_size=vocab_size, max_len=max_len,
    num_choice_classes=num_choice_classes, enable_score=True,
).to(device)
baseline_model.load_state_dict(torch.load("jev_clone_multitask_calibrated.pt", map_location=device))
baseline_results = eval_choice_typo_robustness(baseline_model)

print("\n" + "-" * 70)
print("COMPARISON: pre-fix baseline (freshly re-evaluated) vs typo-augmented model")
print("-" * 70)
print(f"{'Metric':30s} {'Baseline':>10s} {'Augmented':>10s} {'Delta':>10s}")
for key, label in [("clean_acc", "Clean-test accuracy"), ("clean_conf", "Clean-test confidence"),
                    ("typo_acc", "Typo-test accuracy"), ("typo_conf", "Typo-test confidence")]:
    b, a = baseline_results[key], aug_results[key]
    print(f"{label:30s} {b:>10.4f} {a:>10.4f} {a-b:>+10.4f}")
baseline_acc_drop = baseline_results["clean_acc"] - baseline_results["typo_acc"]
baseline_conf_drop = baseline_results["clean_conf"] - baseline_results["typo_conf"]
aug_acc_drop = aug_results["clean_acc"] - aug_results["typo_acc"]
aug_conf_drop = aug_results["clean_conf"] - aug_results["typo_conf"]
print(f"{'Accuracy drop under typos':30s} {baseline_acc_drop:>10.4f} {aug_acc_drop:>10.4f} {aug_acc_drop-baseline_acc_drop:>+10.4f}")
print(f"{'Confidence drop under typos':30s} {baseline_conf_drop:>10.4f} {aug_conf_drop:>10.4f} {aug_conf_drop-baseline_conf_drop:>+10.4f}")
print(f"\n{'Verdict:':10s} accuracy-collapse gap {'NARROWED' if aug_acc_drop < baseline_acc_drop - 0.01 else ('WIDENED' if aug_acc_drop > baseline_acc_drop + 0.01 else 'UNCHANGED')}, "
      f"calibration gap (conf drop vs acc drop mismatch) "
      f"{'NARROWED' if (aug_acc_drop - aug_conf_drop) < (baseline_acc_drop - baseline_conf_drop) - 0.01 else ('WIDENED' if (aug_acc_drop - aug_conf_drop) > (baseline_acc_drop - baseline_conf_drop) + 0.01 else 'UNCHANGED')}")

with open("augment_results.pkl", "wb") as f:
    pickle.dump({
        "augmented": aug_results, "baseline": baseline_results,
        "n_choice_test": n_choice, "augment_frac": AUGMENT_FRAC,
    }, f)
print("\nSaved augment_results.pkl")
