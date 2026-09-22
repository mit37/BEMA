"""
Workstream D: a second, independent Choice-type task on a genuinely
different domain, to test whether the typo-miscalibration finding
(FINDINGS.md section 5, from BANKING77) generalizes or was an artifact of
that one dataset.

Dataset: CLINC150 (`clinc/oos-eval`), Larson et al. 2019. 150 real,
crowdsourced intent categories spanning many domains (banking, travel,
utility, work, small-talk, etc. -- much broader than BANKING77's single
banking domain) PLUS an explicit, human-curated out-of-scope ("oos")
class of queries that don't belong to any of the 150 intents. License:
CC-BY 3.0, verified directly from the repo's own LICENSE file (see
DATA_LICENSES.md). Labels are the dataset's original crowd-annotated
intent labels, not LLM-generated or inferred by this repo.

This script is intentionally self-contained (prepare -> train ->
calibrate -> conformal -> adversarial typo test in one run), matching the
style of augment_and_retrain.py, since it's a standalone additional task
rather than part of the shared multitask trunk (a 151-class head this
different in domain doesn't share the multitask encoder's vocabulary or
training data).

Usage: python3 clinc150_pipeline.py
"""
import json
import math
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizers import ByteLevelBPETokenizer

from model import JevCloneEncoder

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

MAX_LEN = 32
VOCAB_SIZE = 4000
BATCH = 32
EPOCHS = 12
ALPHA = 0.10  # conformal target: 90% coverage

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------------
with open("data/raw/clinc150_data_full.json") as f:
    raw = json.load(f)

intents = sorted({label for _, label in raw["train"]})
assert len(intents) == 150, f"expected 150 in-scope intents, got {len(intents)}"
categories = intents + ["oos"]
label_to_idx = {name: i for i, name in enumerate(categories)}


def to_rows(split_name, oos_split_name):
    rows = [{"text": t, "label": label_to_idx[lab]} for t, lab in raw[split_name]]
    rows += [{"text": t, "label": label_to_idx["oos"]} for t, lab in raw[oos_split_name]]
    return rows


train_rows = to_rows("train", "oos_train")
val_rows = to_rows("val", "oos_val")
test_rows = to_rows("test", "oos_test")
random.Random(SEED).shuffle(train_rows)
print(f"CLINC150: {len(categories)} classes (150 intents + oos)  "
      f"train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}")

corpus_path = "data/_clinc150_corpus.txt"
with open(corpus_path, "w", encoding="utf-8") as f:
    for r in train_rows:
        f.write(r["text"].replace("\n", " ") + "\n")

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=[corpus_path], vocab_size=VOCAB_SIZE, min_frequency=2,
                 special_tokens=["<pad>", "<unk>"])
import os
os.remove(corpus_path)
pad_id = tokenizer.token_to_id("<pad>")
assert pad_id == 0
vocab_size = tokenizer.get_vocab_size()
tokenizer.save("data/tokenizer_clinc150.json")
print(f"Dedicated CLINC150 BPE vocab size: {vocab_size}")


def encode(text):
    enc = tokenizer.encode(text)
    ids = enc.ids[:MAX_LEN]
    mask = [1] * len(ids)
    pad_n = MAX_LEN - len(ids)
    ids = ids + [pad_id] * pad_n
    mask = mask + [0] * pad_n
    return ids, mask


def build(rows):
    out = []
    for r in rows:
        ids, mask = encode(r["text"])
        out.append({"ids": ids, "mask": mask, "label": r["label"], "text": r["text"]})
    return out


data = {
    "train": build(train_rows),
    "val": build(val_rows),
    "test": build(test_rows),
    "num_classes": len(categories),
    "categories": categories,
    "vocab_size": vocab_size,
    "max_len": MAX_LEN,
    "pad_id": pad_id,
}
with open("data/prepared_clinc150.pkl", "wb") as f:
    pickle.dump(data, f)
print("Saved data/prepared_clinc150.pkl and data/tokenizer_clinc150.json")


class ChoiceDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return (torch.tensor(r["ids"], dtype=torch.long),
                torch.tensor(r["mask"], dtype=torch.long),
                torch.tensor(r["label"], dtype=torch.long))


def eval_acc(model, loader):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask, label = ids.to(device), mask.to(device), label.to(device)
            pooled = model.encode(ids, mask)
            pred = model.choice_head(pooled).argmax(dim=-1)
            correct += (pred == label).sum().item()
            n += label.size(0)
    return correct / n


# ---------------------------------------------------------------------------
# 2. Train (Choice head only -- this is a standalone single-task model)
# ---------------------------------------------------------------------------
train_ds = ChoiceDataset(data["train"])
val_ds = ChoiceDataset(data["val"])
test_ds = ChoiceDataset(data["test"])

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

model = JevCloneEncoder(
    vocab_size=vocab_size, d_model=96, nhead=6, num_layers=3, max_len=MAX_LEN,
    num_choice_classes=len(categories), enable_score=False,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
steps_per_epoch = math.ceil(len(train_ds) / BATCH)

best_acc, best_state = -1.0, None
print("\nTraining CLINC150 Choice head (150 intents + oos, standalone encoder)...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for ids, mask, label in train_loader:
        ids, mask, label = ids.to(device), mask.to(device), label.to(device)
        optimizer.zero_grad()
        pooled = model.encode(ids, mask)
        logits = model.choice_head(pooled)
        loss = criterion(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    val_acc = eval_acc(model, val_loader)
    print(f"Epoch {epoch+1:2d}  loss={total_loss/steps_per_epoch:.4f}  val_acc={val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
torch.save(model.state_dict(), "jev_clone_clinc150.pt")
print(f"Saved jev_clone_clinc150.pt (best val_acc={best_acc:.4f})")

# ---------------------------------------------------------------------------
# 3. Calibrate (temperature scaling on val, ECE on raw held-out test)
# ---------------------------------------------------------------------------
def ece(confidences, correct, n_bins=10):
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


def collect_logits(loader):
    logits_all, labels_all = [], []
    model.eval()
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask = ids.to(device), mask.to(device)
            pooled = model.encode(ids, mask)
            logits_all.append(model.choice_head(pooled).cpu())
            labels_all.append(label)
    return torch.cat(logits_all), torch.cat(labels_all)


val_logits, val_labels = collect_logits(val_loader)
test_logits, test_labels = collect_logits(test_loader)


def conf_correct(logits, labels, T=1.0):
    probs = torch.softmax(logits / T, dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels)
    return conf.tolist(), correct.tolist(), (pred == labels).float().mean().item()


raw_conf, raw_correct, raw_acc = conf_correct(test_logits, test_labels, T=1.0)
raw_ece = ece(raw_conf, raw_correct)

T = nn.Parameter(torch.ones(1) * 1.5)
opt = torch.optim.LBFGS([T], lr=0.01, max_iter=200)
ce = nn.CrossEntropyLoss()


def closure():
    opt.zero_grad()
    loss = ce(val_logits / T, val_labels)
    loss.backward()
    return loss


opt.step(closure)
T_val = T.item()
print(f"\nLearned temperature T = {T_val:.4f}")

cal_conf, cal_correct, cal_acc = conf_correct(test_logits, test_labels, T=T_val)
cal_ece = ece(cal_conf, cal_correct)
print(f"Test accuracy: {raw_acc:.4f}  (unaffected by temperature scaling, as expected)")
print(f"ECE  raw={raw_ece:.4f}   calibrated={cal_ece:.4f}")

model.choice_temperature.data = torch.tensor([T_val])
torch.save(model.state_dict(), "jev_clone_clinc150_calibrated.pt")
print("Saved jev_clone_clinc150_calibrated.pt")

# ---------------------------------------------------------------------------
# 4. Split conformal prediction (APS), same method/hygiene as conformal.py
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SPLIT CONFORMAL PREDICTION (APS), target coverage 90%")
print("=" * 70)


def split_val(rows, frac_a=0.5, seed=123):
    rows = list(rows)
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    n_a = int(len(rows) * frac_a)
    return rows[:n_a], rows[n_a:]


val_a_rows, val_b_rows = split_val(data["val"])
val_a_loader = DataLoader(ChoiceDataset(val_a_rows), batch_size=64)
val_b_loader = DataLoader(ChoiceDataset(val_b_rows), batch_size=64)

val_a_logits, val_a_labels = collect_logits(val_a_loader)
T2 = nn.Parameter(torch.ones(1) * 1.5)
opt2 = torch.optim.LBFGS([T2], lr=0.01, max_iter=200)


def closure2():
    opt2.zero_grad()
    loss = ce(val_a_logits / T2, val_a_labels)
    loss.backward()
    return loss


opt2.step(closure2)
T_conformal = T2.item()
print(f"Temperature refit on val_a only (leakage-safe, n={len(val_a_rows)}): T = {T_conformal:.4f}")

val_b_logits, val_b_labels = collect_logits(val_b_loader)
probs_b = torch.softmax(val_b_logits / T_conformal, dim=-1).numpy()
y_b = val_b_labels.numpy()


def aps_score(probs, true_labels):
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    scores = np.zeros(len(true_labels))
    for i in range(len(true_labels)):
        rank = np.where(order[i] == true_labels[i])[0][0]
        scores[i] = cumsum[i, rank]
    return scores


def conformal_quantile(scores, alpha):
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    if q_level >= 1.0:
        return float(np.max(scores))
    return float(np.quantile(scores, q_level, method="higher"))


scores_b = aps_score(probs_b, y_b)
qhat = conformal_quantile(scores_b, ALPHA)
print(f"Calibration set n={len(val_b_rows)}   qhat={qhat:.4f}")

probs_test = torch.softmax(test_logits / T_conformal, dim=-1).numpy()
y_test = test_labels.numpy()


def aps_prediction_set_sizes_and_coverage(probs, true_labels, qhat):
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    set_sizes = np.zeros(len(true_labels), dtype=int)
    covered = np.zeros(len(true_labels), dtype=bool)
    for i in range(len(true_labels)):
        k = np.searchsorted(cumsum[i], qhat, side="left")
        k = min(k, len(cumsum[i]) - 1)
        set_sizes[i] = k + 1
        true_rank = np.where(order[i] == true_labels[i])[0][0]
        covered[i] = true_rank <= k
    return set_sizes, covered


set_sizes, covered = aps_prediction_set_sizes_and_coverage(probs_test, y_test, qhat)
coverage = covered.mean()
avg_set_size = set_sizes.mean()
print(f"\nTEST SET (n={len(test_rows)}):")
print(f"  Target coverage: {1-ALPHA:.0%}   Measured coverage: {coverage:.4f}")
print(f"  Average prediction set size: {avg_set_size:.2f}  (out of {len(categories)} possible classes)")
print(f"  Median set size: {np.median(set_sizes):.0f}   Max: {set_sizes.max()}   Min: {set_sizes.min()}")

# ---------------------------------------------------------------------------
# 5. Adversarial typo stress test -- does the BANKING77 finding generalize?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ADVERSARIAL TYPO STRESS TEST (same 2-adjacent-char-swap perturbation")
print("as adversarial_stress_test.py) -- does the typo-miscalibration")
print("finding generalize to a different domain/dataset?")
print("=" * 70)


def perturb_typo(text, n_typos=2, seed=None):
    rnd = random.Random(seed)
    chars = list(text)
    for _ in range(n_typos):
        if len(chars) < 3:
            break
        i = rnd.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


rnd = random.Random(7)
typo_rows = [{"text": perturb_typo(r["text"], seed=rnd.randint(0, 10**9)), "label": r["label"]}
             for r in test_rows]
typo_encoded = [encode(r["text"]) for r in typo_rows]

model.eval()
with torch.no_grad():
    clean_correct, clean_confs = 0, []
    typo_correct, typo_confs = 0, []
    for i in range(len(test_rows)):
        ids_c = torch.tensor([data["test"][i]["ids"]], dtype=torch.long).to(device)
        mask_c = torch.tensor([data["test"][i]["mask"]], dtype=torch.long).to(device)
        pooled_c = model.encode(ids_c, mask_c)
        logits_c = model.choice_head(pooled_c) / T_val
        prob_c = torch.softmax(logits_c, dim=-1)[0]
        conf_c, pred_c = prob_c.max(dim=-1)
        clean_correct += int(pred_c.item() == test_rows[i]["label"])
        clean_confs.append(conf_c.item())

        ids_t_raw, mask_t_raw = typo_encoded[i]
        ids_t = torch.tensor([ids_t_raw], dtype=torch.long).to(device)
        mask_t = torch.tensor([mask_t_raw], dtype=torch.long).to(device)
        pooled_t = model.encode(ids_t, mask_t)
        logits_t = model.choice_head(pooled_t) / T_val
        prob_t = torch.softmax(logits_t, dim=-1)[0]
        conf_t, pred_t = prob_t.max(dim=-1)
        typo_correct += int(pred_t.item() == typo_rows[i]["label"])
        typo_confs.append(conf_t.item())

n = len(test_rows)
clean_acc = clean_correct / n
clean_conf = sum(clean_confs) / n
typo_acc = typo_correct / n
typo_conf = sum(typo_confs) / n

print(f"n={n}")
print(f"Clean queries    accuracy={clean_acc:.4f}  mean_confidence={clean_conf:.4f}")
print(f"With typos       accuracy={typo_acc:.4f}  mean_confidence={typo_conf:.4f}")
print(f"Accuracy drop under typos:   {clean_acc-typo_acc:.4f}")
print(f"Confidence drop under typos: {clean_conf-typo_conf:.4f}")
if (clean_acc - typo_acc) > 0 and abs((clean_conf-typo_conf) - (clean_acc-typo_acc)) > 0.03:
    print("FINDING GENERALIZES: confidence drop does not track accuracy drop, "
          "same miscalibration pattern as BANKING77.")
else:
    print("FINDING DOES NOT CLEARLY GENERALIZE on this dataset/run (see numbers above).")

results = {
    "raw_acc": raw_acc, "raw_ece": raw_ece, "cal_ece": cal_ece,
    "conformal_coverage": coverage, "conformal_avg_set_size": avg_set_size,
    "conformal_n_classes": len(categories),
    "clean_acc": clean_acc, "clean_conf": clean_conf,
    "typo_acc": typo_acc, "typo_conf": typo_conf,
    "n_test": n,
}
with open("clinc150_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved clinc150_results.pkl")
