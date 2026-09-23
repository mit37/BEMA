"""
Calibrate the multi-task model's classification heads (Noul, Choice) via
temperature scaling fit on each task's VALIDATION split (never test), then
report before/after ECE on each task's held-out TEST split.

The Score head is NOT temperature-scaled the same way: it's a regression
head, not a classifier, so "calibration" for it means something different
(does the predicted value's error distribution match what the model's
implicit uncertainty would suggest), which this toy setup does not
attempt to measure. We report MAE and Pearson correlation for Score
instead of ECE, and say so plainly rather than implying a calibration
number that isn't actually a calibration number.
"""
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import JevCloneEncoder
from calib_utils import ece
from train_multitask import TaskDataset, data, vocab_size, max_len, num_choice_classes, device

model = JevCloneEncoder(
    vocab_size=vocab_size, max_len=max_len,
    num_choice_classes=num_choice_classes, enable_score=True,
).to(device)
model.load_state_dict(torch.load("jev_clone_multitask.pt", map_location=device))
model.eval()



# ---------------------------------------------------------------------------
# Noul: temperature scaling on val, measured on held-out test
# ---------------------------------------------------------------------------
noul_val_loader = DataLoader(TaskDataset(data["noul"]["val"], torch.float), batch_size=64)
noul_test_loader = DataLoader(TaskDataset(data["noul"]["test"], torch.float), batch_size=64)

val_logits, val_labels = [], []
with torch.no_grad():
    for ids, mask, label in noul_val_loader:
        ids, mask = ids.to(device), mask.to(device)
        val_logits.append(model.forward(ids, mask).cpu())
        val_labels.append(label)
val_logits, val_labels = torch.cat(val_logits), torch.cat(val_labels)

T_noul = nn.Parameter(torch.ones(1) * 1.5)
opt = torch.optim.LBFGS([T_noul], lr=0.01, max_iter=200)
bce = nn.BCEWithLogitsLoss()


def closure():
    opt.zero_grad()
    loss = bce(val_logits / T_noul, val_labels)
    loss.backward()
    return loss


opt.step(closure)
model.temperature.data = torch.tensor([T_noul.item()])
print(f"Noul: learned temperature T = {T_noul.item():.4f}")


def eval_noul_calibration(loader, calibrated):
    confs, correct = [], []
    acc_n, acc_correct = 0, 0
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask = ids.to(device), mask.to(device)
            logit = model.forward(ids, mask)
            if calibrated:
                logit = logit / model.temperature
            prob = torch.sigmoid(logit).cpu()
            pred = (prob > 0.5).float()
            conf = torch.where(pred == 1, prob, 1 - prob)
            confs.extend(conf.tolist())
            correct.extend((pred == label).tolist())
            acc_n += label.size(0)
            acc_correct += (pred == label).sum().item()
    return acc_correct / acc_n, ece(confs, correct)


noul_acc_raw, noul_ece_raw = eval_noul_calibration(noul_test_loader, calibrated=False)
noul_acc_cal, noul_ece_cal = eval_noul_calibration(noul_test_loader, calibrated=True)
print(f"Noul  test  BEFORE: acc={noul_acc_raw:.4f} ece={noul_ece_raw:.4f}   "
      f"AFTER: acc={noul_acc_cal:.4f} ece={noul_ece_cal:.4f}")

# ---------------------------------------------------------------------------
# Choice: temperature scaling on val, measured on held-out test
# ---------------------------------------------------------------------------
choice_val_loader = DataLoader(TaskDataset(data["choice"]["val"], torch.long), batch_size=64)
choice_test_loader = DataLoader(TaskDataset(data["choice"]["test"], torch.long), batch_size=64)

val_logits_c, val_labels_c = [], []
with torch.no_grad():
    for ids, mask, label in choice_val_loader:
        ids, mask = ids.to(device), mask.to(device)
        pooled = model.encode(ids, mask)
        val_logits_c.append(model.choice_head(pooled).cpu())
        val_labels_c.append(label)
val_logits_c, val_labels_c = torch.cat(val_logits_c), torch.cat(val_labels_c)

T_choice = nn.Parameter(torch.ones(1) * 1.5)
opt_c = torch.optim.LBFGS([T_choice], lr=0.01, max_iter=200)
ce = nn.CrossEntropyLoss()


def closure_c():
    opt_c.zero_grad()
    loss = ce(val_logits_c / T_choice, val_labels_c)
    loss.backward()
    return loss


opt_c.step(closure_c)
model.choice_temperature.data = torch.tensor([T_choice.item()])
print(f"Choice: learned temperature T = {T_choice.item():.4f}")


def eval_choice_calibration(loader, calibrated):
    confs, correct = [], []
    acc_n, acc_correct = 0, 0
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask = ids.to(device), mask.to(device)
            pooled = model.encode(ids, mask)
            logits = model.choice_head(pooled)
            if calibrated:
                logits = logits / model.choice_temperature
            probs = torch.softmax(logits, dim=-1).cpu()
            conf, pred = probs.max(dim=-1)
            confs.extend(conf.tolist())
            correct.extend((pred == label).tolist())
            acc_n += label.size(0)
            acc_correct += (pred == label).sum().item()
    return acc_correct / acc_n, ece(confs, correct)


choice_acc_raw, choice_ece_raw = eval_choice_calibration(choice_test_loader, calibrated=False)
choice_acc_cal, choice_ece_cal = eval_choice_calibration(choice_test_loader, calibrated=True)
print(f"Choice test  BEFORE: acc={choice_acc_raw:.4f} ece={choice_ece_raw:.4f}   "
      f"AFTER: acc={choice_acc_cal:.4f} ece={choice_ece_cal:.4f}")

# ---------------------------------------------------------------------------
# Score: NOT temperature-scaled classification-style. Report regression
# metrics honestly (MAE, Pearson r) on held-out test -- no ECE claimed.
# ---------------------------------------------------------------------------
score_test_loader = DataLoader(TaskDataset(data["score"]["test"], torch.float), batch_size=64)
preds, labels = [], []
with torch.no_grad():
    for ids, mask, label in score_test_loader:
        ids, mask = ids.to(device), mask.to(device)
        pooled = model.encode(ids, mask)
        pred = torch.sigmoid(model.score_head(pooled).squeeze(-1)).cpu()
        preds.extend(pred.tolist())
        labels.extend(label.tolist())

preds_t, labels_t = torch.tensor(preds), torch.tensor(labels)
mae = (preds_t - labels_t).abs().mean().item()
pearson = torch.corrcoef(torch.stack([preds_t, labels_t]))[0, 1].item()
print(f"Score test   MAE={mae:.4f}  Pearson r={pearson:.4f}  (regression metrics, NOT an ECE)")

torch.save(model.state_dict(), "jev_clone_multitask_calibrated.pt")

print("\n" + "=" * 60)
print("SUMMARY (multi-task, shared encoder, held-out test splits)")
print("=" * 60)
print(f"{'Task':10} {'Metric':20} {'Before':>10} {'After':>10}")
print(f"{'Noul':10} {'acc / ece':20} {noul_acc_raw:.4f}/{noul_ece_raw:.4f}  {noul_acc_cal:.4f}/{noul_ece_cal:.4f}")
print(f"{'Choice':10} {'acc / ece':20} {choice_acc_raw:.4f}/{choice_ece_raw:.4f}  {choice_acc_cal:.4f}/{choice_ece_cal:.4f}")
print(f"{'Score':10} {'MAE / Pearson r':20} {mae:.4f}/{pearson:.4f}  (no calibration fit applied)")
