"""
Calibration pass: fit a single temperature parameter T on the VALIDATION
split (never test) so that sigmoid(logit / T) produces confidence scores
that actually match empirical accuracy — the core claim Jev makes
("calibrated probabilities"). Then re-measure ECE on the held-out test
split, which the temperature was never fit on, and report before/after.
"""
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import JevCloneEncoder
from train import SpamDataset, data, vocab_size, max_len, device

model = JevCloneEncoder(vocab_size=vocab_size, max_len=max_len).to(device)
model.load_state_dict(torch.load("jev_clone.pt", map_location=device))
model.eval()

val_loader = DataLoader(SpamDataset(data["val"]), batch_size=64)
test_loader = DataLoader(SpamDataset(data["test"]), batch_size=64)

# Collect raw logits + labels on val split
val_logits, val_labels = [], []
with torch.no_grad():
    for ids, mask, label in val_loader:
        ids, mask = ids.to(device), mask.to(device)
        logit = model(ids, mask)
        val_logits.append(logit.cpu())
        val_labels.append(label)
val_logits = torch.cat(val_logits)
val_labels = torch.cat(val_labels)

# Fit temperature T by minimizing NLL on validation set (standard
# temperature-scaling calibration method, Guo et al. 2017)
temperature = nn.Parameter(torch.ones(1) * 1.5)
optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=200)
criterion = nn.BCEWithLogitsLoss()


def closure():
    optimizer.zero_grad()
    loss = criterion(val_logits / temperature, val_labels)
    loss.backward()
    return loss


optimizer.step(closure)
learned_T = temperature.item()
print(f"Learned temperature T = {learned_T:.4f}")

model.temperature.data = torch.tensor([learned_T])
torch.save(model.state_dict(), "jev_clone_calibrated.pt")

print("\n" + "=" * 50)
print("BEFORE calibration (T=1.0, raw sigmoid)")
print("=" * 50)
with open("eval_raw.pkl", "rb") as f:
    raw = pickle.load(f)
print(f"Accuracy: {raw['acc']:.4f}   ECE: {raw['ece']:.4f}")

print("\n" + "=" * 50)
print("AFTER calibration (temperature scaling, fit on val, measured on held-out test)")
print("=" * 50)


def evaluate_calibrated(model, loader, n_bins=10):
    model.eval()
    all_conf, all_preds, all_labels = [], [], []
    correct, n = 0, 0
    with torch.no_grad():
        for ids, mask, lbl in loader:
            ids, mask = ids.to(device), mask.to(device)
            is_spam, confidence, prob_spam = model.predict(ids, mask)
            pred = is_spam.float().cpu()
            conf = confidence.cpu()
            all_conf.extend(conf.tolist())
            all_preds.extend(pred.tolist())
            all_labels.extend(lbl.tolist())
            correct += (pred == lbl).sum().item()
            n += lbl.size(0)
    acc = correct / n

    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_report = []
    for i in range(n_bins):
        lo, hi = bins[i].item(), bins[i + 1].item()
        idxs = [j for j, c in enumerate(all_conf) if lo < c <= hi or (i == 0 and c == lo)]
        if not idxs:
            continue
        bin_conf = sum(all_conf[j] for j in idxs) / len(idxs)
        bin_acc = sum(1 for j in idxs if all_preds[j] == all_labels[j]) / len(idxs)
        weight = len(idxs) / len(all_conf)
        ece += weight * abs(bin_acc - bin_conf)
        bin_report.append((lo, hi, len(idxs), bin_conf, bin_acc))

    print(f"Accuracy: {acc:.4f}   ECE: {ece:.4f}")
    print(f"{'bin':>12} {'n':>6} {'avg_conf':>9} {'avg_acc':>9}")
    for lo, hi, cnt, bc, ba in bin_report:
        print(f"{lo:.1f}-{hi:.1f}    {cnt:6d}   {bc:.4f}    {ba:.4f}")
    return acc, ece


calibrated_acc, calibrated_ece = evaluate_calibrated(model, test_loader)

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"{'':20} {'Accuracy':>10} {'ECE':>10}")
print(f"{'Before (T=1.0)':20} {raw['acc']:>10.4f} {raw['ece']:>10.4f}")
print(f"{'After (T='+f'{learned_T:.2f})':20} {calibrated_acc:>10.4f} {calibrated_ece:>10.4f}")
print(f"\nECE improvement: {(1 - calibrated_ece/raw['ece'])*100:.1f}% reduction")
