"""
Train the Jev-clone encoder on real labeled data, then evaluate accuracy
and calibration (ECE) BEFORE any calibration fix — this is the "naive"
baseline that calibrate.py will improve on.
"""
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import JevCloneEncoder

torch.manual_seed(42)

with open("data/prepared.pkl", "rb") as f:
    data = pickle.load(f)

vocab = data["vocab"]
max_len = data["max_len"]


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


train_ds = SpamDataset(data["train"])
val_ds = SpamDataset(data["val"])
test_ds = SpamDataset(data["test"])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = JevCloneEncoder(vocab_size=len(vocab), max_len=max_len).to(device)

# Class weighting for imbalance (747 spam vs 4825 ham)
n_pos = sum(r["label"] for r in data["train"])
n_neg = len(data["train"]) - n_pos
pos_weight = torch.tensor([n_neg / n_pos]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

EPOCHS = 15
best_val_acc = 0
best_state = None

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for ids, mask, label in train_loader:
        ids, mask, label = ids.to(device), mask.to(device), label.to(device)
        optimizer.zero_grad()
        logit = model(ids, mask)
        loss = criterion(logit, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * ids.size(0)

    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for ids, mask, label in val_loader:
            ids, mask, label = ids.to(device), mask.to(device), label.to(device)
            logit = model(ids, mask)
            pred = (torch.sigmoid(logit) > 0.5).float()
            correct += (pred == label).sum().item()
            n += label.size(0)
    val_acc = correct / n
    print(f"Epoch {epoch+1:2d}  train_loss={total_loss/len(train_ds):.4f}  val_acc={val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
torch.save(model.state_dict(), "jev_clone.pt")
print(f"\nBest val acc: {best_val_acc:.4f} — saved jev_clone.pt")


def evaluate_calibration(model, loader, n_bins=10, label=""):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    correct, n = 0, 0
    with torch.no_grad():
        for ids, mask, lbl in loader:
            ids, mask = ids.to(device), mask.to(device)
            logit = model(ids, mask)
            prob = torch.sigmoid(logit).cpu()
            pred = (prob > 0.5).float()
            conf = torch.where(pred == 1, prob, 1 - prob)
            all_probs.extend(conf.tolist())
            all_preds.extend(pred.tolist())
            all_labels.extend(lbl.tolist())
            correct += (pred == lbl).sum().item()
            n += lbl.size(0)
    acc = correct / n

    # Expected Calibration Error (ECE)
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_report = []
    for i in range(n_bins):
        lo, hi = bins[i].item(), bins[i + 1].item()
        idxs = [j for j, c in enumerate(all_probs) if lo < c <= hi or (i == 0 and c == lo)]
        if not idxs:
            continue
        bin_conf = sum(all_probs[j] for j in idxs) / len(idxs)
        bin_acc = sum(1 for j in idxs if all_preds[j] == all_labels[j]) / len(idxs)
        weight = len(idxs) / len(all_probs)
        ece += weight * abs(bin_acc - bin_conf)
        bin_report.append((lo, hi, len(idxs), bin_conf, bin_acc))

    print(f"\n[{label}] Accuracy: {acc:.4f}   ECE: {ece:.4f}")
    print(f"{'bin':>12} {'n':>6} {'avg_conf':>9} {'avg_acc':>9}")
    for lo, hi, cnt, bc, ba in bin_report:
        print(f"{lo:.1f}-{hi:.1f}    {cnt:6d}   {bc:.4f}    {ba:.4f}")
    return acc, ece, all_probs, all_preds, all_labels


print("\n" + "=" * 50)
print("CALIBRATION CHECK — BEFORE any calibration fix")
print("=" * 50)
test_acc, test_ece, probs, preds, labels = evaluate_calibration(model, test_loader, label="test (uncalibrated)")

with open("eval_raw.pkl", "wb") as f:
    pickle.dump({"acc": test_acc, "ece": test_ece, "probs": probs, "preds": preds, "labels": labels}, f)
