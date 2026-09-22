"""
Joint multi-task training: ONE shared encoder trunk, three typed heads
(Noul, Choice, Score), each supervised by its own REAL human-labeled
dataset (see prepare_multitask.py for provenance). Every optimizer step
draws one batch per task and backprops a combined loss through the SAME
shared trunk -- this is what makes "one encoder, many typed questions in
parallel" a real architectural claim rather than three bolted-together
models that happen to share a file.

Usage: python3 prepare_multitask.py && python3 train_multitask.py
"""
import math
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import JevCloneEncoder

torch.manual_seed(42)

with open("data/prepared_multitask.pkl", "rb") as f:
    data = pickle.load(f)

vocab_size = data["vocab_size"]
max_len = data["max_len"]
num_choice_classes = data["num_choice_classes"]
device = "cuda" if torch.cuda.is_available() else "cpu"


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
            ids, mask, label = ids.to(device), mask.to(device), label.to(device)
            logit = model.forward(ids, mask)
            pred = (torch.sigmoid(logit) > 0.5).float()
            correct += (pred == label).sum().item()
            n += label.size(0)
    return correct / n


def eval_choice(model, loader):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask, label = ids.to(device), mask.to(device), label.to(device)
            pooled = model.encode(ids, mask)
            logits = model.choice_head(pooled)
            pred = logits.argmax(dim=-1)
            correct += (pred == label).sum().item()
            n += label.size(0)
    return correct / n


def eval_score_mae(model, loader):
    model.eval()
    total_abs_err, n = 0.0, 0
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask, label = ids.to(device), mask.to(device), label.to(device)
            pooled = model.encode(ids, mask)
            pred = torch.sigmoid(model.score_head(pooled).squeeze(-1))
            total_abs_err += (pred - label).abs().sum().item()
            n += label.size(0)
    return total_abs_err / n


def train_model():
    noul_train = TaskDataset(data["noul"]["train"], torch.float)
    choice_train = TaskDataset(data["choice"]["train"], torch.long)
    score_train = TaskDataset(data["score"]["train"], torch.float)

    BATCH = 32
    noul_loader = DataLoader(noul_train, batch_size=BATCH, shuffle=True, drop_last=True)
    choice_loader = DataLoader(choice_train, batch_size=BATCH, shuffle=True, drop_last=True)
    score_loader = DataLoader(score_train, batch_size=BATCH, shuffle=True, drop_last=True)

    noul_iter = infinite(noul_loader)
    choice_iter = infinite(choice_loader)
    score_iter = infinite(score_loader)

    # One epoch = one pass over the LARGEST task dataset (choice, 8503 rows);
    # smaller tasks (noul 3900, score 5749) cycle around within an epoch.
    steps_per_epoch = math.ceil(len(choice_train) / BATCH)
    EPOCHS = 12

    model = JevCloneEncoder(
        vocab_size=vocab_size, max_len=max_len,
        num_choice_classes=num_choice_classes, enable_score=True,
    ).to(device)

    n_pos = sum(r["label"] for r in data["noul"]["train"])
    n_neg = len(data["noul"]["train"]) - n_pos
    noul_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos]).to(device))
    choice_criterion = nn.CrossEntropyLoss()
    score_criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    noul_val_loader = DataLoader(TaskDataset(data["noul"]["val"], torch.float), batch_size=64)
    choice_val_loader = DataLoader(TaskDataset(data["choice"]["val"], torch.long), batch_size=64)
    score_val_loader = DataLoader(TaskDataset(data["score"]["val"], torch.float), batch_size=64)

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

        # Track a combined score (lower MAE is better, so use 1-MAE) to pick
        # the best checkpoint across all three tasks jointly.
        combined = noul_acc + choice_acc + (1 - score_mae)
        if combined > best_score_sum:
            best_score_sum = combined
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), "jev_clone_multitask.pt")
    print(f"\nSaved jev_clone_multitask.pt (best combined val score={best_score_sum:.4f})")
    return model


if __name__ == "__main__":
    train_model()
