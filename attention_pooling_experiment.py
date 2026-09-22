"""
Workstream E (background, lowest priority): one scoped architecture
experiment -- does learned attention pooling beat mean pooling for the
shared multitask trunk? This project's rule for this workstream is to
report deltas honestly and expect small effects; only one variant is
tried here (not a full sweep), consistent with Workstream E being
explicitly lowest priority and only pursued given idle capacity after
Workstreams A-D.

Everything else (encoder depth/width, layer count, learning rate,
epochs, data) is held IDENTICAL to train_multitask.py -- the only change
is how the final hidden states are pooled into one state vector:
  - Baseline (jev_clone_multitask_calibrated.pt): mean-pool over
    non-padding tokens (model.py's existing encode()).
  - This experiment: a learned attention pool -- a single trainable
    query vector scores every token position, softmax over the
    non-padded positions, and the pooled vector is the resulting
    weighted sum. This lets the model learn to weight informative
    tokens more than padding-adjacent or low-information ones, instead
    of treating every real token equally.

Comparison is against the existing baseline checkpoint, freshly
re-evaluated live in this run (not hardcoded), on the same held-out test
data used everywhere else in this repo.
"""
import math
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import PositionalEncoding
from train_multitask import TaskDataset, data, vocab_size, max_len, num_choice_classes, device

torch.manual_seed(42)


class AttnPoolEncoder(nn.Module):
    """Identical to JevCloneEncoder except encode() uses learned attention
    pooling instead of mean pooling. Duplicated (not subclassed) so the
    baseline model.py stays untouched -- this is a one-off experiment,
    not a replacement architecture."""

    def __init__(self, vocab_size, d_model=96, nhead=6, num_layers=3, max_len=48,
                 dropout=0.15, num_choice_classes=None, enable_score=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_norm = nn.LayerNorm(d_model)

        # Learned attention pooling: a single query vector scores each
        # token position; softmax over non-padded positions gives
        # per-token weights, pooled = weighted sum of hidden states.
        self.attn_query = nn.Linear(d_model, 1)

        self.noul_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.choice_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                          nn.Linear(d_model, num_choice_classes))
        self.score_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

        self.temperature = nn.Parameter(torch.ones(1) * 1.0, requires_grad=False)
        self.choice_temperature = nn.Parameter(torch.ones(1) * 1.0, requires_grad=False)

    def encode(self, ids, mask):
        x = self.embed(ids)
        x = self.pos(x)
        key_padding_mask = mask == 0
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)

        scores = self.attn_query(h).squeeze(-1)  # [batch, seq]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [batch, seq, 1]
        pooled = (h * weights).sum(1)
        return self.pool_norm(pooled)


def eval_noul(model, loader):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask, label = ids.to(device), mask.to(device), label.to(device)
            pooled = model.encode(ids, mask)
            logit = model.noul_head(pooled).squeeze(-1)
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
            pred = model.choice_head(pooled).argmax(dim=-1)
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


def train_attn_model():
    noul_train = TaskDataset(data["noul"]["train"], torch.float)
    choice_train = TaskDataset(data["choice"]["train"], torch.long)
    score_train = TaskDataset(data["score"]["train"], torch.float)

    BATCH = 32
    noul_loader = DataLoader(noul_train, batch_size=BATCH, shuffle=True, drop_last=True)
    choice_loader = DataLoader(choice_train, batch_size=BATCH, shuffle=True, drop_last=True)
    score_loader = DataLoader(score_train, batch_size=BATCH, shuffle=True, drop_last=True)

    def infinite(loader):
        while True:
            for batch in loader:
                yield batch

    noul_iter, choice_iter, score_iter = infinite(noul_loader), infinite(choice_loader), infinite(score_loader)
    steps_per_epoch = math.ceil(len(choice_train) / BATCH)
    EPOCHS = 12

    model = AttnPoolEncoder(
        vocab_size=vocab_size, max_len=max_len, num_choice_classes=num_choice_classes,
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

    best_score_sum, best_state = -1, None
    print("Training attention-pooling variant (same recipe as train_multitask.py)...")
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
    torch.save(model.state_dict(), "jev_clone_attnpool.pt")
    print(f"Saved jev_clone_attnpool.pt (best combined val score={best_score_sum:.4f})")
    return model


if __name__ == "__main__":
    attn_model = train_attn_model()

    noul_test_loader = DataLoader(TaskDataset(data["noul"]["test"], torch.float), batch_size=64)
    choice_test_loader = DataLoader(TaskDataset(data["choice"]["test"], torch.long), batch_size=64)
    score_test_loader = DataLoader(TaskDataset(data["score"]["test"], torch.float), batch_size=64)

    attn_noul_acc = eval_noul(attn_model, noul_test_loader)
    attn_choice_acc = eval_choice(attn_model, choice_test_loader)
    attn_score_mae = eval_score_mae(attn_model, score_test_loader)

    print("\nLoading baseline (mean-pool) model for a live, non-hardcoded comparison...")
    from model import JevCloneEncoder
    baseline = JevCloneEncoder(
        vocab_size=vocab_size, max_len=max_len, num_choice_classes=num_choice_classes, enable_score=True,
    ).to(device)
    baseline.load_state_dict(torch.load("jev_clone_multitask_calibrated.pt", map_location=device))

    def eval_noul_base(model, loader):
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

    def eval_choice_base(model, loader):
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

    def eval_score_base(model, loader):
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

    base_noul_acc = eval_noul_base(baseline, noul_test_loader)
    base_choice_acc = eval_choice_base(baseline, choice_test_loader)
    base_score_mae = eval_score_base(baseline, score_test_loader)

    print("\n" + "=" * 70)
    print("COMPARISON: mean pooling (baseline) vs learned attention pooling")
    print("=" * 70)
    print(f"{'Metric':<28}{'Mean-pool':>12}{'Attn-pool':>12}{'Delta':>12}")
    print(f"{'Noul test accuracy':<28}{base_noul_acc:>12.4f}{attn_noul_acc:>12.4f}{attn_noul_acc-base_noul_acc:>+12.4f}")
    print(f"{'Choice test accuracy':<28}{base_choice_acc:>12.4f}{attn_choice_acc:>12.4f}{attn_choice_acc-base_choice_acc:>+12.4f}")
    print(f"{'Score test MAE (lower=better)':<28}{base_score_mae:>12.4f}{attn_score_mae:>12.4f}{attn_score_mae-base_score_mae:>+12.4f}")

    results = {
        "base_noul_acc": base_noul_acc, "attn_noul_acc": attn_noul_acc,
        "base_choice_acc": base_choice_acc, "attn_choice_acc": attn_choice_acc,
        "base_score_mae": base_score_mae, "attn_score_mae": attn_score_mae,
    }
    with open("attnpool_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nSaved attnpool_results.pkl")
