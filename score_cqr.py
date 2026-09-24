"""
Conformalized quantile regression (CQR, Romano, Patterson & Candes 2019)
for the Score head.

conformal.py's absolute-residual intervals have one width for every
input, and on this task that width (1.07 on the [0,1] rating scale) is
wider than the whole output range. CQR instead learns input-dependent
lower/upper quantiles, then conformalizes them so the 90% coverage
guarantee still holds.

Setup (no retraining of the shared encoder):
  - Freeze the calibrated multitask encoder; its pooled state is the input.
  - Fit a small head predicting the 5th and 95th percentile (pinball loss)
    on the Score TRAIN split; pick the epoch on val_a.
  - Conformalize on val_b (same val_a/val_b split as conformal.py).
  - Report coverage and interval width once, on the untouched TEST split.

Usage: python3 score_cqr.py  (needs jev_clone_multitask_calibrated.pt)
"""
import pickle

import numpy as np
import torch
import torch.nn as nn

from calib_utils import conformal_quantile, split_val
from model import JevCloneEncoder
from train_multitask import data, device, max_len, num_choice_classes, vocab_size

ALPHA = 0.10
QUANTILES = torch.tensor([ALPHA / 2, 1 - ALPHA / 2])
torch.manual_seed(0)

encoder = JevCloneEncoder(vocab_size=vocab_size, max_len=max_len,
                          num_choice_classes=num_choice_classes, enable_score=True).to(device)
encoder.load_state_dict(torch.load("jev_clone_multitask_calibrated.pt", map_location=device))
encoder.eval()


def features(rows):
    ids = torch.tensor([r["ids"] for r in rows])
    mask = torch.tensor([r["mask"] for r in rows])
    feats, points = [], []
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            pooled = encoder.encode(ids[i:i + 256].to(device), mask[i:i + 256].to(device))
            feats.append(pooled.cpu())
            points.append(torch.sigmoid(encoder.score_head(pooled).squeeze(-1)).cpu())
    y = torch.tensor([r["label"] for r in rows], dtype=torch.float)
    return torch.cat(feats), torch.cat(points), y


val_a, val_b = split_val(data["score"]["val"])
X_tr, _, y_tr = features(data["score"]["train"])
X_a, point_a, y_a = features(val_a)
X_b, point_b, y_b = features(val_b)
X_te, point_te, y_te = features(data["score"]["test"])


def pinball(pred, y):
    diff = y.unsqueeze(1) - pred
    return torch.maximum(QUANTILES * diff, (QUANTILES - 1) * diff).mean()


d = X_tr.shape[1]
head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2))
opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
best_loss, best_state, best_epoch = float("inf"), None, -1
for epoch in range(60):
    head.train()
    perm = torch.randperm(len(X_tr))
    for i in range(0, len(perm), 64):
        idx = perm[i:i + 64]
        opt.zero_grad()
        pinball(head(X_tr[idx]), y_tr[idx]).backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        loss = pinball(head(X_a), y_a).item()
    if loss < best_loss:
        best_loss, best_epoch = loss, epoch
        best_state = {k: v.clone() for k, v in head.state_dict().items()}
head.load_state_dict(best_state)
print(f"Quantile head: best val_a pinball loss {best_loss:.4f} at epoch {best_epoch + 1}")


def bounds(X):
    with torch.no_grad():
        q = head(X).numpy()
    return np.minimum(q[:, 0], q[:, 1]), np.maximum(q[:, 0], q[:, 1])


lo_b, hi_b = bounds(X_b)
qhat = conformal_quantile(np.maximum(lo_b - y_b.numpy(), y_b.numpy() - hi_b), ALPHA)
lo_te, hi_te = bounds(X_te)
y = y_te.numpy()
lo, hi = np.clip(lo_te - qhat, 0, 1), np.clip(hi_te + qhat, 0, 1)
width = hi - lo
cov = float(((y >= lo) & (y <= hi)).mean())

# The existing absolute-residual interval from conformal.py, recomputed on the
# same val_b/test so both methods are compared on identical data.
q_abs = conformal_quantile(np.abs(point_b.numpy() - y_b.numpy()), ALPHA)
p = point_te.numpy()
lo_abs, hi_abs = np.clip(p - q_abs, 0, 1), np.clip(p + q_abs, 0, 1)
cov_abs = float(((y >= lo_abs) & (y <= hi_abs)).mean())
width_abs = hi_abs - lo_abs

stars = np.rint(y * 4 + 1).astype(int)
print(f"\nTEST (n={len(y)}), target coverage {1 - ALPHA:.0%}, widths on the [0,1] scale "
      f"(1.0 = the whole 1-5 star range), clipped to [0,1]:")
print(f"{'Method':<24}{'coverage':>10}{'mean width':>12}{'median':>9}{'width<0.5':>11}")
print(f"{'absolute residual':<24}{cov_abs:>10.4f}{width_abs.mean():>12.4f}"
      f"{np.median(width_abs):>9.4f}{(width_abs < 0.5).mean():>11.4f}")
print(f"{'CQR':<24}{cov:>10.4f}{width.mean():>12.4f}{np.median(width):>9.4f}{(width < 0.5).mean():>11.4f}")
print(f"CQR conformal correction qhat = {qhat:+.4f}")

# Group-conditional ("Mondrian") CQR: conformalize separately within three
# groups of the model's PREDICTED rating (equal-count cut points set on
# val_a). This guarantees coverage within each predicted-rating group; it
# cannot guarantee coverage by TRUE rating, which is unknown at test time.
cuts = np.quantile(point_a.numpy(), [1 / 3, 2 / 3])
g_b, g_te = np.digitize(point_b.numpy(), cuts), np.digitize(p, cuts)
scores_b = np.maximum(lo_b - y_b.numpy(), y_b.numpy() - hi_b)
qhat_g = np.array([conformal_quantile(scores_b[g_b == g], ALPHA) for g in range(3)])
lo_m, hi_m = np.clip(lo_te - qhat_g[g_te], 0, 1), np.clip(hi_te + qhat_g[g_te], 0, 1)
width_m = hi_m - lo_m
cov_m = float(((y >= lo_m) & (y <= hi_m)).mean())
print(f"{'CQR, per predicted group':<24}{cov_m:>10.4f}{width_m.mean():>12.4f}"
      f"{np.median(width_m):>9.4f}{(width_m < 0.5).mean():>11.4f}")
print(f"Predicted-rating cut points (val_a tertiles, [0,1] scale): {cuts.round(3).tolist()}; "
      f"per-group qhat {qhat_g.round(4).tolist()} (val_b n per group {np.bincount(g_b).tolist()})")


def covered(lo_, hi_, m):
    return ((y[m] >= lo_[m]) & (y[m] <= hi_[m])).mean()


print("\nBy true star rating (does the interval adapt, and who is under-covered?):")
print(f"  {'':<8}{'n':>6}{'CQR cov':>10}{'width':>8}{'per-group cov':>15}{'width':>8}")
for s in range(1, 6):
    m = stars == s
    print(f"  {s} stars {m.sum():>6}{covered(lo, hi, m):>10.4f}{width[m].mean():>8.3f}"
          f"{covered(lo_m, hi_m, m):>15.4f}{width_m[m].mean():>8.3f}")
print("By predicted-rating group (what per-group calibration does guarantee):")
for g in range(3):
    m = g_te == g
    print(f"  group {g} n={m.sum():>5}  CQR {covered(lo, hi, m):.4f}  per-group {covered(lo_m, hi_m, m):.4f}")

with open("score_cqr_results.pkl", "wb") as f:
    pickle.dump({"alpha": ALPHA, "n_test": len(y), "cqr": {"coverage": cov, "mean_width": float(width.mean())},
                 "abs_residual": {"coverage": cov_abs, "mean_width": float(width_abs.mean())},
                 "cqr_per_group": {"coverage": cov_m, "mean_width": float(width_m.mean())}}, f)
print("\nSaved score_cqr_results.pkl")
