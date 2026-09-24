"""
Workstream A: split conformal prediction for the Noul, Choice, and Score
heads. Unlike temperature/Platt/isotonic scaling (already in this repo),
which calibrate confidence *on average* across a test set, split conformal
prediction produces prediction SETS/intervals with a distribution-free,
finite-sample coverage GUARANTEE: at target coverage 1-alpha, the true
label is provably in the returned set with probability >= 1-alpha, under
only an exchangeability assumption (no distributional assumptions on the
model or the data). This script verifies that guarantee empirically on
held-out test data, the same discipline already applied to ECE elsewhere
in this repo.

STATISTICAL HYGIENE: split conformal requires a calibration set that (a)
was not used to fit the point predictor's weights, and (b) was not used
to fit anything else the nonconformity score depends on (here: the
learned temperature scalars for Noul/Choice). The existing val split was
already used by calibrate_multitask.py to fit temperature/choice_temperature
on its FULL contents. Reusing that same full val split for conformal
calibration would leak: the temperature parameter would have "seen" the
exact data used to calibrate the conformal quantile. To avoid this, this
script splits val 50/50 into val_a/val_b (fixed seed) and REFITS
temperature/choice_temperature from scratch on val_a only, then computes
conformal nonconformity scores and the calibration quantile on val_b only
-- fully disjoint from both the temperature fit and the final test set.
The Score head has no such calibration-layer parameter (calibrate_multitask.py
never fits anything for Score beyond the trained weights), so no refit is
needed there; any half of val is a clean conformal-calibration split.

Coverage target: 90% (alpha = 0.10) for all three heads.
"""
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import JevCloneEncoder
from calib_utils import aps_prediction_set_sizes_and_coverage, aps_score, conformal_quantile, split_val
from train_multitask import TaskDataset, data, vocab_size, max_len, num_choice_classes, device

ALPHA = 0.10  # target: 90% coverage
SEED = 123  # distinct from the seeds used elsewhere, to keep this split independent in spirit

random.seed(SEED)
torch.manual_seed(SEED)

model = JevCloneEncoder(
    vocab_size=vocab_size, max_len=max_len,
    num_choice_classes=num_choice_classes, enable_score=True,
).to(device)
model.load_state_dict(torch.load("jev_clone_multitask_calibrated.pt", map_location=device))
model.eval()



# =============================================================================
# NOUL (binary): split conformal classification
# =============================================================================
print("=" * 70)
print("NOUL (binary spam/ham) -- split conformal prediction sets")
print("=" * 70)

noul_val_a, noul_val_b = split_val(data["noul"]["val"])
noul_test = data["noul"]["test"]
print(f"val_a (temperature refit) n={len(noul_val_a)}   "
      f"val_b (conformal calibration) n={len(noul_val_b)}   test n={len(noul_test)}")


def get_noul_logits(rows):
    loader = DataLoader(TaskDataset(rows, torch.float), batch_size=64)
    logits, labels = [], []
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask = ids.to(device), mask.to(device)
            logits.append(model.forward(ids, mask).cpu())
            labels.append(label)
    return torch.cat(logits), torch.cat(labels)


val_a_logits, val_a_labels = get_noul_logits(noul_val_a)

# Refit temperature on val_a ONLY (fresh fit, independent of val_b/test)
T_noul = nn.Parameter(torch.ones(1) * 1.5)
opt = torch.optim.LBFGS([T_noul], lr=0.01, max_iter=200)
bce = nn.BCEWithLogitsLoss()


def closure():
    opt.zero_grad()
    loss = bce(val_a_logits / T_noul, val_a_labels)
    loss.backward()
    return loss


opt.step(closure)
T_noul_val = T_noul.item()
print(f"Temperature refit on val_a only: T={T_noul_val:.4f} "
      f"(NOT the same fit as calibrate_multitask.py, which used the full val set)")


def noul_probs(rows, T):
    ids_l, mask_l, labels = [], [], []
    for r in rows:
        ids_l.append(r["ids"])
        mask_l.append(r["mask"])
        labels.append(r["label"])
    ids_t = torch.tensor(ids_l, dtype=torch.long).to(device)
    mask_t = torch.tensor(mask_l, dtype=torch.long).to(device)
    with torch.no_grad():
        logit = model.forward(ids_t, mask_t) / T
        p1 = torch.sigmoid(logit).cpu().numpy()
    return p1, np.array(labels)


# Nonconformity score for the TRUE label: s(x, y) = 1 - p_hat(y | x)
p1_b, y_b = noul_probs(noul_val_b, T_noul_val)
p_true_b = np.where(y_b == 1, p1_b, 1 - p1_b)
noul_scores_b = 1 - p_true_b
noul_qhat = conformal_quantile(noul_scores_b, ALPHA)
print(f"Conformal quantile (qhat) at alpha={ALPHA}: {noul_qhat:.4f}")

p1_test, y_test = noul_probs(noul_test, T_noul_val)
p0_test = 1 - p1_test
# Prediction set = {y : 1 - p_hat(y|x) <= qhat} = {y : p_hat(y|x) >= 1 - qhat}
threshold = 1 - noul_qhat
in_set_0 = p0_test >= threshold
in_set_1 = p1_test >= threshold
set_sizes = in_set_0.astype(int) + in_set_1.astype(int)
covered = np.where(y_test == 1, in_set_1, in_set_0)
empty_sets = (set_sizes == 0).sum()

noul_coverage = covered.mean()
noul_avg_set_size = set_sizes.mean()
print(f"\nTEST SET RESULTS (n={len(noul_test)}):")
print(f"  Target coverage: {1-ALPHA:.0%}   Measured coverage: {noul_coverage:.4f}")
print(f"  Average prediction set size: {noul_avg_set_size:.3f} (out of 2 possible classes)")
print(f"  Empty prediction sets: {empty_sets} ({empty_sets/len(noul_test):.4%})")
print(f"  Set size distribution: size=1 (confident, single-class): "
      f"{(set_sizes==1).sum()} ({(set_sizes==1).mean():.4f}), "
      f"size=2 (uncertain, both classes): {(set_sizes==2).sum()} ({(set_sizes==2).mean():.4f})")

# =============================================================================
# CHOICE (77-class): split conformal via Adaptive Prediction Sets (APS)
# =============================================================================
print("\n" + "=" * 70)
print("CHOICE (banking77) -- split conformal prediction sets (APS, non-randomized)")
print("=" * 70)

choice_val_a, choice_val_b = split_val(data["choice"]["val"])
choice_test = data["choice"]["test"]
print(f"val_a (temperature refit) n={len(choice_val_a)}   "
      f"val_b (conformal calibration) n={len(choice_val_b)}   test n={len(choice_test)}")


def get_choice_logits(rows):
    loader = DataLoader(TaskDataset(rows, torch.long), batch_size=64)
    logits, labels = [], []
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask = ids.to(device), mask.to(device)
            pooled = model.encode(ids, mask)
            logits.append(model.choice_head(pooled).cpu())
            labels.append(label)
    return torch.cat(logits), torch.cat(labels)


choice_val_a_logits, choice_val_a_labels = get_choice_logits(choice_val_a)

T_choice = nn.Parameter(torch.ones(1) * 1.5)
opt_c = torch.optim.LBFGS([T_choice], lr=0.01, max_iter=200)
ce = nn.CrossEntropyLoss()


def closure_c():
    opt_c.zero_grad()
    loss = ce(choice_val_a_logits / T_choice, choice_val_a_labels)
    loss.backward()
    return loss


opt_c.step(closure_c)
T_choice_val = T_choice.item()
print(f"choice_temperature refit on val_a only: T={T_choice_val:.4f}")


def choice_probs(rows, T):
    loader = DataLoader(TaskDataset(rows, torch.long), batch_size=64)
    probs, labels = [], []
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask = ids.to(device), mask.to(device)
            pooled = model.encode(ids, mask)
            logits = model.choice_head(pooled) / T
            probs.append(torch.softmax(logits, dim=-1).cpu())
            labels.append(label)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()



probs_b, y_b_choice = choice_probs(choice_val_b, T_choice_val)
choice_scores_b = aps_score(probs_b, y_b_choice)
choice_qhat = conformal_quantile(choice_scores_b, ALPHA)
print(f"Conformal quantile (qhat) at alpha={ALPHA}: {choice_qhat:.4f}")

probs_test, y_test_choice = choice_probs(choice_test, T_choice_val)



choice_set_sizes, choice_covered = aps_prediction_set_sizes_and_coverage(probs_test, y_test_choice, choice_qhat)
choice_coverage = choice_covered.mean()
choice_avg_set_size = choice_set_sizes.mean()
print(f"\nTEST SET RESULTS (n={len(choice_test)}):")
print(f"  Target coverage: {1-ALPHA:.0%}   Measured coverage: {choice_coverage:.4f}")
print(f"  Average prediction set size: {choice_avg_set_size:.2f} (out of 77 possible classes)")
print(f"  Median set size: {np.median(choice_set_sizes):.0f}   "
      f"Max set size: {choice_set_sizes.max()}   Min set size: {choice_set_sizes.min()}")
print(f"  Set size <= 5: {(choice_set_sizes <= 5).mean():.4f} of test examples")

# =============================================================================
# SCORE (continuous): split conformal regression (absolute-residual method)
# =============================================================================
print("\n" + "=" * 70)
print("SCORE (Amazon Fine Food Reviews rating) -- split conformal prediction intervals")
print("=" * 70)
print("Method: absolute-residual split conformal (Vovk et al./Lei et al. 2018) --")
print("NOT conformalized quantile regression (CQR). This gives constant-width")
print("intervals (same half-width for every test point), a real, documented")
print("limitation vs CQR's input-adaptive interval width -- noted honestly in")
print("CONFORMAL.md rather than presented as adaptive when it is not.")

score_val_a, score_val_b = split_val(data["score"]["val"])
score_test = data["score"]["test"]
print(f"val_a (unused here -- Score has no calibration-layer parameter) n={len(score_val_a)}   "
      f"val_b (conformal calibration) n={len(score_val_b)}   test n={len(score_test)}")


def score_preds(rows):
    loader = DataLoader(TaskDataset(rows, torch.float), batch_size=64)
    preds, labels = [], []
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask = ids.to(device), mask.to(device)
            pooled = model.encode(ids, mask)
            pred = torch.sigmoid(model.score_head(pooled).squeeze(-1)).cpu()
            preds.append(pred)
            labels.append(label)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


preds_b, y_b_score = score_preds(score_val_b)
score_residuals_b = np.abs(preds_b - y_b_score)
score_qhat = conformal_quantile(score_residuals_b, ALPHA)
print(f"Conformal quantile (qhat, half-width) at alpha={ALPHA}: {score_qhat:.4f} (on a [0,1] scale)")

preds_test, y_test_score = score_preds(score_test)
lower = preds_test - score_qhat
upper = preds_test + score_qhat
score_covered = (y_test_score >= lower) & (y_test_score <= upper)
score_coverage = score_covered.mean()
score_avg_width = 2 * score_qhat  # constant width by construction

print(f"\nTEST SET RESULTS (n={len(score_test)}):")
print(f"  Target coverage: {1-ALPHA:.0%}   Measured coverage: {score_coverage:.4f}")
print(f"  Interval width: {score_avg_width:.4f} (constant, out of a [0,1]-scale target range)")
print(f"  For context, this repo's Score head has a point-prediction MAE of ~0.19 on this")
print(f"  test set (see FINDINGS.md) -- a conformal half-width of {score_qhat:.4f} is "
      f"{'wider' if score_qhat > 0.19 else 'narrower'} than the typical point-prediction error.")

# =============================================================================
# Save results for CONFORMAL.md
# =============================================================================
import pickle

results = {
    "alpha": ALPHA,
    "noul": {
        "n_val_a": len(noul_val_a), "n_val_b": len(noul_val_b), "n_test": len(noul_test),
        "T_refit": T_noul_val, "qhat": noul_qhat,
        "coverage": noul_coverage, "avg_set_size": noul_avg_set_size,
        "empty_sets": int(empty_sets),
        "frac_size1": float((set_sizes == 1).mean()), "frac_size2": float((set_sizes == 2).mean()),
    },
    "choice": {
        "n_val_a": len(choice_val_a), "n_val_b": len(choice_val_b), "n_test": len(choice_test),
        "T_refit": T_choice_val, "qhat": choice_qhat,
        "coverage": choice_coverage, "avg_set_size": choice_avg_set_size,
        "median_set_size": float(np.median(choice_set_sizes)),
        "max_set_size": int(choice_set_sizes.max()), "min_set_size": int(choice_set_sizes.min()),
        "frac_set_size_le5": float((choice_set_sizes <= 5).mean()),
    },
    "score": {
        "n_val_b": len(score_val_b), "n_test": len(score_test),
        "qhat": score_qhat, "coverage": score_coverage, "interval_width": score_avg_width,
    },
}
with open("conformal_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved conformal_results.pkl")
