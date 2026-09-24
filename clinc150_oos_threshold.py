"""
Can the CLINC150 model's own confidence flag out-of-scope queries?

clinc150_pipeline.py's model treats "oos" as just one more class and
recognizes only 16.1% of out-of-scope test queries. CLINC150's training
set is balanced (100 examples for every class, oos included), so this is
not a class-frequency problem: oos gets the same 100-example budget as a
single narrow intent while having to cover everything else.

The standard remedy from the CLINC150 paper (Larson et al. 2019) is
confidence thresholding: answer "oos" whenever the top-class probability
is below a threshold tau. This script:
  1. measures threshold-free separability: AUROC of (1 - max prob) for
     telling oos test queries apart from in-scope ones;
  2. picks tau on the VALIDATION split only (maximizing the mean of
     in-scope accuracy and oos recall) and reports held-out TEST results.

Usage: python3 clinc150_oos_threshold.py  (after clinc150_pipeline.py)
"""
import pickle

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from model import JevCloneEncoder

with open("data/prepared_clinc150.pkl", "rb") as f:
    data = pickle.load(f)
OOS = data["categories"].index("oos")

model = JevCloneEncoder(vocab_size=data["vocab_size"], max_len=data["max_len"],
                        num_choice_classes=data["num_classes"])
model.load_state_dict(torch.load("jev_clone_clinc150_calibrated.pt", map_location="cpu"))
model.eval()


def probs_and_labels(rows):
    ids = torch.tensor([r["ids"] for r in rows])
    mask = torch.tensor([r["mask"] for r in rows])
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            out.append(model.predict_choice(ids[i:i + 256], mask[i:i + 256])[2])
    return torch.cat(out).numpy(), np.array([r["label"] for r in rows])


def predict(probs, tau):
    pred = probs.argmax(1)
    return np.where(probs.max(1) < tau, OOS, pred)


def rates(pred, y):
    ins, oos = y != OOS, y == OOS
    return {"inscope_acc": float((pred[ins] == y[ins]).mean()),
            "oos_recall": float((pred[oos] == OOS).mean()),
            "oos_precision": float((y[pred == OOS] == OOS).mean()) if (pred == OOS).any() else float("nan"),
            "overall_acc": float((pred == y).mean())}


val_p, val_y = probs_and_labels(data["val"])
test_p, test_y = probs_and_labels(data["test"])

is_oos = test_y == OOS
auroc = roc_auc_score(is_oos, 1 - test_p.max(1))
print(f"Test n={len(test_y)} ({is_oos.sum()} oos, {(~is_oos).sum()} in-scope); "
      f"val n={len(val_y)} ({(val_y == OOS).sum()} oos)")
print(f"AUROC of (1 - max prob) for detecting oos on test: {auroc:.4f}  (0.5 = chance, 1.0 = perfect)")
print(f"Mean max prob on test: in-scope {test_p[~is_oos].max(1).mean():.4f}, "
      f"oos {test_p[is_oos].max(1).mean():.4f}")

taus = np.round(np.arange(0.0, 1.0, 0.01), 2)
val_scores = [np.mean([rates(predict(val_p, t), val_y)[k] for k in ("inscope_acc", "oos_recall")])
              for t in taus]
tau = float(taus[int(np.argmax(val_scores))])

base, thr = rates(predict(test_p, 0.0), test_y), rates(predict(test_p, tau), test_y)
print(f"\nThreshold chosen on validation: tau = {tau:.2f}")
print(f"{'Test metric':<16}{'argmax only':>14}{'tau rejection':>16}")
for k in ("inscope_acc", "oos_recall", "oos_precision", "overall_acc"):
    print(f"{k:<16}{base[k]:>14.4f}{thr[k]:>16.4f}")

print("\nTradeoff curve on test (for reference only; tau above was picked on val):")
for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    r = rates(predict(test_p, t), test_y)
    print(f"  tau={t:.1f}  in-scope acc {r['inscope_acc']:.4f}  oos recall {r['oos_recall']:.4f}")

with open("clinc150_oos_results.pkl", "wb") as f:
    pickle.dump({"auroc": auroc, "tau": tau, "argmax": base, "threshold": thr}, f)
print("\nSaved clinc150_oos_results.pkl")
