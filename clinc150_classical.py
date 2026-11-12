"""
Does FINDINGS.md §9 replicate on CLINC150? The TF-IDF linear baseline and
the n-gram-trunk BEMA (ngram_model.py), on the same 151-class task
(150 intents + oos), splits and typo perturbation as clinc150_pipeline.py,
compared with that script's transformer (numbers in CLINC150.md).

  - Linear: TF-IDF word 1-2 + char 2-5 grams (linear_baseline.features),
    logistic regression, C chosen on validation accuracy.
  - n-gram BEMA: Choice head only, seeds 42/123/2024, the configuration
    tuned on BANKING77 (width 256, dropout 0.4, embedding lr 3e-3,
    8 epochs) reused without re-tuning; epoch chosen on validation.
  - Ensembles: the 3 seeds averaged, and that average mixed with the
    linear model, mixing weight chosen on validation accuracy.

Every model's confidence is temperature-calibrated on validation. Out-of-
scope handling is reported both as plain argmax and with the confidence
threshold rule of clinc150_oos_threshold.py (tau chosen on validation).
Test is scored once.

Usage: python3 clinc150_classical.py
"""
import json
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from calib_utils import ece, fit_temperature, paired_bootstrap_diff
from linear_baseline import features
from ngram_model import NgramEncoder, collate, ngram_features

SEEDS = [42, 123, 2024]
BATCH = 32
EPOCHS = 8
D_MODEL, DROPOUT, SPARSE_LR = 256, 0.4, 3e-3
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)

with open("data/raw/clinc150_data_full.json") as f:
    raw = json.load(f)
intents = sorted({label for _, label in raw["train"]})
categories = intents + ["oos"]
OOS = categories.index("oos")
label_to_idx = {name: i for i, name in enumerate(categories)}


def split(name):
    """Texts and labels, in-scope rows then oos rows, as clinc150_pipeline.to_rows."""
    pairs = raw[name] + raw[f"oos_{name}"]
    return [t for t, _ in pairs], np.array([label_to_idx[lab] for _, lab in pairs])


def perturb_typo(text, n_typos=2, seed=None):
    """clinc150_pipeline.py's perturbation (two adjacent-character swaps), copied so results pair."""
    rnd = random.Random(seed)
    chars = list(text)
    for _ in range(n_typos):
        if len(chars) < 3:
            break
        i = rnd.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def typo_texts(texts):
    rnd = random.Random(7)
    return [perturb_typo(t, seed=rnd.randint(0, 10 ** 9)) for t in texts]


def thresholded(probs, tau):
    return np.where(probs.max(1) < tau, OOS, probs.argmax(1))


def rates(pred, y):
    ins, oos = y != OOS, y == OOS
    return {"inscope_acc": float((pred[ins] == y[ins]).mean()),
            "oos_recall": float((pred[oos] == OOS).mean()),
            "oos_precision": float((y[pred == OOS] == OOS).mean()) if (pred == OOS).any() else float("nan"),
            "overall_acc": float((pred == y).mean())}


def choose_tau(val_p, val_y):
    """As clinc150_oos_threshold.choose_tau: maximize mean(in-scope accuracy, oos recall) on validation."""
    taus = np.round(np.arange(0.0, 1.0, 0.01), 2)
    scores = [np.mean([rates(thresholded(val_p, t), val_y)[k] for k in ("inscope_acc", "oos_recall")])
              for t in taus]
    return float(taus[int(np.argmax(scores))])


def calibrated(log_p_val, y_val):
    """Temperature fitted on validation; returns a function mapping log-probs to calibrated probs."""
    T = fit_temperature(log_p_val.float(), torch.tensor(y_val))
    return lambda log_p: torch.softmax(log_p.float() / T, -1).numpy(), T


def fit_linear(xtr, ytr, xva, yva):
    vec = features()
    Xtr, Xva = vec.fit_transform(xtr), vec.transform(xva)
    fitted = [(c, LogisticRegression(C=c, max_iter=3000).fit(Xtr, ytr)) for c in (5, 20, 50)]
    c, model = max(fitted, key=lambda cm: (cm[1].predict(Xva) == yva).mean())
    print(f"Linear: C={c} chosen on validation")
    return lambda texts: torch.tensor(np.log(np.clip(model.predict_proba(vec.transform(texts)), 1e-12, 1)))


def ngram_logits(model, feats, chunk=512):
    model.eval()
    with torch.no_grad():
        return torch.cat([model.choice_head(model.encode(*collate(feats[i:i + chunk])))
                          for i in range(0, len(feats), chunk)])


def train_ngram(seed, xtr, ytr, xva, yva):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = NgramEncoder(d_model=D_MODEL, dropout=DROPOUT, num_choice_classes=len(categories))
    ftr = [ngram_features(t, model.buckets) for t in xtr]
    fva = [ngram_features(t, model.buckets) for t in xva]
    sparse_opt = torch.optim.SparseAdam(list(model.bag.parameters()), lr=SPARSE_LR)
    dense = [p for n, p in model.named_parameters() if not n.startswith("bag.") and p.requires_grad]
    dense_opt = torch.optim.AdamW(dense, lr=1e-3, weight_decay=1e-4)
    best, best_state = -1.0, None
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(ftr), generator=gen).tolist()
        for i in range(0, len(order) - BATCH + 1, BATCH):
            idx = order[i:i + BATCH]
            logits = model.choice_head(model.encode(*collate([ftr[j] for j in idx])))
            loss = nn.functional.cross_entropy(logits, torch.tensor(ytr[idx]))
            sparse_opt.zero_grad()
            dense_opt.zero_grad()
            loss.backward()
            sparse_opt.step()
            dense_opt.step()
        acc = (ngram_logits(model, fva).argmax(-1).numpy() == yva).mean()
        print(f"[seed {seed}] epoch {epoch + 1}  val acc {acc:.4f}", flush=True)
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval()


def report(name, p_val, y_val, p_test, p_typo, y_test):
    """All test metrics for one model given calibrated probabilities."""
    pred, conf = p_test.argmax(1), p_test.max(1)
    correct = (pred == y_test).astype(float)
    typo_correct = (p_typo.argmax(1) == y_test).astype(float)
    tau = choose_tau(p_val, y_val)
    r = {"argmax": rates(pred, y_test), "tau": tau, "threshold": rates(thresholded(p_test, tau), y_test),
         "ece": ece(conf.tolist(), correct.tolist()),
         "auroc": roc_auc_score(y_test == OOS, 1 - conf),
         "typo_acc": typo_correct.mean(), "typo_overconf": p_typo.max(1).mean() - typo_correct.mean(),
         "clean_overconf": conf.mean() - correct.mean(),
         "correct": correct, "typo_correct": typo_correct}
    print(f"\n{name}: test n={len(y_test)} ({(y_test == OOS).sum()} oos)")
    print(f"  argmax   overall {r['argmax']['overall_acc']:.4f}  in-scope {r['argmax']['inscope_acc']:.4f}  "
          f"oos recall {r['argmax']['oos_recall']:.4f}  ECE {r['ece']:.4f}  overconf {r['clean_overconf']:+.4f}")
    print(f"  tau={tau:.2f} overall {r['threshold']['overall_acc']:.4f}  in-scope {r['threshold']['inscope_acc']:.4f}  "
          f"oos recall {r['threshold']['oos_recall']:.4f}  oos precision {r['threshold']['oos_precision']:.4f}")
    print(f"  oos AUROC {r['auroc']:.4f}   typos: acc {r['typo_acc']:.4f} (drop {correct.mean() - r['typo_acc']:.4f}), "
          f"overconf {r['typo_overconf']:+.4f}")
    return r


if __name__ == "__main__":
    (xtr, ytr), (xva, yva), (xte, yte) = split("train"), split("val"), split("test")
    xty = typo_texts(xte)
    print(f"CLINC150: {len(categories)} classes; train={len(ytr)} val={len(yva)} test={len(yte)}")

    lin = fit_linear(xtr, ytr, xva, yva)
    lin_cal, lin_T = calibrated(lin(xva), yva)
    probs = {"linear": {s: lin_cal(lin(x)) for s, x in (("val", xva), ("test", xte), ("typo", xty))}}

    seed_probs = []
    for seed in SEEDS:
        model = train_ngram(seed, xtr, ytr, xva, yva)
        logit = {s: ngram_logits(model, [ngram_features(t, model.buckets) for t in x])
                 for s, x in (("val", xva), ("test", xte), ("typo", xty))}
        cal, _ = calibrated(logit["val"], yva)
        seed_probs.append({s: cal(v) for s, v in logit.items()})
        probs[f"ngram_seed{seed}"] = seed_probs[-1]

    avg = {s: np.mean([p[s] for p in seed_probs], 0) for s in ("val", "test", "typo")}
    val_acc = {w: ((1 - w) * avg["val"] + w * probs["linear"]["val"]).argmax(1).__eq__(yva).mean() for w in WEIGHTS}
    for w in WEIGHTS:
        print(f"  w={w:.2f}  val accuracy {val_acc[w]:.4f}")
    w = max(WEIGHTS, key=lambda k: (val_acc[k], -abs(k - 0.5)))
    print(f"Mixing weight chosen on validation: w={w}")
    for name, ww in (("ngram3", 0.0), ("mix", w)):
        mixed = {s: (1 - ww) * avg[s] + ww * probs["linear"][s] for s in avg}
        cal, _ = calibrated(torch.log(torch.tensor(mixed["val"]).clamp_min(1e-12)), yva)
        probs[name] = {s: cal(torch.log(torch.tensor(v).clamp_min(1e-12))) for s, v in mixed.items()}

    results = {"w": w, "linear_T": lin_T}
    for name, p in probs.items():
        results[name] = report(name, p["val"], yva, p["test"], p["typo"], yte)

    print("\nPaired bootstrap (B=2000, 95% interval), overall argmax accuracy minus the linear model's:")
    for name in ("ngram3", "mix"):
        for key in ("correct", "typo_correct"):
            point, lo, hi = paired_bootstrap_diff(results[name][key], results["linear"][key])
            results[f"{name}_minus_linear_{key}"] = (point, lo, hi)
            print(f"  {name:<7}{'clean' if key == 'correct' else 'typos':<7}{point:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
                  f"n={len(yte)}")
    single = np.array([results[f"ngram_seed{s}"]["argmax"]["overall_acc"] for s in SEEDS])
    print(f"\nSingle n-gram models: overall accuracy {single.mean():.4f} +/- {single.std(ddof=1):.4f} over seeds {SEEDS}")
    with open("clinc150_classical_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved clinc150_classical_results.pkl")
