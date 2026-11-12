"""
How much did the old temperature fitter matter? FINDINGS.md section 10.

Until 2026-11-12, calib_utils.fit_temperature (and copies of the same code
in other scripts) ran LBFGS on T with a fixed step of 0.01 from T=1.5,
which stops short of the cross-entropy minimum. This script fits both the
old and the fixed version on the same validation logits and reports the
temperatures and the test ECE / overconfidence each gives, clean and
under the held-out typo sets, for the models that can be rebuilt here:
the TF-IDF linear baseline and the tuned n-gram BEMA seeds, plus the
multitask transformer (Choice and Noul heads) if its checkpoint exists.

Usage: python3 temperature_audit.py   (after train_ngram.py --tag _tuned)
"""
import os
import pickle

import numpy as np
import torch

from calib_utils import ece, fit_temperature, fit_temperature_binary
from heldout_noise_test import perturbed
from linear_baseline import acc, fit_best, split
from ngram_model import NgramEncoder, ngram_features
from sklearn.linear_model import LogisticRegression
from train_multitask import num_choice_classes
from train_ngram import NOISES, SEEDS, predict


def old_fit_temperature(logits, labels):
    """The pre-fix fitter, verbatim."""
    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=200)
    ce = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = ce(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return T.item()


def old_fit_temperature_binary(logits, labels):
    """The pre-fix binary fitter (calibrate_multitask.py's Noul fit), verbatim."""
    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=200)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return T.item()


def audit_binary(name, val_logits, yv, test_logits, yt):
    """Single-logit head: ECE of the predicted class's probability."""
    yv, yt = torch.tensor(yv).float(), np.asarray(yt)
    out = {}
    for fitter, fit in (("old", old_fit_temperature_binary), ("fixed", fit_temperature_binary)):
        T = fit(val_logits, yv)
        p = torch.sigmoid(test_logits / T).numpy()
        conf, pred = np.maximum(p, 1 - p), (p > 0.5).astype(int)
        correct = (pred == yt).astype(float)
        out[fitter] = {"T": T, "clean_ece": ece(conf.tolist(), correct.tolist()),
                       "clean_overconf": conf.mean() - correct.mean(),
                       "val_nll": torch.nn.functional.binary_cross_entropy_with_logits(val_logits / T, yv).item()}
    o, f = out["old"], out["fixed"]
    print(f"\n{name}: T old {o['T']:.3f} -> fixed {f['T']:.3f}   val NLL {o['val_nll']:.4f} -> {f['val_nll']:.4f}")
    print(f"  clean      ECE {o['clean_ece']:.4f} -> {f['clean_ece']:.4f}   "
          f"overconf {o['clean_overconf']:+.4f} -> {f['clean_overconf']:+.4f}")
    return out


def nll(logits, labels, T):
    return torch.nn.functional.cross_entropy(logits / T, labels).item()


def audit(name, val_logits, yv, test_logits, yt):
    """test_logits: dict noise -> logits on the (perturbed) test set."""
    yv, yt = torch.tensor(yv), np.asarray(yt)
    out = {}
    for fitter, fit in (("old", old_fit_temperature), ("fixed", fit_temperature)):
        T = fit(val_logits, yv)
        r = {"T": T, "val_nll": nll(val_logits, yv, T)}
        for noise, lg in test_logits.items():
            conf, pred = torch.softmax(lg / T, -1).max(-1)
            correct = (pred.numpy() == yt).astype(float)
            r[f"{noise}_ece"] = ece(conf.tolist(), correct.tolist())
            r[f"{noise}_overconf"] = conf.mean().item() - correct.mean()
        out[fitter] = r
    o, f = out["old"], out["fixed"]
    print(f"\n{name}: T old {o['T']:.3f} -> fixed {f['T']:.3f}   val NLL {o['val_nll']:.4f} -> {f['val_nll']:.4f}")
    for noise in test_logits:
        print(f"  {noise:<10} ECE {o[f'{noise}_ece']:.4f} -> {f[f'{noise}_ece']:.4f}   "
              f"overconf {o[f'{noise}_overconf']:+.4f} -> {f[f'{noise}_overconf']:+.4f}")
    return out


if __name__ == "__main__":
    xv, yv = split("choice", "val")
    xt, yt = split("choice", "test")
    texts = {"clean": xt, **{n: perturbed(xt, n) for n in NOISES}}
    results = {}

    vec, lin, c = fit_best("choice", lambda c: LogisticRegression(C=c, max_iter=3000), (5, 20, 50), acc)
    log_p = lambda t: torch.tensor(np.log(np.clip(lin.predict_proba(vec.transform(t)), 1e-12, 1))).float()
    results["linear"] = audit(f"Linear TF-IDF (C={c})", log_p(xv), yv, {n: log_p(t) for n, t in texts.items()}, yt)

    for seed in SEEDS:
        m = NgramEncoder(d_model=256, num_choice_classes=num_choice_classes, enable_score=True)
        m.load_state_dict(torch.load(f"bema_ngram_tuned_seed{seed}.pt"))
        lg = lambda t: predict(m, [ngram_features(s, m.buckets) for s in t])["choice"]
        results[f"ngram_seed{seed}"] = audit(f"n-gram BEMA tuned, seed {seed}", lg(xv), yv,
                                             {n: lg(t) for n, t in texts.items()}, yt)

    if os.path.exists("jev_clone_multitask.pt"):
        from heldout_noise_test import encode_batch, load
        model = load("jev_clone_multitask.pt")

        def tlg(t):
            ids, mask = encode_batch(t)
            with torch.no_grad():
                return torch.cat([model.choice_head(model.encode(ids[i:i + 256], mask[i:i + 256]))
                                  for i in range(0, len(t), 256)])
        results["transformer"] = audit("Multitask transformer (train_multitask.py, seed 42)", tlg(xv), yv,
                                       {n: tlg(t) for n, t in texts.items()}, yt)

        def nlg(t):
            ids, mask = encode_batch(t)
            with torch.no_grad():
                return torch.cat([model.forward(ids[i:i + 256], mask[i:i + 256]) for i in range(0, len(t), 256)])
        (nxv, nyv), (nxt, nyt) = split("noul", "val"), split("noul", "test")
        results["transformer_noul"] = audit_binary(f"Multitask transformer, Noul head (val n={len(nyv)}, test n={len(nyt)})",
                                                   nlg(nxv), nyv, nlg(nxt), nyt)

    with open("temperature_audit_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nSaved temperature_audit_results.pkl")
