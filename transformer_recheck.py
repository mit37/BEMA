"""
The from-scratch transformer's test metrics (FINDINGS.md section 8's
"From scratch" column, which section 9 compares against), recomputed
with the fixed temperature fitter (section 10). Same models, seeds and
evaluation as self_pretrain.py: multiseed_sweep.get_model("mean", seed)
and self_pretrain.typo_metrics, for seeds 42, 123 and 2024.

Usage: python3 transformer_recheck.py   (trains any missing seed, ~20 min each on 4 CPUs)
"""
import pickle

import numpy as np

from multiseed_sweep import evaluate, get_model
from self_pretrain import SEEDS, typo_metrics

if __name__ == "__main__":
    results = {}
    for seed in SEEDS:
        model = get_model("mean", seed)
        results[seed] = {**evaluate(model), **typo_metrics(model)}
        print(f"[seed {seed}] " + " ".join(f"{k}={v:.4f}" for k, v in results[seed].items()), flush=True)
    print(f"\nFrom-scratch transformer, test metrics, mean +/- sample std over seeds {SEEDS}:")
    for k in results[SEEDS[0]]:
        v = np.array([results[s][k] for s in SEEDS])
        print(f"  {k:<20}{v.mean():.4f} +/- {v.std(ddof=1):.4f}")
    with open("transformer_recheck_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved transformer_recheck_results.pkl")
