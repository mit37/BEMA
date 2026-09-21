# Jev-clone: toy System-One decision model

A minimal, real (not simulated) reproduction of Jev's core interface shape:
send text state in, get a typed decision + calibrated confidence back in
one forward pass — no token generation.

## What's real here
- **Data**: SMS Spam Collection, 5,572 real human-labeled messages (ham/spam),
  not LLM-generated labels.
- **Architecture**: small transformer encoder trained from scratch in PyTorch
  (no pretrained weights — HuggingFace hub wasn't reachable from this sandbox),
  with a typed "Noul" head (binary decision) and Score output.
- **Calibration**: temperature scaling (Guo et al. 2017) fit on a held-out
  validation split, measured against a separate held-out test split.
- **Results** (verified by a clean re-run of the pipeline on 2026-09-21, seed
  fixed at 42): 97.13% test accuracy; Expected Calibration Error (ECE) dropped
  from 0.0231 (raw) to 0.0131 (calibrated) — a 43.4% reduction, meaning the
  confidence scores now more closely track empirical accuracy, though not as
  cleanly as an earlier run of this code reported (97.4% acc, ECE
  0.0238→0.0081, 66% reduction). Despite the fixed seed, exact numbers are
  NOT bit-reproducible run to run in this environment (likely PyTorch/CPU
  non-determinism in the transformer encoder's attention op) — treat the
  qualitative result (calibration meaningfully improves ECE without hurting
  accuracy) as the reliable claim, not the specific decimal values. Re-run
  `train.py` + `calibrate.py` yourself for current numbers rather than
  trusting either figure quoted here.

## Files
- `prepare_data.py` — loads spam.csv, builds vocab, train/val/test splits
- `model.py` — the encoder + typed decision head
- `train.py` — trains the model, reports uncalibrated ECE
- `calibrate.py` — fits temperature scaling, reports before/after ECE
- `serve.py` — the typed API: `python3 serve.py "some text"` -> `{is_spam, confidence}`

## Run it
```
pip install -r requirements.txt
python3 prepare_data.py
python3 train.py
python3 calibrate.py
python3 serve.py --demo
```

## Where this stops being "a Jev"
- One domain (binary spam classification), not general-purpose typed
  questions over arbitrary schemas.
- Calibration validated on one held-out split from one dataset — not
  stress-tested against adversarial or out-of-distribution inputs.
- Toy-scale encoder (~64-dim, 2 layers) trained from scratch, not a
  production-grade pretrained backbone.
- No Choice-type (multi-class) or continuous Score-type decisions yet,
  only Noul (binary).

This is scoped as reconnaissance into the technical claims, per the plan to
identify where a real Jev competitor's wedge might be — not a shippable
product.
