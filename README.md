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
- **Results**: 97.4% test accuracy; Expected Calibration Error (ECE) dropped
  from 0.0238 (raw) to 0.0081 (calibrated) — a 66% reduction, meaning the
  confidence scores now roughly mean what they claim to mean.

## Files
- `prepare_data.py` — loads spam.csv, builds vocab, train/val/test splits
- `model.py` — the encoder + typed decision head
- `train.py` — trains the model, reports uncalibrated ECE
- `calibrate.py` — fits temperature scaling, reports before/after ECE
- `serve.py` — the typed API: `python3 serve.py "some text"` -> `{is_spam, confidence}`

## Run it
```
pip install torch transformers scikit-learn pandas
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
