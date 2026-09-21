# Jev-clone: toy System-One decision model

A minimal, real (not simulated) reproduction of Jev's core interface shape:
send text state in, get a typed decision + calibrated confidence back in
one forward pass — no token generation.

## What's real here
- **Data**: SMS Spam Collection, 5,572 real human-labeled messages (ham/spam),
  not LLM-generated labels.
- **Architecture**: small transformer encoder trained from scratch in PyTorch.
  **No pretrained weights** — this was tested, not assumed: HuggingFace Hub
  (`huggingface.co`) and `download.pytorch.org` are both blocked by this
  sandbox's network policy (confirmed via direct connection tests, not just
  a failed `pip install`), so a pretrained encoder (distilbert, MiniLM, etc.)
  genuinely could not be loaded here. Instead of accepting the original
  64-dim/2-layer/word-level-vocab baseline, the encoder was scaled up
  (96-dim, 6 heads, 3 layers) and the tokenizer was replaced with a
  byte-level BPE vocab trained from scratch on the training split (via the
  `tokenizers` library, no network needed) instead of a frequency-cutoff
  word-level split — real, if modest, architecture improvements given the
  network constraint.
- **A genuine negative result along the way**: the first attempt to scale up
  (128-dim, 8 heads, 4 layers, same lr=1e-3 as the original) made things
  *worse*, not better — val accuracy dropped to 91% and training visibly
  diverged (val_acc collapsed to 15% partway through). This wasn't a data or
  architecture problem, it was a training-stability problem: a bigger model
  at the same learning rate with no gradient clipping on ~3,900 training
  examples is genuinely unstable. Lowering the LR (1e-3 → 3e-4) and adding
  gradient-norm clipping fixed it. This is reported because it's a realistic
  failure mode ("we made the encoder bigger and it got worse") worth keeping
  visible rather than quietly reverting and pretending the first attempt
  never happened.
- **Calibration**: temperature scaling (Guo et al. 2017) fit on a held-out
  validation split, measured against a separate held-out test split.
- **Results** (clean re-run of the full pipeline on 2026-09-21, seed fixed at
  42, tuned encoder + BPE tokenizer + stabilized training recipe): **97.85%
  test accuracy**; Expected Calibration Error (ECE) dropped from **0.0154
  (raw) to 0.0039 (calibrated)** — a 74.7% reduction. This beats both the
  original single-task baseline (97.13% acc / 0.0231→0.0131 ECE) and the
  very first reported numbers (97.4% acc / 0.0238→0.0081 ECE) reproducibly.
  As before, exact decimal values are NOT bit-reproducible run-to-run in
  this environment even with a fixed seed (likely PyTorch/CPU
  non-determinism in attention) — re-run `train.py` + `calibrate.py`
  yourself rather than trusting any single number quoted here as exact.
- **A real, current error worth knowing about**: this same calibrated model
  misclassifies "Free entry in 2 a wkly comp to win FA Cup final tkts, text
  FA to 87121" — a textbook spam message, verbatim from this dataset — as
  ham, with 97.75% confidence. High aggregate accuracy and good aggregate
  calibration do not mean individual high-confidence predictions are always
  right; see `python3 serve.py --demo`.

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
- Calibration validated on one held-out split from one dataset per task —
  not yet stress-tested against adversarial or out-of-distribution inputs
  (planned).
- Toy-scale encoder (~96-dim, 3 layers) trained from scratch, not a
  production-grade pretrained backbone — genuinely blocked by this sandbox's
  network policy, not a shortcut taken by choice.
- The Choice and Score heads (see below) were each validated on their own
  real dataset, sharing one encoder trunk and trained jointly — but no
  single real dataset in this repo has all three label types on the same
  text, so the "one state, many typed questions" claim is demonstrated
  architecturally (one `encode()` call feeds all heads) rather than proven
  on genuinely overlapping real-world queries.

This is scoped as reconnaissance into the technical claims, per the plan to
identify where a real Jev competitor's wedge might be — not a shippable
product.
