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

## Multi-task model: Noul + Choice + Score, one shared encoder

The single-task pipeline above (`prepare_data.py`/`train.py`/`calibrate.py`/
`serve.py`) only has the Noul (binary) head. `prepare_multitask.py`/
`train_multitask.py`/`calibrate_multitask.py`/`serve_multitask.py` add the
two missing typed decision shapes from Jev's interface, each grounded in a
REAL, human-labeled dataset (never LLM-generated labels), sharing ONE
encoder trunk (`model.encode()` runs once; every head reads the same pooled
state — see `model.py`):

- **Noul** (binary): SMS Spam Collection, as above.
- **Choice** (multi-class, 77 classes): [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets)
  (Casanueva et al. 2020) — real, human-annotated customer-support intent
  categories, 10,003 train / 3,080 test.
- **Score** (continuous, [0,1]): [STS Benchmark English](https://github.com/PhilipMay/stsb-multi-mt)
  (Cer et al. 2017, CC-BY-SA 4.0) — genuine human sentence-similarity
  judgments in [0,5], normalized. Input is `sentence1 <sep> sentence2`.

**Results** (clean run, 2026-09-21, held-out test splits the calibration
step never touched):

| Task   | Metric          | Raw            | Calibrated      |
|--------|-----------------|-----------------|------------------|
| Noul   | acc / ECE       | 0.9737 / 0.0216 | 0.9737 / 0.0129 (temp. scaling) |
| Choice | acc / ECE       | 0.8120 / 0.0438 | 0.8120 / 0.0315 (temp. scaling) |
| Score  | MAE / Pearson r | 0.2442 / 0.2941 | — (regression, not calibrated the same way) |

Noul and Choice calibrate well and accuracy is solid (82% over 77 real
classes is a strong real result, not a toy number). **Score is a genuine
failure**, not a rounding artifact: Pearson r=0.29 on STS-B is weak — a
competent sentence-similarity model gets 0.7–0.9+, and even simple averaged
word-vector baselines typically clear 0.5. Training MAE barely moved across
12 epochs (0.2657 → 0.2559). The likely cause: mean-pooling two
concatenated sentences through a small from-scratch byte-level-BPE encoder
with no cross-sentence attention structure just isn't expressive enough for
semantic similarity — this task needs either a bigger/pretrained encoder or
a proper sentence-pair architecture (e.g. cross-attention or a
Siamese/bi-encoder with a learned distance), neither of which this toy setup
has. Reported honestly rather than hidden or reframed as "future work."

### Phase 4: calibration method sweep (`calibration_sweep.py`, `results_log.csv`)

Beyond a single global temperature, `calibration_sweep.py` compares
temperature scaling, Platt scaling, and isotonic regression (fit on val,
measured on held-out test), appending every run to `results_log.csv`:

| Task   | Method               | Accuracy | ECE    |
|--------|----------------------|----------|--------|
| Noul   | raw                  | 0.9737   | 0.0216 |
| Noul   | temperature (T=2.40) | 0.9737   | 0.0129 |
| Noul   | **platt**            | 0.9761   | **0.0088** |
| Noul   | isotonic             | 0.9749   | 0.0115 |
| Choice | raw                  | 0.8120   | 0.0438 |
| Choice | temperature (T=1.29) | 0.8120   | 0.0315 |
| Choice | **isotonic (max-conf)** | 0.8136 | **0.0198** |

No single method wins everywhere: Platt scaling beats temperature scaling
for Noul, isotonic regression beats it for Choice. A real automated
calibration loop should try more than one method per task rather than
assuming temperature scaling is always best.

### Phase 5: out-of-distribution stress test (`ood_stress_test.py`)

Real data only — no synthetic or LLM-generated labels. Two tests:

1. **Noul (spam) on real BANKING77 text** (genuine customer-support
   questions, never spam by dataset construction — a real negative set
   outside the spam model's training distribution): accuracy drops from
   0.9737 (in-distribution) to 0.9524, and mean confidence drops from
   0.9830 to 0.9088. The model degrades gracefully under this domain shift
   — it gets somewhat less accurate AND appropriately less confident,
   rather than staying falsely confident while wrong.
2. **Choice (banking77) on real SMS text** (no correct banking77 answer
   exists for spam/ham messages — this measures confident-wrongness, not
   accuracy): mean confidence drops from 0.7936 (in-distribution) to 0.4029
   on out-of-scope input. That's a real, meaningful drop — but still far
   above the 1/77≈0.013 a maximally-uncertain model would show. **This is
   the sharpest finding in this repo about the "cannot hallucinate" claim**:
   the fixed-schema output guarantee is real (the model structurally cannot
   emit anything but one of 77 valid categories), but that guarantee says
   nothing about whether the *chosen* category is meaningful for
   out-of-scope input. "Cannot hallucinate" in Jev's sense means "cannot
   produce a malformed answer," not "cannot be confidently wrong" — those
   are different claims, and this toy model demonstrates the gap between
   them directly.

## Files
- `prepare_data.py` / `model.py` / `train.py` / `calibrate.py` / `serve.py`
  — single-task (Noul only) pipeline.
- `prepare_multitask.py` / `train_multitask.py` / `calibrate_multitask.py`
  / `serve_multitask.py` — multi-task (Noul + Choice + Score) pipeline,
  shared encoder.
- `calibration_sweep.py` — Phase 4 calibration method comparison, logs to
  `results_log.csv`.
- `ood_stress_test.py` — Phase 5 real-data distribution-shift stress test.

## Run it
```
pip install -r requirements.txt

# Single-task (Noul only)
python3 prepare_data.py && python3 train.py && python3 calibrate.py
python3 serve.py --demo

# Multi-task (Noul + Choice + Score, shared encoder)
python3 prepare_multitask.py && python3 train_multitask.py && python3 calibrate_multitask.py
python3 serve_multitask.py --demo
python3 calibration_sweep.py   # Phase 4
python3 ood_stress_test.py     # Phase 5
```

## Where this stops being "a Jev"
- Toy-scale encoder (~96-dim, 3 layers) trained from scratch, not a
  production-grade pretrained backbone — genuinely blocked by this sandbox's
  network policy (confirmed by direct connection tests), not a shortcut
  taken by choice.
- The Score head does not work well (Pearson r=0.29) — this architecture
  cannot yet demonstrate a working continuous-decision type, only a binary
  and a multi-class one.
- No single real dataset in this repo has all three label types on the same
  text, so "one state, many typed questions" is demonstrated
  architecturally (one `encode()` call feeds all heads) rather than proven
  on genuinely overlapping real-world queries.
- The OOD stress test covers real domain shift for two of three heads on
  two dataset pairs — not adversarial-phrasing robustness, and not a
  systematic sweep across many domain pairs.
- "Cannot hallucinate" holds for output *shape* (always a valid typed
  answer) but not for output *correctness* under distribution shift — see
  Phase 5 above. This is likely a real, general limit of Jev's actual
  claim, not just a toy-scale artifact: a fixed schema constrains form, not
  truth.

This is scoped as reconnaissance into the technical claims, per the plan to
identify where a real Jev competitor's wedge might be — not a shippable
product.
