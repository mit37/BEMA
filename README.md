# Jev-clone: toy System-One decision model

A minimal, real (not simulated) reproduction of Jev's core interface shape:
send text state in, get a typed decision + calibrated confidence back in
one forward pass — no token generation.

## What's real here
- **Data**: SMS Spam Collection, 5,572 real human-labeled messages (ham/spam).
  This is a long-standing, widely cited academic NLP benchmark (Almeida &
  Gómez Hidalgo); the "not LLM-generated" claim rests on that publication
  history and this dataset's age (pre-dating widespread LLM use), not on
  independent verification of the original annotation records from inside
  this repo. See "Dataset provenance & licensing" below for what is and
  isn't independently verified here, and for a licensing question this
  repo has NOT resolved and flags explicitly rather than guessing at.
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
- **Results** (`python3 prepare_data.py && python3 train.py && python3 calibrate.py`,
  seed fixed at 42, tuned encoder + BPE tokenizer + stabilized training
  recipe): **97.85% test accuracy**; Expected Calibration Error (ECE)
  dropped from **0.0154 (raw) to 0.0039 (calibrated)** — a 74.7% reduction.
  This beats the original single-task baseline reported earlier in this
  repo's history (97.13% acc / 0.0231→0.0131 ECE, from before the encoder/
  tokenizer upgrade). Re-verified with a full fresh-clone-style re-run
  (all generated artifacts deleted and regenerated) on 2026-09-21: this
  exact command sequence with this exact code reproduces these exact
  numbers bit-for-bit — an earlier claim in this repo's history that
  results were "not bit-reproducible even with a fixed seed" was based on
  comparing runs across DIFFERENT versions of this code (before vs. after
  the encoder/tokenizer changes), not genuine nondeterminism; that claim
  was incorrect and has been corrected here. `seed_variation_sweep.py`
  separately measures genuine cross-seed variance (different data splits
  and model inits, not just re-running the same seed) — see the Phase 4
  section below for that real uncertainty range.
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
  categories, 10,003 train / 3,080 test. License: CC-BY 4.0, confirmed by
  reading the LICENSE file in the source repository directly (not just
  citing the paper) — see below.
- **Score** (continuous, [0,1]): [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
  (McAuley & Leskovec 2013, SNAP/Stanford) — real customer star ratings
  (1-5, normalized to [0,1]) paired with the reviewer's own review text.
  License: **CC0 (public domain)**, published under that tag by the
  dataset's own creators' Kaggle account. Single-text input, no
  sentence-pair concatenation. (This replaces an earlier version of this
  Score task that used the STS Benchmark, which was dropped entirely
  after a review found its underlying text mixed several sub-sources
  with unresolved, non-uniform licensing — see FINDINGS.md and the git
  history for that decision.)

## Dataset provenance & licensing

This section exists because an internal review of this repo (see
FINDINGS.md) found the earlier dataset citations here were incomplete or
inconsistent with what this repo actually verified — and because the
Score dataset was later replaced entirely for licensing reasons. Exact
sources and what is/isn't independently confirmed:

| Dataset | Source (fetched from) | Retrieved | License |
|---|---|---|---|
| SMS Spam Collection | `data/spam.csv` — bundled with this project from its start, original upstream source not independently re-fetched or re-verified by this repo | n/a (pre-existing file) | **Not independently verified.** Long-standing academic benchmark (Almeida & Gómez Hidalgo), historically distributed for research use; this repo has not confirmed the current authoritative license text against the original source (`archive.ics.uci.edu`, which this sandbox's network policy blocks) or against the dataset's canonical upstream page. Treat as research-use-only until verified. |
| BANKING77 | `https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{train,test}.csv` and `categories.json` | 2026-09-21 | **CC-BY 4.0 — confirmed** by fetching and reading `https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/LICENSE` directly (an "Attribution 4.0 International" / CC-BY 4.0 license file), not just citing the paper. |
| Amazon Fine Food Reviews | `data/raw/amazon_food_reviews.csv` — the first 10,112 rows of the original ~568,454-row dataset, obtained via a third-party GitHub mirror since Kaggle itself requires a login this sandbox cannot provide | 2026-09-21 | **CC0 (public domain) — corroborated by multiple independent sources** (the dataset's own Kaggle listing under the SNAP/creator account, and its Hugging Face mirror, both tagging it CC0-1.0), though this sandbox could not reach kaggle.com directly to read the license field itself — see the caveat below. |

**On verifying the Amazon Fine Food Reviews license, specifically**: unlike
BANKING77 (where this repo fetched and read an actual LICENSE file byte
for byte), `kaggle.com` is unreachable from this sandbox, so the CC0 claim
here rests on corroborating web-search results describing the Kaggle
listing's own license tag plus a matching tag on a Hugging Face mirror,
not a first-party document this repo read directly. This is a weaker
form of verification than BANKING77's, though still meaningfully stronger
than the earlier STS-B situation (a downstream mirror's own license file,
read directly, that turned out to describe non-uniform sub-licenses this
repo hadn't accounted for) — here, multiple independent third parties
agree on a single, simple, maximally permissive tag (CC0) for a dataset
published directly by its original creators, and CC0 is not a NC/SA-
style license that residual ambiguity could turn into a compliance trap
the way STS-B's did. Flagged transparently rather than overstated as a
first-party confirmation it isn't.

The Amazon Fine Food Reviews CSV bundled here is a truncated slice (first
10,112 of ~568,454 rows), not the full dataset — its first two rows were
spot-checked against the well-known, widely-reproduced first rows of the
original dataset ("Good Quality Dog Food" / "Not as Advertised") to
confirm the mirror is genuine and not fabricated or reordered.

This repository's own code is MIT-licensed (see `LICENSE`). That covers
the code in this repo (Python scripts, `model.py`, etc.) only — it does
NOT change or override the separate licenses on the bundled third-party
datasets described in the table above, each of which keeps its own
terms regardless of this repo's code license.

**Results** (clean run, 2026-09-21, held-out test splits the calibration
step never touched, Score dataset = Amazon Fine Food Reviews as described
above):

| Task   | Metric          | Raw            | Calibrated      |
|--------|-----------------|-----------------|------------------|
| Noul   | acc / ECE       | 0.9785 / 0.0159 | 0.9785 / 0.0046 (temp. scaling) |
| Choice | acc / ECE       | 0.8188 / 0.0210 | 0.8188 / 0.0371 (temp. scaling — see note below) |
| Score  | MAE / Pearson r | 0.1872 / 0.5340 | — (regression, not calibrated the same way) |

**Score works reasonably well here**, unlike the STS-B task it replaced:
Pearson r=0.53 on a genuinely held-out test split, with training MAE
improving steadily across all 12 epochs (0.262→0.177) rather than
plateauing immediately the way the STS-B run did. r=0.53 is still short
of a strong sentiment/rating model (competent models on similar review-
rating tasks often reach 0.7+), but it's a real, working, generalizing
Score head — `serve_multitask.py --demo`'s third example ("Terrible.
Stale, tasteless, and arrived broken.") scores 0.15, correctly low,
despite that exact phrasing never appearing in the training reviews.

**A genuine, honest surprise: temperature scaling made Choice's ECE
*worse*, not better** (0.0210 raw → 0.0371 after temp. scaling). This can
legitimately happen — a global temperature is fit on validation data and
doesn't guarantee improvement on a different held-out test split,
especially when the raw model happens to already be well-calibrated on
that specific test split. Isotonic regression is a safer bet here: see
the Phase 4 sweep below, where it recovers a lower ECE (0.0113) than both
raw and temperature scaling for Choice. This is reported because it's a
real result from this exact run, not smoothed into "worked as expected."

Noul and Choice calibrate well and accuracy is solid (82% over 77 real
classes is a strong real result, not a toy number). See FINDINGS.md for
how the Score task performed on this dataset and what that does and
doesn't tell us, including the now-inapplicable history of a prior
diagnosis/fix effort that was specific to the STS-B sentence-pair
architecture this repo no longer uses.

### Phase 4: calibration method sweep (`calibration_sweep.py`, `results_log.csv`)

Beyond a single global temperature, `calibration_sweep.py` compares
temperature scaling, Platt scaling, and isotonic regression (fit on val,
measured on held-out test), appending every run to `results_log.csv`:

| Task   | Method               | Accuracy | ECE    |
|--------|----------------------|----------|--------|
| Noul   | raw                  | 0.9785   | 0.0159 |
| Noul   | temperature (T=2.36) | 0.9785   | 0.0046 |
| Noul   | **platt**            | 0.9797   | **0.0030** |
| Noul   | isotonic             | 0.9773   | 0.0049 |
| Choice | raw                  | 0.8188   | 0.0210 |
| Choice | temperature (T=1.18) | 0.8188   | 0.0371 (worse than raw — see note above) |
| Choice | **isotonic (max-conf)** | 0.8188 | **0.0113** |

No single method wins everywhere: Platt scaling beats temperature scaling
for Noul; for Choice, temperature scaling actually made ECE worse than
doing nothing, while isotonic regression clearly won. A real automated
calibration loop should try more than one method per task rather than
assuming temperature scaling is always best — this run makes that case
more sharply than the last one, since temperature wasn't just
suboptimal here, it was actively counterproductive.

(An earlier version of this table had a real bug: the Choice isotonic
method's implementation overwrote one class's probability in-place before
re-taking the argmax for prediction, which could silently flip the
recorded prediction to a class the model never actually chose. Fixed in
`calibration_sweep.py` to compute the prediction from the raw,
uncalibrated softmax always, and only ever recalibrate the reported
confidence value — the numbers above are all from the corrected code.)

### Phase 5: out-of-distribution stress test (`ood_stress_test.py`)

Real data only — no synthetic or LLM-generated labels. Two tests:

1. **Noul (spam) on real BANKING77 text** (genuine customer-support
   questions, never spam by dataset construction — a real negative set
   outside the spam model's training distribution): every example here
   is ground-truth ham, so the 0.9317 figure is a pure true-negative rate
   (specificity), NOT the same statistic as the 0.9785 in-distribution
   accuracy, which is a class-weighted blend of true-positive and
   true-negative rates over the original ~13%-spam/87%-ham test set —
   these are different metrics measured on differently-composed
   populations, not a clean apples-to-apples before/after (the OOD
   script's own output honestly labels this `accuracy(=1-false_spam_rate)`
   rather than plain "accuracy"). What IS a clean, comparable measurement
   is confidence: mean confidence drops from 0.9780 (in-distribution) to
   0.9036 on this real non-spam OOD text — a same-metric, appropriately-
   calibrated-looking degradation under domain shift.
2. **Choice (banking77) on real SMS text** (no correct banking77 answer
   exists for spam/ham messages — this measures confident-wrongness, not
   accuracy): mean confidence drops from 0.7906 (in-distribution, measured
   on the Choice head's held-out test split, not the split its own
   temperature was fit on) to 0.4097 on out-of-scope input. That's a real,
   meaningful drop — but still far
   above the 1/77≈0.013 a maximally-uncertain model would show. This is a
   real limit on the "cannot hallucinate" claim: the fixed-schema output
   guarantee is real (the model structurally cannot emit anything but one
   of 77 valid categories), but that guarantee says nothing about whether
   the *chosen* category is meaningful for out-of-scope input. "Cannot
   hallucinate" in Jev's sense means "cannot produce a malformed answer,"
   not "cannot be confidently wrong" — those are different claims, and
   this toy model demonstrates the gap between them directly.

### Phase 5b: adversarial / ambiguous-input stress test (`adversarial_stress_test.py`)

Cross-domain shift (Phase 5 above) isn't the only way real input differs
from a friendly test split. Three more tests, all built from real,
correctly-labeled data (documented perturbations of real text, never
synthetic labels):

1. **Real hard negatives**: real ham SMS messages that happen to contain
   classic spam-trigger words ("free", "win", "prize", "urgent", etc.) —
   100% accuracy on this real 50-example subset vs 99.1% on other ham.
   The model is not meaningfully fooled by superficial lexical cues on
   real messages.
2. **Mechanical filter-evasion perturbation** of real, correctly-
   classified spam (character spacing, leetspeak substitution, case
   noise — the kind of thing real spammers actually do to evade
   filters): accuracy drops only slightly (89.7%→88.8%) with confidence
   dropping proportionally (0.972→0.926) — reasonably graceful
   degradation.
3. **Typo robustness on real banking77 queries** (2 adjacent-character
   swaps per query — an ordinary fast-typing typo, verified by hand:
   "How do I locate my card?" → "Howd o I locate my crad?", fully
   readable to a human): **accuracy collapses from 81.9% to 51.2% — a
   30.7-point drop — while confidence only drops from 0.791 to 0.612.**

**This, not the cross-domain OOD result above, is the sharpest finding in
this repo.** A tiny amount of completely realistic noise — the kind every
real customer-service deployment sees constantly, not an adversarial
attack — nearly halves accuracy while confidence drops by less than a
third. The model isn't just "confidently wrong on inputs outside its
schema" (Phase 5's finding); it's **badly miscalibrated on inputs squarely
inside its intended domain**, the moment those inputs contain the kind of
noise real text always has. Good in-distribution ECE numbers (Phase
3/4 above) say nothing about this failure mode — it would not show up in
any of this repo's other calibration measurements, because those all
evaluate on clean, unperturbed test text. Anyone evaluating this category
of model for real deployment should demand a typo/noise robustness number
specifically, not just clean-test-set calibration metrics — this repo's
own numbers would have looked strong without this test, and were not.

## Files
- `prepare_data.py` / `model.py` / `train.py` / `calibrate.py` / `serve.py`
  — single-task (Noul only) pipeline.
- `prepare_multitask.py` / `train_multitask.py` / `calibrate_multitask.py`
  / `serve_multitask.py` — multi-task (Noul + Choice + Score) pipeline,
  shared encoder.
- `calibration_sweep.py` — Phase 4 calibration method comparison, logs to
  `results_log.csv`.
- `ood_stress_test.py` — Phase 5 real-data distribution-shift stress test.
- `adversarial_stress_test.py` — Phase 5b real-data adversarial/typo
  robustness stress test (the repo's sharpest finding — see above).
- `seed_variation_sweep.py` — measures real cross-seed variance on the
  Noul pipeline's accuracy/ECE (3 independent seeds), logs to
  `results_log.csv`.
- `benchmark_speed.py` — measures single-example CPU inference latency
  for the multi-task model (the "~1.4ms" figure cited in FINDINGS.md).

`results_log.csv` is intentionally tracked in git (unlike model
checkpoints or pickled data caches, which are gitignored as reproducible
build output) — it is designed to accumulate results across runs over
time, per its own purpose as an inspectable history, so `git status`
being non-clean after running the scripts below that append to it is
expected, not a bug.

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
python3 adversarial_stress_test.py  # Phase 5b

# Real cross-seed uncertainty measurement (Noul pipeline, ~10-15 min for 3 seeds)
python3 seed_variation_sweep.py

# CPU inference latency benchmark (requires the multi-task pipeline above)
python3 benchmark_speed.py
```

## Where this stops being "a Jev"
- Toy-scale encoder (~96-dim, 3 layers) trained from scratch, not a
  production-grade pretrained backbone — genuinely blocked by this sandbox's
  network policy (confirmed by direct connection tests), not a shortcut
  taken by choice.
- The Score head reaches Pearson r=0.53 (Amazon Fine Food Reviews star
  ratings) — real, generalizing signal, but short of what a competent
  sentiment/rating model would reach (0.7+ is common). Encoder capacity
  and the lack of pretrained weights (see above) are the most plausible
  remaining bottleneck.
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
