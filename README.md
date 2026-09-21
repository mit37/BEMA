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
- **Score** (continuous, [0,1]): [STS Benchmark English](https://github.com/PhilipMay/stsb-multi-mt)
  (Cer et al. 2017) — genuine human sentence-similarity judgments in
  [0,5], normalized. Input is `sentence1 <sep> sentence2`. Licensing is
  NOT a single clean tag — see "Dataset provenance & licensing" below,
  which flags a real, unresolved compliance question about this dataset
  rather than the earlier (incorrect) blanket "CC-BY-SA 4.0" this section
  used to state.

## Dataset provenance & licensing

This section exists because an internal review of this repo (see
FINDINGS.md) found the earlier dataset citations here were incomplete or
inconsistent with what this repo actually verified. Corrected below, with
exact sources and what is/isn't independently confirmed:

| Dataset | Source (fetched from) | Retrieved | License |
|---|---|---|---|
| SMS Spam Collection | `data/spam.csv` — bundled with this project from its start, original upstream source not independently re-fetched or re-verified by this repo | n/a (pre-existing file) | **Not independently verified.** Long-standing academic benchmark (Almeida & Gómez Hidalgo), historically distributed for research use; this repo has not confirmed the current authoritative license text against the original source (`archive.ics.uci.edu`, which this sandbox's network policy blocks) or against the dataset's canonical upstream page. Treat as research-use-only until verified. |
| BANKING77 | `https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{train,test}.csv` and `categories.json` | 2026-09-21 | **CC-BY 4.0 — confirmed** by fetching and reading `https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/LICENSE` directly (an "Attribution 4.0 International" / CC-BY 4.0 license file), not just citing the paper. |
| STS Benchmark English (via stsb-multi-mt) | `https://raw.githubusercontent.com/PhilipMay/stsb-multi-mt/master/data/stsb-{train,dev,test}.csv` | 2026-09-21 | **Not a single license — genuinely mixed, and NOT fully addressed by this repo (flagged, not resolved).** See below. |

**The STS-B licensing question, specifically** (this is the one item in
this repo that needed a human decision and didn't get one yet — see the
note at the end of this section): the stsb-multi-mt mirror's own
`LICENSE` file (fetched and read directly, not assumed) states that while
the numeric *scores* are CC-BY-SA 4.0, the underlying *sentence text* is
NOT uniformly licensed — it's assembled from multiple original sources,
each with its own terms:
- **MSRpar** and **MSR-Video** subsets: require researchers to separately
  agree to Microsoft Research's own license terms — not simply CC-licensed.
- **answers-answers** / **answers-forums** subsets: real Stack Exchange
  user content, CC-BY-SA **3.0** (not 4.0), with specific per-post,
  per-author attribution requirements (a hyperlink to each source post AND
  each author's profile) that this repo's redistributed CSVs do not carry.
- **headlines**: European Media Monitor content, requesting acknowledgment
  when reused.
- **track5.en-en** (SNLI-derived) and **image** (PASCAL VOC-2008
  captions): CC-BY-SA (4.0 for track5.en-en; version unspecified for image
  captions).

`data/raw/stsb_{train,dev,test}.csv` in this repo mixes rows from all of
these sub-sources without per-row sub-license tracking, and this repo has
NOT implemented the Stack Exchange per-post/per-author attribution or
documented agreement to the Microsoft Research terms that some rows
require. **This is a real, currently-unresolved licensing compliance gap
in already-committed data, not a hypothetical one** — flagged here
explicitly per this project's own ground rules (keep dataset sourcing
legal and above-board; stop and flag rather than guess when a real
decision is needed) rather than smoothed over with an incorrect blanket
license tag. Options going forward (not decided in this repo, needs a
human call): (a) accept the risk for internal/research-only use and
document that clearly, (b) do the attribution work required for the
Stack Exchange subset and confirm the MSR terms for MSRpar/MSR-Video, or
(c) replace the Score task's dataset with one that has cleaner, single-
license terms. This repo does not resolve that choice on its own.

There is also no repository-level LICENSE file for this project itself —
another item that needs an explicit decision from whoever is publishing
this repo, not an assumption made on their behalf.

**Results** (clean run, 2026-09-21, held-out test splits the calibration
step never touched):

| Task   | Metric          | Raw            | Calibrated      |
|--------|-----------------|-----------------|------------------|
| Noul   | acc / ECE       | 0.9737 / 0.0216 | 0.9737 / 0.0129 (temp. scaling) |
| Choice | acc / ECE       | 0.8120 / 0.0438 | 0.8120 / 0.0315 (temp. scaling) |
| Score (v1, concat+meanpool+MLP) | MAE / Pearson r | 0.2442 / 0.2941 | — (regression, not calibrated the same way) |

Noul and Choice calibrate well and accuracy is solid (82% over 77 real
classes is a strong real result, not a toy number). **Score (v1) was
initially a genuine failure**, not a rounding artifact: Pearson r=0.29 on
STS-B is weak — a competent sentence-similarity model gets 0.7–0.9+, and
even simple averaged word-vector baselines typically clear 0.5.

**Follow-up diagnosis and partial fix** (`diagnose_score_head.py`,
`fix_score_head.py`): the failure was the scoring *architecture*, not the
encoder. Encoding the two sentences SEPARATELY through the exact same
trained encoder (no retraining) and just taking cosine similarity — zero
additional training — already scores r=0.48, beating the trained v1 head
outright. Replacing the head with a standard bi-encoder/SBERT-style
regression (`[u, v, |u-v|]` features from the two separately-encoded
sentences, fit on real STS-B train, frozen encoder) gets MAE 0.2442→0.2174
and Pearson r 0.2941→0.4883, roughly matching the untrained-cosine ceiling.
The remaining gap to a "competent" 0.7–0.9+ model is now most plausibly
encoder capacity/pretraining, not the scoring architecture — concatenating
a sentence pair into one string and mean-pooling was actively destroying
signal a bi-encoder recovers "for free" from the same weights. Left as
future work: fine-tuning the shared encoder itself against a similarity
objective (not just the frozen-trunk regression head done here), or
swapping in pretrained weights if this sandbox's network policy ever
allows it.

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
| Choice | **isotonic (max-conf)** | 0.8120 | **0.0189** |

(Corrected from an earlier version of this table, which had a real bug: the
Choice isotonic method's implementation overwrote one class's probability
in-place before re-taking the argmax for prediction, which could silently
flip the recorded prediction to a class the model never actually chose —
that's why Choice/isotonic's earlier accuracy, 0.8136, differed from raw/
temperature's 0.8120 when it structurally shouldn't have, since isotonic
here is documented as confidence-only recalibration. Fixed in
`calibration_sweep.py` to compute the prediction from the raw, uncalibrated
softmax always, and only ever recalibrate the reported confidence value.)

No single method wins everywhere: Platt scaling beats temperature scaling
for Noul, isotonic regression beats it for Choice. A real automated
calibration loop should try more than one method per task rather than
assuming temperature scaling is always best.

### Phase 5: out-of-distribution stress test (`ood_stress_test.py`)

Real data only — no synthetic or LLM-generated labels. Two tests:

1. **Noul (spam) on real BANKING77 text** (genuine customer-support
   questions, never spam by dataset construction — a real negative set
   outside the spam model's training distribution): every example here
   is ground-truth ham, so the 0.9524 figure is a pure true-negative rate
   (specificity), NOT the same statistic as the 0.9737 in-distribution
   accuracy, which is a class-weighted blend of true-positive and
   true-negative rates over the original ~13%-spam/87%-ham test set —
   these are different metrics measured on differently-composed
   populations, not a clean apples-to-apples before/after (the OOD
   script's own output honestly labels this `accuracy(=1-false_spam_rate)`
   rather than plain "accuracy"). What IS a clean, comparable measurement
   is confidence: mean confidence drops from 0.9830 (in-distribution) to
   0.9088 on this real non-spam OOD text — a same-metric, appropriately-
   calibrated-looking degradation under domain shift.
2. **Choice (banking77) on real SMS text** (no correct banking77 answer
   exists for spam/ham messages — this measures confident-wrongness, not
   accuracy): mean confidence drops from 0.7949 (in-distribution, measured
   on the Choice head's held-out test split, not the split its own
   temperature was fit on) to 0.4029 on out-of-scope input. That's a real,
   meaningful drop — but still far
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
- `seed_variation_sweep.py` — measures real cross-seed variance on the
  Noul pipeline's accuracy/ECE (3 independent seeds), logs to
  `results_log.csv`.
- `diagnose_score_head.py` / `fix_score_head.py` — Score head follow-up:
  diagnoses the concat+meanpool architecture's failure and fixes it with
  a bi-encoder scoring head on the same frozen encoder.
- `benchmark_speed.py` — measures single-example CPU inference latency
  for the multi-task model (the "1.51ms" figure cited in FINDINGS.md).

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

# Score head follow-up (diagnosis + bi-encoder fix)
python3 diagnose_score_head.py
python3 fix_score_head.py

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
- The Score head, even after the bi-encoder fix, tops out around Pearson
  r=0.49 — real signal, well short of a "competent" 0.7-0.9+ similarity
  model. The scoring-architecture bug is fixed; encoder capacity/pretraining
  is the remaining, harder-to-fix bottleneck.
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
