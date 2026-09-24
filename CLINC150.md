# Workstream D: CLINC150 — a second, independent domain-breadth task

Date: 2026-09-22. Dataset: CLINC150 (`clinc/oos-eval`, Larson et al.
2019), 150 real crowdsourced intent categories spanning many domains
(banking, travel, utilities, work, small talk, and more — broader than
BANKING77's single banking domain) plus an explicit, human-curated
out-of-scope ("oos") class. License: **CC-BY 3.0**, verified directly
from `https://raw.githubusercontent.com/clinc/oos-eval/master/LICENSE`
(see `DATA_LICENSES.md`). Labels are the dataset's original
crowd-annotated intent categories — not LLM-generated, not relabeled by
this repo.

Purpose: test whether the typo-miscalibration finding from BANKING77
(FINDINGS.md §5 — ordinary 2-adjacent-character-swap typos collapse
accuracy far more than confidence drops) is a property specific to that
one dataset, or generalizes to a structurally different task (151-way
instead of 77-way, different domain mix, different-sized vocabulary,
dedicated tokenizer trained from scratch on this corpus alone rather than
the shared multitask vocabulary).

This is a **standalone single-task model** (`clinc150_pipeline.py`), not
part of the shared multitask trunk — a 151-class head this different in
domain doesn't share the multitask encoder's vocabulary or training
data, so it gets its own encoder, its own dedicated BPE tokenizer
(4,000-token vocab, trained on this corpus only), and its own
checkpoint. Same architecture (`JevCloneEncoder`, d_model=96, nhead=6,
3 layers) and training recipe (12 epochs, AdamW lr=3e-4, gradient-clipped)
as every other model in this repo.

## Results (n as noted; held-out test split, never touched during
training/calibration/conformal fitting)

| Metric | Value |
|---|---|
| Train / val / test sizes | 15,100 / 3,100 / 5,500 (150 in-scope intents + oos, combined) |
| Best val accuracy (training selection) | 0.8177 |
| **Test accuracy (overall, n=5,500)** | **0.7180** |
| Test accuracy, in-scope only (n=4,500) | 0.8418 |
| Test recall on the out-of-scope class (n=1,000) | **0.1610** |
| ECE, raw (uncalibrated) | 0.1375 |
| ECE, temperature-scaled | 0.0412 |
| Learned temperature | 1.5739 |
| Conformal (APS) coverage at 90% target (n=5,500) | 0.9827 |
| Conformal average prediction set size | 17.31 / 151 classes |
| Conformal median / max / min set size | 12 / 74 / 1 |
| Clean-test typo-robustness accuracy (n=5,500) | 0.7180 |
| Typo-perturbed accuracy | 0.4504 |
| Clean confidence | 0.7514 |
| Typo-perturbed confidence | 0.5615 |

## Finding 1: the typo-miscalibration gap generalizes

Same mechanism as `adversarial_stress_test.py` (2 adjacent-character
swaps, ordinary fast-typing typos, not adversarial intent), applied here
to a structurally different 151-class task with its own dedicated
tokenizer:

- **Accuracy** dropped 0.7180 → 0.4504 (a 26.8-point / ~37% relative
  collapse).
- **Confidence** dropped only 0.7514 → 0.5615 (a 19.0-point / ~25%
  relative drop).

This is the same qualitative pattern found on BANKING77 (81.9%→51.2%
accuracy collapse vs 0.791→0.612 confidence drop, a ~37% relative
accuracy drop against a ~23% relative confidence drop) — the magnitudes
are close, not just the direction. **This is the strongest evidence yet
that the typo-miscalibration finding is not an artifact of BANKING77's
tokenizer or domain specifically**: two different datasets, different
vocabularies (8,000-token shared multitask BPE vs this task's own
4,000-token dedicated BPE), different class counts (77 vs 151), and
different domains (single-domain banking vs 150-domain broad intent) all
show confidence failing to track the real accuracy loss under the exact
same ordinary-typo perturbation, by a similar relative margin. This
strengthens, rather than merely repeats, the wedge described in
FINDINGS.md §5.

## Finding 2: the "oos" class alone catches few out-of-scope queries, but the confidence score can catch most of them, at a price

CLINC150's design gives this model something BANKING77 never had: a
genuine, human-labeled "none of the above" class to learn, directly
addressing the gap FINDINGS.md §5 identifies ("a learned 'none of the
above' option... as a first-class feature").

**Used as just one more softmax class, it mostly fails: the model
recognizes an actual out-of-scope query only 16.1% of the time** (recall
on the 1,000-example oos test set), while reaching 84.2% accuracy on the
4,500 in-scope test examples. The overall 71.8% test accuracy is a
weighted blend of these two very different numbers and is not meaningful
on its own; both halves are reported explicitly.

**Correction to an earlier version of this document.** It blamed the low
recall on "severe class imbalance" (oos being 0.67% of training data,
"150:1"). That was wrong: CLINC150's training set is balanced, with
exactly 100 examples for every class, oos included (verified by counting
`data_full.json`), so every class is about 0.66% of training data and
frequency-based class weighting would change nothing. The real
difference is what the 100 examples must cover: an in-scope intent is
one narrow request type, while oos is everything else. The test split
also contains 1,000 oos queries versus 30 per intent.

**The confidence score carries most of the missing signal**
(`clinc150_oos_threshold.py`). On the test split, the model's top-class
probability averages 0.82 on in-scope queries but 0.44 on out-of-scope
ones, and (1 - top probability) separates the two with AUROC 0.88. Using
the CLINC150 paper's standard remedy, answering "oos" whenever the top
probability is below a threshold tau chosen on the validation split only
(tau = 0.67):

| Test metric (n=5,500) | Argmax only | Threshold rejection (tau = 0.67) |
|---|---|---|
| In-scope accuracy (n=4,500) | 0.8418 | 0.7449 |
| Out-of-scope recall (n=1,000) | 0.1610 | **0.8570** |
| Out-of-scope precision | 0.8895 | 0.4612 |
| Overall accuracy | 0.7180 | 0.7653 |

So the model does, in a usable sense, "know when it doesn't know" more
often than its argmax answer suggests. But the separation is far from
clean. Catching 86% of out-of-scope queries also rejects enough real
in-scope queries to cost almost 10 points of in-scope accuracy, and more
than half of all "oos" answers are then wrong. Moving tau trades one
against the other (on test: tau 0.4 gives 82.4% in-scope accuracy and
60.3% oos recall; tau 0.8 gives 68.4% and 92.7%). The fixed schema
guarantees a well-formed answer; how often the answer means "I don't
know" when it should is a tunable tradeoff, not a property you get for
free.

## Why val accuracy (0.8177) and test accuracy (0.7180) differ so much

Not overfitting, and not a bug: CLINC150's own train/val/test split design
puts far more out-of-scope examples in test (1,000, 18.2% of the split)
than in val (100, 3.2% of the split) — train has only 100 as well. Val
accuracy is measured on a split that's overwhelmingly in-scope (like
training), so it doesn't reflect the model's (poor) out-of-scope
performance nearly as much as the test split, which was deliberately
built with 10x more out-of-scope examples specifically to stress-test
out-of-scope detection. This is documented explicitly here rather than
left as an unexplained accuracy gap, since it's the kind of thing this
project's rules require flagging rather than glossing over.

## Conformal prediction result: valid, and here genuinely not very useful

Coverage (98.27%) comfortably exceeds the 90% target — the guarantee
holds, consistent with Workstream A's other results, all of which
over-covered rather than under-covered (a conservative, not a broken,
result). But the average prediction set size (17.31 of 151 classes,
~11.5%) is proportionally similar to BANKING77's Choice result (avg ~7 of
77, ~9%) — in both cases, guaranteeing 90% coverage over a many-class
softmax with real per-class ambiguity requires a set that's too wide to
narrow a user's next action much on its own. This is consistent with
CONFORMAL.md's finding that a valid conformal set can still be
practically loose — reproduced again here on an independent dataset, not
contradicted.

## What this does and doesn't add to the project

- **Confirms** (does not merely repeat) the typo-miscalibration wedge on
  a second, structurally different dataset — meaningfully strengthens
  that finding's claim to generalize beyond one dataset's quirks.
- **Adds a new finding on out-of-scope detection**: an explicit oos class
  used as one more softmax output catches only 16.1% of out-of-scope
  queries, even though training is balanced. The confidence score
  separates oos from in-scope reasonably well (AUROC 0.88), and a
  validation-tuned threshold raises oos recall to 85.7%, at a cost of
  about 10 points of in-scope accuracy. This is a measured version of the
  "cannot hallucinate ≠ knows what it doesn't know" gap in FINDINGS.md §5:
  the signal exists, but using it is a tradeoff.
- **Does not** try a dedicated OOD head, extra oos training data, or
  outlier exposure, which would be the next experiments for a better
  detector.
