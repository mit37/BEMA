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

## Finding 2 (new, not previously observed in this repo): an explicit
"I don't know" class doesn't help much if it's severely underrepresented
in training

CLINC150's design gives this model something BANKING77 never had: a
genuine, human-labeled "none of the above" class to learn, directly
addressing the exact gap FINDINGS.md §5 identifies as missing from this
project's other tasks ("a learned 'none of the above' option... as a
first-class feature"). The result is a genuinely useful, honest negative
finding: **having the class in the schema is not sufficient if the
training data is severely imbalanced.**

The out-of-scope class makes up only 100 of 15,000 training examples
(0.67%) — a real property of CLINC150's own split design, not something
this repo chose. The consequence: **the model recognizes an actual
out-of-scope query only 16.1% of the time** (recall on the 1,000-example
oos test set), while achieving 84.2% accuracy on the 4,500 in-scope test
examples. The overall 71.8% test accuracy is a weighted blend of these
two very different numbers — it is NOT a single meaningful accuracy
figure on its own, and reporting it without the in-scope/out-of-scope
split (as `clinc150_pipeline.py`'s summary line alone would) would
understate how well the model handles in-domain queries while also
overstating how well it handles unfamiliar ones. Both halves are reported
here explicitly rather than collapsed into one number.

This is a direct, concrete confirmation of FINDINGS.md §5's wedge finding
("the model cannot produce a malformed answer, but it absolutely can,
and does, produce a confidently wrong one") — except this time the model
*had* the correct typed answer available in its schema and mostly failed
to use it, because of a training-data imbalance problem rather than an
architectural one. A team building a real out-of-scope detector on this
architecture would need either substantially more OOS training examples,
class-weighted loss (as this repo already does for Noul's spam/ham
imbalance — see `train_multitask.py`'s `pos_weight` — but did not apply
here, since CLINC150's imbalance (150:1) is far more extreme and a
150-way softmax cross-entropy doesn't have as direct an equivalent), or a
dedicated OOD-detection mechanism on the pooled state (as FINDINGS.md §5
already recommends) rather than relying on the softmax class alone.

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
- **Adds a new, previously-untested finding**: an explicit out-of-scope
  class in the schema is not sufficient for reliable out-of-scope
  detection if training data for that class is severely imbalanced — a
  concrete, measured instance of the "cannot hallucinate ≠ knows what it
  doesn't know" gap FINDINGS.md §5 already argued for on more general
  grounds.
- **Does not** attempt a from-scratch fix for the OOS-imbalance problem
  (e.g. class-weighted loss, oversampling the oos class, a dedicated
  binary OOD head) — that's a natural next experiment but out of scope
  for this workstream, which was about domain breadth and generalization
  of the typo finding, not about building a better OOD detector.
