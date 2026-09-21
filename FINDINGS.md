# Findings: reconnaissance into System-One decision models (Jev-clone)

Date: 2026-09-21. This document summarizes what an independent, from-scratch
reproduction of Jev's core interface shape (typed state in, typed
calibrated decision out, one non-autoregressive forward pass) found when
actually built and tested, rather than assumed. All numbers below are
measured on held-out test splits the relevant fitting step never touched,
on real human-labeled data (SMS Spam Collection, BANKING77, STS Benchmark
English) — see README.md for full dataset provenance and licensing.

## 1. What worked

- **Binary decisions (Noul) calibrate well and are accurate.** 97.37-97.85%
  accuracy on spam detection (numbers vary slightly run-to-run — see
  §3), with temperature scaling reliably cutting ECE by 40-75% depending
  on the run. This is the strongest, most reproducible result in the repo.
- **Multi-class decisions (Choice) work at real scale.** 77 real classes
  (BANKING77), 81-83% accuracy from a from-scratch toy encoder — a
  legitimately hard task, not a toy 3-5-class demo, and calibration
  (isotonic regression, in this case) cut ECE from 0.0438 to 0.0198.
- **One shared encoder genuinely serves multiple typed heads.** The
  architectural claim — one `encode()` forward pass, multiple typed
  questions answered from the same pooled state — is real in this repo,
  not just asserted: `model.py`'s `forward_all()` computes the trunk once
  and reads off three separate heads from it, and `serve_multitask.py`
  demonstrates this working end-to-end on live text.
- **Calibration method choice matters, and no single method dominates.**
  The Phase 4 sweep (`calibration_sweep.py`, `results_log.csv`) found
  Platt scaling beat temperature scaling for the binary head (ECE 0.0088
  vs 0.0129) while isotonic regression beat it for the multi-class head
  (0.0198 vs 0.0315). A serious implementation of this category should
  not assume temperature scaling is always the right calibration method.
- **Inference is fast**, as claimed. 1.51ms mean CPU latency for a single
  example across all three typed heads, in one forward pass, on a toy
  model with no GPU. See §4 for why this number needs a caveat despite
  being real.
- **Graceful degradation on real domain shift, at least for the binary
  head.** Evaluated on real, human-written text the spam model was never
  trained on (BANKING77 customer-support questions, which are provably
  never spam by how the dataset was built), accuracy dropped from 97.37%
  to 95.24% AND mean confidence dropped from 0.983 to 0.909 — the model
  got both less accurate and appropriately less confident under shift,
  rather than staying falsely confident while wrong. That's the
  calibration promise actually holding up somewhere it wasn't fit.

## 2. What didn't work

- **The Score (continuous) head is a real failure, not a rounding
  artifact.** Pearson r=0.29 against real human similarity judgments
  (STS-B). A competent sentence-similarity model scores 0.7-0.9+; even
  weak baselines (averaged word vectors) typically clear 0.5. Training
  MAE barely moved across 12 epochs (0.2657 → 0.2559) — the model
  essentially isn't learning this task. Root cause, most likely:
  mean-pooling a naively concatenated sentence pair through a small,
  from-scratch, byte-level-BPE encoder with no cross-sentence attention
  structure just isn't expressive enough for semantic similarity. This
  needs either a much larger/pretrained encoder or a proper sentence-pair
  architecture (cross-attention, or a Siamese/bi-encoder with a learned
  distance metric) — neither of which this toy setup has, and neither of
  which is a quick fix.
- **A naive "make the encoder bigger" attempt made things measurably
  worse before it got better.** Doubling encoder size at the original
  learning rate (1e-3, no gradient clipping) caused visible training
  divergence (val accuracy collapsed from ~90% to 15% partway through a
  15-epoch run). This wasn't a data problem, it was a training-stability
  problem specific to scaling a small model on a small dataset (~3,900
  examples) without adjusting the training recipe. Fixed by lowering the
  learning rate and adding gradient clipping — but it's a real reminder
  that "just make it bigger" doesn't compose for free even at toy scale.
- **"Cannot hallucinate" holds for output shape, not output
  correctness, under distribution shift.** This is the most important
  negative-but-informative result in this repo — see §5.
- **The Choice head's uncertainty response to out-of-scope input is real
  but incomplete.** Confidence dropped from 0.794 (in-distribution) to
  0.403 (on real SMS text with no valid banking-intent answer) — a
  meaningful, real signal. But 0.403 is still ~30x higher than the
  ~0.013 a maximally uncertain 77-way classifier would show. The model
  knows *something* is off, but doesn't know it's completely off-schema.

## 3. Where the calibration numbers are trustworthy vs. not

**Trustworthy, with the stated caveats:**
- The qualitative claim "temperature scaling (or Platt/isotonic) reduces
  ECE on held-out test data without changing accuracy" is well-supported:
  it held across every run in this repo, on two different real
  classification tasks, with proper val/test separation (temperature/
  Platt/isotonic parameters were fit ONLY on validation splits and
  ECE/accuracy were measured ONLY on test splits the fitting step never
  saw).
- The Choice head's 81-83% accuracy on 77 real classes is a solid,
  reproducible number from a genuinely small toy encoder.

**Not trustworthy as exact figures, and this is explicitly documented in
the code/README rather than hidden:**
- **Run-to-run exact values are not bit-reproducible even with a fixed
  seed.** Three separate clean re-runs of the single-task spam pipeline
  in this same environment, same seed (42), produced accuracy ranging
  97.13%-97.85% and ECE reduction ranging 43%-75%. This is very likely
  PyTorch/CPU non-determinism in the transformer attention implementation
  across environment/library versions, not a code bug — but it means any
  single decimal number quoted from a single run (including in this
  document, including in earlier commits' messages) should be treated as
  "roughly this, within a few points," not exact. Anyone building on this
  work should re-run the pipeline and look at the range, not trust one
  number.
- **All calibration numbers come from ONE held-out split per dataset, not
  cross-validation or repeated splits.** A single 70/15/15 train/val/test
  split (spam) or the dataset's own provided splits (BANKING77, STS-B)
  each give one ECE estimate. With test sets in the 800-3,000 example
  range, the ECE estimates themselves have non-trivial variance that this
  repo does not quantify (no bootstrap confidence intervals were
  computed). Before anyone relies on a specific ECE number as a real
  guarantee, it needs bootstrapped or cross-validated uncertainty bounds,
  which this toy-scale project did not have the scope to add.
- **The Score head's numbers (MAE=0.24, Pearson r=0.29) are trustworthy
  as "this doesn't work," not trustworthy as a precise measurement of how
  badly** — with a model this undertrained on this task, small
  architecture or hyperparameter changes could plausibly move these
  numbers substantially without changing the qualitative conclusion.

## 4. An honest estimate of what % of Jev's value proposition this reproduces

Breaking Jev's claimed value proposition into components, and rating how
much this toy reproduction actually demonstrates:

| Claim | Reproduced here? | Confidence |
|---|---|---|
| Non-autoregressive, single forward pass | **Yes, fully.** Architecturally real: one `encode()` call, typed heads read from it. | High |
| Fixed-schema output ("cannot emit malformed output") | **Yes, fully.** Structural, not learned — heads are fixed-shape linear projections. | High |
| Calibrated confidence that tracks real accuracy | **Partially.** True for Noul/Choice on in-distribution data across multiple calibration methods. Not measured with statistical rigor (no CI on ECE). Explicitly does NOT hold for "confident but wrong on out-of-scope input" (see §5). | Medium |
| Multiple typed decision shapes (Noul/Choice/Score) | **Two of three work.** Score is a real, unresolved failure at this scale. | Low-Medium |
| Speed/cost advantage over LLMs (40-200x claimed) | **Not independently verified.** This repo's own model is fast (1.51ms/example measured), but there is no LLM API in this sandbox to benchmark against directly — the comparison uses a documented industry reference figure for LLM latency, not a live measurement. The *shape* of the claim (a small non-autoregressive forward pass beats an LLM API round-trip) is directionally very plausible and structurally makes sense, but "40-200x" specifically is Jev's number, not something this repo measured against a real LLM. | Low (weakest-evidenced claim in this repo) |
| Domain breadth / general-purpose typed questions over arbitrary schemas | **No.** Three narrow, single-domain tasks (spam/ham, 77 banking intents, sentence similarity), each needing its own dataset and largely its own calibration. Jev's actual product claim is schema generality across arbitrary domains without per-domain retraining — nothing here demonstrates that. | Very Low |

**Overall**: this reproduction validates the *architectural* core of Jev's
claim (one shared encoder, typed fixed-schema heads, real calibration
improvement via standard methods) convincingly at toy scale. It does NOT
validate the *product* claims that would matter for a real
market-competitiveness assessment — domain generality, robustness at
scale, and a verified speed/cost advantage over actual LLM APIs.

**Compute/data bottlenecks, named specifically (not blurred into "needs
more time"):**
- **No pretrained encoder access.** Confirmed via direct network tests
  (not assumed): `huggingface.co` and `download.pytorch.org` are both
  blocked by this sandbox's network policy. This is the single biggest
  ceiling on quality — every number in this repo comes from a
  from-scratch encoder trained on a few thousand examples per task. A
  real Jev competitor would almost certainly start from pretrained
  weights.
- **CPU-only, single environment.** All training ran on CPU; the
  multi-task run (3 tasks x ~266 steps x 12 epochs) took ~30 minutes for
  a 96-dim/3-layer encoder. Anything meaningfully larger, or a genuine
  hyperparameter search (rather than the two or three configurations
  actually tried here), was out of reach in the time available.
- **No sentence-pair architecture for the Score task.** This isn't a
  data or compute problem, it's an architecture gap — mean-pooling two
  concatenated sentences through one encoder is a known-weak approach
  for semantic similarity, and fixing it needs a different model
  structure, not just more of the same training.
- **Small held-out test sets (800-3,080 examples per task).** Enough for
  point estimates, not enough (without bootstrapping, which wasn't done)
  for tight confidence intervals on ECE or accuracy.

## 5. The real technical wedge (or weakness) this exercise surfaced

**"Cannot hallucinate" is a claim about output *form*, not output
*truth*, and that distinction has a measurable, reproducible gap.**

The fixed-schema guarantee in this repo is completely real: the model
structurally cannot emit anything except one of a declared set of typed
answers. That part of Jev's pitch is not marketing — it's a genuine,
verifiable architectural property that autoregressive LLMs (which can, in
principle, emit any token sequence, including malformed JSON or an
off-schema answer) don't share.

But Phase 5's OOD test shows this guarantee doesn't extend to correctness.
Feed the banking-intent classifier text that has no valid banking-intent
answer (real SMS spam/ham messages), and it doesn't refuse, abstain, or
even show much uncertainty — it picks one of 77 categories with 40%
average confidence, down from 79% in-distribution but nowhere near the 1%
a genuinely "I don't know" response would imply. The model cannot produce
a *malformed* answer, but it absolutely can, and does, produce a
*confidently wrong* one when asked something outside its intended scope.

This matters commercially, not just academically: if "cannot hallucinate"
is read by a customer as "the confidence score is a trustworthy AND
complete signal that the model is out of its depth," this toy reproduction
suggests that's not automatically true even with textbook-correct
calibration (temperature/Platt/isotonic scaling, all applied and all
measurably improving ECE here). A model can be well-calibrated on its
training distribution and still fail silently — meaning confidently,
without a low-confidence flag — the moment real-world input drifts
outside that distribution. Any team evaluating this category as a
competitor to LLM-based classification should specifically demand
evidence of out-of-scope detection or rejection behavior, not just
in-distribution ECE numbers, before trusting "cannot hallucinate" as a
safety property rather than a formatting property.

That's the wedge: **the defensible claim is "cannot emit a malformed
answer." The claim that would actually matter for safety-critical use —
"knows when it doesn't know" — is a separate, harder problem that this
reproduction shows is not solved for free by fixed-schema output and
standard calibration alone.** A team pursuing this category seriously
would need explicit out-of-distribution/rejection modeling (e.g. a
learned "none of the above" option, or OOD detection on the pooled state
before committing to a typed head) as a first-class feature, not an
afterthought — and that's exactly the kind of thing worth verifying is
present (or absent) in any real competitor's actual product before taking
"cannot hallucinate" at face value.
