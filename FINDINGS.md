# Findings: reconnaissance into System-One decision models (Jev-clone)

Date: 2026-09-21. This document summarizes what an independent, from-scratch
reproduction of Jev's core interface shape (typed state in, typed
calibrated decision out, one non-autoregressive forward pass) found when
actually built and tested, rather than assumed. All numbers below are
measured on held-out test splits the relevant fitting step never touched,
on real human-labeled data (SMS Spam Collection, BANKING77, Amazon Fine
Food Reviews) — see README.md's "Dataset provenance & licensing" section
for what is and isn't independently verified for each dataset.

**Note on the Score dataset**: the Score task originally used the STS
Benchmark (sentence-pair similarity). That dataset was replaced entirely
(not just re-licensed or worked around) after a review found its
underlying text mixed multiple sub-sources with unresolved, non-uniform
licensing -- a real compliance gap in already-committed data, described
in an earlier version of this document and in the git history. It now
uses Amazon Fine Food Reviews (CC0, single-text star-rating regression)
instead. Section 2 below keeps the STS-B diagnosis-and-fix work as
historical record, since the methodology lesson it produced (a scoring
architecture bug can look like a data/capacity problem) is still real
and worth keeping visible, but the numbers under "what worked" (§1) and
the value-proposition table (§4) reflect the current Amazon Fine Food
Reviews Score task, not STS-B.

## 1. What worked

- **Binary decisions (Noul) calibrate well and are accurate.** 97.37-97.85%
  accuracy on spam detection (numbers vary slightly run-to-run — see
  §3), with temperature scaling reliably cutting ECE by 40-75% depending
  on the run. This is the strongest, most reproducible result in the repo.
- **Multi-class decisions (Choice) work at real scale.** 77 real classes
  (BANKING77), 82% accuracy from a from-scratch toy encoder — a
  legitimately hard task, not a toy 3-5-class demo, and calibration
  (isotonic regression, in this case) cut ECE from 0.0210 to 0.0113.
- **A working Score head, on the second dataset tried.** Amazon Fine
  Food Reviews (real 1-5 star ratings from the reviewer, normalized;
  see the dataset-replacement note above) gets Pearson r=0.53 on
  held-out test, with training MAE improving steadily across all 12
  epochs (0.262→0.177) rather than the flat, stuck curve the earlier
  STS-B attempt showed. Not a strong result by NLP-sentiment-model
  standards (0.7+ is common there), but a real, generalizing one — see
  §2 for the STS-B failure this replaced and why the new dataset worked
  better architecturally as well as legally.
- **One shared encoder genuinely serves multiple typed heads.** The
  architectural claim — one `encode()` forward pass, multiple typed
  questions answered from the same pooled state — is real in this repo,
  not just asserted: `model.py`'s `forward_all()` computes the trunk once
  and reads off three separate heads from it, and `serve_multitask.py`
  demonstrates this working end-to-end on live text.
- **Calibration method choice matters, and no single method dominates —
  and can even hurt.** The Phase 4 sweep (`calibration_sweep.py`,
  `results_log.csv`) found Platt scaling beat temperature scaling for
  the binary head (ECE 0.0030 vs 0.0046) while for the multi-class head,
  temperature scaling actually made ECE WORSE than doing nothing (0.0210
  raw -> 0.0371), and isotonic regression was the only method that
  helped (0.0113). A serious implementation of this category should not
  assume temperature scaling is always safe, let alone always best — on
  this run, for this task, it was actively counterproductive.
- **Inference is fast**, as claimed. ~1.4ms mean CPU latency for a single
  example across all three typed heads, in one forward pass, on a toy
  model with no GPU. See §4 for why this number needs a caveat despite
  being real.
- **Graceful degradation on real domain shift, at least for the binary
  head's confidence.** Evaluated on real, human-written text the spam
  model was never trained on (BANKING77 customer-support questions, which
  are provably never spam by how the dataset was built), mean confidence
  dropped from 0.978 (in-distribution) to 0.904 on this real OOD text —
  a same-metric, appropriately-calibrated-looking degradation under
  shift. (An earlier draft of this document also cited a matching
  accuracy before/after pair; that OOD figure is actually a pure
  true-negative rate on an all-ham OOD set, not the same statistic as
  the blended in-distribution accuracy, so it has been removed as a
  direct comparison here — see README.md's
  Phase 5 section for the corrected framing.) The confidence result is
  the calibration promise actually holding up somewhere it wasn't fit.

## 2. What didn't work (and what was fixable vs. not)

- **[HISTORICAL -- superseded, kept for the methodology lesson] The
  Score (continuous) head v1, on the STS Benchmark dataset since
  replaced for licensing reasons (see the note at the top of this
  document), was a real failure, not a rounding artifact — and the root
  cause turned out to be fixable.** This entire bullet describes work
  against a dataset no longer in this repo; the diagnostic scripts it
  references (`diagnose_score_head.py`, `fix_score_head.py`) have been
  removed along with the STS-B data, since their bi-encoder sentence-pair
  logic doesn't apply to the current single-text Amazon Fine Food Reviews
  Score task. Kept below because the methodology lesson is real and
  transferable even though the specific numbers no longer describe
  anything in this repo. The original
  architecture (concatenate the sentence pair into one string, mean-pool
  through the shared encoder, predict from one vector) scored Pearson
  r=0.29 against real human similarity judgments (STS-B), with training
  MAE barely moving across 12 epochs (0.2657 → 0.2559). A follow-up
  diagnostic (`diagnose_score_head.py`) isolated the cause precisely: encode
  the two sentences SEPARATELY through the exact same trained encoder, with
  ZERO additional training, and plain cosine similarity between the two
  pooled vectors already scores r=0.48 — beating the trained v1 head
  outright. The scoring *architecture*, not the encoder's representations,
  was the bottleneck; concatenating the pair into one string and mean-
  pooling was actively destroying signal the encoder had already captured.
  Replacing it with a standard bi-encoder/SBERT-style regression head
  (`[u, v, |u-v|]` features, Reimers & Gurevych 2019, `fix_score_head.py`)
  on the SAME frozen encoder brought MAE to 0.2174 and Pearson r to 0.4883
  — matching the untrained-cosine ceiling. r≈0.49 is still short of a
  "competent" 0.7-0.9+ similarity model, and that remaining gap is now most
  plausibly encoder capacity/pretraining (a harder, more expensive fix,
  not attempted here) rather than a scoring-architecture bug (which was
  fixable, and was fixed, in about an hour of follow-up work). The lesson:
  a "this task doesn't work" result is worth one round of architectural
  diagnosis before being written off as a data/compute ceiling — sometimes
  it's neither.
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
  but incomplete.** Confidence dropped from 0.795 (in-distribution,
  measured on the held-out test split, not the split its own temperature
  was fit on) to 0.403 (on real SMS text with no valid banking-intent
  answer) — a
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

**Not trustworthy as exact figures, but now with a real measured range
instead of a guess — this is explicitly documented in the code/README
rather than hidden:**
- **Cross-seed variance is real and now measured; a claim that the SAME
  seed and code were non-reproducible was checked and was wrong.** An
  earlier draft of this document stated that re-running the identical
  pipeline with the identical fixed seed produced different numbers
  (citing an early 97.13% acc / ECE 0.0131 run vs. a later 97.85% / 0.0039
  run as if both were seed=42 on the same code). A subsequent fresh-clone
  sanity check disproved this: the early number was from an OLDER version
  of the code (before the encoder/tokenizer upgrade in a later commit),
  and the later number is what the CURRENT code reproduces bit-for-bit,
  every time, seed=42, verified by two independent full re-runs. There is
  no PyTorch/CPU nondeterminism here — the earlier claim conflated a code
  change with run-to-run noise, and has been corrected in the README.
  What IS real and worth reporting is genuine cross-SEED variance:
  `seed_variation_sweep.py` runs 3 independent seeds (42, 123, 2024),
  each with its own data split, tokenizer, and training run (not the same
  seed re-run), giving calibrated accuracy 0.9785-0.9833 (mean 0.9801,
  std 0.0028) and calibrated ECE 0.0039-0.0070 (mean 0.0054, std 0.0015).
  (These std values use the unbiased sample-variance formula, dividing by
  n-1=2; an earlier version of this script divided by n=3, understating
  std by about 18% -- fixed, and the numbers above reflect the corrected,
  slightly wider figures.) That is still a tight, reassuring range for
  how much a genuinely different split/init changes the outcome. Note
  this sweep was run once, for one task (Noul); the Choice and Score
  heads have not had the same treatment and their numbers carry the
  same un-quantified uncertainty this section describes.
- **Calibration numbers otherwise still come from ONE held-out split per
  dataset**, not cross-validation or bootstrapping within a single split.
  The seed sweep above measures cross-run variance (different splits
  entirely), which is a stronger and more honest signal than a
  within-split bootstrap would have been, but it was only done for one
  of the three tasks due to the ~3-5 minutes of compute per seed adding
  up across tasks. Extending it to Choice and Score is straightforward
  future work using the same script as a template.
- **The Score task's numbers below reflect a single run on the new
  (Amazon Fine Food Reviews) dataset**, with no seed sweep done for it
  yet (see the point above) -- read them the same way as any single-run
  number in this repo: directionally informative, not exact.

## 4. An honest estimate of what % of Jev's value proposition this reproduces

Breaking Jev's claimed value proposition into components, and rating how
much this toy reproduction actually demonstrates:

| Claim | Reproduced here? | Confidence |
|---|---|---|
| Non-autoregressive, single forward pass | **Yes, fully.** Architecturally real: one `encode()` call, typed heads read from it. | High |
| Fixed-schema output ("cannot emit malformed output") | **Yes, fully.** Structural, not learned — heads are fixed-shape linear projections. | High |
| Calibrated confidence that tracks real accuracy | **Mostly, for the Noul task specifically.** True for Noul/Choice on in-distribution data across multiple calibration methods; for Noul, now backed by a real 3-seed uncertainty range (calibrated ECE 0.0039-0.0070, §3) rather than one number. Choice and Score have not had the same multi-seed treatment. Explicitly does NOT hold for "confident but wrong on out-of-scope input" (see §5). | Medium-High for Noul, Medium for Choice/Score |
| Multiple typed decision shapes (Noul/Choice/Score) | **All three now show real, working signal.** Score (Amazon Fine Food Reviews, r=0.53) is weaker than Noul/Choice but genuinely learns and generalizes, unlike the STS-B attempt it replaced (r=0.29, flat training curve). None reach a "production-grade" bar, but none is a dead task either. | Low-Medium |
| Speed/cost advantage over LLMs (40-200x claimed) | **Not independently verified.** This repo's own model is fast (1.51ms/example measured), but there is no LLM API in this sandbox to benchmark against directly — the comparison uses a documented industry reference figure for LLM latency, not a live measurement. The *shape* of the claim (a small non-autoregressive forward pass beats an LLM API round-trip) is directionally very plausible and structurally makes sense, but "40-200x" specifically is Jev's number, not something this repo measured against a real LLM. | Low (weakest-evidenced claim in this repo) |
| Domain breadth / general-purpose typed questions over arbitrary schemas | **No.** Three narrow, single-domain tasks (spam/ham, 77 banking intents, food-review star ratings), each needing its own dataset and largely its own calibration. Jev's actual product claim is schema generality across arbitrary domains without per-domain retraining — nothing here demonstrates that. | Very Low |

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
- **No sentence-pair architecture question anymore, since there's no
  sentence pair.** The STS-B-era finding that a naive concat+meanpool
  architecture actively hurts a sentence-pair task (§2, historical) no
  longer applies now that the Score task is single-text star-rating
  regression -- there's no pair to mishandle. The remaining gap between
  r=0.53 and a "strong" sentiment-regression result is most plausibly
  the same encoder-capacity/no-pretrained-weights bottleneck named
  above, not a new architecture problem specific to this dataset.
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
