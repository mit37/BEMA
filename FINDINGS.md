# Findings: reconnaissance into System-One decision models (Jev-clone)

Date: 2026-09-21, updated 2026-09-22 after Workstreams A-E (split
conformal prediction, adversarial typo-noise augmentation, the real-LLM
benchmark disposition, a second independent domain-breadth dataset, and
one architecture experiment — see CONFORMAL.md, CLINC150.md, and the
sections below for each), and 2026-09-23 with follow-ups that closed the
gaps those left open: augmentation tested on unseen noise types,
out-of-scope rejection on CLINC150, adaptive (CQR) intervals for Score,
and a multi-seed pooling comparison. That round also corrected two
earlier claims (CLINC150 "class imbalance"; the Score interval being
"wider than the output range"), marked where they appear. This document summarizes what an independent,
from-scratch reproduction of Jev's core interface shape (typed state in,
typed calibrated decision out, one non-autoregressive forward pass)
found when actually built and tested, rather than assumed. All numbers
below are measured on held-out test splits the relevant fitting step
never touched, on real human-labeled data (SMS Spam Collection,
BANKING77, Amazon Fine Food Reviews, and CLINC150 for the standalone
Workstream D task) — see README.md's "Dataset provenance & licensing"
section and `DATA_LICENSES.md` for what is and isn't independently
verified for each dataset.

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
- **Conformal prediction's coverage guarantee genuinely holds, verified
  empirically on held-out test data — but "valid" and "useful" turn out
  to be different questions per head.** See `CONFORMAL.md` for the full
  writeup. Noul hit 90.91% coverage (target 90%) with tight sets, but
  8.37% of test examples got an EMPTY prediction set (neither class
  confident enough) — a real, honest refusal rate a product would need
  a policy for. Choice hit 98.83% coverage (conservative, as the
  non-randomized APS method used here is known to be) with an average
  set size of 7 of 77 classes — informative but not tight. Score hit
  89.85% coverage, but its constant-width interval spans on average
  0.656 of the [0,1] rating scale (about 2.6 stars) once clipped to the
  valid range. (An earlier version of this document called the interval
  "wider than the entire output range"; that described the raw,
  unclipped width of 1.066 and overstated the problem.) Conformalized
  quantile regression (`score_cqr.py`) keeps coverage at 90.11% with a
  narrower, input-adaptive interval (mean 0.608; 41% of intervals under
  2 stars), but coverage is uneven: 97% for 5-star reviews and only 52%
  for 1-star reviews. The 90% guarantee is an average, and it holds here
  because 61% of reviews are 5-star.
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
  correctness, under distribution shift.** This is an important
  negative-but-informative result in this repo — see §5.
- **The Choice head's uncertainty response to out-of-scope input is real
  but incomplete.** Confidence dropped from 0.795 (in-distribution,
  measured on the held-out test split, not the split its own temperature
  was fit on) to 0.403 (on real SMS text with no valid banking-intent
  answer) — a
  meaningful, real signal. But 0.403 is still ~30x higher than the
  ~0.013 a maximally uncertain 77-way classifier would show. The model
  knows *something* is off, but doesn't know it's completely off-schema.
- **The most severe result in the whole repo, on discovery: ordinary
  typos, not domain shift, break the Choice head's calibration.** Phase
  5b (`adversarial_stress_test.py`) applied 2 adjacent-character swaps
  (verified by hand to read as an ordinary fast-typing typo, e.g. "How
  do I locate my card?" -> "Howd o I locate my crad?") to real,
  correctly-classified banking77 test queries -- still squarely inside
  the model's intended domain, same schema, same task. **n=3080** (the
  full BANKING77 test split, not a small sample). Accuracy collapsed
  from 81.9% to 51.2% while mean confidence dropped only from 0.791 to
  0.612. Every other calibration number in this repo (Phase 3, Phase 4)
  is measured on clean test text and would completely miss this. **Two
  separate claims here, not one** -- see the tokenizer-audit caveat in
  §5: the calibration GAP (confidence not tracking the accuracy drop)
  is treated as the tokenizer-independent finding; the exact MAGNITUDE
  of the 81.9%->51.2% collapse is not, since it runs partly through a
  small, from-scratch BPE vocabulary's specific fragmentation behavior
  on this input. **Status as of Workstream D: confirmed to generalize**
  to a second, structurally different dataset (CLINC150) at a similar
  relative magnitude -- not a one-dataset artifact. **Status as of
  Workstream B: partially, not fully, mitigated** -- training on
  typo-augmented data narrowed BANKING77's accuracy-collapse gap by
  roughly half (31.4pp -> 16.0pp drop) and the calibration mismatch
  similarly, with no clean-accuracy cost, but did not close either gap.
  The improvement also carries over to typo types the model never
  trained on (keyboard-neighbor substitutions, deletions, insertions;
  +11 to +15 points accuracy, n=3080; see §5).
  See §5 for the full discussion, including why this finding (its
  existence, generalization, and partial-mitigation result together),
  not the OOD result above, is this repo's sharpest finding as of the
  most recent work.

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
  how much a genuinely different split/init changes the outcome.
- **Choice and Score now have seed variance too, but only over training
  seeds.** `multiseed_sweep.py` retrains the multitask model with seeds
  42, 123 and 2024 on the SAME data split (so it measures init and
  batch-order noise, not split noise, and is a narrower measure than the
  Noul sweep above). With mean pooling: Choice test accuracy 0.8195 ±
  0.0011 (n=3080), calibrated Choice ECE 0.0341 ± 0.0042, Score MAE
  0.1858 ± 0.0055 and Score Pearson r 0.5605 ± 0.0234 (n=1517). Choice
  accuracy is very stable across seeds. Score r is not: it ranges
  0.534-0.579, and the single-run r=0.53 quoted elsewhere in this repo
  is the lowest of the three, so it slightly understates the typical
  result.
- **Calibration numbers otherwise still come from ONE held-out split per
  dataset**, not cross-validation or bootstrapping within a single split.
  The Noul seed sweep measures cross-run variance with different splits
  entirely; the Choice/Score sweep varies only the training seed. Neither
  resamples the multitask test split, so split-to-split variance for
  Choice and Score is still unmeasured.
- **The Score task's numbers below reflect a single run on the new
  (Amazon Fine Food Reviews) dataset** unless stated otherwise. The
  training-seed sweep above puts that single run's r=0.53 at the low end
  of a 0.53-0.58 range.

## 4. An honest estimate of what % of Jev's value proposition this reproduces

Breaking Jev's claimed value proposition into components, and rating how
much this toy reproduction actually demonstrates:

| Claim | Reproduced here? | Confidence |
|---|---|---|
| Non-autoregressive, single forward pass | **Yes, fully.** Architecturally real: one `encode()` call, typed heads read from it. | High |
| Fixed-schema output ("cannot emit malformed output") | **Yes, fully.** Structural, not learned — heads are fixed-shape linear projections. | High |
| Calibrated confidence that tracks real accuracy | **On clean test data, mostly yes for Noul specifically** (calibrated ECE 0.0039-0.0070 across 3 seeds, §3). **On realistic noisy input, no by default** — Phase 5b found ordinary typos collapse Choice accuracy 81.9%→51.2% while confidence drops only 0.791→0.612, a real miscalibration inside the model's own domain, not just at OOD edges (see §5). Clean-test-set ECE and noisy-input calibration are demonstrably different properties here. **Workstream B found this is substantially, not fully, fixable**: training on 50%-typo-augmented data narrowed the accuracy-collapse gap by roughly half (31.4pp→16.0pp drop) and the calibration mismatch similarly, with no clean-accuracy tradeoff, and the gain carries over to typo types never seen in training (+11-15 points accuracy, ECE roughly halved). But the gap did not close: accuracy under noise stays 13-24 points below clean (see §5). | Medium for clean input, Low-Medium once realistic noise is introduced (partially mitigable with targeted augmentation) |
| Multiple typed decision shapes (Noul/Choice/Score) | **All three now show real, working signal.** Score (Amazon Fine Food Reviews, r=0.53) is weaker than Noul/Choice but genuinely learns and generalizes, unlike the STS-B attempt it replaced (r=0.29, flat training curve). None reach a "production-grade" bar, but none is a dead task either. | Low-Medium |
| Speed/cost advantage over LLMs (40-200x claimed) | **Not independently verified — confirmed unreachable, not just untried** (see `LLM_BENCHMARK.md`). This repo's own model is fast (~1.4ms/example measured), but a real LLM benchmark requires an LLM API and no usable one exists in this sandbox: `api.anthropic.com` is network-reachable but no API credentials are present (`ANTHROPIC_API_KEY`/`CLAUDE_CODE_OAUTH_TOKEN`/`ANTHROPIC_AUTH_TOKEN` all unset), no local LLM server is running, and no LLM SDK is installed. The comparison still uses a documented industry reference figure for LLM latency, not a live measurement. The *shape* of the claim (a small non-autoregressive forward pass beats an LLM API round-trip) is directionally very plausible, but "40-200x" specifically remains Jev's number, not something this repo measured against a real LLM. | Low (weakest-evidenced claim in this repo) |
| Domain breadth / general-purpose typed questions over arbitrary schemas | **No, still the weakest part of this reproduction, though now tested on one more task.** Four narrow, single-purpose tasks total (spam/ham, 77 banking intents, food-review star ratings, and now CLINC150's 151-way intent+oos, `CLINC150.md`), each needing its own dataset, its own tokenizer/encoder in CLINC150's case, and largely its own calibration. Jev's actual product claim is schema generality across arbitrary domains WITHOUT per-domain retraining — nothing here demonstrates that; each new task in this repo required a full retrain, not zero-shot or few-shot adaptation of an existing model. | Very Low |
| A formal, verifiable confidence guarantee (not just observed calibration) | **Implemented and verified (`CONFORMAL.md`), with mixed practical results per head.** Split conformal prediction's coverage guarantee held on held-out test data for all three heads (Noul 90.91%, Choice 98.83%, Score 89.85%, target 90%) — the guarantee itself is real, not just observed-and-hoped-for. But "valid" isn't "useful": Noul refuses to answer (empty set) 8.4% of the time, Choice needs an average set of 7 of 77 classes to guarantee coverage, and Score's intervals average about 2.4-2.6 stars wide (CQR narrows them somewhat but covers 1-star reviews only 52% of the time while 5-star reviews get 97%, so the 90% average hides the group that matters most). A team relying on "cannot hallucinate" as a complete safety story should be asked specifically whether their confidence numbers come with this kind of guarantee, and if so, whether the resulting sets/intervals are actually narrow enough to be useful — this repo shows both can be true or false independently. | Medium (guarantee verified real; usefulness varies sharply by head) |

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

**A second, sharper wedge: the failure mode above needs a domain shift to
show up. This one doesn't.** `adversarial_stress_test.py` (Phase 5b) ran
the Choice head on real, correctly-labeled banking77 queries with 2
adjacent-character swaps -- an ordinary fast-typing typo, not an
adversarial attack, verified by hand to be fully readable ("How do I
locate my card?" -> "Howd o I locate my crad?"). Accuracy collapsed from
81.9% to 51.2% -- nearly halved -- while mean confidence dropped only
from 0.791 to 0.612. That gap (confidence down ~23%, accuracy down ~37
relative percentage points) is a real, quantified miscalibration, and it
happens on input that is squarely INSIDE the model's intended domain,
using the exact same schema, on the exact kind of noise every real text
interface encounters constantly. Every calibration number elsewhere in
this repo (Phase 3's temperature/Platt/isotonic scaling, Phase 4's
sweep) is measured on clean, unperturbed test text and would not have
caught this. If a team ships "cannot hallucinate, and here's our ECE on
held-out test data" as the whole trustworthiness story, this result says
that story is incomplete in a way that matters immediately, not just at
the edges of the domain: ordinary text noise, not adversarial intent or
distribution shift, is enough to break the confidence-accuracy
relationship. A serious evaluation of this category needs a typo/noise
robustness number as a first-class metric alongside ECE, not an
afterthought -- clean-test-set calibration and real-world-noise
calibration are evidently not the same thing, and only one of them is
usually reported.

**Tokenizer audit on this finding, and what it does and doesn't tell
us.** Before treating the 81.9%→51.2% collapse as a general property of
"System-One-style architectures," it's worth being precise about what's
actually happening mechanically, since that changes how far the finding
generalizes.

*What tokenizer does the Choice head use, exactly?* Confirmed directly
from the code, not assumed: the shared byte-level BPE tokenizer
(`data/tokenizer_multitask.json`, 8,000-token vocabulary, trained by
`prepare_multitask.py`'s `ByteLevelBPETokenizer` on this repo's own
training text only). This is NOT a fixed word-level vocabulary with
`<unk>` OOV collapse -- that design existed only in an early version of
`prepare_data.py` and was retired during the Phase 2 encoder upgrade,
well before the Choice/Score work existed. Byte-level BPE has no true
unknown-token collapse: any input string always decomposes to some
sequence of byte-level tokens.

*So what actually happens to the typo'd text?* `adversarial_stress_test.py`
now prints this directly (added as a permanent diagnostic, not a one-off
check): on a random sample, the 2-character swap produces zero `<unk>`
tokens, but it frequently fragments a content word into several smaller,
less-informative subword pieces -- e.g. "available" -> "vaailable"
tokenizes as `['v', 'aa', 'ila', 'ble']` instead of the one clean
`['available']` token the unperturbed text gets. 4 of 5 sampled examples
showed measurable fragmentation (+2 to +5 extra tokens); one showed none,
confirming this is a real but not universal mechanism, not an artifact
of a single cherry-picked example.

*Two claims, and only one of them is the tokenizer-independent finding:*
1. **The calibration gap** -- confidence (0.791→0.612, a ~23% relative
   drop) not tracking the accuracy collapse (81.9%→51.2%, a ~37% relative
   drop) -- is treated as real and likely tokenizer-independent. Whatever
   specific mechanism corrupts the model's internal representation under
   this perturbation, a well-calibrated model's confidence should track
   however much its accuracy actually degrades. It didn't. This is this
   repo's headline finding and is not walked back by the tokenizer audit.
2. **The exact magnitude of the accuracy collapse** is NOT claimed to
   generalize to a production-grade subword-tokenized encoder. This
   repo's tokenizer is small (8,000 tokens) and trained on a narrow,
   single-domain corpus, which plausibly makes it more prone to
   fragmenting near-miss spellings than a pretrained tokenizer (e.g. a
   BERT/DistilBERT WordPiece vocabulary, ~30,000 tokens, trained on a
   huge and diverse corpus) would be, since common near-miss spellings
   are more likely to already exist as recognized subword pieces in a
   much larger, more thoroughly-trained vocabulary. The 30.7-point
   collapse specifically should be read as "this toy implementation's
   number," not "System-One models in general will show a 30.7-point
   collapse under typos."

*The experiment that would actually resolve this* -- swap in a
pretrained, subword-tokenized encoder for the Choice head, retrain, and
re-run the same typo test -- was attempted and could not be run: this
sandbox's network policy blocks both HuggingFace Hub (`huggingface.co`)
and `download.pytorch.org`, confirmed again via direct connection tests
immediately before writing this section (both return a hard connection
rejection, not a timeout or a missing-package error). No pretrained
subword-tokenized encoder of any kind is reachable from this environment.
Per this project's own rule against faking or approximating an
unreachable experiment, no from-scratch substitute was used as a stand-in
for "pretrained encoder" -- the byte-level BPE tokenizer already in this
repo IS a subword tokenizer, but it is not a pretrained one, and
conflating the two would answer a different question than the one that
matters here. **This experiment is reported as genuinely unresolved, not
attempted-and-passing or attempted-and-failing.** Anyone able to run it
in an environment with Hub access would learn something this repo
cannot: whether the calibration gap survives with a production-grade
tokenizer even if the accuracy-collapse magnitude shrinks, which is the
single result that would tell us whether this is a real property of
System-One-style architectures or an artifact of this toy
implementation's tokenizer.

**Does adversarial data augmentation fix the typo-collapse finding?
Mostly yes, on this toy task — a genuinely encouraging, fully-measured
result.** `augment_and_retrain.py` duplicated 50% of each task's training
rows with the exact same 2-adjacent-character-swap perturbation used by
`adversarial_stress_test.py` (Choice: 8,503→12,754 train rows; Noul and
Score augmented identically for consistency, though the typo-collapse
finding itself was only ever measured on Choice), retrained a fresh
encoder from scratch with the same architecture/hyperparameters as the
baseline, and recalibrated. Val/test splits were left completely
untouched — no perturbed example ever appears outside training. The
comparison below re-evaluates the actual pre-fix baseline checkpoint live
in the same run (not hardcoded historical numbers), applying the
identical fixed-seed typo perturbations to both models:

| Metric | Baseline | Augmented | Delta |
|---|---|---|---|
| Clean-test accuracy (Choice, n=3,080) | 0.8188 | 0.8347 | **+0.0159** |
| Clean-test mean confidence | 0.7906 | 0.8133 | +0.0228 |
| Typo-test accuracy | 0.5052 | 0.6744 | **+0.1692** |
| Typo-test mean confidence | 0.6122 | 0.7046 | +0.0925 |
| Accuracy drop under typos | 0.3136 (−31.4pp) | 0.1604 (−16.0pp) | **gap narrowed by ~half** |
| Confidence drop under typos | 0.1784 | 0.1087 | gap narrowed by ~39% |

Both halves of the original finding improve substantially:
1. **The accuracy-collapse gap narrowed by roughly half** — typo-test
   accuracy rose from 50.5% to 67.4%, cutting the accuracy drop under
   typos from 31.4 points to 16.0 points. Augmentation clearly helps the
   model generalize past small character-order perturbations it now
   sees in training.
2. **The calibration gap (confidence not tracking the real accuracy
   drop) also narrowed**, from an 0.31-vs-0.18 mismatch (accuracy drop
   nearly double the confidence drop) to a 0.16-vs-0.11 mismatch — still
   present, but meaningfully closer to tracking.
3. **No clean-accuracy tradeoff was observed — if anything, the
   augmented model is slightly better on clean, unperturbed test data
   too** (+1.6 points accuracy, +2.3 points confidence). This is not the
   classic robustness/accuracy tradeoff pattern; augmentation with a
   fixed typo mechanism at 50% duplication rate looks like a
   close-to-free improvement on this toy task, possibly because the
   duplicated rows also act as a mild data-volume increase (train set
   grew ~50%) independent of the typo perturbation itself — this repo
   does not separately test augmenting with non-typo duplicate data to
   isolate that effect, so some of the clean-accuracy gain may be a
   generic more-data effect rather than typo-specific.

**What this does and doesn't establish.** The gap is narrowed, not
closed — typo-test accuracy (67.4%) is still well below clean-test
accuracy (83.5%), and confidence still doesn't fully track the remaining
accuracy loss. This is a real, partial, measured improvement, not a
solved problem: report it as "augmentation meaningfully helps and costs
nothing on this toy task's clean accuracy," not as "typo miscalibration
is fixed."

**Does the improvement carry over to typo types it never trained on?
Yes.** The augmented model only ever saw 2 adjacent-character swaps, so
its gain could have been narrow memorization of that one corruption.
`heldout_noise_test.py` scores both calibrated models on the same
BANKING77 test queries (n=3080) under four corruptions absent from
training (QWERTY-neighbor substitution, deletion, insertion, and a mix
of all three), generated once with a fixed seed so both models see
identical text:

| Noise (2 edits unless noted) | In training? | Baseline acc / ECE | Augmented acc / ECE | Acc gain |
|---|---|---|---|---|
| none (clean) | - | 0.8188 / 0.037 | 0.8347 / 0.034 | +0.016 |
| adjacent swap | yes | 0.5000 / 0.112 | 0.6718 / 0.032 | +0.172 |
| adjacent swap ×4 | same kind, heavier | 0.3276 / 0.202 | 0.5295 / 0.095 | +0.202 |
| keyboard-neighbor substitution | no | 0.5276 / 0.094 | 0.6558 / 0.035 | +0.128 |
| deletion | no | 0.5503 / 0.092 | 0.6870 / 0.026 | +0.137 |
| insertion | no | 0.5899 / 0.062 | 0.7026 / 0.024 | +0.113 |
| substitution + deletion + insertion | no | 0.4461 / 0.139 | 0.5958 / 0.066 | +0.150 |

(The baseline's swap accuracy here, 0.5000, differs slightly from the
0.5052 above because the swaps are a different random draw.) On every
held-out corruption the augmented model is 11-15 points more accurate
and its ECE falls by roughly half or more, so the fix is not limited to
the exact noise it was trained on. It is still not a cure: accuracy
under held-out noise stays 13-24 points below clean accuracy, the model
remains overconfident under every corruption (mean confidence exceeds
accuracy by 0.9-9.5 points), and none of these synthetic corruptions is
a substitute for testing on real user-typed text.

**Does the typo-miscalibration finding generalize past BANKING77? Yes —
confirmed on a second, structurally different dataset (Workstream D, full
writeup in `CLINC150.md`).** CLINC150 (150 real crowdsourced intents
across many domains, plus a genuine out-of-scope class; CC-BY 3.0,
verified in `DATA_LICENSES.md`) was run through the identical typo
perturbation with its own dedicated encoder, its own dedicated
4,000-token tokenizer trained from scratch on this corpus alone (not the
shared multitask vocabulary), and 151 classes instead of 77. Result:
accuracy 71.80%→45.04% (a ~37% relative collapse) while confidence only
dropped 0.7514→0.5615 (a ~25% relative drop) — the same qualitative
pattern, at a similar relative magnitude, as BANKING77's 81.9%→51.2%
accuracy collapse against a 0.791→0.612 confidence drop. Two different
datasets, different vocabularies, different domains, different class
counts, same result: this meaningfully strengthens the claim that the
typo-miscalibration gap is a real property worth checking for in any
System-One-style architecture, not a quirk of one dataset's tokenizer or
domain.

CLINC150 also tested something BANKING77's schema couldn't: an explicit,
human-labeled out-of-scope class, the "learned 'none of the above'
option" this section recommends above. Used as just one more softmax
class, it recognized true out-of-scope queries only 16.1% of the time
(vs 84.2% accuracy on in-scope queries). An earlier version of this
paragraph blamed class imbalance; that was wrong, because CLINC150's
training set has exactly 100 examples for every class, oos included. The
oos class simply has to cover "everything else" with the same budget as
one narrow intent. The more useful result is that the confidence score
carries the missing signal (`clinc150_oos_threshold.py`): top-class
probability averages 0.82 on in-scope vs 0.44 on out-of-scope test
queries (AUROC 0.88), and rejecting answers below a validation-tuned
threshold raises oos recall to 85.7%, at a cost of in-scope accuracy
84.2%→74.5% (oos precision 46%). So "knows when it doesn't know" is
partly true here, but only as a tunable tradeoff, not for free. See
`CLINC150.md` for the full table and tradeoff curve.

## 6. Item 3 disposition: pretrained subword-tokenized encoder swap (skipped, not approximated)

Requested experiment: swap a pretrained, subword-tokenized encoder
(distilbert-base-uncased, MiniLM, or similar) into the Choice head,
retrain, and re-run `adversarial_stress_test.py`'s typo test against it,
to determine whether the calibration gap found in §5 is a property of
System-One-style architectures generally or an artifact of this repo's
own from-scratch, narrow-vocabulary tokenizer.

**Result: not run. Re-verified, freshly, immediately before writing this
section (2026-09-22T02:12:10Z UTC), that no pretrained encoder is
reachable from this sandbox:**

```
huggingface.co/distilbert-base-uncased/resolve/main/config.json -> connection tunnel rejected (HTTP 403, policy denial)
download.pytorch.org/whl/torch/                                 -> connection tunnel rejected (HTTP 403, policy denial)
```

Both failures are hard connection rejections from the sandbox's egress
proxy (explicit policy denial), not timeouts, DNS failures, or missing
local packages -- the same result found earlier in this project (Phase
2) when first checking pretrained-weights availability, now confirmed
unchanged. No pretrained subword-tokenized encoder of any kind
(HuggingFace Hub, PyTorch Hub, or otherwise) is reachable from this
environment.

Per this project's explicit instruction not to fake or approximate this
experiment, no substitute was run. Specifically NOT done, and not
presented as equivalent to the requested experiment: retraining the
existing from-scratch encoder with a different from-scratch subword
tokenizer (e.g. a WordPiece vocabulary instead of byte-level BPE) would
still lack pretrained representations and would not answer the actual
question, which is whether PRETRAINED subword embeddings -- not merely
subword tokenization, which this repo's BPE tokenizer already has --
close the calibration gap.

**This item is reported as genuinely unresolved, and Workstream D's
CLINC150 task (§5, `CLINC150.md`) does not resolve it either, despite
using its own separate from-scratch tokenizer.** CLINC150 answers a
different, narrower question than this one: whether the calibration gap
survives across two different DATASETS/DOMAINS when both still use a
from-scratch tokenizer (it does — see §5). It does not touch the
pretrained-vs-from-scratch axis at all, since CLINC150's dedicated
4,000-token BPE tokenizer is exactly the kind of "different from-scratch
subword tokenizer" already named above as insufficient to answer this
item's actual question. The two findings are complementary, not
duplicates: one shows the gap generalizes across domains, the other
(still open) is about whether it would shrink or vanish with real
pretrained subword embeddings.

It is not evidence
either for or against the calibration gap being a general
System-One-architecture property; it is an open question this
environment cannot answer. Anyone with HuggingFace Hub access can run
this experiment directly against this repo's `adversarial_stress_test.py`
and `train_multitask.py` by swapping in a pretrained encoder for the
Choice head -- that is the specific, well-defined next step this
reconnaissance effort could not complete.

## 7. Workstream E: attention pooling vs. mean pooling

Lowest-priority workstream. The encoder's mean pooling (averaging
per-token hidden states over non-padded positions) was compared with a
learned attention pool (one trainable query vector scores each token;
the pooled vector is the softmax-weighted sum), available as
`JevCloneEncoder(pooling="attn")`. Everything else (layers, dimensions,
training recipe, data split) is identical to `train_multitask.py`.

A first single-seed run showed attention pooling slightly ahead (Choice
0.8247 vs 0.8188, Score MAE 0.1787 vs 0.1872) and was flagged at the time
as possibly seed noise. `multiseed_sweep.py` settles that by training
both variants with three seeds (42, 123, 2024) on the same data split:

| Test metric | Mean pooling (3 seeds) | Attention pooling (3 seeds) | Paired difference (attn − mean) |
|---|---|---|---|
| Noul accuracy (n=836) | 0.9801 ± 0.0050 | 0.9809 ± 0.0041 | +0.0008 ± 0.0014 |
| Choice accuracy (n=3080) | 0.8195 ± 0.0011 | 0.8202 ± 0.0143 | +0.0008 ± 0.0136 |
| Choice ECE, calibrated | 0.0341 ± 0.0042 | 0.0314 ± 0.0046 | −0.0027 ± 0.0050 |
| Score MAE (lower is better, n=1517) | 0.1858 ± 0.0055 | 0.1809 ± 0.0041 | −0.0049 ± 0.0095 |
| Score Pearson r | 0.5605 ± 0.0234 | 0.5666 ± 0.0199 | +0.0061 ± 0.0190 |

(mean ± sample standard deviation over seeds.)

**Result: no measurable difference.** Every paired difference is smaller
than its own seed-to-seed spread. The single-seed Choice edge did not
hold up: attention pooling won by 0.0058 at seed 42 and 0.0110 at seed
123, but lost by 0.0146 at seed 2024. If anything, attention pooling made
Choice accuracy less stable across seeds (std 0.0143 vs 0.0011). With
only three seeds this cannot rule out a small real effect, but nothing
here supports switching pooling methods. That fits the usual expectation
that pooling choice matters little for short inputs (48-64 tokens) and a
small encoder, and it says nothing about attention pooling at larger
scale.
