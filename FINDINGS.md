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
"wider than the output range"), marked where they appear. A third pass
(2026-09-24) added bootstrap confidence intervals (§3), conformal
coverage under typo noise, and group-conditional CQR. A fourth pass
(2026-09-30) pretrained BEMA's encoder in-house with masked-token
prediction, since no pretrained encoder can be downloaded (§8). A fifth
pass (2026-10-04) added a classical TF-IDF baseline, which beat the
transformer everywhere, then rebuilt BEMA's trunk from hashed n-grams and
tuned and ensembled it for accuracy (§9). A sixth pass (2026-11-12)
found and fixed a temperature-fitting bug that affected every temperature-scaled
number in the repo (temperatures, calibrated ECE and overconfidence), measured how much it mattered and revised §9 (§10),
added paired bootstrap intervals for §9's ensemble-versus-linear comparisons, and repeated §9
on CLINC150 (§11). This document summarizes what an independent,
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
  because 61% of reviews are 5-star. Calibrating separately within
  predicted-rating groups evens coverage across those groups but only
  lifts 1-star coverage to 61%: the model rarely predicts a low rating at
  all, so the gap is in the point predictor, not the calibration.
  Under typo noise (`conformal_typo.py`), Choice's sets react where its
  point confidence did not: calibrated on clean text, coverage stays at
  89-92% while sets nearly double (7 to 13-15 classes); calibrating on
  typo'd text restores the 90% guarantee, and the typo-augmented model
  keeps 96-98% coverage with smaller sets.
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
  is treated as the tokenizer-independent finding (for this BPE +
  transformer trunk; with §9's n-gram trunk the large overconfidence
  under typos does not appear); the exact MAGNITUDE
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
  +11 to +15 points accuracy, n=3080; see §5). **Status as of §9:
  specific to the from-scratch transformer.** A TF-IDF linear model and
  the n-gram-trunk BEMA, trained on the same clean data, lose about 6-10
  points of Choice accuracy under keyboard and mixed typos (vs 31-38 for
  the transformer, n=3080), and they do not show the transformer's large
  overconfidence under mixed typos (+0.14); see §9 for their figures.
  See §5 for the full discussion, including why this finding (its
  existence, generalization, and partial-mitigation result together),
  not the OOD result above, is this repo's sharpest finding about the
  transformer BEMA (§9 shows it largely does not carry over to the
  linear or n-gram models).

## 3. Where the calibration numbers are trustworthy vs. not

*Every temperature-scaled number in sections 1-8 was fitted with
temperature code that did not converge (§10). For the multitask
transformer, the corrected fit moves Choice ECE from 0.0371 to 0.0303,
Noul ECE from 0.0046 to 0.0090, and overconfidence under typos up by
0.012-0.013; the qualitative conclusions below hold, except that Noul's
temperature scaling looks less good than stated, and the first bullet's "held across every run" was never true for the transformer's Choice head: there temperature scaling leaves test ECE worse than raw with either fit (raw 0.0210; old fit 0.0371, fixed fit 0.0303) (§10).*

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
  and the later number is what the code of that date reproduced
  bit-for-bit, every time, seed=42, verified by two independent full
  re-runs. (A 2026-11-12 re-run with the corrected temperature fitter,
  §10, reproduced the accuracy and raw ECE exactly; the calibrated ECE is
  now 0.0048, not 0.0039.) There is
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
  entirely; the Choice/Score sweep varies only the training seed.
- **Test-set sampling uncertainty is now measured** (`bootstrap_ci.py`:
  2000 bootstrap resamples of each fixed test split, existing
  checkpoints, paired for differences). 95% intervals: Noul accuracy
  0.9785 [0.968, 0.988] and calibrated ECE 0.0046 [0.0038, 0.0153]
  (n=836); Choice accuracy 0.8188 [0.806, 0.832] and calibrated ECE
  0.0371 [0.033, 0.051] (n=3080); Score MAE 0.187 [0.176, 0.199] and
  Pearson r 0.534 [0.491, 0.576] (n=1517). Two things follow. The
  differences this repo leans on are well outside the noise (e.g. typo
  augmentation's gain on unseen keyboard typos, +12.8 points [11.2,
  14.5]; its small clean-text gain, +1.6 [0.5, 2.8]; CLINC150's threshold
  cost to in-scope accuracy, −9.7 [−10.5, −8.9]). But single ECE values
  are soft: binned ECE is noisy and biased upward at these sample sizes,
  so the headline "calibrated Noul ECE ≈ 0.005" is better read as "at
  most about 0.015". BANKING77's test split is the dataset's official
  one, so resampling that fixed set is the relevant uncertainty; for the
  randomly split Noul and Score data, a different split would add further
  variation (measured for Noul by the seed sweep above).
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
| Calibrated confidence that tracks real accuracy | **On clean test data, mostly yes for Noul specifically** (calibrated ECE 0.0039-0.0070 across 3 seeds, §3). **On realistic noisy input, no by default for the from-scratch transformer** — Phase 5b found ordinary typos collapse Choice accuracy 81.9%→51.2% while confidence drops only 0.791→0.612, a real miscalibration inside the model's own domain, not just at OOD edges (see §5). Clean-test-set ECE and noisy-input calibration are demonstrably different properties here. **Workstream B found this is substantially, not fully, fixable**: training on 50%-typo-augmented data narrowed the accuracy-collapse gap by roughly half (31.4pp→16.0pp drop) and the calibration mismatch similarly, with no clean-accuracy tradeoff, and the gain carries over to typo types never seen in training (+11-15 points accuracy, ECE roughly halved). But the gap did not close: accuracy under noise stays 13-24 points below clean (see §5). **§9: the typo collapse is tied to the from-scratch transformer/BPE trunk, not the typed heads.** The same typed heads on a hashed n-gram trunk, trained on clean data only, lose 6.5-9.8 points of Choice accuracy under keyboard and mixed typos (transformer: 31-38; 3-seed means, n=3080), and they do not show the transformer's large overconfidence under mixed typos (+0.14); see §9 for their figures. | Medium for clean input; Low-Medium for the transformer under realistic noise (partly fixed by augmentation); Medium with the n-gram trunk (§9) |
| Multiple typed decision shapes (Noul/Choice/Score) | **All three now show real, working signal.** Score (Amazon Fine Food Reviews, r=0.53) is weaker than Noul/Choice but genuinely learns and generalizes, unlike the STS-B attempt it replaced (r=0.29, flat training curve). None reach a "production-grade" bar, but none is a dead task either. | Low-Medium |
| Speed/cost advantage over LLMs (40-200x claimed) | **Not independently verified — confirmed unreachable, not just untried** (see `LLM_BENCHMARK.md`). This repo's own model is fast (~1.4ms/example measured), but a real LLM benchmark requires an LLM API and no usable one exists in this sandbox: `api.anthropic.com` is network-reachable but no API credentials are present (`ANTHROPIC_API_KEY`/`CLAUDE_CODE_OAUTH_TOKEN`/`ANTHROPIC_AUTH_TOKEN` all unset), no local LLM server is running, and no LLM SDK is installed. The comparison still uses a documented industry reference figure for LLM latency, not a live measurement. The *shape* of the claim (a small non-autoregressive forward pass beats an LLM API round-trip) is directionally very plausible, but "40-200x" specifically remains Jev's number, not something this repo measured against a real LLM. | Low (weakest-evidenced claim in this repo) |
| Domain breadth / general-purpose typed questions over arbitrary schemas | **No, still the weakest part of this reproduction, though now tested on one more task.** Four narrow, single-purpose tasks total (spam/ham, 77 banking intents, food-review star ratings, and now CLINC150's 151-way intent+oos, `CLINC150.md`), each needing its own dataset, its own tokenizer/encoder in CLINC150's case, and largely its own calibration. Jev's actual product claim is schema generality across arbitrary domains WITHOUT per-domain retraining — nothing here demonstrates that; each new task in this repo required a full retrain, not zero-shot or few-shot adaptation of an existing model. | Very Low |
| A formal, verifiable confidence guarantee (not just observed calibration) | **Implemented and verified (`CONFORMAL.md`), with mixed practical results per head.** Split conformal prediction's coverage guarantee held on held-out test data for all three heads (Noul 90.91%, Choice 98.83%, Score 89.85%, target 90%) — the guarantee itself is real, not just observed-and-hoped-for. But "valid" isn't "useful": Noul refuses to answer (empty set) 8.4% of the time, Choice needs an average set of 7 of 77 classes to guarantee coverage, and Score's intervals average about 2.4-2.6 stars wide (CQR narrows them somewhat but covers 1-star reviews only 52% of the time while 5-star reviews get 97%, so the 90% average hides the group that matters most; per-group calibration only lifts that to 61%). Under typo noise the Choice sets widen instead of silently failing (coverage 89-92% with clean calibration, back above 90% when calibrated on typo'd text). A team relying on "cannot hallucinate" as a complete safety story should be asked specifically whether their confidence numbers come with this kind of guarantee, and if so, whether the resulting sets/intervals are actually narrow enough to be useful — this repo shows both can be true or false independently. | Medium (guarantee verified real; usefulness varies sharply by head) |

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
  blocked by this sandbox's network policy. What real pretrained weights
  would add is untested here (§6, §8). What was measured: in-house
  masked-token pretraining added +0.0087 Choice accuracy and +0.0286
  Score r (§8). Replacing the from-scratch transformer with a hashed
  n-gram trunk, still with no pretrained weights, added +0.0771 Choice
  accuracy (0.8195 to 0.8966 tuned; +0.0703 untuned) and +0.221 Score r
  (0.5605 to 0.7818) (§9; 3-seed means, Choice n=3080, Score n=1517). So
  the largest measured gain came from replacing the transformer, not
  from pretraining it. Every model here (transformer, n-gram and TF-IDF
  linear) is trained from scratch on this repo's training splits. A
  real Jev competitor would almost certainly start from pretrained
  weights.
- **CPU-only, single environment.** All training ran on CPU; the
  multi-task run (3 tasks x ~266 steps x 12 epochs) took ~30 minutes for
  a 96-dim/3-layer encoder. Anything meaningfully larger, or a genuine
  hyperparameter search for the transformer (rather than the two or
  three configurations actually tried), was out of reach in the time
  available. The much cheaper n-gram BEMA got a small 8-setting,
  validation-only search (seed 42, §9), which chose the tuned
  configuration cited above.
- **No sentence-pair architecture question anymore, since there's no
  sentence pair.** The STS-B-era finding that a naive concat+meanpool
  architecture actively hurts a sentence-pair task (§2, historical) no
  longer applies now that the Score task is single-text star-rating
  regression -- there's no pair to mishandle. §9 has since shown where
  most of the gap between r=0.53 and a "strong" sentiment-regression
  result came from. On the same split, with no pretrained weights, the
  same Score-head design on a hashed n-gram trunk reaches r=0.78
  (0.7818 ± 0.0022, 3 seeds, n=1517), and a TF-IDF Ridge model reaches
  0.7821. In-house pretraining of the transformer (§8) added only
  +0.0286. Part of the gap is input length, not the model: the
  transformer reads only the first 64 tokens of each review, and the
  TF-IDF model cut to that same view reaches r=0.7148 (§9). So the
  r=0.53-0.58 was mainly a limit of the small from-scratch transformer
  trunk, with about 30% of the gap from truncation (0.0673 of 0.2216 in r, against the 3-seed mean 0.5605, n=1517): reaching 0.71 on
  the same input, and 0.78 on the whole review, did not need pretrained
  weights. Whether a real pretrained encoder would do better than 0.78
  is untested.
- **Small held-out test sets (836-3,080 examples per task).** Bootstrap
  intervals (§3) show accuracies are pinned to about ±1-2 points, but
  ECE values are only good to within roughly a factor of 2-3.

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
calibration (temperature/Platt/isotonic scaling, all applied; Platt and
isotonic lowered test ECE wherever applied, and temperature scaling lowered
Noul's, but it made the transformer's Choice ECE worse than raw with either
fit: 0.0210 raw → 0.0371 old fit / 0.0303 fixed fit, §10). A model can be well-calibrated on its
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
   (Qualified by §9: when the BPE + transformer trunk is replaced by
   hashed word/character n-grams, with the same head design and the same
   clean training data, the large gap does not appear: those models lose
   6.5-9.8 points under keyboard and mixed typos (3-seed means, n=3080)
   and do not show the transformer's large overconfidence (+0.14 under
   mixed typos); §9 has their figures. So the large gap holds for this
   BPE + transformer trunk, not across input representations. §9 changes
   tokenizer and encoder together, so it does not separate the two.)
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
implementation's tokenizer. (§9 has since shown it is not a property of
System-One-style typed heads as such: BEMA with the same heads on a
hashed n-gram trunk does not show the transformer's large overconfidence
under typos. Whether a pretrained subword transformer would show it is
still open.)

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
dropped 0.7489→0.5583 (a ~25% relative drop; 0.7514→0.5615 with the old
temperature fit, §10) — the same qualitative
pattern, at a similar relative magnitude, as BANKING77's 81.9%→51.2%
accuracy collapse against a 0.791→0.612 confidence drop. Two different
datasets, different vocabularies, different domains, different class
counts, same result: this meaningfully strengthens the claim that the
typo-miscalibration gap is a real property of this from-scratch
transformer encoder, not a quirk of one dataset's tokenizer or domain.
Both runs use the same `JevCloneEncoder` architecture, so this is
generalization across datasets, not across architectures. §9 shows that
on BANKING77 the same typed heads on a hashed n-gram trunk, and a TF-IDF
linear model, do not show the transformer's large overconfidence under
mixed typos (+0.14); see §9 for their figures. The gap is still worth
checking for in any System-One-style model, but it is not a property of
those models in general. (§11 confirms this on CLINC150 itself, n=5,500: under the same typos the transformer's overconfidence rises by 0.0770 (from +0.0309 clean to +0.1079), while the overconfidence of every linear and n-gram model, single or ensembled, moves by less than 0.011 in either direction. The linear and n-gram models also lose only 7-9.5 points of accuracy, where the transformer loses 27.)

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
probability averages 0.82 on in-scope vs 0.43 on out-of-scope test
queries (AUROC 0.88), and rejecting answers below a validation-tuned
threshold raises oos recall to 86.1%, at a cost of in-scope accuracy
84.2%→74.4% (oos precision 46%; re-run with the fixed temperature fit,
§10). So "knows when it doesn't know" is
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
section (2026-09-22T02:12:10Z UTC; re-checked again 2026-09-24T06:40Z
with the same result), that no pretrained encoder is reachable from this
sandbox:**

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
pretrained subword embeddings. §8 adds the closest experiment possible
here, pretraining BEMA's own encoder on ~1.1M tokens: it improves clean
accuracy slightly but leaves the typo collapse unchanged.

It is not evidence
either for or against the calibration gap being a general
System-One-architecture property. §9 later answered that general
question in this environment: BEMA with the same typed heads on a hashed
n-gram trunk, trained on the same clean data, loses 6.5-9.8 points
under keyboard and mixed typos (transformer: 31-38; 3-seed means,
n=3080) and does not show the transformer's large overconfidence. What
stays open is the narrower question of whether a pretrained subword
transformer would show the gap. Anyone with HuggingFace Hub access can
run this experiment directly against this repo's
`adversarial_stress_test.py` and `train_multitask.py` by swapping in a
pretrained encoder for the Choice head -- that is the specific,
well-defined next step this reconnaissance effort could not complete.

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

## 8. Self-supervised pretraining, done here instead of downloaded

§6's experiment (swap in a downloaded pretrained encoder) is still
blocked, so `self_pretrain.py` does the nearest thing that is possible:
it pretrains BEMA's own encoder with masked-token prediction (BERT-style:
hide 15% of tokens, predict them) on unlabeled text, then fine-tunes the
multitask model with exactly `train_multitask.py`'s recipe.

- **Corpus:** the text of every training split this repo uses (SMS,
  BANKING77, Amazon reviews, CLINC150), labels ignored, no validation or
  test text: 34,581 texts, cut into 42,699 windows of up to 64 tokens,
  1.13M tokens in total. 5% of windows were held out to pick the epoch.
- **Pretraining:** 20 epochs (~90 minutes on 4 CPU cores). Held-out loss
  fell from 6.80 to 5.23 and masked-token accuracy rose from 5.7% to
  21.6%, i.e. the encoder learned real word statistics, but from a tiny
  corpus by pretraining standards.
- **Comparison:** fine-tuned with seeds 42, 123 and 2024 and compared with
  the from-scratch models for the same seeds (fixed data split; Choice
  n=3080, Score n=1517, Noul n=836; typo sets as in `heldout_noise_test.py`).

| Test metric (mean ± std over 3 seeds) | From scratch | Self-pretrained | Paired difference |
|---|---|---|---|
| Choice accuracy (clean) | 0.8195 ± 0.0011 | 0.8281 ± 0.0048 | **+0.0087 ± 0.0047** (better on all 3 seeds) |
| Score Pearson r | 0.5605 ± 0.0234 | 0.5892 ± 0.0103 | **+0.0286 ± 0.0182** (better on all 3 seeds) |
| Noul accuracy | 0.9801 ± 0.0050 | 0.9833 ± 0.0032 | +0.0032 ± 0.0070 (mixed signs) |
| Choice ECE (calibrated, clean) | 0.0341 ± 0.0042 | 0.0342 ± 0.0026 | +0.0001 ± 0.0046 |
| Score MAE | 0.1858 ± 0.0055 | 0.1857 ± 0.0046 | −0.0000 ± 0.0071 |
| Choice accuracy drop, keyboard typos | 0.3088 ± 0.0154 | 0.3126 ± 0.0054 | +0.0038 ± 0.0171 |
| Choice accuracy drop, mixed typos | 0.3827 ± 0.0112 | 0.3851 ± 0.0031 | +0.0024 ± 0.0142 |
| Overconfidence under mixed typos | 0.1395 ± 0.0006 | 0.1389 ± 0.0009 | −0.0006 ± 0.0015 |

**Result: small, consistent gains on clean text; no effect at all on the
typo problem.** Pretraining raised clean Choice accuracy by about 0.9
points and Score correlation by about 0.03, both on every seed, with
calibration on clean text unchanged. But the accuracy collapse under
typos and the overconfidence that comes with it are the same as training
from scratch, to within seed noise, for every corruption tested. For
comparison, training on typo-augmented data (§5) cut the same collapse
by 11-17 points. On this evidence, the typo miscalibration is not caused
by a lack of general language knowledge that pretraining supplies.
Training on corrupted text partly fixes it (§5), and pretraining does
not. Never seeing corrupted text is not enough on its own to cause it,
though: §9's TF-IDF linear model and n-gram-trunk BEMA, trained on the
same clean text, lose only about 6-10 points under the same keyboard and
mixed typos (31-38 here), and they do not show this model's large
overconfidence (+0.14 under mixed typos); §9 has their figures. So the
problem is tied to the from-scratch transformer/BPE trunk, not to clean
training data alone. Augmentation narrows it but does not remove it.

**What this does not settle.** 1.1M tokens of clean, in-domain text is
orders of magnitude less than a production pretrained encoder sees, and a
web-scale corpus contains plenty of real typos. So this does not answer
§6's question about a model like BERT or DistilBERT, which could behave
differently. It does show that pretraining in itself, at the scale
possible here, is not a remedy.

## 9. Accuracy: a linear baseline, an n-gram trunk, and ensembles

*Revised 2026-11-12. The first version of this section used a temperature
fitter that did not converge (§10). Single-model accuracy and all Score and Noul numbers are
unchanged by the fix; ECE, overconfidence and the ensembles' Choice accuracies (the ensembles average calibrated probabilities, so e.g. the mix's clean accuracy moved from 0.9201 to 0.9198) are recomputed with the fixed fitter, and paired bootstrap intervals are
added. Two claims of the first version shrank under the fix: the linear model's and the ensembles' underconfidence under typos (−0.02 to −0.04 then, −0.006 to −0.020 now; mostly an artifact of the fitter), and the ECE cost of ensembling (the 3-seed ensemble's clean ECE is 0.0120 against a mean of 0.0074 for its three members, not about 0.02 against 0.008).*

**The baseline came first, and it was unflattering.** `linear_baseline.py`
fits TF-IDF features (word 1-2 grams plus character 2-5 grams) with a
linear model per task, regularization chosen on validation. On the same
test splits it beat the transformer BEMA on every task, and under typos it
loses 6-9 points with overconfidence near zero (−0.006 to −0.014; the
transformer: +0.11 to +0.15, 3-seed means). So the typo
miscalibration in §5 is a property of the small from-scratch transformer,
not of typed-decision models in general.

**BEMA with an n-gram trunk.** `ngram_model.py` keeps BEMA's design (one
shared state per input, read by fixed-shape Noul / Choice / Score heads,
all three trained jointly on `train_multitask.py`'s splits with its
checkpoint rule), but builds the state from the mean embedding of hashed
word 1-2 grams and character 2-5 grams instead of a transformer. These are
the same n-gram types the baseline uses, but unweighted (no TF-IDF, no
min_df cut, hashing collisions possible). Hyperparameters were chosen by
running `train_ngram.py --val-only --epochs 8` once per setting: 8
settings, seed 42, validation only, scored by the combined validation
score (Noul accuracy + Choice accuracy + 1 − Score MAE):

| width | dropout | embedding lr | combined val score |
|---|---|---|---|
| 96 | 0.15 | 3e-3 | 2.7519 |
| 96 | 0.15 | 1e-3 | 2.7622 |
| 96 | 0.4 | 3e-3 | 2.7672 |
| 96 | 0.4 | 1e-3 | 2.7655 |
| 256 | 0.15 | 3e-3 | 2.7592 |
| 256 | 0.15 | 1e-3 | 2.7653 |
| **256** | **0.4** | **3e-3** | **2.7723** |
| 256 | 0.4 | 1e-3 | 2.7682 |

Test was scored once for the chosen setting and once for the untuned
default (width 96, dropout 0.15, lr 3e-3, 12 epochs). The tuned run was
repeated on 2026-11-12 on a different machine and torch version and
reproduced every accuracy, Score and Noul number to four decimals.

**Ensembles.** `ensemble.py` averages the three tuned seeds' calibrated
probabilities, and mixes that average with the linear baseline. The
mixing weight was picked on Choice validation accuracy from
{0, .25, .5, .75, 1}: 0.9153, 0.9207, **0.9253**, 0.9173, 0.9093, so
w = 0.5. The ensemble temperature was refitted on validation.

Test results. Single models are mean ± std over seeds 42/123/2024;
ensembles and the linear model are single deterministic runs. Choice
n=3080, Score n=1517, Noul n=836.

| Test metric | Transformer BEMA (§8) | Linear TF-IDF | n-gram BEMA, default | n-gram BEMA, tuned | 3-seed n-gram ensemble | n-gram ensemble + linear (w=0.5) |
|---|---|---|---|---|---|---|
| Choice accuracy (clean) | 0.8195 ± 0.0011 | 0.9114 | 0.8898 ± 0.0020 | 0.8966 ± 0.0058 | 0.9153 | **0.9198** |
| Choice ECE (calibrated) | 0.0281 ± 0.0026 | 0.0127 | 0.0141 ± 0.0044 | **0.0074 ± 0.0016** | 0.0120 | 0.0116 |
| Choice accuracy, keyboard typos | 0.5107 ± 0.0147 | 0.8471 | 0.8156 ± 0.0081 | 0.8315 ± 0.0089 | 0.8594 | **0.8679** |
| Choice accuracy, mixed typos | 0.4368 ± 0.0102 | 0.8253 | 0.7923 ± 0.0111 | 0.8075 ± 0.0074 | 0.8419 | **0.8506** |
| Drop under keyboard / mixed typos | 0.309 / 0.383 | 0.064 / 0.086 | 0.074 / 0.098 | 0.065 / 0.089 | 0.056 / 0.073 | 0.052 / 0.069 |
| Overconfidence, mixed typos | +0.1507 ± 0.0024 | −0.0142 | +0.0106 ± 0.0077 | +0.0139 ± 0.0059 | −0.0115 | −0.0201 |
| Score MAE | 0.1858 ± 0.0055 | 0.1503 | 0.1298 ± 0.0015 | 0.1244 ± 0.0016 | **0.1232** | 0.1351 |
| Score Pearson r | 0.5605 ± 0.0234 | 0.7821 | 0.7776 ± 0.0040 | 0.7818 ± 0.0022 | 0.7896 | **0.8012** |
| Noul accuracy | 0.9801 ± 0.0050 | 0.9880 | 0.9852 ± 0.0030 | 0.9852 ± 0.0042 | 0.9868 | **0.9904** |

The transformer column is §8's from-scratch models (same three seeds)
re-evaluated with the fixed fitter by `transformer_recheck.py`, retrained
on 2026-11-12. Its accuracies, Score and Noul numbers reproduce §8
exactly; with the fixed fit its ECE is 0.0281 (old fit 0.0341) and its
overconfidence under mixed typos +0.1507 (old fit +0.1395).

Paired bootstrap differences from the linear baseline (`ensemble.py`;
B=2000 resamples of the test examples, 95% percentile intervals; positive
means better than linear; for Score, linear MAE minus the model's MAE):

| | 3-seed n-gram ensemble − linear | ensemble + linear mix − linear |
|---|---|---|
| Choice accuracy, clean (n=3080) | +0.0039 [−0.0036, +0.0110] | **+0.0084 [+0.0032, +0.0136]** |
| Choice accuracy, swap typos | **+0.0237 [+0.0133, +0.0344]** | **+0.0256 [+0.0185, +0.0331]** |
| Choice accuracy, keyboard typos | **+0.0123 [+0.0023, +0.0221]** | **+0.0208 [+0.0133, +0.0279]** |
| Choice accuracy, mixed typos | **+0.0166 [+0.0055, +0.0282]** | **+0.0253 [+0.0169, +0.0334]** |
| Score, MAE reduction (n=1517) | **+0.0272 [+0.0220, +0.0325]** | **+0.0152 [+0.0126, +0.0179]** |
| Noul accuracy (n=836) | −0.0012 [−0.0096, +0.0072] | +0.0024 [−0.0036, +0.0084] |

What this shows, and what it does not:

- **Choice accuracy went from 81.95% (transformer) to 91.98%.** The trunk
  swap gives most of it (+7.0 points untuned, +7.7 tuned); seed
  ensembling adds 1.9 more; mixing with the linear model another 0.45.
  A single tuned n-gram model (89.66%) is still 1.5 points below the
  linear baseline. The 3-seed ensemble's 0.39-point lead over linear is
  within noise (interval spans zero). The mix's lead, 0.84 points (26 of
  3080 queries), is small but its interval excludes zero.
- **Typo robustness:** every n-gram variant loses 5-10 points under
  keyboard and mixed typos (single models 6.5-9.8, ensembles 5.2-7.3),
  against 31-38 for the transformer, and both ensembles are significantly more accurate than the linear model under every typo type (the paired intervals are for accuracy under typos; the size of the drop itself was not bootstrapped). Mean
  overconfidence under typos is at most +0.014 for the single n-gram
  models (worst single seed +0.018, tuned, mixed typos), against +0.15
  for the transformer. The linear model and the ensembles are slightly
  *under*confident under typos (−0.006 to −0.020).
- **Calibration:** with a converged temperature, clean ECE is 0.0074-0.0141
  for every linear and n-gram model here (single models: 3-seed means);
  the transformer is far worse (0.0281 ± 0.0026 with the fixed fit;
  0.0341 with the old one). Among the others the tuned single model is
  lowest (0.0074) and the untuned one highest (0.0141); the ensembles
  (0.0120, 0.0116) and the linear model (0.0127) differ by at most
  0.0011, no interval was computed for that gap, and it is too small to
  rank them given how soft single ECE values are (§3).
- **Score and Noul:** the mixing weight was chosen on Choice validation
  only and reused for Score and Noul. It gives the best Score correlation
  (0.80) and Noul accuracy, but its Score MAE (0.135) is worse than every
  n-gram model without the mix (0.123-0.130), because the linear Ridge
  model's MAE is worse. Both ensembles beat the linear baseline on Score
  MAE; neither Noul difference from linear is distinguishable from zero
  (0.9904 vs 0.9880 is 2 of 836 messages).
- **Cost:** the 3-seed ensemble is three forward passes, and the mix adds
  a TF-IDF transform and a linear model. It is still non-autoregressive
  with a fixed output shape, but it is not "one forward pass".
- **Unequal input length (a confound for Score):** the transformer reads
  only the first 64 BPE tokens of each input (`prepare_multitask.py`
  MAX_LEN=64); the TF-IDF and n-gram models read the whole text.
  For Score this matters: 57.9% of test reviews (n=1517) are
  longer than 64 tokens. `truncation_control.py` refits the linear
  baseline on exactly the transformer's 64-token view: Score r falls
  from 0.7821 to 0.7148 and MAE rises from 0.1503 to 0.1662. So about
  0.07 of the 0.22 gap in r between the linear model and the transformer
  comes from reading more text, and about 0.15 (0.7148 vs 0.5605) remains
  on equal input. For Choice (0.1% of test inputs cut) and Noul (3.6%)
  it makes no material difference: 0.9114 vs 0.9107 and 0.9880 vs
  0.9880. The n-gram BEMA was not re-run on truncated text, so its Score
  numbers carry the same advantage.
- **For the Jev comparison:** the accuracy gain comes from swapping the
  learned-from-scratch transformer for n-gram features, i.e. from a
  better input representation (for Score, partly also from reading the
  whole review, as above), not from anything specific to the typed-head
  design. Jev's claims should be judged against strong
  classical baselines like this one, not against a weak neural one.


## 10. Correction: the temperature fitter did not converge

**What was wrong.** Every temperature in this repo was fitted with the
same few lines, copied thirteen times into eight scripts and also used
by `calib_utils.fit_temperature` (which six more scripts call): LBFGS on T, starting at T=1.5, with a fixed step of 0.01 and no line
search. That moves T only a small fraction of the way to the
cross-entropy minimum per iteration, and when the best T is below 1 the
loss is concave in T near the start, where this update is unreliable: it can stop well short of the minimum or run through T=0 to a negative temperature. On
synthetic logits with a known best temperature, it returned 0.725 for a
true 0.4, −41 for a true 0.8, 2.46 for a true 3.0, and was close only
near 1-1.6 (1.012 for 1.0, 1.207 for 1.2). The fit was found by an
independent code audit on 2026-11-12 (two reviewers per finding) and
confirmed with that synthetic check. `calib_utils` now fits 1/T, in
which the loss is convex, with a strong-Wolfe line search
(`fit_temperature`, plus `fit_temperature_binary` for the Noul head),
unit-tested against a fine grid minimum for best temperatures 0.4, 1.0 and 3.0 (0.4 and 3.0 for the binary fit).
Every script now calls these helpers.

**What it affected.** Temperatures, calibrated ECE and overconfidence,
everywhere: Noul (`calibrate.py`, `calibrate_multitask.py`,
`seed_variation_sweep.py`, `conformal.py`), Choice
(`calibrate_multitask.py`, `multiseed_sweep.py`, `self_pretrain.py`,
`conformal.py`, `conformal_typo.py`, `augment_and_retrain.py`), Phase 4
(`calibration_sweep.py`), CLINC150 (`clinc150_pipeline.py`) and §9; `augment_and_retrain.py` fits Noul too. Numbers computed from the temperatures these scripts store in the calibrated checkpoints inherit the old fit as well: `ood_stress_test.py` (Phase 5), `adversarial_stress_test.py` (Phase 5b), `heldout_noise_test.py`, `bootstrap_ci.py` and `clinc150_oos_threshold.py`.
Accuracy, Score and argmax predictions do not depend on the temperature, except inside §9's ensembles, which average calibrated probabilities, and in CLINC150's out-of-scope rule (`clinc150_oos_threshold.py`), which thresholds the temperature-scaled top probability. That rule, and `clinc150_pipeline.py`, were re-run with the fixed fit (CLINC150.md, §5 and §11 now show the re-run); only §3's bootstrap interval for the rule's in-scope cost (−9.7 [−10.5, −8.9]; the point estimate is now −9.8) was not recomputed.
Split-conformal coverage is guaranteed on exchangeable data for any score function fixed before calibration, and `conformal.py` fits the temperature on val_a and the conformal quantile on val_b, so the clean-test guarantee (at least 90%) holds with either fitter. The measured Choice figures in CONFORMAL.md (APS coverage 98.83%, the set sizes, and all coverage under typos, where exchangeability fails and no guarantee applies) depend on the temperature and were not recomputed. Noul's threshold sets are unaffected (the temperature rescales a single logit, a monotone change to the score), and the Score head has no temperature.

**How much it mattered** (`temperature_audit.py`: old and fixed fits on
the same validation logits, test n=3080 for Choice, 836 for Noul). The
transformer is `train_multitask.py` retrained here with seed 42; it
reproduces the published temperatures and ECEs exactly (Choice T=1.18,
ECE 0.0371; Noul T=2.36, ECE 0.0046; README Phase 4), so its "old"
column is the published one.

| Model, head | T, old → fixed | Clean ECE, old → fixed | Overconfidence under mixed typos, old → fixed |
|---|---|---|---|
| Transformer, Choice | 1.181 → 1.143 | 0.0371 → 0.0303 | +0.1391 → **+0.1519** |
| Transformer, Noul | 2.363 → 2.610 | 0.0046 → **0.0090** | — |
| Linear TF-IDF, Choice | 0.794 → 0.720 | 0.0159 → 0.0127 | −0.0421 → −0.0142 |
| n-gram BEMA tuned, Choice, seed 42 | 1.553 → 1.561 | 0.0073 → 0.0083 | +0.0085 → +0.0071 |
| n-gram BEMA tuned, Choice, seed 123 | 1.519 → 1.524 | 0.0085 → 0.0083 | +0.0169 → +0.0162 |
| n-gram BEMA tuned, Choice, seed 2024 | 1.705 → 1.741 | 0.0084 → 0.0056 | +0.0233 → +0.0183 |

Validation NLL fell or stayed equal in every row, as it must.

What changes in the conclusions:

- **The transformer's typo miscalibration (§5, §8) stands, and is
  slightly larger.** With the fixed fit its overconfidence under typos
  rises by 0.012-0.013 (seed 42: swap +0.112 → +0.124, keyboard +0.094 →
  +0.106, mixed +0.139 → +0.152). Over all three from-scratch seeds
  (`transformer_recheck.py`, test n=3080): clean ECE 0.0341 ± 0.0042 →
  0.0281 ± 0.0026, mixed-typo overconfidence +0.1395 ± 0.0006 → +0.1507
  ± 0.0024, with accuracies unchanged. The +0.14 figures quoted in §2-§8
  are mixed-typo figures from the old fit and understate it by about
  0.011; they were left as the record. §8's self-pretrained column was
  not re-run (`self_pretrain.py` takes about 2.5 hours).
- **"Temperature scaling made Choice ECE worse than raw" (README, Phase
  4) stands:** the fixed fit gives 0.0303, still worse than raw 0.0210.
- **The published Noul ECE was flattered by the bug.** The
  cross-entropy-optimal temperature gives ECE 0.0090, not 0.0046; still
  better than raw (0.0159), but Platt scaling (0.0030) and isotonic
  regression (0.0049) now beat temperature scaling on this test split (no interval was computed for these differences). With 836 test
  messages and about 18 errors, Noul ECE is noisy either way. The 3-seed
  Noul figures in §3 (calibrated ECE 0.0039-0.0070) used the same old
  fit and were not re-run.
- **§9's calibration claims changed the most**, because the linear model (T=0.720)
  and the ensemble mix (T=0.860; the 3-seed ensemble's own refitted temperature is not logged) have best temperatures below 1, where the old fit
  failed worst. The first version of §9 reported them slightly
  underconfident under typos (−0.02 to −0.04); with a converged
  temperature that is −0.006 to −0.020. §9 has been revised.
- **The tuned n-gram models barely move**: their best temperatures (1.52-1.74)
  are where the old fit happened to start. The untuned default n-gram models
  were not in this audit and their temperatures were not logged. Retrained
  with the fixed fit, they give the same accuracies as before, but their
  calibration moved more. Means over 3 seeds, Choice test n=3080: mixed-typo
  overconfidence went from +0.0039 to +0.0106, and clean ECE from 0.0158 to
  0.0141 (§9).
- **CLINC150's transformer was re-run** with the fixed fit
  (`clinc150_pipeline.py`, `clinc150_oos_threshold.py`; test n=5,500).
  Accuracy reproduced exactly (0.7180). The temperature moved 1.5739 →
  1.5860, scaled ECE 0.0412 → 0.0407, typo-test confidence 0.5615 →
  0.5583, and the threshold rule's overall accuracy 0.7653 → 0.7655 (tau
  unchanged at 0.67); conformal coverage is unchanged (0.9827), set size
  17.31 → 17.27. CLINC150.md shows the re-run.

## 11. Does §9 replicate on CLINC150?

`clinc150_classical.py` repeats §9's comparison on CLINC150 (151 classes:
150 intents plus out-of-scope, "oos"), with `clinc150_pipeline.py`'s
splits and its typo perturbation (two adjacent-character swaps, same
seed), so every typo number below pairs with the transformer's in
CLINC150.md. The n-gram configuration is the one tuned on BANKING77,
reused without re-tuning; the linear model's C (50) and the mixing weight
(w = 0.25, from {0, .25, .5, .75, 1}: val accuracy 0.9119, **0.9148**,
0.9119, 0.9106, 0.9055) were chosen on validation, and every model's
temperature was fitted on validation with the fixed fitter. "oos
threshold" is `clinc150_oos_threshold.py`'s rule: answer oos when the top
probability is below tau, with tau chosen on validation.

Test n=5,500 (4,500 in-scope, 1,000 oos). Single n-gram models are mean ±
std over seeds 42/123/2024.

| Test metric | Transformer (CLINC150.md) | Linear TF-IDF | n-gram BEMA, single | 3-seed n-gram ensemble | ensemble + linear (w=0.25) |
|---|---|---|---|---|---|
| Overall accuracy (argmax) | 0.7180 | 0.7949 | 0.7885 ± 0.0109 | 0.8098 | **0.8118** |
| In-scope accuracy (n=4,500) | 0.8418 | 0.9231 | 0.9116 ± 0.0040 | 0.9267 | **0.9302** |
| oos recall, argmax (n=1,000) | 0.1610 | 0.2180 | 0.2350 ± 0.0720 | **0.2840** | 0.2790 |
| oos AUROC of 1 − top probability | 0.8802 | **0.9365** | 0.9068 ± 0.0124 | 0.9246 | 0.9306 |
| Overall accuracy with oos threshold | 0.7655 (tau 0.67) | **0.8645** (tau 0.76) | 0.8158 ± 0.0028 | 0.8571 (tau 0.78) | 0.8596 (tau 0.81) |
| Calibrated ECE (clean) | 0.0407 | 0.0442 | 0.0447 ± 0.0035 | **0.0319** | 0.0343 |
| Overconfidence, clean | +0.0309 | +0.0384 | +0.0444 ± 0.0037 | **+0.0319** | +0.0343 |
| Accuracy with typos | 0.4504 | 0.7004 | 0.7018 ± 0.0097 | 0.7367 | **0.7405** |
| Accuracy drop under typos | 0.2676 | 0.0945 | 0.0867 ± 0.0016 | 0.0731 | **0.0713** |
| Overconfidence, typos | +0.1079 | +0.0308 | +0.0522 ± 0.0055 | +0.0264 | **+0.0245** |

The transformer column is `clinc150_pipeline.py` and
`clinc150_oos_threshold.py` re-run on 2026-11-12 with the fixed
temperature fit (T=1.5860; the old fit gave 1.5739). Its accuracy
reproduces CLINC150.md exactly; its temperature-dependent numbers moved by
at most 0.003 (§10).

Paired bootstrap of overall argmax accuracy against the linear model
(B=2000, 95% intervals, n=5,500): 3-seed ensemble +0.0149 [+0.0084,
+0.0213] clean and +0.0364 [+0.0278, +0.0449] with typos; ensemble +
linear +0.0169 [+0.0109, +0.0229] clean and +0.0402 [+0.0324, +0.0480]
with typos.

What replicates, and what does not:

- **The main §9 result replicates.** The linear model and the n-gram
  BEMA both beat the transformer by large margins (overall accuracy
  0.79 vs 0.72; in-scope 0.91-0.92 vs 0.84), and under typos they lose
  7-9.5 points where the transformer loses 27. On average a single n-gram model
  is again slightly below the linear model (mean 0.7885 vs 0.7949, n=3
  seeds), but not every seed is: seed 42 alone reaches 0.7995, and the
  seed std (0.0109) is larger than the gap (0.0064). Both ensembles beat
  the linear model with intervals that exclude zero, clean and with typos
  (on BANKING77 the 3-seed ensemble's clean interval spanned zero). Here the gain is
  larger than on BANKING77 (+1.5 to +1.7 points clean, +3.6 to +4.0
  with typos).
- **Out-of-scope detection is better too.** With the validation-chosen
  threshold the linear model reaches 0.8645 overall (in-scope 0.8598,
  oos recall 0.886, oos precision 0.627) against the transformer's
  0.7655 (0.7442 / 0.861 / 0.461). The linear model's confidence
  separates oos best (AUROC 0.937); single n-gram models are weaker here
  (0.907), and their threshold results trail the linear model's by about
  5 points overall (0.8158 vs 0.8645).
- **Calibration does not replicate cleanly.** Every model, the
  transformer included, is overconfident on CLINC150's clean test set by
  +0.03 to +0.05. The likely cause is the split design rather than the
  models: validation is 3% oos (100 of 3,100) while test is 18% (1,000 of
  5,500), and oos queries are where every model is wrong most often (oos
  recall 0.16-0.31 against in-scope accuracy 0.84-0.93), so a temperature
  fitted on validation is too low for test (CLINC150.md discusses the
  same val/test gap for accuracy). Under typos the
  transformer's overconfidence rises by +0.077 over clean; the linear
  model's falls by 0.008, the ensembles' by 0.006-0.010, and single
  n-gram models' rises by +0.008.
