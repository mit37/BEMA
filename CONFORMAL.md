# Conformal prediction: results and honest assessment

Date: 2026-09-22. Implements Workstream A: split conformal prediction for
all three typed heads (Noul, Choice, Score), verified empirically on
held-out test data the calibration procedure never touched. See
`conformal.py` for the full implementation.

## Why conformal prediction, on top of what's already here

Temperature/Platt/isotonic scaling (already in this repo, see FINDINGS.md
§1/§3) calibrate confidence *on average* across a test set — a well-
calibrated model's stated 90% confidence should, in aggregate, be right
about 90% of the time. That's a real property, but it says nothing about
any *individual* prediction's reliability, and it's an empirical property
observed on one split, not a mathematical guarantee.

Split conformal prediction is different in kind: given only an
exchangeability assumption between the calibration set and the test set
(no assumption about the model, the data distribution, or whether the
underlying probabilities are well-calibrated at all), it produces
prediction SETS (not point estimates) with a *distribution-free,
finite-sample coverage guarantee*: at target 1-alpha, the true label is
in the returned set with probability >= 1-alpha, by construction. This
section verifies that guarantee actually holds here, the same way
FINDINGS.md verifies ECE rather than assuming it.

## Statistical hygiene: the calibration split

`calibrate_multitask.py` already fit `temperature` (Noul) and
`choice_temperature` (Choice) on the FULL validation split. Reusing that
same val split for conformal calibration would be a real, if subtle,
leakage: the nonconformity scores would depend on a temperature that had
already "seen" that exact data. To avoid this, `conformal.py` splits val
50/50 (fixed seed) into `val_a`/`val_b`, **refits temperature and
choice_temperature from scratch on val_a only**, and computes conformal
nonconformity scores and quantiles on val_b only — fully disjoint from
both the temperature fit and the final test set. (The Score head has no
such calibration-layer parameter, so this split matters less there, but
the same val_a/val_b split is used for consistency.) The temperature
values from this refit (T=2.46 for Noul, T=1.19 for Choice) are close to
but not identical to `calibrate_multitask.py`'s full-val fit (T=2.36,
T=1.18) — expected, since they're fit on half the data with a different
(fixed) seed; both are legitimate estimates of the same quantity.

## Results

Target coverage: 90% (alpha=0.10) for all three heads.

| Task | n (calib / test) | Method | Target | Measured coverage | Set size / interval width |
|---|---|---|---|---|---|
| Noul | 418 / 836 | Threshold (LAC) conformal | 90% | **90.91%** | avg 0.92 of 2 classes (see note below) |
| Choice | 750 / 3080 | APS (non-randomized) | 90% | **98.83%** | avg 7.07 of 77 classes (median 6) |
| Score | 759 / 1517 | Absolute-residual conformal | 90% | **89.85%** | raw width 1.066; mean 0.656 after clipping to the [0,1] label range |
| Score | 759 / 1517 | CQR (`score_cqr.py`) | 90% | **90.11%** | mean 0.608, median 0.570, clipped to [0,1] |

**The coverage guarantee held in all three cases** — measured coverage
was at or above target in every case (89.85% for Score is a hair under
90% due to finite-sample noise at n=1517, well within the expected
fluctuation band for this sample size; the other two are comfortably at
or above target). That's the headline result: conformal prediction's
core promise is real and verified here, not just asserted.

**But "valid" and "useful" are different questions, and the answer is
different per head:**

### Noul: tight sets, but a notable empty-set rate

Average prediction set size 0.92 (not 1.0) is not a typo — it reflects
that this repo's binary "least ambiguous set" (threshold/LAC) conformal
method can return an EMPTY set when neither class's probability clears
the calibrated threshold. That happened for **70 of 836 test examples
(8.37%)**. Breaking down the numbers: 91.63% of test examples get a
confident singleton set ({spam} or {ham}), 0% get the uninformative
both-classes set, and 8.37% get nothing. Cross-checking: the overall
miss rate is 1 - 0.9091 = 9.09% (76 of 836 examples). 70 of those 76
misses are empty sets; the remaining 6 are wrong singleton predictions
(6 of 766 singleton predictions, i.e. 0.72 percentage points of the
whole test set, or 99.22% accuracy specifically among the singleton
predictions) — meaning **when this method commits to a single class, it
is right 99.2% of the time**, and essentially all of its formal
"coverage failures" are honest refusals (empty sets) rather than
confidently wrong singleton predictions. Read charitably,
this is a good property: the method is telling you outright when it
can't confidently place an example in either class, rather than forcing
a guess. Read strictly, an empty set is not a valid "answer" in Jev's
typed-decision sense (a typed API caller still needs a schema-valid
Noul, and "neither" isn't one) — a real product built on this would need
an explicit policy for what to do with the ~8% of inputs the conformal
layer refuses to place.

### Choice: valid but noticeably conservative (over-covering)

98.83% measured coverage against a 90% target is not a bug — the
non-randomized APS method used here is known to be conservative by
construction (it rounds up to the next class boundary rather than
interpolating exactly to the target quantile, which the randomized
variant of APS would do). The guarantee (coverage >= 90%) held with room
to spare, at a real cost: average set size 7.07 out of 77 classes. That
is genuinely informative — a random-guess baseline would need a set of
size ~70 to hit 90% coverage with a poorly-discriminating model, and
7.07 shows the model's probability estimates carry real signal — but it
is also not "tight": half of test examples need a set of 6 or more
categories to reach the guarantee, and the largest set seen was 35 of
77 categories. A caller expecting a short, actionable shortlist most of
the time would find this useful roughly half the time (49.4% of
examples get a set of 5 or fewer) and unwieldy the rest.

### Score: valid, wide, and uneven across rating levels

89.85% coverage against the 90% target, essentially exact. The raw
interval is ±0.533 on the [0,1] rating scale (width 1.066).

**Correction to an earlier version of this document.** It called that
width "wider than the entire possible range of the label" and the result
"practically useless". The first part is true of the raw interval, but
the label can only lie in [0,1], so clipping the interval to [0,1] loses
no coverage. After clipping, the mean width is 0.656 (median 0.589),
about 2.6 of the 4 steps on the 1-5 star scale. That is wide, and no test
interval is narrower than 2 stars, but it is not literally uninformative.

Why so wide: absolute-residual conformal gives every input the same
width, sized to the 90th percentile of calibration-set error. The Score
head's error is heavily right-skewed (median absolute error 0.090, 90th
percentile 0.538, 95th 0.724) because the test set is mostly 5-star
reviews (932 of 1517, 61%), which the model predicts well, plus a
minority of lower ratings it predicts poorly. One width has to cover
that hard tail, so it is too wide for the easy majority.

**Conformalized quantile regression (CQR)** is the standard fix and is
now implemented (`score_cqr.py`). Two quantile heads (5th and 95th
percentile, pinball loss) are fit on the frozen encoder's pooled
features using the training split, the epoch is picked on val_a, the
intervals are conformalized on val_b, and the result is measured once on
the test set:

| Method (test n=1517, clipped to [0,1]) | Coverage | Mean width | Median width | Intervals under 2 stars (<0.5) |
|---|---|---|---|---|
| Absolute residual | 0.8985 | 0.656 | 0.589 | 0% |
| CQR | 0.9011 | 0.608 | 0.570 | 41% |

CQR keeps the guarantee, is modestly narrower on average, and adapts:
41% of its intervals span less than 2 stars. But its coverage by true
rating shows what "90% coverage" does and does not promise:

| True rating | n | CQR coverage | CQR mean width |
|---|---|---|---|
| 1 star | 162 | 52.5% | 0.824 |
| 2 stars | 82 | 78.0% | 0.830 |
| 3 stars | 119 | 86.6% | 0.788 |
| 4 stars | 222 | 95.0% | 0.642 |
| 5 stars | 932 | 97.0% | 0.521 |

The guarantee is *marginal*: 90% averaged over the test distribution.
Because 61% of reviews are 5-star and covered 97% of the time, the
average reaches 90% even though a 1-star review's true rating falls
inside its interval only about half the time. Split conformal methods
cannot guarantee coverage conditional on the true label (it is unknown
at prediction time). A deployment that cares most about negative
reviews would need to measure coverage on that group directly. An
average coverage number alone would hide the problem.

**Group-conditional CQR** (also in `score_cqr.py`) conformalizes
separately within three predicted-rating groups (equal-count cut points
from val_a). It evens coverage across those groups (lowest-predicted
group 85.4% → 89.6%; all three ≈ 90%), keeps 90.2% overall, and narrows
intervals slightly (mean width 0.590). By true rating it barely helps:
1-star coverage 52.5% → 60.5%, 2-star unchanged at 78.0%. The lowest
predicted third already starts at 0.87 on the [0,1] scale (about 4.5
stars): the model almost never predicts a low rating, so no grouping by
prediction can single out the reviews it gets wrong. That gap has to be
fixed in the point predictor (more low-rating data, a stronger
encoder), not in the calibration layer.

## What this adds over temperature/Platt/isotonic scaling, honestly

- **A real, verified guarantee**, not just an observed average property:
  all three heads hit their coverage target on held-out data. This is
  a genuine methodological upgrade over ECE alone.
- **It does not fix the underlying model.** Conformal prediction
  calibrates the SET, not the point estimate — a weak Score model still
  produces wide intervals (CQR narrows them somewhat but leaves 1-star
  reviews covered only about half the time); a Choice model with real but
  imperfect discriminative power still needs a set of ~7 classes on
  average to guarantee coverage. Conformal prediction makes weakness
  visible and quantified rather than hidden behind a single miscalibrated
  point estimate — which is valuable — but it is not a substitute for a
  better point predictor.
- **It directly contradicts an unqualified "cannot hallucinate + typed
  output" pitch as a complete trustworthiness story.** A typed API that
  must always return a SINGLE typed answer (as Jev's basic interface
  does) cannot, by definition, return "I need to say {6 possible
  categories}" or "somewhere between 2 and 5 stars" — it has to pick one
  value. Conformal prediction's honest answer, when forced into a
  single-point interface, is either the point estimate alone (silently
  discarding the uncertainty this section just measured) or a visibly
  unwieldy set or interval exposed to the caller. Neither resolves the
  tension; this section's job was to make the tension visible and
  measured, which it does.

## Conformal sets under typo noise (and Workstream B)

This repo's sharpest finding (FINDINGS.md §5) is that ordinary typos
collapse Choice accuracy while the point confidence barely drops.
`conformal_typo.py` asks the same question of the conformal sets. Split
conformal's guarantee assumes calibration and test data are exchangeable,
which calibrating on clean text and serving typo'd text breaks. Same
protocol as above (temperature on val_a, APS on val_b, test n=3080);
the baseline row on clean text reproduces the 98.83% / 7.07 result.

| Model | Calibrated on | Test text | Coverage | Avg set size |
|---|---|---|---|---|
| baseline | clean | clean | 0.9883 | 7.07 |
| baseline | clean | 2 swaps | 0.9172 | 13.34 |
| baseline | clean | 2 keyboard typos | 0.9247 | 12.95 |
| baseline | clean | mixed (sub+del+ins) | **0.8903** | 14.51 |
| baseline | matching typos | mixed | 0.9094 | 16.55 |
| augmented | clean | clean | 0.9906 | 7.30 |
| augmented | clean | mixed | 0.9682 | 12.90 |
| augmented | matching typos | mixed | 0.9614 | 11.24 |

(Full grid, including the swap and keyboard rows for every setting, is
printed by the script.)

- **The sets respond to typo uncertainty; the point confidence mostly
  did not.** Sets nearly double under typos, so a caller looking at the
  set sees that the model is less sure.
- **The guarantee is not formally kept, but nearly.** Non-randomized
  APS over-covers clean text, and that slack absorbs most of the shift:
  coverage stays at 91.7-92.5% on swap and keyboard typos and dips to
  89.0% only on the mixed corruption (about 1.8 standard errors below
  90% at n=3080). Calibrating on typo'd validation text, i.e. on data
  that looks like deployment, restores coverage above 90% (90.9% on
  mixed), as exchangeability predicts.
- **Typo augmentation helps here too.** The augmented model keeps 96-98%
  coverage under every corruption, with smaller sets than the baseline
  (9.5-11 classes when calibrated on matching noise, vs 12-17).
