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
| Score | 759 / 1517 | Absolute-residual conformal | 90% | **89.85%** | constant width 1.066 (on [0,1] scale) |

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

### Score: valid, but the interval is not actionable

89.85% coverage against 90% target — essentially exact. But the
interval width is **1.066 on a [0,1]-scale target variable** — wider
than the entire possible range of the label. A predicted score of 0.5
with this interval spans roughly [-0.03, 1.03], i.e. effectively "the
score is somewhere in [0,1]," which is true by construction of the task
and tells a caller nothing. **This is a technically valid but practically
useless result, and it is reported as such rather than hidden or
qualified away.**

Why: this repo's absolute-residual conformal method (see below) produces
one CONSTANT width for every test point, calibrated to cover the 90th
percentile of calibration-set absolute error. The Score head's error
distribution is heavily right-skewed (median absolute error 0.090, but
the 90th percentile is 0.538 and the 95th is 0.724) — verified directly:
the test set is majority 5-star reviews (932 of 1517, 61%), which the
model predicts well, alongside a harder minority of lower-star reviews
it predicts poorly. A constant-width interval calibrated to cover the
90th percentile of error across ALL points has to be wide enough for the
hard minority tail, which makes it needlessly (uselessly) wide for the
easy majority. This is exactly the scenario conformalized quantile
regression (CQR) is designed to fix — CQR would very plausibly give
narrow, useful intervals for easy/majority-class inputs and wide,
honest intervals for the hard minority-class ones, rather than one
number for everyone. **CQR was not implemented here** (this repo used
the simpler standard method, as the original brief for this workstream
explicitly allowed); this result is the concrete evidence for why CQR
would likely matter here specifically, not a generic caveat.

## What this adds over temperature/Platt/isotonic scaling, honestly

- **A real, verified guarantee**, not just an observed average property:
  all three heads hit their coverage target on held-out data. This is
  a genuine methodological upgrade over ECE alone.
- **It does not fix the underlying model.** Conformal prediction
  calibrates the SET, not the point estimate — a weak Score model still
  produces a wide (here, useless) interval; a Choice model with real but
  imperfect discriminative power still needs a set of ~7 classes on
  average to guarantee coverage. Conformal prediction makes weakness
  visible and quantified rather than hidden behind a single miscalibrated
  point estimate — which is valuable — but it is not a substitute for a
  better point predictor.
- **It directly contradicts an unqualified "cannot hallucinate + typed
  output" pitch as a complete trustworthiness story.** A typed API that
  must always return a SINGLE typed answer (as Jev's basic interface
  does) cannot, by definition, return "I need to say {6 possible
  categories}" or "this could be anywhere in [0,1]" — it has to pick one
  value. Conformal prediction's honest answer, when forced into a
  single-point interface, is either the point estimate alone (silently
  discarding the uncertainty this section just measured) or a visibly
  unwieldy/useless set exposed to the caller. Neither resolves the
  tension; this section's job was to make the tension visible and
  measured, which it does.

## Relation to Workstream B

This repo's earlier, sharpest finding (FINDINGS.md §5: ordinary typo
noise collapses Choice accuracy 81.9%→51.2% while confidence drops only
0.791→0.612) predates this conformal analysis and used point-estimate
confidence, not conformal sets. Workstream B (retraining with typo
augmentation) is evaluated separately and does not currently re-run the
conformal pipeline above — that combination (does typo-augmented
training also tighten/loosen conformal set sizes under typo noise) is a
natural follow-up not attempted in this pass, noted here rather than
silently left out.
