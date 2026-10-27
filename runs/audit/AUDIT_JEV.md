# Independent audit of the Jev benchmark numbers

Auditor's own code, written from scratch (it does not import `bench_jev.py`, `redteam_jev.py` or `calib_utils.py`): `runs/audit/audit_abc.py` (checks A-C; imports `noise.py` only for check B, as instructed), `audit_metrics.py` (D, E, F), `audit_extra.py` and `audit_extra2.py` (F), `compare.py` (reported vs recomputed), `write_md.py` (this file). Raw outputs are in `abc_results.json`, `audit_metrics.json`, `audit_extra*.json` and `comparison.json`. No network calls were made.

Scope: Jev entries only, real runs only (`runs/mock/` ignored). Reports were regenerated first with `python3 bench_jev.py report` and `python3 redteam_jev.py report` (copies from before regeneration: `runs/audit/*_before_regen.json`).

## Verdict

**403 of 409 reported Jev numbers reproduce exactly from the raw files. 6 do not, and none of the 6 changes a conclusion.** The inputs, labels, typo generation, hashes and record counts are all clean. The two biggest problems are in how some numbers are framed, not in their arithmetic (see the list at the end).

| FAIL | reported | recomputed (spec) | difference | cause |
|---|---|---|---|---|
| spam/clean `ece` | 0.06382 | 0.06575 | +0.00194 | float32 bin edge: confidence values of exactly 0.9 and 0.7 are put in the bin above |
| score_amazon_clean `pearson_r_expected_stars` | 0.92044 | 0.92045 | +0.00001 | the report divides expected stars by sum(probabilities), which is 0.99 on some rows; the spec does not |
| score_amazon_clean `mae_stars_expected` | 0.35426 | 0.35453 | +0.00027 | the report divides expected stars by sum(probabilities), which is 0.99 on some rows; the spec does not |
| score_amazon_swap2 `pearson_r_expected_stars` | 0.92001 | 0.91997 | -0.00004 | the report divides expected stars by sum(probabilities), which is 0.99 on some rows; the spec does not |
| score_amazon_swap2 `mae_stars_expected` | 0.36061 | 0.36074 | +0.00013 | the report divides expected stars by sum(probabilities), which is 0.99 on some rows; the spec does not |
| injection_oos_clinc `ece` | 0.06018 | 0.06138 | +0.00120 | float32 bin edge: confidence values of exactly 0.9 and 0.7 are put in the bin above |

## Checks A-F

### A. Ground truth: PASS
- `categories.json` has 77 unique names and is identical to `data/raw/banking_categories.json`.
- BANKING77 clean (3,080 rows) vs `data/raw/banking_test.csv` (3,080 rows): **0 label mismatches** by position and **0** when matched by exact text; 0 text mismatches; 0 texts with two categories. 77 intents, exactly 40 rows each.
- Spam (836 rows, 116 spam) vs `data/spam.csv` matched by exact text: **0 label mismatches**. One row (idx 23) only matches after normalising `\r` to `\n`. That row is a malformed CSV line in the source, and its label (ham) still agrees. After normalising, 0 rows are missing and 0 disagree.
- Note: the spam test set holds 17 duplicate texts (819 distinct). Removing them moves Jev's clean accuracy from 0.9605 to 0.9597, which is immaterial.

### B. Typo inputs regenerated from noise.py: PASS
| condition | n | text mismatches | label changes vs clean | texts left unchanged by the noise |
|---|---|---|---|---|
| banking77_swap2 | 3,080 | 0 | 0 | 62 |
| banking77_swap4 | 3,080 | 0 | 0 | 7 |
| banking77_keyboard2 | 3,080 | 0 | 0 | 19 |
| banking77_delete2 | 3,080 | 0 | 0 | 0 |
| banking77_insert2 | 3,080 | 0 | 0 | 0 |
| banking77_mixed3 | 3,080 | 0 | 0 | 0 |
| spam_swap2 | 836 | 0 | 0 | 24 |
| spam_keyboard2 | 836 | 0 | 0 | 5 |
| spam_mixed3 | 836 | 0 | 0 | 0 |

(Swaps of two identical adjacent characters, or keyboard substitutions undone by a later operation, leave some texts unchanged, e.g. 62 of 3,080 for swap2. They still count in the typo condition, so the drops are very slightly understated.)

### C. Integrity: PASS
- Core: 24,904 Jev records in 11 files; red team: 24,766 records in 19 files. Every record has status `ok` and took 1 attempt. Each idx/key has exactly one ok record (0 duplicates, 0 missing, 0 extra). **0 sha256 mismatches** between records and the sha256 of the input text; 0 input files whose stored hash differs from their text.
- `response.model` values seen: {'jev-1.14.0': 49670} (= 24,904 core + 24,766 red-team). No `mock-jev-0.0` record appears in any real file, and `mock` is false in both reports.
- Record timestamps (16:48-17:19 UTC, 27 Oct 2026) agree with the file mtimes and the run logs (`runs/jev_core.log`, `jev_redteam*.log`: 0 errors).

### D. Recompute every Jev metric: 403/409 PASS
Tolerances: 1e-9 for counts, means, rates, latencies and tokens; 0.001 for ECE (per the brief); 0.005 for bootstrap CI endpoints. The full table is at the end.

### E. Bootstrap CI, BANKING77 clean accuracy: PASS
- Reported [0.791883, 0.82013]. Own numpy bootstrap (10k resamples, seed 2026): [0.792208, 0.819805]. Own pure-Python bootstrap (4k, seed 12345): [0.791558, 0.819805]. Wilson interval: [0.791497, 0.819429]. All within 0.0005 of the reported values.

### F. Things that are misleading or fragile

**F1. ECE depends on float32 bin edges (this causes 2 FAILs).** The ECE code builds bin edges with float32 `torch.linspace`, so the 0.7 edge is 0.69999999 and the 0.9 edge is 0.89999998. Jev rounds probabilities to 2 decimals, so many confidences land exactly on an edge: 36 banking-clean answers sit at exactly 0.90, and in spam/clean 34 answers at 0.90 and 5 at 0.70 move up a bin. On spam/clean this lowers the reported ECE from 0.0658 (spec) to 0.0638. On injection_oos_clinc it goes from 0.0614 to 0.0602. Elsewhere the effect is under 0.001, and on every BANKING77 condition the two edge conventions agree to 5 decimals. The fix is to quote the spec values or say 'float32 edges'.

**F2. Amazon expected stars are renormalised (4 tiny FAILs).** The report divides by sum(probabilities). Jev's rounded probabilities sum to 0.99 on 22 of 1,517 clean rows and 9 of 1,517 swap2 rows. Using the definition in the brief (no renormalisation): clean r = 0.92045 (reported 0.92044), MAE = 0.3545 (reported 0.3543); swap2 r = 0.91997, MAE = 0.3607. With renormalisation my code reproduces the reported values exactly. This is a matter of definition, and the effect is under 0.0003.

**F3. Spam accuracy hides poor precision.** The test set is 116/836 spam, so always answering 'ham' scores 0.861. Jev scores 0.961 on clean with recall 0.991 (1 miss) but **precision 0.782**: 32 of 720 real messages are flagged as spam. Precision falls to 0.752 on mixed3. Spam 'overconfidence' is negative (Jev is underconfident), so its ECE measures underconfidence.

**F4. Spam typo drops are not statistically distinguishable from zero.** Paired bootstrap 95% CIs (clean minus condition, same idx): swap2 +0.0048 [-0.0036, +0.0144], keyboard2 +0.0060 [-0.0048, +0.0167], mixed3 +0.0072 [-0.0036, +0.0191]. All BANKING77 drops are significant, but insert2's is borderline: +0.0078 [+0.0013, +0.0146]. Every confidence drop is significant. The report gives CIs for levels but not for drops, so a statement like 'spam accuracy drops under typos' is unsupported.

**F5. Jev is not deterministic, and the repeat test covers a narrow slice.** Sending an identical request again changes the answer on 3% of queries, and the probabilities are identical in only 48.5% of queries. Accuracy on the same 200 queries ranged 0.850-0.865 across the 5 rounds. The 200 repeat queries are the first 200 of the sorted sample (idx 0-598), so they cover only **15 of 77 intents**. '97% same answer' should not be read as the figure for all of BANKING77.

**F6. Order-shuffle sensitivity should be read against that noise floor.** Shuffled options change the answer on 0.050 of the 1,000 queries. On the 200 queries that were also repeated, shuffling disagrees with the core run 0.045 of the time, against 0.012 for a plain resend. So roughly 3 of the 5 points come from option order. (The response key order matched the expected shuffle in 1000/1000 cases, so the shuffle really was sent.)

**F7. The cost figure depends on output tokens being free.** Core: 37,465,151 input tokens give $1.5735. Jev also returned 17,931,243 output tokens (48% of input volume). If output were billed at the input rate, the core run would cost $2.33. The price (an OpenRouter page 'checked 2026-10-27') cannot be checked offline, and the $ figures are tokens times a posted price, not an invoice. Total across core and red team: 83,621,901 input tokens, $3.51.

**F8. What the latency numbers measure.** The core median of 104 ms is client-side wall clock at the default concurrency of 8 from a cloud workspace, including the network. It pools BANKING77 (median 105 ms, max 2857 ms) and spam (94 ms). The 'latency stays flat with more questions' result rests on 60 calls per level, short SMS texts and yes/no questions only: the ratio of medians is 1.09 and the paired per-message median ratio is 1.12. Input tokens rise from 302 to 533, so cost is not flat. CLINC150 calls (151 options) have a median of about 190 ms.

**F9. A degenerate CI.** evasion_spam reports accuracy 1.0 with a bootstrap CI of [1.0, 1.0] (116/116). A percentile bootstrap cannot produce an interval at 100%. The Wilson interval is [0.9679, 1]. The same applies, more mildly, to injection_spam (n = 116) and ood_spam_on_banking (4 false alarms in 1,000).

**F10. The response shape, as observed.** Every `choice` is a valid option. The probability keys equal the option list in every response, and BANKING77 probabilities sum to 0.99-1.00 (2-decimal rounding). Jev's own `confidence` field differs from probabilities[choice] in 402 of 3,080 clean answers, always by exactly -0.01. The report uses probabilities[choice], the higher of the two, so Jev's overconfidence is overstated by about 0.001, which is conservative. In one clean case (idx 2481) `choice` is not the argmax (0.47 vs 0.48, and the argmax was the gold answer). There are 5 exact ties. Scoring by argmax instead of `choice` would give 0.8065 instead of 0.8058. 36% of clean answers have confidence 1.00 (0.980 of them correct).

**F11. Subsets.** Every injection, order and OOD test uses the same 1,000-query sample (seed 7). Jev's clean accuracy on that sample is 0.800, against 0.806 on the full set, and the report correctly compares against `clean_accuracy_same_examples`. The injection_* banking tests share identical targets (0 differences), and `picked_target_without_injection` = 0.005 (5/1000) is computed from the core clean run on the same idx, as claimed. CLINC150 `in_scope_accuracy`, `oos_recall` and `false_oos_rate` use the correct subsets (4,500 in-scope / 1,000 oos). OOS precision, which is not reported, is 0.856 clean and 0.772 swap2.

**F12. Stale report before regeneration.** The `runs/redteam_report.json` on disk before this audit lacked `latency_sequential_banking`, and its red-team cost (45,987,872 tokens, $1.9315) was out of date. After regeneration it reads 46,156,750 tokens, $1.9386. `runs/report.json` changed only in its timestamp. Cite only the regenerated files.

**F13. No mixing with mock data.** The report code reads only `runs/jev` and `runs/redteam/jev`. Every model string is `jev-1.14.0`, and the token totals equal the sum over the real files.

## What a skeptical interviewer could attack

1. **'Jev 80.6% vs BEMA' is apples to oranges.** Jev is zero-shot from category names, while BEMA is trained on 8.5k in-domain examples. The repo says so itself. Lead with the drops, not the levels.
2. **Spam '96% accuracy'** is only 10 points above always answering ham. Precision is 0.78 (32 false positives on 720 ham). Quote recall and precision.
3. **'Jev degrades under typos'** is true for BANKING77 (1-4 points, significant) and not shown for spam (all CIs include 0). The drop CIs are not in the report.
4. **Non-determinism:** a resend changes 3% of answers, and the run-to-run accuracy spread on 200 queries is about 1.5 points. Single-run differences under about 1 point (e.g. insert2's 0.8-point drop) are close to that noise.
5. **ECE is fragile** with 2-decimal probabilities. Bin-edge handling moves spam ECE by 0.002. Quote ECE to 2 decimals, or use overconfidence, which here is nearly identical and has no binning.
6. **'Latency is flat in number of questions'** rests on n = 60 per level, short texts, yes/no questions only, and client-side timing at concurrency 1. The paired ratio is 1.12x, not 1.0x.
7. **Cost** assumes output tokens cost $0 (17.9M output tokens in the core run) and comes from a price page, not a bill.
8. **The red-team repeat/consistency test** covers 15 of 77 intents. **evasion_spam's CI [1.0, 1.0]** is a bootstrap artefact. Use Wilson [0.968, 1.0].
9. **Single provider snapshot:** one model version (jev-1.14.0), one day, one ~30-minute window. A different day or version may differ.

## Full metric table (reported vs recomputed)

| section | metric | reported | recomputed | tol | result |
|---|---|---|---|---|---|
| core banking77/clean | n | 3,080 | 3,080 | 1.000e-09 | PASS |
| core banking77/clean | accuracy | 0.805844 | 0.805844 | 1.000e-09 | PASS |
| core banking77/clean | mean_confidence | 0.878906 | 0.878906 | 1.000e-09 | PASS |
| core banking77/clean | overconfidence | 0.0730617 | 0.0730617 | 1.000e-09 | PASS |
| core banking77/clean | n_missing | 0 | 0 | 1.000e-09 | PASS |
| core banking77/clean | n_failed_calls | 0 | 0 | 1.000e-09 | PASS |
| core banking77/clean | ece (spec: exact decimal bin edges) | 0.0730617 | 0.0730617 | 0.001 | PASS |
| core banking77/clean | ece_using_jev_confidence_field | 0.0717565 | 0.0717565 | 0.001 | PASS |
| core banking77/clean | accuracy_ci95 (own bootstrap, 10k, seed 2026) | [0.791883, 0.82013] | [0.792208, 0.819805] | 0.005 | PASS |
| core banking77/swap2 | n | 3,080 | 3,080 | 1.000e-09 | PASS |
| core banking77/swap2 | accuracy | 0.79026 | 0.79026 | 1.000e-09 | PASS |
| core banking77/swap2 | mean_confidence | 0.857429 | 0.857429 | 1.000e-09 | PASS |
| core banking77/swap2 | overconfidence | 0.0671688 | 0.0671688 | 1.000e-09 | PASS |
| core banking77/swap2 | n_missing | 0 | 0 | 1.000e-09 | PASS |
| core banking77/swap2 | n_failed_calls | 0 | 0 | 1.000e-09 | PASS |
| core banking77/swap2 | accuracy_drop_from_clean | 0.0155844 | 0.0155844 | 1.000e-09 | PASS |
| core banking77/swap2 | confidence_drop_from_clean | 0.0214773 | 0.0214773 | 1.000e-09 | PASS |
| core banking77/swap2 | ece (spec: exact decimal bin edges) | 0.0672078 | 0.0672078 | 0.001 | PASS |
| core banking77/swap2 | ece_using_jev_confidence_field | 0.0655195 | 0.0655195 | 0.001 | PASS |
| core banking77/swap2 | accuracy_ci95 (own bootstrap, 10k, seed 2026) | [0.775974, 0.805195] | [0.775974, 0.804221] | 0.005 | PASS |
| core banking77/swap4 | n | 3,080 | 3,080 | 1.000e-09 | PASS |
| core banking77/swap4 | accuracy | 0.77013 | 0.77013 | 1.000e-09 | PASS |
| core banking77/swap4 | mean_confidence | 0.834714 | 0.834714 | 1.000e-09 | PASS |
| core banking77/swap4 | overconfidence | 0.0645844 | 0.0645844 | 1.000e-09 | PASS |
| core banking77/swap4 | n_missing | 0 | 0 | 1.000e-09 | PASS |
| core banking77/swap4 | n_failed_calls | 0 | 0 | 1.000e-09 | PASS |
| core banking77/swap4 | accuracy_drop_from_clean | 0.0357143 | 0.0357143 | 1.000e-09 | PASS |
| core banking77/swap4 | confidence_drop_from_clean | 0.0441916 | 0.0441916 | 1.000e-09 | PASS |
| core banking77/swap4 | ece (spec: exact decimal bin edges) | 0.0662987 | 0.0662987 | 0.001 | PASS |
| core banking77/swap4 | ece_using_jev_confidence_field | 0.0638344 | 0.0638344 | 0.001 | PASS |
| core banking77/swap4 | accuracy_ci95 (own bootstrap, 10k, seed 2026) | [0.755519, 0.784748] | [0.755519, 0.78474] | 0.005 | PASS |
| core banking77/keyboard2 | n | 3,080 | 3,080 | 1.000e-09 | PASS |
| core banking77/keyboard2 | accuracy | 0.775325 | 0.775325 | 1.000e-09 | PASS |
| core banking77/keyboard2 | mean_confidence | 0.842458 | 0.842458 | 1.000e-09 | PASS |
| core banking77/keyboard2 | overconfidence | 0.0671331 | 0.0671331 | 1.000e-09 | PASS |
| core banking77/keyboard2 | n_missing | 0 | 0 | 1.000e-09 | PASS |
| core banking77/keyboard2 | n_failed_calls | 0 | 0 | 1.000e-09 | PASS |
| core banking77/keyboard2 | accuracy_drop_from_clean | 0.0305195 | 0.0305195 | 1.000e-09 | PASS |
| core banking77/keyboard2 | confidence_drop_from_clean | 0.0364481 | 0.0364481 | 1.000e-09 | PASS |
| core banking77/keyboard2 | ece (spec: exact decimal bin edges) | 0.0680032 | 0.0680032 | 0.001 | PASS |
| core banking77/keyboard2 | ece_using_jev_confidence_field | 0.0663084 | 0.0663084 | 0.001 | PASS |
| core banking77/keyboard2 | accuracy_ci95 (own bootstrap, 10k, seed 2026) | [0.761364, 0.790909] | [0.760706, 0.789935] | 0.005 | PASS |
| core banking77/delete2 | n | 3,080 | 3,080 | 1.000e-09 | PASS |
| core banking77/delete2 | accuracy | 0.787013 | 0.787013 | 1.000e-09 | PASS |
| core banking77/delete2 | mean_confidence | 0.85051 | 0.85051 | 1.000e-09 | PASS |
| core banking77/delete2 | overconfidence | 0.0634968 | 0.0634968 | 1.000e-09 | PASS |
| core banking77/delete2 | n_missing | 0 | 0 | 1.000e-09 | PASS |
| core banking77/delete2 | n_failed_calls | 0 | 0 | 1.000e-09 | PASS |
| core banking77/delete2 | accuracy_drop_from_clean | 0.0188312 | 0.0188312 | 1.000e-09 | PASS |
| core banking77/delete2 | confidence_drop_from_clean | 0.0283961 | 0.0283961 | 1.000e-09 | PASS |
| core banking77/delete2 | ece (spec: exact decimal bin edges) | 0.0649253 | 0.0649253 | 0.001 | PASS |
| core banking77/delete2 | ece_using_jev_confidence_field | 0.0638442 | 0.0638442 | 0.001 | PASS |
| core banking77/delete2 | accuracy_ci95 (own bootstrap, 10k, seed 2026) | [0.773369, 0.800982] | [0.772719, 0.801299] | 0.005 | PASS |
| core banking77/insert2 | n | 3,080 | 3,080 | 1.000e-09 | PASS |
| core banking77/insert2 | accuracy | 0.798052 | 0.798052 | 1.000e-09 | PASS |
| core banking77/insert2 | mean_confidence | 0.866692 | 0.866692 | 1.000e-09 | PASS |
| core banking77/insert2 | overconfidence | 0.0686396 | 0.0686396 | 1.000e-09 | PASS |
| core banking77/insert2 | n_missing | 0 | 0 | 1.000e-09 | PASS |
| core banking77/insert2 | n_failed_calls | 0 | 0 | 1.000e-09 | PASS |
| core banking77/insert2 | accuracy_drop_from_clean | 0.00779221 | 0.00779221 | 1.000e-09 | PASS |
| core banking77/insert2 | confidence_drop_from_clean | 0.0122143 | 0.0122143 | 1.000e-09 | PASS |
| core banking77/insert2 | ece (spec: exact decimal bin edges) | 0.0689123 | 0.0689123 | 0.001 | PASS |
| core banking77/insert2 | ece_using_jev_confidence_field | 0.0685844 | 0.0685844 | 0.001 | PASS |
| core banking77/insert2 | accuracy_ci95 (own bootstrap, 10k, seed 2026) | [0.784416, 0.812021] | [0.784091, 0.812013] | 0.005 | PASS |
| core banking77/mixed3 | n | 3,080 | 3,080 | 1.000e-09 | PASS |
| core banking77/mixed3 | accuracy | 0.774026 | 0.774026 | 1.000e-09 | PASS |
| core banking77/mixed3 | mean_confidence | 0.837594 | 0.837594 | 1.000e-09 | PASS |
| core banking77/mixed3 | overconfidence | 0.0635682 | 0.0635682 | 1.000e-09 | PASS |
| core banking77/mixed3 | n_missing | 0 | 0 | 1.000e-09 | PASS |
| core banking77/mixed3 | n_failed_calls | 0 | 0 | 1.000e-09 | PASS |
| core banking77/mixed3 | accuracy_drop_from_clean | 0.0318182 | 0.0318182 | 1.000e-09 | PASS |
| core banking77/mixed3 | confidence_drop_from_clean | 0.0413117 | 0.0413117 | 1.000e-09 | PASS |
| core banking77/mixed3 | ece (spec: exact decimal bin edges) | 0.0635682 | 0.0635682 | 0.001 | PASS |
| core banking77/mixed3 | ece_using_jev_confidence_field | 0.0617175 | 0.0617175 | 0.001 | PASS |
| core banking77/mixed3 | accuracy_ci95 (own bootstrap, 10k, seed 2026) | [0.760065, 0.78961] | [0.759407, 0.788636] | 0.005 | PASS |
| core spam/clean | n | 836 | 836 | 1.000e-09 | PASS |
| core spam/clean | accuracy | 0.960526 | 0.960526 | 1.000e-09 | PASS |
| core spam/clean | mean_confidence | 0.896711 | 0.896711 | 1.000e-09 | PASS |
| core spam/clean | overconfidence | -0.0638158 | -0.0638158 | 1.000e-09 | PASS |
| core spam/clean | n_missing | 0 | 0 | 1.000e-09 | PASS |
| core spam/clean | n_failed_calls | 0 | 0 | 1.000e-09 | PASS |
| core spam/clean | ece (spec: exact decimal bin edges) | 0.0638158 | 0.0657536 | 0.001 | FAIL: float32 bin edges (see F1) |
| core spam/clean | accuracy_ci95 (own bootstrap, 10k, seed 2026) | [0.947368, 0.973684] | [0.946172, 0.972518] | 0.005 | PASS |
| core spam/swap2 | n | 836 | 836 | 1.000e-09 | PASS |
| core spam/swap2 | accuracy | 0.955742 | 0.955742 | 1.000e-09 | PASS |
| core spam/swap2 | mean_confidence | 0.88238 | 0.88238 | 1.000e-09 | PASS |
| core spam/swap2 | overconfidence | -0.0733612 | -0.0733612 | 1.000e-09 | PASS |
| core spam/swap2 | n_missing | 0 | 0 | 1.000e-09 | PASS |
| core spam/swap2 | n_failed_calls | 0 | 0 | 1.000e-09 | PASS |
| core spam/swap2 | accuracy_drop_from_clean | 0.00478469 | 0.00478469 | 1.000e-09 | PASS |
| core spam/swap2 | confidence_drop_from_clean | 0.0143301 | 0.0143301 | 1.000e-09 | PASS |
| core spam/swap2 | ece (spec: exact decimal bin edges) | 0.0733612 | 0.0733612 | 0.001 | PASS |
| core spam/swap2 | accuracy_ci95 (own bootstrap, 10k, seed 2026) | [0.941388, 0.9689] | [0.941388, 0.9689] | 0.005 | PASS |
| core spam/keyboard2 | n | 836 | 836 | 1.000e-09 | PASS |
| core spam/keyboard2 | accuracy | 0.954545 | 0.954545 | 1.000e-09 | PASS |
| core spam/keyboard2 | mean_confidence | 0.879988 | 0.879988 | 1.000e-09 | PASS |
| core spam/keyboard2 | overconfidence | -0.0745574 | -0.0745574 | 1.000e-09 | PASS |
| core spam/keyboard2 | n_missing | 0 | 0 | 1.000e-09 | PASS |
| core spam/keyboard2 | n_failed_calls | 0 | 0 | 1.000e-09 | PASS |
| core spam/keyboard2 | accuracy_drop_from_clean | 0.00598086 | 0.00598086 | 1.000e-09 | PASS |
| core spam/keyboard2 | confidence_drop_from_clean | 0.0167225 | 0.0167225 | 1.000e-09 | PASS |
| core spam/keyboard2 | ece (spec: exact decimal bin edges) | 0.0745574 | 0.0745574 | 0.001 | PASS |
| core spam/keyboard2 | accuracy_ci95 (own bootstrap, 10k, seed 2026) | [0.940161, 0.9689] | [0.940191, 0.967703] | 0.005 | PASS |
| core spam/mixed3 | n | 836 | 836 | 1.000e-09 | PASS |
| core spam/mixed3 | accuracy | 0.953349 | 0.953349 | 1.000e-09 | PASS |
| core spam/mixed3 | mean_confidence | 0.878301 | 0.878301 | 1.000e-09 | PASS |
| core spam/mixed3 | overconfidence | -0.0750478 | -0.0750478 | 1.000e-09 | PASS |
| core spam/mixed3 | n_missing | 0 | 0 | 1.000e-09 | PASS |
| core spam/mixed3 | n_failed_calls | 0 | 0 | 1.000e-09 | PASS |
| core spam/mixed3 | accuracy_drop_from_clean | 0.00717703 | 0.00717703 | 1.000e-09 | PASS |
| core spam/mixed3 | confidence_drop_from_clean | 0.0184091 | 0.0184091 | 1.000e-09 | PASS |
| core spam/mixed3 | ece (spec: exact decimal bin edges) | 0.0750478 | 0.0750478 | 0.001 | PASS |
| core spam/mixed3 | accuracy_ci95 (own bootstrap, 10k, seed 2026) | [0.938995, 0.966507] | [0.938995, 0.966507] | 0.005 | PASS |
| core latency | n | 24,904 | 24,904 | 1.000e-09 | PASS |
| core latency | median_ms | 104.01 | 104.01 | 1.000e-09 | PASS |
| core latency | p90_ms (index rule s[int(0.9n)]) | 138.655 | 138.655 | 1.000e-09 | PASS |
| core latency | p99_ms (index rule s[int(0.99n)]) | 192.213 | 192.213 | 1.000e-09 | PASS |
| core cost | input_tokens | 37,465,151 | 37,465,151 | 1.000e-09 | PASS |
| core cost | usd | 1.57354 | 1.57354 | 1.000e-09 | PASS |
| core | jev_failed_calls_total | 0 | 0 | 1.000e-09 | PASS |
| core | model versions | {"jev-1.14.0": 24904} | {"jev-1.14.0": 24904} | exact | PASS |
| core | mock flag | False | False | exact | PASS |
| redteam injection_banking | n | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam injection_banking | n_missing | 0 | 0 | 1.000e-09 | PASS |
| redteam injection_banking | accuracy | 0.474 | 0.474 | 1.000e-09 | PASS |
| redteam injection_banking | mean_confidence | 0.70736 | 0.70736 | 1.000e-09 | PASS |
| redteam injection_banking | overconfidence | 0.23336 | 0.23336 | 1.000e-09 | PASS |
| redteam injection_banking | clean_accuracy_same_examples | 0.8 | 0.8 | 1.000e-09 | PASS |
| redteam injection_banking | followed_injected_label | 0.457 | 0.457 | 1.000e-09 | PASS |
| redteam injection_banking | mean_confidence_when_following_injection | 0.697484 | 0.697484 | 1.000e-09 | PASS |
| redteam injection_banking | share_following_injection_with_confidence_ge_0.9 | 0.129103 | 0.129103 | 1.000e-09 | PASS |
| redteam injection_banking | picked_target_without_injection | 0.005 | 0.005 | 1.000e-09 | PASS |
| redteam injection_banking | ece | 0.23478 | 0.23478 | 0.001 | PASS |
| redteam injection_banking | accuracy_ci95 | [0.444, 0.503] | [0.443, 0.504] | 0.005 | PASS |
| redteam injection_banking | calls.ok | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam injection_banking | calls.input_tokens | 1,705,932 | 1,705,932 | 1.000e-09 | PASS |
| redteam injection_banking | calls.median_latency_ms | 102.583 | 102.583 | 1.000e-09 | PASS |
| redteam injection_banking | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam injection_spam | n | 116 | 116 | 1.000e-09 | PASS |
| redteam injection_spam | n_missing | 0 | 0 | 1.000e-09 | PASS |
| redteam injection_spam | accuracy | 0.965517 | 0.965517 | 1.000e-09 | PASS |
| redteam injection_spam | mean_confidence | 0.90931 | 0.90931 | 1.000e-09 | PASS |
| redteam injection_spam | overconfidence | -0.0562069 | -0.0562069 | 1.000e-09 | PASS |
| redteam injection_spam | mean_p_spam | 0.897241 | 0.897241 | 1.000e-09 | PASS |
| redteam injection_spam | clean_recall_same_examples | 0.991379 | 0.991379 | 1.000e-09 | PASS |
| redteam injection_spam | ece | 0.0767241 | 0.0767241 | 0.001 | PASS |
| redteam injection_spam | accuracy_ci95 | [0.931034, 0.991379] | [0.931034, 0.991379] | 0.005 | PASS |
| redteam injection_spam | calls.ok | 116 | 116 | 1.000e-09 | PASS |
| redteam injection_spam | calls.input_tokens | 39,189 | 39,189 | 1.000e-09 | PASS |
| redteam injection_spam | calls.median_latency_ms | 95.6135 | 95.6135 | 1.000e-09 | PASS |
| redteam injection_spam | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam evasion_spam | n | 116 | 116 | 1.000e-09 | PASS |
| redteam evasion_spam | n_missing | 0 | 0 | 1.000e-09 | PASS |
| redteam evasion_spam | accuracy | 1 | 1 | 1.000e-09 | PASS |
| redteam evasion_spam | mean_confidence | 0.952586 | 0.952586 | 1.000e-09 | PASS |
| redteam evasion_spam | overconfidence | -0.0474138 | -0.0474138 | 1.000e-09 | PASS |
| redteam evasion_spam | mean_p_spam | 0.952586 | 0.952586 | 1.000e-09 | PASS |
| redteam evasion_spam | clean_recall_same_examples | 0.991379 | 0.991379 | 1.000e-09 | PASS |
| redteam evasion_spam | ece | 0.0474138 | 0.0474138 | 0.001 | PASS |
| redteam evasion_spam | accuracy_ci95 | [1, 1] | [1, 1] | 0.005 | PASS |
| redteam evasion_spam | calls.ok | 116 | 116 | 1.000e-09 | PASS |
| redteam evasion_spam | calls.input_tokens | 43,298 | 43,298 | 1.000e-09 | PASS |
| redteam evasion_spam | calls.median_latency_ms | 94.0382 | 94.0382 | 1.000e-09 | PASS |
| redteam evasion_spam | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam ood_spam_on_banking | n | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam ood_spam_on_banking | n_missing | 0 | 0 | 1.000e-09 | PASS |
| redteam ood_spam_on_banking | accuracy | 0.996 | 0.996 | 1.000e-09 | PASS |
| redteam ood_spam_on_banking | mean_confidence | 0.86942 | 0.86942 | 1.000e-09 | PASS |
| redteam ood_spam_on_banking | overconfidence | -0.12658 | -0.12658 | 1.000e-09 | PASS |
| redteam ood_spam_on_banking | mean_p_spam | 0.13112 | 0.13112 | 1.000e-09 | PASS |
| redteam ood_spam_on_banking | ece | 0.12658 | 0.12658 | 0.001 | PASS |
| redteam ood_spam_on_banking | accuracy_ci95 | [0.992, 0.999] | [0.992, 0.999] | 0.005 | PASS |
| redteam ood_spam_on_banking | calls.ok | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam ood_spam_on_banking | calls.input_tokens | 284,632 | 284,632 | 1.000e-09 | PASS |
| redteam ood_spam_on_banking | calls.median_latency_ms | 92.2679 | 92.2679 | 1.000e-09 | PASS |
| redteam ood_spam_on_banking | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam gibberish_banking | digits_symbols: n | 25 | 25 | 1.000e-09 | PASS |
| redteam gibberish_banking | digits_symbols: mean_confidence | 0.1028 | 0.1028 | 1.000e-09 | PASS |
| redteam gibberish_banking | digits_symbols: share_confidence_ge_0.9 | 0 | 0 | 1.000e-09 | PASS |
| redteam gibberish_banking | digits_symbols: share_confidence_ge_0.5 | 0 | 0 | 1.000e-09 | PASS |
| redteam gibberish_banking | random_letters: n | 25 | 25 | 1.000e-09 | PASS |
| redteam gibberish_banking | random_letters: mean_confidence | 0.0824 | 0.0824 | 1.000e-09 | PASS |
| redteam gibberish_banking | random_letters: share_confidence_ge_0.9 | 0 | 0 | 1.000e-09 | PASS |
| redteam gibberish_banking | random_letters: share_confidence_ge_0.5 | 0 | 0 | 1.000e-09 | PASS |
| redteam gibberish_banking | real_offtopic_query (CLINC150 oos_test): n | 25 | 25 | 1.000e-09 | PASS |
| redteam gibberish_banking | real_offtopic_query (CLINC150 oos_test): mean_confidence | 0.266 | 0.266 | 1.000e-09 | PASS |
| redteam gibberish_banking | real_offtopic_query (CLINC150 oos_test): share_confidence_ge_0.9 | 0.04 | 0.04 | 1.000e-09 | PASS |
| redteam gibberish_banking | real_offtopic_query (CLINC150 oos_test): share_confidence_ge_0.5 | 0.12 | 0.12 | 1.000e-09 | PASS |
| redteam gibberish_banking | scrambled_real_query: n | 25 | 25 | 1.000e-09 | PASS |
| redteam gibberish_banking | scrambled_real_query: mean_confidence | 0.1276 | 0.1276 | 1.000e-09 | PASS |
| redteam gibberish_banking | scrambled_real_query: share_confidence_ge_0.9 | 0 | 0 | 1.000e-09 | PASS |
| redteam gibberish_banking | scrambled_real_query: share_confidence_ge_0.5 | 0.04 | 0.04 | 1.000e-09 | PASS |
| redteam gibberish_banking | all: n | 100 | 100 | 1.000e-09 | PASS |
| redteam gibberish_banking | all: mean_confidence | 0.1447 | 0.1447 | 1.000e-09 | PASS |
| redteam gibberish_banking | all: share_confidence_ge_0.9 | 0.01 | 0.01 | 1.000e-09 | PASS |
| redteam gibberish_banking | all: share_confidence_ge_0.5 | 0.04 | 0.04 | 1.000e-09 | PASS |
| redteam gibberish_banking | calls.ok | 100 | 100 | 1.000e-09 | PASS |
| redteam gibberish_banking | calls.input_tokens | 169,084 | 169,084 | 1.000e-09 | PASS |
| redteam gibberish_banking | calls.median_latency_ms | 104.353 | 104.353 | 1.000e-09 | PASS |
| redteam gibberish_banking | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam order_shuffle_banking | n | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam order_shuffle_banking | n_missing | 0 | 0 | 1.000e-09 | PASS |
| redteam order_shuffle_banking | accuracy | 0.799 | 0.799 | 1.000e-09 | PASS |
| redteam order_shuffle_banking | mean_confidence | 0.87343 | 0.87343 | 1.000e-09 | PASS |
| redteam order_shuffle_banking | overconfidence | 0.07443 | 0.07443 | 1.000e-09 | PASS |
| redteam order_shuffle_banking | clean_accuracy_same_examples | 0.8 | 0.8 | 1.000e-09 | PASS |
| redteam order_shuffle_banking | same_answer_as_original_order | 0.95 | 0.95 | 1.000e-09 | PASS |
| redteam order_shuffle_banking | mean_abs_change_in_p_gold | 0.03238 | 0.03238 | 1.000e-09 | PASS |
| redteam order_shuffle_banking | ece | 0.07629 | 0.07629 | 0.001 | PASS |
| redteam order_shuffle_banking | accuracy_ci95 | [0.774, 0.823] | [0.774, 0.823] | 0.005 | PASS |
| redteam order_shuffle_banking | calls.ok | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam order_shuffle_banking | calls.input_tokens | 1,689,632 | 1,689,632 | 1.000e-09 | PASS |
| redteam order_shuffle_banking | calls.median_latency_ms | 103.108 | 103.108 | 1.000e-09 | PASS |
| redteam order_shuffle_banking | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam repeat_banking | n_queries | 200 | 200 | 1.000e-09 | PASS |
| redteam repeat_banking | calls | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam repeat_banking | share_same_answer_all_repeats | 0.97 | 0.97 | 1.000e-09 | PASS |
| redteam repeat_banking | share_identical_probabilities_all_repeats | 0.485 | 0.485 | 1.000e-09 | PASS |
| redteam repeat_banking | mean_confidence_spread_max_minus_min | 0.0223 | 0.0223 | 1.000e-09 | PASS |
| redteam repeat_banking | calls.ok | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam repeat_banking | calls.input_tokens | 1,689,260 | 1,689,260 | 1.000e-09 | PASS |
| redteam repeat_banking | calls.median_latency_ms | 105.802 | 105.802 | 1.000e-09 | PASS |
| redteam repeat_banking | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam question_count | n=1: calls | 60 | 60 | 1.000e-09 | PASS |
| redteam question_count | n=1: median_latency_ms | 96.1983 | 96.1983 | 1.000e-09 | PASS |
| redteam question_count | n=1: p90_latency_ms | 129.059 | 129.059 | 1.000e-09 | PASS |
| redteam question_count | n=1: mean_input_tokens | 302.333 | 302.333 | 1.000e-09 | PASS |
| redteam question_count | n=1: all_questions_answered | True | True | exact | PASS |
| redteam question_count | n=2: calls | 60 | 60 | 1.000e-09 | PASS |
| redteam question_count | n=2: median_latency_ms | 93.0347 | 93.0347 | 1.000e-09 | PASS |
| redteam question_count | n=2: p90_latency_ms | 123.674 | 123.674 | 1.000e-09 | PASS |
| redteam question_count | n=2: mean_input_tokens | 320.333 | 320.333 | 1.000e-09 | PASS |
| redteam question_count | n=2: all_questions_answered | True | True | exact | PASS |
| redteam question_count | n=4: calls | 60 | 60 | 1.000e-09 | PASS |
| redteam question_count | n=4: median_latency_ms | 99.049 | 99.049 | 1.000e-09 | PASS |
| redteam question_count | n=4: p90_latency_ms | 146.886 | 146.886 | 1.000e-09 | PASS |
| redteam question_count | n=4: mean_input_tokens | 353.333 | 353.333 | 1.000e-09 | PASS |
| redteam question_count | n=4: all_questions_answered | True | True | exact | PASS |
| redteam question_count | n=8: calls | 60 | 60 | 1.000e-09 | PASS |
| redteam question_count | n=8: median_latency_ms | 94.9717 | 94.9717 | 1.000e-09 | PASS |
| redteam question_count | n=8: p90_latency_ms | 134.824 | 134.824 | 1.000e-09 | PASS |
| redteam question_count | n=8: mean_input_tokens | 410.333 | 410.333 | 1.000e-09 | PASS |
| redteam question_count | n=8: all_questions_answered | True | True | exact | PASS |
| redteam question_count | n=16: calls | 60 | 60 | 1.000e-09 | PASS |
| redteam question_count | n=16: median_latency_ms | 104.733 | 104.733 | 1.000e-09 | PASS |
| redteam question_count | n=16: p90_latency_ms | 134.195 | 134.195 | 1.000e-09 | PASS |
| redteam question_count | n=16: mean_input_tokens | 533.333 | 533.333 | 1.000e-09 | PASS |
| redteam question_count | n=16: all_questions_answered | True | True | exact | PASS |
| redteam question_count | latency_ratio_16_vs_1 | 1.08872 | 1.08872 | 1.000e-09 | PASS |
| redteam question_count | calls.ok | 300 | 300 | 1.000e-09 | PASS |
| redteam question_count | calls.input_tokens | 115,180 | 115,180 | 1.000e-09 | PASS |
| redteam question_count | calls.median_latency_ms | 97.3718 | 97.3718 | 1.000e-09 | PASS |
| redteam question_count | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam score_amazon_clean | n | 1,517 | 1,517 | 1.000e-09 | PASS |
| redteam score_amazon_clean | n_missing | 0 | 0 | 1.000e-09 | PASS |
| redteam score_amazon_clean | pearson_r_expected_stars | 0.920444 | 0.92045 | 1.000e-09 | FAIL (definition, tiny; see F2) |
| redteam score_amazon_clean | mae_stars_expected | 0.354258 | 0.354529 | 1.000e-09 | FAIL (definition, tiny; see F2) |
| redteam score_amazon_clean | exact_star_accuracy_argmax | 0.752802 | 0.752802 | 1.000e-09 | PASS |
| redteam score_amazon_clean | within_one_star_argmax | 0.969018 | 0.969018 | 1.000e-09 | PASS |
| redteam score_amazon_clean | ece_argmax | 0.0580949 | 0.0580949 | 0.001 | PASS |
| redteam score_amazon_clean | gold_distribution | {"1": 162, "2": 82, "3": 119, "4": 222, "5": 932} | {"1": 162, "2": 82, "3": 119, "4": 222, "5": 932} | exact | PASS |
| redteam score_amazon_clean | calls.ok | 1,517 | 1,517 | 1.000e-09 | PASS |
| redteam score_amazon_clean | calls.input_tokens | 648,851 | 648,851 | 1.000e-09 | PASS |
| redteam score_amazon_clean | calls.median_latency_ms | 113.904 | 113.904 | 1.000e-09 | PASS |
| redteam score_amazon_clean | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam score_amazon_swap2 | n | 1,517 | 1,517 | 1.000e-09 | PASS |
| redteam score_amazon_swap2 | n_missing | 0 | 0 | 1.000e-09 | PASS |
| redteam score_amazon_swap2 | pearson_r_expected_stars | 0.920014 | 0.91997 | 1.000e-09 | FAIL (definition, tiny; see F2) |
| redteam score_amazon_swap2 | mae_stars_expected | 0.36061 | 0.360745 | 1.000e-09 | FAIL (definition, tiny; see F2) |
| redteam score_amazon_swap2 | exact_star_accuracy_argmax | 0.755438 | 0.755438 | 1.000e-09 | PASS |
| redteam score_amazon_swap2 | within_one_star_argmax | 0.969677 | 0.969677 | 1.000e-09 | PASS |
| redteam score_amazon_swap2 | ece_argmax | 0.0513975 | 0.0513975 | 0.001 | PASS |
| redteam score_amazon_swap2 | gold_distribution | {"1": 162, "2": 82, "3": 119, "4": 222, "5": 932} | {"1": 162, "2": 82, "3": 119, "4": 222, "5": 932} | exact | PASS |
| redteam score_amazon_swap2 | calls.ok | 1,517 | 1,517 | 1.000e-09 | PASS |
| redteam score_amazon_swap2 | calls.input_tokens | 652,230 | 652,230 | 1.000e-09 | PASS |
| redteam score_amazon_swap2 | calls.median_latency_ms | 117.564 | 117.564 | 1.000e-09 | PASS |
| redteam score_amazon_swap2 | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam clinc150_clean | n | 5,500 | 5,500 | 1.000e-09 | PASS |
| redteam clinc150_clean | n_missing | 0 | 0 | 1.000e-09 | PASS |
| redteam clinc150_clean | accuracy | 0.908545 | 0.908545 | 1.000e-09 | PASS |
| redteam clinc150_clean | mean_confidence | 0.916593 | 0.916593 | 1.000e-09 | PASS |
| redteam clinc150_clean | overconfidence | 0.00804727 | 0.00804727 | 1.000e-09 | PASS |
| redteam clinc150_clean | in_scope_accuracy | 0.907778 | 0.907778 | 1.000e-09 | PASS |
| redteam clinc150_clean | oos_recall | 0.912 | 0.912 | 1.000e-09 | PASS |
| redteam clinc150_clean | false_oos_rate_on_in_scope | 0.034 | 0.034 | 1.000e-09 | PASS |
| redteam clinc150_clean | n_in_scope | 4,500 | 4,500 | 1.000e-09 | PASS |
| redteam clinc150_clean | n_oos | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam clinc150_clean | ece | 0.0170873 | 0.0170873 | 0.001 | PASS |
| redteam clinc150_clean | accuracy_ci95 | [0.900909, 0.916005] | [0.900727, 0.916] | 0.005 | PASS |
| redteam clinc150_clean | calls.ok | 5,500 | 5,500 | 1.000e-09 | PASS |
| redteam clinc150_clean | calls.input_tokens | 13,915,632 | 13,915,632 | 1.000e-09 | PASS |
| redteam clinc150_clean | calls.median_latency_ms | 190.522 | 190.522 | 1.000e-09 | PASS |
| redteam clinc150_clean | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam clinc150_swap2 | n | 5,500 | 5,500 | 1.000e-09 | PASS |
| redteam clinc150_swap2 | n_missing | 0 | 0 | 1.000e-09 | PASS |
| redteam clinc150_swap2 | accuracy | 0.872182 | 0.872182 | 1.000e-09 | PASS |
| redteam clinc150_swap2 | mean_confidence | 0.884293 | 0.884293 | 1.000e-09 | PASS |
| redteam clinc150_swap2 | overconfidence | 0.0121109 | 0.0121109 | 1.000e-09 | PASS |
| redteam clinc150_swap2 | in_scope_accuracy | 0.864889 | 0.864889 | 1.000e-09 | PASS |
| redteam clinc150_swap2 | oos_recall | 0.905 | 0.905 | 1.000e-09 | PASS |
| redteam clinc150_swap2 | false_oos_rate_on_in_scope | 0.0595556 | 0.0595556 | 1.000e-09 | PASS |
| redteam clinc150_swap2 | n_in_scope | 4,500 | 4,500 | 1.000e-09 | PASS |
| redteam clinc150_swap2 | n_oos | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam clinc150_swap2 | ece | 0.0181036 | 0.0181036 | 0.001 | PASS |
| redteam clinc150_swap2 | accuracy_ci95 | [0.863273, 0.881091] | [0.863091, 0.880909] | 0.005 | PASS |
| redteam clinc150_swap2 | calls.ok | 5,500 | 5,500 | 1.000e-09 | PASS |
| redteam clinc150_swap2 | calls.input_tokens | 13,926,553 | 13,926,553 | 1.000e-09 | PASS |
| redteam clinc150_swap2 | calls.median_latency_ms | 187.347 | 187.347 | 1.000e-09 | PASS |
| redteam clinc150_swap2 | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam injection_control_banking | n | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam injection_control_banking | n_missing | 0 | 0 | 1.000e-09 | PASS |
| redteam injection_control_banking | accuracy | 0.8 | 0.8 | 1.000e-09 | PASS |
| redteam injection_control_banking | mean_confidence | 0.8773 | 0.8773 | 1.000e-09 | PASS |
| redteam injection_control_banking | overconfidence | 0.0773 | 0.0773 | 1.000e-09 | PASS |
| redteam injection_control_banking | clean_accuracy_same_examples | 0.8 | 0.8 | 1.000e-09 | PASS |
| redteam injection_control_banking | ece | 0.07814 | 0.07814 | 0.001 | PASS |
| redteam injection_control_banking | accuracy_ci95 | [0.776, 0.825] | [0.775, 0.824] | 0.005 | PASS |
| redteam injection_control_banking | calls.ok | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam injection_control_banking | calls.input_tokens | 1,704,632 | 1,704,632 | 1.000e-09 | PASS |
| redteam injection_control_banking | calls.median_latency_ms | 98.3049 | 98.3049 | 1.000e-09 | PASS |
| redteam injection_control_banking | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam injection_prefix_banking | n | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam injection_prefix_banking | n_missing | 0 | 0 | 1.000e-09 | PASS |
| redteam injection_prefix_banking | accuracy | 0.62 | 0.62 | 1.000e-09 | PASS |
| redteam injection_prefix_banking | mean_confidence | 0.71943 | 0.71943 | 1.000e-09 | PASS |
| redteam injection_prefix_banking | overconfidence | 0.09943 | 0.09943 | 1.000e-09 | PASS |
| redteam injection_prefix_banking | clean_accuracy_same_examples | 0.8 | 0.8 | 1.000e-09 | PASS |
| redteam injection_prefix_banking | followed_injected_label | 0.277 | 0.277 | 1.000e-09 | PASS |
| redteam injection_prefix_banking | mean_confidence_when_following_injection | 0.660289 | 0.660289 | 1.000e-09 | PASS |
| redteam injection_prefix_banking | share_following_injection_with_confidence_ge_0.9 | 0.104693 | 0.104693 | 1.000e-09 | PASS |
| redteam injection_prefix_banking | picked_target_without_injection | 0.005 | 0.005 | 1.000e-09 | PASS |
| redteam injection_prefix_banking | ece | 0.09943 | 0.09943 | 0.001 | PASS |
| redteam injection_prefix_banking | accuracy_ci95 | [0.589975, 0.65] | [0.591, 0.649] | 0.005 | PASS |
| redteam injection_prefix_banking | calls.ok | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam injection_prefix_banking | calls.input_tokens | 1,704,925 | 1,704,925 | 1.000e-09 | PASS |
| redteam injection_prefix_banking | calls.median_latency_ms | 99.6034 | 99.6034 | 1.000e-09 | PASS |
| redteam injection_prefix_banking | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam injection_casual_banking | n | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam injection_casual_banking | n_missing | 0 | 0 | 1.000e-09 | PASS |
| redteam injection_casual_banking | accuracy | 0.326 | 0.326 | 1.000e-09 | PASS |
| redteam injection_casual_banking | mean_confidence | 0.73497 | 0.73497 | 1.000e-09 | PASS |
| redteam injection_casual_banking | overconfidence | 0.40897 | 0.40897 | 1.000e-09 | PASS |
| redteam injection_casual_banking | clean_accuracy_same_examples | 0.8 | 0.8 | 1.000e-09 | PASS |
| redteam injection_casual_banking | followed_injected_label | 0.635 | 0.635 | 1.000e-09 | PASS |
| redteam injection_casual_banking | mean_confidence_when_following_injection | 0.753874 | 0.753874 | 1.000e-09 | PASS |
| redteam injection_casual_banking | share_following_injection_with_confidence_ge_0.9 | 0.248819 | 0.248819 | 1.000e-09 | PASS |
| redteam injection_casual_banking | picked_target_without_injection | 0.005 | 0.005 | 1.000e-09 | PASS |
| redteam injection_casual_banking | ece | 0.40955 | 0.40955 | 0.001 | PASS |
| redteam injection_casual_banking | accuracy_ci95 | [0.297, 0.355] | [0.297, 0.354] | 0.005 | PASS |
| redteam injection_casual_banking | calls.ok | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam injection_casual_banking | calls.input_tokens | 1,700,923 | 1,700,923 | 1.000e-09 | PASS |
| redteam injection_casual_banking | calls.median_latency_ms | 103.27 | 103.27 | 1.000e-09 | PASS |
| redteam injection_casual_banking | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam injection_defended_banking | n | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam injection_defended_banking | n_missing | 0 | 0 | 1.000e-09 | PASS |
| redteam injection_defended_banking | accuracy | 0.778 | 0.778 | 1.000e-09 | PASS |
| redteam injection_defended_banking | mean_confidence | 0.84839 | 0.84839 | 1.000e-09 | PASS |
| redteam injection_defended_banking | overconfidence | 0.07039 | 0.07039 | 1.000e-09 | PASS |
| redteam injection_defended_banking | clean_accuracy_same_examples | 0.8 | 0.8 | 1.000e-09 | PASS |
| redteam injection_defended_banking | followed_injected_label | 0.027 | 0.027 | 1.000e-09 | PASS |
| redteam injection_defended_banking | mean_confidence_when_following_injection | 0.667407 | 0.667407 | 1.000e-09 | PASS |
| redteam injection_defended_banking | share_following_injection_with_confidence_ge_0.9 | 0.185185 | 0.185185 | 1.000e-09 | PASS |
| redteam injection_defended_banking | picked_target_without_injection | 0.005 | 0.005 | 1.000e-09 | PASS |
| redteam injection_defended_banking | ece | 0.07419 | 0.07419 | 0.001 | PASS |
| redteam injection_defended_banking | accuracy_ci95 | [0.752, 0.803] | [0.752, 0.803] | 0.005 | PASS |
| redteam injection_defended_banking | calls.ok | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam injection_defended_banking | calls.input_tokens | 1,729,932 | 1,729,932 | 1.000e-09 | PASS |
| redteam injection_defended_banking | calls.median_latency_ms | 102.49 | 102.49 | 1.000e-09 | PASS |
| redteam injection_defended_banking | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam injection_oos_clinc | n | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam injection_oos_clinc | n_missing | 0 | 0 | 1.000e-09 | PASS |
| redteam injection_oos_clinc | accuracy | 0.805 | 0.805 | 1.000e-09 | PASS |
| redteam injection_oos_clinc | mean_confidence | 0.77308 | 0.77308 | 1.000e-09 | PASS |
| redteam injection_oos_clinc | overconfidence | -0.03192 | -0.03192 | 1.000e-09 | PASS |
| redteam injection_oos_clinc | said_oos_with_note | 0.16 | 0.16 | 1.000e-09 | PASS |
| redteam injection_oos_clinc | said_oos_without_note_same_requests | 0.027 | 0.027 | 1.000e-09 | PASS |
| redteam injection_oos_clinc | clean_accuracy_same_requests | 0.92 | 0.92 | 1.000e-09 | PASS |
| redteam injection_oos_clinc | ece | 0.06018 | 0.06138 | 0.001 | FAIL: float32 bin edges (see F1) |
| redteam injection_oos_clinc | accuracy_ci95 | [0.78, 0.83] | [0.779, 0.829] | 0.005 | PASS |
| redteam injection_oos_clinc | calls.ok | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam injection_oos_clinc | calls.input_tokens | 2,543,064 | 2,543,064 | 1.000e-09 | PASS |
| redteam injection_oos_clinc | calls.median_latency_ms | 134.766 | 134.766 | 1.000e-09 | PASS |
| redteam injection_oos_clinc | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam injection_casual_defended_banking | n | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam injection_casual_defended_banking | n_missing | 0 | 0 | 1.000e-09 | PASS |
| redteam injection_casual_defended_banking | accuracy | 0.63 | 0.63 | 1.000e-09 | PASS |
| redteam injection_casual_defended_banking | mean_confidence | 0.75539 | 0.75539 | 1.000e-09 | PASS |
| redteam injection_casual_defended_banking | overconfidence | 0.12539 | 0.12539 | 1.000e-09 | PASS |
| redteam injection_casual_defended_banking | clean_accuracy_same_examples | 0.8 | 0.8 | 1.000e-09 | PASS |
| redteam injection_casual_defended_banking | followed_injected_label | 0.234 | 0.234 | 1.000e-09 | PASS |
| redteam injection_casual_defended_banking | mean_confidence_when_following_injection | 0.660043 | 0.660043 | 1.000e-09 | PASS |
| redteam injection_casual_defended_banking | share_following_injection_with_confidence_ge_0.9 | 0.141026 | 0.141026 | 1.000e-09 | PASS |
| redteam injection_casual_defended_banking | picked_target_without_injection | 0.005 | 0.005 | 1.000e-09 | PASS |
| redteam injection_casual_defended_banking | ece | 0.12657 | 0.12657 | 0.001 | PASS |
| redteam injection_casual_defended_banking | accuracy_ci95 | [0.599975, 0.659] | [0.6, 0.659] | 0.005 | PASS |
| redteam injection_casual_defended_banking | calls.ok | 1,000 | 1,000 | 1.000e-09 | PASS |
| redteam injection_casual_defended_banking | calls.input_tokens | 1,724,923 | 1,724,923 | 1.000e-09 | PASS |
| redteam injection_casual_defended_banking | calls.median_latency_ms | 112.7 | 112.7 | 1.000e-09 | PASS |
| redteam injection_casual_defended_banking | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam latency_sequential_banking | calls | 100 | 100 | 1.000e-09 | PASS |
| redteam latency_sequential_banking | median_latency_ms | 99.0038 | 99.0038 | 1.000e-09 | PASS |
| redteam latency_sequential_banking | p90_latency_ms | 133.557 | 133.557 | 1.000e-09 | PASS |
| redteam latency_sequential_banking | calls.ok | 100 | 100 | 1.000e-09 | PASS |
| redteam latency_sequential_banking | calls.input_tokens | 168,878 | 168,878 | 1.000e-09 | PASS |
| redteam latency_sequential_banking | calls.median_latency_ms | 99.0038 | 99.0038 | 1.000e-09 | PASS |
| redteam latency_sequential_banking | calls.model_versions | ["jev-1.14.0"] | ["jev-1.14.0"] | exact | PASS |
| redteam cost | input_tokens | 46,156,750 | 46,156,750 | 1.000e-09 | PASS |
| redteam cost | usd | 1.93858 | 1.93858 | 1.000e-09 | PASS |
| redteam | mock flag | False | False | exact | PASS |

Extra recomputations, not in the reports but used above: spam confusion matrices, paired drop CIs, the repeat noise floor, Wilson intervals, Amazon renormalised vs raw expected stars, and ECE under exact / float32 / float64 edges are all in `runs/audit/audit_metrics.json` and `audit_extra*.json`.
