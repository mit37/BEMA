# Audit pass 3: regenerated results_summary.json, llm_report.json and the quiet BEMA timings

This pass checks the files as they stand: `runs/results_summary.json` (generated 2026-10-27T21:38:53Z), `runs/llm_report.json` (21:38:20Z), and `runs/report.json` / `runs/redteam_report.json` (21:38:30Z / 21:38:46Z). The code is at commit d7aa954. All code is in `runs/audit/pass3/`, adapted from the pass-2 scripts. I did not rerun any report command, changed nothing outside `runs/audit/pass3/` and this file, made no network calls, and did not re-time anything.

## Verdict

**PASS. All 3,061 values I recomputed match, with 0 FAILs.**

| check | values | result |
|---|---|---|
| 1. results_summary.json: banking77_typos, spam, slice250_all_models, injection, banking77_paired_vs_jev, speed_cost (+ spend, redteam_other) | 586 | 586/586 PASS |
| 2. llm_report.json vs my last-object parser, every model and test, including the new strict_json_reply_rate | 1269 | 1269/1269 PASS; 0 replies parse differently |
| Regression: BEMA and zero-shot values in report.json / redteam_report.json (pass-2 code), incl. `bema_latency` = timing_quiet.json | 1143 | 1143/1143 PASS |
| Regression: Jev rows after the pass-1 fixes | 63 | 63/63 PASS |
| 3. runs/bema/timing_quiet.json plausibility | | plausible, with 3 caveats (below) |

All pass-2 FAILs are fixed:
- The spend is now split by provider: OpenRouter $3.115 and Anthropic $110.575, which match my sums.
- Sonnet, Opus, Fable and zs_offshelf_bge_base now appear in every relevant section. No model with raw files is missing.
- The slice250 section now notes that BEMA's defended answers are its undefended ones. The predictions are identical, and the state text of the two input files is identical.
- llm_report now matches my parser on all rows. For example, Sonnet banking77_clean has 0 invalid outputs and accuracy 0.8458; Haiku banking77_clean ECE is 0.0857.

## 1. results_summary.json

Every accuracy_pct, n, ECE (4 dp), mean confidence and followed-injection value matches my own recomputation from raw files. So do the clean references, the majority baseline, Jev cost per 1,000 decisions, all latency medians and LLM cost per 1,000 calls, and the spend figures. The `redteam_other` block equals the current redteam_report.json. slice250 uses the identical 250 keys for every model and row.

**Spam duplicates, 83 vs 88.** Your 83 is an exact-text match, and I confirm it (exact match: 83). My pass-2 figure of 88 matched after lower-casing and collapsing whitespace. The 5 extra messages differ from a training message only by one trailing space, so they are duplicates in substance:
- idx 51: "YOU HAVE WON! As a valued Vodafone customer our computer has picked YO..."
- idx 143: "A å£400 XMAS REWARD IS WAITING FOR YOU! Our computer has randomly pick..."
- idx 399: "Eerie Nokia tones 4u, rply TONE TITLE to 8007 eg TONE DRACULA to 8007 ..."
- idx 484: "GENT! We are trying to contact you. Last weekends draw shows that you ..."
- idx 752: "If you don't, your prize will go to another customer. T&C at www.t-c.b..."

The definition barely matters. Excluding all 88 instead of 83 changes only Qwen at one decimal place (81.3% to 81.1%; n 753 to 748). The rest are unchanged: jev 96.8, bema_baseline 97.6, bema_v2 98.5, claude-haiku-5.5 94.3, qwen3.8-flash 81.1, claude-sonnet-5.5 97.2, claude-opus-5.5 99.7. I suggest the summary say 'exact text match', and optionally 'plus 5 that differ only by a trailing space'.

**banking77_paired_vs_jev.** Every n and every point gap (1 dp) matches. The CIs match my own 10,000-resample bootstrap within the 0.2-point tolerance; they differ by at most 0.1 point because the seeds differ:

| model | reported gap [CI] | audit gap [CI] |
|---|---|---|
| bema_baseline | +1.3 [-0.3, +2.9] | +1.27 [-0.39, +2.92] |
| bema_v2 | +1.8 [+0.2, +3.3] | +1.82 [+0.19, +3.44] |
| bema_v2_spell | +1.8 [+0.2, +3.3] | +1.82 [+0.19, +3.44] |
| claude-haiku-5.5 | -3.8 [-5.0, -2.6] | -3.80 [-5.00, -2.63] |
| qwen3.8-flash | -1.2 [-2.3, -0.1] | -1.23 [-2.37, -0.13] |
| claude-sonnet-5.5 | +4.0 [+3.0, +4.9] | +3.99 [+3.05, +4.94] |
| claude-opus-5.5 | +8.2 [+7.1, +9.3] | +8.21 [+7.08, +9.32] |

Not errors, but worth fixing:
1. The paired section leaves out the zero-shot models, which are the fairest comparison with Jev. From pass 2: zs_offshelf_bge_base -11.1 points [-12.6, -9.6], zs_from_clinc -17.5 [-19.1, -16.0], zs_offshelf -18.4 [-19.9, -16.8].
2. The `injection` section's `source` string still reads 'runs/llm (Haiku, Qwen)', but its rows now include Sonnet and Opus. The numbers are right; the label is stale.

## 2. llm_report.json

Using my pass-2 parser (last JSON object holding the key; exact 77-label match; confidence a number in [0, 1]) and my own item sets, all 1269 values match. They cover accuracy, invalid count and rate, n_with_confidence, mean stated confidence, ECE, overconfidence, followed_injected_label, latency median/p90, cost and cost per 1,000 calls, tokens, thinking tokens, providers, Qwen's token probability, and the new `strict_json_reply_rate`. The repo's new `parse()` and mine agree on every reply (0 disagreements).

`strict_json_reply_rate` means the whole reply parses as JSON. A ```json fence or any surrounding prose counts as not strict. On that definition the lowest rates are Opus banking77_mixed3 at 0.988 (37 of 3,080) and Haiku injection_casual_defended at 0.970. My recomputation matches.

## 3. runs/bema/timing_quiet.json: plausible, with caveats

I checked this against `measure_latency.py` (unchanged since commit e5af50f, 11:53 local) without re-timing anything:

| model | quiet median (p90) | busy timing.json median (p90) | tokenize (excluded) |
|---|---|---|---|
| bema_baseline | 1.70 ms (2.12) | 100.5 ms (387.8) | 0.11 ms |
| bema_v2 | 14.02 ms (19.74) | 12.39 ms (18.16) | 0.45 ms |
| bema_v2_spell | 14.07 ms (speller 0.054 ms) | | |

**Consistent with the code:**
- The protocol (first 300 clean BANKING77 queries, 20 warm-up passes, batch 1, tokenisation timed separately and excluded, all 3 heads, 2 torch threads on 2 cores) is what `measure_latency.py` does, and every field in the JSON is one the script writes.
- bema_v2_spell's median is computed as median(forward + speller): 14.069 ms vs 14.070 ms for bema_v2 plus the speller median, as expected.
- The magnitudes are plausible. v1 is a 3-layer, 96-dimension encoder fed 64 padded positions (about 24M MACs, at most). v2 is a 6-layer, 384-dimension MiniLM on about 13 tokens (about 139M MACs). The measured ratio is 8.3x and the implied throughput is about 10-14 GMAC/s on 2 cores. Both are reasonable for batch-1 CPU inference with per-op overhead.
- The quiet tails are tight (p90/median 1.25 and 1.41). The busy v1 figure (100 ms, p90/median 3.9) made the small model slower than the large one, which is impossible without contention. The quiet file resolves that.
- `report.json` `bema_latency` and the BEMA rows of results_summary `speed_cost` equal timing_quiet.json exactly. The contended timing.json is no longer quoted anywhere I checked.

**Caveats a skeptical reader could raise:**
1. **The machine was not provably idle.** The 1-minute load average at start was 1.71 on 2 cores, with 5- and 15-minute averages of 1.98 and 2.58. Training was terminated at 13:44:01 local and the file was written at 13:44:26. If training held about 2 runnable threads and the machine then went idle, a 1-minute load of 1.71 is what you would see about 9 s after the kill. So the reading is consistent with the job's lag, but it does not prove the machine was idle, and the load was not recorded at the end. The docstring's 'nothing else running' is plausible but unverified. My own audit scripts were not running in that window: my next heavy script wrote its output at 13:45:15, and it takes about 11 s. (That output file's mtime is now 14:35 because I re-ran it to time it.)
2. **bema_v2's quiet figure is slower than its busy one** (14.0 vs 12.4 ms, +1.6 ms). So contention only clearly inflated bema_v1. A possible cause is that the quiet loop runs the speller and tokeniser between each forward pass. Quote bema_v2 as roughly 12-14 ms, not 14.0 ms precisely.
3. **The speller was timed on clean text only.** In the first 300 clean queries the speller changes 4 texts, against 294 on swap2 and 297 on mixed3. Nearly every clean token takes the fast exact-lookup path, so the 54 µs speller cost (and the 14.07 ms total) is unmeasured on the typo'd inputs where the speller matters.

All BEMA latencies are single-run, local CPU forward passes that exclude tokenisation. Jev's and the LLMs' latencies include the network round trip. The speed_cost `source` field says this, and it needs to stay attached wherever the numbers are quoted.

## Appendix: every value checked

### results_summary.json (586/586 PASS)

| section | metric | reported | recomputed | result |
|---|---|---|---|---|
| banking77_typos | clean/jev n | 3,080 | 3,080 | PASS |
| banking77_typos | clean/jev accuracy_pct | 80.6 | 80.6 | PASS |
| banking77_typos | clean/jev ece (4 dp) | 0.0731 | 0.0731 | PASS |
| banking77_typos | clean/bema_baseline n | 3,080 | 3,080 | PASS |
| banking77_typos | clean/bema_baseline accuracy_pct | 81.9 | 81.9 | PASS |
| banking77_typos | clean/bema_baseline ece (4 dp) | 0.0399 | 0.0399 | PASS |
| banking77_typos | clean/bema_v2 n | 3,080 | 3,080 | PASS |
| banking77_typos | clean/bema_v2 accuracy_pct | 82.4 | 82.4 | PASS |
| banking77_typos | clean/bema_v2 ece (4 dp) | 0.025 | 0.025 | PASS |
| banking77_typos | clean/bema_v2_spell n | 3,080 | 3,080 | PASS |
| banking77_typos | clean/bema_v2_spell accuracy_pct | 82.4 | 82.4 | PASS |
| banking77_typos | clean/bema_v2_spell ece (4 dp) | 0.0245 | 0.0245 | PASS |
| banking77_typos | clean/zs_offshelf n | 3,080 | 3,080 | PASS |
| banking77_typos | clean/zs_offshelf accuracy_pct | 62.2 | 62.2 | PASS |
| banking77_typos | clean/zs_offshelf ece (4 dp) | 0.0433 | 0.0433 | PASS |
| banking77_typos | clean/zs_from_clinc n | 3,080 | 3,080 | PASS |
| banking77_typos | clean/zs_from_clinc accuracy_pct | 63.1 | 63.1 | PASS |
| banking77_typos | clean/zs_from_clinc ece (4 dp) | 0.0907 | 0.0907 | PASS |
| banking77_typos | clean/zs_offshelf_bge_base n | 3,080 | 3,080 | PASS |
| banking77_typos | clean/zs_offshelf_bge_base accuracy_pct | 69.5 | 69.5 | PASS |
| banking77_typos | clean/zs_offshelf_bge_base ece (4 dp) | 0.0254 | 0.0254 | PASS |
| banking77_typos | clean/claude-haiku-5.5 n | 3,080 | 3,080 | PASS |
| banking77_typos | clean/claude-haiku-5.5 accuracy_pct | 76.8 | 76.8 | PASS |
| banking77_typos | clean/claude-haiku-5.5 ece (4 dp) | 0.0857 | 0.0857 | PASS |
| banking77_typos | clean/qwen3.8-flash n | 3,080 | 3,080 | PASS |
| banking77_typos | clean/qwen3.8-flash accuracy_pct | 79.4 | 79.4 | PASS |
| banking77_typos | clean/qwen3.8-flash ece (4 dp) | 0.1553 | 0.1553 | PASS |
| banking77_typos | clean/claude-sonnet-5.5 n | 3,080 | 3,080 | PASS |
| banking77_typos | clean/claude-sonnet-5.5 accuracy_pct | 84.6 | 84.6 | PASS |
| banking77_typos | clean/claude-sonnet-5.5 ece (4 dp) | 0.0676 | 0.0676 | PASS |
| banking77_typos | clean/claude-opus-5.5 n | 3,080 | 3,080 | PASS |
| banking77_typos | clean/claude-opus-5.5 accuracy_pct | 88.8 | 88.8 | PASS |
| banking77_typos | clean/claude-opus-5.5 ece (4 dp) | 0.093 | 0.093 | PASS |
| banking77_typos | swap2/jev n | 3,080 | 3,080 | PASS |
| banking77_typos | swap2/jev accuracy_pct | 79 | 79 | PASS |
| banking77_typos | swap2/jev ece (4 dp) | 0.0672 | 0.0672 | PASS |
| banking77_typos | swap2/bema_baseline n | 3,080 | 3,080 | PASS |
| banking77_typos | swap2/bema_baseline accuracy_pct | 50.8 | 50.8 | PASS |
| banking77_typos | swap2/bema_baseline ece (4 dp) | 0.1027 | 0.1027 | PASS |
| banking77_typos | swap2/bema_v2 n | 3,080 | 3,080 | PASS |
| banking77_typos | swap2/bema_v2 accuracy_pct | 67.3 | 67.3 | PASS |
| banking77_typos | swap2/bema_v2 ece (4 dp) | 0.066 | 0.066 | PASS |
| banking77_typos | swap2/bema_v2_spell n | 3,080 | 3,080 | PASS |
| banking77_typos | swap2/bema_v2_spell accuracy_pct | 79.4 | 79.4 | PASS |
| banking77_typos | swap2/bema_v2_spell ece (4 dp) | 0.0283 | 0.0283 | PASS |
| banking77_typos | swap2/zs_offshelf n | 3,080 | 3,080 | PASS |
| banking77_typos | swap2/zs_offshelf accuracy_pct | 47.3 | 47.3 | PASS |
| banking77_typos | swap2/zs_offshelf ece (4 dp) | 0.0287 | 0.0287 | PASS |
| banking77_typos | swap2/zs_from_clinc n | 3,080 | 3,080 | PASS |
| banking77_typos | swap2/zs_from_clinc accuracy_pct | 47.4 | 47.4 | PASS |
| banking77_typos | swap2/zs_from_clinc ece (4 dp) | 0.1147 | 0.1147 | PASS |
| banking77_typos | swap2/zs_offshelf_bge_base n | 3,080 | 3,080 | PASS |
| banking77_typos | swap2/zs_offshelf_bge_base accuracy_pct | 54.4 | 54.4 | PASS |
| banking77_typos | swap2/zs_offshelf_bge_base ece (4 dp) | 0.021 | 0.021 | PASS |
| banking77_typos | swap2/claude-haiku-5.5 n | 3,080 | 3,080 | PASS |
| banking77_typos | swap2/claude-haiku-5.5 accuracy_pct | 74.7 | 74.7 | PASS |
| banking77_typos | swap2/claude-haiku-5.5 ece (4 dp) | 0.1028 | 0.1028 | PASS |
| banking77_typos | swap2/qwen3.8-flash n | 3,080 | 3,080 | PASS |
| banking77_typos | swap2/qwen3.8-flash accuracy_pct | 77.9 | 77.9 | PASS |
| banking77_typos | swap2/qwen3.8-flash ece (4 dp) | 0.1672 | 0.1672 | PASS |
| banking77_typos | swap2/claude-sonnet-5.5 n | 3,080 | 3,080 | PASS |
| banking77_typos | swap2/claude-sonnet-5.5 accuracy_pct | 84.4 | 84.4 | PASS |
| banking77_typos | swap2/claude-sonnet-5.5 ece (4 dp) | 0.0629 | 0.0629 | PASS |
| banking77_typos | swap2/claude-opus-5.5 n | 3,080 | 3,080 | PASS |
| banking77_typos | swap2/claude-opus-5.5 accuracy_pct | 88.8 | 88.8 | PASS |
| banking77_typos | swap2/claude-opus-5.5 ece (4 dp) | 0.092 | 0.092 | PASS |
| banking77_typos | swap4/jev n | 3,080 | 3,080 | PASS |
| banking77_typos | swap4/jev accuracy_pct | 77 | 77 | PASS |
| banking77_typos | swap4/jev ece (4 dp) | 0.0663 | 0.0663 | PASS |
| banking77_typos | swap4/bema_baseline n | 3,080 | 3,080 | PASS |
| banking77_typos | swap4/bema_baseline accuracy_pct | 32.7 | 32.7 | PASS |
| banking77_typos | swap4/bema_baseline ece (4 dp) | 0.2005 | 0.2005 | PASS |
| banking77_typos | swap4/bema_v2 n | 3,080 | 3,080 | PASS |
| banking77_typos | swap4/bema_v2 accuracy_pct | 53.6 | 53.6 | PASS |
| banking77_typos | swap4/bema_v2 ece (4 dp) | 0.1081 | 0.1081 | PASS |
| banking77_typos | swap4/bema_v2_spell n | 3,080 | 3,080 | PASS |
| banking77_typos | swap4/bema_v2_spell accuracy_pct | 74.6 | 74.6 | PASS |
| banking77_typos | swap4/bema_v2_spell ece (4 dp) | 0.0463 | 0.0463 | PASS |
| banking77_typos | swap4/zs_offshelf n | 3,080 | 3,080 | PASS |
| banking77_typos | swap4/zs_offshelf accuracy_pct | 36.4 | 36.4 | PASS |
| banking77_typos | swap4/zs_offshelf ece (4 dp) | 0.0301 | 0.0301 | PASS |
| banking77_typos | swap4/zs_from_clinc n | 3,080 | 3,080 | PASS |
| banking77_typos | swap4/zs_from_clinc accuracy_pct | 35.4 | 35.4 | PASS |
| banking77_typos | swap4/zs_from_clinc ece (4 dp) | 0.1434 | 0.1434 | PASS |
| banking77_typos | keyboard2/jev n | 3,080 | 3,080 | PASS |
| banking77_typos | keyboard2/jev accuracy_pct | 77.5 | 77.5 | PASS |
| banking77_typos | keyboard2/jev ece (4 dp) | 0.068 | 0.068 | PASS |
| banking77_typos | keyboard2/bema_baseline n | 3,080 | 3,080 | PASS |
| banking77_typos | keyboard2/bema_baseline accuracy_pct | 53.2 | 53.2 | PASS |
| banking77_typos | keyboard2/bema_baseline ece (4 dp) | 0.087 | 0.087 | PASS |
| banking77_typos | keyboard2/bema_v2 n | 3,080 | 3,080 | PASS |
| banking77_typos | keyboard2/bema_v2 accuracy_pct | 69.1 | 69.1 | PASS |
| banking77_typos | keyboard2/bema_v2 ece (4 dp) | 0.0505 | 0.0505 | PASS |
| banking77_typos | keyboard2/bema_v2_spell n | 3,080 | 3,080 | PASS |
| banking77_typos | keyboard2/bema_v2_spell accuracy_pct | 80.4 | 80.4 | PASS |
| banking77_typos | keyboard2/bema_v2_spell ece (4 dp) | 0.0297 | 0.0297 | PASS |
| banking77_typos | keyboard2/zs_offshelf n | 3,080 | 3,080 | PASS |
| banking77_typos | keyboard2/zs_offshelf accuracy_pct | 48.7 | 48.7 | PASS |
| banking77_typos | keyboard2/zs_offshelf ece (4 dp) | 0.0329 | 0.0329 | PASS |
| banking77_typos | keyboard2/zs_from_clinc n | 3,080 | 3,080 | PASS |
| banking77_typos | keyboard2/zs_from_clinc accuracy_pct | 47 | 47 | PASS |
| banking77_typos | keyboard2/zs_from_clinc ece (4 dp) | 0.1278 | 0.1278 | PASS |
| banking77_typos | delete2/jev n | 3,080 | 3,080 | PASS |
| banking77_typos | delete2/jev accuracy_pct | 78.7 | 78.7 | PASS |
| banking77_typos | delete2/jev ece (4 dp) | 0.0649 | 0.0649 | PASS |
| banking77_typos | delete2/bema_baseline n | 3,080 | 3,080 | PASS |
| banking77_typos | delete2/bema_baseline accuracy_pct | 55.1 | 55.1 | PASS |
| banking77_typos | delete2/bema_baseline ece (4 dp) | 0.09 | 0.09 | PASS |
| banking77_typos | delete2/bema_v2 n | 3,080 | 3,080 | PASS |
| banking77_typos | delete2/bema_v2 accuracy_pct | 70.1 | 70.1 | PASS |
| banking77_typos | delete2/bema_v2 ece (4 dp) | 0.0525 | 0.0525 | PASS |
| banking77_typos | delete2/bema_v2_spell n | 3,080 | 3,080 | PASS |
| banking77_typos | delete2/bema_v2_spell accuracy_pct | 76.8 | 76.8 | PASS |
| banking77_typos | delete2/bema_v2_spell ece (4 dp) | 0.0444 | 0.0444 | PASS |
| banking77_typos | delete2/zs_offshelf n | 3,080 | 3,080 | PASS |
| banking77_typos | delete2/zs_offshelf accuracy_pct | 49.3 | 49.3 | PASS |
| banking77_typos | delete2/zs_offshelf ece (4 dp) | 0.037 | 0.037 | PASS |
| banking77_typos | delete2/zs_from_clinc n | 3,080 | 3,080 | PASS |
| banking77_typos | delete2/zs_from_clinc accuracy_pct | 49.1 | 49.1 | PASS |
| banking77_typos | delete2/zs_from_clinc ece (4 dp) | 0.1209 | 0.1209 | PASS |
| banking77_typos | insert2/jev n | 3,080 | 3,080 | PASS |
| banking77_typos | insert2/jev accuracy_pct | 79.8 | 79.8 | PASS |
| banking77_typos | insert2/jev ece (4 dp) | 0.0689 | 0.0689 | PASS |
| banking77_typos | insert2/bema_baseline n | 3,080 | 3,080 | PASS |
| banking77_typos | insert2/bema_baseline accuracy_pct | 58.7 | 58.7 | PASS |
| banking77_typos | insert2/bema_baseline ece (4 dp) | 0.0652 | 0.0652 | PASS |
| banking77_typos | insert2/bema_v2 n | 3,080 | 3,080 | PASS |
| banking77_typos | insert2/bema_v2 accuracy_pct | 72.1 | 72.1 | PASS |
| banking77_typos | insert2/bema_v2 ece (4 dp) | 0.0484 | 0.0484 | PASS |
| banking77_typos | insert2/bema_v2_spell n | 3,080 | 3,080 | PASS |
| banking77_typos | insert2/bema_v2_spell accuracy_pct | 82.1 | 82.1 | PASS |
| banking77_typos | insert2/bema_v2_spell ece (4 dp) | 0.0248 | 0.0248 | PASS |
| banking77_typos | insert2/zs_offshelf n | 3,080 | 3,080 | PASS |
| banking77_typos | insert2/zs_offshelf accuracy_pct | 53.1 | 53.1 | PASS |
| banking77_typos | insert2/zs_offshelf ece (4 dp) | 0.0168 | 0.0168 | PASS |
| banking77_typos | insert2/zs_from_clinc n | 3,080 | 3,080 | PASS |
| banking77_typos | insert2/zs_from_clinc accuracy_pct | 51.9 | 51.9 | PASS |
| banking77_typos | insert2/zs_from_clinc ece (4 dp) | 0.1079 | 0.1079 | PASS |
| banking77_typos | mixed3/jev n | 3,080 | 3,080 | PASS |
| banking77_typos | mixed3/jev accuracy_pct | 77.4 | 77.4 | PASS |
| banking77_typos | mixed3/jev ece (4 dp) | 0.0636 | 0.0636 | PASS |
| banking77_typos | mixed3/bema_baseline n | 3,080 | 3,080 | PASS |
| banking77_typos | mixed3/bema_baseline accuracy_pct | 44.7 | 44.7 | PASS |
| banking77_typos | mixed3/bema_baseline ece (4 dp) | 0.1365 | 0.1365 | PASS |
| banking77_typos | mixed3/bema_v2 n | 3,080 | 3,080 | PASS |
| banking77_typos | mixed3/bema_v2 accuracy_pct | 64.4 | 64.4 | PASS |
| banking77_typos | mixed3/bema_v2 ece (4 dp) | 0.0678 | 0.0678 | PASS |
| banking77_typos | mixed3/bema_v2_spell n | 3,080 | 3,080 | PASS |
| banking77_typos | mixed3/bema_v2_spell accuracy_pct | 78.7 | 78.7 | PASS |
| banking77_typos | mixed3/bema_v2_spell ece (4 dp) | 0.0295 | 0.0295 | PASS |
| banking77_typos | mixed3/zs_offshelf n | 3,080 | 3,080 | PASS |
| banking77_typos | mixed3/zs_offshelf accuracy_pct | 43.7 | 43.7 | PASS |
| banking77_typos | mixed3/zs_offshelf ece (4 dp) | 0.0319 | 0.0319 | PASS |
| banking77_typos | mixed3/zs_from_clinc n | 3,080 | 3,080 | PASS |
| banking77_typos | mixed3/zs_from_clinc accuracy_pct | 44.2 | 44.2 | PASS |
| banking77_typos | mixed3/zs_from_clinc ece (4 dp) | 0.1234 | 0.1234 | PASS |
| banking77_typos | mixed3/claude-haiku-5.5 n | 3,080 | 3,080 | PASS |
| banking77_typos | mixed3/claude-haiku-5.5 accuracy_pct | 72.6 | 72.6 | PASS |
| banking77_typos | mixed3/claude-haiku-5.5 ece (4 dp) | 0.1071 | 0.1071 | PASS |
| banking77_typos | mixed3/qwen3.8-flash n | 3,080 | 3,080 | PASS |
| banking77_typos | mixed3/qwen3.8-flash accuracy_pct | 76.9 | 76.9 | PASS |
| banking77_typos | mixed3/qwen3.8-flash ece (4 dp) | 0.1738 | 0.1738 | PASS |
| banking77_typos | mixed3/claude-sonnet-5.5 n | 3,080 | 3,080 | PASS |
| banking77_typos | mixed3/claude-sonnet-5.5 accuracy_pct | 83.4 | 83.4 | PASS |
| banking77_typos | mixed3/claude-sonnet-5.5 ece (4 dp) | 0.0589 | 0.0589 | PASS |
| banking77_typos | mixed3/claude-opus-5.5 n | 3,080 | 3,080 | PASS |
| banking77_typos | mixed3/claude-opus-5.5 accuracy_pct | 88.3 | 88.3 | PASS |
| banking77_typos | mixed3/claude-opus-5.5 ece (4 dp) | 0.0981 | 0.0981 | PASS |
| spam | clean/jev n | 836 | 836 | PASS |
| spam | clean/jev accuracy_pct | 96.1 | 96.1 | PASS |
| spam | clean/jev ece (4 dp) | 0.0658 | 0.0658 | PASS |
| spam | clean/bema_baseline n | 836 | 836 | PASS |
| spam | clean/bema_baseline accuracy_pct | 97.8 | 97.8 | PASS |
| spam | clean/bema_baseline ece (4 dp) | 0.006 | 0.006 | PASS |
| spam | clean/bema_v2 n | 836 | 836 | PASS |
| spam | clean/bema_v2 accuracy_pct | 98.7 | 98.7 | PASS |
| spam | clean/bema_v2 ece (4 dp) | 0.0052 | 0.0052 | PASS |
| spam | clean/bema_v2_spell n | 836 | 836 | PASS |
| spam | clean/bema_v2_spell accuracy_pct | 98.7 | 98.7 | PASS |
| spam | clean/bema_v2_spell ece (4 dp) | 0.0071 | 0.0071 | PASS |
| spam | clean/claude-haiku-5.5 n | 836 | 836 | PASS |
| spam | clean/claude-haiku-5.5 accuracy_pct | 94 | 94 | PASS |
| spam | clean/claude-haiku-5.5 ece (4 dp) | 0.035 | 0.035 | PASS |
| spam | clean/qwen3.8-flash n | 836 | 836 | PASS |
| spam | clean/qwen3.8-flash accuracy_pct | 79.9 | 79.9 | PASS |
| spam | clean/qwen3.8-flash ece (4 dp) | 0.1582 | 0.1582 | PASS |
| spam | clean/claude-sonnet-5.5 n | 836 | 836 | PASS |
| spam | clean/claude-sonnet-5.5 accuracy_pct | 96.8 | 96.8 | PASS |
| spam | clean/claude-sonnet-5.5 ece (4 dp) | 0.0397 | 0.0397 | PASS |
| spam | clean/claude-opus-5.5 n | 836 | 836 | PASS |
| spam | clean/claude-opus-5.5 accuracy_pct | 99.6 | 99.6 | PASS |
| spam | clean/claude-opus-5.5 ece (4 dp) | 0.0323 | 0.0323 | PASS |
| spam | swap2/jev n | 836 | 836 | PASS |
| spam | swap2/jev accuracy_pct | 95.6 | 95.6 | PASS |
| spam | swap2/jev ece (4 dp) | 0.0734 | 0.0734 | PASS |
| spam | swap2/bema_baseline n | 836 | 836 | PASS |
| spam | swap2/bema_baseline accuracy_pct | 97.2 | 97.2 | PASS |
| spam | swap2/bema_baseline ece (4 dp) | 0.0066 | 0.0066 | PASS |
| spam | swap2/bema_v2 n | 836 | 836 | PASS |
| spam | swap2/bema_v2 accuracy_pct | 98.4 | 98.4 | PASS |
| spam | swap2/bema_v2 ece (4 dp) | 0.0059 | 0.0059 | PASS |
| spam | swap2/bema_v2_spell n | 836 | 836 | PASS |
| spam | swap2/bema_v2_spell accuracy_pct | 98.6 | 98.6 | PASS |
| spam | swap2/bema_v2_spell ece (4 dp) | 0.0068 | 0.0068 | PASS |
| spam | swap2/claude-haiku-5.5 n | 836 | 836 | PASS |
| spam | swap2/claude-haiku-5.5 accuracy_pct | 91.7 | 91.7 | PASS |
| spam | swap2/claude-haiku-5.5 ece (4 dp) | 0.056 | 0.056 | PASS |
| spam | swap2/qwen3.8-flash n | 836 | 836 | PASS |
| spam | swap2/qwen3.8-flash accuracy_pct | 67.9 | 67.9 | PASS |
| spam | swap2/qwen3.8-flash ece (4 dp) | 0.3 | 0.3 | PASS |
| spam | swap2/claude-sonnet-5.5 n | 836 | 836 | PASS |
| spam | swap2/claude-sonnet-5.5 accuracy_pct | 96.1 | 96.1 | PASS |
| spam | swap2/claude-sonnet-5.5 ece (4 dp) | 0.0326 | 0.0326 | PASS |
| spam | swap2/claude-opus-5.5 n | 836 | 836 | PASS |
| spam | swap2/claude-opus-5.5 accuracy_pct | 99.6 | 99.6 | PASS |
| spam | swap2/claude-opus-5.5 ece (4 dp) | 0.0356 | 0.0356 | PASS |
| spam | keyboard2/jev n | 836 | 836 | PASS |
| spam | keyboard2/jev accuracy_pct | 95.5 | 95.5 | PASS |
| spam | keyboard2/jev ece (4 dp) | 0.0746 | 0.0746 | PASS |
| spam | keyboard2/bema_baseline n | 836 | 836 | PASS |
| spam | keyboard2/bema_baseline accuracy_pct | 97.1 | 97.1 | PASS |
| spam | keyboard2/bema_baseline ece (4 dp) | 0.0067 | 0.0067 | PASS |
| spam | keyboard2/bema_v2 n | 836 | 836 | PASS |
| spam | keyboard2/bema_v2 accuracy_pct | 98.4 | 98.4 | PASS |
| spam | keyboard2/bema_v2 ece (4 dp) | 0.0081 | 0.0081 | PASS |
| spam | keyboard2/bema_v2_spell n | 836 | 836 | PASS |
| spam | keyboard2/bema_v2_spell accuracy_pct | 98.6 | 98.6 | PASS |
| spam | keyboard2/bema_v2_spell ece (4 dp) | 0.0054 | 0.0054 | PASS |
| spam | mixed3/jev n | 836 | 836 | PASS |
| spam | mixed3/jev accuracy_pct | 95.3 | 95.3 | PASS |
| spam | mixed3/jev ece (4 dp) | 0.075 | 0.075 | PASS |
| spam | mixed3/bema_baseline n | 836 | 836 | PASS |
| spam | mixed3/bema_baseline accuracy_pct | 97.6 | 97.6 | PASS |
| spam | mixed3/bema_baseline ece (4 dp) | 0.0075 | 0.0075 | PASS |
| spam | mixed3/bema_v2 n | 836 | 836 | PASS |
| spam | mixed3/bema_v2 accuracy_pct | 98.3 | 98.3 | PASS |
| spam | mixed3/bema_v2 ece (4 dp) | 0.004 | 0.004 | PASS |
| spam | mixed3/bema_v2_spell n | 836 | 836 | PASS |
| spam | mixed3/bema_v2_spell accuracy_pct | 98.7 | 98.7 | PASS |
| spam | mixed3/bema_v2_spell ece (4 dp) | 0.006 | 0.006 | PASS |
| spam | majority_baseline_pct | 86.1 | 86.1 | PASS |
| slice250 | n_keys | 250 | 250 | PASS |
| slice250 | clean/jev n | 250 | 250 | PASS |
| slice250 | clean/jev accuracy_pct | 77.2 | 77.2 | PASS |
| slice250 | clean/bema_v2 n | 250 | 250 | PASS |
| slice250 | clean/bema_v2 accuracy_pct | 80 | 80 | PASS |
| slice250 | clean/bema_v2_spell n | 250 | 250 | PASS |
| slice250 | clean/bema_v2_spell accuracy_pct | 80 | 80 | PASS |
| slice250 | clean/claude-haiku-5.5 n | 250 | 250 | PASS |
| slice250 | clean/claude-haiku-5.5 accuracy_pct | 72 | 72 | PASS |
| slice250 | clean/qwen3.8-flash n | 250 | 250 | PASS |
| slice250 | clean/qwen3.8-flash accuracy_pct | 79.6 | 79.6 | PASS |
| slice250 | clean/gpt-6-luna n | 250 | 250 | PASS |
| slice250 | clean/gpt-6-luna accuracy_pct | 75.6 | 75.6 | PASS |
| slice250 | clean/gpt-6.1-terra n | 250 | 250 | PASS |
| slice250 | clean/gpt-6.1-terra accuracy_pct | 81.6 | 81.6 | PASS |
| slice250 | clean/claude-sonnet-5.5 n | 250 | 250 | PASS |
| slice250 | clean/claude-sonnet-5.5 accuracy_pct | 84 | 84 | PASS |
| slice250 | clean/claude-opus-5.5 n | 250 | 250 | PASS |
| slice250 | clean/claude-opus-5.5 accuracy_pct | 88.4 | 88.4 | PASS |
| slice250 | clean/claude-fable-5.1 n | 250 | 250 | PASS |
| slice250 | clean/claude-fable-5.1 accuracy_pct | 84 | 84 | PASS |
| slice250 | swap2/jev n | 250 | 250 | PASS |
| slice250 | swap2/jev accuracy_pct | 76.4 | 76.4 | PASS |
| slice250 | swap2/bema_v2 n | 250 | 250 | PASS |
| slice250 | swap2/bema_v2 accuracy_pct | 64.4 | 64.4 | PASS |
| slice250 | swap2/bema_v2_spell n | 250 | 250 | PASS |
| slice250 | swap2/bema_v2_spell accuracy_pct | 76 | 76 | PASS |
| slice250 | swap2/claude-haiku-5.5 n | 250 | 250 | PASS |
| slice250 | swap2/claude-haiku-5.5 accuracy_pct | 69.2 | 69.2 | PASS |
| slice250 | swap2/qwen3.8-flash n | 250 | 250 | PASS |
| slice250 | swap2/qwen3.8-flash accuracy_pct | 77.2 | 77.2 | PASS |
| slice250 | swap2/gpt-6-luna n | 250 | 250 | PASS |
| slice250 | swap2/gpt-6-luna accuracy_pct | 74.8 | 74.8 | PASS |
| slice250 | swap2/gpt-6.1-terra n | 250 | 250 | PASS |
| slice250 | swap2/gpt-6.1-terra accuracy_pct | 81.6 | 81.6 | PASS |
| slice250 | swap2/claude-sonnet-5.5 n | 250 | 250 | PASS |
| slice250 | swap2/claude-sonnet-5.5 accuracy_pct | 84.4 | 84.4 | PASS |
| slice250 | swap2/claude-opus-5.5 n | 250 | 250 | PASS |
| slice250 | swap2/claude-opus-5.5 accuracy_pct | 88 | 88 | PASS |
| slice250 | swap2/claude-fable-5.1 n | 250 | 250 | PASS |
| slice250 | swap2/claude-fable-5.1 accuracy_pct | 84 | 84 | PASS |
| slice250 | injection_explicit/jev n | 250 | 250 | PASS |
| slice250 | injection_explicit/jev accuracy_pct | 42 | 42 | PASS |
| slice250 | injection_explicit/jev followed_injection_pct | 51.6 | 51.6 | PASS |
| slice250 | injection_explicit/bema_v2 n | 250 | 250 | PASS |
| slice250 | injection_explicit/bema_v2 accuracy_pct | 42.8 | 42.8 | PASS |
| slice250 | injection_explicit/bema_v2 followed_injection_pct | 25.6 | 25.6 | PASS |
| slice250 | injection_explicit/claude-haiku-5.5 n | 250 | 250 | PASS |
| slice250 | injection_explicit/claude-haiku-5.5 accuracy_pct | 56.4 | 56.4 | PASS |
| slice250 | injection_explicit/claude-haiku-5.5 followed_injection_pct | 27.2 | 27.2 | PASS |
| slice250 | injection_explicit/qwen3.8-flash n | 250 | 250 | PASS |
| slice250 | injection_explicit/qwen3.8-flash accuracy_pct | 22.8 | 22.8 | PASS |
| slice250 | injection_explicit/qwen3.8-flash followed_injection_pct | 72.8 | 72.8 | PASS |
| slice250 | injection_explicit/gpt-6-luna n | 250 | 250 | PASS |
| slice250 | injection_explicit/gpt-6-luna accuracy_pct | 1.2 | 1.2 | PASS |
| slice250 | injection_explicit/gpt-6-luna followed_injection_pct | 97.6 | 97.6 | PASS |
| slice250 | injection_explicit/gpt-6.1-terra n | 250 | 250 | PASS |
| slice250 | injection_explicit/gpt-6.1-terra accuracy_pct | 81.2 | 81.2 | PASS |
| slice250 | injection_explicit/gpt-6.1-terra followed_injection_pct | 1.2 | 1.2 | PASS |
| slice250 | injection_explicit/claude-sonnet-5.5 n | 250 | 250 | PASS |
| slice250 | injection_explicit/claude-sonnet-5.5 accuracy_pct | 81.2 | 81.2 | PASS |
| slice250 | injection_explicit/claude-sonnet-5.5 followed_injection_pct | 2.8 | 2.8 | PASS |
| slice250 | injection_explicit/claude-opus-5.5 n | 250 | 250 | PASS |
| slice250 | injection_explicit/claude-opus-5.5 accuracy_pct | 80.8 | 80.8 | PASS |
| slice250 | injection_explicit/claude-opus-5.5 followed_injection_pct | 7.6 | 7.6 | PASS |
| slice250 | injection_explicit/claude-fable-5.1 n | 250 | 250 | PASS |
| slice250 | injection_explicit/claude-fable-5.1 accuracy_pct | 82 | 82 | PASS |
| slice250 | injection_explicit/claude-fable-5.1 followed_injection_pct | 0.4 | 0.4 | PASS |
| slice250 | injection_control/jev n | 250 | 250 | PASS |
| slice250 | injection_control/jev accuracy_pct | 76.8 | 76.8 | PASS |
| slice250 | injection_control/bema_v2 n | 250 | 250 | PASS |
| slice250 | injection_control/bema_v2 accuracy_pct | 66.4 | 66.4 | PASS |
| slice250 | injection_control/claude-haiku-5.5 n | 250 | 250 | PASS |
| slice250 | injection_control/claude-haiku-5.5 accuracy_pct | 72.8 | 72.8 | PASS |
| slice250 | injection_control/qwen3.8-flash n | 250 | 250 | PASS |
| slice250 | injection_control/qwen3.8-flash accuracy_pct | 79.2 | 79.2 | PASS |
| slice250 | injection_control/gpt-6-luna n | 250 | 250 | PASS |
| slice250 | injection_control/gpt-6-luna accuracy_pct | 75.6 | 75.6 | PASS |
| slice250 | injection_control/gpt-6.1-terra n | 250 | 250 | PASS |
| slice250 | injection_control/gpt-6.1-terra accuracy_pct | 82 | 82 | PASS |
| slice250 | injection_control/claude-sonnet-5.5 n | 250 | 250 | PASS |
| slice250 | injection_control/claude-sonnet-5.5 accuracy_pct | 82.8 | 82.8 | PASS |
| slice250 | injection_control/claude-opus-5.5 n | 250 | 250 | PASS |
| slice250 | injection_control/claude-opus-5.5 accuracy_pct | 88 | 88 | PASS |
| slice250 | injection_control/claude-fable-5.1 n | 250 | 250 | PASS |
| slice250 | injection_control/claude-fable-5.1 accuracy_pct | 84 | 84 | PASS |
| slice250 | injection_casual/jev n | 250 | 250 | PASS |
| slice250 | injection_casual/jev accuracy_pct | 27.6 | 27.6 | PASS |
| slice250 | injection_casual/jev followed_injection_pct | 68.4 | 68.4 | PASS |
| slice250 | injection_casual/bema_v2 n | 250 | 250 | PASS |
| slice250 | injection_casual/bema_v2 accuracy_pct | 45.2 | 45.2 | PASS |
| slice250 | injection_casual/bema_v2 followed_injection_pct | 28.8 | 28.8 | PASS |
| slice250 | injection_casual/claude-haiku-5.5 n | 250 | 250 | PASS |
| slice250 | injection_casual/claude-haiku-5.5 accuracy_pct | 61.6 | 61.6 | PASS |
| slice250 | injection_casual/claude-haiku-5.5 followed_injection_pct | 18 | 18 | PASS |
| slice250 | injection_casual/qwen3.8-flash n | 250 | 250 | PASS |
| slice250 | injection_casual/qwen3.8-flash accuracy_pct | 56 | 56 | PASS |
| slice250 | injection_casual/qwen3.8-flash followed_injection_pct | 33.2 | 33.2 | PASS |
| slice250 | injection_casual/gpt-6-luna n | 250 | 250 | PASS |
| slice250 | injection_casual/gpt-6-luna accuracy_pct | 48 | 48 | PASS |
| slice250 | injection_casual/gpt-6-luna followed_injection_pct | 40.4 | 40.4 | PASS |
| slice250 | injection_casual/gpt-6.1-terra n | 250 | 250 | PASS |
| slice250 | injection_casual/gpt-6.1-terra accuracy_pct | 67.2 | 67.2 | PASS |
| slice250 | injection_casual/gpt-6.1-terra followed_injection_pct | 20 | 20 | PASS |
| slice250 | injection_casual/claude-sonnet-5.5 n | 250 | 250 | PASS |
| slice250 | injection_casual/claude-sonnet-5.5 accuracy_pct | 62.4 | 62.4 | PASS |
| slice250 | injection_casual/claude-sonnet-5.5 followed_injection_pct | 30.8 | 30.8 | PASS |
| slice250 | injection_casual/claude-opus-5.5 n | 250 | 250 | PASS |
| slice250 | injection_casual/claude-opus-5.5 accuracy_pct | 84.4 | 84.4 | PASS |
| slice250 | injection_casual/claude-opus-5.5 followed_injection_pct | 4 | 4 | PASS |
| slice250 | injection_casual/claude-fable-5.1 n | 250 | 250 | PASS |
| slice250 | injection_casual/claude-fable-5.1 accuracy_pct | 82.8 | 82.8 | PASS |
| slice250 | injection_casual/claude-fable-5.1 followed_injection_pct | 2.8 | 2.8 | PASS |
| slice250 | injection_casual_defended/jev n | 250 | 250 | PASS |
| slice250 | injection_casual_defended/jev accuracy_pct | 57.2 | 57.2 | PASS |
| slice250 | injection_casual_defended/jev followed_injection_pct | 27.6 | 27.6 | PASS |
| slice250 | injection_casual_defended/bema_v2 n | 250 | 250 | PASS |
| slice250 | injection_casual_defended/bema_v2 accuracy_pct | 45.2 | 45.2 | PASS |
| slice250 | injection_casual_defended/bema_v2 followed_injection_pct | 28.8 | 28.8 | PASS |
| slice250 | injection_casual_defended/claude-haiku-5.5 n | 250 | 250 | PASS |
| slice250 | injection_casual_defended/claude-haiku-5.5 accuracy_pct | 68 | 68 | PASS |
| slice250 | injection_casual_defended/claude-haiku-5.5 followed_injection_pct | 8.8 | 8.8 | PASS |
| slice250 | injection_casual_defended/qwen3.8-flash n | 250 | 250 | PASS |
| slice250 | injection_casual_defended/qwen3.8-flash accuracy_pct | 66.8 | 66.8 | PASS |
| slice250 | injection_casual_defended/qwen3.8-flash followed_injection_pct | 16 | 16 | PASS |
| slice250 | injection_casual_defended/gpt-6-luna n | 250 | 250 | PASS |
| slice250 | injection_casual_defended/gpt-6-luna accuracy_pct | 68 | 68 | PASS |
| slice250 | injection_casual_defended/gpt-6-luna followed_injection_pct | 11.6 | 11.6 | PASS |
| slice250 | injection_casual_defended/gpt-6.1-terra n | 250 | 250 | PASS |
| slice250 | injection_casual_defended/gpt-6.1-terra accuracy_pct | 80.4 | 80.4 | PASS |
| slice250 | injection_casual_defended/gpt-6.1-terra followed_injection_pct | 1.2 | 1.2 | PASS |
| slice250 | injection_casual_defended/claude-sonnet-5.5 n | 250 | 250 | PASS |
| slice250 | injection_casual_defended/claude-sonnet-5.5 accuracy_pct | 80.4 | 80.4 | PASS |
| slice250 | injection_casual_defended/claude-sonnet-5.5 followed_injection_pct | 2.8 | 2.8 | PASS |
| slice250 | injection_casual_defended/claude-opus-5.5 n | 250 | 250 | PASS |
| slice250 | injection_casual_defended/claude-opus-5.5 accuracy_pct | 89.2 | 89.2 | PASS |
| slice250 | injection_casual_defended/claude-opus-5.5 followed_injection_pct | 0.4 | 0.4 | PASS |
| slice250 | injection_casual_defended/claude-fable-5.1 n | 250 | 250 | PASS |
| slice250 | injection_casual_defended/claude-fable-5.1 accuracy_pct | 82.8 | 82.8 | PASS |
| slice250 | injection_casual_defended/claude-fable-5.1 followed_injection_pct | 0.8 | 0.8 | PASS |
| slice250 | note labels BEMA defended reuse | True | True | PASS |
| injection | explicit_note_end/jev n | 1,000 | 1,000 | PASS |
| injection | explicit_note_end/jev accuracy_pct | 47.4 | 47.4 | PASS |
| injection | explicit_note_end/jev mean_confidence (3 dp) | 0.707 | 0.707 | PASS |
| injection | explicit_note_end/jev followed_injection_pct | 45.7 | 45.7 | PASS |
| injection | explicit_note_end/bema_baseline n | 1,000 | 1,000 | PASS |
| injection | explicit_note_end/bema_baseline accuracy_pct | 35.1 | 35.1 | PASS |
| injection | explicit_note_end/bema_baseline mean_confidence (3 dp) | 0.582 | 0.582 | PASS |
| injection | explicit_note_end/bema_baseline followed_injection_pct | 17.4 | 17.4 | PASS |
| injection | explicit_note_end/bema_v2 n | 1,000 | 1,000 | PASS |
| injection | explicit_note_end/bema_v2 accuracy_pct | 46.9 | 46.9 | PASS |
| injection | explicit_note_end/bema_v2 mean_confidence (3 dp) | 0.688 | 0.688 | PASS |
| injection | explicit_note_end/bema_v2 followed_injection_pct | 23.7 | 23.7 | PASS |
| injection | explicit_note_end/claude-haiku-5.5 n | 1,000 | 1,000 | PASS |
| injection | explicit_note_end/claude-haiku-5.5 accuracy_pct | 59.1 | 59.1 | PASS |
| injection | explicit_note_end/claude-haiku-5.5 mean_confidence (3 dp) | 0.814 | 0.814 | PASS |
| injection | explicit_note_end/claude-haiku-5.5 followed_injection_pct | 24.1 | 24.1 | PASS |
| injection | explicit_note_end/qwen3.8-flash n | 1,000 | 1,000 | PASS |
| injection | explicit_note_end/qwen3.8-flash accuracy_pct | 24.4 | 24.4 | PASS |
| injection | explicit_note_end/qwen3.8-flash mean_confidence (3 dp) | 0.981 | 0.981 | PASS |
| injection | explicit_note_end/qwen3.8-flash followed_injection_pct | 71.5 | 71.5 | PASS |
| injection | explicit_note_end/claude-sonnet-5.5 n | 1,000 | 1,000 | PASS |
| injection | explicit_note_end/claude-sonnet-5.5 accuracy_pct | 82.5 | 82.5 | PASS |
| injection | explicit_note_end/claude-sonnet-5.5 mean_confidence (3 dp) | 0.845 | 0.845 | PASS |
| injection | explicit_note_end/claude-sonnet-5.5 followed_injection_pct | 2.1 | 2.1 | PASS |
| injection | explicit_note_end/claude-opus-5.5 n | 1,000 | 1,000 | PASS |
| injection | explicit_note_end/claude-opus-5.5 accuracy_pct | 82.2 | 82.2 | PASS |
| injection | explicit_note_end/claude-opus-5.5 mean_confidence (3 dp) | 0.803 | 0.803 | PASS |
| injection | explicit_note_end/claude-opus-5.5 followed_injection_pct | 8.2 | 8.2 | PASS |
| injection | explicit_note_start/jev n | 1,000 | 1,000 | PASS |
| injection | explicit_note_start/jev accuracy_pct | 62 | 62 | PASS |
| injection | explicit_note_start/jev mean_confidence (3 dp) | 0.719 | 0.719 | PASS |
| injection | explicit_note_start/jev followed_injection_pct | 27.7 | 27.7 | PASS |
| injection | explicit_note_start/bema_baseline n | 1,000 | 1,000 | PASS |
| injection | explicit_note_start/bema_baseline accuracy_pct | 21.8 | 21.8 | PASS |
| injection | explicit_note_start/bema_baseline mean_confidence (3 dp) | 0.56 | 0.56 | PASS |
| injection | explicit_note_start/bema_baseline followed_injection_pct | 24.2 | 24.2 | PASS |
| injection | explicit_note_start/bema_v2 n | 1,000 | 1,000 | PASS |
| injection | explicit_note_start/bema_v2 accuracy_pct | 34 | 34 | PASS |
| injection | explicit_note_start/bema_v2 mean_confidence (3 dp) | 0.662 | 0.662 | PASS |
| injection | explicit_note_start/bema_v2 followed_injection_pct | 33.6 | 33.6 | PASS |
| injection | casual_ps/jev n | 1,000 | 1,000 | PASS |
| injection | casual_ps/jev accuracy_pct | 32.6 | 32.6 | PASS |
| injection | casual_ps/jev mean_confidence (3 dp) | 0.735 | 0.735 | PASS |
| injection | casual_ps/jev followed_injection_pct | 63.5 | 63.5 | PASS |
| injection | casual_ps/bema_baseline n | 1,000 | 1,000 | PASS |
| injection | casual_ps/bema_baseline accuracy_pct | 43.9 | 43.9 | PASS |
| injection | casual_ps/bema_baseline mean_confidence (3 dp) | 0.628 | 0.628 | PASS |
| injection | casual_ps/bema_baseline followed_injection_pct | 22.3 | 22.3 | PASS |
| injection | casual_ps/bema_v2 n | 1,000 | 1,000 | PASS |
| injection | casual_ps/bema_v2 accuracy_pct | 50.8 | 50.8 | PASS |
| injection | casual_ps/bema_v2 mean_confidence (3 dp) | 0.735 | 0.735 | PASS |
| injection | casual_ps/bema_v2 followed_injection_pct | 26.5 | 26.5 | PASS |
| injection | casual_ps/claude-haiku-5.5 n | 1,000 | 1,000 | PASS |
| injection | casual_ps/claude-haiku-5.5 accuracy_pct | 66.7 | 66.7 | PASS |
| injection | casual_ps/claude-haiku-5.5 mean_confidence (3 dp) | 0.705 | 0.705 | PASS |
| injection | casual_ps/claude-haiku-5.5 followed_injection_pct | 13.4 | 13.4 | PASS |
| injection | casual_ps/qwen3.8-flash n | 1,000 | 1,000 | PASS |
| injection | casual_ps/qwen3.8-flash accuracy_pct | 57.7 | 57.7 | PASS |
| injection | casual_ps/qwen3.8-flash mean_confidence (3 dp) | 0.944 | 0.944 | PASS |
| injection | casual_ps/qwen3.8-flash followed_injection_pct | 32.6 | 32.6 | PASS |
| injection | casual_ps/claude-sonnet-5.5 n | 1,000 | 1,000 | PASS |
| injection | casual_ps/claude-sonnet-5.5 accuracy_pct | 63.4 | 63.4 | PASS |
| injection | casual_ps/claude-sonnet-5.5 mean_confidence (3 dp) | 0.609 | 0.609 | PASS |
| injection | casual_ps/claude-sonnet-5.5 followed_injection_pct | 29.4 | 29.4 | PASS |
| injection | casual_ps/claude-opus-5.5 n | 1,000 | 1,000 | PASS |
| injection | casual_ps/claude-opus-5.5 accuracy_pct | 85.1 | 85.1 | PASS |
| injection | casual_ps/claude-opus-5.5 mean_confidence (3 dp) | 0.721 | 0.721 | PASS |
| injection | casual_ps/claude-opus-5.5 followed_injection_pct | 3.9 | 3.9 | PASS |
| injection | neutral_note_control/jev n | 1,000 | 1,000 | PASS |
| injection | neutral_note_control/jev accuracy_pct | 80 | 80 | PASS |
| injection | neutral_note_control/jev mean_confidence (3 dp) | 0.877 | 0.877 | PASS |
| injection | neutral_note_control/bema_baseline n | 1,000 | 1,000 | PASS |
| injection | neutral_note_control/bema_baseline accuracy_pct | 45.5 | 45.5 | PASS |
| injection | neutral_note_control/bema_baseline mean_confidence (3 dp) | 0.618 | 0.618 | PASS |
| injection | neutral_note_control/bema_v2 n | 1,000 | 1,000 | PASS |
| injection | neutral_note_control/bema_v2 accuracy_pct | 68 | 68 | PASS |
| injection | neutral_note_control/bema_v2 mean_confidence (3 dp) | 0.704 | 0.704 | PASS |
| injection | neutral_note_control/claude-haiku-5.5 n | 1,000 | 1,000 | PASS |
| injection | neutral_note_control/claude-haiku-5.5 accuracy_pct | 76 | 76 | PASS |
| injection | neutral_note_control/claude-haiku-5.5 mean_confidence (3 dp) | 0.871 | 0.871 | PASS |
| injection | neutral_note_control/qwen3.8-flash n | 1,000 | 1,000 | PASS |
| injection | neutral_note_control/qwen3.8-flash accuracy_pct | 79.7 | 79.7 | PASS |
| injection | neutral_note_control/qwen3.8-flash mean_confidence (3 dp) | 0.953 | 0.953 | PASS |
| injection | neutral_note_control/claude-sonnet-5.5 n | 1,000 | 1,000 | PASS |
| injection | neutral_note_control/claude-sonnet-5.5 accuracy_pct | 84.4 | 84.4 | PASS |
| injection | neutral_note_control/claude-sonnet-5.5 mean_confidence (3 dp) | 0.788 | 0.788 | PASS |
| injection | neutral_note_control/claude-opus-5.5 n | 1,000 | 1,000 | PASS |
| injection | neutral_note_control/claude-opus-5.5 accuracy_pct | 89.1 | 89.1 | PASS |
| injection | neutral_note_control/claude-opus-5.5 mean_confidence (3 dp) | 0.798 | 0.798 | PASS |
| injection | explicit_note_defended/jev n | 1,000 | 1,000 | PASS |
| injection | explicit_note_defended/jev accuracy_pct | 77.8 | 77.8 | PASS |
| injection | explicit_note_defended/jev mean_confidence (3 dp) | 0.848 | 0.848 | PASS |
| injection | explicit_note_defended/jev followed_injection_pct | 2.7 | 2.7 | PASS |
| injection | explicit_note_defended/bema_baseline n | 1,000 | 1,000 | PASS |
| injection | explicit_note_defended/bema_baseline accuracy_pct | 35.1 | 35.1 | PASS |
| injection | explicit_note_defended/bema_baseline mean_confidence (3 dp) | 0.582 | 0.582 | PASS |
| injection | explicit_note_defended/bema_baseline followed_injection_pct | 17.4 | 17.4 | PASS |
| injection | explicit_note_defended/bema_v2 n | 1,000 | 1,000 | PASS |
| injection | explicit_note_defended/bema_v2 accuracy_pct | 46.9 | 46.9 | PASS |
| injection | explicit_note_defended/bema_v2 mean_confidence (3 dp) | 0.688 | 0.688 | PASS |
| injection | explicit_note_defended/bema_v2 followed_injection_pct | 23.7 | 23.7 | PASS |
| injection | casual_ps_defended/jev n | 1,000 | 1,000 | PASS |
| injection | casual_ps_defended/jev accuracy_pct | 63 | 63 | PASS |
| injection | casual_ps_defended/jev mean_confidence (3 dp) | 0.755 | 0.755 | PASS |
| injection | casual_ps_defended/jev followed_injection_pct | 23.4 | 23.4 | PASS |
| injection | casual_ps_defended/bema_baseline n | 1,000 | 1,000 | PASS |
| injection | casual_ps_defended/bema_baseline accuracy_pct | 43.9 | 43.9 | PASS |
| injection | casual_ps_defended/bema_baseline mean_confidence (3 dp) | 0.628 | 0.628 | PASS |
| injection | casual_ps_defended/bema_baseline followed_injection_pct | 22.3 | 22.3 | PASS |
| injection | casual_ps_defended/bema_v2 n | 1,000 | 1,000 | PASS |
| injection | casual_ps_defended/bema_v2 accuracy_pct | 50.8 | 50.8 | PASS |
| injection | casual_ps_defended/bema_v2 mean_confidence (3 dp) | 0.735 | 0.735 | PASS |
| injection | casual_ps_defended/bema_v2 followed_injection_pct | 26.5 | 26.5 | PASS |
| injection | casual_ps_defended/claude-haiku-5.5 n | 1,000 | 1,000 | PASS |
| injection | casual_ps_defended/claude-haiku-5.5 accuracy_pct | 71.3 | 71.3 | PASS |
| injection | casual_ps_defended/claude-haiku-5.5 mean_confidence (3 dp) | 0.787 | 0.787 | PASS |
| injection | casual_ps_defended/claude-haiku-5.5 followed_injection_pct | 6.5 | 6.5 | PASS |
| injection | casual_ps_defended/qwen3.8-flash n | 1,000 | 1,000 | PASS |
| injection | casual_ps_defended/qwen3.8-flash accuracy_pct | 70.8 | 70.8 | PASS |
| injection | casual_ps_defended/qwen3.8-flash mean_confidence (3 dp) | 0.95 | 0.95 | PASS |
| injection | casual_ps_defended/qwen3.8-flash followed_injection_pct | 13.3 | 13.3 | PASS |
| injection | casual_ps_defended/claude-sonnet-5.5 n | 1,000 | 1,000 | PASS |
| injection | casual_ps_defended/claude-sonnet-5.5 accuracy_pct | 83.3 | 83.3 | PASS |
| injection | casual_ps_defended/claude-sonnet-5.5 mean_confidence (3 dp) | 0.806 | 0.806 | PASS |
| injection | casual_ps_defended/claude-sonnet-5.5 followed_injection_pct | 1.8 | 1.8 | PASS |
| injection | casual_ps_defended/claude-opus-5.5 n | 1,000 | 1,000 | PASS |
| injection | casual_ps_defended/claude-opus-5.5 accuracy_pct | 88.8 | 88.8 | PASS |
| injection | casual_ps_defended/claude-opus-5.5 mean_confidence (3 dp) | 0.808 | 0.808 | PASS |
| injection | casual_ps_defended/claude-opus-5.5 followed_injection_pct | 0.3 | 0.3 | PASS |
| injection | clean_reference/jev | 80 | 80 | PASS |
| injection | clean_reference/bema_baseline | 82 | 82 | PASS |
| injection | clean_reference/bema_v2 | 81.3 | 81.3 | PASS |
| speed_cost | jev median_ms_concurrent_core_run | 104.01 | 104.01 | PASS |
| speed_cost | jev median_ms_one_at_a_time | 99.0038 | 99.0038 | PASS |
| speed_cost | jev usd_per_1000_banking_decisions | 0.071 | 0.071 | PASS |
| speed_cost | jev usd_per_1000_spam_decisions | 0.0124 | 0.0124 | PASS |
| speed_cost | _bema_timing_machine equals timing_quiet._machine | {"cpu_count": 2, "load_avg_at_start": [1.70751953125, 1.984375, 2.58056640625], "torch_threads": 2} | {"cpu_count": 2, "load_avg_at_start": [1.70751953125, 1.984375, 2.58056640625], "torch_threads": 2} | PASS |
| speed_cost | claude-haiku-5.5 median_ms_one_at_a_time | 836.789 | 836.789 | PASS |
| speed_cost | claude-haiku-5.5 usd_per_1000_banking_decisions | 0.1004 | 0.1004 | PASS |
| speed_cost | claude-haiku-5.5 usd_per_1000_spam_decisions | 0.019 | 0.019 | PASS |
| speed_cost | gpt-6-luna median_ms_one_at_a_time | 1193.5 | 1193.5 | PASS |
| speed_cost | gpt-6-luna usd_per_1000_banking_decisions | 0.0582 | 0.0582 | PASS |
| speed_cost | gpt-6-luna usd_per_1000_spam_decisions | None | None | PASS |
| speed_cost | qwen3.8-flash median_ms_one_at_a_time | 983.565 | 983.565 | PASS |
| speed_cost | qwen3.8-flash usd_per_1000_banking_decisions | 0.0478 | 0.0478 | PASS |
| speed_cost | qwen3.8-flash usd_per_1000_spam_decisions | 0.0193 | 0.0193 | PASS |
| speed_cost | gpt-6.1-terra median_ms_one_at_a_time | 1269.46 | 1269.46 | PASS |
| speed_cost | gpt-6.1-terra usd_per_1000_banking_decisions | 0.5821 | 0.5821 | PASS |
| speed_cost | gpt-6.1-terra usd_per_1000_spam_decisions | None | None | PASS |
| speed_cost | claude-sonnet-5.5 median_ms_one_at_a_time | 1137.52 | 1137.52 | PASS |
| speed_cost | claude-sonnet-5.5 usd_per_1000_banking_decisions | 2.0075 | 2.0075 | PASS |
| speed_cost | claude-sonnet-5.5 usd_per_1000_spam_decisions | 0.3807 | 0.3807 | PASS |
| speed_cost | claude-opus-5.5 median_ms_one_at_a_time | 1234.16 | 1234.16 | PASS |
| speed_cost | claude-opus-5.5 usd_per_1000_banking_decisions | 4.5404 | 4.5404 | PASS |
| speed_cost | claude-opus-5.5 usd_per_1000_spam_decisions | 0.8579 | 0.8579 | PASS |
| speed_cost | claude-fable-5.1 median_ms_one_at_a_time | 1298.8 | 1298.8 | PASS |
| speed_cost | claude-fable-5.1 usd_per_1000_banking_decisions | 10.0488 | 10.0488 | PASS |
| speed_cost | claude-fable-5.1 usd_per_1000_spam_decisions | None | None | PASS |
| speed_cost | bema_baseline median_ms_one_at_a_time (= timing_quiet.json median_ms) | 1.69778 | 1.69778 | PASS |
| speed_cost | bema_baseline note | BEMA v1 forward pass, all 3 heads, batch 1, CPU | BEMA v1 forward pass, all 3 heads, batch 1, CPU | PASS |
| speed_cost | bema_v2 median_ms_one_at_a_time (= timing_quiet.json median_ms) | 14.0163 | 14.0163 | PASS |
| speed_cost | bema_v2 note | BEMA v2 forward pass (MiniLM-L6 + 3 heads), batch 1, CPU | BEMA v2 forward pass (MiniLM-L6 + 3 heads), batch 1, CPU | PASS |
| speed_cost | bema_v2_spell median_ms_one_at_a_time (= timing_quiet.json median_ms) | 14.0686 | 14.0686 | PASS |
| speed_cost | bema_v2_spell note | BEMA v2 forward pass plus the spelling correction step | BEMA v2 forward pass plus the spelling correction step | PASS |
| spend | jev_usd_estimate | 3.583 | 3.583 | PASS |
| spend | openrouter_usd_reported_by_openrouter | 3.115 | 3.115 | PASS |
| spend | anthropic_usd_from_tokens_x_list_price | 110.575 | 110.575 | PASS |
| redteam_other | copied blocks equal current runs/redteam_report.json | True | True | PASS |
| spam | test_messages_also_in_training (exact text) | 83 | 83 | PASS |
| spam | clean_excluding_duplicates/jev n | 753 | 753 | PASS |
| spam | clean_excluding_duplicates/jev accuracy_pct | 96.8 | 96.8 | PASS |
| spam | clean_excluding_duplicates/bema_baseline n | 753 | 753 | PASS |
| spam | clean_excluding_duplicates/bema_baseline accuracy_pct | 97.6 | 97.6 | PASS |
| spam | clean_excluding_duplicates/bema_v2 n | 753 | 753 | PASS |
| spam | clean_excluding_duplicates/bema_v2 accuracy_pct | 98.5 | 98.5 | PASS |
| spam | clean_excluding_duplicates/claude-haiku-5.5 n | 753 | 753 | PASS |
| spam | clean_excluding_duplicates/claude-haiku-5.5 accuracy_pct | 94.3 | 94.3 | PASS |
| spam | clean_excluding_duplicates/qwen3.8-flash n | 753 | 753 | PASS |
| spam | clean_excluding_duplicates/qwen3.8-flash accuracy_pct | 81.3 | 81.3 | PASS |
| spam | clean_excluding_duplicates/claude-sonnet-5.5 n | 753 | 753 | PASS |
| spam | clean_excluding_duplicates/claude-sonnet-5.5 accuracy_pct | 97.2 | 97.2 | PASS |
| spam | clean_excluding_duplicates/claude-opus-5.5 n | 753 | 753 | PASS |
| spam | clean_excluding_duplicates/claude-opus-5.5 accuracy_pct | 99.7 | 99.7 | PASS |
| paired_vs_jev | bema_baseline n | 3,080 | 3,080 | PASS |
| paired_vs_jev | bema_baseline minus_jev_points (1 dp) | 1.3 | 1.3 | PASS |
| paired_vs_jev | bema_baseline ci95_points (own bootstrap, tol 0.2 pts) | [-0.3, 2.9] | [-0.4, 2.9] | PASS |
| paired_vs_jev | bema_v2 n | 3,080 | 3,080 | PASS |
| paired_vs_jev | bema_v2 minus_jev_points (1 dp) | 1.8 | 1.8 | PASS |
| paired_vs_jev | bema_v2 ci95_points (own bootstrap, tol 0.2 pts) | [0.2, 3.3] | [0.2, 3.4] | PASS |
| paired_vs_jev | bema_v2_spell n | 3,080 | 3,080 | PASS |
| paired_vs_jev | bema_v2_spell minus_jev_points (1 dp) | 1.8 | 1.8 | PASS |
| paired_vs_jev | bema_v2_spell ci95_points (own bootstrap, tol 0.2 pts) | [0.2, 3.3] | [0.2, 3.4] | PASS |
| paired_vs_jev | claude-haiku-5.5 n | 3,080 | 3,080 | PASS |
| paired_vs_jev | claude-haiku-5.5 minus_jev_points (1 dp) | -3.8 | -3.8 | PASS |
| paired_vs_jev | claude-haiku-5.5 ci95_points (own bootstrap, tol 0.2 pts) | [-5, -2.6] | [-5, -2.6] | PASS |
| paired_vs_jev | qwen3.8-flash n | 3,080 | 3,080 | PASS |
| paired_vs_jev | qwen3.8-flash minus_jev_points (1 dp) | -1.2 | -1.2 | PASS |
| paired_vs_jev | qwen3.8-flash ci95_points (own bootstrap, tol 0.2 pts) | [-2.3, -0.1] | [-2.4, -0.1] | PASS |
| paired_vs_jev | claude-sonnet-5.5 n | 3,080 | 3,080 | PASS |
| paired_vs_jev | claude-sonnet-5.5 minus_jev_points (1 dp) | 4 | 4 | PASS |
| paired_vs_jev | claude-sonnet-5.5 ci95_points (own bootstrap, tol 0.2 pts) | [3, 4.9] | [3.1, 4.9] | PASS |
| paired_vs_jev | claude-opus-5.5 n | 3,080 | 3,080 | PASS |
| paired_vs_jev | claude-opus-5.5 minus_jev_points (1 dp) | 8.2 | 8.2 | PASS |
| paired_vs_jev | claude-opus-5.5 ci95_points (own bootstrap, tol 0.2 pts) | [7.1, 9.3] | [7.1, 9.3] | PASS |

### llm_report.json (my parser) (1269/1269 PASS)

| section | metric | reported | recomputed | result |
|---|---|---|---|---|
| llm claude-haiku-5.5/banking77_clean | n | 3,080 | 3,080 | PASS |
| llm claude-haiku-5.5/banking77_clean | n_missing | 0 | 0 | PASS |
| llm claude-haiku-5.5/banking77_clean | invalid_output | 2 | 2 | PASS |
| llm claude-haiku-5.5/banking77_clean | invalid_output_rate | 0.000649351 | 0.000649351 | PASS |
| llm claude-haiku-5.5/banking77_clean | accuracy_invalid_counted_wrong | 0.767857 | 0.767857 | PASS |
| llm claude-haiku-5.5/banking77_clean | n_with_confidence | 3,078 | 3,078 | PASS |
| llm claude-haiku-5.5/banking77_clean | mean_stated_confidence | 0.854058 | 0.854058 | PASS |
| llm claude-haiku-5.5/banking77_clean | accuracy_on_those | 0.768356 | 0.768356 | PASS |
| llm claude-haiku-5.5/banking77_clean | overconfidence | 0.0857018 | 0.0857018 | PASS |
| llm claude-haiku-5.5/banking77_clean | cost_usd | 0.309223 | 0.309223 | PASS |
| llm claude-haiku-5.5/banking77_clean | cost_per_1000_calls_usd | 0.100397 | 0.100397 | PASS |
| llm claude-haiku-5.5/banking77_clean | mean_prompt_tokens | 867.758 | 867.758 | PASS |
| llm claude-haiku-5.5/banking77_clean | mean_output_tokens | 27.2425 | 27.2425 | PASS |
| llm claude-haiku-5.5/banking77_clean | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-haiku-5.5/banking77_clean | strict_json_reply_rate | 0.999351 | 0.999351 | PASS |
| llm claude-haiku-5.5/banking77_clean | ece_stated_confidence | 0.0857018 | 0.0857018 | PASS |
| llm claude-haiku-5.5/banking77_clean | accuracy_ci95 (Wilson) | [0.752619, 0.782428] | [0.752619, 0.782428] | PASS |
| llm claude-haiku-5.5/banking77_clean | latency median | 1066.89 | 1066.89 | PASS |
| llm claude-haiku-5.5/banking77_clean | latency p90 (index rule) | 1239.27 | 1239.27 | PASS |
| llm claude-haiku-5.5/banking77_clean | providers | ["Claude Platform on AWS"] | ["Claude Platform on AWS"] | PASS |
| llm claude-haiku-5.5/banking77_swap2 | n | 3,080 | 3,080 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | n_missing | 0 | 0 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | invalid_output | 5 | 5 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | invalid_output_rate | 0.00162338 | 0.00162338 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | accuracy_invalid_counted_wrong | 0.747078 | 0.747078 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | n_with_confidence | 3,075 | 3,075 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | mean_stated_confidence | 0.850146 | 0.850146 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | accuracy_on_those | 0.748293 | 0.748293 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | overconfidence | 0.101854 | 0.101854 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | cost_usd | 0.310079 | 0.310079 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | cost_per_1000_calls_usd | 0.100675 | 0.100675 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | mean_prompt_tokens | 870.488 | 870.488 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | mean_output_tokens | 27.2526 | 27.2526 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | strict_json_reply_rate | 0.998377 | 0.998377 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | ece_stated_confidence | 0.102751 | 0.102751 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | accuracy_ci95 (Wilson) | [0.731425, 0.762115] | [0.731425, 0.762115] | PASS |
| llm claude-haiku-5.5/banking77_swap2 | latency median | 1077.34 | 1077.34 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | latency p90 (index rule) | 1256.39 | 1256.39 | PASS |
| llm claude-haiku-5.5/banking77_swap2 | providers | ["Claude Platform on AWS"] | ["Claude Platform on AWS"] | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | n | 3,080 | 3,080 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | n_missing | 0 | 0 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | invalid_output | 5 | 5 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | invalid_output_rate | 0.00162338 | 0.00162338 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | accuracy_invalid_counted_wrong | 0.725649 | 0.725649 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | n_with_confidence | 3,075 | 3,075 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | mean_stated_confidence | 0.832667 | 0.832667 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | accuracy_on_those | 0.726829 | 0.726829 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | overconfidence | 0.105837 | 0.105837 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | cost_usd | 0.310247 | 0.310247 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | cost_per_1000_calls_usd | 0.10073 | 0.10073 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | mean_prompt_tokens | 870.81 | 870.81 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | mean_output_tokens | 27.2971 | 27.2971 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | strict_json_reply_rate | 0.999026 | 0.999026 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | ece_stated_confidence | 0.107119 | 0.107119 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | accuracy_ci95 (Wilson) | [0.709618, 0.741119] | [0.709618, 0.741119] | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | latency median | 1098.82 | 1098.82 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | latency p90 (index rule) | 1297.38 | 1297.38 | PASS |
| llm claude-haiku-5.5/banking77_mixed3 | providers | ["Claude Platform on AWS"] | ["Claude Platform on AWS"] | PASS |
| llm claude-haiku-5.5/spam_clean | n | 836 | 836 | PASS |
| llm claude-haiku-5.5/spam_clean | n_missing | 0 | 0 | PASS |
| llm claude-haiku-5.5/spam_clean | invalid_output | 0 | 0 | PASS |
| llm claude-haiku-5.5/spam_clean | invalid_output_rate | 0 | 0 | PASS |
| llm claude-haiku-5.5/spam_clean | accuracy_invalid_counted_wrong | 0.940191 | 0.940191 | PASS |
| llm claude-haiku-5.5/spam_clean | n_with_confidence | 836 | 836 | PASS |
| llm claude-haiku-5.5/spam_clean | mean_stated_confidence | 0.941471 | 0.941471 | PASS |
| llm claude-haiku-5.5/spam_clean | accuracy_on_those | 0.940191 | 0.940191 | PASS |
| llm claude-haiku-5.5/spam_clean | overconfidence | 0.0012799 | 0.0012799 | PASS |
| llm claude-haiku-5.5/spam_clean | cost_usd | 0.0159113 | 0.0159113 | PASS |
| llm claude-haiku-5.5/spam_clean | cost_per_1000_calls_usd | 0.0190327 | 0.0190327 | PASS |
| llm claude-haiku-5.5/spam_clean | mean_prompt_tokens | 120.327 | 120.327 | PASS |
| llm claude-haiku-5.5/spam_clean | mean_output_tokens | 14 | 14 | PASS |
| llm claude-haiku-5.5/spam_clean | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-haiku-5.5/spam_clean | strict_json_reply_rate | 1 | 1 | PASS |
| llm claude-haiku-5.5/spam_clean | ece_stated_confidence | 0.0349641 | 0.0349641 | PASS |
| llm claude-haiku-5.5/spam_clean | accuracy_ci95 (Wilson) | [0.922014, 0.954341] | [0.922014, 0.954341] | PASS |
| llm claude-haiku-5.5/spam_clean | latency median | 1043.59 | 1043.59 | PASS |
| llm claude-haiku-5.5/spam_clean | latency p90 (index rule) | 1249.53 | 1249.53 | PASS |
| llm claude-haiku-5.5/spam_clean | providers | ["Claude Platform on AWS"] | ["Claude Platform on AWS"] | PASS |
| llm claude-haiku-5.5/spam_swap2 | n | 836 | 836 | PASS |
| llm claude-haiku-5.5/spam_swap2 | n_missing | 0 | 0 | PASS |
| llm claude-haiku-5.5/spam_swap2 | invalid_output | 0 | 0 | PASS |
| llm claude-haiku-5.5/spam_swap2 | invalid_output_rate | 0 | 0 | PASS |
| llm claude-haiku-5.5/spam_swap2 | accuracy_invalid_counted_wrong | 0.917464 | 0.917464 | PASS |
| llm claude-haiku-5.5/spam_swap2 | n_with_confidence | 836 | 836 | PASS |
| llm claude-haiku-5.5/spam_swap2 | mean_stated_confidence | 0.929414 | 0.929414 | PASS |
| llm claude-haiku-5.5/spam_swap2 | accuracy_on_those | 0.917464 | 0.917464 | PASS |
| llm claude-haiku-5.5/spam_swap2 | overconfidence | 0.0119498 | 0.0119498 | PASS |
| llm claude-haiku-5.5/spam_swap2 | cost_usd | 0.0160787 | 0.0160787 | PASS |
| llm claude-haiku-5.5/spam_swap2 | cost_per_1000_calls_usd | 0.0192329 | 0.0192329 | PASS |
| llm claude-haiku-5.5/spam_swap2 | mean_prompt_tokens | 122.329 | 122.329 | PASS |
| llm claude-haiku-5.5/spam_swap2 | mean_output_tokens | 14 | 14 | PASS |
| llm claude-haiku-5.5/spam_swap2 | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-haiku-5.5/spam_swap2 | strict_json_reply_rate | 1 | 1 | PASS |
| llm claude-haiku-5.5/spam_swap2 | ece_stated_confidence | 0.0560407 | 0.0560407 | PASS |
| llm claude-haiku-5.5/spam_swap2 | accuracy_ci95 (Wilson) | [0.896846, 0.934263] | [0.896846, 0.934263] | PASS |
| llm claude-haiku-5.5/spam_swap2 | latency median | 1052.22 | 1052.22 | PASS |
| llm claude-haiku-5.5/spam_swap2 | latency p90 (index rule) | 1204.95 | 1204.95 | PASS |
| llm claude-haiku-5.5/spam_swap2 | providers | ["Claude Platform on AWS"] | ["Claude Platform on AWS"] | PASS |
| llm claude-haiku-5.5/injection_banking | n | 1,000 | 1,000 | PASS |
| llm claude-haiku-5.5/injection_banking | n_missing | 0 | 0 | PASS |
| llm claude-haiku-5.5/injection_banking | invalid_output | 11 | 11 | PASS |
| llm claude-haiku-5.5/injection_banking | invalid_output_rate | 0.011 | 0.011 | PASS |
| llm claude-haiku-5.5/injection_banking | accuracy_invalid_counted_wrong | 0.591 | 0.591 | PASS |
| llm claude-haiku-5.5/injection_banking | n_with_confidence | 989 | 989 | PASS |
| llm claude-haiku-5.5/injection_banking | mean_stated_confidence | 0.814206 | 0.814206 | PASS |
| llm claude-haiku-5.5/injection_banking | accuracy_on_those | 0.597573 | 0.597573 | PASS |
| llm claude-haiku-5.5/injection_banking | overconfidence | 0.216633 | 0.216633 | PASS |
| llm claude-haiku-5.5/injection_banking | followed_injected_label | 0.241 | 0.241 | PASS |
| llm claude-haiku-5.5/injection_banking | cost_usd | 0.103295 | 0.103295 | PASS |
| llm claude-haiku-5.5/injection_banking | cost_per_1000_calls_usd | 0.103295 | 0.103295 | PASS |
| llm claude-haiku-5.5/injection_banking | mean_prompt_tokens | 891.996 | 891.996 | PASS |
| llm claude-haiku-5.5/injection_banking | mean_output_tokens | 28.191 | 28.191 | PASS |
| llm claude-haiku-5.5/injection_banking | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-haiku-5.5/injection_banking | strict_json_reply_rate | 0.981 | 0.981 | PASS |
| llm claude-haiku-5.5/injection_banking | ece_stated_confidence | 0.239646 | 0.239646 | PASS |
| llm claude-haiku-5.5/injection_banking | accuracy_ci95 (Wilson) | [0.560236, 0.621068] | [0.560236, 0.621068] | PASS |
| llm claude-haiku-5.5/injection_banking | latency median | 1108.14 | 1108.14 | PASS |
| llm claude-haiku-5.5/injection_banking | latency p90 (index rule) | 1295.05 | 1295.05 | PASS |
| llm claude-haiku-5.5/injection_banking | providers | ["Claude Platform on AWS"] | ["Claude Platform on AWS"] | PASS |
| llm claude-haiku-5.5/injection_control_banking | n | 1,000 | 1,000 | PASS |
| llm claude-haiku-5.5/injection_control_banking | n_missing | 0 | 0 | PASS |
| llm claude-haiku-5.5/injection_control_banking | invalid_output | 0 | 0 | PASS |
| llm claude-haiku-5.5/injection_control_banking | invalid_output_rate | 0 | 0 | PASS |
| llm claude-haiku-5.5/injection_control_banking | accuracy_invalid_counted_wrong | 0.76 | 0.76 | PASS |
| llm claude-haiku-5.5/injection_control_banking | n_with_confidence | 1,000 | 1,000 | PASS |
| llm claude-haiku-5.5/injection_control_banking | mean_stated_confidence | 0.87111 | 0.87111 | PASS |
| llm claude-haiku-5.5/injection_control_banking | accuracy_on_those | 0.76 | 0.76 | PASS |
| llm claude-haiku-5.5/injection_control_banking | overconfidence | 0.11111 | 0.11111 | PASS |
| llm claude-haiku-5.5/injection_control_banking | cost_usd | 0.102583 | 0.102583 | PASS |
| llm claude-haiku-5.5/injection_control_banking | cost_per_1000_calls_usd | 0.102583 | 0.102583 | PASS |
| llm claude-haiku-5.5/injection_control_banking | mean_prompt_tokens | 889.91 | 889.91 | PASS |
| llm claude-haiku-5.5/injection_control_banking | mean_output_tokens | 27.185 | 27.185 | PASS |
| llm claude-haiku-5.5/injection_control_banking | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-haiku-5.5/injection_control_banking | strict_json_reply_rate | 1 | 1 | PASS |
| llm claude-haiku-5.5/injection_control_banking | ece_stated_confidence | 0.11111 | 0.11111 | PASS |
| llm claude-haiku-5.5/injection_control_banking | accuracy_ci95 (Wilson) | [0.732567, 0.785443] | [0.732567, 0.785443] | PASS |
| llm claude-haiku-5.5/injection_control_banking | latency median | 1116.44 | 1116.44 | PASS |
| llm claude-haiku-5.5/injection_control_banking | latency p90 (index rule) | 1301.22 | 1301.22 | PASS |
| llm claude-haiku-5.5/injection_control_banking | providers | ["Claude Platform on AWS"] | ["Claude Platform on AWS"] | PASS |
| llm claude-haiku-5.5/injection_casual_banking | n | 1,000 | 1,000 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | n_missing | 0 | 0 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | invalid_output | 0 | 0 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | invalid_output_rate | 0 | 0 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | accuracy_invalid_counted_wrong | 0.667 | 0.667 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | n_with_confidence | 1,000 | 1,000 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | mean_stated_confidence | 0.70478 | 0.70478 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | accuracy_on_those | 0.667 | 0.667 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | overconfidence | 0.03778 | 0.03778 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | followed_injected_label | 0.134 | 0.134 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | cost_usd | 0.102103 | 0.102103 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | cost_per_1000_calls_usd | 0.102103 | 0.102103 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | mean_prompt_tokens | 882.996 | 882.996 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | mean_output_tokens | 27.607 | 27.607 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | strict_json_reply_rate | 0.993 | 0.993 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | ece_stated_confidence | 0.04574 | 0.04574 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | accuracy_ci95 (Wilson) | [0.6372, 0.695522] | [0.6372, 0.695522] | PASS |
| llm claude-haiku-5.5/injection_casual_banking | latency median | 1120.77 | 1120.77 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | latency p90 (index rule) | 1286.97 | 1286.97 | PASS |
| llm claude-haiku-5.5/injection_casual_banking | providers | ["Claude Platform on AWS"] | ["Claude Platform on AWS"] | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | n | 1,000 | 1,000 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | n_missing | 0 | 0 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | invalid_output | 13 | 13 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | invalid_output_rate | 0.013 | 0.013 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | accuracy_invalid_counted_wrong | 0.713 | 0.713 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | n_with_confidence | 987 | 987 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | mean_stated_confidence | 0.786869 | 0.786869 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | accuracy_on_those | 0.722391 | 0.722391 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | overconfidence | 0.0644782 | 0.0644782 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | followed_injected_label | 0.065 | 0.065 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | cost_usd | 0.106233 | 0.106233 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | cost_per_1000_calls_usd | 0.106233 | 0.106233 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | mean_prompt_tokens | 918.996 | 918.996 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | mean_output_tokens | 28.667 | 28.667 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | strict_json_reply_rate | 0.97 | 0.97 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | ece_stated_confidence | 0.0729889 | 0.0729889 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | accuracy_ci95 (Wilson) | [0.68419, 0.74018] | [0.68419, 0.74018] | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | latency median | 1092.49 | 1092.49 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | latency p90 (index rule) | 1278.4 | 1278.4 | PASS |
| llm claude-haiku-5.5/injection_casual_defended_banking | providers | ["Claude Platform on AWS"] | ["Claude Platform on AWS"] | PASS |
| llm claude-haiku-5.5/latency_sequential | n | 100 | 100 | PASS |
| llm claude-haiku-5.5/latency_sequential | n_missing | 0 | 0 | PASS |
| llm claude-haiku-5.5/latency_sequential | invalid_output | 0 | 0 | PASS |
| llm claude-haiku-5.5/latency_sequential | invalid_output_rate | 0 | 0 | PASS |
| llm claude-haiku-5.5/latency_sequential | accuracy_invalid_counted_wrong | 0.82 | 0.82 | PASS |
| llm claude-haiku-5.5/latency_sequential | n_with_confidence | 100 | 100 | PASS |
| llm claude-haiku-5.5/latency_sequential | mean_stated_confidence | 0.8959 | 0.8959 | PASS |
| llm claude-haiku-5.5/latency_sequential | accuracy_on_those | 0.82 | 0.82 | PASS |
| llm claude-haiku-5.5/latency_sequential | overconfidence | 0.0759 | 0.0759 | PASS |
| llm claude-haiku-5.5/latency_sequential | cost_usd | 0.009878 | 0.009878 | PASS |
| llm claude-haiku-5.5/latency_sequential | cost_per_1000_calls_usd | 0.09878 | 0.09878 | PASS |
| llm claude-haiku-5.5/latency_sequential | mean_prompt_tokens | 865.55 | 865.55 | PASS |
| llm claude-haiku-5.5/latency_sequential | mean_output_tokens | 24.45 | 24.45 | PASS |
| llm claude-haiku-5.5/latency_sequential | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-haiku-5.5/latency_sequential | strict_json_reply_rate | 1 | 1 | PASS |
| llm claude-haiku-5.5/latency_sequential | ece_stated_confidence | 0.1135 | 0.1135 | PASS |
| llm claude-haiku-5.5/latency_sequential | accuracy_ci95 (Wilson) | [0.733326, 0.882998] | [0.733326, 0.882998] | PASS |
| llm claude-haiku-5.5/latency_sequential | latency median | 836.789 | 836.789 | PASS |
| llm claude-haiku-5.5/latency_sequential | latency p90 (index rule) | 952.361 | 952.361 | PASS |
| llm claude-haiku-5.5/latency_sequential | providers | ["Claude Platform on AWS"] | ["Claude Platform on AWS"] | PASS |
| llm gpt-6-luna/latency_sequential | n | 100 | 100 | PASS |
| llm gpt-6-luna/latency_sequential | n_missing | 0 | 0 | PASS |
| llm gpt-6-luna/latency_sequential | invalid_output | 0 | 0 | PASS |
| llm gpt-6-luna/latency_sequential | invalid_output_rate | 0 | 0 | PASS |
| llm gpt-6-luna/latency_sequential | accuracy_invalid_counted_wrong | 0.82 | 0.82 | PASS |
| llm gpt-6-luna/latency_sequential | n_with_confidence | 100 | 100 | PASS |
| llm gpt-6-luna/latency_sequential | mean_stated_confidence | 0.9457 | 0.9457 | PASS |
| llm gpt-6-luna/latency_sequential | accuracy_on_those | 0.82 | 0.82 | PASS |
| llm gpt-6-luna/latency_sequential | overconfidence | 0.1257 | 0.1257 | PASS |
| llm gpt-6-luna/latency_sequential | cost_usd | 0.0057536 | 0.0057536 | PASS |
| llm gpt-6-luna/latency_sequential | cost_per_1000_calls_usd | 0.057536 | 0.057536 | PASS |
| llm gpt-6-luna/latency_sequential | mean_prompt_tokens | 490.41 | 490.41 | PASS |
| llm gpt-6-luna/latency_sequential | mean_output_tokens | 16.99 | 16.99 | PASS |
| llm gpt-6-luna/latency_sequential | reasoning_tokens_total | 0 | 0 | PASS |
| llm gpt-6-luna/latency_sequential | strict_json_reply_rate | 1 | 1 | PASS |
| llm gpt-6-luna/latency_sequential | ece_stated_confidence | 0.1499 | 0.1499 | PASS |
| llm gpt-6-luna/latency_sequential | accuracy_ci95 (Wilson) | [0.733326, 0.882998] | [0.733326, 0.882998] | PASS |
| llm gpt-6-luna/latency_sequential | latency median | 1193.5 | 1193.5 | PASS |
| llm gpt-6-luna/latency_sequential | latency p90 (index rule) | 1725.97 | 1725.97 | PASS |
| llm gpt-6-luna/latency_sequential | providers | ["OpenAI"] | ["OpenAI"] | PASS |
| llm gpt-6-luna/banking77_clean_s250 | n | 250 | 250 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | n_missing | 0 | 0 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | invalid_output | 2 | 2 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | invalid_output_rate | 0.008 | 0.008 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | accuracy_invalid_counted_wrong | 0.756 | 0.756 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | n_with_confidence | 248 | 248 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | mean_stated_confidence | 0.883911 | 0.883911 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | accuracy_on_those | 0.762097 | 0.762097 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | overconfidence | 0.121815 | 0.121815 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | cost_usd | 0.0145458 | 0.0145458 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | cost_per_1000_calls_usd | 0.0581832 | 0.0581832 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | mean_prompt_tokens | 491.392 | 491.392 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | mean_output_tokens | 18.088 | 18.088 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | ece_stated_confidence | 0.149234 | 0.149234 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | accuracy_ci95 (Wilson) | [0.699149, 0.805103] | [0.699149, 0.805103] | PASS |
| llm gpt-6-luna/banking77_clean_s250 | latency median | 1206.82 | 1206.82 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | latency p90 (index rule) | 1896 | 1896 | PASS |
| llm gpt-6-luna/banking77_clean_s250 | providers | ["OpenAI"] | ["OpenAI"] | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | n | 250 | 250 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | n_missing | 0 | 0 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | invalid_output | 0 | 0 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | invalid_output_rate | 0 | 0 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | accuracy_invalid_counted_wrong | 0.748 | 0.748 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | n_with_confidence | 250 | 250 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | mean_stated_confidence | 0.88016 | 0.88016 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | accuracy_on_those | 0.748 | 0.748 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | overconfidence | 0.13216 | 0.13216 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | cost_usd | 0.0146036 | 0.0146036 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | cost_per_1000_calls_usd | 0.0584144 | 0.0584144 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | mean_prompt_tokens | 493.504 | 493.504 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | mean_output_tokens | 18.128 | 18.128 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | ece_stated_confidence | 0.13744 | 0.13744 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | accuracy_ci95 (Wilson) | [0.690706, 0.797788] | [0.690706, 0.797788] | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | latency median | 1216.72 | 1216.72 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | latency p90 (index rule) | 1789.86 | 1789.86 | PASS |
| llm gpt-6-luna/banking77_swap2_s250 | providers | ["OpenAI"] | ["OpenAI"] | PASS |
| llm gpt-6-luna/injection_banking_s250 | n | 250 | 250 | PASS |
| llm gpt-6-luna/injection_banking_s250 | n_missing | 0 | 0 | PASS |
| llm gpt-6-luna/injection_banking_s250 | invalid_output | 2 | 2 | PASS |
| llm gpt-6-luna/injection_banking_s250 | invalid_output_rate | 0.008 | 0.008 | PASS |
| llm gpt-6-luna/injection_banking_s250 | accuracy_invalid_counted_wrong | 0.012 | 0.012 | PASS |
| llm gpt-6-luna/injection_banking_s250 | n_with_confidence | 248 | 248 | PASS |
| llm gpt-6-luna/injection_banking_s250 | mean_stated_confidence | 0.994032 | 0.994032 | PASS |
| llm gpt-6-luna/injection_banking_s250 | accuracy_on_those | 0.0120968 | 0.0120968 | PASS |
| llm gpt-6-luna/injection_banking_s250 | overconfidence | 0.981935 | 0.981935 | PASS |
| llm gpt-6-luna/injection_banking_s250 | followed_injected_label | 0.976 | 0.976 | PASS |
| llm gpt-6-luna/injection_banking_s250 | cost_usd | 0.0148939 | 0.0148939 | PASS |
| llm gpt-6-luna/injection_banking_s250 | cost_per_1000_calls_usd | 0.0595756 | 0.0595756 | PASS |
| llm gpt-6-luna/injection_banking_s250 | mean_prompt_tokens | 507.956 | 507.956 | PASS |
| llm gpt-6-luna/injection_banking_s250 | mean_output_tokens | 17.56 | 17.56 | PASS |
| llm gpt-6-luna/injection_banking_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm gpt-6-luna/injection_banking_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm gpt-6-luna/injection_banking_s250 | ece_stated_confidence | 0.981935 | 0.981935 | PASS |
| llm gpt-6-luna/injection_banking_s250 | accuracy_ci95 (Wilson) | [0.00408932, 0.0346808] | [0.00408932, 0.0346808] | PASS |
| llm gpt-6-luna/injection_banking_s250 | latency median | 1154.77 | 1154.77 | PASS |
| llm gpt-6-luna/injection_banking_s250 | latency p90 (index rule) | 1560.95 | 1560.95 | PASS |
| llm gpt-6-luna/injection_banking_s250 | providers | ["OpenAI"] | ["OpenAI"] | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | n | 250 | 250 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | n_missing | 0 | 0 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | invalid_output | 1 | 1 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | invalid_output_rate | 0.004 | 0.004 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | accuracy_invalid_counted_wrong | 0.756 | 0.756 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | n_with_confidence | 249 | 249 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | mean_stated_confidence | 0.900402 | 0.900402 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | accuracy_on_those | 0.759036 | 0.759036 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | overconfidence | 0.141365 | 0.141365 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | cost_usd | 0.0149323 | 0.0149323 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | cost_per_1000_calls_usd | 0.0597292 | 0.0597292 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | mean_prompt_tokens | 506.632 | 506.632 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | mean_output_tokens | 18.132 | 18.132 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | ece_stated_confidence | 0.163133 | 0.163133 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | accuracy_ci95 (Wilson) | [0.699149, 0.805103] | [0.699149, 0.805103] | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | latency median | 1281.02 | 1281.02 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | latency p90 (index rule) | 1992.28 | 1992.28 | PASS |
| llm gpt-6-luna/injection_control_banking_s250 | providers | ["OpenAI"] | ["OpenAI"] | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | n | 250 | 250 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | n_missing | 0 | 0 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | invalid_output | 0 | 0 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | invalid_output_rate | 0 | 0 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | accuracy_invalid_counted_wrong | 0.48 | 0.48 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | n_with_confidence | 250 | 250 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | mean_stated_confidence | 0.93104 | 0.93104 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | accuracy_on_those | 0.48 | 0.48 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | overconfidence | 0.45104 | 0.45104 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | followed_injected_label | 0.404 | 0.404 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | cost_usd | 0.0148484 | 0.0148484 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | cost_per_1000_calls_usd | 0.0593936 | 0.0593936 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | mean_prompt_tokens | 501.956 | 501.956 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | mean_output_tokens | 18.396 | 18.396 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | ece_stated_confidence | 0.45104 | 0.45104 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | accuracy_ci95 (Wilson) | [0.418842, 0.541763] | [0.418842, 0.541763] | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | latency median | 1518.37 | 1518.37 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | latency p90 (index rule) | 2689.1 | 2689.1 | PASS |
| llm gpt-6-luna/injection_casual_banking_s250 | providers | ["OpenAI"] | ["OpenAI"] | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | n | 250 | 250 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | n_missing | 0 | 0 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | invalid_output | 1 | 1 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | invalid_output_rate | 0.004 | 0.004 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | accuracy_invalid_counted_wrong | 0.68 | 0.68 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | n_with_confidence | 249 | 249 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | mean_stated_confidence | 0.950361 | 0.950361 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | accuracy_on_those | 0.682731 | 0.682731 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | overconfidence | 0.267631 | 0.267631 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | followed_injected_label | 0.116 | 0.116 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | cost_usd | 0.0154269 | 0.0154269 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | cost_per_1000_calls_usd | 0.0617076 | 0.0617076 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | mean_prompt_tokens | 525.956 | 525.956 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | mean_output_tokens | 18.224 | 18.224 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | ece_stated_confidence | 0.268675 | 0.268675 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | accuracy_ci95 (Wilson) | [0.619827, 0.734725] | [0.619827, 0.734725] | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | latency median | 1139.72 | 1139.72 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | latency p90 (index rule) | 1435.16 | 1435.16 | PASS |
| llm gpt-6-luna/injection_casual_defended_banking_s250 | providers | ["OpenAI"] | ["OpenAI"] | PASS |
| llm qwen3.8-flash/banking77_clean | n | 3,080 | 3,080 | PASS |
| llm qwen3.8-flash/banking77_clean | n_missing | 0 | 0 | PASS |
| llm qwen3.8-flash/banking77_clean | invalid_output | 12 | 12 | PASS |
| llm qwen3.8-flash/banking77_clean | invalid_output_rate | 0.0038961 | 0.0038961 | PASS |
| llm qwen3.8-flash/banking77_clean | accuracy_invalid_counted_wrong | 0.793506 | 0.793506 | PASS |
| llm qwen3.8-flash/banking77_clean | n_with_confidence | 3,068 | 3,068 | PASS |
| llm qwen3.8-flash/banking77_clean | mean_stated_confidence | 0.951874 | 0.951874 | PASS |
| llm qwen3.8-flash/banking77_clean | accuracy_on_those | 0.79661 | 0.79661 | PASS |
| llm qwen3.8-flash/banking77_clean | overconfidence | 0.155264 | 0.155264 | PASS |
| llm qwen3.8-flash/banking77_clean | cost_usd | 0.147195 | 0.147195 | PASS |
| llm qwen3.8-flash/banking77_clean | cost_per_1000_calls_usd | 0.0477905 | 0.0477905 | PASS |
| llm qwen3.8-flash/banking77_clean | mean_prompt_tokens | 488.365 | 488.365 | PASS |
| llm qwen3.8-flash/banking77_clean | mean_output_tokens | 18.8081 | 18.8081 | PASS |
| llm qwen3.8-flash/banking77_clean | reasoning_tokens_total | 0 | 0 | PASS |
| llm qwen3.8-flash/banking77_clean | mean_token_probability | 0.88529 | 0.88529 | PASS |
| llm qwen3.8-flash/banking77_clean | n_token_probability | 3,068 | 3,068 | PASS |
| llm qwen3.8-flash/banking77_clean | strict_json_reply_rate | 1 | 1 | PASS |
| llm qwen3.8-flash/banking77_clean | ece_stated_confidence | 0.155264 | 0.155264 | PASS |
| llm qwen3.8-flash/banking77_clean | ece_token_probability | 0.0886802 | 0.0886802 | PASS |
| llm qwen3.8-flash/banking77_clean | accuracy_ci95 (Wilson) | [0.77885, 0.807432] | [0.77885, 0.807432] | PASS |
| llm qwen3.8-flash/banking77_clean | latency median | 1271.01 | 1271.01 | PASS |
| llm qwen3.8-flash/banking77_clean | latency p90 (index rule) | 2167.43 | 2167.43 | PASS |
| llm qwen3.8-flash/banking77_clean | providers | ["Alibaba"] | ["Alibaba"] | PASS |
| llm qwen3.8-flash/banking77_swap2 | n | 3,080 | 3,080 | PASS |
| llm qwen3.8-flash/banking77_swap2 | n_missing | 0 | 0 | PASS |
| llm qwen3.8-flash/banking77_swap2 | invalid_output | 12 | 12 | PASS |
| llm qwen3.8-flash/banking77_swap2 | invalid_output_rate | 0.0038961 | 0.0038961 | PASS |
| llm qwen3.8-flash/banking77_swap2 | accuracy_invalid_counted_wrong | 0.778571 | 0.778571 | PASS |
| llm qwen3.8-flash/banking77_swap2 | n_with_confidence | 3,068 | 3,068 | PASS |
| llm qwen3.8-flash/banking77_swap2 | mean_stated_confidence | 0.948866 | 0.948866 | PASS |
| llm qwen3.8-flash/banking77_swap2 | accuracy_on_those | 0.781617 | 0.781617 | PASS |
| llm qwen3.8-flash/banking77_swap2 | overconfidence | 0.167249 | 0.167249 | PASS |
| llm qwen3.8-flash/banking77_swap2 | cost_usd | 0.148355 | 0.148355 | PASS |
| llm qwen3.8-flash/banking77_swap2 | cost_per_1000_calls_usd | 0.0481671 | 0.0481671 | PASS |
| llm qwen3.8-flash/banking77_swap2 | mean_prompt_tokens | 490.51 | 490.51 | PASS |
| llm qwen3.8-flash/banking77_swap2 | mean_output_tokens | 18.9484 | 18.9484 | PASS |
| llm qwen3.8-flash/banking77_swap2 | reasoning_tokens_total | 0 | 0 | PASS |
| llm qwen3.8-flash/banking77_swap2 | mean_token_probability | 0.871576 | 0.871576 | PASS |
| llm qwen3.8-flash/banking77_swap2 | n_token_probability | 3,068 | 3,068 | PASS |
| llm qwen3.8-flash/banking77_swap2 | strict_json_reply_rate | 1 | 1 | PASS |
| llm qwen3.8-flash/banking77_swap2 | ece_stated_confidence | 0.167249 | 0.167249 | PASS |
| llm qwen3.8-flash/banking77_swap2 | ece_token_probability | 0.0899598 | 0.0899598 | PASS |
| llm qwen3.8-flash/banking77_swap2 | accuracy_ci95 (Wilson) | [0.763566, 0.792883] | [0.763566, 0.792883] | PASS |
| llm qwen3.8-flash/banking77_swap2 | latency median | 1249.12 | 1249.12 | PASS |
| llm qwen3.8-flash/banking77_swap2 | latency p90 (index rule) | 1824.63 | 1824.63 | PASS |
| llm qwen3.8-flash/banking77_swap2 | providers | ["Alibaba"] | ["Alibaba"] | PASS |
| llm qwen3.8-flash/banking77_mixed3 | n | 3,080 | 3,080 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | n_missing | 0 | 0 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | invalid_output | 11 | 11 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | invalid_output_rate | 0.00357143 | 0.00357143 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | accuracy_invalid_counted_wrong | 0.769156 | 0.769156 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | n_with_confidence | 3,069 | 3,069 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | mean_stated_confidence | 0.945516 | 0.945516 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | accuracy_on_those | 0.771913 | 0.771913 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | overconfidence | 0.173604 | 0.173604 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | cost_usd | 0.148519 | 0.148519 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | cost_per_1000_calls_usd | 0.0482204 | 0.0482204 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | mean_prompt_tokens | 490.889 | 490.889 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | mean_output_tokens | 18.9172 | 18.9172 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | reasoning_tokens_total | 0 | 0 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | mean_token_probability | 0.861655 | 0.861655 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | n_token_probability | 3,069 | 3,069 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | strict_json_reply_rate | 1 | 1 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | ece_stated_confidence | 0.173832 | 0.173832 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | ece_token_probability | 0.0897419 | 0.0897419 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | accuracy_ci95 (Wilson) | [0.753945, 0.783696] | [0.753945, 0.783696] | PASS |
| llm qwen3.8-flash/banking77_mixed3 | latency median | 1244.21 | 1244.21 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | latency p90 (index rule) | 1728 | 1728 | PASS |
| llm qwen3.8-flash/banking77_mixed3 | providers | ["Alibaba"] | ["Alibaba"] | PASS |
| llm qwen3.8-flash/spam_clean | n | 836 | 836 | PASS |
| llm qwen3.8-flash/spam_clean | n_missing | 0 | 0 | PASS |
| llm qwen3.8-flash/spam_clean | invalid_output | 0 | 0 | PASS |
| llm qwen3.8-flash/spam_clean | invalid_output_rate | 0 | 0 | PASS |
| llm qwen3.8-flash/spam_clean | accuracy_invalid_counted_wrong | 0.799043 | 0.799043 | PASS |
| llm qwen3.8-flash/spam_clean | n_with_confidence | 836 | 836 | PASS |
| llm qwen3.8-flash/spam_clean | mean_stated_confidence | 0.941699 | 0.941699 | PASS |
| llm qwen3.8-flash/spam_clean | accuracy_on_those | 0.799043 | 0.799043 | PASS |
| llm qwen3.8-flash/spam_clean | overconfidence | 0.142656 | 0.142656 | PASS |
| llm qwen3.8-flash/spam_clean | cost_usd | 0.0161538 | 0.0161538 | PASS |
| llm qwen3.8-flash/spam_clean | cost_per_1000_calls_usd | 0.0193227 | 0.0193227 | PASS |
| llm qwen3.8-flash/spam_clean | mean_prompt_tokens | 95.311 | 95.311 | PASS |
| llm qwen3.8-flash/spam_clean | mean_output_tokens | 10.6938 | 10.6938 | PASS |
| llm qwen3.8-flash/spam_clean | reasoning_tokens_total | 0 | 0 | PASS |
| llm qwen3.8-flash/spam_clean | strict_json_reply_rate | 1 | 1 | PASS |
| llm qwen3.8-flash/spam_clean | ece_stated_confidence | 0.158206 | 0.158206 | PASS |
| llm qwen3.8-flash/spam_clean | accuracy_ci95 (Wilson) | [0.77054, 0.824811] | [0.77054, 0.824811] | PASS |
| llm qwen3.8-flash/spam_clean | latency median | 1283.71 | 1283.71 | PASS |
| llm qwen3.8-flash/spam_clean | latency p90 (index rule) | 6186.61 | 6186.61 | PASS |
| llm qwen3.8-flash/spam_clean | providers | ["Alibaba"] | ["Alibaba"] | PASS |
| llm qwen3.8-flash/spam_swap2 | n | 836 | 836 | PASS |
| llm qwen3.8-flash/spam_swap2 | n_missing | 0 | 0 | PASS |
| llm qwen3.8-flash/spam_swap2 | invalid_output | 0 | 0 | PASS |
| llm qwen3.8-flash/spam_swap2 | invalid_output_rate | 0 | 0 | PASS |
| llm qwen3.8-flash/spam_swap2 | accuracy_invalid_counted_wrong | 0.679426 | 0.679426 | PASS |
| llm qwen3.8-flash/spam_swap2 | n_with_confidence | 836 | 836 | PASS |
| llm qwen3.8-flash/spam_swap2 | mean_stated_confidence | 0.940778 | 0.940778 | PASS |
| llm qwen3.8-flash/spam_swap2 | accuracy_on_those | 0.679426 | 0.679426 | PASS |
| llm qwen3.8-flash/spam_swap2 | overconfidence | 0.261352 | 0.261352 | PASS |
| llm qwen3.8-flash/spam_swap2 | cost_usd | 0.0163783 | 0.0163783 | PASS |
| llm qwen3.8-flash/spam_swap2 | cost_per_1000_calls_usd | 0.0195913 | 0.0195913 | PASS |
| llm qwen3.8-flash/spam_swap2 | mean_prompt_tokens | 96.9964 | 96.9964 | PASS |
| llm qwen3.8-flash/spam_swap2 | mean_output_tokens | 10.7273 | 10.7273 | PASS |
| llm qwen3.8-flash/spam_swap2 | reasoning_tokens_total | 0 | 0 | PASS |
| llm qwen3.8-flash/spam_swap2 | strict_json_reply_rate | 1 | 1 | PASS |
| llm qwen3.8-flash/spam_swap2 | ece_stated_confidence | 0.299988 | 0.299988 | PASS |
| llm qwen3.8-flash/spam_swap2 | accuracy_ci95 (Wilson) | [0.647031, 0.710179] | [0.647031, 0.710179] | PASS |
| llm qwen3.8-flash/spam_swap2 | latency median | 1074.74 | 1074.74 | PASS |
| llm qwen3.8-flash/spam_swap2 | latency p90 (index rule) | 5877.24 | 5877.24 | PASS |
| llm qwen3.8-flash/spam_swap2 | providers | ["Alibaba"] | ["Alibaba"] | PASS |
| llm qwen3.8-flash/injection_banking | n | 1,000 | 1,000 | PASS |
| llm qwen3.8-flash/injection_banking | n_missing | 0 | 0 | PASS |
| llm qwen3.8-flash/injection_banking | invalid_output | 19 | 19 | PASS |
| llm qwen3.8-flash/injection_banking | invalid_output_rate | 0.019 | 0.019 | PASS |
| llm qwen3.8-flash/injection_banking | accuracy_invalid_counted_wrong | 0.244 | 0.244 | PASS |
| llm qwen3.8-flash/injection_banking | n_with_confidence | 981 | 981 | PASS |
| llm qwen3.8-flash/injection_banking | mean_stated_confidence | 0.980989 | 0.980989 | PASS |
| llm qwen3.8-flash/injection_banking | accuracy_on_those | 0.248726 | 0.248726 | PASS |
| llm qwen3.8-flash/injection_banking | overconfidence | 0.732263 | 0.732263 | PASS |
| llm qwen3.8-flash/injection_banking | followed_injected_label | 0.715 | 0.715 | PASS |
| llm qwen3.8-flash/injection_banking | cost_usd | 0.0496153 | 0.0496153 | PASS |
| llm qwen3.8-flash/injection_banking | cost_per_1000_calls_usd | 0.0496153 | 0.0496153 | PASS |
| llm qwen3.8-flash/injection_banking | mean_prompt_tokens | 504.867 | 504.867 | PASS |
| llm qwen3.8-flash/injection_banking | mean_output_tokens | 17.424 | 17.424 | PASS |
| llm qwen3.8-flash/injection_banking | reasoning_tokens_total | 0 | 0 | PASS |
| llm qwen3.8-flash/injection_banking | mean_token_probability | 0.697182 | 0.697182 | PASS |
| llm qwen3.8-flash/injection_banking | n_token_probability | 981 | 981 | PASS |
| llm qwen3.8-flash/injection_banking | strict_json_reply_rate | 1 | 1 | PASS |
| llm qwen3.8-flash/injection_banking | ece_stated_confidence | 0.732263 | 0.732263 | PASS |
| llm qwen3.8-flash/injection_banking | ece_token_probability | 0.448456 | 0.448456 | PASS |
| llm qwen3.8-flash/injection_banking | accuracy_ci95 (Wilson) | [0.218393, 0.271566] | [0.218393, 0.271566] | PASS |
| llm qwen3.8-flash/injection_banking | latency median | 1022.51 | 1022.51 | PASS |
| llm qwen3.8-flash/injection_banking | latency p90 (index rule) | 1619.17 | 1619.17 | PASS |
| llm qwen3.8-flash/injection_banking | providers | ["Alibaba"] | ["Alibaba"] | PASS |
| llm qwen3.8-flash/injection_control_banking | n | 1,000 | 1,000 | PASS |
| llm qwen3.8-flash/injection_control_banking | n_missing | 0 | 0 | PASS |
| llm qwen3.8-flash/injection_control_banking | invalid_output | 5 | 5 | PASS |
| llm qwen3.8-flash/injection_control_banking | invalid_output_rate | 0.005 | 0.005 | PASS |
| llm qwen3.8-flash/injection_control_banking | accuracy_invalid_counted_wrong | 0.797 | 0.797 | PASS |
| llm qwen3.8-flash/injection_control_banking | n_with_confidence | 995 | 995 | PASS |
| llm qwen3.8-flash/injection_control_banking | mean_stated_confidence | 0.953126 | 0.953126 | PASS |
| llm qwen3.8-flash/injection_control_banking | accuracy_on_those | 0.801005 | 0.801005 | PASS |
| llm qwen3.8-flash/injection_control_banking | overconfidence | 0.152121 | 0.152121 | PASS |
| llm qwen3.8-flash/injection_control_banking | cost_usd | 0.0500841 | 0.0500841 | PASS |
| llm qwen3.8-flash/injection_control_banking | cost_per_1000_calls_usd | 0.0500841 | 0.0500841 | PASS |
| llm qwen3.8-flash/injection_control_banking | mean_prompt_tokens | 503.59 | 503.59 | PASS |
| llm qwen3.8-flash/injection_control_banking | mean_output_tokens | 18.902 | 18.902 | PASS |
| llm qwen3.8-flash/injection_control_banking | reasoning_tokens_total | 0 | 0 | PASS |
| llm qwen3.8-flash/injection_control_banking | mean_token_probability | 0.874436 | 0.874436 | PASS |
| llm qwen3.8-flash/injection_control_banking | n_token_probability | 995 | 995 | PASS |
| llm qwen3.8-flash/injection_control_banking | strict_json_reply_rate | 1 | 1 | PASS |
| llm qwen3.8-flash/injection_control_banking | ece_stated_confidence | 0.152121 | 0.152121 | PASS |
| llm qwen3.8-flash/injection_control_banking | ece_token_probability | 0.0749959 | 0.0749959 | PASS |
| llm qwen3.8-flash/injection_control_banking | accuracy_ci95 (Wilson) | [0.770955, 0.820772] | [0.770955, 0.820772] | PASS |
| llm qwen3.8-flash/injection_control_banking | latency median | 989.155 | 989.155 | PASS |
| llm qwen3.8-flash/injection_control_banking | latency p90 (index rule) | 1375.64 | 1375.64 | PASS |
| llm qwen3.8-flash/injection_control_banking | providers | ["Alibaba"] | ["Alibaba"] | PASS |
| llm qwen3.8-flash/injection_casual_banking | n | 1,000 | 1,000 | PASS |
| llm qwen3.8-flash/injection_casual_banking | n_missing | 0 | 0 | PASS |
| llm qwen3.8-flash/injection_casual_banking | invalid_output | 3 | 3 | PASS |
| llm qwen3.8-flash/injection_casual_banking | invalid_output_rate | 0.003 | 0.003 | PASS |
| llm qwen3.8-flash/injection_casual_banking | accuracy_invalid_counted_wrong | 0.577 | 0.577 | PASS |
| llm qwen3.8-flash/injection_casual_banking | n_with_confidence | 997 | 997 | PASS |
| llm qwen3.8-flash/injection_casual_banking | mean_stated_confidence | 0.944213 | 0.944213 | PASS |
| llm qwen3.8-flash/injection_casual_banking | accuracy_on_those | 0.578736 | 0.578736 | PASS |
| llm qwen3.8-flash/injection_casual_banking | overconfidence | 0.365476 | 0.365476 | PASS |
| llm qwen3.8-flash/injection_casual_banking | followed_injected_label | 0.326 | 0.326 | PASS |
| llm qwen3.8-flash/injection_casual_banking | cost_usd | 0.0496323 | 0.0496323 | PASS |
| llm qwen3.8-flash/injection_casual_banking | cost_per_1000_calls_usd | 0.0496323 | 0.0496323 | PASS |
| llm qwen3.8-flash/injection_casual_banking | mean_prompt_tokens | 498.867 | 498.867 | PASS |
| llm qwen3.8-flash/injection_casual_banking | mean_output_tokens | 19.375 | 19.375 | PASS |
| llm qwen3.8-flash/injection_casual_banking | reasoning_tokens_total | 0 | 0 | PASS |
| llm qwen3.8-flash/injection_casual_banking | mean_token_probability | 0.766453 | 0.766453 | PASS |
| llm qwen3.8-flash/injection_casual_banking | n_token_probability | 997 | 997 | PASS |
| llm qwen3.8-flash/injection_casual_banking | strict_json_reply_rate | 1 | 1 | PASS |
| llm qwen3.8-flash/injection_casual_banking | ece_stated_confidence | 0.365476 | 0.365476 | PASS |
| llm qwen3.8-flash/injection_casual_banking | ece_token_probability | 0.190724 | 0.190724 | PASS |
| llm qwen3.8-flash/injection_casual_banking | accuracy_ci95 (Wilson) | [0.546142, 0.607268] | [0.546142, 0.607268] | PASS |
| llm qwen3.8-flash/injection_casual_banking | latency median | 992.113 | 992.113 | PASS |
| llm qwen3.8-flash/injection_casual_banking | latency p90 (index rule) | 1447.58 | 1447.58 | PASS |
| llm qwen3.8-flash/injection_casual_banking | providers | ["Alibaba"] | ["Alibaba"] | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | n | 1,000 | 1,000 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | n_missing | 0 | 0 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | invalid_output | 6 | 6 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | invalid_output_rate | 0.006 | 0.006 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | accuracy_invalid_counted_wrong | 0.708 | 0.708 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | n_with_confidence | 994 | 994 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | mean_stated_confidence | 0.949899 | 0.949899 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | accuracy_on_those | 0.712274 | 0.712274 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | overconfidence | 0.237626 | 0.237626 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | followed_injected_label | 0.133 | 0.133 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | cost_usd | 0.0528859 | 0.0528859 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | cost_per_1000_calls_usd | 0.0528859 | 0.0528859 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | mean_prompt_tokens | 522.867 | 522.867 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | mean_output_tokens | 18.492 | 18.492 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | reasoning_tokens_total | 0 | 0 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | mean_token_probability | 0.818212 | 0.818212 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | n_token_probability | 994 | 994 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | strict_json_reply_rate | 0.997 | 0.997 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | ece_stated_confidence | 0.237626 | 0.237626 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | ece_token_probability | 0.105938 | 0.105938 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | accuracy_ci95 (Wilson) | [0.679066, 0.735342] | [0.679066, 0.735342] | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | latency median | 1005.49 | 1005.49 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | latency p90 (index rule) | 1488.07 | 1488.07 | PASS |
| llm qwen3.8-flash/injection_casual_defended_banking | providers | ["Alibaba"] | ["Alibaba"] | PASS |
| llm qwen3.8-flash/latency_sequential | n | 100 | 100 | PASS |
| llm qwen3.8-flash/latency_sequential | n_missing | 0 | 0 | PASS |
| llm qwen3.8-flash/latency_sequential | invalid_output | 0 | 0 | PASS |
| llm qwen3.8-flash/latency_sequential | invalid_output_rate | 0 | 0 | PASS |
| llm qwen3.8-flash/latency_sequential | accuracy_invalid_counted_wrong | 0.64 | 0.64 | PASS |
| llm qwen3.8-flash/latency_sequential | n_with_confidence | 100 | 100 | PASS |
| llm qwen3.8-flash/latency_sequential | mean_stated_confidence | 0.9559 | 0.9559 | PASS |
| llm qwen3.8-flash/latency_sequential | accuracy_on_those | 0.64 | 0.64 | PASS |
| llm qwen3.8-flash/latency_sequential | overconfidence | 0.3159 | 0.3159 | PASS |
| llm qwen3.8-flash/latency_sequential | cost_usd | 0.0047086 | 0.0047086 | PASS |
| llm qwen3.8-flash/latency_sequential | cost_per_1000_calls_usd | 0.047086 | 0.047086 | PASS |
| llm qwen3.8-flash/latency_sequential | mean_prompt_tokens | 487.61 | 487.61 | PASS |
| llm qwen3.8-flash/latency_sequential | mean_output_tokens | 17.55 | 17.55 | PASS |
| llm qwen3.8-flash/latency_sequential | reasoning_tokens_total | 0 | 0 | PASS |
| llm qwen3.8-flash/latency_sequential | mean_token_probability | 0.913603 | 0.913603 | PASS |
| llm qwen3.8-flash/latency_sequential | n_token_probability | 100 | 100 | PASS |
| llm qwen3.8-flash/latency_sequential | strict_json_reply_rate | 1 | 1 | PASS |
| llm qwen3.8-flash/latency_sequential | ece_stated_confidence | 0.3159 | 0.3159 | PASS |
| llm qwen3.8-flash/latency_sequential | ece_token_probability | 0.289881 | 0.289881 | PASS |
| llm qwen3.8-flash/latency_sequential | accuracy_ci95 (Wilson) | [0.542354, 0.727288] | [0.542354, 0.727288] | PASS |
| llm qwen3.8-flash/latency_sequential | latency median | 983.565 | 983.565 | PASS |
| llm qwen3.8-flash/latency_sequential | latency p90 (index rule) | 1253.32 | 1253.32 | PASS |
| llm qwen3.8-flash/latency_sequential | providers | ["Alibaba"] | ["Alibaba"] | PASS |
| llm gpt-6.1-terra/latency_sequential | n | 100 | 100 | PASS |
| llm gpt-6.1-terra/latency_sequential | n_missing | 0 | 0 | PASS |
| llm gpt-6.1-terra/latency_sequential | invalid_output | 0 | 0 | PASS |
| llm gpt-6.1-terra/latency_sequential | invalid_output_rate | 0 | 0 | PASS |
| llm gpt-6.1-terra/latency_sequential | accuracy_invalid_counted_wrong | 0.88 | 0.88 | PASS |
| llm gpt-6.1-terra/latency_sequential | n_with_confidence | 100 | 100 | PASS |
| llm gpt-6.1-terra/latency_sequential | mean_stated_confidence | 0.9397 | 0.9397 | PASS |
| llm gpt-6.1-terra/latency_sequential | accuracy_on_those | 0.88 | 0.88 | PASS |
| llm gpt-6.1-terra/latency_sequential | overconfidence | 0.0597 | 0.0597 | PASS |
| llm gpt-6.1-terra/latency_sequential | cost_usd | 0.057526 | 0.057526 | PASS |
| llm gpt-6.1-terra/latency_sequential | cost_per_1000_calls_usd | 0.57526 | 0.57526 | PASS |
| llm gpt-6.1-terra/latency_sequential | mean_prompt_tokens | 490.41 | 490.41 | PASS |
| llm gpt-6.1-terra/latency_sequential | mean_output_tokens | 16.97 | 16.97 | PASS |
| llm gpt-6.1-terra/latency_sequential | reasoning_tokens_total | 0 | 0 | PASS |
| llm gpt-6.1-terra/latency_sequential | strict_json_reply_rate | 1 | 1 | PASS |
| llm gpt-6.1-terra/latency_sequential | ece_stated_confidence | 0.0957 | 0.0957 | PASS |
| llm gpt-6.1-terra/latency_sequential | accuracy_ci95 (Wilson) | [0.801879, 0.930006] | [0.801879, 0.930006] | PASS |
| llm gpt-6.1-terra/latency_sequential | latency median | 1269.46 | 1269.46 | PASS |
| llm gpt-6.1-terra/latency_sequential | latency p90 (index rule) | 1770.21 | 1770.21 | PASS |
| llm gpt-6.1-terra/latency_sequential | providers | ["OpenAI"] | ["OpenAI"] | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | n | 250 | 250 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | n_missing | 0 | 0 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | invalid_output | 0 | 0 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | invalid_output_rate | 0 | 0 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | accuracy_invalid_counted_wrong | 0.816 | 0.816 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | n_with_confidence | 250 | 250 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | mean_stated_confidence | 0.889424 | 0.889424 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | accuracy_on_those | 0.816 | 0.816 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | overconfidence | 0.073424 | 0.073424 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | cost_usd | 0.145518 | 0.145518 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | cost_per_1000_calls_usd | 0.582072 | 0.582072 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | mean_prompt_tokens | 491.392 | 491.392 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | mean_output_tokens | 18.136 | 18.136 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | ece_stated_confidence | 0.081264 | 0.081264 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | accuracy_ci95 (Wilson) | [0.763311, 0.859124] | [0.763311, 0.859124] | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | latency median | 1255.86 | 1255.86 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | latency p90 (index rule) | 1654.13 | 1654.13 | PASS |
| llm gpt-6.1-terra/banking77_clean_s250 | providers | ["OpenAI"] | ["OpenAI"] | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | n | 250 | 250 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | n_missing | 0 | 0 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | invalid_output | 0 | 0 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | invalid_output_rate | 0 | 0 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | accuracy_invalid_counted_wrong | 0.816 | 0.816 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | n_with_confidence | 250 | 250 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | mean_stated_confidence | 0.882492 | 0.882492 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | accuracy_on_those | 0.816 | 0.816 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | overconfidence | 0.066492 | 0.066492 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | cost_usd | 0.146051 | 0.146051 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | cost_per_1000_calls_usd | 0.584204 | 0.584204 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | mean_prompt_tokens | 493.504 | 493.504 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | mean_output_tokens | 18.14 | 18.14 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | ece_stated_confidence | 0.082252 | 0.082252 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | accuracy_ci95 (Wilson) | [0.763311, 0.859124] | [0.763311, 0.859124] | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | latency median | 1255.09 | 1255.09 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | latency p90 (index rule) | 1669.87 | 1669.87 | PASS |
| llm gpt-6.1-terra/banking77_swap2_s250 | providers | ["OpenAI"] | ["OpenAI"] | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | n | 250 | 250 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | n_missing | 0 | 0 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | invalid_output | 0 | 0 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | invalid_output_rate | 0 | 0 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | accuracy_invalid_counted_wrong | 0.812 | 0.812 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | n_with_confidence | 250 | 250 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | mean_stated_confidence | 0.91596 | 0.91596 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | accuracy_on_those | 0.812 | 0.812 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | overconfidence | 0.10396 | 0.10396 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | followed_injected_label | 0.012 | 0.012 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | cost_usd | 0.149669 | 0.149669 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | cost_per_1000_calls_usd | 0.598676 | 0.598676 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | mean_prompt_tokens | 507.956 | 507.956 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | mean_output_tokens | 18.144 | 18.144 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | ece_stated_confidence | 0.12716 | 0.12716 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | accuracy_ci95 (Wilson) | [0.758983, 0.855574] | [0.758983, 0.855574] | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | latency median | 1231.79 | 1231.79 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | latency p90 (index rule) | 1632.39 | 1632.39 | PASS |
| llm gpt-6.1-terra/injection_banking_s250 | providers | ["OpenAI"] | ["OpenAI"] | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | n | 250 | 250 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | n_missing | 0 | 0 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | invalid_output | 0 | 0 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | invalid_output_rate | 0 | 0 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | accuracy_invalid_counted_wrong | 0.82 | 0.82 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | n_with_confidence | 250 | 250 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | mean_stated_confidence | 0.893988 | 0.893988 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | accuracy_on_those | 0.82 | 0.82 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | overconfidence | 0.073988 | 0.073988 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | cost_usd | 0.149358 | 0.149358 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | cost_per_1000_calls_usd | 0.597432 | 0.597432 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | mean_prompt_tokens | 506.632 | 506.632 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | mean_output_tokens | 18.16 | 18.16 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | ece_stated_confidence | 0.096308 | 0.096308 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | accuracy_ci95 (Wilson) | [0.767648, 0.862667] | [0.767648, 0.862667] | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | latency median | 1218.81 | 1218.81 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | latency p90 (index rule) | 1542.97 | 1542.97 | PASS |
| llm gpt-6.1-terra/injection_control_banking_s250 | providers | ["OpenAI"] | ["OpenAI"] | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | n | 250 | 250 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | n_missing | 0 | 0 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | invalid_output | 0 | 0 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | invalid_output_rate | 0 | 0 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | accuracy_invalid_counted_wrong | 0.672 | 0.672 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | n_with_confidence | 250 | 250 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | mean_stated_confidence | 0.90332 | 0.90332 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | accuracy_on_those | 0.672 | 0.672 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | overconfidence | 0.23132 | 0.23132 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | followed_injected_label | 0.2 | 0.2 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | cost_usd | 0.148284 | 0.148284 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | cost_per_1000_calls_usd | 0.593136 | 0.593136 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | mean_prompt_tokens | 501.956 | 501.956 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | mean_output_tokens | 18.236 | 18.236 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | ece_stated_confidence | 0.23132 | 0.23132 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | accuracy_ci95 (Wilson) | [0.611584, 0.727211] | [0.611584, 0.727211] | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | latency median | 1244.43 | 1244.43 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | latency p90 (index rule) | 1631.23 | 1631.23 | PASS |
| llm gpt-6.1-terra/injection_casual_banking_s250 | providers | ["OpenAI"] | ["OpenAI"] | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | n | 250 | 250 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | n_missing | 0 | 0 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | invalid_output | 0 | 0 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | invalid_output_rate | 0 | 0 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | accuracy_invalid_counted_wrong | 0.804 | 0.804 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | n_with_confidence | 250 | 250 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | mean_stated_confidence | 0.906116 | 0.906116 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | accuracy_on_those | 0.804 | 0.804 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | overconfidence | 0.102116 | 0.102116 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | followed_injected_label | 0.012 | 0.012 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | cost_usd | 0.154219 | 0.154219 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | cost_per_1000_calls_usd | 0.616876 | 0.616876 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | mean_prompt_tokens | 525.956 | 525.956 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | mean_output_tokens | 18.184 | 18.184 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | ece_stated_confidence | 0.102676 | 0.102676 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | accuracy_ci95 (Wilson) | [0.750349, 0.84845] | [0.750349, 0.84845] | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | latency median | 1253.27 | 1253.27 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | latency p90 (index rule) | 1583.86 | 1583.86 | PASS |
| llm gpt-6.1-terra/injection_casual_defended_banking_s250 | providers | ["OpenAI"] | ["OpenAI"] | PASS |
| llm claude-sonnet-5.5/banking77_clean | n | 3,080 | 3,080 | PASS |
| llm claude-sonnet-5.5/banking77_clean | n_missing | 0 | 0 | PASS |
| llm claude-sonnet-5.5/banking77_clean | invalid_output | 0 | 0 | PASS |
| llm claude-sonnet-5.5/banking77_clean | invalid_output_rate | 0 | 0 | PASS |
| llm claude-sonnet-5.5/banking77_clean | accuracy_invalid_counted_wrong | 0.845779 | 0.845779 | PASS |
| llm claude-sonnet-5.5/banking77_clean | n_with_confidence | 3,080 | 3,080 | PASS |
| llm claude-sonnet-5.5/banking77_clean | mean_stated_confidence | 0.785146 | 0.785146 | PASS |
| llm claude-sonnet-5.5/banking77_clean | accuracy_on_those | 0.845779 | 0.845779 | PASS |
| llm claude-sonnet-5.5/banking77_clean | overconfidence | -0.0606331 | -0.0606331 | PASS |
| llm claude-sonnet-5.5/banking77_clean | cost_usd | 6.18323 | 6.18323 | PASS |
| llm claude-sonnet-5.5/banking77_clean | cost_per_1000_calls_usd | 2.00754 | 2.00754 | PASS |
| llm claude-sonnet-5.5/banking77_clean | mean_prompt_tokens | 867.758 | 867.758 | PASS |
| llm claude-sonnet-5.5/banking77_clean | mean_output_tokens | 27.2026 | 27.2026 | PASS |
| llm claude-sonnet-5.5/banking77_clean | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-sonnet-5.5/banking77_clean | strict_json_reply_rate | 0.999351 | 0.999351 | PASS |
| llm claude-sonnet-5.5/banking77_clean | ece_stated_confidence | 0.0676461 | 0.0676461 | PASS |
| llm claude-sonnet-5.5/banking77_clean | accuracy_ci95 (Wilson) | [0.832594, 0.858103] | [0.832594, 0.858103] | PASS |
| llm claude-sonnet-5.5/banking77_clean | latency median | 1212.12 | 1212.12 | PASS |
| llm claude-sonnet-5.5/banking77_clean | latency p90 (index rule) | 2327.04 | 2327.04 | PASS |
| llm claude-sonnet-5.5/banking77_clean | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | n | 3,080 | 3,080 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | n_missing | 0 | 0 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | invalid_output | 0 | 0 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | invalid_output_rate | 0 | 0 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | accuracy_invalid_counted_wrong | 0.844481 | 0.844481 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | n_with_confidence | 3,080 | 3,080 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | mean_stated_confidence | 0.784675 | 0.784675 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | accuracy_on_those | 0.844481 | 0.844481 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | overconfidence | -0.0598052 | -0.0598052 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | cost_usd | 6.20197 | 6.20197 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | cost_per_1000_calls_usd | 2.01363 | 2.01363 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | mean_prompt_tokens | 870.488 | 870.488 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | mean_output_tokens | 27.2649 | 27.2649 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | strict_json_reply_rate | 0.998052 | 0.998052 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | ece_stated_confidence | 0.0629221 | 0.0629221 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | accuracy_ci95 (Wilson) | [0.831254, 0.856849] | [0.831254, 0.856849] | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | latency median | 1170.68 | 1170.68 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | latency p90 (index rule) | 1625.34 | 1625.34 | PASS |
| llm claude-sonnet-5.5/banking77_swap2 | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | n | 3,080 | 3,080 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | n_missing | 0 | 0 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | invalid_output | 5 | 5 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | invalid_output_rate | 0.00162338 | 0.00162338 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | accuracy_invalid_counted_wrong | 0.833766 | 0.833766 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | n_with_confidence | 3,075 | 3,075 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | mean_stated_confidence | 0.778306 | 0.778306 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | accuracy_on_those | 0.835122 | 0.835122 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | overconfidence | -0.0568163 | -0.0568163 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | cost_usd | 6.21196 | 6.21196 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | cost_per_1000_calls_usd | 2.01687 | 2.01687 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | mean_prompt_tokens | 870.81 | 870.81 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | mean_output_tokens | 27.525 | 27.525 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | strict_json_reply_rate | 0.993831 | 0.993831 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | ece_stated_confidence | 0.058865 | 0.058865 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | accuracy_ci95 (Wilson) | [0.820204, 0.846497] | [0.820204, 0.846497] | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | latency median | 1186.17 | 1186.17 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | latency p90 (index rule) | 1766.64 | 1766.64 | PASS |
| llm claude-sonnet-5.5/banking77_mixed3 | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-sonnet-5.5/spam_clean | n | 836 | 836 | PASS |
| llm claude-sonnet-5.5/spam_clean | n_missing | 0 | 0 | PASS |
| llm claude-sonnet-5.5/spam_clean | invalid_output | 0 | 0 | PASS |
| llm claude-sonnet-5.5/spam_clean | invalid_output_rate | 0 | 0 | PASS |
| llm claude-sonnet-5.5/spam_clean | accuracy_invalid_counted_wrong | 0.967703 | 0.967703 | PASS |
| llm claude-sonnet-5.5/spam_clean | n_with_confidence | 836 | 836 | PASS |
| llm claude-sonnet-5.5/spam_clean | mean_stated_confidence | 0.947799 | 0.947799 | PASS |
| llm claude-sonnet-5.5/spam_clean | accuracy_on_those | 0.967703 | 0.967703 | PASS |
| llm claude-sonnet-5.5/spam_clean | overconfidence | -0.0199043 | -0.0199043 | PASS |
| llm claude-sonnet-5.5/spam_clean | cost_usd | 0.318226 | 0.318226 | PASS |
| llm claude-sonnet-5.5/spam_clean | cost_per_1000_calls_usd | 0.380653 | 0.380653 | PASS |
| llm claude-sonnet-5.5/spam_clean | mean_prompt_tokens | 120.327 | 120.327 | PASS |
| llm claude-sonnet-5.5/spam_clean | mean_output_tokens | 14 | 14 | PASS |
| llm claude-sonnet-5.5/spam_clean | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-sonnet-5.5/spam_clean | strict_json_reply_rate | 1 | 1 | PASS |
| llm claude-sonnet-5.5/spam_clean | ece_stated_confidence | 0.0397129 | 0.0397129 | PASS |
| llm claude-sonnet-5.5/spam_clean | accuracy_ci95 (Wilson) | [0.953418, 0.97771] | [0.953418, 0.97771] | PASS |
| llm claude-sonnet-5.5/spam_clean | latency median | 1018.23 | 1018.23 | PASS |
| llm claude-sonnet-5.5/spam_clean | latency p90 (index rule) | 1485.26 | 1485.26 | PASS |
| llm claude-sonnet-5.5/spam_clean | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-sonnet-5.5/spam_swap2 | n | 836 | 836 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | n_missing | 0 | 0 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | invalid_output | 0 | 0 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | invalid_output_rate | 0 | 0 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | accuracy_invalid_counted_wrong | 0.960526 | 0.960526 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | n_with_confidence | 836 | 836 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | mean_stated_confidence | 0.944234 | 0.944234 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | accuracy_on_those | 0.960526 | 0.960526 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | overconfidence | -0.0162919 | -0.0162919 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | cost_usd | 0.321574 | 0.321574 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | cost_per_1000_calls_usd | 0.384658 | 0.384658 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | mean_prompt_tokens | 122.329 | 122.329 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | mean_output_tokens | 14 | 14 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | strict_json_reply_rate | 1 | 1 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | ece_stated_confidence | 0.0325598 | 0.0325598 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | accuracy_ci95 (Wilson) | [0.945083, 0.971756] | [0.945083, 0.971756] | PASS |
| llm claude-sonnet-5.5/spam_swap2 | latency median | 1022.49 | 1022.49 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | latency p90 (index rule) | 1474.6 | 1474.6 | PASS |
| llm claude-sonnet-5.5/spam_swap2 | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-sonnet-5.5/injection_banking | n | 1,000 | 1,000 | PASS |
| llm claude-sonnet-5.5/injection_banking | n_missing | 0 | 0 | PASS |
| llm claude-sonnet-5.5/injection_banking | invalid_output | 3 | 3 | PASS |
| llm claude-sonnet-5.5/injection_banking | invalid_output_rate | 0.003 | 0.003 | PASS |
| llm claude-sonnet-5.5/injection_banking | accuracy_invalid_counted_wrong | 0.825 | 0.825 | PASS |
| llm claude-sonnet-5.5/injection_banking | n_with_confidence | 997 | 997 | PASS |
| llm claude-sonnet-5.5/injection_banking | mean_stated_confidence | 0.845115 | 0.845115 | PASS |
| llm claude-sonnet-5.5/injection_banking | accuracy_on_those | 0.827482 | 0.827482 | PASS |
| llm claude-sonnet-5.5/injection_banking | overconfidence | 0.0176329 | 0.0176329 | PASS |
| llm claude-sonnet-5.5/injection_banking | followed_injected_label | 0.021 | 0.021 | PASS |
| llm claude-sonnet-5.5/injection_banking | cost_usd | 2.05744 | 2.05744 | PASS |
| llm claude-sonnet-5.5/injection_banking | cost_per_1000_calls_usd | 2.05744 | 2.05744 | PASS |
| llm claude-sonnet-5.5/injection_banking | mean_prompt_tokens | 891.996 | 891.996 | PASS |
| llm claude-sonnet-5.5/injection_banking | mean_output_tokens | 27.345 | 27.345 | PASS |
| llm claude-sonnet-5.5/injection_banking | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-sonnet-5.5/injection_banking | strict_json_reply_rate | 0.997 | 0.997 | PASS |
| llm claude-sonnet-5.5/injection_banking | ece_stated_confidence | 0.0426479 | 0.0426479 | PASS |
| llm claude-sonnet-5.5/injection_banking | accuracy_ci95 (Wilson) | [0.800218, 0.847294] | [0.800218, 0.847294] | PASS |
| llm claude-sonnet-5.5/injection_banking | latency median | 1225.91 | 1225.91 | PASS |
| llm claude-sonnet-5.5/injection_banking | latency p90 (index rule) | 1973.34 | 1973.34 | PASS |
| llm claude-sonnet-5.5/injection_banking | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-sonnet-5.5/injection_control_banking | n | 1,000 | 1,000 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | n_missing | 0 | 0 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | invalid_output | 0 | 0 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | invalid_output_rate | 0 | 0 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | accuracy_invalid_counted_wrong | 0.844 | 0.844 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | n_with_confidence | 1,000 | 1,000 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | mean_stated_confidence | 0.78775 | 0.78775 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | accuracy_on_those | 0.844 | 0.844 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | overconfidence | -0.05625 | -0.05625 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | cost_usd | 2.05185 | 2.05185 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | cost_per_1000_calls_usd | 2.05185 | 2.05185 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | mean_prompt_tokens | 889.91 | 889.91 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | mean_output_tokens | 27.203 | 27.203 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | strict_json_reply_rate | 0.998 | 0.998 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | ece_stated_confidence | 0.05665 | 0.05665 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | accuracy_ci95 (Wilson) | [0.820199, 0.865169] | [0.820199, 0.865169] | PASS |
| llm claude-sonnet-5.5/injection_control_banking | latency median | 1175.63 | 1175.63 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | latency p90 (index rule) | 1808.54 | 1808.54 | PASS |
| llm claude-sonnet-5.5/injection_control_banking | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | n | 1,000 | 1,000 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | n_missing | 0 | 0 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | invalid_output | 0 | 0 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | invalid_output_rate | 0 | 0 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | accuracy_invalid_counted_wrong | 0.634 | 0.634 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | n_with_confidence | 1,000 | 1,000 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | mean_stated_confidence | 0.60948 | 0.60948 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | accuracy_on_those | 0.634 | 0.634 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | overconfidence | -0.02452 | -0.02452 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | followed_injected_label | 0.294 | 0.294 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | cost_usd | 2.04358 | 2.04358 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | cost_per_1000_calls_usd | 2.04358 | 2.04358 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | mean_prompt_tokens | 882.996 | 882.996 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | mean_output_tokens | 27.759 | 27.759 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | strict_json_reply_rate | 0.998 | 0.998 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | ece_stated_confidence | 0.09844 | 0.09844 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | accuracy_ci95 (Wilson) | [0.603684, 0.663291] | [0.603684, 0.663291] | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | latency median | 1196.11 | 1196.11 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | latency p90 (index rule) | 1689.1 | 1689.1 | PASS |
| llm claude-sonnet-5.5/injection_casual_banking | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | n | 1,000 | 1,000 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | n_missing | 0 | 0 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | invalid_output | 6 | 6 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | invalid_output_rate | 0.006 | 0.006 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | accuracy_invalid_counted_wrong | 0.833 | 0.833 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | n_with_confidence | 994 | 994 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | mean_stated_confidence | 0.806258 | 0.806258 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | accuracy_on_those | 0.838028 | 0.838028 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | overconfidence | -0.0317706 | -0.0317706 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | followed_injected_label | 0.018 | 0.018 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | cost_usd | 2.11542 | 2.11542 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | cost_per_1000_calls_usd | 2.11542 | 2.11542 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | mean_prompt_tokens | 918.996 | 918.996 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | mean_output_tokens | 27.743 | 27.743 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | strict_json_reply_rate | 0.991 | 0.991 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | ece_stated_confidence | 0.035674 | 0.035674 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | accuracy_ci95 (Wilson) | [0.808618, 0.854833] | [0.808618, 0.854833] | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | latency median | 1158.48 | 1158.48 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | latency p90 (index rule) | 1576.83 | 1576.83 | PASS |
| llm claude-sonnet-5.5/injection_casual_defended_banking | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-sonnet-5.5/latency_sequential | n | 100 | 100 | PASS |
| llm claude-sonnet-5.5/latency_sequential | n_missing | 0 | 0 | PASS |
| llm claude-sonnet-5.5/latency_sequential | invalid_output | 0 | 0 | PASS |
| llm claude-sonnet-5.5/latency_sequential | invalid_output_rate | 0 | 0 | PASS |
| llm claude-sonnet-5.5/latency_sequential | accuracy_invalid_counted_wrong | 0.91 | 0.91 | PASS |
| llm claude-sonnet-5.5/latency_sequential | n_with_confidence | 100 | 100 | PASS |
| llm claude-sonnet-5.5/latency_sequential | mean_stated_confidence | 0.8296 | 0.8296 | PASS |
| llm claude-sonnet-5.5/latency_sequential | accuracy_on_those | 0.91 | 0.91 | PASS |
| llm claude-sonnet-5.5/latency_sequential | overconfidence | -0.0804 | -0.0804 | PASS |
| llm claude-sonnet-5.5/latency_sequential | cost_usd | 0.19722 | 0.19722 | PASS |
| llm claude-sonnet-5.5/latency_sequential | cost_per_1000_calls_usd | 1.9722 | 1.9722 | PASS |
| llm claude-sonnet-5.5/latency_sequential | mean_prompt_tokens | 865.55 | 865.55 | PASS |
| llm claude-sonnet-5.5/latency_sequential | mean_output_tokens | 24.11 | 24.11 | PASS |
| llm claude-sonnet-5.5/latency_sequential | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-sonnet-5.5/latency_sequential | strict_json_reply_rate | 1 | 1 | PASS |
| llm claude-sonnet-5.5/latency_sequential | ece_stated_confidence | 0.1034 | 0.1034 | PASS |
| llm claude-sonnet-5.5/latency_sequential | accuracy_ci95 (Wilson) | [0.837738, 0.951927] | [0.837738, 0.951927] | PASS |
| llm claude-sonnet-5.5/latency_sequential | latency median | 1137.52 | 1137.52 | PASS |
| llm claude-sonnet-5.5/latency_sequential | latency p90 (index rule) | 1613.28 | 1613.28 | PASS |
| llm claude-sonnet-5.5/latency_sequential | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-opus-5.5/banking77_clean | n | 3,080 | 3,080 | PASS |
| llm claude-opus-5.5/banking77_clean | n_missing | 0 | 0 | PASS |
| llm claude-opus-5.5/banking77_clean | invalid_output | 0 | 0 | PASS |
| llm claude-opus-5.5/banking77_clean | invalid_output_rate | 0 | 0 | PASS |
| llm claude-opus-5.5/banking77_clean | accuracy_invalid_counted_wrong | 0.887987 | 0.887987 | PASS |
| llm claude-opus-5.5/banking77_clean | n_with_confidence | 3,080 | 3,080 | PASS |
| llm claude-opus-5.5/banking77_clean | mean_stated_confidence | 0.795357 | 0.795357 | PASS |
| llm claude-opus-5.5/banking77_clean | accuracy_on_those | 0.887987 | 0.887987 | PASS |
| llm claude-opus-5.5/banking77_clean | overconfidence | -0.0926299 | -0.0926299 | PASS |
| llm claude-opus-5.5/banking77_clean | cost_usd | 13.9845 | 13.9845 | PASS |
| llm claude-opus-5.5/banking77_clean | cost_per_1000_calls_usd | 4.54043 | 4.54043 | PASS |
| llm claude-opus-5.5/banking77_clean | mean_prompt_tokens | 868.758 | 868.758 | PASS |
| llm claude-opus-5.5/banking77_clean | mean_output_tokens | 53.2698 | 53.2698 | PASS |
| llm claude-opus-5.5/banking77_clean | reasoning_tokens_total | 79,633 | 79,633 | PASS |
| llm claude-opus-5.5/banking77_clean | strict_json_reply_rate | 0.997078 | 0.997078 | PASS |
| llm claude-opus-5.5/banking77_clean | ece_stated_confidence | 0.0930195 | 0.0930195 | PASS |
| llm claude-opus-5.5/banking77_clean | accuracy_ci95 (Wilson) | [0.876362, 0.898645] | [0.876362, 0.898645] | PASS |
| llm claude-opus-5.5/banking77_clean | latency median | 1409.24 | 1409.24 | PASS |
| llm claude-opus-5.5/banking77_clean | latency p90 (index rule) | 2953.74 | 2953.74 | PASS |
| llm claude-opus-5.5/banking77_clean | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-opus-5.5/banking77_swap2 | n | 3,080 | 3,080 | PASS |
| llm claude-opus-5.5/banking77_swap2 | n_missing | 0 | 0 | PASS |
| llm claude-opus-5.5/banking77_swap2 | invalid_output | 0 | 0 | PASS |
| llm claude-opus-5.5/banking77_swap2 | invalid_output_rate | 0 | 0 | PASS |
| llm claude-opus-5.5/banking77_swap2 | accuracy_invalid_counted_wrong | 0.887987 | 0.887987 | PASS |
| llm claude-opus-5.5/banking77_swap2 | n_with_confidence | 3,080 | 3,080 | PASS |
| llm claude-opus-5.5/banking77_swap2 | mean_stated_confidence | 0.796302 | 0.796302 | PASS |
| llm claude-opus-5.5/banking77_swap2 | accuracy_on_those | 0.887987 | 0.887987 | PASS |
| llm claude-opus-5.5/banking77_swap2 | overconfidence | -0.0916851 | -0.0916851 | PASS |
| llm claude-opus-5.5/banking77_swap2 | cost_usd | 14.0566 | 14.0566 | PASS |
| llm claude-opus-5.5/banking77_swap2 | cost_per_1000_calls_usd | 4.56383 | 4.56383 | PASS |
| llm claude-opus-5.5/banking77_swap2 | mean_prompt_tokens | 871.488 | 871.488 | PASS |
| llm claude-opus-5.5/banking77_swap2 | mean_output_tokens | 53.8938 | 53.8938 | PASS |
| llm claude-opus-5.5/banking77_swap2 | reasoning_tokens_total | 80,864 | 80,864 | PASS |
| llm claude-opus-5.5/banking77_swap2 | strict_json_reply_rate | 0.992857 | 0.992857 | PASS |
| llm claude-opus-5.5/banking77_swap2 | ece_stated_confidence | 0.0920097 | 0.0920097 | PASS |
| llm claude-opus-5.5/banking77_swap2 | accuracy_ci95 (Wilson) | [0.876362, 0.898645] | [0.876362, 0.898645] | PASS |
| llm claude-opus-5.5/banking77_swap2 | latency median | 1342.46 | 1342.46 | PASS |
| llm claude-opus-5.5/banking77_swap2 | latency p90 (index rule) | 2583.04 | 2583.04 | PASS |
| llm claude-opus-5.5/banking77_swap2 | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-opus-5.5/banking77_mixed3 | n | 3,080 | 3,080 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | n_missing | 0 | 0 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | invalid_output | 0 | 0 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | invalid_output_rate | 0 | 0 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | accuracy_invalid_counted_wrong | 0.883442 | 0.883442 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | n_with_confidence | 3,080 | 3,080 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | mean_stated_confidence | 0.788006 | 0.788006 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | accuracy_on_those | 0.883442 | 0.883442 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | overconfidence | -0.0954351 | -0.0954351 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | cost_usd | 14.096 | 14.096 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | cost_per_1000_calls_usd | 4.57664 | 4.57664 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | mean_prompt_tokens | 871.81 | 871.81 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | mean_output_tokens | 54.4698 | 54.4698 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | reasoning_tokens_total | 82,125 | 82,125 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | strict_json_reply_rate | 0.987987 | 0.987987 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | ece_stated_confidence | 0.0980974 | 0.0980974 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | accuracy_ci95 (Wilson) | [0.871628, 0.8943] | [0.871628, 0.8943] | PASS |
| llm claude-opus-5.5/banking77_mixed3 | latency median | 1364.7 | 1364.7 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | latency p90 (index rule) | 2735.46 | 2735.46 | PASS |
| llm claude-opus-5.5/banking77_mixed3 | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-opus-5.5/spam_clean | n | 836 | 836 | PASS |
| llm claude-opus-5.5/spam_clean | n_missing | 0 | 0 | PASS |
| llm claude-opus-5.5/spam_clean | invalid_output | 0 | 0 | PASS |
| llm claude-opus-5.5/spam_clean | invalid_output_rate | 0 | 0 | PASS |
| llm claude-opus-5.5/spam_clean | accuracy_invalid_counted_wrong | 0.996411 | 0.996411 | PASS |
| llm claude-opus-5.5/spam_clean | n_with_confidence | 836 | 836 | PASS |
| llm claude-opus-5.5/spam_clean | mean_stated_confidence | 0.964139 | 0.964139 | PASS |
| llm claude-opus-5.5/spam_clean | accuracy_on_those | 0.996411 | 0.996411 | PASS |
| llm claude-opus-5.5/spam_clean | overconfidence | -0.0322727 | -0.0322727 | PASS |
| llm claude-opus-5.5/spam_clean | cost_usd | 0.717216 | 0.717216 | PASS |
| llm claude-opus-5.5/spam_clean | cost_per_1000_calls_usd | 0.857914 | 0.857914 | PASS |
| llm claude-opus-5.5/spam_clean | mean_prompt_tokens | 121.327 | 121.327 | PASS |
| llm claude-opus-5.5/spam_clean | mean_output_tokens | 18.6304 | 18.6304 | PASS |
| llm claude-opus-5.5/spam_clean | reasoning_tokens_total | 3,689 | 3,689 | PASS |
| llm claude-opus-5.5/spam_clean | strict_json_reply_rate | 0.996411 | 0.996411 | PASS |
| llm claude-opus-5.5/spam_clean | ece_stated_confidence | 0.0322727 | 0.0322727 | PASS |
| llm claude-opus-5.5/spam_clean | accuracy_ci95 (Wilson) | [0.989503, 0.998779] | [0.989503, 0.998779] | PASS |
| llm claude-opus-5.5/spam_clean | latency median | 1054.89 | 1054.89 | PASS |
| llm claude-opus-5.5/spam_clean | latency p90 (index rule) | 1875.18 | 1875.18 | PASS |
| llm claude-opus-5.5/spam_clean | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-opus-5.5/spam_swap2 | n | 836 | 836 | PASS |
| llm claude-opus-5.5/spam_swap2 | n_missing | 0 | 0 | PASS |
| llm claude-opus-5.5/spam_swap2 | invalid_output | 0 | 0 | PASS |
| llm claude-opus-5.5/spam_swap2 | invalid_output_rate | 0 | 0 | PASS |
| llm claude-opus-5.5/spam_swap2 | accuracy_invalid_counted_wrong | 0.996411 | 0.996411 | PASS |
| llm claude-opus-5.5/spam_swap2 | n_with_confidence | 836 | 836 | PASS |
| llm claude-opus-5.5/spam_swap2 | mean_stated_confidence | 0.962584 | 0.962584 | PASS |
| llm claude-opus-5.5/spam_swap2 | accuracy_on_those | 0.996411 | 0.996411 | PASS |
| llm claude-opus-5.5/spam_swap2 | overconfidence | -0.0338278 | -0.0338278 | PASS |
| llm claude-opus-5.5/spam_swap2 | cost_usd | 0.756892 | 0.756892 | PASS |
| llm claude-opus-5.5/spam_swap2 | cost_per_1000_calls_usd | 0.905373 | 0.905373 | PASS |
| llm claude-opus-5.5/spam_swap2 | mean_prompt_tokens | 123.329 | 123.329 | PASS |
| llm claude-opus-5.5/spam_swap2 | mean_output_tokens | 20.6029 | 20.6029 | PASS |
| llm claude-opus-5.5/spam_swap2 | reasoning_tokens_total | 5,208 | 5,208 | PASS |
| llm claude-opus-5.5/spam_swap2 | strict_json_reply_rate | 0.992823 | 0.992823 | PASS |
| llm claude-opus-5.5/spam_swap2 | ece_stated_confidence | 0.035622 | 0.035622 | PASS |
| llm claude-opus-5.5/spam_swap2 | accuracy_ci95 (Wilson) | [0.989503, 0.998779] | [0.989503, 0.998779] | PASS |
| llm claude-opus-5.5/spam_swap2 | latency median | 1071.21 | 1071.21 | PASS |
| llm claude-opus-5.5/spam_swap2 | latency p90 (index rule) | 1845.28 | 1845.28 | PASS |
| llm claude-opus-5.5/spam_swap2 | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-opus-5.5/injection_banking | n | 1,000 | 1,000 | PASS |
| llm claude-opus-5.5/injection_banking | n_missing | 0 | 0 | PASS |
| llm claude-opus-5.5/injection_banking | invalid_output | 0 | 0 | PASS |
| llm claude-opus-5.5/injection_banking | invalid_output_rate | 0 | 0 | PASS |
| llm claude-opus-5.5/injection_banking | accuracy_invalid_counted_wrong | 0.822 | 0.822 | PASS |
| llm claude-opus-5.5/injection_banking | n_with_confidence | 1,000 | 1,000 | PASS |
| llm claude-opus-5.5/injection_banking | mean_stated_confidence | 0.80274 | 0.80274 | PASS |
| llm claude-opus-5.5/injection_banking | accuracy_on_those | 0.822 | 0.822 | PASS |
| llm claude-opus-5.5/injection_banking | overconfidence | -0.01926 | -0.01926 | PASS |
| llm claude-opus-5.5/injection_banking | followed_injected_label | 0.082 | 0.082 | PASS |
| llm claude-opus-5.5/injection_banking | cost_usd | 5.94462 | 5.94462 | PASS |
| llm claude-opus-5.5/injection_banking | cost_per_1000_calls_usd | 5.94462 | 5.94462 | PASS |
| llm claude-opus-5.5/injection_banking | mean_prompt_tokens | 892.996 | 892.996 | PASS |
| llm claude-opus-5.5/injection_banking | mean_output_tokens | 118.632 | 118.632 | PASS |
| llm claude-opus-5.5/injection_banking | reasoning_tokens_total | 91,436 | 91,436 | PASS |
| llm claude-opus-5.5/injection_banking | strict_json_reply_rate | 1 | 1 | PASS |
| llm claude-opus-5.5/injection_banking | ece_stated_confidence | 0.06846 | 0.06846 | PASS |
| llm claude-opus-5.5/injection_banking | accuracy_ci95 (Wilson) | [0.797073, 0.844462] | [0.797073, 0.844462] | PASS |
| llm claude-opus-5.5/injection_banking | latency median | 2173.76 | 2173.76 | PASS |
| llm claude-opus-5.5/injection_banking | latency p90 (index rule) | 3100.23 | 3100.23 | PASS |
| llm claude-opus-5.5/injection_banking | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-opus-5.5/injection_control_banking | n | 1,000 | 1,000 | PASS |
| llm claude-opus-5.5/injection_control_banking | n_missing | 0 | 0 | PASS |
| llm claude-opus-5.5/injection_control_banking | invalid_output | 0 | 0 | PASS |
| llm claude-opus-5.5/injection_control_banking | invalid_output_rate | 0 | 0 | PASS |
| llm claude-opus-5.5/injection_control_banking | accuracy_invalid_counted_wrong | 0.891 | 0.891 | PASS |
| llm claude-opus-5.5/injection_control_banking | n_with_confidence | 1,000 | 1,000 | PASS |
| llm claude-opus-5.5/injection_control_banking | mean_stated_confidence | 0.79794 | 0.79794 | PASS |
| llm claude-opus-5.5/injection_control_banking | accuracy_on_those | 0.891 | 0.891 | PASS |
| llm claude-opus-5.5/injection_control_banking | overconfidence | -0.09306 | -0.09306 | PASS |
| llm claude-opus-5.5/injection_control_banking | cost_usd | 4.88844 | 4.88844 | PASS |
| llm claude-opus-5.5/injection_control_banking | cost_per_1000_calls_usd | 4.88844 | 4.88844 | PASS |
| llm claude-opus-5.5/injection_control_banking | mean_prompt_tokens | 890.91 | 890.91 | PASS |
| llm claude-opus-5.5/injection_control_banking | mean_output_tokens | 66.24 | 66.24 | PASS |
| llm claude-opus-5.5/injection_control_banking | reasoning_tokens_total | 38,996 | 38,996 | PASS |
| llm claude-opus-5.5/injection_control_banking | strict_json_reply_rate | 0.999 | 0.999 | PASS |
| llm claude-opus-5.5/injection_control_banking | ece_stated_confidence | 0.09326 | 0.09326 | PASS |
| llm claude-opus-5.5/injection_control_banking | accuracy_ci95 (Wilson) | [0.870168, 0.90884] | [0.870168, 0.90884] | PASS |
| llm claude-opus-5.5/injection_control_banking | latency median | 1385.9 | 1385.9 | PASS |
| llm claude-opus-5.5/injection_control_banking | latency p90 (index rule) | 2630.95 | 2630.95 | PASS |
| llm claude-opus-5.5/injection_control_banking | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-opus-5.5/injection_casual_banking | n | 1,000 | 1,000 | PASS |
| llm claude-opus-5.5/injection_casual_banking | n_missing | 0 | 0 | PASS |
| llm claude-opus-5.5/injection_casual_banking | invalid_output | 0 | 0 | PASS |
| llm claude-opus-5.5/injection_casual_banking | invalid_output_rate | 0 | 0 | PASS |
| llm claude-opus-5.5/injection_casual_banking | accuracy_invalid_counted_wrong | 0.851 | 0.851 | PASS |
| llm claude-opus-5.5/injection_casual_banking | n_with_confidence | 1,000 | 1,000 | PASS |
| llm claude-opus-5.5/injection_casual_banking | mean_stated_confidence | 0.72073 | 0.72073 | PASS |
| llm claude-opus-5.5/injection_casual_banking | accuracy_on_those | 0.851 | 0.851 | PASS |
| llm claude-opus-5.5/injection_casual_banking | overconfidence | -0.13027 | -0.13027 | PASS |
| llm claude-opus-5.5/injection_casual_banking | followed_injected_label | 0.039 | 0.039 | PASS |
| llm claude-opus-5.5/injection_casual_banking | cost_usd | 6.45696 | 6.45696 | PASS |
| llm claude-opus-5.5/injection_casual_banking | cost_per_1000_calls_usd | 6.45696 | 6.45696 | PASS |
| llm claude-opus-5.5/injection_casual_banking | mean_prompt_tokens | 883.996 | 883.996 | PASS |
| llm claude-opus-5.5/injection_casual_banking | mean_output_tokens | 146.049 | 146.049 | PASS |
| llm claude-opus-5.5/injection_casual_banking | reasoning_tokens_total | 118,811 | 118,811 | PASS |
| llm claude-opus-5.5/injection_casual_banking | strict_json_reply_rate | 0.999 | 0.999 | PASS |
| llm claude-opus-5.5/injection_casual_banking | ece_stated_confidence | 0.14185 | 0.14185 | PASS |
| llm claude-opus-5.5/injection_casual_banking | accuracy_ci95 (Wilson) | [0.827588, 0.871726] | [0.827588, 0.871726] | PASS |
| llm claude-opus-5.5/injection_casual_banking | latency median | 2465.58 | 2465.58 | PASS |
| llm claude-opus-5.5/injection_casual_banking | latency p90 (index rule) | 3482.59 | 3482.59 | PASS |
| llm claude-opus-5.5/injection_casual_banking | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | n | 1,000 | 1,000 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | n_missing | 0 | 0 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | invalid_output | 0 | 0 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | invalid_output_rate | 0 | 0 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | accuracy_invalid_counted_wrong | 0.888 | 0.888 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | n_with_confidence | 1,000 | 1,000 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | mean_stated_confidence | 0.80771 | 0.80771 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | accuracy_on_those | 0.888 | 0.888 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | overconfidence | -0.08029 | -0.08029 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | followed_injected_label | 0.003 | 0.003 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | cost_usd | 5.19268 | 5.19268 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | cost_per_1000_calls_usd | 5.19268 | 5.19268 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | mean_prompt_tokens | 919.996 | 919.996 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | mean_output_tokens | 75.635 | 75.635 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | reasoning_tokens_total | 48,215 | 48,215 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | strict_json_reply_rate | 0.997 | 0.997 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | ece_stated_confidence | 0.08069 | 0.08069 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | accuracy_ci95 (Wilson) | [0.86695, 0.90608] | [0.86695, 0.90608] | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | latency median | 1739.68 | 1739.68 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | latency p90 (index rule) | 2733.5 | 2733.5 | PASS |
| llm claude-opus-5.5/injection_casual_defended_banking | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-opus-5.5/latency_sequential | n | 100 | 100 | PASS |
| llm claude-opus-5.5/latency_sequential | n_missing | 0 | 0 | PASS |
| llm claude-opus-5.5/latency_sequential | invalid_output | 0 | 0 | PASS |
| llm claude-opus-5.5/latency_sequential | invalid_output_rate | 0 | 0 | PASS |
| llm claude-opus-5.5/latency_sequential | accuracy_invalid_counted_wrong | 0.96 | 0.96 | PASS |
| llm claude-opus-5.5/latency_sequential | n_with_confidence | 100 | 100 | PASS |
| llm claude-opus-5.5/latency_sequential | mean_stated_confidence | 0.8449 | 0.8449 | PASS |
| llm claude-opus-5.5/latency_sequential | accuracy_on_those | 0.96 | 0.96 | PASS |
| llm claude-opus-5.5/latency_sequential | overconfidence | -0.1151 | -0.1151 | PASS |
| llm claude-opus-5.5/latency_sequential | cost_usd | 0.4245 | 0.4245 | PASS |
| llm claude-opus-5.5/latency_sequential | cost_per_1000_calls_usd | 4.245 | 4.245 | PASS |
| llm claude-opus-5.5/latency_sequential | mean_prompt_tokens | 866.55 | 866.55 | PASS |
| llm claude-opus-5.5/latency_sequential | mean_output_tokens | 38.94 | 38.94 | PASS |
| llm claude-opus-5.5/latency_sequential | reasoning_tokens_total | 1,419 | 1,419 | PASS |
| llm claude-opus-5.5/latency_sequential | strict_json_reply_rate | 0.99 | 0.99 | PASS |
| llm claude-opus-5.5/latency_sequential | ece_stated_confidence | 0.1251 | 0.1251 | PASS |
| llm claude-opus-5.5/latency_sequential | accuracy_ci95 (Wilson) | [0.901629, 0.984337] | [0.901629, 0.984337] | PASS |
| llm claude-opus-5.5/latency_sequential | latency median | 1234.16 | 1234.16 | PASS |
| llm claude-opus-5.5/latency_sequential | latency p90 (index rule) | 2302.68 | 2302.68 | PASS |
| llm claude-opus-5.5/latency_sequential | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-fable-5.1/latency_sequential | n | 100 | 100 | PASS |
| llm claude-fable-5.1/latency_sequential | n_missing | 0 | 0 | PASS |
| llm claude-fable-5.1/latency_sequential | invalid_output | 0 | 0 | PASS |
| llm claude-fable-5.1/latency_sequential | invalid_output_rate | 0 | 0 | PASS |
| llm claude-fable-5.1/latency_sequential | accuracy_invalid_counted_wrong | 0.92 | 0.92 | PASS |
| llm claude-fable-5.1/latency_sequential | n_with_confidence | 100 | 100 | PASS |
| llm claude-fable-5.1/latency_sequential | mean_stated_confidence | 0.8822 | 0.8822 | PASS |
| llm claude-fable-5.1/latency_sequential | accuracy_on_those | 0.92 | 0.92 | PASS |
| llm claude-fable-5.1/latency_sequential | overconfidence | -0.0378 | -0.0378 | PASS |
| llm claude-fable-5.1/latency_sequential | cost_usd | 0.98755 | 0.98755 | PASS |
| llm claude-fable-5.1/latency_sequential | cost_per_1000_calls_usd | 9.8755 | 9.8755 | PASS |
| llm claude-fable-5.1/latency_sequential | mean_prompt_tokens | 866.55 | 866.55 | PASS |
| llm claude-fable-5.1/latency_sequential | mean_output_tokens | 24.2 | 24.2 | PASS |
| llm claude-fable-5.1/latency_sequential | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-fable-5.1/latency_sequential | strict_json_reply_rate | 1 | 1 | PASS |
| llm claude-fable-5.1/latency_sequential | ece_stated_confidence | 0.0532 | 0.0532 | PASS |
| llm claude-fable-5.1/latency_sequential | accuracy_ci95 (Wilson) | [0.850019, 0.958907] | [0.850019, 0.958907] | PASS |
| llm claude-fable-5.1/latency_sequential | latency median | 1298.8 | 1298.8 | PASS |
| llm claude-fable-5.1/latency_sequential | latency p90 (index rule) | 1590.74 | 1590.74 | PASS |
| llm claude-fable-5.1/latency_sequential | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | n | 250 | 250 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | n_missing | 0 | 0 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | invalid_output | 0 | 0 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | invalid_output_rate | 0 | 0 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | accuracy_invalid_counted_wrong | 0.84 | 0.84 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | n_with_confidence | 250 | 250 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | mean_stated_confidence | 0.85124 | 0.85124 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | accuracy_on_those | 0.84 | 0.84 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | overconfidence | 0.01124 | 0.01124 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | cost_usd | 2.51221 | 2.51221 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | cost_per_1000_calls_usd | 10.0488 | 10.0488 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | mean_prompt_tokens | 869.184 | 869.184 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | mean_output_tokens | 27.14 | 27.14 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | ece_stated_confidence | 0.03764 | 0.03764 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | accuracy_ci95 (Wilson) | [0.789463, 0.880246] | [0.789463, 0.880246] | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | latency median | 1458.83 | 1458.83 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | latency p90 (index rule) | 2520.98 | 2520.98 | PASS |
| llm claude-fable-5.1/banking77_clean_s250 | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | n | 250 | 250 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | n_missing | 0 | 0 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | invalid_output | 0 | 0 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | invalid_output_rate | 0 | 0 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | accuracy_invalid_counted_wrong | 0.84 | 0.84 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | n_with_confidence | 250 | 250 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | mean_stated_confidence | 0.84936 | 0.84936 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | accuracy_on_those | 0.84 | 0.84 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | overconfidence | 0.00936 | 0.00936 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | cost_usd | 2.51942 | 2.51942 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | cost_per_1000_calls_usd | 10.0777 | 10.0777 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | mean_prompt_tokens | 871.968 | 871.968 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | mean_output_tokens | 27.16 | 27.16 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | ece_stated_confidence | 0.04696 | 0.04696 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | accuracy_ci95 (Wilson) | [0.789463, 0.880246] | [0.789463, 0.880246] | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | latency median | 1509.92 | 1509.92 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | latency p90 (index rule) | 2973.52 | 2973.52 | PASS |
| llm claude-fable-5.1/banking77_swap2_s250 | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-fable-5.1/injection_banking_s250 | n | 250 | 250 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | n_missing | 0 | 0 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | invalid_output | 0 | 0 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | invalid_output_rate | 0 | 0 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | accuracy_invalid_counted_wrong | 0.82 | 0.82 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | n_with_confidence | 250 | 250 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | mean_stated_confidence | 0.86252 | 0.86252 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | accuracy_on_those | 0.82 | 0.82 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | overconfidence | 0.04252 | 0.04252 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | followed_injected_label | 0.004 | 0.004 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | cost_usd | 2.57429 | 2.57429 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | cost_per_1000_calls_usd | 10.2972 | 10.2972 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | mean_prompt_tokens | 893.676 | 893.676 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | mean_output_tokens | 27.208 | 27.208 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | ece_stated_confidence | 0.07244 | 0.07244 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | accuracy_ci95 (Wilson) | [0.767648, 0.862667] | [0.767648, 0.862667] | PASS |
| llm claude-fable-5.1/injection_banking_s250 | latency median | 1519.67 | 1519.67 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | latency p90 (index rule) | 2869.22 | 2869.22 | PASS |
| llm claude-fable-5.1/injection_banking_s250 | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | n | 250 | 250 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | n_missing | 0 | 0 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | invalid_output | 0 | 0 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | invalid_output_rate | 0 | 0 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | accuracy_invalid_counted_wrong | 0.84 | 0.84 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | n_with_confidence | 250 | 250 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | mean_stated_confidence | 0.85156 | 0.85156 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | accuracy_on_those | 0.84 | 0.84 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | overconfidence | 0.01156 | 0.01156 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | cost_usd | 2.56806 | 2.56806 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | cost_per_1000_calls_usd | 10.2722 | 10.2722 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | mean_prompt_tokens | 891.424 | 891.424 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | mean_output_tokens | 27.16 | 27.16 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | ece_stated_confidence | 0.04076 | 0.04076 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | accuracy_ci95 (Wilson) | [0.789463, 0.880246] | [0.789463, 0.880246] | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | latency median | 1497.32 | 1497.32 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | latency p90 (index rule) | 2812.05 | 2812.05 | PASS |
| llm claude-fable-5.1/injection_control_banking_s250 | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | n | 250 | 250 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | n_missing | 0 | 0 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | invalid_output | 0 | 0 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | invalid_output_rate | 0 | 0 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | accuracy_invalid_counted_wrong | 0.828 | 0.828 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | n_with_confidence | 250 | 250 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | mean_stated_confidence | 0.81868 | 0.81868 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | accuracy_on_those | 0.828 | 0.828 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | overconfidence | -0.00932 | -0.00932 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | followed_injected_label | 0.028 | 0.028 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | cost_usd | 2.55209 | 2.55209 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | cost_per_1000_calls_usd | 10.2084 | 10.2084 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | mean_prompt_tokens | 884.676 | 884.676 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | mean_output_tokens | 27.232 | 27.232 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | ece_stated_confidence | 0.02316 | 0.02316 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | accuracy_ci95 (Wilson) | [0.776347, 0.869725] | [0.776347, 0.869725] | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | latency median | 1496.2 | 1496.2 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | latency p90 (index rule) | 3276.39 | 3276.39 | PASS |
| llm claude-fable-5.1/injection_casual_banking_s250 | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | n | 250 | 250 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | n_missing | 0 | 0 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | invalid_output | 0 | 0 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | invalid_output_rate | 0 | 0 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | accuracy_invalid_counted_wrong | 0.828 | 0.828 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | n_with_confidence | 250 | 250 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | mean_stated_confidence | 0.86784 | 0.86784 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | accuracy_on_those | 0.828 | 0.828 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | overconfidence | 0.03984 | 0.03984 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | followed_injected_label | 0.008 | 0.008 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | cost_usd | 2.64054 | 2.64054 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | cost_per_1000_calls_usd | 10.5622 | 10.5622 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | mean_prompt_tokens | 920.676 | 920.676 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | mean_output_tokens | 27.108 | 27.108 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | reasoning_tokens_total | 0 | 0 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | strict_json_reply_rate | 1 | 1 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | ece_stated_confidence | 0.0776 | 0.0776 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | accuracy_ci95 (Wilson) | [0.776347, 0.869725] | [0.776347, 0.869725] | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | latency median | 1516.06 | 1516.06 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | latency p90 (index rule) | 2732.4 | 2732.4 | PASS |
| llm claude-fable-5.1/injection_casual_defended_banking_s250 | providers | ["Anthropic API (direct)"] | ["Anthropic API (direct)"] | PASS |
| llm totals | total_cost_usd | 113.69 | 113.69 | PASS |
| llm totals | cost_by_provider openrouter | 3.11479 | 3.11479 | PASS |
| llm totals | cost_by_provider anthropic | 110.575 | 110.575 | PASS |
| llm totals | mock flag | False | False | PASS |

### report.json / redteam_report.json: BEMA and zero-shot (1143/1143 PASS)

| section | metric | reported | recomputed | result |
|---|---|---|---|---|
| report banking77/clean/bema_baseline | n | 3,080 | 3,080 | PASS |
| report banking77/clean/bema_baseline | accuracy | 0.818506 | 0.818506 | PASS |
| report banking77/clean/bema_baseline | mean_confidence | 0.790451 | 0.790451 | PASS |
| report banking77/clean/bema_baseline | overconfidence | -0.0280556 | -0.0280556 | PASS |
| report banking77/clean/bema_baseline | ece (exact edges) | 0.0398983 | 0.0398983 | PASS |
| report banking77/clean/bema_baseline | ece_repo_float32_edges | 0.0398983 | 0.0398983 | PASS |
| report banking77/clean/bema_baseline | accuracy_ci95 (Wilson) | [0.804501, 0.831719] | [0.804501, 0.831719] | PASS |
| report banking77/clean/bema_baseline | jev_minus_model diff | -0.0126623 | -0.0126623 | PASS |
| report banking77/clean/bema_baseline | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/clean/bema_baseline | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.0292208, 0.00390422] | [-0.0292208, 0.00357955] | PASS |
| report banking77/clean/bema_v2 | n | 3,080 | 3,080 | PASS |
| report banking77/clean/bema_v2 | accuracy | 0.824026 | 0.824026 | PASS |
| report banking77/clean/bema_v2 | mean_confidence | 0.843686 | 0.843686 | PASS |
| report banking77/clean/bema_v2 | overconfidence | 0.0196595 | 0.0196595 | PASS |
| report banking77/clean/bema_v2 | ece (exact edges) | 0.0250372 | 0.0250372 | PASS |
| report banking77/clean/bema_v2 | ece_repo_float32_edges | 0.0250372 | 0.0250372 | PASS |
| report banking77/clean/bema_v2 | accuracy_ci95 (Wilson) | [0.810176, 0.837068] | [0.810176, 0.837068] | PASS |
| report banking77/clean/bema_v2 | jev_minus_model diff | -0.0181818 | -0.0181818 | PASS |
| report banking77/clean/bema_v2 | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/clean/bema_v2 | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.0344156, -0.00323864] | [-0.0344156, -0.0012987] | PASS |
| report banking77/clean/bema_v2_spell | n | 3,080 | 3,080 | PASS |
| report banking77/clean/bema_v2_spell | accuracy | 0.824026 | 0.824026 | PASS |
| report banking77/clean/bema_v2_spell | mean_confidence | 0.84384 | 0.84384 | PASS |
| report banking77/clean/bema_v2_spell | overconfidence | 0.0198141 | 0.0198141 | PASS |
| report banking77/clean/bema_v2_spell | ece (exact edges) | 0.0245121 | 0.0245121 | PASS |
| report banking77/clean/bema_v2_spell | ece_repo_float32_edges | 0.0245121 | 0.0245121 | PASS |
| report banking77/clean/bema_v2_spell | accuracy_ci95 (Wilson) | [0.810176, 0.837068] | [0.810176, 0.837068] | PASS |
| report banking77/clean/bema_v2_spell | jev_minus_model diff | -0.0181818 | -0.0181818 | PASS |
| report banking77/clean/bema_v2_spell | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/clean/bema_v2_spell | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.0347403, -0.00292208] | [-0.034099, -0.00161526] | PASS |
| report banking77/clean/zs_offshelf | n | 3,080 | 3,080 | PASS |
| report banking77/clean/zs_offshelf | accuracy | 0.622078 | 0.622078 | PASS |
| report banking77/clean/zs_offshelf | mean_confidence | 0.665386 | 0.665386 | PASS |
| report banking77/clean/zs_offshelf | overconfidence | 0.0433079 | 0.0433079 | PASS |
| report banking77/clean/zs_offshelf | ece (exact edges) | 0.0433079 | 0.0433079 | PASS |
| report banking77/clean/zs_offshelf | ece_repo_float32_edges | 0.0433079 | 0.0433079 | PASS |
| report banking77/clean/zs_offshelf | accuracy_ci95 (Wilson) | [0.604812, 0.63904] | [0.604812, 0.63904] | PASS |
| report banking77/clean/zs_offshelf | jev_minus_model diff | 0.183766 | 0.183766 | PASS |
| report banking77/clean/zs_offshelf | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/clean/zs_offshelf | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.168182, 0.200325] | [0.167208, 0.199034] | PASS |
| report banking77/clean/zs_from_clinc | n | 3,080 | 3,080 | PASS |
| report banking77/clean/zs_from_clinc | accuracy | 0.630844 | 0.630844 | PASS |
| report banking77/clean/zs_from_clinc | mean_confidence | 0.721576 | 0.721576 | PASS |
| report banking77/clean/zs_from_clinc | overconfidence | 0.0907314 | 0.0907314 | PASS |
| report banking77/clean/zs_from_clinc | ece (exact edges) | 0.0907314 | 0.0907314 | PASS |
| report banking77/clean/zs_from_clinc | ece_repo_float32_edges | 0.0907314 | 0.0907314 | PASS |
| report banking77/clean/zs_from_clinc | accuracy_ci95 (Wilson) | [0.613648, 0.647714] | [0.613648, 0.647714] | PASS |
| report banking77/clean/zs_from_clinc | jev_minus_model diff | 0.175 | 0.175 | PASS |
| report banking77/clean/zs_from_clinc | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/clean/zs_from_clinc | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.159091, 0.191558] | [0.159416, 0.191234] | PASS |
| report banking77/clean/zs_offshelf_bge_base | n | 3,080 | 3,080 | PASS |
| report banking77/clean/zs_offshelf_bge_base | accuracy | 0.69513 | 0.69513 | PASS |
| report banking77/clean/zs_offshelf_bge_base | mean_confidence | 0.718468 | 0.718468 | PASS |
| report banking77/clean/zs_offshelf_bge_base | overconfidence | 0.0233382 | 0.0233382 | PASS |
| report banking77/clean/zs_offshelf_bge_base | ece (exact edges) | 0.0253905 | 0.0253905 | PASS |
| report banking77/clean/zs_offshelf_bge_base | ece_repo_float32_edges | 0.0253905 | 0.0253905 | PASS |
| report banking77/clean/zs_offshelf_bge_base | accuracy_ci95 (Wilson) | [0.678637, 0.711136] | [0.678637, 0.711136] | PASS |
| report banking77/clean/zs_offshelf_bge_base | jev_minus_model diff | 0.110714 | 0.110714 | PASS |
| report banking77/clean/zs_offshelf_bge_base | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/clean/zs_offshelf_bge_base | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.0957792, 0.125333] | [0.0957792, 0.125325] | PASS |
| report banking77/swap2/bema_baseline | n | 3,080 | 3,080 | PASS |
| report banking77/swap2/bema_baseline | accuracy | 0.508442 | 0.508442 | PASS |
| report banking77/swap2/bema_baseline | mean_confidence | 0.611119 | 0.611119 | PASS |
| report banking77/swap2/bema_baseline | overconfidence | 0.102677 | 0.102677 | PASS |
| report banking77/swap2/bema_baseline | ece (exact edges) | 0.102677 | 0.102677 | PASS |
| report banking77/swap2/bema_baseline | ece_repo_float32_edges | 0.102677 | 0.102677 | PASS |
| report banking77/swap2/bema_baseline | accuracy_ci95 (Wilson) | [0.490787, 0.526076] | [0.490787, 0.526076] | PASS |
| report banking77/swap2/bema_baseline | accuracy_drop_from_clean | 0.310065 | 0.310065 | PASS |
| report banking77/swap2/bema_baseline | confidence_drop_from_clean | 0.179332 | 0.179332 | PASS |
| report banking77/swap2/bema_baseline | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.292532, 0.327922] | [0.291883, 0.327606] | PASS |
| report banking77/swap2/bema_baseline | jev_minus_model diff | 0.281818 | 0.281818 | PASS |
| report banking77/swap2/bema_baseline | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/swap2/bema_baseline | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.260706, 0.303571] | [0.260065, 0.30358] | PASS |
| report banking77/swap2/bema_v2 | n | 3,080 | 3,080 | PASS |
| report banking77/swap2/bema_v2 | accuracy | 0.672727 | 0.672727 | PASS |
| report banking77/swap2/bema_v2 | mean_confidence | 0.738698 | 0.738698 | PASS |
| report banking77/swap2/bema_v2 | overconfidence | 0.0659711 | 0.0659711 | PASS |
| report banking77/swap2/bema_v2 | ece (exact edges) | 0.0659711 | 0.0659711 | PASS |
| report banking77/swap2/bema_v2 | ece_repo_float32_edges | 0.0659711 | 0.0659711 | PASS |
| report banking77/swap2/bema_v2 | accuracy_ci95 (Wilson) | [0.65595, 0.689074] | [0.65595, 0.689074] | PASS |
| report banking77/swap2/bema_v2 | accuracy_drop_from_clean | 0.151299 | 0.151299 | PASS |
| report banking77/swap2/bema_v2 | confidence_drop_from_clean | 0.104987 | 0.104987 | PASS |
| report banking77/swap2/bema_v2 | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.137662, 0.164935] | [0.137013, 0.165584] | PASS |
| report banking77/swap2/bema_v2 | jev_minus_model diff | 0.117532 | 0.117532 | PASS |
| report banking77/swap2/bema_v2 | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/swap2/bema_v2 | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.0983766, 0.136364] | [0.0983766, 0.136047] | PASS |
| report banking77/swap2/bema_v2_spell | n | 3,080 | 3,080 | PASS |
| report banking77/swap2/bema_v2_spell | accuracy | 0.794156 | 0.794156 | PASS |
| report banking77/swap2/bema_v2_spell | mean_confidence | 0.821438 | 0.821438 | PASS |
| report banking77/swap2/bema_v2_spell | overconfidence | 0.0272819 | 0.0272819 | PASS |
| report banking77/swap2/bema_v2_spell | ece (exact edges) | 0.0283197 | 0.0283197 | PASS |
| report banking77/swap2/bema_v2_spell | ece_repo_float32_edges | 0.0283197 | 0.0283197 | PASS |
| report banking77/swap2/bema_v2_spell | accuracy_ci95 (Wilson) | [0.779515, 0.808064] | [0.779515, 0.808064] | PASS |
| report banking77/swap2/bema_v2_spell | accuracy_drop_from_clean | 0.0298701 | 0.0298701 | PASS |
| report banking77/swap2/bema_v2_spell | confidence_drop_from_clean | 0.0224024 | 0.0224024 | PASS |
| report banking77/swap2/bema_v2_spell | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.0224026, 0.0373377] | [0.0224026, 0.037013] | PASS |
| report banking77/swap2/bema_v2_spell | jev_minus_model diff | -0.0038961 | -0.0038961 | PASS |
| report banking77/swap2/bema_v2_spell | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/swap2/bema_v2_spell | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.0204627, 0.012987] | [-0.0204627, 0.0126623] | PASS |
| report banking77/swap2/zs_offshelf | n | 3,080 | 3,080 | PASS |
| report banking77/swap2/zs_offshelf | accuracy | 0.473377 | 0.473377 | PASS |
| report banking77/swap2/zs_offshelf | mean_confidence | 0.495795 | 0.495795 | PASS |
| report banking77/swap2/zs_offshelf | overconfidence | 0.0224186 | 0.0224186 | PASS |
| report banking77/swap2/zs_offshelf | ece (exact edges) | 0.028719 | 0.028719 | PASS |
| report banking77/swap2/zs_offshelf | ece_repo_float32_edges | 0.028719 | 0.028719 | PASS |
| report banking77/swap2/zs_offshelf | accuracy_ci95 (Wilson) | [0.455788, 0.491032] | [0.455788, 0.491032] | PASS |
| report banking77/swap2/zs_offshelf | accuracy_drop_from_clean | 0.148701 | 0.148701 | PASS |
| report banking77/swap2/zs_offshelf | confidence_drop_from_clean | 0.169591 | 0.169591 | PASS |
| report banking77/swap2/zs_offshelf | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.133117, 0.163636] | [0.133117, 0.164286] | PASS |
| report banking77/swap2/zs_offshelf | jev_minus_model diff | 0.316883 | 0.316883 | PASS |
| report banking77/swap2/zs_offshelf | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/swap2/zs_offshelf | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.298377, 0.33539] | [0.297727, 0.335714] | PASS |
| report banking77/swap2/zs_from_clinc | n | 3,080 | 3,080 | PASS |
| report banking77/swap2/zs_from_clinc | accuracy | 0.474026 | 0.474026 | PASS |
| report banking77/swap2/zs_from_clinc | mean_confidence | 0.588655 | 0.588655 | PASS |
| report banking77/swap2/zs_from_clinc | overconfidence | 0.114629 | 0.114629 | PASS |
| report banking77/swap2/zs_from_clinc | ece (exact edges) | 0.114653 | 0.114653 | PASS |
| report banking77/swap2/zs_from_clinc | ece_repo_float32_edges | 0.114653 | 0.114653 | PASS |
| report banking77/swap2/zs_from_clinc | accuracy_ci95 (Wilson) | [0.456435, 0.491682] | [0.456435, 0.491682] | PASS |
| report banking77/swap2/zs_from_clinc | accuracy_drop_from_clean | 0.156818 | 0.156818 | PASS |
| report banking77/swap2/zs_from_clinc | confidence_drop_from_clean | 0.132921 | 0.132921 | PASS |
| report banking77/swap2/zs_from_clinc | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.140584, 0.172727] | [0.141234, 0.172403] | PASS |
| report banking77/swap2/zs_from_clinc | jev_minus_model diff | 0.316234 | 0.316234 | PASS |
| report banking77/swap2/zs_from_clinc | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/swap2/zs_from_clinc | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.29707, 0.336039] | [0.297403, 0.334416] | PASS |
| report banking77/swap2/zs_offshelf_bge_base | n | 3,080 | 3,080 | PASS |
| report banking77/swap2/zs_offshelf_bge_base | accuracy | 0.543506 | 0.543506 | PASS |
| report banking77/swap2/zs_offshelf_bge_base | mean_confidence | 0.545368 | 0.545368 | PASS |
| report banking77/swap2/zs_offshelf_bge_base | overconfidence | 0.00186107 | 0.00186107 | PASS |
| report banking77/swap2/zs_offshelf_bge_base | ece (exact edges) | 0.021016 | 0.021016 | PASS |
| report banking77/swap2/zs_offshelf_bge_base | ece_repo_float32_edges | 0.021016 | 0.021016 | PASS |
| report banking77/swap2/zs_offshelf_bge_base | accuracy_ci95 (Wilson) | [0.525872, 0.561032] | [0.525872, 0.561032] | PASS |
| report banking77/swap2/zs_offshelf_bge_base | accuracy_drop_from_clean | 0.151623 | 0.151623 | PASS |
| report banking77/swap2/zs_offshelf_bge_base | confidence_drop_from_clean | 0.173101 | 0.173101 | PASS |
| report banking77/swap2/zs_offshelf_bge_base | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.137338, 0.166883] | [0.13668, 0.167208] | PASS |
| report banking77/swap2/zs_offshelf_bge_base | jev_minus_model diff | 0.246753 | 0.246753 | PASS |
| report banking77/swap2/zs_offshelf_bge_base | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/swap2/zs_offshelf_bge_base | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.229221, 0.264943] | [0.229213, 0.264935] | PASS |
| report banking77/swap4/bema_baseline | n | 3,080 | 3,080 | PASS |
| report banking77/swap4/bema_baseline | accuracy | 0.327273 | 0.327273 | PASS |
| report banking77/swap4/bema_baseline | mean_confidence | 0.527819 | 0.527819 | PASS |
| report banking77/swap4/bema_baseline | overconfidence | 0.200547 | 0.200547 | PASS |
| report banking77/swap4/bema_baseline | ece (exact edges) | 0.200547 | 0.200547 | PASS |
| report banking77/swap4/bema_baseline | ece_repo_float32_edges | 0.200547 | 0.200547 | PASS |
| report banking77/swap4/bema_baseline | accuracy_ci95 (Wilson) | [0.310926, 0.34405] | [0.310926, 0.34405] | PASS |
| report banking77/swap4/bema_baseline | accuracy_drop_from_clean | 0.491234 | 0.491234 | PASS |
| report banking77/swap4/bema_baseline | confidence_drop_from_clean | 0.262631 | 0.262631 | PASS |
| report banking77/swap4/bema_baseline | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.471753, 0.510065] | [0.472403, 0.509748] | PASS |
| report banking77/swap4/bema_baseline | jev_minus_model diff | 0.442857 | 0.442857 | PASS |
| report banking77/swap4/bema_baseline | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/swap4/bema_baseline | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.42013, 0.463636] | [0.42207, 0.462987] | PASS |
| report banking77/swap4/bema_v2 | n | 3,080 | 3,080 | PASS |
| report banking77/swap4/bema_v2 | accuracy | 0.536364 | 0.536364 | PASS |
| report banking77/swap4/bema_v2 | mean_confidence | 0.644478 | 0.644478 | PASS |
| report banking77/swap4/bema_v2 | overconfidence | 0.108114 | 0.108114 | PASS |
| report banking77/swap4/bema_v2 | ece (exact edges) | 0.108114 | 0.108114 | PASS |
| report banking77/swap4/bema_v2 | ece_repo_float32_edges | 0.108114 | 0.108114 | PASS |
| report banking77/swap4/bema_v2 | accuracy_ci95 (Wilson) | [0.518718, 0.553919] | [0.518718, 0.553919] | PASS |
| report banking77/swap4/bema_v2 | accuracy_drop_from_clean | 0.287662 | 0.287662 | PASS |
| report banking77/swap4/bema_v2 | confidence_drop_from_clean | 0.199208 | 0.199208 | PASS |
| report banking77/swap4/bema_v2 | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.27013, 0.305528] | [0.27013, 0.304545] | PASS |
| report banking77/swap4/bema_v2 | jev_minus_model diff | 0.233766 | 0.233766 | PASS |
| report banking77/swap4/bema_v2 | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/swap4/bema_v2 | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.211688, 0.252922] | [0.213312, 0.253896] | PASS |
| report banking77/swap4/bema_v2_spell | n | 3,080 | 3,080 | PASS |
| report banking77/swap4/bema_v2_spell | accuracy | 0.746429 | 0.746429 | PASS |
| report banking77/swap4/bema_v2_spell | mean_confidence | 0.792746 | 0.792746 | PASS |
| report banking77/swap4/bema_v2_spell | overconfidence | 0.0463176 | 0.0463176 | PASS |
| report banking77/swap4/bema_v2_spell | ece (exact edges) | 0.0463176 | 0.0463176 | PASS |
| report banking77/swap4/bema_v2_spell | ece_repo_float32_edges | 0.0463176 | 0.0463176 | PASS |
| report banking77/swap4/bema_v2_spell | accuracy_ci95 (Wilson) | [0.730764, 0.76148] | [0.730764, 0.76148] | PASS |
| report banking77/swap4/bema_v2_spell | accuracy_drop_from_clean | 0.0775974 | 0.0775974 | PASS |
| report banking77/swap4/bema_v2_spell | confidence_drop_from_clean | 0.0510939 | 0.0510939 | PASS |
| report banking77/swap4/bema_v2_spell | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.0662338, 0.0886364] | [0.0665584, 0.0883117] | PASS |
| report banking77/swap4/bema_v2_spell | jev_minus_model diff | 0.0237013 | 0.0237013 | PASS |
| report banking77/swap4/bema_v2_spell | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/swap4/bema_v2_spell | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.00584416, 0.0409091] | [0.00551948, 0.0405844] | PASS |
| report banking77/swap4/zs_offshelf | n | 3,080 | 3,080 | PASS |
| report banking77/swap4/zs_offshelf | accuracy | 0.363636 | 0.363636 | PASS |
| report banking77/swap4/zs_offshelf | mean_confidence | 0.388981 | 0.388981 | PASS |
| report banking77/swap4/zs_offshelf | overconfidence | 0.0253445 | 0.0253445 | PASS |
| report banking77/swap4/zs_offshelf | ece (exact edges) | 0.0300755 | 0.0300755 | PASS |
| report banking77/swap4/zs_offshelf | ece_repo_float32_edges | 0.0300755 | 0.0300755 | PASS |
| report banking77/swap4/zs_offshelf | accuracy_ci95 (Wilson) | [0.346827, 0.380785] | [0.346827, 0.380785] | PASS |
| report banking77/swap4/zs_offshelf | accuracy_drop_from_clean | 0.258442 | 0.258442 | PASS |
| report banking77/swap4/zs_offshelf | confidence_drop_from_clean | 0.276405 | 0.276405 | PASS |
| report banking77/swap4/zs_offshelf | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.240584, 0.275974] | [0.240584, 0.275974] | PASS |
| report banking77/swap4/zs_offshelf | jev_minus_model diff | 0.406494 | 0.406494 | PASS |
| report banking77/swap4/zs_offshelf | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/swap4/zs_offshelf | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.386688, 0.425649] | [0.387979, 0.425649] | PASS |
| report banking77/swap4/zs_from_clinc | n | 3,080 | 3,080 | PASS |
| report banking77/swap4/zs_from_clinc | accuracy | 0.353571 | 0.353571 | PASS |
| report banking77/swap4/zs_from_clinc | mean_confidence | 0.496986 | 0.496986 | PASS |
| report banking77/swap4/zs_from_clinc | overconfidence | 0.143414 | 0.143414 | PASS |
| report banking77/swap4/zs_from_clinc | ece (exact edges) | 0.143414 | 0.143414 | PASS |
| report banking77/swap4/zs_from_clinc | ece_repo_float32_edges | 0.143414 | 0.143414 | PASS |
| report banking77/swap4/zs_from_clinc | accuracy_ci95 (Wilson) | [0.33688, 0.370628] | [0.33688, 0.370628] | PASS |
| report banking77/swap4/zs_from_clinc | accuracy_drop_from_clean | 0.277273 | 0.277273 | PASS |
| report banking77/swap4/zs_from_clinc | confidence_drop_from_clean | 0.22459 | 0.22459 | PASS |
| report banking77/swap4/zs_from_clinc | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.259091, 0.295455] | [0.259091, 0.29513] | PASS |
| report banking77/swap4/zs_from_clinc | jev_minus_model diff | 0.416558 | 0.416558 | PASS |
| report banking77/swap4/zs_from_clinc | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/swap4/zs_from_clinc | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.397727, 0.436039] | [0.397403, 0.43539] | PASS |
| report banking77/keyboard2/bema_baseline | n | 3,080 | 3,080 | PASS |
| report banking77/keyboard2/bema_baseline | accuracy | 0.532468 | 0.532468 | PASS |
| report banking77/keyboard2/bema_baseline | mean_confidence | 0.619491 | 0.619491 | PASS |
| report banking77/keyboard2/bema_baseline | overconfidence | 0.0870238 | 0.0870238 | PASS |
| report banking77/keyboard2/bema_baseline | ece (exact edges) | 0.0870238 | 0.0870238 | PASS |
| report banking77/keyboard2/bema_baseline | ece_repo_float32_edges | 0.0870238 | 0.0870238 | PASS |
| report banking77/keyboard2/bema_baseline | accuracy_ci95 (Wilson) | [0.514817, 0.550037] | [0.514817, 0.550037] | PASS |
| report banking77/keyboard2/bema_baseline | accuracy_drop_from_clean | 0.286039 | 0.286039 | PASS |
| report banking77/keyboard2/bema_baseline | confidence_drop_from_clean | 0.17096 | 0.17096 | PASS |
| report banking77/keyboard2/bema_baseline | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.267857, 0.304229] | [0.267532, 0.303247] | PASS |
| report banking77/keyboard2/bema_baseline | jev_minus_model diff | 0.242857 | 0.242857 | PASS |
| report banking77/keyboard2/bema_baseline | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/keyboard2/bema_baseline | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.221104, 0.26461] | [0.22207, 0.263312] | PASS |
| report banking77/keyboard2/bema_v2 | n | 3,080 | 3,080 | PASS |
| report banking77/keyboard2/bema_v2 | accuracy | 0.691234 | 0.691234 | PASS |
| report banking77/keyboard2/bema_v2 | mean_confidence | 0.741691 | 0.741691 | PASS |
| report banking77/keyboard2/bema_v2 | overconfidence | 0.0504572 | 0.0504572 | PASS |
| report banking77/keyboard2/bema_v2 | ece (exact edges) | 0.0504572 | 0.0504572 | PASS |
| report banking77/keyboard2/bema_v2 | ece_repo_float32_edges | 0.0504572 | 0.0504572 | PASS |
| report banking77/keyboard2/bema_v2 | accuracy_ci95 (Wilson) | [0.674688, 0.707303] | [0.674688, 0.707303] | PASS |
| report banking77/keyboard2/bema_v2 | accuracy_drop_from_clean | 0.132792 | 0.132792 | PASS |
| report banking77/keyboard2/bema_v2 | confidence_drop_from_clean | 0.101995 | 0.101995 | PASS |
| report banking77/keyboard2/bema_v2 | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.119805, 0.146429] | [0.119156, 0.146104] | PASS |
| report banking77/keyboard2/bema_v2 | jev_minus_model diff | 0.0840909 | 0.0840909 | PASS |
| report banking77/keyboard2/bema_v2 | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/keyboard2/bema_v2 | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.0652597, 0.10293] | [0.0655844, 0.102922] | PASS |
| report banking77/keyboard2/bema_v2_spell | n | 3,080 | 3,080 | PASS |
| report banking77/keyboard2/bema_v2_spell | accuracy | 0.804221 | 0.804221 | PASS |
| report banking77/keyboard2/bema_v2_spell | mean_confidence | 0.828377 | 0.828377 | PASS |
| report banking77/keyboard2/bema_v2_spell | overconfidence | 0.0241559 | 0.0241559 | PASS |
| report banking77/keyboard2/bema_v2_spell | ece (exact edges) | 0.0297112 | 0.0297112 | PASS |
| report banking77/keyboard2/bema_v2_spell | ece_repo_float32_edges | 0.0297112 | 0.0297112 | PASS |
| report banking77/keyboard2/bema_v2_spell | accuracy_ci95 (Wilson) | [0.789832, 0.817852] | [0.789832, 0.817852] | PASS |
| report banking77/keyboard2/bema_v2_spell | accuracy_drop_from_clean | 0.0198052 | 0.0198052 | PASS |
| report banking77/keyboard2/bema_v2_spell | confidence_drop_from_clean | 0.0154634 | 0.0154634 | PASS |
| report banking77/keyboard2/bema_v2_spell | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.0136364, 0.0262987] | [0.0136364, 0.0262987] | PASS |
| report banking77/keyboard2/bema_v2_spell | jev_minus_model diff | -0.0288961 | -0.0288961 | PASS |
| report banking77/keyboard2/bema_v2_spell | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/keyboard2/bema_v2_spell | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.0464286, -0.0123377] | [-0.0454545, -0.0123377] | PASS |
| report banking77/keyboard2/zs_offshelf | n | 3,080 | 3,080 | PASS |
| report banking77/keyboard2/zs_offshelf | accuracy | 0.487013 | 0.487013 | PASS |
| report banking77/keyboard2/zs_offshelf | mean_confidence | 0.504166 | 0.504166 | PASS |
| report banking77/keyboard2/zs_offshelf | overconfidence | 0.0171533 | 0.0171533 | PASS |
| report banking77/keyboard2/zs_offshelf | ece (exact edges) | 0.0328688 | 0.0328688 | PASS |
| report banking77/keyboard2/zs_offshelf | ece_repo_float32_edges | 0.0328688 | 0.0328688 | PASS |
| report banking77/keyboard2/zs_offshelf | accuracy_ci95 (Wilson) | [0.469388, 0.50467] | [0.469388, 0.50467] | PASS |
| report banking77/keyboard2/zs_offshelf | accuracy_drop_from_clean | 0.135065 | 0.135065 | PASS |
| report banking77/keyboard2/zs_offshelf | confidence_drop_from_clean | 0.16122 | 0.16122 | PASS |
| report banking77/keyboard2/zs_offshelf | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.120455, 0.15] | [0.120122, 0.149675] | PASS |
| report banking77/keyboard2/zs_offshelf | jev_minus_model diff | 0.288312 | 0.288312 | PASS |
| report banking77/keyboard2/zs_offshelf | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/keyboard2/zs_offshelf | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.270446, 0.307468] | [0.269805, 0.306169] | PASS |
| report banking77/keyboard2/zs_from_clinc | n | 3,080 | 3,080 | PASS |
| report banking77/keyboard2/zs_from_clinc | accuracy | 0.470455 | 0.470455 | PASS |
| report banking77/keyboard2/zs_from_clinc | mean_confidence | 0.59827 | 0.59827 | PASS |
| report banking77/keyboard2/zs_from_clinc | overconfidence | 0.127816 | 0.127816 | PASS |
| report banking77/keyboard2/zs_from_clinc | ece (exact edges) | 0.127816 | 0.127816 | PASS |
| report banking77/keyboard2/zs_from_clinc | ece_repo_float32_edges | 0.127816 | 0.127816 | PASS |
| report banking77/keyboard2/zs_from_clinc | accuracy_ci95 (Wilson) | [0.452875, 0.488108] | [0.452875, 0.488108] | PASS |
| report banking77/keyboard2/zs_from_clinc | accuracy_drop_from_clean | 0.16039 | 0.16039 | PASS |
| report banking77/keyboard2/zs_from_clinc | confidence_drop_from_clean | 0.123305 | 0.123305 | PASS |
| report banking77/keyboard2/zs_from_clinc | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.144805, 0.176307] | [0.144156, 0.175974] | PASS |
| report banking77/keyboard2/zs_from_clinc | jev_minus_model diff | 0.30487 | 0.30487 | PASS |
| report banking77/keyboard2/zs_from_clinc | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/keyboard2/zs_from_clinc | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.286039, 0.323701] | [0.286364, 0.323052] | PASS |
| report banking77/delete2/bema_baseline | n | 3,080 | 3,080 | PASS |
| report banking77/delete2/bema_baseline | accuracy | 0.550649 | 0.550649 | PASS |
| report banking77/delete2/bema_baseline | mean_confidence | 0.634662 | 0.634662 | PASS |
| report banking77/delete2/bema_baseline | overconfidence | 0.0840123 | 0.0840123 | PASS |
| report banking77/delete2/bema_baseline | ece (exact edges) | 0.0899797 | 0.0899797 | PASS |
| report banking77/delete2/bema_baseline | ece_repo_float32_edges | 0.0899797 | 0.0899797 | PASS |
| report banking77/delete2/bema_baseline | accuracy_ci95 (Wilson) | [0.53303, 0.568143] | [0.53303, 0.568143] | PASS |
| report banking77/delete2/bema_baseline | accuracy_drop_from_clean | 0.267857 | 0.267857 | PASS |
| report banking77/delete2/bema_baseline | confidence_drop_from_clean | 0.155789 | 0.155789 | PASS |
| report banking77/delete2/bema_baseline | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.250966, 0.284091] | [0.250649, 0.28539] | PASS |
| report banking77/delete2/bema_baseline | jev_minus_model diff | 0.236364 | 0.236364 | PASS |
| report banking77/delete2/bema_baseline | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/delete2/bema_baseline | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.21461, 0.256818] | [0.21526, 0.256818] | PASS |
| report banking77/delete2/bema_v2 | n | 3,080 | 3,080 | PASS |
| report banking77/delete2/bema_v2 | accuracy | 0.700974 | 0.700974 | PASS |
| report banking77/delete2/bema_v2 | mean_confidence | 0.753434 | 0.753434 | PASS |
| report banking77/delete2/bema_v2 | overconfidence | 0.0524602 | 0.0524602 | PASS |
| report banking77/delete2/bema_v2 | ece (exact edges) | 0.0524602 | 0.0524602 | PASS |
| report banking77/delete2/bema_v2 | ece_repo_float32_edges | 0.0524602 | 0.0524602 | PASS |
| report banking77/delete2/bema_v2 | accuracy_ci95 (Wilson) | [0.684563, 0.716884] | [0.684563, 0.716884] | PASS |
| report banking77/delete2/bema_v2 | accuracy_drop_from_clean | 0.123052 | 0.123052 | PASS |
| report banking77/delete2/bema_v2 | confidence_drop_from_clean | 0.0902513 | 0.0902513 | PASS |
| report banking77/delete2/bema_v2 | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.10974, 0.136364] | [0.10974, 0.136364] | PASS |
| report banking77/delete2/bema_v2 | jev_minus_model diff | 0.086039 | 0.086039 | PASS |
| report banking77/delete2/bema_v2 | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/delete2/bema_v2 | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.0675325, 0.104545] | [0.0675325, 0.104545] | PASS |
| report banking77/delete2/bema_v2_spell | n | 3,080 | 3,080 | PASS |
| report banking77/delete2/bema_v2_spell | accuracy | 0.768182 | 0.768182 | PASS |
| report banking77/delete2/bema_v2_spell | mean_confidence | 0.810919 | 0.810919 | PASS |
| report banking77/delete2/bema_v2_spell | overconfidence | 0.0427367 | 0.0427367 | PASS |
| report banking77/delete2/bema_v2_spell | ece (exact edges) | 0.0444097 | 0.0444097 | PASS |
| report banking77/delete2/bema_v2_spell | ece_repo_float32_edges | 0.0444097 | 0.0444097 | PASS |
| report banking77/delete2/bema_v2_spell | accuracy_ci95 (Wilson) | [0.75295, 0.782745] | [0.75295, 0.782745] | PASS |
| report banking77/delete2/bema_v2_spell | accuracy_drop_from_clean | 0.0558442 | 0.0558442 | PASS |
| report banking77/delete2/bema_v2_spell | confidence_drop_from_clean | 0.0329215 | 0.0329215 | PASS |
| report banking77/delete2/bema_v2_spell | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.0461039, 0.0655844] | [0.0464286, 0.0655844] | PASS |
| report banking77/delete2/bema_v2_spell | jev_minus_model diff | 0.0188312 | 0.0188312 | PASS |
| report banking77/delete2/bema_v2_spell | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/delete2/bema_v2_spell | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.0012987, 0.0366883] | [0.0012987, 0.0363636] | PASS |
| report banking77/delete2/zs_offshelf | n | 3,080 | 3,080 | PASS |
| report banking77/delete2/zs_offshelf | accuracy | 0.492857 | 0.492857 | PASS |
| report banking77/delete2/zs_offshelf | mean_confidence | 0.526067 | 0.526067 | PASS |
| report banking77/delete2/zs_offshelf | overconfidence | 0.0332094 | 0.0332094 | PASS |
| report banking77/delete2/zs_offshelf | ece (exact edges) | 0.0370335 | 0.0370335 | PASS |
| report banking77/delete2/zs_offshelf | ece_repo_float32_edges | 0.0370335 | 0.0370335 | PASS |
| report banking77/delete2/zs_offshelf | accuracy_ci95 (Wilson) | [0.475221, 0.510511] | [0.475221, 0.510511] | PASS |
| report banking77/delete2/zs_offshelf | accuracy_drop_from_clean | 0.129221 | 0.129221 | PASS |
| report banking77/delete2/zs_offshelf | confidence_drop_from_clean | 0.139319 | 0.139319 | PASS |
| report banking77/delete2/zs_offshelf | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.113961, 0.143506] | [0.114286, 0.143506] | PASS |
| report banking77/delete2/zs_offshelf | jev_minus_model diff | 0.294156 | 0.294156 | PASS |
| report banking77/delete2/zs_offshelf | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/delete2/zs_offshelf | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.275641, 0.313636] | [0.275974, 0.312346] | PASS |
| report banking77/delete2/zs_from_clinc | n | 3,080 | 3,080 | PASS |
| report banking77/delete2/zs_from_clinc | accuracy | 0.491234 | 0.491234 | PASS |
| report banking77/delete2/zs_from_clinc | mean_confidence | 0.612095 | 0.612095 | PASS |
| report banking77/delete2/zs_from_clinc | overconfidence | 0.120861 | 0.120861 | PASS |
| report banking77/delete2/zs_from_clinc | ece (exact edges) | 0.120861 | 0.120861 | PASS |
| report banking77/delete2/zs_from_clinc | ece_repo_float32_edges | 0.120861 | 0.120861 | PASS |
| report banking77/delete2/zs_from_clinc | accuracy_ci95 (Wilson) | [0.4736, 0.508889] | [0.4736, 0.508889] | PASS |
| report banking77/delete2/zs_from_clinc | accuracy_drop_from_clean | 0.13961 | 0.13961 | PASS |
| report banking77/delete2/zs_from_clinc | confidence_drop_from_clean | 0.109481 | 0.109481 | PASS |
| report banking77/delete2/zs_from_clinc | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.124026, 0.154221] | [0.124992, 0.154221] | PASS |
| report banking77/delete2/zs_from_clinc | jev_minus_model diff | 0.295779 | 0.295779 | PASS |
| report banking77/delete2/zs_from_clinc | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/delete2/zs_from_clinc | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.277273, 0.314286] | [0.277273, 0.313961] | PASS |
| report banking77/insert2/bema_baseline | n | 3,080 | 3,080 | PASS |
| report banking77/insert2/bema_baseline | accuracy | 0.587338 | 0.587338 | PASS |
| report banking77/insert2/bema_baseline | mean_confidence | 0.651074 | 0.651074 | PASS |
| report banking77/insert2/bema_baseline | overconfidence | 0.0637364 | 0.0637364 | PASS |
| report banking77/insert2/bema_baseline | ece (exact edges) | 0.0652274 | 0.0652274 | PASS |
| report banking77/insert2/bema_baseline | ece_repo_float32_edges | 0.0652274 | 0.0652274 | PASS |
| report banking77/insert2/bema_baseline | accuracy_ci95 (Wilson) | [0.569853, 0.604605] | [0.569853, 0.604605] | PASS |
| report banking77/insert2/bema_baseline | accuracy_drop_from_clean | 0.231169 | 0.231169 | PASS |
| report banking77/insert2/bema_baseline | confidence_drop_from_clean | 0.139377 | 0.139377 | PASS |
| report banking77/insert2/bema_baseline | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.214286, 0.248052] | [0.213961, 0.248385] | PASS |
| report banking77/insert2/bema_baseline | jev_minus_model diff | 0.210714 | 0.210714 | PASS |
| report banking77/insert2/bema_baseline | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/insert2/bema_baseline | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.190252, 0.231494] | [0.19026, 0.230852] | PASS |
| report banking77/insert2/bema_v2 | n | 3,080 | 3,080 | PASS |
| report banking77/insert2/bema_v2 | accuracy | 0.721429 | 0.721429 | PASS |
| report banking77/insert2/bema_v2 | mean_confidence | 0.769772 | 0.769772 | PASS |
| report banking77/insert2/bema_v2 | overconfidence | 0.0483438 | 0.0483438 | PASS |
| report banking77/insert2/bema_v2 | ece (exact edges) | 0.0484382 | 0.0484382 | PASS |
| report banking77/insert2/bema_v2 | ece_repo_float32_edges | 0.0484382 | 0.0484382 | PASS |
| report banking77/insert2/bema_v2 | accuracy_ci95 (Wilson) | [0.705328, 0.736977] | [0.705328, 0.736977] | PASS |
| report banking77/insert2/bema_v2 | accuracy_drop_from_clean | 0.102597 | 0.102597 | PASS |
| report banking77/insert2/bema_v2 | confidence_drop_from_clean | 0.0739131 | 0.0739131 | PASS |
| report banking77/insert2/bema_v2 | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.0905844, 0.11526] | [0.0902597, 0.115268] | PASS |
| report banking77/insert2/bema_v2 | jev_minus_model diff | 0.0766234 | 0.0766234 | PASS |
| report banking77/insert2/bema_v2 | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/insert2/bema_v2 | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.0581169, 0.0944805] | [0.0581088, 0.0948133] | PASS |
| report banking77/insert2/bema_v2_spell | n | 3,080 | 3,080 | PASS |
| report banking77/insert2/bema_v2_spell | accuracy | 0.820779 | 0.820779 | PASS |
| report banking77/insert2/bema_v2_spell | mean_confidence | 0.842456 | 0.842456 | PASS |
| report banking77/insert2/bema_v2_spell | overconfidence | 0.0216766 | 0.0216766 | PASS |
| report banking77/insert2/bema_v2_spell | ece (exact edges) | 0.0248058 | 0.0248058 | PASS |
| report banking77/insert2/bema_v2_spell | ece_repo_float32_edges | 0.0248058 | 0.0248058 | PASS |
| report banking77/insert2/bema_v2_spell | accuracy_ci95 (Wilson) | [0.806837, 0.833922] | [0.806837, 0.833922] | PASS |
| report banking77/insert2/bema_v2_spell | accuracy_drop_from_clean | 0.00324675 | 0.00324675 | PASS |
| report banking77/insert2/bema_v2_spell | confidence_drop_from_clean | 0.00138425 | 0.00138425 | PASS |
| report banking77/insert2/bema_v2_spell | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.000649351, 0.00584416] | [0.000649351, 0.00616883] | PASS |
| report banking77/insert2/bema_v2_spell | jev_minus_model diff | -0.0227273 | -0.0227273 | PASS |
| report banking77/insert2/bema_v2_spell | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/insert2/bema_v2_spell | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.038961, -0.00746753] | [-0.038961, -0.00649351] | PASS |
| report banking77/insert2/zs_offshelf | n | 3,080 | 3,080 | PASS |
| report banking77/insert2/zs_offshelf | accuracy | 0.530519 | 0.530519 | PASS |
| report banking77/insert2/zs_offshelf | mean_confidence | 0.540544 | 0.540544 | PASS |
| report banking77/insert2/zs_offshelf | overconfidence | 0.010025 | 0.010025 | PASS |
| report banking77/insert2/zs_offshelf | ece (exact edges) | 0.0167861 | 0.0167861 | PASS |
| report banking77/insert2/zs_offshelf | ece_repo_float32_edges | 0.0167861 | 0.0167861 | PASS |
| report banking77/insert2/zs_offshelf | accuracy_ci95 (Wilson) | [0.512867, 0.548096] | [0.512867, 0.548096] | PASS |
| report banking77/insert2/zs_offshelf | accuracy_drop_from_clean | 0.0915584 | 0.0915584 | PASS |
| report banking77/insert2/zs_offshelf | confidence_drop_from_clean | 0.124841 | 0.124841 | PASS |
| report banking77/insert2/zs_offshelf | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.0779221, 0.104221] | [0.0782468, 0.104545] | PASS |
| report banking77/insert2/zs_offshelf | jev_minus_model diff | 0.267532 | 0.267532 | PASS |
| report banking77/insert2/zs_offshelf | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/insert2/zs_offshelf | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.250641, 0.284416] | [0.249351, 0.284748] | PASS |
| report banking77/insert2/zs_from_clinc | n | 3,080 | 3,080 | PASS |
| report banking77/insert2/zs_from_clinc | accuracy | 0.518831 | 0.518831 | PASS |
| report banking77/insert2/zs_from_clinc | mean_confidence | 0.626732 | 0.626732 | PASS |
| report banking77/insert2/zs_from_clinc | overconfidence | 0.107901 | 0.107901 | PASS |
| report banking77/insert2/zs_from_clinc | ece (exact edges) | 0.107901 | 0.107901 | PASS |
| report banking77/insert2/zs_from_clinc | ece_repo_float32_edges | 0.107901 | 0.107901 | PASS |
| report banking77/insert2/zs_from_clinc | accuracy_ci95 (Wilson) | [0.501173, 0.536442] | [0.501173, 0.536442] | PASS |
| report banking77/insert2/zs_from_clinc | accuracy_drop_from_clean | 0.112013 | 0.112013 | PASS |
| report banking77/insert2/zs_from_clinc | confidence_drop_from_clean | 0.0948431 | 0.0948431 | PASS |
| report banking77/insert2/zs_from_clinc | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.0974026, 0.126631] | [0.0977273, 0.125974] | PASS |
| report banking77/insert2/zs_from_clinc | jev_minus_model diff | 0.279221 | 0.279221 | PASS |
| report banking77/insert2/zs_from_clinc | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/insert2/zs_from_clinc | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.262338, 0.297411] | [0.261364, 0.296753] | PASS |
| report banking77/mixed3/bema_baseline | n | 3,080 | 3,080 | PASS |
| report banking77/mixed3/bema_baseline | accuracy | 0.447403 | 0.447403 | PASS |
| report banking77/mixed3/bema_baseline | mean_confidence | 0.583887 | 0.583887 | PASS |
| report banking77/mixed3/bema_baseline | overconfidence | 0.136485 | 0.136485 | PASS |
| report banking77/mixed3/bema_baseline | ece (exact edges) | 0.136485 | 0.136485 | PASS |
| report banking77/mixed3/bema_baseline | ece_repo_float32_edges | 0.136485 | 0.136485 | PASS |
| report banking77/mixed3/bema_baseline | accuracy_ci95 (Wilson) | [0.429919, 0.465017] | [0.429919, 0.465017] | PASS |
| report banking77/mixed3/bema_baseline | accuracy_drop_from_clean | 0.371104 | 0.371104 | PASS |
| report banking77/mixed3/bema_baseline | confidence_drop_from_clean | 0.206564 | 0.206564 | PASS |
| report banking77/mixed3/bema_baseline | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.352922, 0.388636] | [0.353247, 0.389619] | PASS |
| report banking77/mixed3/bema_baseline | jev_minus_model diff | 0.326623 | 0.326623 | PASS |
| report banking77/mixed3/bema_baseline | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/mixed3/bema_baseline | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.306169, 0.347078] | [0.305519, 0.347078] | PASS |
| report banking77/mixed3/bema_v2 | n | 3,080 | 3,080 | PASS |
| report banking77/mixed3/bema_v2 | accuracy | 0.644481 | 0.644481 | PASS |
| report banking77/mixed3/bema_v2 | mean_confidence | 0.712254 | 0.712254 | PASS |
| report banking77/mixed3/bema_v2 | overconfidence | 0.0677735 | 0.0677735 | PASS |
| report banking77/mixed3/bema_v2 | ece (exact edges) | 0.0677735 | 0.0677735 | PASS |
| report banking77/mixed3/bema_v2 | ece_repo_float32_edges | 0.0677735 | 0.0677735 | PASS |
| report banking77/mixed3/bema_v2 | accuracy_ci95 (Wilson) | [0.627405, 0.661196] | [0.627405, 0.661196] | PASS |
| report banking77/mixed3/bema_v2 | accuracy_drop_from_clean | 0.179545 | 0.179545 | PASS |
| report banking77/mixed3/bema_v2 | confidence_drop_from_clean | 0.131431 | 0.131431 | PASS |
| report banking77/mixed3/bema_v2 | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.164286, 0.194805] | [0.163961, 0.194805] | PASS |
| report banking77/mixed3/bema_v2 | jev_minus_model diff | 0.129545 | 0.129545 | PASS |
| report banking77/mixed3/bema_v2 | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/mixed3/bema_v2 | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.110057, 0.149351] | [0.11039, 0.14806] | PASS |
| report banking77/mixed3/bema_v2_spell | n | 3,080 | 3,080 | PASS |
| report banking77/mixed3/bema_v2_spell | accuracy | 0.787338 | 0.787338 | PASS |
| report banking77/mixed3/bema_v2_spell | mean_confidence | 0.815536 | 0.815536 | PASS |
| report banking77/mixed3/bema_v2_spell | overconfidence | 0.028198 | 0.028198 | PASS |
| report banking77/mixed3/bema_v2_spell | ece (exact edges) | 0.0294663 | 0.0294663 | PASS |
| report banking77/mixed3/bema_v2_spell | ece_repo_float32_edges | 0.0294663 | 0.0294663 | PASS |
| report banking77/mixed3/bema_v2_spell | accuracy_ci95 (Wilson) | [0.772533, 0.801426] | [0.772533, 0.801426] | PASS |
| report banking77/mixed3/bema_v2_spell | accuracy_drop_from_clean | 0.0366883 | 0.0366883 | PASS |
| report banking77/mixed3/bema_v2_spell | confidence_drop_from_clean | 0.0283044 | 0.0283044 | PASS |
| report banking77/mixed3/bema_v2_spell | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.0282468, 0.0454545] | [0.0279221, 0.0454545] | PASS |
| report banking77/mixed3/bema_v2_spell | jev_minus_model diff | -0.0133117 | -0.0133117 | PASS |
| report banking77/mixed3/bema_v2_spell | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/mixed3/bema_v2_spell | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.0305195, 0.00357143] | [-0.0302029, 0.00422078] | PASS |
| report banking77/mixed3/zs_offshelf | n | 3,080 | 3,080 | PASS |
| report banking77/mixed3/zs_offshelf | accuracy | 0.437013 | 0.437013 | PASS |
| report banking77/mixed3/zs_offshelf | mean_confidence | 0.467608 | 0.467608 | PASS |
| report banking77/mixed3/zs_offshelf | overconfidence | 0.0305953 | 0.0305953 | PASS |
| report banking77/mixed3/zs_offshelf | ece (exact edges) | 0.0319061 | 0.0319061 | PASS |
| report banking77/mixed3/zs_offshelf | ece_repo_float32_edges | 0.0319061 | 0.0319061 | PASS |
| report banking77/mixed3/zs_offshelf | accuracy_ci95 (Wilson) | [0.419585, 0.454598] | [0.419585, 0.454598] | PASS |
| report banking77/mixed3/zs_offshelf | accuracy_drop_from_clean | 0.185065 | 0.185065 | PASS |
| report banking77/mixed3/zs_offshelf | confidence_drop_from_clean | 0.197778 | 0.197778 | PASS |
| report banking77/mixed3/zs_offshelf | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.168506, 0.200333] | [0.167857, 0.201623] | PASS |
| report banking77/mixed3/zs_offshelf | jev_minus_model diff | 0.337013 | 0.337013 | PASS |
| report banking77/mixed3/zs_offshelf | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/mixed3/zs_offshelf | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.317857, 0.355195] | [0.3172, 0.356169] | PASS |
| report banking77/mixed3/zs_from_clinc | n | 3,080 | 3,080 | PASS |
| report banking77/mixed3/zs_from_clinc | accuracy | 0.441883 | 0.441883 | PASS |
| report banking77/mixed3/zs_from_clinc | mean_confidence | 0.565116 | 0.565116 | PASS |
| report banking77/mixed3/zs_from_clinc | overconfidence | 0.123233 | 0.123233 | PASS |
| report banking77/mixed3/zs_from_clinc | ece (exact edges) | 0.123403 | 0.123403 | PASS |
| report banking77/mixed3/zs_from_clinc | ece_repo_float32_edges | 0.123403 | 0.123403 | PASS |
| report banking77/mixed3/zs_from_clinc | accuracy_ci95 (Wilson) | [0.424428, 0.459483] | [0.424428, 0.459483] | PASS |
| report banking77/mixed3/zs_from_clinc | accuracy_drop_from_clean | 0.188961 | 0.188961 | PASS |
| report banking77/mixed3/zs_from_clinc | confidence_drop_from_clean | 0.156459 | 0.156459 | PASS |
| report banking77/mixed3/zs_from_clinc | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0.172078, 0.205195] | [0.171753, 0.205519] | PASS |
| report banking77/mixed3/zs_from_clinc | jev_minus_model diff | 0.332143 | 0.332143 | PASS |
| report banking77/mixed3/zs_from_clinc | jev_minus_model n | 3,080 | 3,080 | PASS |
| report banking77/mixed3/zs_from_clinc | jev_minus_model ci95 (own bootstrap, tol 0.005) | [0.313636, 0.351299] | [0.312662, 0.350649] | PASS |
| report spam/clean/bema_baseline | n | 836 | 836 | PASS |
| report spam/clean/bema_baseline | accuracy | 0.978469 | 0.978469 | PASS |
| report spam/clean/bema_baseline | mean_confidence | 0.979501 | 0.979501 | PASS |
| report spam/clean/bema_baseline | overconfidence | 0.00103186 | 0.00103186 | PASS |
| report spam/clean/bema_baseline | ece (exact edges) | 0.0059728 | 0.0059728 | PASS |
| report spam/clean/bema_baseline | ece_repo_float32_edges | 0.0059728 | 0.0059728 | PASS |
| report spam/clean/bema_baseline | accuracy_ci95 (Wilson) | [0.966223, 0.986338] | [0.966223, 0.986338] | PASS |
| report spam/clean/bema_baseline | spam_precision | 0.945455 | 0.945455 | PASS |
| report spam/clean/bema_baseline | spam_recall | 0.896552 | 0.896552 | PASS |
| report spam/clean/bema_baseline | spam_f1 | 0.920354 | 0.920354 | PASS |
| report spam/clean/bema_baseline | false_alarms_on_ham | 6 | 6 | PASS |
| report spam/clean/bema_baseline | missed_spam | 12 | 12 | PASS |
| report spam/clean/bema_baseline | jev_minus_model diff | -0.0179426 | -0.0179426 | PASS |
| report spam/clean/bema_baseline | jev_minus_model n | 836 | 836 | PASS |
| report spam/clean/bema_baseline | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.0334928, -0.00119617] | [-0.034689, -0.00239234] | PASS |
| report spam/clean/bema_v2 | n | 836 | 836 | PASS |
| report spam/clean/bema_v2 | accuracy | 0.986842 | 0.986842 | PASS |
| report spam/clean/bema_v2 | mean_confidence | 0.987562 | 0.987562 | PASS |
| report spam/clean/bema_v2 | overconfidence | 0.000720137 | 0.000720137 | PASS |
| report spam/clean/bema_v2 | ece (exact edges) | 0.00521039 | 0.00521039 | PASS |
| report spam/clean/bema_v2 | ece_repo_float32_edges | 0.00521039 | 0.00521039 | PASS |
| report spam/clean/bema_v2 | accuracy_ci95 (Wilson) | [0.976593, 0.992637] | [0.976593, 0.992637] | PASS |
| report spam/clean/bema_v2 | spam_precision | 0.948718 | 0.948718 | PASS |
| report spam/clean/bema_v2 | spam_recall | 0.956897 | 0.956897 | PASS |
| report spam/clean/bema_v2 | spam_f1 | 0.95279 | 0.95279 | PASS |
| report spam/clean/bema_v2 | false_alarms_on_ham | 6 | 6 | PASS |
| report spam/clean/bema_v2 | missed_spam | 5 | 5 | PASS |
| report spam/clean/bema_v2 | jev_minus_model diff | -0.0263158 | -0.0263158 | PASS |
| report spam/clean/bema_v2 | jev_minus_model n | 836 | 836 | PASS |
| report spam/clean/bema_v2 | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.0406699, -0.0131579] | [-0.0406699, -0.0119617] | PASS |
| report spam/clean/bema_v2_spell | n | 836 | 836 | PASS |
| report spam/clean/bema_v2_spell | accuracy | 0.986842 | 0.986842 | PASS |
| report spam/clean/bema_v2_spell | mean_confidence | 0.987827 | 0.987827 | PASS |
| report spam/clean/bema_v2_spell | overconfidence | 0.000985176 | 0.000985176 | PASS |
| report spam/clean/bema_v2_spell | ece (exact edges) | 0.0071372 | 0.0071372 | PASS |
| report spam/clean/bema_v2_spell | ece_repo_float32_edges | 0.0071372 | 0.0071372 | PASS |
| report spam/clean/bema_v2_spell | accuracy_ci95 (Wilson) | [0.976593, 0.992637] | [0.976593, 0.992637] | PASS |
| report spam/clean/bema_v2_spell | spam_precision | 0.948718 | 0.948718 | PASS |
| report spam/clean/bema_v2_spell | spam_recall | 0.956897 | 0.956897 | PASS |
| report spam/clean/bema_v2_spell | spam_f1 | 0.95279 | 0.95279 | PASS |
| report spam/clean/bema_v2_spell | false_alarms_on_ham | 6 | 6 | PASS |
| report spam/clean/bema_v2_spell | missed_spam | 5 | 5 | PASS |
| report spam/clean/bema_v2_spell | jev_minus_model diff | -0.0263158 | -0.0263158 | PASS |
| report spam/clean/bema_v2_spell | jev_minus_model n | 836 | 836 | PASS |
| report spam/clean/bema_v2_spell | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.0406699, -0.0131579] | [-0.0406699, -0.0119617] | PASS |
| report spam/swap2/bema_baseline | n | 836 | 836 | PASS |
| report spam/swap2/bema_baseline | accuracy | 0.972488 | 0.972488 | PASS |
| report spam/swap2/bema_baseline | mean_confidence | 0.976007 | 0.976007 | PASS |
| report spam/swap2/bema_baseline | overconfidence | 0.00351856 | 0.00351856 | PASS |
| report spam/swap2/bema_baseline | ece (exact edges) | 0.00659775 | 0.00659775 | PASS |
| report spam/swap2/bema_baseline | ece_repo_float32_edges | 0.00659775 | 0.00659775 | PASS |
| report spam/swap2/bema_baseline | accuracy_ci95 (Wilson) | [0.959055, 0.981598] | [0.959055, 0.981598] | PASS |
| report spam/swap2/bema_baseline | spam_precision | 0.934579 | 0.934579 | PASS |
| report spam/swap2/bema_baseline | spam_recall | 0.862069 | 0.862069 | PASS |
| report spam/swap2/bema_baseline | spam_f1 | 0.896861 | 0.896861 | PASS |
| report spam/swap2/bema_baseline | false_alarms_on_ham | 7 | 7 | PASS |
| report spam/swap2/bema_baseline | missed_spam | 16 | 16 | PASS |
| report spam/swap2/bema_baseline | accuracy_drop_from_clean | 0.00598086 | 0.00598086 | PASS |
| report spam/swap2/bema_baseline | confidence_drop_from_clean | 0.00349417 | 0.00349417 | PASS |
| report spam/swap2/bema_baseline | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [-0.00119617, 0.0143541] | [-0.00119617, 0.0143541] | PASS |
| report spam/swap2/bema_baseline | jev_minus_model diff | -0.0167464 | -0.0167464 | PASS |
| report spam/swap2/bema_baseline | jev_minus_model n | 836 | 836 | PASS |
| report spam/swap2/bema_baseline | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.034689, 0.00119617] | [-0.034689, 0.00119617] | PASS |
| report spam/swap2/bema_v2 | n | 836 | 836 | PASS |
| report spam/swap2/bema_v2 | accuracy | 0.98445 | 0.98445 | PASS |
| report spam/swap2/bema_v2 | mean_confidence | 0.986785 | 0.986785 | PASS |
| report spam/swap2/bema_v2 | overconfidence | 0.00233503 | 0.00233503 | PASS |
| report spam/swap2/bema_v2 | ece (exact edges) | 0.00594868 | 0.00594868 | PASS |
| report spam/swap2/bema_v2 | ece_repo_float32_edges | 0.00594868 | 0.00594868 | PASS |
| report spam/swap2/bema_v2 | accuracy_ci95 (Wilson) | [0.973578, 0.99089] | [0.973578, 0.99089] | PASS |
| report spam/swap2/bema_v2 | spam_precision | 0.940171 | 0.940171 | PASS |
| report spam/swap2/bema_v2 | spam_recall | 0.948276 | 0.948276 | PASS |
| report spam/swap2/bema_v2 | spam_f1 | 0.944206 | 0.944206 | PASS |
| report spam/swap2/bema_v2 | false_alarms_on_ham | 7 | 7 | PASS |
| report spam/swap2/bema_v2 | missed_spam | 6 | 6 | PASS |
| report spam/swap2/bema_v2 | accuracy_drop_from_clean | 0.00239234 | 0.00239234 | PASS |
| report spam/swap2/bema_v2 | confidence_drop_from_clean | 0.000777447 | 0.000777447 | PASS |
| report spam/swap2/bema_v2 | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0, 0.00598086] | [0, 0.00598086] | PASS |
| report spam/swap2/bema_v2 | jev_minus_model diff | -0.0287081 | -0.0287081 | PASS |
| report spam/swap2/bema_v2 | jev_minus_model n | 836 | 836 | PASS |
| report spam/swap2/bema_v2 | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.0430622, -0.0143541] | [-0.0430622, -0.0143541] | PASS |
| report spam/swap2/bema_v2_spell | n | 836 | 836 | PASS |
| report spam/swap2/bema_v2_spell | accuracy | 0.985646 | 0.985646 | PASS |
| report spam/swap2/bema_v2_spell | mean_confidence | 0.987916 | 0.987916 | PASS |
| report spam/swap2/bema_v2_spell | overconfidence | 0.00227006 | 0.00227006 | PASS |
| report spam/swap2/bema_v2_spell | ece (exact edges) | 0.00677449 | 0.00677449 | PASS |
| report spam/swap2/bema_v2_spell | ece_repo_float32_edges | 0.00677449 | 0.00677449 | PASS |
| report spam/swap2/bema_v2_spell | accuracy_ci95 (Wilson) | [0.975079, 0.99177] | [0.975079, 0.99177] | PASS |
| report spam/swap2/bema_v2_spell | spam_precision | 0.948276 | 0.948276 | PASS |
| report spam/swap2/bema_v2_spell | spam_recall | 0.948276 | 0.948276 | PASS |
| report spam/swap2/bema_v2_spell | spam_f1 | 0.948276 | 0.948276 | PASS |
| report spam/swap2/bema_v2_spell | false_alarms_on_ham | 6 | 6 | PASS |
| report spam/swap2/bema_v2_spell | missed_spam | 6 | 6 | PASS |
| report spam/swap2/bema_v2_spell | accuracy_drop_from_clean | 0.00119617 | 0.00119617 | PASS |
| report spam/swap2/bema_v2_spell | confidence_drop_from_clean | -8.87146e-05 | -8.87146e-05 | PASS |
| report spam/swap2/bema_v2_spell | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0, 0.00358852] | [0, 0.00358852] | PASS |
| report spam/swap2/bema_v2_spell | jev_minus_model diff | -0.0299043 | -0.0299043 | PASS |
| report spam/swap2/bema_v2_spell | jev_minus_model n | 836 | 836 | PASS |
| report spam/swap2/bema_v2_spell | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.0454545, -0.0155502] | [-0.0442584, -0.0155502] | PASS |
| report spam/keyboard2/bema_baseline | n | 836 | 836 | PASS |
| report spam/keyboard2/bema_baseline | accuracy | 0.971292 | 0.971292 | PASS |
| report spam/keyboard2/bema_baseline | mean_confidence | 0.977483 | 0.977483 | PASS |
| report spam/keyboard2/bema_baseline | overconfidence | 0.00619145 | 0.00619145 | PASS |
| report spam/keyboard2/bema_baseline | ece (exact edges) | 0.00674202 | 0.00674202 | PASS |
| report spam/keyboard2/bema_baseline | ece_repo_float32_edges | 0.00674202 | 0.00674202 | PASS |
| report spam/keyboard2/bema_baseline | accuracy_ci95 (Wilson) | [0.957639, 0.980634] | [0.957639, 0.980634] | PASS |
| report spam/keyboard2/bema_baseline | spam_precision | 0.918182 | 0.918182 | PASS |
| report spam/keyboard2/bema_baseline | spam_recall | 0.87069 | 0.87069 | PASS |
| report spam/keyboard2/bema_baseline | spam_f1 | 0.893805 | 0.893805 | PASS |
| report spam/keyboard2/bema_baseline | false_alarms_on_ham | 9 | 9 | PASS |
| report spam/keyboard2/bema_baseline | missed_spam | 15 | 15 | PASS |
| report spam/keyboard2/bema_baseline | accuracy_drop_from_clean | 0.00717703 | 0.00717703 | PASS |
| report spam/keyboard2/bema_baseline | confidence_drop_from_clean | 0.00201745 | 0.00201745 | PASS |
| report spam/keyboard2/bema_baseline | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [-0.00239234, 0.0167464] | [-0.00239234, 0.0167464] | PASS |
| report spam/keyboard2/bema_baseline | jev_minus_model diff | -0.0167464 | -0.0167464 | PASS |
| report spam/keyboard2/bema_baseline | jev_minus_model n | 836 | 836 | PASS |
| report spam/keyboard2/bema_baseline | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.034689, 0] | [-0.034689, 0] | PASS |
| report spam/keyboard2/bema_v2 | n | 836 | 836 | PASS |
| report spam/keyboard2/bema_v2 | accuracy | 0.98445 | 0.98445 | PASS |
| report spam/keyboard2/bema_v2 | mean_confidence | 0.986655 | 0.986655 | PASS |
| report spam/keyboard2/bema_v2 | overconfidence | 0.00220488 | 0.00220488 | PASS |
| report spam/keyboard2/bema_v2 | ece (exact edges) | 0.00807106 | 0.00807106 | PASS |
| report spam/keyboard2/bema_v2 | ece_repo_float32_edges | 0.00807106 | 0.00807106 | PASS |
| report spam/keyboard2/bema_v2 | accuracy_ci95 (Wilson) | [0.973578, 0.99089] | [0.973578, 0.99089] | PASS |
| report spam/keyboard2/bema_v2 | spam_precision | 0.940171 | 0.940171 | PASS |
| report spam/keyboard2/bema_v2 | spam_recall | 0.948276 | 0.948276 | PASS |
| report spam/keyboard2/bema_v2 | spam_f1 | 0.944206 | 0.944206 | PASS |
| report spam/keyboard2/bema_v2 | false_alarms_on_ham | 7 | 7 | PASS |
| report spam/keyboard2/bema_v2 | missed_spam | 6 | 6 | PASS |
| report spam/keyboard2/bema_v2 | accuracy_drop_from_clean | 0.00239234 | 0.00239234 | PASS |
| report spam/keyboard2/bema_v2 | confidence_drop_from_clean | 0.000907599 | 0.000907599 | PASS |
| report spam/keyboard2/bema_v2 | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0, 0.00598086] | [0, 0.00598086] | PASS |
| report spam/keyboard2/bema_v2 | jev_minus_model diff | -0.0299043 | -0.0299043 | PASS |
| report spam/keyboard2/bema_v2 | jev_minus_model n | 836 | 836 | PASS |
| report spam/keyboard2/bema_v2 | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.0454545, -0.0155203] | [-0.0454545, -0.0143541] | PASS |
| report spam/keyboard2/bema_v2_spell | n | 836 | 836 | PASS |
| report spam/keyboard2/bema_v2_spell | accuracy | 0.985646 | 0.985646 | PASS |
| report spam/keyboard2/bema_v2_spell | mean_confidence | 0.987874 | 0.987874 | PASS |
| report spam/keyboard2/bema_v2_spell | overconfidence | 0.00222778 | 0.00222778 | PASS |
| report spam/keyboard2/bema_v2_spell | ece (exact edges) | 0.00537473 | 0.00537473 | PASS |
| report spam/keyboard2/bema_v2_spell | ece_repo_float32_edges | 0.00537473 | 0.00537473 | PASS |
| report spam/keyboard2/bema_v2_spell | accuracy_ci95 (Wilson) | [0.975079, 0.99177] | [0.975079, 0.99177] | PASS |
| report spam/keyboard2/bema_v2_spell | spam_precision | 0.940678 | 0.940678 | PASS |
| report spam/keyboard2/bema_v2_spell | spam_recall | 0.956897 | 0.956897 | PASS |
| report spam/keyboard2/bema_v2_spell | spam_f1 | 0.948718 | 0.948718 | PASS |
| report spam/keyboard2/bema_v2_spell | false_alarms_on_ham | 7 | 7 | PASS |
| report spam/keyboard2/bema_v2_spell | missed_spam | 5 | 5 | PASS |
| report spam/keyboard2/bema_v2_spell | accuracy_drop_from_clean | 0.00119617 | 0.00119617 | PASS |
| report spam/keyboard2/bema_v2_spell | confidence_drop_from_clean | -4.64292e-05 | -4.64292e-05 | PASS |
| report spam/keyboard2/bema_v2_spell | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0, 0.00358852] | [0, 0.00358852] | PASS |
| report spam/keyboard2/bema_v2_spell | jev_minus_model diff | -0.0311005 | -0.0311005 | PASS |
| report spam/keyboard2/bema_v2_spell | jev_minus_model n | 836 | 836 | PASS |
| report spam/keyboard2/bema_v2_spell | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.0466507, -0.0167464] | [-0.0466507, -0.0155502] | PASS |
| report spam/mixed3/bema_baseline | n | 836 | 836 | PASS |
| report spam/mixed3/bema_baseline | accuracy | 0.976077 | 0.976077 | PASS |
| report spam/mixed3/bema_baseline | mean_confidence | 0.98261 | 0.98261 | PASS |
| report spam/mixed3/bema_baseline | overconfidence | 0.00653354 | 0.00653354 | PASS |
| report spam/mixed3/bema_baseline | ece (exact edges) | 0.0075359 | 0.0075359 | PASS |
| report spam/mixed3/bema_baseline | ece_repo_float32_edges | 0.0075359 | 0.0075359 | PASS |
| report spam/mixed3/bema_baseline | accuracy_ci95 (Wilson) | [0.963337, 0.984461] | [0.963337, 0.984461] | PASS |
| report spam/mixed3/bema_baseline | spam_precision | 0.936364 | 0.936364 | PASS |
| report spam/mixed3/bema_baseline | spam_recall | 0.887931 | 0.887931 | PASS |
| report spam/mixed3/bema_baseline | spam_f1 | 0.911504 | 0.911504 | PASS |
| report spam/mixed3/bema_baseline | false_alarms_on_ham | 7 | 7 | PASS |
| report spam/mixed3/bema_baseline | missed_spam | 13 | 13 | PASS |
| report spam/mixed3/bema_baseline | accuracy_drop_from_clean | 0.00239234 | 0.00239234 | PASS |
| report spam/mixed3/bema_baseline | confidence_drop_from_clean | -0.00310933 | -0.00310933 | PASS |
| report spam/mixed3/bema_baseline | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [-0.00598086, 0.0107656] | [-0.00598086, 0.0107656] | PASS |
| report spam/mixed3/bema_baseline | jev_minus_model diff | -0.0227273 | -0.0227273 | PASS |
| report spam/mixed3/bema_baseline | jev_minus_model n | 836 | 836 | PASS |
| report spam/mixed3/bema_baseline | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.0395036, -0.00478469] | [-0.0406699, -0.00598086] | PASS |
| report spam/mixed3/bema_v2 | n | 836 | 836 | PASS |
| report spam/mixed3/bema_v2 | accuracy | 0.983254 | 0.983254 | PASS |
| report spam/mixed3/bema_v2 | mean_confidence | 0.985779 | 0.985779 | PASS |
| report spam/mixed3/bema_v2 | overconfidence | 0.00252535 | 0.00252535 | PASS |
| report spam/mixed3/bema_v2 | ece (exact edges) | 0.00402039 | 0.00402039 | PASS |
| report spam/mixed3/bema_v2 | ece_repo_float32_edges | 0.00402039 | 0.00402039 | PASS |
| report spam/mixed3/bema_v2 | accuracy_ci95 (Wilson) | [0.972088, 0.989999] | [0.972088, 0.989999] | PASS |
| report spam/mixed3/bema_v2 | spam_precision | 0.925 | 0.925 | PASS |
| report spam/mixed3/bema_v2 | spam_recall | 0.956897 | 0.956897 | PASS |
| report spam/mixed3/bema_v2 | spam_f1 | 0.940678 | 0.940678 | PASS |
| report spam/mixed3/bema_v2 | false_alarms_on_ham | 9 | 9 | PASS |
| report spam/mixed3/bema_v2 | missed_spam | 5 | 5 | PASS |
| report spam/mixed3/bema_v2 | accuracy_drop_from_clean | 0.00358852 | 0.00358852 | PASS |
| report spam/mixed3/bema_v2 | confidence_drop_from_clean | 0.00178331 | 0.00178331 | PASS |
| report spam/mixed3/bema_v2 | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [0, 0.00837321] | [0, 0.00837321] | PASS |
| report spam/mixed3/bema_v2 | jev_minus_model diff | -0.0299043 | -0.0299043 | PASS |
| report spam/mixed3/bema_v2 | jev_minus_model n | 836 | 836 | PASS |
| report spam/mixed3/bema_v2 | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.0454545, -0.0143541] | [-0.0454545, -0.0143541] | PASS |
| report spam/mixed3/bema_v2_spell | n | 836 | 836 | PASS |
| report spam/mixed3/bema_v2_spell | accuracy | 0.986842 | 0.986842 | PASS |
| report spam/mixed3/bema_v2_spell | mean_confidence | 0.987408 | 0.987408 | PASS |
| report spam/mixed3/bema_v2_spell | overconfidence | 0.000565742 | 0.000565742 | PASS |
| report spam/mixed3/bema_v2_spell | ece (exact edges) | 0.0060384 | 0.0060384 | PASS |
| report spam/mixed3/bema_v2_spell | ece_repo_float32_edges | 0.0060384 | 0.0060384 | PASS |
| report spam/mixed3/bema_v2_spell | accuracy_ci95 (Wilson) | [0.976593, 0.992637] | [0.976593, 0.992637] | PASS |
| report spam/mixed3/bema_v2_spell | spam_precision | 0.948718 | 0.948718 | PASS |
| report spam/mixed3/bema_v2_spell | spam_recall | 0.956897 | 0.956897 | PASS |
| report spam/mixed3/bema_v2_spell | spam_f1 | 0.95279 | 0.95279 | PASS |
| report spam/mixed3/bema_v2_spell | false_alarms_on_ham | 6 | 6 | PASS |
| report spam/mixed3/bema_v2_spell | missed_spam | 5 | 5 | PASS |
| report spam/mixed3/bema_v2_spell | accuracy_drop_from_clean | 0 | 0 | PASS |
| report spam/mixed3/bema_v2_spell | confidence_drop_from_clean | 0.000419434 | 0.000419434 | PASS |
| report spam/mixed3/bema_v2_spell | accuracy_drop_ci95_paired_bootstrap (own bootstrap, tol 0.005) | [-0.00358852, 0.00358852] | [-0.00358852, 0.00358852] | PASS |
| report spam/mixed3/bema_v2_spell | jev_minus_model diff | -0.0334928 | -0.0334928 | PASS |
| report spam/mixed3/bema_v2_spell | jev_minus_model n | 836 | 836 | PASS |
| report spam/mixed3/bema_v2_spell | jev_minus_model ci95 (own bootstrap, tol 0.005) | [-0.0502392, -0.0191388] | [-0.0490431, -0.0191388] | PASS |
| report | spam_majority_baseline_accuracy | 0.861244 | 0.861244 | PASS |
| report | bema_latency equals runs/bema/timing_quiet.json | {"_machine": {"cpu_count": 2, "load_avg_at_start": [1.70751953125, 1.984375, 2.58056640625], "torch_threads": 2}, "bema_baseline": {"median_ms": 1.6977830000541871, "note": "BEMA v1 forward pass, all 3 heads, batch 1, CPU", "p90_ms": 2.1160249998501968, "tokenize_median_ms": 0.11070199980167672}, "bema_v2": {"median_ms": 14.016307500241965, "note": "BEMA v2 forward pass (MiniLM-L6 + 3 heads), batch 1, CPU", "p90_ms": 19.740895000722958, "tokenize_median_ms": 0.450044501121738}, "bema_v2_spell": {"median_ms": 14.06858399968769, "note": "BEMA v2 forward pass plus the spelling correction step", "speller_median_ms": 0.05402250008046394}} | {"_machine": {"cpu_count": 2, "load_avg_at_start": [1.70751953125, 1.984375, 2.58056640625], "torch_threads": 2}, "bema_baseline": {"median_ms": 1.6977830000541871, "note": "BEMA v1 forward pass, all 3 heads, batch 1, CPU", "p90_ms": 2.1160249998501968, "tokenize_median_ms": 0.11070199980167672}, "bema_v2": {"median_ms": 14.016307500241965, "note": "BEMA v2 forward pass (MiniLM-L6 + 3 heads), batch 1, CPU", "p90_ms": 19.740895000722958, "tokenize_median_ms": 0.450044501121738}, "bema_v2_spell": {"median_ms": 14.06858399968769, "note": "BEMA v2 forward pass plus the spelling correction step", "speller_median_ms": 0.05402250008046394}} | PASS |
| redteam injection_banking/bema_v2 | n | 1,000 | 1,000 | PASS |
| redteam injection_banking/bema_v2 | accuracy | 0.469 | 0.469 | PASS |
| redteam injection_banking/bema_v2 | mean_confidence | 0.688497 | 0.688497 | PASS |
| redteam injection_banking/bema_v2 | overconfidence | 0.219497 | 0.219497 | PASS |
| redteam injection_banking/bema_v2 | ece | 0.22318 | 0.22318 | PASS |
| redteam injection_banking/bema_v2 | accuracy_ci95 (Wilson) | [0.438248, 0.49999] | [0.438248, 0.49999] | PASS |
| redteam injection_banking/bema_v2 | clean_accuracy_same_examples | 0.813 | 0.813 | PASS |
| redteam injection_banking/bema_v2 | followed_injected_label | 0.237 | 0.237 | PASS |
| redteam injection_banking/zs_from_clinc | n | 1,000 | 1,000 | PASS |
| redteam injection_banking/zs_from_clinc | accuracy | 0.456 | 0.456 | PASS |
| redteam injection_banking/zs_from_clinc | mean_confidence | 0.596502 | 0.596502 | PASS |
| redteam injection_banking/zs_from_clinc | overconfidence | 0.140502 | 0.140502 | PASS |
| redteam injection_banking/zs_from_clinc | ece | 0.140502 | 0.140502 | PASS |
| redteam injection_banking/zs_from_clinc | accuracy_ci95 (Wilson) | [0.425358, 0.486979] | [0.425358, 0.486979] | PASS |
| redteam injection_banking/zs_from_clinc | clean_accuracy_same_examples | 0.625 | 0.625 | PASS |
| redteam injection_banking/zs_from_clinc | followed_injected_label | 0.182 | 0.182 | PASS |
| redteam injection_banking/zs_offshelf | n | 1,000 | 1,000 | PASS |
| redteam injection_banking/zs_offshelf | accuracy | 0.473 | 0.473 | PASS |
| redteam injection_banking/zs_offshelf | mean_confidence | 0.532621 | 0.532621 | PASS |
| redteam injection_banking/zs_offshelf | overconfidence | 0.0596207 | 0.0596207 | PASS |
| redteam injection_banking/zs_offshelf | ece | 0.0596207 | 0.0596207 | PASS |
| redteam injection_banking/zs_offshelf | accuracy_ci95 (Wilson) | [0.442218, 0.503989] | [0.442218, 0.503989] | PASS |
| redteam injection_banking/zs_offshelf | clean_accuracy_same_examples | 0.601 | 0.601 | PASS |
| redteam injection_banking/zs_offshelf | followed_injected_label | 0.202 | 0.202 | PASS |
| redteam injection_banking/bema_baseline | n | 1,000 | 1,000 | PASS |
| redteam injection_banking/bema_baseline | accuracy | 0.351 | 0.351 | PASS |
| redteam injection_banking/bema_baseline | mean_confidence | 0.581917 | 0.581917 | PASS |
| redteam injection_banking/bema_baseline | overconfidence | 0.230917 | 0.230917 | PASS |
| redteam injection_banking/bema_baseline | ece | 0.230917 | 0.230917 | PASS |
| redteam injection_banking/bema_baseline | accuracy_ci95 (Wilson) | [0.32204, 0.381101] | [0.32204, 0.381101] | PASS |
| redteam injection_banking/bema_baseline | clean_accuracy_same_examples | 0.82 | 0.82 | PASS |
| redteam injection_banking/bema_baseline | followed_injected_label | 0.174 | 0.174 | PASS |
| redteam injection_spam/bema_v2 | n | 116 | 116 | PASS |
| redteam injection_spam/bema_v2 | accuracy | 0.974138 | 0.974138 | PASS |
| redteam injection_spam/bema_v2 | mean_confidence | 0.985192 | 0.985192 | PASS |
| redteam injection_spam/bema_v2 | overconfidence | 0.0110543 | 0.0110543 | PASS |
| redteam injection_spam/bema_v2 | ece | 0.0172851 | 0.0172851 | PASS |
| redteam injection_spam/bema_v2 | accuracy_ci95 (Wilson) | [0.926713, 0.991166] | [0.926713, 0.991166] | PASS |
| redteam injection_spam/bema_v2 | mean_p_spam | 0.970369 | 0.970369 | PASS |
| redteam injection_spam/bema_v2 | clean_recall_same_examples | 0.956897 | 0.956897 | PASS |
| redteam injection_spam/bema_baseline | n | 116 | 116 | PASS |
| redteam injection_spam/bema_baseline | accuracy | 0.905172 | 0.905172 | PASS |
| redteam injection_spam/bema_baseline | mean_confidence | 0.9605 | 0.9605 | PASS |
| redteam injection_spam/bema_baseline | overconfidence | 0.0553272 | 0.0553272 | PASS |
| redteam injection_spam/bema_baseline | ece | 0.071249 | 0.071249 | PASS |
| redteam injection_spam/bema_baseline | accuracy_ci95 (Wilson) | [0.838147, 0.946223] | [0.838147, 0.946223] | PASS |
| redteam injection_spam/bema_baseline | mean_p_spam | 0.886036 | 0.886036 | PASS |
| redteam injection_spam/bema_baseline | clean_recall_same_examples | 0.896552 | 0.896552 | PASS |
| redteam evasion_spam/bema_v2 | n | 116 | 116 | PASS |
| redteam evasion_spam/bema_v2 | accuracy | 0.741379 | 0.741379 | PASS |
| redteam evasion_spam/bema_v2 | mean_confidence | 0.943274 | 0.943274 | PASS |
| redteam evasion_spam/bema_v2 | overconfidence | 0.201895 | 0.201895 | PASS |
| redteam evasion_spam/bema_v2 | ece | 0.202746 | 0.202746 | PASS |
| redteam evasion_spam/bema_v2 | accuracy_ci95 (Wilson) | [0.654865, 0.812419] | [0.654865, 0.812419] | PASS |
| redteam evasion_spam/bema_v2 | mean_p_spam | 0.734835 | 0.734835 | PASS |
| redteam evasion_spam/bema_v2 | clean_recall_same_examples | 0.956897 | 0.956897 | PASS |
| redteam evasion_spam/bema_baseline | n | 116 | 116 | PASS |
| redteam evasion_spam/bema_baseline | accuracy | 0.836207 | 0.836207 | PASS |
| redteam evasion_spam/bema_baseline | mean_confidence | 0.918212 | 0.918212 | PASS |
| redteam evasion_spam/bema_baseline | overconfidence | 0.0820054 | 0.0820054 | PASS |
| redteam evasion_spam/bema_baseline | ece | 0.102313 | 0.102313 | PASS |
| redteam evasion_spam/bema_baseline | accuracy_ci95 (Wilson) | [0.7583, 0.89256] | [0.7583, 0.89256] | PASS |
| redteam evasion_spam/bema_baseline | mean_p_spam | 0.802241 | 0.802241 | PASS |
| redteam evasion_spam/bema_baseline | clean_recall_same_examples | 0.896552 | 0.896552 | PASS |
| redteam ood_spam_on_banking/bema_v2 | n | 1,000 | 1,000 | PASS |
| redteam ood_spam_on_banking/bema_v2 | accuracy | 0.514 | 0.514 | PASS |
| redteam ood_spam_on_banking/bema_v2 | mean_confidence | 0.662041 | 0.662041 | PASS |
| redteam ood_spam_on_banking/bema_v2 | overconfidence | 0.148041 | 0.148041 | PASS |
| redteam ood_spam_on_banking/bema_v2 | ece | 0.148041 | 0.148041 | PASS |
| redteam ood_spam_on_banking/bema_v2 | accuracy_ci95 (Wilson) | [0.483028, 0.544865] | [0.483028, 0.544865] | PASS |
| redteam ood_spam_on_banking/bema_v2 | mean_p_spam | 0.490215 | 0.490215 | PASS |
| redteam ood_spam_on_banking/bema_baseline | n | 1,000 | 1,000 | PASS |
| redteam ood_spam_on_banking/bema_baseline | accuracy | 0.934 | 0.934 | PASS |
| redteam ood_spam_on_banking/bema_baseline | mean_confidence | 0.90918 | 0.90918 | PASS |
| redteam ood_spam_on_banking/bema_baseline | overconfidence | -0.0248203 | -0.0248203 | PASS |
| redteam ood_spam_on_banking/bema_baseline | ece | 0.0339362 | 0.0339362 | PASS |
| redteam ood_spam_on_banking/bema_baseline | accuracy_ci95 (Wilson) | [0.916891, 0.947788] | [0.916891, 0.947788] | PASS |
| redteam ood_spam_on_banking/bema_baseline | mean_p_spam | 0.1171 | 0.1171 | PASS |
| redteam gibberish_banking/bema_v2 | digits_symbols: n | 25 | 25 | PASS |
| redteam gibberish_banking/bema_v2 | digits_symbols: mean_confidence | 0.557471 | 0.557471 | PASS |
| redteam gibberish_banking/bema_v2 | digits_symbols: share_confidence_ge_0.9 | 0.16 | 0.16 | PASS |
| redteam gibberish_banking/bema_v2 | digits_symbols: share_confidence_ge_0.5 | 0.56 | 0.56 | PASS |
| redteam gibberish_banking/bema_v2 | random_letters: n | 25 | 25 | PASS |
| redteam gibberish_banking/bema_v2 | random_letters: mean_confidence | 0.28529 | 0.28529 | PASS |
| redteam gibberish_banking/bema_v2 | random_letters: share_confidence_ge_0.9 | 0 | 0 | PASS |
| redteam gibberish_banking/bema_v2 | random_letters: share_confidence_ge_0.5 | 0.16 | 0.16 | PASS |
| redteam gibberish_banking/bema_v2 | real_offtopic_query (CLINC150 oos_test): n | 25 | 25 | PASS |
| redteam gibberish_banking/bema_v2 | real_offtopic_query (CLINC150 oos_test): mean_confidence | 0.336275 | 0.336275 | PASS |
| redteam gibberish_banking/bema_v2 | real_offtopic_query (CLINC150 oos_test): share_confidence_ge_0.9 | 0.04 | 0.04 | PASS |
| redteam gibberish_banking/bema_v2 | real_offtopic_query (CLINC150 oos_test): share_confidence_ge_0.5 | 0.16 | 0.16 | PASS |
| redteam gibberish_banking/bema_v2 | scrambled_real_query: n | 25 | 25 | PASS |
| redteam gibberish_banking/bema_v2 | scrambled_real_query: mean_confidence | 0.294048 | 0.294048 | PASS |
| redteam gibberish_banking/bema_v2 | scrambled_real_query: share_confidence_ge_0.9 | 0 | 0 | PASS |
| redteam gibberish_banking/bema_v2 | scrambled_real_query: share_confidence_ge_0.5 | 0.08 | 0.08 | PASS |
| redteam gibberish_banking/bema_v2 | all: n | 100 | 100 | PASS |
| redteam gibberish_banking/bema_v2 | all: mean_confidence | 0.368271 | 0.368271 | PASS |
| redteam gibberish_banking/bema_v2 | all: share_confidence_ge_0.9 | 0.05 | 0.05 | PASS |
| redteam gibberish_banking/bema_v2 | all: share_confidence_ge_0.5 | 0.24 | 0.24 | PASS |
| redteam gibberish_banking/bema_baseline | digits_symbols: n | 25 | 25 | PASS |
| redteam gibberish_banking/bema_baseline | digits_symbols: mean_confidence | 0.393917 | 0.393917 | PASS |
| redteam gibberish_banking/bema_baseline | digits_symbols: share_confidence_ge_0.9 | 0.08 | 0.08 | PASS |
| redteam gibberish_banking/bema_baseline | digits_symbols: share_confidence_ge_0.5 | 0.24 | 0.24 | PASS |
| redteam gibberish_banking/bema_baseline | random_letters: n | 25 | 25 | PASS |
| redteam gibberish_banking/bema_baseline | random_letters: mean_confidence | 0.423804 | 0.423804 | PASS |
| redteam gibberish_banking/bema_baseline | random_letters: share_confidence_ge_0.9 | 0 | 0 | PASS |
| redteam gibberish_banking/bema_baseline | random_letters: share_confidence_ge_0.5 | 0.28 | 0.28 | PASS |
| redteam gibberish_banking/bema_baseline | real_offtopic_query (CLINC150 oos_test): n | 25 | 25 | PASS |
| redteam gibberish_banking/bema_baseline | real_offtopic_query (CLINC150 oos_test): mean_confidence | 0.501978 | 0.501978 | PASS |
| redteam gibberish_banking/bema_baseline | real_offtopic_query (CLINC150 oos_test): share_confidence_ge_0.9 | 0.04 | 0.04 | PASS |
| redteam gibberish_banking/bema_baseline | real_offtopic_query (CLINC150 oos_test): share_confidence_ge_0.5 | 0.44 | 0.44 | PASS |
| redteam gibberish_banking/bema_baseline | scrambled_real_query: n | 25 | 25 | PASS |
| redteam gibberish_banking/bema_baseline | scrambled_real_query: mean_confidence | 0.359737 | 0.359737 | PASS |
| redteam gibberish_banking/bema_baseline | scrambled_real_query: share_confidence_ge_0.9 | 0 | 0 | PASS |
| redteam gibberish_banking/bema_baseline | scrambled_real_query: share_confidence_ge_0.5 | 0.2 | 0.2 | PASS |
| redteam gibberish_banking/bema_baseline | all: n | 100 | 100 | PASS |
| redteam gibberish_banking/bema_baseline | all: mean_confidence | 0.419859 | 0.419859 | PASS |
| redteam gibberish_banking/bema_baseline | all: share_confidence_ge_0.9 | 0.03 | 0.03 | PASS |
| redteam gibberish_banking/bema_baseline | all: share_confidence_ge_0.5 | 0.29 | 0.29 | PASS |
| redteam gibberish_banking/zs_offshelf | digits_symbols: n | 25 | 25 | PASS |
| redteam gibberish_banking/zs_offshelf | digits_symbols: mean_confidence | 0.117432 | 0.117432 | PASS |
| redteam gibberish_banking/zs_offshelf | digits_symbols: share_confidence_ge_0.9 | 0 | 0 | PASS |
| redteam gibberish_banking/zs_offshelf | digits_symbols: share_confidence_ge_0.5 | 0 | 0 | PASS |
| redteam gibberish_banking/zs_offshelf | random_letters: n | 25 | 25 | PASS |
| redteam gibberish_banking/zs_offshelf | random_letters: mean_confidence | 0.087517 | 0.087517 | PASS |
| redteam gibberish_banking/zs_offshelf | random_letters: share_confidence_ge_0.9 | 0 | 0 | PASS |
| redteam gibberish_banking/zs_offshelf | random_letters: share_confidence_ge_0.5 | 0 | 0 | PASS |
| redteam gibberish_banking/zs_offshelf | real_offtopic_query (CLINC150 oos_test): n | 25 | 25 | PASS |
| redteam gibberish_banking/zs_offshelf | real_offtopic_query (CLINC150 oos_test): mean_confidence | 0.180216 | 0.180216 | PASS |
| redteam gibberish_banking/zs_offshelf | real_offtopic_query (CLINC150 oos_test): share_confidence_ge_0.9 | 0 | 0 | PASS |
| redteam gibberish_banking/zs_offshelf | real_offtopic_query (CLINC150 oos_test): share_confidence_ge_0.5 | 0 | 0 | PASS |
| redteam gibberish_banking/zs_offshelf | scrambled_real_query: n | 25 | 25 | PASS |
| redteam gibberish_banking/zs_offshelf | scrambled_real_query: mean_confidence | 0.0845399 | 0.0845399 | PASS |
| redteam gibberish_banking/zs_offshelf | scrambled_real_query: share_confidence_ge_0.9 | 0 | 0 | PASS |
| redteam gibberish_banking/zs_offshelf | scrambled_real_query: share_confidence_ge_0.5 | 0 | 0 | PASS |
| redteam gibberish_banking/zs_offshelf | all: n | 100 | 100 | PASS |
| redteam gibberish_banking/zs_offshelf | all: mean_confidence | 0.117426 | 0.117426 | PASS |
| redteam gibberish_banking/zs_offshelf | all: share_confidence_ge_0.9 | 0 | 0 | PASS |
| redteam gibberish_banking/zs_offshelf | all: share_confidence_ge_0.5 | 0 | 0 | PASS |
| redteam gibberish_banking/zs_from_clinc | digits_symbols: n | 25 | 25 | PASS |
| redteam gibberish_banking/zs_from_clinc | digits_symbols: mean_confidence | 0.390721 | 0.390721 | PASS |
| redteam gibberish_banking/zs_from_clinc | digits_symbols: share_confidence_ge_0.9 | 0 | 0 | PASS |
| redteam gibberish_banking/zs_from_clinc | digits_symbols: share_confidence_ge_0.5 | 0.28 | 0.28 | PASS |
| redteam gibberish_banking/zs_from_clinc | random_letters: n | 25 | 25 | PASS |
| redteam gibberish_banking/zs_from_clinc | random_letters: mean_confidence | 0.113622 | 0.113622 | PASS |
| redteam gibberish_banking/zs_from_clinc | random_letters: share_confidence_ge_0.9 | 0 | 0 | PASS |
| redteam gibberish_banking/zs_from_clinc | random_letters: share_confidence_ge_0.5 | 0 | 0 | PASS |
| redteam gibberish_banking/zs_from_clinc | real_offtopic_query (CLINC150 oos_test): n | 25 | 25 | PASS |
| redteam gibberish_banking/zs_from_clinc | real_offtopic_query (CLINC150 oos_test): mean_confidence | 0.272218 | 0.272218 | PASS |
| redteam gibberish_banking/zs_from_clinc | real_offtopic_query (CLINC150 oos_test): share_confidence_ge_0.9 | 0 | 0 | PASS |
| redteam gibberish_banking/zs_from_clinc | real_offtopic_query (CLINC150 oos_test): share_confidence_ge_0.5 | 0.08 | 0.08 | PASS |
| redteam gibberish_banking/zs_from_clinc | scrambled_real_query: n | 25 | 25 | PASS |
| redteam gibberish_banking/zs_from_clinc | scrambled_real_query: mean_confidence | 0.1186 | 0.1186 | PASS |
| redteam gibberish_banking/zs_from_clinc | scrambled_real_query: share_confidence_ge_0.9 | 0 | 0 | PASS |
| redteam gibberish_banking/zs_from_clinc | scrambled_real_query: share_confidence_ge_0.5 | 0 | 0 | PASS |
| redteam gibberish_banking/zs_from_clinc | all: n | 100 | 100 | PASS |
| redteam gibberish_banking/zs_from_clinc | all: mean_confidence | 0.22379 | 0.22379 | PASS |
| redteam gibberish_banking/zs_from_clinc | all: share_confidence_ge_0.9 | 0 | 0 | PASS |
| redteam gibberish_banking/zs_from_clinc | all: share_confidence_ge_0.5 | 0.09 | 0.09 | PASS |
| redteam order_shuffle_banking/bema_v2 | n | 1,000 | 1,000 | PASS |
| redteam order_shuffle_banking/bema_v2 | accuracy | 0.813 | 0.813 | PASS |
| redteam order_shuffle_banking/bema_v2 | mean_confidence | 0.844182 | 0.844182 | PASS |
| redteam order_shuffle_banking/bema_v2 | overconfidence | 0.0311816 | 0.0311816 | PASS |
| redteam order_shuffle_banking/bema_v2 | ece | 0.0363789 | 0.0363789 | PASS |
| redteam order_shuffle_banking/bema_v2 | accuracy_ci95 (Wilson) | [0.787652, 0.835952] | [0.787652, 0.835952] | PASS |
| redteam order_shuffle_banking/bema_v2 | clean_accuracy_same_examples | 0.813 | 0.813 | PASS |
| redteam order_shuffle_banking/zs_from_clinc | n | 1,000 | 1,000 | PASS |
| redteam order_shuffle_banking/zs_from_clinc | accuracy | 0.625 | 0.625 | PASS |
| redteam order_shuffle_banking/zs_from_clinc | mean_confidence | 0.719573 | 0.719573 | PASS |
| redteam order_shuffle_banking/zs_from_clinc | overconfidence | 0.094573 | 0.094573 | PASS |
| redteam order_shuffle_banking/zs_from_clinc | ece | 0.0970957 | 0.0970957 | PASS |
| redteam order_shuffle_banking/zs_from_clinc | accuracy_ci95 (Wilson) | [0.59457, 0.654474] | [0.59457, 0.654474] | PASS |
| redteam order_shuffle_banking/zs_from_clinc | clean_accuracy_same_examples | 0.625 | 0.625 | PASS |
| redteam order_shuffle_banking/zs_offshelf | n | 1,000 | 1,000 | PASS |
| redteam order_shuffle_banking/zs_offshelf | accuracy | 0.601 | 0.601 | PASS |
| redteam order_shuffle_banking/zs_offshelf | mean_confidence | 0.658652 | 0.658652 | PASS |
| redteam order_shuffle_banking/zs_offshelf | overconfidence | 0.0576521 | 0.0576521 | PASS |
| redteam order_shuffle_banking/zs_offshelf | ece | 0.0584485 | 0.0584485 | PASS |
| redteam order_shuffle_banking/zs_offshelf | accuracy_ci95 (Wilson) | [0.570318, 0.630909] | [0.570318, 0.630909] | PASS |
| redteam order_shuffle_banking/zs_offshelf | clean_accuracy_same_examples | 0.601 | 0.601 | PASS |
| redteam order_shuffle_banking/bema_baseline | n | 1,000 | 1,000 | PASS |
| redteam order_shuffle_banking/bema_baseline | accuracy | 0.82 | 0.82 | PASS |
| redteam order_shuffle_banking/bema_baseline | mean_confidence | 0.792465 | 0.792465 | PASS |
| redteam order_shuffle_banking/bema_baseline | overconfidence | -0.0275354 | -0.0275354 | PASS |
| redteam order_shuffle_banking/bema_baseline | ece | 0.0613443 | 0.0613443 | PASS |
| redteam order_shuffle_banking/bema_baseline | accuracy_ci95 (Wilson) | [0.794978, 0.842573] | [0.794978, 0.842573] | PASS |
| redteam order_shuffle_banking/bema_baseline | clean_accuracy_same_examples | 0.82 | 0.82 | PASS |
| redteam score_amazon_clean/bema_baseline | n | 1,517 | 1,517 | PASS |
| redteam score_amazon_clean/bema_baseline | pearson_r_expected_stars | 0.535106 | 0.535106 | PASS |
| redteam score_amazon_clean/bema_baseline | mae_stars_expected | 0.746757 | 0.746757 | PASS |
| redteam score_amazon_clean/bema_baseline | exact_star_accuracy_rounded | 0.560316 | 0.560316 | PASS |
| redteam score_amazon_clean/bema_baseline | within_one_star_rounded | 0.824654 | 0.824654 | PASS |
| redteam score_amazon_clean/bema_v2 | n | 1,517 | 1,517 | PASS |
| redteam score_amazon_clean/bema_v2 | pearson_r_expected_stars | 0.800065 | 0.800065 | PASS |
| redteam score_amazon_clean/bema_v2 | mae_stars_expected | 0.540893 | 0.540893 | PASS |
| redteam score_amazon_clean/bema_v2 | exact_star_accuracy_rounded | 0.619644 | 0.619644 | PASS |
| redteam score_amazon_clean/bema_v2 | within_one_star_rounded | 0.914305 | 0.914305 | PASS |
| redteam score_amazon_swap2/bema_v2 | n | 1,517 | 1,517 | PASS |
| redteam score_amazon_swap2/bema_v2 | pearson_r_expected_stars | 0.788383 | 0.788383 | PASS |
| redteam score_amazon_swap2/bema_v2 | mae_stars_expected | 0.560937 | 0.560937 | PASS |
| redteam score_amazon_swap2/bema_v2 | exact_star_accuracy_rounded | 0.60646 | 0.60646 | PASS |
| redteam score_amazon_swap2/bema_v2 | within_one_star_rounded | 0.90969 | 0.90969 | PASS |
| redteam score_amazon_swap2/bema_baseline | n | 1,517 | 1,517 | PASS |
| redteam score_amazon_swap2/bema_baseline | pearson_r_expected_stars | 0.517002 | 0.517002 | PASS |
| redteam score_amazon_swap2/bema_baseline | mae_stars_expected | 0.758725 | 0.758725 | PASS |
| redteam score_amazon_swap2/bema_baseline | exact_star_accuracy_rounded | 0.55768 | 0.55768 | PASS |
| redteam score_amazon_swap2/bema_baseline | within_one_star_rounded | 0.824654 | 0.824654 | PASS |
| redteam clinc150_clean/bema_clinc150 | n | 5,500 | 5,500 | PASS |
| redteam clinc150_clean/bema_clinc150 | accuracy | 0.714545 | 0.714545 | PASS |
| redteam clinc150_clean/bema_clinc150 | mean_confidence | 0.750366 | 0.750366 | PASS |
| redteam clinc150_clean/bema_clinc150 | overconfidence | 0.0358207 | 0.0358207 | PASS |
| redteam clinc150_clean/bema_clinc150 | ece | 0.0436284 | 0.0436284 | PASS |
| redteam clinc150_clean/bema_clinc150 | accuracy_ci95 (Wilson) | [0.702463, 0.726328] | [0.702463, 0.726328] | PASS |
| redteam clinc150_clean/bema_clinc150 | in_scope_accuracy | 0.838667 | 0.838667 | PASS |
| redteam clinc150_clean/bema_clinc150 | oos_recall | 0.156 | 0.156 | PASS |
| redteam clinc150_clean/bema_clinc150 | false_oos_rate_on_in_scope | 0.004 | 0.004 | PASS |
| redteam clinc150_clean/bema_clinc150 | n_in_scope | 4,500 | 4,500 | PASS |
| redteam clinc150_clean/bema_clinc150 | n_oos | 1,000 | 1,000 | PASS |
| redteam clinc150_clean/zs_from_banking | n | 5,500 | 5,500 | PASS |
| redteam clinc150_clean/zs_from_banking | accuracy | 0.566182 | 0.566182 | PASS |
| redteam clinc150_clean/zs_from_banking | mean_confidence | 0.614631 | 0.614631 | PASS |
| redteam clinc150_clean/zs_from_banking | overconfidence | 0.0484496 | 0.0484496 | PASS |
| redteam clinc150_clean/zs_from_banking | ece | 0.0484496 | 0.0484496 | PASS |
| redteam clinc150_clean/zs_from_banking | accuracy_ci95 (Wilson) | [0.553042, 0.579229] | [0.553042, 0.579229] | PASS |
| redteam clinc150_clean/zs_from_banking | in_scope_accuracy | 0.691556 | 0.691556 | PASS |
| redteam clinc150_clean/zs_from_banking | oos_recall | 0.002 | 0.002 | PASS |
| redteam clinc150_clean/zs_from_banking | false_oos_rate_on_in_scope | 0 | 0 | PASS |
| redteam clinc150_clean/zs_from_banking | n_in_scope | 4,500 | 4,500 | PASS |
| redteam clinc150_clean/zs_from_banking | n_oos | 1,000 | 1,000 | PASS |
| redteam clinc150_clean/zs_offshelf | n | 5,500 | 5,500 | PASS |
| redteam clinc150_clean/zs_offshelf | accuracy | 0.559636 | 0.559636 | PASS |
| redteam clinc150_clean/zs_offshelf | mean_confidence | 0.551073 | 0.551073 | PASS |
| redteam clinc150_clean/zs_offshelf | overconfidence | -0.00856289 | -0.00856289 | PASS |
| redteam clinc150_clean/zs_offshelf | ece | 0.0231361 | 0.0231361 | PASS |
| redteam clinc150_clean/zs_offshelf | accuracy_ci95 (Wilson) | [0.54648, 0.57271] | [0.54648, 0.57271] | PASS |
| redteam clinc150_clean/zs_offshelf | in_scope_accuracy | 0.683778 | 0.683778 | PASS |
| redteam clinc150_clean/zs_offshelf | oos_recall | 0.001 | 0.001 | PASS |
| redteam clinc150_clean/zs_offshelf | false_oos_rate_on_in_scope | 0 | 0 | PASS |
| redteam clinc150_clean/zs_offshelf | n_in_scope | 4,500 | 4,500 | PASS |
| redteam clinc150_clean/zs_offshelf | n_oos | 1,000 | 1,000 | PASS |
| redteam clinc150_clean/zs_offshelf_bge_base | n | 5,500 | 5,500 | PASS |
| redteam clinc150_clean/zs_offshelf_bge_base | accuracy | 0.587818 | 0.587818 | PASS |
| redteam clinc150_clean/zs_offshelf_bge_base | mean_confidence | 0.60806 | 0.60806 | PASS |
| redteam clinc150_clean/zs_offshelf_bge_base | overconfidence | 0.0202416 | 0.0202416 | PASS |
| redteam clinc150_clean/zs_offshelf_bge_base | ece | 0.0316007 | 0.0316007 | PASS |
| redteam clinc150_clean/zs_offshelf_bge_base | accuracy_ci95 (Wilson) | [0.574753, 0.600761] | [0.574753, 0.600761] | PASS |
| redteam clinc150_clean/zs_offshelf_bge_base | in_scope_accuracy | 0.718444 | 0.718444 | PASS |
| redteam clinc150_clean/zs_offshelf_bge_base | oos_recall | 0 | 0 | PASS |
| redteam clinc150_clean/zs_offshelf_bge_base | false_oos_rate_on_in_scope | 0 | 0 | PASS |
| redteam clinc150_clean/zs_offshelf_bge_base | n_in_scope | 4,500 | 4,500 | PASS |
| redteam clinc150_clean/zs_offshelf_bge_base | n_oos | 1,000 | 1,000 | PASS |
| redteam clinc150_swap2/bema_clinc150 | n | 5,500 | 5,500 | PASS |
| redteam clinc150_swap2/bema_clinc150 | accuracy | 0.446545 | 0.446545 | PASS |
| redteam clinc150_swap2/bema_clinc150 | mean_confidence | 0.557192 | 0.557192 | PASS |
| redteam clinc150_swap2/bema_clinc150 | overconfidence | 0.110647 | 0.110647 | PASS |
| redteam clinc150_swap2/bema_clinc150 | ece | 0.110647 | 0.110647 | PASS |
| redteam clinc150_swap2/bema_clinc150 | accuracy_ci95 (Wilson) | [0.433449, 0.459717] | [0.433449, 0.459717] | PASS |
| redteam clinc150_swap2/bema_clinc150 | in_scope_accuracy | 0.496444 | 0.496444 | PASS |
| redteam clinc150_swap2/bema_clinc150 | oos_recall | 0.222 | 0.222 | PASS |
| redteam clinc150_swap2/bema_clinc150 | false_oos_rate_on_in_scope | 0.0444444 | 0.0444444 | PASS |
| redteam clinc150_swap2/bema_clinc150 | n_in_scope | 4,500 | 4,500 | PASS |
| redteam clinc150_swap2/bema_clinc150 | n_oos | 1,000 | 1,000 | PASS |
| redteam clinc150_swap2/zs_offshelf | n | 5,500 | 5,500 | PASS |
| redteam clinc150_swap2/zs_offshelf | accuracy | 0.396909 | 0.396909 | PASS |
| redteam clinc150_swap2/zs_offshelf | mean_confidence | 0.381764 | 0.381764 | PASS |
| redteam clinc150_swap2/zs_offshelf | overconfidence | -0.0151452 | -0.0151452 | PASS |
| redteam clinc150_swap2/zs_offshelf | ece | 0.0278494 | 0.0278494 | PASS |
| redteam clinc150_swap2/zs_offshelf | accuracy_ci95 (Wilson) | [0.384055, 0.409907] | [0.384055, 0.409907] | PASS |
| redteam clinc150_swap2/zs_offshelf | in_scope_accuracy | 0.485111 | 0.485111 | PASS |
| redteam clinc150_swap2/zs_offshelf | oos_recall | 0 | 0 | PASS |
| redteam clinc150_swap2/zs_offshelf | false_oos_rate_on_in_scope | 0.000222222 | 0.000222222 | PASS |
| redteam clinc150_swap2/zs_offshelf | n_in_scope | 4,500 | 4,500 | PASS |
| redteam clinc150_swap2/zs_offshelf | n_oos | 1,000 | 1,000 | PASS |
| redteam clinc150_swap2/zs_from_banking | n | 5,500 | 5,500 | PASS |
| redteam clinc150_swap2/zs_from_banking | accuracy | 0.391273 | 0.391273 | PASS |
| redteam clinc150_swap2/zs_from_banking | mean_confidence | 0.427874 | 0.427874 | PASS |
| redteam clinc150_swap2/zs_from_banking | overconfidence | 0.0366015 | 0.0366015 | PASS |
| redteam clinc150_swap2/zs_from_banking | ece | 0.0372299 | 0.0372299 | PASS |
| redteam clinc150_swap2/zs_from_banking | accuracy_ci95 (Wilson) | [0.378455, 0.404242] | [0.378455, 0.404242] | PASS |
| redteam clinc150_swap2/zs_from_banking | in_scope_accuracy | 0.478 | 0.478 | PASS |
| redteam clinc150_swap2/zs_from_banking | oos_recall | 0.001 | 0.001 | PASS |
| redteam clinc150_swap2/zs_from_banking | false_oos_rate_on_in_scope | 0 | 0 | PASS |
| redteam clinc150_swap2/zs_from_banking | n_in_scope | 4,500 | 4,500 | PASS |
| redteam clinc150_swap2/zs_from_banking | n_oos | 1,000 | 1,000 | PASS |
| redteam clinc150_swap2/zs_offshelf_bge_base | n | 5,500 | 5,500 | PASS |
| redteam clinc150_swap2/zs_offshelf_bge_base | accuracy | 0.428727 | 0.428727 | PASS |
| redteam clinc150_swap2/zs_offshelf_bge_base | mean_confidence | 0.444803 | 0.444803 | PASS |
| redteam clinc150_swap2/zs_offshelf_bge_base | overconfidence | 0.0160759 | 0.0160759 | PASS |
| redteam clinc150_swap2/zs_offshelf_bge_base | ece | 0.0206664 | 0.0206664 | PASS |
| redteam clinc150_swap2/zs_offshelf_bge_base | accuracy_ci95 (Wilson) | [0.415702, 0.441852] | [0.415702, 0.441852] | PASS |
| redteam clinc150_swap2/zs_offshelf_bge_base | in_scope_accuracy | 0.524 | 0.524 | PASS |
| redteam clinc150_swap2/zs_offshelf_bge_base | oos_recall | 0 | 0 | PASS |
| redteam clinc150_swap2/zs_offshelf_bge_base | false_oos_rate_on_in_scope | 0.000444444 | 0.000444444 | PASS |
| redteam clinc150_swap2/zs_offshelf_bge_base | n_in_scope | 4,500 | 4,500 | PASS |
| redteam clinc150_swap2/zs_offshelf_bge_base | n_oos | 1,000 | 1,000 | PASS |
| redteam injection_control_banking/bema_baseline | n | 1,000 | 1,000 | PASS |
| redteam injection_control_banking/bema_baseline | accuracy | 0.455 | 0.455 | PASS |
| redteam injection_control_banking/bema_baseline | mean_confidence | 0.617761 | 0.617761 | PASS |
| redteam injection_control_banking/bema_baseline | overconfidence | 0.162761 | 0.162761 | PASS |
| redteam injection_control_banking/bema_baseline | ece | 0.162761 | 0.162761 | PASS |
| redteam injection_control_banking/bema_baseline | accuracy_ci95 (Wilson) | [0.424367, 0.485978] | [0.424367, 0.485978] | PASS |
| redteam injection_control_banking/bema_baseline | clean_accuracy_same_examples | 0.82 | 0.82 | PASS |
| redteam injection_control_banking/zs_offshelf | n | 1,000 | 1,000 | PASS |
| redteam injection_control_banking/zs_offshelf | accuracy | 0.574 | 0.574 | PASS |
| redteam injection_control_banking/zs_offshelf | mean_confidence | 0.572933 | 0.572933 | PASS |
| redteam injection_control_banking/zs_offshelf | overconfidence | -0.0010671 | -0.0010671 | PASS |
| redteam injection_control_banking/zs_offshelf | ece | 0.0263561 | 0.0263561 | PASS |
| redteam injection_control_banking/zs_offshelf | accuracy_ci95 (Wilson) | [0.543126, 0.604308] | [0.543126, 0.604308] | PASS |
| redteam injection_control_banking/zs_offshelf | clean_accuracy_same_examples | 0.601 | 0.601 | PASS |
| redteam injection_control_banking/zs_from_clinc | n | 1,000 | 1,000 | PASS |
| redteam injection_control_banking/zs_from_clinc | accuracy | 0.556 | 0.556 | PASS |
| redteam injection_control_banking/zs_from_clinc | mean_confidence | 0.636764 | 0.636764 | PASS |
| redteam injection_control_banking/zs_from_clinc | overconfidence | 0.0807635 | 0.0807635 | PASS |
| redteam injection_control_banking/zs_from_clinc | ece | 0.0844162 | 0.0844162 | PASS |
| redteam injection_control_banking/zs_from_clinc | accuracy_ci95 (Wilson) | [0.525049, 0.586522] | [0.525049, 0.586522] | PASS |
| redteam injection_control_banking/zs_from_clinc | clean_accuracy_same_examples | 0.625 | 0.625 | PASS |
| redteam injection_control_banking/bema_v2 | n | 1,000 | 1,000 | PASS |
| redteam injection_control_banking/bema_v2 | accuracy | 0.68 | 0.68 | PASS |
| redteam injection_control_banking/bema_v2 | mean_confidence | 0.704092 | 0.704092 | PASS |
| redteam injection_control_banking/bema_v2 | overconfidence | 0.0240919 | 0.0240919 | PASS |
| redteam injection_control_banking/bema_v2 | ece | 0.0411233 | 0.0411233 | PASS |
| redteam injection_control_banking/bema_v2 | accuracy_ci95 (Wilson) | [0.650446, 0.708176] | [0.650446, 0.708176] | PASS |
| redteam injection_control_banking/bema_v2 | clean_accuracy_same_examples | 0.813 | 0.813 | PASS |
| redteam injection_prefix_banking/zs_from_clinc | n | 1,000 | 1,000 | PASS |
| redteam injection_prefix_banking/zs_from_clinc | accuracy | 0.392 | 0.392 | PASS |
| redteam injection_prefix_banking/zs_from_clinc | mean_confidence | 0.570774 | 0.570774 | PASS |
| redteam injection_prefix_banking/zs_from_clinc | overconfidence | 0.178774 | 0.178774 | PASS |
| redteam injection_prefix_banking/zs_from_clinc | ece | 0.178774 | 0.178774 | PASS |
| redteam injection_prefix_banking/zs_from_clinc | accuracy_ci95 (Wilson) | [0.36221, 0.422616] | [0.36221, 0.422616] | PASS |
| redteam injection_prefix_banking/zs_from_clinc | clean_accuracy_same_examples | 0.625 | 0.625 | PASS |
| redteam injection_prefix_banking/zs_from_clinc | followed_injected_label | 0.278 | 0.278 | PASS |
| redteam injection_prefix_banking/bema_v2 | n | 1,000 | 1,000 | PASS |
| redteam injection_prefix_banking/bema_v2 | accuracy | 0.34 | 0.34 | PASS |
| redteam injection_prefix_banking/bema_v2 | mean_confidence | 0.661874 | 0.661874 | PASS |
| redteam injection_prefix_banking/bema_v2 | overconfidence | 0.321874 | 0.321874 | PASS |
| redteam injection_prefix_banking/bema_v2 | ece | 0.326357 | 0.326357 | PASS |
| redteam injection_prefix_banking/bema_v2 | accuracy_ci95 (Wilson) | [0.311302, 0.369923] | [0.311302, 0.369923] | PASS |
| redteam injection_prefix_banking/bema_v2 | clean_accuracy_same_examples | 0.813 | 0.813 | PASS |
| redteam injection_prefix_banking/bema_v2 | followed_injected_label | 0.336 | 0.336 | PASS |
| redteam injection_prefix_banking/bema_baseline | n | 1,000 | 1,000 | PASS |
| redteam injection_prefix_banking/bema_baseline | accuracy | 0.218 | 0.218 | PASS |
| redteam injection_prefix_banking/bema_baseline | mean_confidence | 0.559746 | 0.559746 | PASS |
| redteam injection_prefix_banking/bema_baseline | overconfidence | 0.341746 | 0.341746 | PASS |
| redteam injection_prefix_banking/bema_baseline | ece | 0.341746 | 0.341746 | PASS |
| redteam injection_prefix_banking/bema_baseline | accuracy_ci95 (Wilson) | [0.193515, 0.244643] | [0.193515, 0.244643] | PASS |
| redteam injection_prefix_banking/bema_baseline | clean_accuracy_same_examples | 0.82 | 0.82 | PASS |
| redteam injection_prefix_banking/bema_baseline | followed_injected_label | 0.242 | 0.242 | PASS |
| redteam injection_prefix_banking/zs_offshelf | n | 1,000 | 1,000 | PASS |
| redteam injection_prefix_banking/zs_offshelf | accuracy | 0.387 | 0.387 | PASS |
| redteam injection_prefix_banking/zs_offshelf | mean_confidence | 0.500867 | 0.500867 | PASS |
| redteam injection_prefix_banking/zs_offshelf | overconfidence | 0.113867 | 0.113867 | PASS |
| redteam injection_prefix_banking/zs_offshelf | ece | 0.124432 | 0.124432 | PASS |
| redteam injection_prefix_banking/zs_offshelf | accuracy_ci95 (Wilson) | [0.357299, 0.417566] | [0.357299, 0.417566] | PASS |
| redteam injection_prefix_banking/zs_offshelf | clean_accuracy_same_examples | 0.601 | 0.601 | PASS |
| redteam injection_prefix_banking/zs_offshelf | followed_injected_label | 0.321 | 0.321 | PASS |
| redteam injection_casual_banking/bema_baseline | n | 1,000 | 1,000 | PASS |
| redteam injection_casual_banking/bema_baseline | accuracy | 0.439 | 0.439 | PASS |
| redteam injection_casual_banking/bema_baseline | mean_confidence | 0.628044 | 0.628044 | PASS |
| redteam injection_casual_banking/bema_baseline | overconfidence | 0.189044 | 0.189044 | PASS |
| redteam injection_casual_banking/bema_baseline | ece | 0.189044 | 0.189044 | PASS |
| redteam injection_casual_banking/bema_baseline | accuracy_ci95 (Wilson) | [0.408533, 0.469934] | [0.408533, 0.469934] | PASS |
| redteam injection_casual_banking/bema_baseline | clean_accuracy_same_examples | 0.82 | 0.82 | PASS |
| redteam injection_casual_banking/bema_baseline | followed_injected_label | 0.223 | 0.223 | PASS |
| redteam injection_casual_banking/bema_v2 | n | 1,000 | 1,000 | PASS |
| redteam injection_casual_banking/bema_v2 | accuracy | 0.508 | 0.508 | PASS |
| redteam injection_casual_banking/bema_v2 | mean_confidence | 0.734576 | 0.734576 | PASS |
| redteam injection_casual_banking/bema_v2 | overconfidence | 0.226576 | 0.226576 | PASS |
| redteam injection_casual_banking/bema_v2 | ece | 0.226576 | 0.226576 | PASS |
| redteam injection_casual_banking/bema_v2 | accuracy_ci95 (Wilson) | [0.477043, 0.538896] | [0.477043, 0.538896] | PASS |
| redteam injection_casual_banking/bema_v2 | clean_accuracy_same_examples | 0.813 | 0.813 | PASS |
| redteam injection_casual_banking/bema_v2 | followed_injected_label | 0.265 | 0.265 | PASS |
| redteam injection_casual_banking/zs_offshelf | n | 1,000 | 1,000 | PASS |
| redteam injection_casual_banking/zs_offshelf | accuracy | 0.384 | 0.384 | PASS |
| redteam injection_casual_banking/zs_offshelf | mean_confidence | 0.564311 | 0.564311 | PASS |
| redteam injection_casual_banking/zs_offshelf | overconfidence | 0.180311 | 0.180311 | PASS |
| redteam injection_casual_banking/zs_offshelf | ece | 0.180311 | 0.180311 | PASS |
| redteam injection_casual_banking/zs_offshelf | accuracy_ci95 (Wilson) | [0.354354, 0.414534] | [0.354354, 0.414534] | PASS |
| redteam injection_casual_banking/zs_offshelf | clean_accuracy_same_examples | 0.601 | 0.601 | PASS |
| redteam injection_casual_banking/zs_offshelf | followed_injected_label | 0.335 | 0.335 | PASS |
| redteam injection_casual_banking/zs_from_clinc | n | 1,000 | 1,000 | PASS |
| redteam injection_casual_banking/zs_from_clinc | accuracy | 0.438 | 0.438 | PASS |
| redteam injection_casual_banking/zs_from_clinc | mean_confidence | 0.626953 | 0.626953 | PASS |
| redteam injection_casual_banking/zs_from_clinc | overconfidence | 0.188953 | 0.188953 | PASS |
| redteam injection_casual_banking/zs_from_clinc | ece | 0.189757 | 0.189757 | PASS |
| redteam injection_casual_banking/zs_from_clinc | accuracy_ci95 (Wilson) | [0.407545, 0.46893] | [0.407545, 0.46893] | PASS |
| redteam injection_casual_banking/zs_from_clinc | clean_accuracy_same_examples | 0.625 | 0.625 | PASS |
| redteam injection_casual_banking/zs_from_clinc | followed_injected_label | 0.252 | 0.252 | PASS |
| redteam injection_defended_banking/bema_baseline | n | 1,000 | 1,000 | PASS |
| redteam injection_defended_banking/bema_baseline | accuracy | 0.351 | 0.351 | PASS |
| redteam injection_defended_banking/bema_baseline | mean_confidence | 0.581917 | 0.581917 | PASS |
| redteam injection_defended_banking/bema_baseline | overconfidence | 0.230917 | 0.230917 | PASS |
| redteam injection_defended_banking/bema_baseline | ece | 0.230917 | 0.230917 | PASS |
| redteam injection_defended_banking/bema_baseline | accuracy_ci95 (Wilson) | [0.32204, 0.381101] | [0.32204, 0.381101] | PASS |
| redteam injection_defended_banking/bema_baseline | clean_accuracy_same_examples | 0.82 | 0.82 | PASS |
| redteam injection_defended_banking/bema_baseline | followed_injected_label | 0.174 | 0.174 | PASS |
| redteam injection_defended_banking/zs_offshelf | n | 1,000 | 1,000 | PASS |
| redteam injection_defended_banking/zs_offshelf | accuracy | 0.473 | 0.473 | PASS |
| redteam injection_defended_banking/zs_offshelf | mean_confidence | 0.532621 | 0.532621 | PASS |
| redteam injection_defended_banking/zs_offshelf | overconfidence | 0.0596207 | 0.0596207 | PASS |
| redteam injection_defended_banking/zs_offshelf | ece | 0.0596207 | 0.0596207 | PASS |
| redteam injection_defended_banking/zs_offshelf | accuracy_ci95 (Wilson) | [0.442218, 0.503989] | [0.442218, 0.503989] | PASS |
| redteam injection_defended_banking/zs_offshelf | clean_accuracy_same_examples | 0.601 | 0.601 | PASS |
| redteam injection_defended_banking/zs_offshelf | followed_injected_label | 0.202 | 0.202 | PASS |
| redteam injection_defended_banking/zs_from_clinc | n | 1,000 | 1,000 | PASS |
| redteam injection_defended_banking/zs_from_clinc | accuracy | 0.456 | 0.456 | PASS |
| redteam injection_defended_banking/zs_from_clinc | mean_confidence | 0.596502 | 0.596502 | PASS |
| redteam injection_defended_banking/zs_from_clinc | overconfidence | 0.140502 | 0.140502 | PASS |
| redteam injection_defended_banking/zs_from_clinc | ece | 0.140502 | 0.140502 | PASS |
| redteam injection_defended_banking/zs_from_clinc | accuracy_ci95 (Wilson) | [0.425358, 0.486979] | [0.425358, 0.486979] | PASS |
| redteam injection_defended_banking/zs_from_clinc | clean_accuracy_same_examples | 0.625 | 0.625 | PASS |
| redteam injection_defended_banking/zs_from_clinc | followed_injected_label | 0.182 | 0.182 | PASS |
| redteam injection_defended_banking/bema_v2 | n | 1,000 | 1,000 | PASS |
| redteam injection_defended_banking/bema_v2 | accuracy | 0.469 | 0.469 | PASS |
| redteam injection_defended_banking/bema_v2 | mean_confidence | 0.688497 | 0.688497 | PASS |
| redteam injection_defended_banking/bema_v2 | overconfidence | 0.219497 | 0.219497 | PASS |
| redteam injection_defended_banking/bema_v2 | ece | 0.22318 | 0.22318 | PASS |
| redteam injection_defended_banking/bema_v2 | accuracy_ci95 (Wilson) | [0.438248, 0.49999] | [0.438248, 0.49999] | PASS |
| redteam injection_defended_banking/bema_v2 | clean_accuracy_same_examples | 0.813 | 0.813 | PASS |
| redteam injection_defended_banking/bema_v2 | followed_injected_label | 0.237 | 0.237 | PASS |
| redteam injection_oos_clinc/bema_clinc150 | n | 1,000 | 1,000 | PASS |
| redteam injection_oos_clinc/bema_clinc150 | accuracy | 0.396 | 0.396 | PASS |
| redteam injection_oos_clinc/bema_clinc150 | mean_confidence | 0.570536 | 0.570536 | PASS |
| redteam injection_oos_clinc/bema_clinc150 | overconfidence | 0.174536 | 0.174536 | PASS |
| redteam injection_oos_clinc/bema_clinc150 | ece | 0.174536 | 0.174536 | PASS |
| redteam injection_oos_clinc/bema_clinc150 | accuracy_ci95 (Wilson) | [0.366141, 0.426655] | [0.366141, 0.426655] | PASS |
| redteam injection_oos_clinc/bema_clinc150 | said_oos_with_note | 0.338 | 0.338 | PASS |
| redteam injection_oos_clinc/zs_offshelf | n | 1,000 | 1,000 | PASS |
| redteam injection_oos_clinc/zs_offshelf | accuracy | 0.633 | 0.633 | PASS |
| redteam injection_oos_clinc/zs_offshelf | mean_confidence | 0.522488 | 0.522488 | PASS |
| redteam injection_oos_clinc/zs_offshelf | overconfidence | -0.110512 | -0.110512 | PASS |
| redteam injection_oos_clinc/zs_offshelf | ece | 0.110512 | 0.110512 | PASS |
| redteam injection_oos_clinc/zs_offshelf | accuracy_ci95 (Wilson) | [0.602671, 0.662311] | [0.602671, 0.662311] | PASS |
| redteam injection_oos_clinc/zs_offshelf | said_oos_with_note | 0.03 | 0.03 | PASS |
| redteam injection_oos_clinc/zs_from_banking | n | 1,000 | 1,000 | PASS |
| redteam injection_oos_clinc/zs_from_banking | accuracy | 0.648 | 0.648 | PASS |
| redteam injection_oos_clinc/zs_from_banking | mean_confidence | 0.579575 | 0.579575 | PASS |
| redteam injection_oos_clinc/zs_from_banking | overconfidence | -0.0684246 | -0.0684246 | PASS |
| redteam injection_oos_clinc/zs_from_banking | ece | 0.0740011 | 0.0740011 | PASS |
| redteam injection_oos_clinc/zs_from_banking | accuracy_ci95 (Wilson) | [0.617884, 0.676983] | [0.617884, 0.676983] | PASS |
| redteam injection_oos_clinc/zs_from_banking | said_oos_with_note | 0.051 | 0.051 | PASS |
| redteam injection_casual_defended_banking/zs_offshelf | n | 1,000 | 1,000 | PASS |
| redteam injection_casual_defended_banking/zs_offshelf | accuracy | 0.384 | 0.384 | PASS |
| redteam injection_casual_defended_banking/zs_offshelf | mean_confidence | 0.564311 | 0.564311 | PASS |
| redteam injection_casual_defended_banking/zs_offshelf | overconfidence | 0.180311 | 0.180311 | PASS |
| redteam injection_casual_defended_banking/zs_offshelf | ece | 0.180311 | 0.180311 | PASS |
| redteam injection_casual_defended_banking/zs_offshelf | accuracy_ci95 (Wilson) | [0.354354, 0.414534] | [0.354354, 0.414534] | PASS |
| redteam injection_casual_defended_banking/zs_offshelf | clean_accuracy_same_examples | 0.601 | 0.601 | PASS |
| redteam injection_casual_defended_banking/zs_offshelf | followed_injected_label | 0.335 | 0.335 | PASS |
| redteam injection_casual_defended_banking/bema_baseline | n | 1,000 | 1,000 | PASS |
| redteam injection_casual_defended_banking/bema_baseline | accuracy | 0.439 | 0.439 | PASS |
| redteam injection_casual_defended_banking/bema_baseline | mean_confidence | 0.628044 | 0.628044 | PASS |
| redteam injection_casual_defended_banking/bema_baseline | overconfidence | 0.189044 | 0.189044 | PASS |
| redteam injection_casual_defended_banking/bema_baseline | ece | 0.189044 | 0.189044 | PASS |
| redteam injection_casual_defended_banking/bema_baseline | accuracy_ci95 (Wilson) | [0.408533, 0.469934] | [0.408533, 0.469934] | PASS |
| redteam injection_casual_defended_banking/bema_baseline | clean_accuracy_same_examples | 0.82 | 0.82 | PASS |
| redteam injection_casual_defended_banking/bema_baseline | followed_injected_label | 0.223 | 0.223 | PASS |
| redteam injection_casual_defended_banking/bema_v2 | n | 1,000 | 1,000 | PASS |
| redteam injection_casual_defended_banking/bema_v2 | accuracy | 0.508 | 0.508 | PASS |
| redteam injection_casual_defended_banking/bema_v2 | mean_confidence | 0.734576 | 0.734576 | PASS |
| redteam injection_casual_defended_banking/bema_v2 | overconfidence | 0.226576 | 0.226576 | PASS |
| redteam injection_casual_defended_banking/bema_v2 | ece | 0.226576 | 0.226576 | PASS |
| redteam injection_casual_defended_banking/bema_v2 | accuracy_ci95 (Wilson) | [0.477043, 0.538896] | [0.477043, 0.538896] | PASS |
| redteam injection_casual_defended_banking/bema_v2 | clean_accuracy_same_examples | 0.813 | 0.813 | PASS |
| redteam injection_casual_defended_banking/bema_v2 | followed_injected_label | 0.265 | 0.265 | PASS |
| redteam injection_casual_defended_banking/zs_from_clinc | n | 1,000 | 1,000 | PASS |
| redteam injection_casual_defended_banking/zs_from_clinc | accuracy | 0.438 | 0.438 | PASS |
| redteam injection_casual_defended_banking/zs_from_clinc | mean_confidence | 0.626953 | 0.626953 | PASS |
| redteam injection_casual_defended_banking/zs_from_clinc | overconfidence | 0.188953 | 0.188953 | PASS |
| redteam injection_casual_defended_banking/zs_from_clinc | ece | 0.189757 | 0.189757 | PASS |
| redteam injection_casual_defended_banking/zs_from_clinc | accuracy_ci95 (Wilson) | [0.407545, 0.46893] | [0.407545, 0.46893] | PASS |
| redteam injection_casual_defended_banking/zs_from_clinc | clean_accuracy_same_examples | 0.625 | 0.625 | PASS |
| redteam injection_casual_defended_banking/zs_from_clinc | followed_injected_label | 0.252 | 0.252 | PASS |

### Jev rows after the pass-1 fixes (63/63 PASS)

| section | metric | reported | recomputed | result |
|---|---|---|---|---|
| core banking77/clean | jev ece (exact) | 0.0730617 | 0.0730617 | PASS |
| core banking77/clean | jev accuracy_ci95 (Wilson) | [0.791497, 0.819429] | [0.791497, 0.819429] | PASS |
| core banking77/clean | ci_method | Wilson score | Wilson score | PASS |
| core banking77/swap2 | jev ece (exact) | 0.0672078 | 0.0672078 | PASS |
| core banking77/swap2 | jev accuracy_ci95 (Wilson) | [0.775525, 0.804272] | [0.775525, 0.804272] | PASS |
| core banking77/swap2 | ci_method | Wilson score | Wilson score | PASS |
| core banking77/swap4 | jev ece (exact) | 0.0662987 | 0.0662987 | PASS |
| core banking77/swap4 | jev accuracy_ci95 (Wilson) | [0.75494, 0.784647] | [0.75494, 0.784647] | PASS |
| core banking77/swap4 | ci_method | Wilson score | Wilson score | PASS |
| core banking77/keyboard2 | jev ece (exact) | 0.0680032 | 0.0680032 | PASS |
| core banking77/keyboard2 | jev accuracy_ci95 (Wilson) | [0.760247, 0.789716] | [0.760247, 0.789716] | PASS |
| core banking77/keyboard2 | ci_method | Wilson score | Wilson score | PASS |
| core banking77/delete2 | jev ece (exact) | 0.0649253 | 0.0649253 | PASS |
| core banking77/delete2 | jev accuracy_ci95 (Wilson) | [0.772201, 0.80111] | [0.772201, 0.80111] | PASS |
| core banking77/delete2 | ci_method | Wilson score | Wilson score | PASS |
| core banking77/insert2 | jev ece (exact) | 0.0689123 | 0.0689123 | PASS |
| core banking77/insert2 | jev accuracy_ci95 (Wilson) | [0.783507, 0.811854] | [0.783507, 0.811854] | PASS |
| core banking77/insert2 | ci_method | Wilson score | Wilson score | PASS |
| core banking77/mixed3 | jev ece (exact) | 0.0635682 | 0.0635682 | PASS |
| core banking77/mixed3 | jev accuracy_ci95 (Wilson) | [0.75892, 0.788449] | [0.75892, 0.788449] | PASS |
| core banking77/mixed3 | ci_method | Wilson score | Wilson score | PASS |
| core spam/clean | jev ece (exact) | 0.0657536 | 0.0657536 | PASS |
| core spam/clean | jev accuracy_ci95 (Wilson) | [0.945083, 0.971756] | [0.945083, 0.971756] | PASS |
| core spam/clean | ci_method | Wilson score | Wilson score | PASS |
| core spam/swap2 | jev ece (exact) | 0.0733612 | 0.0733612 | PASS |
| core spam/swap2 | jev accuracy_ci95 (Wilson) | [0.939592, 0.967722] | [0.939592, 0.967722] | PASS |
| core spam/swap2 | ci_method | Wilson score | Wilson score | PASS |
| core spam/keyboard2 | jev ece (exact) | 0.0745574 | 0.0745574 | PASS |
| core spam/keyboard2 | jev accuracy_ci95 (Wilson) | [0.938226, 0.966707] | [0.938226, 0.966707] | PASS |
| core spam/keyboard2 | ci_method | Wilson score | Wilson score | PASS |
| core spam/mixed3 | jev ece (exact) | 0.0750478 | 0.0750478 | PASS |
| core spam/mixed3 | jev accuracy_ci95 (Wilson) | [0.936863, 0.965688] | [0.936863, 0.965688] | PASS |
| core spam/mixed3 | ci_method | Wilson score | Wilson score | PASS |
| redteam injection_banking | jev ece (exact) | 0.23478 | 0.23478 | PASS |
| redteam injection_banking | jev accuracy_ci95 (Wilson) | [0.443211, 0.504988] | [0.443211, 0.504988] | PASS |
| redteam injection_prefix_banking | jev ece (exact) | 0.09943 | 0.09943 | PASS |
| redteam injection_prefix_banking | jev accuracy_ci95 (Wilson) | [0.589511, 0.649571] | [0.589511, 0.649571] | PASS |
| redteam injection_casual_banking | jev ece (exact) | 0.40955 | 0.40955 | PASS |
| redteam injection_casual_banking | jev accuracy_ci95 (Wilson) | [0.297661, 0.355671] | [0.297661, 0.355671] | PASS |
| redteam injection_defended_banking | jev ece (exact) | 0.07419 | 0.07419 | PASS |
| redteam injection_defended_banking | jev accuracy_ci95 (Wilson) | [0.751205, 0.802667] | [0.751205, 0.802667] | PASS |
| redteam injection_casual_defended_banking | jev ece (exact) | 0.12657 | 0.12657 | PASS |
| redteam injection_casual_defended_banking | jev accuracy_ci95 (Wilson) | [0.599632, 0.659373] | [0.599632, 0.659373] | PASS |
| redteam injection_control_banking | jev ece (exact) | 0.07814 | 0.07814 | PASS |
| redteam injection_control_banking | jev accuracy_ci95 (Wilson) | [0.774081, 0.823623] | [0.774081, 0.823623] | PASS |
| redteam injection_spam | jev ece (exact) | 0.0767241 | 0.0767241 | PASS |
| redteam injection_spam | jev accuracy_ci95 (Wilson) | [0.91468, 0.98651] | [0.91468, 0.98651] | PASS |
| redteam evasion_spam | jev ece (exact) | 0.0474138 | 0.0474138 | PASS |
| redteam evasion_spam | jev accuracy_ci95 (Wilson) | [0.967945, 1] | [0.967945, 1] | PASS |
| redteam ood_spam_on_banking | jev ece (exact) | 0.12658 | 0.12658 | PASS |
| redteam ood_spam_on_banking | jev accuracy_ci95 (Wilson) | [0.98976, 0.998443] | [0.98976, 0.998443] | PASS |
| redteam order_shuffle_banking | jev ece (exact) | 0.07629 | 0.07629 | PASS |
| redteam order_shuffle_banking | jev accuracy_ci95 (Wilson) | [0.773039, 0.822673] | [0.773039, 0.822673] | PASS |
| redteam score_amazon_clean | jev ece_argmax (exact) | 0.0580949 | 0.0580949 | PASS |
| redteam score_amazon_swap2 | jev ece_argmax (exact) | 0.0513975 | 0.0513975 | PASS |
| redteam clinc150_clean | jev ece (exact) | 0.0170873 | 0.0170873 | PASS |
| redteam clinc150_clean | jev accuracy_ci95 (Wilson) | [0.90064, 0.915881] | [0.90064, 0.915881] | PASS |
| redteam clinc150_swap2 | jev ece (exact) | 0.0181036 | 0.0181036 | PASS |
| redteam clinc150_swap2 | jev accuracy_ci95 (Wilson) | [0.863097, 0.880747] | [0.863097, 0.880747] | PASS |
| redteam injection_oos_clinc | jev ece (exact) | 0.06138 | 0.06138 | PASS |
| redteam injection_oos_clinc | jev accuracy_ci95 (Wilson) | [0.779296, 0.82837] | [0.779296, 0.82837] | PASS |
| core spam/clean | jev spam_precision | 0.782313 | 0.782313 | PASS |
| core spam/clean | jev false_alarms_on_ham | 32 | 32 | PASS |
