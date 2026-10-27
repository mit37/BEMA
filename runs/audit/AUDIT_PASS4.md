# Audit pass 4: the five GPU-run models

I checked `runs/results_summary.json` (generated 2026-10-27T22:34:53Z), `runs/report.json` and `runs/redteam_report.json` against my own recomputation from the raw files. The models are bema_v2_full_gpu, bema_v2_full_gpu_spell, bema_v3, bema_v3_spell and zs3_from_clinc, plus zs3_from_banking on CLINC150. All code is in `runs/audit/pass4/`. `audit4.py` recomputes every metric with its own implementation and imports no repo module. `spell_consistency.py` checks the speller variants. `write_md.py` writes this file. I changed nothing outside `runs/audit/pass4/` and this file, and I made no network calls.

## Verdict

**PASS: 0 FAILs.** 849/849 reported values for the new models match my recomputation within rounding. All 97/97 new prediction files cover exactly their input set, with a matching sha256 on every row.

**One flag on the ECE definition (not counted as a FAIL):** every ECE in the summary and the reports uses **10** equal-width bins (`bench_jev.ece_exact`, `n_bins=10`). The summary does not say so. With the 15 bins you specified, 40 of 51 new-model ECE cells differ beyond rounding, by up to 0.0120. The reported numbers are correct for 10 bins, and I match all of them to 4 decimal places.

| check | result |
|---|---|
| 1. coverage + per-row sha256: 97 files (51 core, 46 red-team) | 97/97 exact |
| 2-3. recomputed vs results_summary.json / report.json / redteam_report.json | 849/849 PASS, 0 FAIL |
| ECE at 15 bins vs reported (10-bin) values | 40/51 cells differ, max 0.0120 (definition, see below) |
| values not reported at all | 3 (`injection.clean_reference` lacks the new models) |
| 4. training log: full runs, epochs and steps as claimed | yes |
| 4. anything selected on test | no evidence in the code or the log (see caveats) |
| 4. BANKING77 test texts also in training (normalised) | 5 in train, 2 in val (0 exact) |
| 4. BEMA v3 93.1% plausible | yes; at or just below published fine-tuned BERT-base |

## 1. Coverage and sha256

Every input row's `text_sha256` / `state_sha256` equals the sha256 of its text (0 mismatches). Each prediction file has exactly the input's idx/key set, with no extras, gaps or duplicates, and the hash matches on every row. Values are well-formed: preds are in range, conf is in [0, 1], and for spam `pred == (p_yes > 0.5)` and `conf == max(p, 1-p)`.

- Core files: 4 BEMA models x 11 (7 BANKING77 + 4 spam) + zs3_from_clinc x 7 = 51.
- Red-team files: 16 each for bema_v2_full_gpu and bema_v3 (every test whose qset is banking, shuffled, defended, spam or stars), 11 for zs3_from_clinc (banking qsets), and 3 for zs3_from_banking (clinc150_clean, clinc150_swap2, injection_oos_clinc). That is 46 files, with none missing and none unexpected.
- Spam label convention: label 1 is the raw CSV's `spam` on all 116 label-1 rows, and label 0 is `ham` on all 720 label-0 rows. So pred 1 = spam = label 1 is consistent. ood_spam_on_banking gold is 0 on every row, and evasion_spam gold is 1 on every row.

## 2-3. Recomputed vs reported

Cells show **recomputed** values. All of them equal the reported value after the report's rounding, and any exception would be marked. Accuracy is in %.

### BANKING77 accuracy, n = 3,080 per condition

| model | clean | swap2 | swap4 | keyboard2 | delete2 | insert2 | mixed3 |
|---|---|---|---|---|---|---|---|
| bema_v2_full_gpu | 82.9 | 69.2 | 56.4 | 69.4 | 71.0 | 73.2 | 65.2 |
| bema_v2_full_gpu_spell | 82.9 | 79.7 | 74.7 | 80.6 | 77.2 | 82.6 | 78.8 |
| bema_v3 | 93.1 | 84.6 | 74.7 | 84.1 | 84.9 | 87.3 | 80.9 |
| bema_v3_spell | 93.1 | 90.5 | 85.7 | 91.0 | 88.4 | 92.9 | 89.1 |
| zs3_from_clinc | 60.1 | 44.2 | 33.5 | 45.7 | 46.4 | 49.0 | 42.6 |

### BANKING77 ECE: reported (10 bins) / my 10-bin / my 15-bin

| model | clean | swap2 | swap4 | keyboard2 | delete2 | insert2 | mixed3 |
|---|---|---|---|---|---|---|---|
| bema_v2_full_gpu | 0.0125 / 0.0125 / 0.0170 | 0.0553 / 0.0553 / 0.0553 | 0.1004 / 0.1004 / 0.1007 | 0.0545 / 0.0545 / 0.0545 | 0.0495 / 0.0495 / 0.0495 | 0.0413 / 0.0413 / 0.0413 | 0.0722 / 0.0722 / 0.0722 |
| bema_v2_full_gpu_spell | 0.0141 / 0.0141 / 0.0171 | 0.0171 / 0.0171 / 0.0188 | 0.0398 / 0.0398 / 0.0398 | 0.0213 / 0.0213 / 0.0203 | 0.0338 / 0.0338 / 0.0338 | 0.0155 / 0.0155 / 0.0187 | 0.0215 / 0.0215 / 0.0215 |
| bema_v3 | 0.0138 / 0.0138 / 0.0147 | 0.0189 / 0.0189 / 0.0228 | 0.0484 / 0.0484 / 0.0475 | 0.0225 / 0.0225 / 0.0231 | 0.0268 / 0.0268 / 0.0295 | 0.0138 / 0.0138 / 0.0166 | 0.0257 / 0.0257 / 0.0271 |
| bema_v3_spell | 0.0136 / 0.0136 / 0.0146 | 0.0201 / 0.0201 / 0.0195 | 0.0314 / 0.0314 / 0.0316 | 0.0143 / 0.0143 / 0.0183 | 0.0226 / 0.0226 / 0.0225 | 0.0141 / 0.0141 / 0.0148 | 0.0207 / 0.0207 / 0.0225 |
| zs3_from_clinc | 0.0207 / 0.0207 / 0.0327 | 0.0408 / 0.0408 / 0.0412 | 0.0640 / 0.0640 / 0.0640 | 0.0285 / 0.0285 / 0.0275 | 0.0240 / 0.0240 / 0.0276 | 0.0332 / 0.0332 / 0.0332 | 0.0325 / 0.0325 / 0.0409 |

### Spam (n = 836; 116 spam): accuracy % and ECE reported (10 bins) / my 15-bin

| model | clean | swap2 | keyboard2 | mixed3 |
|---|---|---|---|---|
| bema_v2_full_gpu | 98.8; 0.0043 / 0.0069 | 98.8; 0.0028 / 0.0046 | 98.7; 0.0046 / 0.0064 | 98.7; 0.0054 / 0.0093 |
| bema_v2_full_gpu_spell | 98.8; 0.0043 / 0.0057 | 98.8; 0.0054 / 0.0064 | 98.7; 0.0044 / 0.0061 | 98.9; 0.0036 / 0.0050 |
| bema_v3 | 99.4; 0.0049 / 0.0076 | 99.2; 0.0068 / 0.0081 | 99.3; 0.0065 / 0.0059 | 98.9; 0.0091 / 0.0090 |
| bema_v3_spell | 99.4; 0.0068 / 0.0068 | 99.3; 0.0038 / 0.0034 | 99.4; 0.0044 / 0.0062 | 99.4; 0.0074 / 0.0085 |

On spam clean, excluding the 83 exact-text duplicates of training messages (88 after case and whitespace normalisation), I get bema_v2_full_gpu 98.7% and bema_v3 99.3% (n 753). Both match the summary. Precision, recall, F1, false alarms and missed spam in report.json also match for every condition.

### Paired accuracy difference vs Jev, clean BANKING77

Jev has 3080/3080 `status: ok` rows, and its accuracy is 80.6% (using the `choice` field, which is in `probabilities` on every row). I used a percentile bootstrap over queries with 10,000 resamples and my own seed. The summary uses 2,000 resamples with seed 0, so I allow 0.2 points on the CI endpoints.

| model | reported gap [CI] | recomputed gap [CI] | model-only right / Jev-only right |
|---|---|---|---|
| bema_v2_full_gpu | +2.3 [+0.8, +3.9] | +2.31 [+0.71, +3.90] | 356 / 285 |
| bema_v2_full_gpu_spell | +2.3 [+0.8, +3.9] | +2.34 [+0.75, +3.93] | 356 / 284 |
| bema_v3 | +12.5 [+11.2, +13.9] | +12.53 [+11.20, +13.90] | 445 / 59 |
| bema_v3_spell | +12.6 [+11.2, +13.9] | +12.56 [+11.27, +13.93] | 445 / 58 |
| zs3_from_clinc | -20.5 [-22.2, -18.9] | -20.52 [-22.21, -18.86] | 87 / 719 |

report.json's `jev_minus_<model>_accuracy` diffs match exactly, with the sign flipped. Its CIs match within 0.002.

### Injection: the 1,000-query sample and the 250-key slice

I rebuilt the sample as `sorted(random.Random(7).sample(range(3080), 1000))`. It equals the key set and `meta.idx` of `runs/redteam/inputs/injection_banking.jsonl`. The slice is `random.Random(11).sample(sample, 250)`, which is the same construction as `llm_baseline.subsample_keys` (`random.Random(11).sample(redteam_jev.sample_idx(), n)`). It covers 75 distinct gold intents, and its first sorted keys are [0, 11, 27, 47, 58, 62]. 'Followed' means pred == `meta.target`.

| test | model | acc % (1,000) | followed % (1,000) | mean conf | acc % (250) | followed % (250) |
|---|---|---|---|---|---|---|
| injection_banking | bema_v2_full_gpu | 56.5 | 17.2 | 0.691 | 52.8 | 18.8 |
| injection_banking | bema_v3 | 60.5 | 27.1 | 0.784 | 60.4 | 26.4 |
| injection_banking | zs3_from_clinc | 45.6 | 16.3 | 0.491 | 44.8 | 16.8 |
| injection_prefix_banking | bema_v2_full_gpu | 37.7 | 32.5 | 0.645 | 32.4 (not in summary) | 34.8 (not in summary) |
| injection_prefix_banking | bema_v3 | 49.1 | 37.3 | 0.770 | 49.2 (not in summary) | 36.0 (not in summary) |
| injection_prefix_banking | zs3_from_clinc | 41.1 | 21.7 | 0.471 | 41.6 (not in summary) | 23.2 (not in summary) |
| injection_casual_banking | bema_v2_full_gpu | 50.9 | 25.7 | 0.721 | 45.6 | 28.8 |
| injection_casual_banking | bema_v3 | 59.6 | 28.0 | 0.798 | 60.4 | 27.2 |
| injection_casual_banking | zs3_from_clinc | 43.9 | 17.2 | 0.489 | 44.0 | 17.2 |
| injection_control_banking | bema_v2_full_gpu | 71.5 |  | 0.730 | 69.2 |  |
| injection_control_banking | bema_v3 | 82.6 |  | 0.792 | 81.6 |  |
| injection_control_banking | zs3_from_clinc | 58.4 |  | 0.549 | 58.8 |  |

All of these match `injection` and `slice250_all_models` in the summary and `followed_injected_label` / accuracy / mean_confidence / ECE / clean_accuracy_same_examples in redteam_report.json. The summary's slice has no prefix row, so those two slice cells are mine only. The summary shows `injection_defended_banking` and `injection_casual_defended_banking` as copies of the undefended rows for these models. That copying is legitimate: the defended input text is identical, and the predictions in the separately written defended files are identical (1000/1000, max conf difference 0.0) for all three models.

Clean BANKING77 slice250: bema_v2_full_gpu 80.0, bema_v2_full_gpu_spell 80.0, bema_v3 93.2, bema_v3_spell 93.2, zs3_from_clinc 58.8. swap2: bema_v2_full_gpu 67.6, bema_v2_full_gpu_spell 76.4, bema_v3 86.4, bema_v3_spell 90.0, zs3_from_clinc 42.0. All match.

### Other red-team tests

| test | metric | bema_v2_full_gpu | bema_v3 | zs3_from_clinc |
|---|---|---|---|---|
| ood_spam_on_banking (n 1,000, gold not-spam) | accuracy % | 47.6 | 53.2 | n/a |
| evasion_spam (n 116) | accuracy % | 75.9 | 76.7 | n/a |
| injection_spam (n 116) | accuracy % | 96.6 | 98.3 | n/a |
| gibberish_banking (n 100) | mean confidence, all | 0.4181 | 0.1718 | 0.1256 |
| gibberish_banking | mean conf, digits_symbols | 0.8176 | 0.1896 | 0.0936 |
| score_amazon_clean (n 1,517) | Pearson r, expected stars vs gold | 0.8209 | 0.8976 | n/a |
| score_amazon_swap2 | Pearson r | 0.8078 | 0.8882 | n/a |
| score_amazon_clean | MAE stars | 0.5158 | 0.3608 | n/a |
| order_shuffle_banking (n 1,000) | accuracy % | 83.0 | 92.8 | 59.6 |

Every cell matches redteam_report.json and the copy in the summary's `redteam_other` at full precision. The checks also cover mean_p_spam, clean_recall_same_examples, the per-kind gibberish confidences, share >= 0.5 and >= 0.9, exact and within-one-star rates, and the ECEs. Amazon gold, `round(4*label)+1`, matches the pickle on all 1,517 rows.

### CLINC150, zs3_from_banking

| test | accuracy % | in-scope acc % | OOS recall % | false-OOS on in-scope % | said OOS % | ECE (10-bin) |
|---|---|---|---|---|---|---|
| clinc150_clean | 55.7 | 68.0 | 0.2 | 0.1 | 0.1 | 0.0438 |
| clinc150_swap2 | 39.7 | 48.4 | 0.2 | 0.2 | 0.2 | 0.0471 |
| injection_oos_clinc | 39.6 | 39.6 | n/a (no OOS gold) | 42.4 | 42.4 | 0.1151 |

All values match. On injection_oos_clinc, zs3_from_banking and bema_clinc150 both score exactly 39.6%. I checked whether one file was a copy of the other. It is not: they agree on only 397/1000 predictions, so the tie is a coincidence.

## 4. Sanity checks

### Training log (`runs/gpu/gpu_run.log`)

| stage | claimed | in the log |
|---|---|---|
| BEMA v2 full (MiniLM-L6) | 3 epochs, 266 steps/epoch | epochs 1-3, each logging steps 0..250 of /266; val choice acc 0.724, 0.835, 0.845; temperatures noul 1.071, choice 0.516 |
| BEMA v3 (bge-base) | 4 epochs, 266 steps/epoch | epochs 1-4, each 0..250 of /266; val choice acc 0.840, 0.907, 0.924, 0.927; temperatures noul 1.307, choice 0.742 |
| zs3_from_clinc | 2 epochs | 2 x 471 steps. 471 = floor((15,000 train + 100 oos_train) / 32) |
| zs3_from_banking | 2 epochs | 2 x 265 steps. 265 = floor(8,503 / 32) |

- **Not a `--quick` run.** Quick mode sets `V2_MAX_STEPS=5, V2_EPOCHS=1` and `ZS_MAX_STEPS=5`. The log shows the full step counts. 266 = ceil(8,503 / 32), the BANKING77 training size. Every stage exits 0, and the input check passed ('All 32 input files match the cloud run exactly').
- The trainable parameter counts (10.8M of 22.7M; 85.7M of 109.5M) are what V2_FREEZE=0 gives with the token and position embeddings frozen. So 'all layers' means all transformer layers. The embeddings stay frozen, as in `bema_v2.py`.
- The speller-changed counts in the log (37, 2986, 3060, 2989, 2631, 3041, 3021 / 214, 775, 805, 817) equal what I count from `runs/bema/spell_texts`. On every text the speller left unchanged, each `_spell` file has the same prediction as its base model in all 22 files, with conf within 1.2e-6. So both come from the same checkpoint.
- The red-team files that hold clean BANKING77 text (order_shuffle, latency_sequential, repeat_banking_spread) reproduce the core clean predictions 100% for all three models, with conf within 1.6e-6. That is GPU batch-padding noise. So the core and red-team files come from the same checkpoint.
- Cross-check vs the cloud run: the cloud `bema_v2_full` (runs/bema_v2_full.log) was killed in epoch 3 (EXIT 143). Its epochs 1-2 track the GPU run closely: loss 3.6367 / 1.6794 vs 3.6379 / 1.6758, and val choice acc 0.709 / 0.835 vs 0.724 / 0.835. That is consistent with the same code and seed on different hardware. There are no cloud test predictions for bema_v2_full to compare.

### Selection on the test set

I found none in the code paths that produced these files. `bema_v2.py train` uses the train split for gradients. It picks the best epoch by a validation score, which here was the last epoch for both models, and fits the temperatures on the clean validation split. `bema_zeroshot.py` trains on the source dataset's train split and fits temperatures on the source validation split, as the log says: 'clinc validation split' for zs3_from_clinc, 'banking validation split' for zs3_from_banking. The speller's dictionaries come from training text only. `gpu_run.py` runs one fixed configuration per model, with no sweep.

There are two things I cannot rule out from the repo:
1. The hyperparameters (bge-base, CLS pooling for zero-shot, learning rates 5e-5 / 3e-5, 3 / 4 epochs) were chosen after the cloud runs' test results were visible.
2. `gpu_run.py` truncates the log on every start (`open(LOG, 'w')`), so earlier attempts on the laptop would leave no trace.

### Train/test overlap (BANKING77)

`prepare_multitask.py` uses the official BANKING77 files. The test split is `banking_test.csv` (3,080 rows, identical to the frozen inputs: True). The training data is `banking_train.csv` (10,003 rows), shuffled with seed 42 and split into 15% val (1,500) and 8,503 train. My rebuilt train + val equals the raw train file as a multiset.

- Exact text matches, test in train: 0.
- After lower-casing, stripping and collapsing whitespace: 5 test texts are in train and 2 in val (7 in either). All 5 train matches carry the same label. Examples: `How do I unblock my PIN?`; `What businesses accept this card?`; `Where can I use my card?`; `At which ATMs can I use this card?`; `I don't live in the UK.  Can I still get a card?`.
- Impact is negligible. bema_v3 scores 93.1% without those 5 rows (it gets all of them right), bema_v2_full_gpu 82.9%, and Jev 80.6%.
- Zero-shot cross-dataset overlap: 2 BANKING77 test texts appear in CLINC150 train, and 0 CLINC150 test texts appear in BANKING77 train.

### Is 93.1% for BEMA v3 plausible?

Yes. I am working from memory with no web access, so treat these figures as approximate. The BANKING77 paper (Casanueva et al., 2020, 'Efficient Intent Detection with Dual Sentence Encoders') reports about 93.7% for fine-tuned BERT-base on the full 10,003-example training set. In the same table, USE+ConveRT sentence-encoder classifiers are about 93.0-93.4% and frozen BERT is about 87%. Later fine-tuned base-size encoders (RoBERTa-base, contrastive pre-finetuning variants) usually land around 93-94.5%.

bge-base-en-v1.5 is a BERT-base-sized encoder (109.5M). Here it was trained on 85% of the training set (8,503 rows), with inputs truncated to 64 tokens, typo augmentation on half the examples, and a multi-task loss, for 4 epochs. 93.1% sits right at, or slightly below, the published fine-tuned BERT-base figure. Its own validation accuracy is 92.7%, so there is no suspicious gap between validation and test. A number well above about 95% would have been a red flag. This is not.

## Concerns (none change a reported number)

1. **ECE bin count is undocumented, and the 15-bin values differ.** The summary and reports use 10 bins. At 15 bins, 40/51 new-model cells change at 4 decimal places, by up to 0.0120. The largest change is zs3_from_clinc clean, 0.0207 -> 0.0327. Some rankings flip. On clean BANKING77, bema_v2_full_gpu has the lowest ECE at 10 bins (0.0125 vs bema_v3 0.0138), but at 15 bins bema_v3 is lower (0.0147 vs 0.0170). State the bin count in the summary, and do not rank models on ECE differences of about 0.001-0.005. These are within binning noise at n = 3,080.
2. **`injection.clean_reference` omits the new models.** It lists only jev, bema_baseline and bema_v2. On the same 1,000 keys, my values are bema_v2_full_gpu 83.0%, bema_v3 92.8%, zs3_from_clinc 59.6%. These are also what redteam_report.json's clean_accuracy_same_examples shows.
3. **The GPU results can't be reproduced from the repo.** The checkpoints (`bema_v2_full_gpu.pt`, `bema_v3.pt`) and the `zs3_from_clinc/` and `zs3_from_banking/` encoders are not in the repo. I could only check the prediction files' internal consistency. That consistency is strong: hashes, same-checkpoint agreement across core, red-team and speller files, and log counts. Archive the checkpoints, or their sha256 plus the gpu_results.zip hash, if these numbers will be published.
4. **MiniLM vs bge-base confounds model size with training budget.** bema_v2_full_gpu's validation accuracy was still rising at its last epoch (0.835 -> 0.845). It got 3 epochs at 5e-5, while v3 got 4 epochs at 3e-5. Its 82.9% is probably below what a longer MiniLM-L6 fine-tune would reach. So the +10.2-point v3 gain should not be read as purely model size. The v3 temperature (choice T = 0.74 < 1) shows the typed heads are under-confident before scaling, which is consistent with short training.
5. **Small documentation drift.**
   - The `bema_zeroshot.py` docstring still says fine-tuning 'trains only the top 3 of 6 encoder layers' on 2 CPU cores. The zs3 runs used bge-base (12 layers) with ZS_FREEZE=0.
   - 'All layers' in the log titles means all transformer layers; the embeddings were frozen.
   - The gpu_run.py docstring calls bema_v2_full_gpu a cross-check of the cloud run, but no completed cloud run exists (see above).
6. **Test-set hygiene can't be fully verified.** See 'Selection on the test set' above: the hyperparameters may have been chosen after the cloud test results were visible, and the log is overwritten on each run. I found nothing in the code that touches test labels before scoring.

## Files

- `runs/audit/pass4/audit4.py`: coverage, the recomputation, the comparisons and the leakage check. Writes `audit4.json`.
- `runs/audit/pass4/spell_consistency.py`: checks that each speller variant matches its base model. Writes `spell_consistency.json`.
- `runs/audit/pass4/write_md.py`: writes this file.
