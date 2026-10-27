# Results summary (generated)

Generated 2026-10-27T23:36:38.796137+00:00 by `python3 results_summary.py` from raw files.

## banking77_typos

```json
{
 "source": "runs/jev, runs/bema, runs/llm raw files; n=3,080 BANKING77 test queries",
 "rows": {
  "clean": {
   "jev": {
    "accuracy_pct": 80.6,
    "n": 3080,
    "ece": 0.0731
   },
   "bema_baseline": {
    "accuracy_pct": 81.9,
    "n": 3080,
    "ece": 0.0399
   },
   "bema_v2": {
    "accuracy_pct": 82.4,
    "n": 3080,
    "ece": 0.025
   },
   "bema_v2_spell": {
    "accuracy_pct": 82.4,
    "n": 3080,
    "ece": 0.0245
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 82.9,
    "n": 3080,
    "ece": 0.0125
   },
   "bema_v2_full_gpu_spell": {
    "accuracy_pct": 82.9,
    "n": 3080,
    "ece": 0.0141
   },
   "bema_v3": {
    "accuracy_pct": 93.1,
    "n": 3080,
    "ece": 0.0138
   },
   "bema_v3_spell": {
    "accuracy_pct": 93.1,
    "n": 3080,
    "ece": 0.0136
   },
   "zs_offshelf": {
    "accuracy_pct": 62.2,
    "n": 3080,
    "ece": 0.0433
   },
   "zs_from_clinc": {
    "accuracy_pct": 63.1,
    "n": 3080,
    "ece": 0.0907
   },
   "zs_offshelf_bge_base": {
    "accuracy_pct": 69.5,
    "n": 3080,
    "ece": 0.0254
   },
   "zs3_from_clinc": {
    "accuracy_pct": 60.1,
    "n": 3080,
    "ece": 0.0207
   },
   "claude-haiku-5.5": {
    "accuracy_pct": 76.8,
    "n": 3080,
    "ece": 0.0857
   },
   "qwen3.8-flash": {
    "accuracy_pct": 79.4,
    "n": 3080,
    "ece": 0.1553
   },
   "claude-sonnet-5.5": {
    "accuracy_pct": 84.6,
    "n": 3080,
    "ece": 0.0676
   },
   "claude-opus-5.5": {
    "accuracy_pct": 88.8,
    "n": 3080,
    "ece": 0.093
   }
  },
  "swap2": {
   "jev": {
    "accuracy_pct": 79.0,
    "n": 3080,
    "ece": 0.0672
   },
   "bema_baseline": {
    "accuracy_pct": 50.8,
    "n": 3080,
    "ece": 0.1027
   },
   "bema_v2": {
    "accuracy_pct": 67.3,
    "n": 3080,
    "ece": 0.066
   },
   "bema_v2_spell": {
    "accuracy_pct": 79.4,
    "n": 3080,
    "ece": 0.0283
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 69.2,
    "n": 3080,
    "ece": 0.0553
   },
   "bema_v2_full_gpu_spell": {
    "accuracy_pct": 79.7,
    "n": 3080,
    "ece": 0.0171
   },
   "bema_v3": {
    "accuracy_pct": 84.6,
    "n": 3080,
    "ece": 0.0189
   },
   "bema_v3_spell": {
    "accuracy_pct": 90.5,
    "n": 3080,
    "ece": 0.0201
   },
   "zs_offshelf": {
    "accuracy_pct": 47.3,
    "n": 3080,
    "ece": 0.0287
   },
   "zs_from_clinc": {
    "accuracy_pct": 47.4,
    "n": 3080,
    "ece": 0.1147
   },
   "zs_offshelf_bge_base": {
    "accuracy_pct": 54.4,
    "n": 3080,
    "ece": 0.021
   },
   "zs3_from_clinc": {
    "accuracy_pct": 44.2,
    "n": 3080,
    "ece": 0.0408
   },
   "claude-haiku-5.5": {
    "accuracy_pct": 74.7,
    "n": 3080,
    "ece": 0.1028
   },
   "qwen3.8-flash": {
    "accuracy_pct": 77.9,
    "n": 3080,
    "ece": 0.1672
   },
   "claude-sonnet-5.5": {
    "accuracy_pct": 84.4,
    "n": 3080,
    "ece": 0.0629
   },
   "claude-opus-5.5": {
    "accuracy_pct": 88.8,
    "n": 3080,
    "ece": 0.092
   }
  },
  "swap4": {
   "jev": {
    "accuracy_pct": 77.0,
    "n": 3080,
    "ece": 0.0663
   },
   "bema_baseline": {
    "accuracy_pct": 32.7,
    "n": 3080,
    "ece": 0.2005
   },
   "bema_v2": {
    "accuracy_pct": 53.6,
    "n": 3080,
    "ece": 0.1081
   },
   "bema_v2_spell": {
    "accuracy_pct": 74.6,
    "n": 3080,
    "ece": 0.0463
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 56.4,
    "n": 3080,
    "ece": 0.1004
   },
   "bema_v2_full_gpu_spell": {
    "accuracy_pct": 74.7,
    "n": 3080,
    "ece": 0.0398
   },
   "bema_v3": {
    "accuracy_pct": 74.7,
    "n": 3080,
    "ece": 0.0484
   },
   "bema_v3_spell": {
    "accuracy_pct": 85.7,
    "n": 3080,
    "ece": 0.0314
   },
   "zs_offshelf": {
    "accuracy_pct": 36.4,
    "n": 3080,
    "ece": 0.0301
   },
   "zs_from_clinc": {
    "accuracy_pct": 35.4,
    "n": 3080,
    "ece": 0.1434
   },
   "zs3_from_clinc": {
    "accuracy_pct": 33.5,
    "n": 3080,
    "ece": 0.064
   }
  },
  "keyboard2": {
   "jev": {
    "accuracy_pct": 77.5,
    "n": 3080,
    "ece": 0.068
   },
   "bema_baseline": {
    "accuracy_pct": 53.2,
    "n": 3080,
    "ece": 0.087
   },
   "bema_v2": {
    "accuracy_pct": 69.1,
    "n": 3080,
    "ece": 0.0505
   },
   "bema_v2_spell": {
    "accuracy_pct": 80.4,
    "n": 3080,
    "ece": 0.0297
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 69.4,
    "n": 3080,
    "ece": 0.0545
   },
   "bema_v2_full_gpu_spell": {
    "accuracy_pct": 80.6,
    "n": 3080,
    "ece": 0.0213
   },
   "bema_v3": {
    "accuracy_pct": 84.1,
    "n": 3080,
    "ece": 0.0225
   },
   "bema_v3_spell": {
    "accuracy_pct": 91.0,
    "n": 3080,
    "ece": 0.0143
   },
   "zs_offshelf": {
    "accuracy_pct": 48.7,
    "n": 3080,
    "ece": 0.0329
   },
   "zs_from_clinc": {
    "accuracy_pct": 47.0,
    "n": 3080,
    "ece": 0.1278
   },
   "zs3_from_clinc": {
    "accuracy_pct": 45.7,
    "n": 3080,
    "ece": 0.0285
   }
  },
  "delete2": {
   "jev": {
    "accuracy_pct": 78.7,
    "n": 3080,
    "ece": 0.0649
   },
   "bema_baseline": {
    "accuracy_pct": 55.1,
    "n": 3080,
    "ece": 0.09
   },
   "bema_v2": {
    "accuracy_pct": 70.1,
    "n": 3080,
    "ece": 0.0525
   },
   "bema_v2_spell": {
    "accuracy_pct": 76.8,
    "n": 3080,
    "ece": 0.0444
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 71.0,
    "n": 3080,
    "ece": 0.0495
   },
   "bema_v2_full_gpu_spell": {
    "accuracy_pct": 77.2,
    "n": 3080,
    "ece": 0.0338
   },
   "bema_v3": {
    "accuracy_pct": 84.9,
    "n": 3080,
    "ece": 0.0268
   },
   "bema_v3_spell": {
    "accuracy_pct": 88.4,
    "n": 3080,
    "ece": 0.0226
   },
   "zs_offshelf": {
    "accuracy_pct": 49.3,
    "n": 3080,
    "ece": 0.037
   },
   "zs_from_clinc": {
    "accuracy_pct": 49.1,
    "n": 3080,
    "ece": 0.1209
   },
   "zs3_from_clinc": {
    "accuracy_pct": 46.4,
    "n": 3080,
    "ece": 0.024
   }
  },
  "insert2": {
   "jev": {
    "accuracy_pct": 79.8,
    "n": 3080,
    "ece": 0.0689
   },
   "bema_baseline": {
    "accuracy_pct": 58.7,
    "n": 3080,
    "ece": 0.0652
   },
   "bema_v2": {
    "accuracy_pct": 72.1,
    "n": 3080,
    "ece": 0.0484
   },
   "bema_v2_spell": {
    "accuracy_pct": 82.1,
    "n": 3080,
    "ece": 0.0248
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 73.2,
    "n": 3080,
    "ece": 0.0413
   },
   "bema_v2_full_gpu_spell": {
    "accuracy_pct": 82.6,
    "n": 3080,
    "ece": 0.0155
   },
   "bema_v3": {
    "accuracy_pct": 87.3,
    "n": 3080,
    "ece": 0.0138
   },
   "bema_v3_spell": {
    "accuracy_pct": 92.9,
    "n": 3080,
    "ece": 0.0141
   },
   "zs_offshelf": {
    "accuracy_pct": 53.1,
    "n": 3080,
    "ece": 0.0168
   },
   "zs_from_clinc": {
    "accuracy_pct": 51.9,
    "n": 3080,
    "ece": 0.1079
   },
   "zs3_from_clinc": {
    "accuracy_pct": 49.0,
    "n": 3080,
    "ece": 0.0332
   }
  },
  "mixed3": {
   "jev": {
    "accuracy_pct": 77.4,
    "n": 3080,
    "ece": 0.0636
   },
   "bema_baseline": {
    "accuracy_pct": 44.7,
    "n": 3080,
    "ece": 0.1365
   },
   "bema_v2": {
    "accuracy_pct": 64.4,
    "n": 3080,
    "ece": 0.0678
   },
   "bema_v2_spell": {
    "accuracy_pct": 78.7,
    "n": 3080,
    "ece": 0.0295
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 65.2,
    "n": 3080,
    "ece": 0.0722
   },
   "bema_v2_full_gpu_spell": {
    "accuracy_pct": 78.8,
    "n": 3080,
    "ece": 0.0215
   },
   "bema_v3": {
    "accuracy_pct": 80.9,
    "n": 3080,
    "ece": 0.0257
   },
   "bema_v3_spell": {
    "accuracy_pct": 89.1,
    "n": 3080,
    "ece": 0.0207
   },
   "zs_offshelf": {
    "accuracy_pct": 43.7,
    "n": 3080,
    "ece": 0.0319
   },
   "zs_from_clinc": {
    "accuracy_pct": 44.2,
    "n": 3080,
    "ece": 0.1234
   },
   "zs3_from_clinc": {
    "accuracy_pct": 42.6,
    "n": 3080,
    "ece": 0.0325
   },
   "claude-haiku-5.5": {
    "accuracy_pct": 72.6,
    "n": 3080,
    "ece": 0.1071
   },
   "qwen3.8-flash": {
    "accuracy_pct": 76.9,
    "n": 3080,
    "ece": 0.1738
   },
   "claude-sonnet-5.5": {
    "accuracy_pct": 83.4,
    "n": 3080,
    "ece": 0.0589
   },
   "claude-opus-5.5": {
    "accuracy_pct": 88.3,
    "n": 3080,
    "ece": 0.0981
   }
  }
 },
 "ece_definition": "expected calibration error over 10 equal-width confidence bins (bench_jev.ece_exact)"
}
```

## spam

```json
{
 "source": "n=836 SMS test messages (116 spam); majority-class baseline in 'majority'",
 "rows": {
  "clean": {
   "jev": {
    "accuracy_pct": 96.1,
    "n": 836,
    "ece": 0.0658
   },
   "bema_baseline": {
    "accuracy_pct": 97.8,
    "n": 836,
    "ece": 0.006
   },
   "bema_v2": {
    "accuracy_pct": 98.7,
    "n": 836,
    "ece": 0.0052
   },
   "bema_v2_spell": {
    "accuracy_pct": 98.7,
    "n": 836,
    "ece": 0.0071
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 98.8,
    "n": 836,
    "ece": 0.0043
   },
   "bema_v2_full_gpu_spell": {
    "accuracy_pct": 98.8,
    "n": 836,
    "ece": 0.0043
   },
   "bema_v3": {
    "accuracy_pct": 99.4,
    "n": 836,
    "ece": 0.0049
   },
   "bema_v3_spell": {
    "accuracy_pct": 99.4,
    "n": 836,
    "ece": 0.0068
   },
   "claude-haiku-5.5": {
    "accuracy_pct": 94.0,
    "n": 836,
    "ece": 0.035
   },
   "qwen3.8-flash": {
    "accuracy_pct": 79.9,
    "n": 836,
    "ece": 0.1582
   },
   "claude-sonnet-5.5": {
    "accuracy_pct": 96.8,
    "n": 836,
    "ece": 0.0397
   },
   "claude-opus-5.5": {
    "accuracy_pct": 99.6,
    "n": 836,
    "ece": 0.0323
   }
  },
  "swap2": {
   "jev": {
    "accuracy_pct": 95.6,
    "n": 836,
    "ece": 0.0734
   },
   "bema_baseline": {
    "accuracy_pct": 97.2,
    "n": 836,
    "ece": 0.0066
   },
   "bema_v2": {
    "accuracy_pct": 98.4,
    "n": 836,
    "ece": 0.0059
   },
   "bema_v2_spell": {
    "accuracy_pct": 98.6,
    "n": 836,
    "ece": 0.0068
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 98.8,
    "n": 836,
    "ece": 0.0028
   },
   "bema_v2_full_gpu_spell": {
    "accuracy_pct": 98.8,
    "n": 836,
    "ece": 0.0054
   },
   "bema_v3": {
    "accuracy_pct": 99.2,
    "n": 836,
    "ece": 0.0068
   },
   "bema_v3_spell": {
    "accuracy_pct": 99.3,
    "n": 836,
    "ece": 0.0038
   },
   "claude-haiku-5.5": {
    "accuracy_pct": 91.7,
    "n": 836,
    "ece": 0.056
   },
   "qwen3.8-flash": {
    "accuracy_pct": 67.9,
    "n": 836,
    "ece": 0.3
   },
   "claude-sonnet-5.5": {
    "accuracy_pct": 96.1,
    "n": 836,
    "ece": 0.0326
   },
   "claude-opus-5.5": {
    "accuracy_pct": 99.6,
    "n": 836,
    "ece": 0.0356
   }
  },
  "keyboard2": {
   "jev": {
    "accuracy_pct": 95.5,
    "n": 836,
    "ece": 0.0746
   },
   "bema_baseline": {
    "accuracy_pct": 97.1,
    "n": 836,
    "ece": 0.0067
   },
   "bema_v2": {
    "accuracy_pct": 98.4,
    "n": 836,
    "ece": 0.0081
   },
   "bema_v2_spell": {
    "accuracy_pct": 98.6,
    "n": 836,
    "ece": 0.0054
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 98.7,
    "n": 836,
    "ece": 0.0046
   },
   "bema_v2_full_gpu_spell": {
    "accuracy_pct": 98.7,
    "n": 836,
    "ece": 0.0044
   },
   "bema_v3": {
    "accuracy_pct": 99.3,
    "n": 836,
    "ece": 0.0065
   },
   "bema_v3_spell": {
    "accuracy_pct": 99.4,
    "n": 836,
    "ece": 0.0044
   }
  },
  "mixed3": {
   "jev": {
    "accuracy_pct": 95.3,
    "n": 836,
    "ece": 0.075
   },
   "bema_baseline": {
    "accuracy_pct": 97.6,
    "n": 836,
    "ece": 0.0075
   },
   "bema_v2": {
    "accuracy_pct": 98.3,
    "n": 836,
    "ece": 0.004
   },
   "bema_v2_spell": {
    "accuracy_pct": 98.7,
    "n": 836,
    "ece": 0.006
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 98.7,
    "n": 836,
    "ece": 0.0054
   },
   "bema_v2_full_gpu_spell": {
    "accuracy_pct": 98.9,
    "n": 836,
    "ece": 0.0036
   },
   "bema_v3": {
    "accuracy_pct": 98.9,
    "n": 836,
    "ece": 0.0091
   },
   "bema_v3_spell": {
    "accuracy_pct": 99.4,
    "n": 836,
    "ece": 0.0074
   }
  }
 },
 "majority_baseline_pct": 86.1,
 "test_messages_also_in_training": 83,
 "duplicate_rule": "exact text match with a training message (88 if case and whitespace are normalised)",
 "clean_excluding_duplicates": {
  "jev": {
   "accuracy_pct": 96.8,
   "n": 753
  },
  "bema_baseline": {
   "accuracy_pct": 97.6,
   "n": 753
  },
  "bema_v2": {
   "accuracy_pct": 98.5,
   "n": 753
  },
  "bema_v2_full_gpu": {
   "accuracy_pct": 98.7,
   "n": 753
  },
  "bema_v3": {
   "accuracy_pct": 99.3,
   "n": 753
  },
  "claude-haiku-5.5": {
   "accuracy_pct": 94.3,
   "n": 753
  },
  "qwen3.8-flash": {
   "accuracy_pct": 81.3,
   "n": 753
  },
  "claude-sonnet-5.5": {
   "accuracy_pct": 97.2,
   "n": 753
  },
  "claude-opus-5.5": {
   "accuracy_pct": 99.7,
   "n": 753
  }
 }
}
```

## banking77_paired_vs_jev

```json
{
 "source": "per-query paired bootstrap, clean BANKING77",
 "rows": {
  "bema_baseline": {
   "minus_jev_points": 1.3,
   "ci95_points": [
    -0.3,
    2.9
   ],
   "n": 3080
  },
  "bema_v2": {
   "minus_jev_points": 1.8,
   "ci95_points": [
    0.2,
    3.3
   ],
   "n": 3080
  },
  "bema_v2_spell": {
   "minus_jev_points": 1.8,
   "ci95_points": [
    0.2,
    3.3
   ],
   "n": 3080
  },
  "bema_v2_full_gpu": {
   "minus_jev_points": 2.3,
   "ci95_points": [
    0.8,
    3.9
   ],
   "n": 3080
  },
  "bema_v2_full_gpu_spell": {
   "minus_jev_points": 2.3,
   "ci95_points": [
    0.8,
    3.9
   ],
   "n": 3080
  },
  "bema_v3": {
   "minus_jev_points": 12.5,
   "ci95_points": [
    11.2,
    13.9
   ],
   "n": 3080
  },
  "bema_v3_spell": {
   "minus_jev_points": 12.6,
   "ci95_points": [
    11.2,
    13.9
   ],
   "n": 3080
  },
  "zs_offshelf": {
   "minus_jev_points": -18.4,
   "ci95_points": [
    -20.0,
    -16.8
   ],
   "n": 3080
  },
  "zs_from_clinc": {
   "minus_jev_points": -17.5,
   "ci95_points": [
    -19.1,
    -15.9
   ],
   "n": 3080
  },
  "zs_offshelf_bge_base": {
   "minus_jev_points": -11.1,
   "ci95_points": [
    -12.6,
    -9.5
   ],
   "n": 3080
  },
  "zs3_from_clinc": {
   "minus_jev_points": -20.5,
   "ci95_points": [
    -22.2,
    -18.9
   ],
   "n": 3080
  },
  "claude-haiku-5.5": {
   "minus_jev_points": -3.8,
   "ci95_points": [
    -5.0,
    -2.6
   ],
   "n": 3080
  },
  "qwen3.8-flash": {
   "minus_jev_points": -1.2,
   "ci95_points": [
    -2.3,
    -0.1
   ],
   "n": 3080
  },
  "claude-sonnet-5.5": {
   "minus_jev_points": 4.0,
   "ci95_points": [
    3.0,
    4.9
   ],
   "n": 3080
  },
  "claude-opus-5.5": {
   "minus_jev_points": 8.2,
   "ci95_points": [
    7.1,
    9.3
   ],
   "n": 3080
  }
 }
}
```

## slice250_all_models

```json
{
 "source": "random 250 of the 1,000-query red-team sample (random.Random(11)); identical keys for every model",
 "n_keys": 250,
 "rows": {
  "clean": {
   "jev": {
    "accuracy_pct": 77.2,
    "n": 250
   },
   "bema_v2": {
    "accuracy_pct": 80.0,
    "n": 250
   },
   "bema_v2_spell": {
    "accuracy_pct": 80.0,
    "n": 250
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 80.0,
    "n": 250
   },
   "bema_v2_full_gpu_spell": {
    "accuracy_pct": 80.0,
    "n": 250
   },
   "bema_v3": {
    "accuracy_pct": 93.2,
    "n": 250
   },
   "bema_v3_spell": {
    "accuracy_pct": 93.2,
    "n": 250
   },
   "zs3_from_clinc": {
    "accuracy_pct": 58.8,
    "n": 250
   },
   "claude-haiku-5.5": {
    "accuracy_pct": 72.0,
    "n": 250
   },
   "qwen3.8-flash": {
    "accuracy_pct": 79.6,
    "n": 250
   },
   "gpt-6-luna": {
    "accuracy_pct": 75.6,
    "n": 250
   },
   "gpt-6.1-terra": {
    "accuracy_pct": 81.6,
    "n": 250
   },
   "claude-sonnet-5.5": {
    "accuracy_pct": 84.0,
    "n": 250
   },
   "claude-opus-5.5": {
    "accuracy_pct": 88.4,
    "n": 250
   },
   "claude-fable-5.1": {
    "accuracy_pct": 84.0,
    "n": 250
   }
  },
  "swap2": {
   "jev": {
    "accuracy_pct": 76.4,
    "n": 250
   },
   "bema_v2": {
    "accuracy_pct": 64.4,
    "n": 250
   },
   "bema_v2_spell": {
    "accuracy_pct": 76.0,
    "n": 250
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 67.6,
    "n": 250
   },
   "bema_v2_full_gpu_spell": {
    "accuracy_pct": 76.4,
    "n": 250
   },
   "bema_v3": {
    "accuracy_pct": 86.4,
    "n": 250
   },
   "bema_v3_spell": {
    "accuracy_pct": 90.0,
    "n": 250
   },
   "zs3_from_clinc": {
    "accuracy_pct": 42.0,
    "n": 250
   },
   "claude-haiku-5.5": {
    "accuracy_pct": 69.2,
    "n": 250
   },
   "qwen3.8-flash": {
    "accuracy_pct": 77.2,
    "n": 250
   },
   "gpt-6-luna": {
    "accuracy_pct": 74.8,
    "n": 250
   },
   "gpt-6.1-terra": {
    "accuracy_pct": 81.6,
    "n": 250
   },
   "claude-sonnet-5.5": {
    "accuracy_pct": 84.4,
    "n": 250
   },
   "claude-opus-5.5": {
    "accuracy_pct": 88.0,
    "n": 250
   },
   "claude-fable-5.1": {
    "accuracy_pct": 84.0,
    "n": 250
   }
  },
  "injection_explicit": {
   "jev": {
    "accuracy_pct": 42.0,
    "n": 250,
    "followed_injection_pct": 51.6
   },
   "bema_v2": {
    "accuracy_pct": 42.8,
    "n": 250,
    "followed_injection_pct": 25.6
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 52.8,
    "n": 250,
    "followed_injection_pct": 18.8
   },
   "bema_v3": {
    "accuracy_pct": 60.4,
    "n": 250,
    "followed_injection_pct": 26.4
   },
   "zs3_from_clinc": {
    "accuracy_pct": 44.8,
    "n": 250,
    "followed_injection_pct": 16.8
   },
   "claude-haiku-5.5": {
    "accuracy_pct": 56.4,
    "n": 250,
    "followed_injection_pct": 27.2
   },
   "qwen3.8-flash": {
    "accuracy_pct": 22.8,
    "n": 250,
    "followed_injection_pct": 72.8
   },
   "gpt-6-luna": {
    "accuracy_pct": 1.2,
    "n": 250,
    "followed_injection_pct": 97.6
   },
   "gpt-6.1-terra": {
    "accuracy_pct": 81.2,
    "n": 250,
    "followed_injection_pct": 1.2
   },
   "claude-sonnet-5.5": {
    "accuracy_pct": 81.2,
    "n": 250,
    "followed_injection_pct": 2.8
   },
   "claude-opus-5.5": {
    "accuracy_pct": 80.8,
    "n": 250,
    "followed_injection_pct": 7.6
   },
   "claude-fable-5.1": {
    "accuracy_pct": 82.0,
    "n": 250,
    "followed_injection_pct": 0.4
   }
  },
  "injection_control": {
   "jev": {
    "accuracy_pct": 76.8,
    "n": 250
   },
   "bema_v2": {
    "accuracy_pct": 66.4,
    "n": 250
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 69.2,
    "n": 250
   },
   "bema_v3": {
    "accuracy_pct": 81.6,
    "n": 250
   },
   "zs3_from_clinc": {
    "accuracy_pct": 58.8,
    "n": 250
   },
   "claude-haiku-5.5": {
    "accuracy_pct": 72.8,
    "n": 250
   },
   "qwen3.8-flash": {
    "accuracy_pct": 79.2,
    "n": 250
   },
   "gpt-6-luna": {
    "accuracy_pct": 75.6,
    "n": 250
   },
   "gpt-6.1-terra": {
    "accuracy_pct": 82.0,
    "n": 250
   },
   "claude-sonnet-5.5": {
    "accuracy_pct": 82.8,
    "n": 250
   },
   "claude-opus-5.5": {
    "accuracy_pct": 88.0,
    "n": 250
   },
   "claude-fable-5.1": {
    "accuracy_pct": 84.0,
    "n": 250
   }
  },
  "injection_casual": {
   "jev": {
    "accuracy_pct": 27.6,
    "n": 250,
    "followed_injection_pct": 68.4
   },
   "bema_v2": {
    "accuracy_pct": 45.2,
    "n": 250,
    "followed_injection_pct": 28.8
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 45.6,
    "n": 250,
    "followed_injection_pct": 28.8
   },
   "bema_v3": {
    "accuracy_pct": 60.4,
    "n": 250,
    "followed_injection_pct": 27.2
   },
   "zs3_from_clinc": {
    "accuracy_pct": 44.0,
    "n": 250,
    "followed_injection_pct": 17.2
   },
   "claude-haiku-5.5": {
    "accuracy_pct": 61.6,
    "n": 250,
    "followed_injection_pct": 18.0
   },
   "qwen3.8-flash": {
    "accuracy_pct": 56.0,
    "n": 250,
    "followed_injection_pct": 33.2
   },
   "gpt-6-luna": {
    "accuracy_pct": 48.0,
    "n": 250,
    "followed_injection_pct": 40.4
   },
   "gpt-6.1-terra": {
    "accuracy_pct": 67.2,
    "n": 250,
    "followed_injection_pct": 20.0
   },
   "claude-sonnet-5.5": {
    "accuracy_pct": 62.4,
    "n": 250,
    "followed_injection_pct": 30.8
   },
   "claude-opus-5.5": {
    "accuracy_pct": 84.4,
    "n": 250,
    "followed_injection_pct": 4.0
   },
   "claude-fable-5.1": {
    "accuracy_pct": 82.8,
    "n": 250,
    "followed_injection_pct": 2.8
   }
  },
  "injection_casual_defended": {
   "jev": {
    "accuracy_pct": 57.2,
    "n": 250,
    "followed_injection_pct": 27.6
   },
   "bema_v2": {
    "accuracy_pct": 45.2,
    "n": 250,
    "followed_injection_pct": 28.8
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 45.6,
    "n": 250,
    "followed_injection_pct": 28.8
   },
   "bema_v3": {
    "accuracy_pct": 60.4,
    "n": 250,
    "followed_injection_pct": 27.2
   },
   "zs3_from_clinc": {
    "accuracy_pct": 44.0,
    "n": 250,
    "followed_injection_pct": 17.2
   },
   "claude-haiku-5.5": {
    "accuracy_pct": 68.0,
    "n": 250,
    "followed_injection_pct": 8.8
   },
   "qwen3.8-flash": {
    "accuracy_pct": 66.8,
    "n": 250,
    "followed_injection_pct": 16.0
   },
   "gpt-6-luna": {
    "accuracy_pct": 68.0,
    "n": 250,
    "followed_injection_pct": 11.6
   },
   "gpt-6.1-terra": {
    "accuracy_pct": 80.4,
    "n": 250,
    "followed_injection_pct": 1.2
   },
   "claude-sonnet-5.5": {
    "accuracy_pct": 80.4,
    "n": 250,
    "followed_injection_pct": 2.8
   },
   "claude-opus-5.5": {
    "accuracy_pct": 89.2,
    "n": 250,
    "followed_injection_pct": 0.4
   },
   "claude-fable-5.1": {
    "accuracy_pct": 82.8,
    "n": 250,
    "followed_injection_pct": 0.8
   }
  }
 },
 "note": "BEMA cannot read instructions: its injection_casual_defended answers are its injection_casual answers."
}
```

## injection

```json
{
 "source": "runs/redteam (Jev, BEMA) and runs/llm (Haiku, Qwen, Sonnet, Opus); 1,000 BANKING77 queries",
 "rows": {
  "explicit_note_end": {
   "jev": {
    "accuracy_pct": 47.4,
    "n": 1000,
    "mean_confidence": 0.707,
    "followed_injection_pct": 45.7
   },
   "bema_baseline": {
    "accuracy_pct": 35.1,
    "n": 1000,
    "mean_confidence": 0.582,
    "followed_injection_pct": 17.4
   },
   "bema_v2": {
    "accuracy_pct": 46.9,
    "n": 1000,
    "mean_confidence": 0.688,
    "followed_injection_pct": 23.7
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 56.5,
    "n": 1000,
    "mean_confidence": 0.691,
    "followed_injection_pct": 17.2
   },
   "bema_v3": {
    "accuracy_pct": 60.5,
    "n": 1000,
    "mean_confidence": 0.784,
    "followed_injection_pct": 27.1
   },
   "zs3_from_clinc": {
    "accuracy_pct": 45.6,
    "n": 1000,
    "mean_confidence": 0.491,
    "followed_injection_pct": 16.3
   },
   "claude-haiku-5.5": {
    "accuracy_pct": 59.1,
    "n": 1000,
    "mean_confidence": 0.814,
    "followed_injection_pct": 24.1
   },
   "qwen3.8-flash": {
    "accuracy_pct": 24.4,
    "n": 1000,
    "mean_confidence": 0.981,
    "followed_injection_pct": 71.5
   },
   "claude-sonnet-5.5": {
    "accuracy_pct": 82.5,
    "n": 1000,
    "mean_confidence": 0.845,
    "followed_injection_pct": 2.1
   },
   "claude-opus-5.5": {
    "accuracy_pct": 82.2,
    "n": 1000,
    "mean_confidence": 0.803,
    "followed_injection_pct": 8.2
   },
   "claude-fable-5.1": {
    "accuracy_pct": 84.2,
    "n": 1000,
    "mean_confidence": 0.874,
    "followed_injection_pct": 0.5
   }
  },
  "explicit_note_start": {
   "jev": {
    "accuracy_pct": 62.0,
    "n": 1000,
    "mean_confidence": 0.719,
    "followed_injection_pct": 27.7
   },
   "bema_baseline": {
    "accuracy_pct": 21.8,
    "n": 1000,
    "mean_confidence": 0.56,
    "followed_injection_pct": 24.2
   },
   "bema_v2": {
    "accuracy_pct": 34.0,
    "n": 1000,
    "mean_confidence": 0.662,
    "followed_injection_pct": 33.6
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 37.7,
    "n": 1000,
    "mean_confidence": 0.645,
    "followed_injection_pct": 32.5
   },
   "bema_v3": {
    "accuracy_pct": 49.1,
    "n": 1000,
    "mean_confidence": 0.77,
    "followed_injection_pct": 37.3
   },
   "zs3_from_clinc": {
    "accuracy_pct": 41.1,
    "n": 1000,
    "mean_confidence": 0.471,
    "followed_injection_pct": 21.7
   }
  },
  "casual_ps": {
   "jev": {
    "accuracy_pct": 32.6,
    "n": 1000,
    "mean_confidence": 0.735,
    "followed_injection_pct": 63.5
   },
   "bema_baseline": {
    "accuracy_pct": 43.9,
    "n": 1000,
    "mean_confidence": 0.628,
    "followed_injection_pct": 22.3
   },
   "bema_v2": {
    "accuracy_pct": 50.8,
    "n": 1000,
    "mean_confidence": 0.735,
    "followed_injection_pct": 26.5
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 50.9,
    "n": 1000,
    "mean_confidence": 0.721,
    "followed_injection_pct": 25.7
   },
   "bema_v3": {
    "accuracy_pct": 59.6,
    "n": 1000,
    "mean_confidence": 0.798,
    "followed_injection_pct": 28.0
   },
   "zs3_from_clinc": {
    "accuracy_pct": 43.9,
    "n": 1000,
    "mean_confidence": 0.489,
    "followed_injection_pct": 17.2
   },
   "claude-haiku-5.5": {
    "accuracy_pct": 66.7,
    "n": 1000,
    "mean_confidence": 0.705,
    "followed_injection_pct": 13.4
   },
   "qwen3.8-flash": {
    "accuracy_pct": 57.7,
    "n": 1000,
    "mean_confidence": 0.944,
    "followed_injection_pct": 32.6
   },
   "claude-sonnet-5.5": {
    "accuracy_pct": 63.4,
    "n": 1000,
    "mean_confidence": 0.609,
    "followed_injection_pct": 29.4
   },
   "claude-opus-5.5": {
    "accuracy_pct": 85.1,
    "n": 1000,
    "mean_confidence": 0.721,
    "followed_injection_pct": 3.9
   },
   "claude-fable-5.1": {
    "accuracy_pct": 83.7,
    "n": 1000,
    "mean_confidence": 0.824,
    "followed_injection_pct": 1.9
   }
  },
  "neutral_note_control": {
   "jev": {
    "accuracy_pct": 80.0,
    "n": 1000,
    "mean_confidence": 0.877
   },
   "bema_baseline": {
    "accuracy_pct": 45.5,
    "n": 1000,
    "mean_confidence": 0.618
   },
   "bema_v2": {
    "accuracy_pct": 68.0,
    "n": 1000,
    "mean_confidence": 0.704
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 71.5,
    "n": 1000,
    "mean_confidence": 0.73
   },
   "bema_v3": {
    "accuracy_pct": 82.6,
    "n": 1000,
    "mean_confidence": 0.792
   },
   "zs3_from_clinc": {
    "accuracy_pct": 58.4,
    "n": 1000,
    "mean_confidence": 0.549
   },
   "claude-haiku-5.5": {
    "accuracy_pct": 76.0,
    "n": 1000,
    "mean_confidence": 0.871
   },
   "qwen3.8-flash": {
    "accuracy_pct": 79.7,
    "n": 1000,
    "mean_confidence": 0.953
   },
   "claude-sonnet-5.5": {
    "accuracy_pct": 84.4,
    "n": 1000,
    "mean_confidence": 0.788
   },
   "claude-opus-5.5": {
    "accuracy_pct": 89.1,
    "n": 1000,
    "mean_confidence": 0.798
   },
   "claude-fable-5.1": {
    "accuracy_pct": 85.9,
    "n": 1000,
    "mean_confidence": 0.864
   }
  },
  "explicit_note_defended": {
   "jev": {
    "accuracy_pct": 77.8,
    "n": 1000,
    "mean_confidence": 0.848,
    "followed_injection_pct": 2.7
   },
   "bema_baseline": {
    "accuracy_pct": 35.1,
    "n": 1000,
    "mean_confidence": 0.582,
    "followed_injection_pct": 17.4
   },
   "bema_v2": {
    "accuracy_pct": 46.9,
    "n": 1000,
    "mean_confidence": 0.688,
    "followed_injection_pct": 23.7
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 56.5,
    "n": 1000,
    "mean_confidence": 0.691,
    "followed_injection_pct": 17.2
   },
   "bema_v3": {
    "accuracy_pct": 60.5,
    "n": 1000,
    "mean_confidence": 0.784,
    "followed_injection_pct": 27.1
   },
   "zs3_from_clinc": {
    "accuracy_pct": 45.6,
    "n": 1000,
    "mean_confidence": 0.491,
    "followed_injection_pct": 16.3
   }
  },
  "casual_ps_defended": {
   "jev": {
    "accuracy_pct": 63.0,
    "n": 1000,
    "mean_confidence": 0.755,
    "followed_injection_pct": 23.4
   },
   "bema_baseline": {
    "accuracy_pct": 43.9,
    "n": 1000,
    "mean_confidence": 0.628,
    "followed_injection_pct": 22.3
   },
   "bema_v2": {
    "accuracy_pct": 50.8,
    "n": 1000,
    "mean_confidence": 0.735,
    "followed_injection_pct": 26.5
   },
   "bema_v2_full_gpu": {
    "accuracy_pct": 50.9,
    "n": 1000,
    "mean_confidence": 0.721,
    "followed_injection_pct": 25.7
   },
   "bema_v3": {
    "accuracy_pct": 59.6,
    "n": 1000,
    "mean_confidence": 0.798,
    "followed_injection_pct": 28.0
   },
   "zs3_from_clinc": {
    "accuracy_pct": 43.9,
    "n": 1000,
    "mean_confidence": 0.489,
    "followed_injection_pct": 17.2
   },
   "claude-haiku-5.5": {
    "accuracy_pct": 71.3,
    "n": 1000,
    "mean_confidence": 0.787,
    "followed_injection_pct": 6.5
   },
   "qwen3.8-flash": {
    "accuracy_pct": 70.8,
    "n": 1000,
    "mean_confidence": 0.95,
    "followed_injection_pct": 13.3
   },
   "claude-sonnet-5.5": {
    "accuracy_pct": 83.3,
    "n": 1000,
    "mean_confidence": 0.806,
    "followed_injection_pct": 1.8
   },
   "claude-opus-5.5": {
    "accuracy_pct": 88.8,
    "n": 1000,
    "mean_confidence": 0.808,
    "followed_injection_pct": 0.3
   },
   "claude-fable-5.1": {
    "accuracy_pct": 84.8,
    "n": 1000,
    "mean_confidence": 0.878,
    "followed_injection_pct": 0.6
   }
  }
 },
 "note": "BEMA cannot read instructions, so its 'defended' rows are its undefended results; clean accuracy on these 1,000 queries: see clean_reference.",
 "clean_reference": {
  "jev": 80.0,
  "bema_baseline": 82.0,
  "bema_v2": 81.3,
  "bema_v2_full_gpu": 83.0,
  "bema_v3": 92.8,
  "zs3_from_clinc": 59.6
 }
}
```

## redteam_other

```json
{
 "source": "runs/redteam_report.json",
 "injection_spam": {
  "jev": {
   "n": 116,
   "n_missing": 0,
   "accuracy": 0.9655172413793104,
   "accuracy_ci95": [
    0.9146804588951801,
    0.9865101725287494
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.9093103448275862,
   "overconfidence": -0.05620689655172417,
   "ece": 0.07672413793103457,
   "ece_repo_float32_edges": 0.07672413793103437,
   "mean_p_spam": 0.8972413793103449,
   "clean_recall_same_examples": 0.9913793103448276
  },
  "bema_v3": {
   "n": 116,
   "accuracy": 0.9827586206896551,
   "accuracy_ci95": [
    0.9393090753779318,
    0.9952589868394763
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.992380451025634,
   "overconfidence": 0.009621830335978854,
   "ece": 0.009621830335978803,
   "ece_repo_float32_edges": 0.009621830335978803,
   "mean_p_spam": 0.9834654184122538,
   "clean_recall_same_examples": 0.9741379310344828
  },
  "bema_v2": {
   "n": 116,
   "accuracy": 0.9741379310344828,
   "accuracy_ci95": [
    0.9267132346240605,
    0.9911661121966082
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.985192270114504,
   "overconfidence": 0.011054339080021292,
   "ece": 0.01728507362563033,
   "ece_repo_float32_edges": 0.01728507362563033,
   "mean_p_spam": 0.9703690987722627,
   "clean_recall_same_examples": 0.9568965517241379
  },
  "bema_v2_full_gpu": {
   "n": 116,
   "accuracy": 0.9655172413793104,
   "accuracy_ci95": [
    0.9146804588951801,
    0.9865101725287494
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.9886385531261049,
   "overconfidence": 0.02312131174679455,
   "ece": 0.025786066877430958,
   "ece_repo_float32_edges": 0.025786066877430958,
   "mean_p_spam": 0.9681695333344797,
   "clean_recall_same_examples": 0.9655172413793104
  },
  "bema_baseline": {
   "n": 116,
   "accuracy": 0.9051724137931034,
   "accuracy_ci95": [
    0.8381469869898026,
    0.9462226366569506
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.9604996381134823,
   "overconfidence": 0.05532722432037884,
   "ece": 0.07124898557005259,
   "ece_repo_float32_edges": 0.07124898557005259,
   "mean_p_spam": 0.8860359538972763,
   "clean_recall_same_examples": 0.896551724137931
  },
  "jev_calls": {
   "ok": 116,
   "input_tokens": 39189,
   "median_latency_ms": 95.61345650013209,
   "model_versions": [
    "jev-1.14.0"
   ]
  }
 },
 "evasion_spam": {
  "jev": {
   "n": 116,
   "n_missing": 0,
   "accuracy": 1.0,
   "accuracy_ci95": [
    0.9679454930108872,
    1.0
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.9525862068965517,
   "overconfidence": -0.04741379310344829,
   "ece": 0.04741379310344823,
   "ece_repo_float32_edges": 0.04741379310344823,
   "mean_p_spam": 0.9525862068965517,
   "clean_recall_same_examples": 0.9913793103448276
  },
  "bema_v2": {
   "n": 116,
   "accuracy": 0.7413793103448276,
   "accuracy_ci95": [
    0.6548646736501386,
    0.8124193574585655
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.9432741901997862,
   "overconfidence": 0.20189487985495858,
   "ece": 0.20274571457813526,
   "ece_repo_float32_edges": 0.20274571457813526,
   "mean_p_spam": 0.7348351936219341,
   "clean_recall_same_examples": 0.9568965517241379
  },
  "bema_v3": {
   "n": 116,
   "accuracy": 0.7672413793103449,
   "accuracy_ci95": [
    0.6825321917307261,
    0.8348179855681963
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.9579683901934788,
   "overconfidence": 0.19072701088313393,
   "ece": 0.20216558925036726,
   "ece_repo_float32_edges": 0.20216558925036726,
   "mean_p_spam": 0.7620478383935827,
   "clean_recall_same_examples": 0.9741379310344828
  },
  "bema_baseline": {
   "n": 116,
   "accuracy": 0.8362068965517241,
   "accuracy_ci95": [
    0.7582995871792455,
    0.8925603132935922
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.9182123351713707,
   "overconfidence": 0.0820054386196466,
   "ece": 0.10231281097592977,
   "ece_repo_float32_edges": 0.10231281097592977,
   "mean_p_spam": 0.8022409898794962,
   "clean_recall_same_examples": 0.896551724137931
  },
  "bema_v2_full_gpu": {
   "n": 116,
   "accuracy": 0.7586206896551724,
   "accuracy_ci95": [
    0.6732697464950692,
    0.8273917154071138
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.9692807958043855,
   "overconfidence": 0.21066010614921316,
   "ece": 0.21225611403070646,
   "ece_repo_float32_edges": 0.21225611403070646,
   "mean_p_spam": 0.7460977573646232,
   "clean_recall_same_examples": 0.9655172413793104
  },
  "jev_calls": {
   "ok": 116,
   "input_tokens": 43298,
   "median_latency_ms": 94.03815799998938,
   "model_versions": [
    "jev-1.14.0"
   ]
  }
 },
 "ood_spam_on_banking": {
  "jev": {
   "n": 1000,
   "n_missing": 0,
   "accuracy": 0.996,
   "accuracy_ci95": [
    0.9897604436522742,
    0.9984434118703971
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.86942,
   "overconfidence": -0.12658000000000003,
   "ece": 0.12658000000000003,
   "ece_repo_float32_edges": 0.12657999999999997,
   "mean_p_spam": 0.13112000000000001
  },
  "bema_v2_full_gpu": {
   "n": 1000,
   "accuracy": 0.476,
   "accuracy_ci95": [
    0.4451969581869969,
    0.5069867262231963
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.6405855491757393,
   "overconfidence": 0.1645855491757393,
   "ece": 0.1645855491757393,
   "ece_repo_float32_edges": 0.1645855491757393,
   "mean_p_spam": 0.4940634817481041
  },
  "bema_v3": {
   "n": 1000,
   "accuracy": 0.532,
   "accuracy_ci95": [
    0.5010103114046786,
    0.5627447760483971
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.5979091175198555,
   "overconfidence": 0.06590911751985551,
   "ece": 0.0661465284228325,
   "ece_repo_float32_edges": 0.0661465284228325,
   "mean_p_spam": 0.4899849531762302
  },
  "bema_v2": {
   "n": 1000,
   "accuracy": 0.514,
   "accuracy_ci95": [
    0.48302810618302305,
    0.5448647445776975
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.6620413485169411,
   "overconfidence": 0.14804134851694106,
   "ece": 0.1480413485169411,
   "ece_repo_float32_edges": 0.1480413485169411,
   "mean_p_spam": 0.4902145897503942
  },
  "bema_baseline": {
   "n": 1000,
   "accuracy": 0.934,
   "accuracy_ci95": [
    0.9168907239451315,
    0.9477876496372061
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.9091797404885292,
   "overconfidence": -0.02482025951147082,
   "ece": 0.03393618279695514,
   "ece_repo_float32_edges": 0.03393618279695514,
   "mean_p_spam": 0.11709959695592988
  },
  "jev_calls": {
   "ok": 1000,
   "input_tokens": 284632,
   "median_latency_ms": 92.26786699991862,
   "model_versions": [
    "jev-1.14.0"
   ]
  }
 },
 "gibberish_banking": {
  "jev": {
   "digits_symbols": {
    "n": 25,
    "mean_confidence": 0.10279999999999999,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.0
   },
   "random_letters": {
    "n": 25,
    "mean_confidence": 0.0824,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.0
   },
   "real_offtopic_query (CLINC150 oos_test)": {
    "n": 25,
    "mean_confidence": 0.266,
    "share_confidence_ge_0.9": 0.04,
    "share_confidence_ge_0.5": 0.12
   },
   "scrambled_real_query": {
    "n": 25,
    "mean_confidence": 0.1276,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.04
   },
   "all": {
    "n": 100,
    "mean_confidence": 0.1447,
    "share_confidence_ge_0.9": 0.01,
    "share_confidence_ge_0.5": 0.04
   }
  },
  "bema_v2": {
   "digits_symbols": {
    "n": 25,
    "mean_confidence": 0.5574714201688766,
    "share_confidence_ge_0.9": 0.16,
    "share_confidence_ge_0.5": 0.56
   },
   "random_letters": {
    "n": 25,
    "mean_confidence": 0.2852896246314049,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.16
   },
   "real_offtopic_query (CLINC150 oos_test)": {
    "n": 25,
    "mean_confidence": 0.3362754842638969,
    "share_confidence_ge_0.9": 0.04,
    "share_confidence_ge_0.5": 0.16
   },
   "scrambled_real_query": {
    "n": 25,
    "mean_confidence": 0.2940478426218033,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.08
   },
   "all": {
    "n": 100,
    "mean_confidence": 0.36827109292149546,
    "share_confidence_ge_0.9": 0.05,
    "share_confidence_ge_0.5": 0.24
   }
  },
  "bema_baseline": {
   "digits_symbols": {
    "n": 25,
    "mean_confidence": 0.3939169818162918,
    "share_confidence_ge_0.9": 0.08,
    "share_confidence_ge_0.5": 0.24
   },
   "random_letters": {
    "n": 25,
    "mean_confidence": 0.42380372285842893,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.28
   },
   "real_offtopic_query (CLINC150 oos_test)": {
    "n": 25,
    "mean_confidence": 0.5019784867763519,
    "share_confidence_ge_0.9": 0.04,
    "share_confidence_ge_0.5": 0.44
   },
   "scrambled_real_query": {
    "n": 25,
    "mean_confidence": 0.3597374075651169,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.2
   },
   "all": {
    "n": 100,
    "mean_confidence": 0.41985914975404737,
    "share_confidence_ge_0.9": 0.03,
    "share_confidence_ge_0.5": 0.29
   }
  },
  "zs_offshelf": {
   "digits_symbols": {
    "n": 25,
    "mean_confidence": 0.11743196770548821,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.0
   },
   "random_letters": {
    "n": 25,
    "mean_confidence": 0.08751699417829513,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.0
   },
   "real_offtopic_query (CLINC150 oos_test)": {
    "n": 25,
    "mean_confidence": 0.1802163428068161,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.0
   },
   "scrambled_real_query": {
    "n": 25,
    "mean_confidence": 0.08453986495733261,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.0
   },
   "all": {
    "n": 100,
    "mean_confidence": 0.11742629241198302,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.0
   }
  },
  "bema_v3": {
   "digits_symbols": {
    "n": 25,
    "mean_confidence": 0.1896394184231758,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.08
   },
   "random_letters": {
    "n": 25,
    "mean_confidence": 0.09734889939427376,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.0
   },
   "real_offtopic_query (CLINC150 oos_test)": {
    "n": 25,
    "mean_confidence": 0.23719175279140473,
    "share_confidence_ge_0.9": 0.04,
    "share_confidence_ge_0.5": 0.08
   },
   "scrambled_real_query": {
    "n": 25,
    "mean_confidence": 0.16320323169231415,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.04
   },
   "all": {
    "n": 100,
    "mean_confidence": 0.1718458255752921,
    "share_confidence_ge_0.9": 0.01,
    "share_confidence_ge_0.5": 0.05
   }
  },
  "bema_v2_full_gpu": {
   "digits_symbols": {
    "n": 25,
    "mean_confidence": 0.8175577008724213,
    "share_confidence_ge_0.9": 0.28,
    "share_confidence_ge_0.5": 0.96
   },
   "random_letters": {
    "n": 25,
    "mean_confidence": 0.2846230459213257,
    "share_confidence_ge_0.9": 0.04,
    "share_confidence_ge_0.5": 0.04
   },
   "real_offtopic_query (CLINC150 oos_test)": {
    "n": 25,
    "mean_confidence": 0.31878347575664523,
    "share_confidence_ge_0.9": 0.04,
    "share_confidence_ge_0.5": 0.16
   },
   "scrambled_real_query": {
    "n": 25,
    "mean_confidence": 0.2515855610370636,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.08
   },
   "all": {
    "n": 100,
    "mean_confidence": 0.4181374458968639,
    "share_confidence_ge_0.9": 0.09,
    "share_confidence_ge_0.5": 0.31
   }
  },
  "zs3_from_clinc": {
   "digits_symbols": {
    "n": 25,
    "mean_confidence": 0.0936475795507431,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.0
   },
   "random_letters": {
    "n": 25,
    "mean_confidence": 0.1132769252359867,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.0
   },
   "real_offtopic_query (CLINC150 oos_test)": {
    "n": 25,
    "mean_confidence": 0.1636958810687065,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.0
   },
   "scrambled_real_query": {
    "n": 25,
    "mean_confidence": 0.13168753653764725,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.0
   },
   "all": {
    "n": 100,
    "mean_confidence": 0.12557698059827088,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.0
   }
  },
  "zs_from_clinc": {
   "digits_symbols": {
    "n": 25,
    "mean_confidence": 0.3907208237051964,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.28
   },
   "random_letters": {
    "n": 25,
    "mean_confidence": 0.11362244471907616,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.0
   },
   "real_offtopic_query (CLINC150 oos_test)": {
    "n": 25,
    "mean_confidence": 0.272217937707901,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.08
   },
   "scrambled_real_query": {
    "n": 25,
    "mean_confidence": 0.11859980508685113,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.0
   },
   "all": {
    "n": 100,
    "mean_confidence": 0.22379025280475617,
    "share_confidence_ge_0.9": 0.0,
    "share_confidence_ge_0.5": 0.09
   }
  },
  "note": "No correct answer exists for these inputs (the 77-intent question has no 'none' option), so a calibrated model's confidence should be low; 1/77 = 0.013 is a uniform guess.",
  "jev_calls": {
   "ok": 100,
   "input_tokens": 169084,
   "median_latency_ms": 104.35317899998608,
   "model_versions": [
    "jev-1.14.0"
   ]
  }
 },
 "order_shuffle_banking": {
  "jev": {
   "n": 1000,
   "n_missing": 0,
   "accuracy": 0.799,
   "accuracy_ci95": [
    0.7730388132545092,
    0.8226727851351658
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.8734299999999999,
   "overconfidence": 0.07442999999999989,
   "ece": 0.07628999999999993,
   "ece_repo_float32_edges": 0.07628999999999998,
   "clean_accuracy_same_examples": 0.8,
   "same_answer_as_original_order": 0.95,
   "mean_abs_change_in_p_gold": 0.032380000000000006
  },
  "bema_v2": {
   "n": 1000,
   "accuracy": 0.813,
   "accuracy_ci95": [
    0.7876522640280523,
    0.835952185122343
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.8441815786063671,
   "overconfidence": 0.031181578606367166,
   "ece": 0.036378882139921215,
   "ece_repo_float32_edges": 0.036378882139921215,
   "clean_accuracy_same_examples": 0.813,
   "note": "BEMA has a fixed output layer, so option order cannot affect it"
  },
  "zs3_from_clinc": {
   "n": 1000,
   "accuracy": 0.596,
   "accuracy_ci95": [
    0.5652754707472443,
    0.6259897916119823
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.5967255310937762,
   "overconfidence": 0.000725531093776266,
   "ece": 0.023855041213333594,
   "ece_repo_float32_edges": 0.023855041213333598,
   "clean_accuracy_same_examples": 0.596,
   "note": "BEMA has a fixed output layer, so option order cannot affect it"
  },
  "zs_from_clinc": {
   "n": 1000,
   "accuracy": 0.625,
   "accuracy_ci95": [
    0.594569604368661,
    0.6544737059949155
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.7195729937106371,
   "overconfidence": 0.0945729937106371,
   "ece": 0.0970957229882479,
   "ece_repo_float32_edges": 0.0970957229882479,
   "clean_accuracy_same_examples": 0.625,
   "note": "BEMA has a fixed output layer, so option order cannot affect it"
  },
  "zs_offshelf": {
   "n": 1000,
   "accuracy": 0.601,
   "accuracy_ci95": [
    0.5703182459971236,
    0.6309087487766462
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.6586520772352815,
   "overconfidence": 0.05765207723528154,
   "ece": 0.05844845492392777,
   "ece_repo_float32_edges": 0.058448454923927765,
   "clean_accuracy_same_examples": 0.601,
   "note": "BEMA has a fixed output layer, so option order cannot affect it"
  },
  "bema_v3": {
   "n": 1000,
   "accuracy": 0.928,
   "accuracy_ci95": [
    0.9102882162639245,
    0.9424360784209613
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.9354059181809425,
   "overconfidence": 0.0074059181809424945,
   "ece": 0.02228577798604966,
   "ece_repo_float32_edges": 0.02228577798604966,
   "clean_accuracy_same_examples": 0.928,
   "note": "BEMA has a fixed output layer, so option order cannot affect it"
  },
  "bema_v2_full_gpu": {
   "n": 1000,
   "accuracy": 0.83,
   "accuracy_ci95": [
    0.8054659348348435,
    0.8520084045249984
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.8323595410734415,
   "overconfidence": 0.0023595410734414957,
   "ece": 0.019787779137492193,
   "ece_repo_float32_edges": 0.019787779137492193,
   "clean_accuracy_same_examples": 0.83,
   "note": "BEMA has a fixed output layer, so option order cannot affect it"
  },
  "bema_baseline": {
   "n": 1000,
   "accuracy": 0.82,
   "accuracy_ci95": [
    0.7949777668912537,
    0.842573107639502
   ],
   "ci_method": "Wilson score",
   "mean_confidence": 0.7924646461308003,
   "overconfidence": -0.027535353869199697,
   "ece": 0.0613443310856819,
   "ece_repo_float32_edges": 0.061344331085681894,
   "clean_accuracy_same_examples": 0.82,
   "note": "BEMA has a fixed output layer, so option order cannot affect it"
  },
  "jev_calls": {
   "ok": 1000,
   "input_tokens": 1689632,
   "median_latency_ms": 103.10792249993028,
   "model_versions": [
    "jev-1.14.0"
   ]
  }
 },
 "question_count": {
  "jev": {
   "1": {
    "calls": 60,
    "median_latency_ms": 96.19828150005105,
    "p90_latency_ms": 129.05930000010812,
    "mean_input_tokens": 302.3333333333333,
    "all_questions_answered": true
   },
   "2": {
    "calls": 60,
    "median_latency_ms": 93.03473900001791,
    "p90_latency_ms": 123.67354899993188,
    "mean_input_tokens": 320.3333333333333,
    "all_questions_answered": true
   },
   "4": {
    "calls": 60,
    "median_latency_ms": 99.04904500001521,
    "p90_latency_ms": 146.8861119999474,
    "mean_input_tokens": 353.3333333333333,
    "all_questions_answered": true
   },
   "8": {
    "calls": 60,
    "median_latency_ms": 94.9717055000292,
    "p90_latency_ms": 134.82437400011804,
    "mean_input_tokens": 410.3333333333333,
    "all_questions_answered": true
   },
   "16": {
    "calls": 60,
    "median_latency_ms": 104.73331850005252,
    "p90_latency_ms": 134.19534900003782,
    "mean_input_tokens": 533.3333333333334,
    "all_questions_answered": true
   },
   "latency_ratio_16_vs_1": 1.088723383275791,
   "median_paired_ratio_16_vs_1_same_message": 1.11871193178987,
   "scope_note": "60 short SMS texts, yes/no questions only, one call at a time; input tokens rise with the number of questions"
  },
  "jev_calls": {
   "ok": 300,
   "input_tokens": 115180,
   "median_latency_ms": 97.37181200011946,
   "model_versions": [
    "jev-1.14.0"
   ]
  }
 },
 "score_amazon_clean": {
  "jev": {
   "n": 1517,
   "n_missing": 0,
   "pearson_r_expected_stars": 0.9204438569786791,
   "mae_stars_expected": 0.3542579386481825,
   "exact_star_accuracy_argmax": 0.7528015820698748,
   "within_one_star_argmax": 0.9690177982860909,
   "ece_argmax": 0.058094924192485145
  },
  "bema_v2_full_gpu": {
   "n": 1517,
   "pearson_r_expected_stars": 0.8208694647691138,
   "mae_stars_expected": 0.5157737260047118,
   "exact_star_accuracy_rounded": 0.6381015161502966,
   "within_one_star_rounded": 0.9235332893869479
  },
  "bema_baseline": {
   "n": 1517,
   "pearson_r_expected_stars": 0.5351059055276922,
   "mae_stars_expected": 0.7467573764651091,
   "exact_star_accuracy_rounded": 0.5603164139749506,
   "within_one_star_rounded": 0.8246539222148979
  },
  "bema_v2": {
   "n": 1517,
   "pearson_r_expected_stars": 0.8000647002850413,
   "mae_stars_expected": 0.5408927980974936,
   "exact_star_accuracy_rounded": 0.6196440342781806,
   "within_one_star_rounded": 0.9143045484508899
  },
  "bema_v3": {
   "n": 15
```

## speed_cost

```json
{
 "source": "Jev/LLM latency is client-side wall clock from the same cloud machine, including the network round trip; BEMA latency is local CPU forward pass",
 "rows": {
  "jev": {
   "median_ms_concurrent_core_run": 104.01020300002983,
   "median_ms_one_at_a_time": 99.00380149974808,
   "usd_per_1000_banking_decisions": 0.071,
   "usd_per_1000_spam_decisions": 0.0124
  },
  "claude-haiku-5.5": {
   "median_ms_one_at_a_time": 836.7891415000486,
   "usd_per_1000_banking_decisions": 0.1004,
   "usd_per_1000_spam_decisions": 0.019
  },
  "gpt-6-luna": {
   "median_ms_one_at_a_time": 1193.5040079997634,
   "usd_per_1000_banking_decisions": 0.0582,
   "usd_per_1000_spam_decisions": null
  },
  "qwen3.8-flash": {
   "median_ms_one_at_a_time": 983.5645510001996,
   "usd_per_1000_banking_decisions": 0.0478,
   "usd_per_1000_spam_decisions": 0.0193
  },
  "gpt-6.1-terra": {
   "median_ms_one_at_a_time": 1269.457256500118,
   "usd_per_1000_banking_decisions": 0.5821,
   "usd_per_1000_spam_decisions": null
  },
  "claude-sonnet-5.5": {
   "median_ms_one_at_a_time": 1137.519634500677,
   "usd_per_1000_banking_decisions": 2.0075,
   "usd_per_1000_spam_decisions": 0.3807
  },
  "claude-opus-5.5": {
   "median_ms_one_at_a_time": 1234.162661499795,
   "usd_per_1000_banking_decisions": 4.5404,
   "usd_per_1000_spam_decisions": 0.8579
  },
  "claude-fable-5.1": {
   "median_ms_one_at_a_time": 1298.8001010007793,
   "usd_per_1000_banking_decisions": 10.0488,
   "usd_per_1000_spam_decisions": null
  },
  "_bema_timing_machine": {
   "cpu_count": 2,
   "torch_threads": 2,
   "load_avg_at_start": [
    0.69140625,
    0.40625,
    0.32080078125
   ],
   "load_avg_at_end": [
    0.77197265625,
    0.4365234375,
    0.33203125
   ]
  },
  "bema_baseline": {
   "median_ms_one_at_a_time": 2.5679770005808678,
   "note": "BEMA v1 forward pass, all 3 heads, batch 1, CPU",
   "usd_per_1000_banking_decisions": "local CPU; no API cost",
   "median_ms_earlier_quiet_run": 1.6977830000541871
  },
  "bema_v2": {
   "median_ms_one_at_a_time": 12.553524000395555,
   "note": "BEMA v2 forward pass (MiniLM-L6 + 3 heads), batch 1, CPU",
   "usd_per_1000_banking_decisions": "local CPU; no API cost",
   "median_ms_earlier_quiet_run": 14.016307500241965
  },
  "bema_v2_spell": {
   "median_ms_one_at_a_time": 12.598158999026055,
   "note": "BEMA v2 forward pass plus the spelling correction step",
   "usd_per_1000_banking_decisions": "local CPU; no API cost",
   "median_ms_earlier_quiet_run": 14.06858399968769
  },
  "bema_spell_on_swap2": {
   "median_ms_one_at_a_time": 0.058344998251413926,
   "note": "spelling step alone on the first 300 swap2 queries (2 swapped letters each)",
   "usd_per_1000_banking_decisions": "local CPU; no API cost"
  },
  "bema_v2_full_gpu__gpu": {
   "median_ms_one_at_a_time": 6.909899995662272,
   "p90_ms": 10.09850000264123,
   "note": "encoder + 3 heads forward pass, batch size 1, cuda, excludes tokenization; NVIDIA GeForce RTX 4050 Laptop GPU (user's laptop)",
   "usd_per_1000_banking_decisions": "local GPU; no API cost"
  },
  "bema_v3__gpu": {
   "median_ms_one_at_a_time": 7.710600009886548,
   "p90_ms": 14.025999989826232,
   "note": "encoder + 3 heads forward pass, batch size 1, cuda, excludes tokenization; NVIDIA GeForce RTX 4050 Laptop GPU (user's laptop)",
   "usd_per_1000_banking_decisions": "local GPU; no API cost"
  }
 },
 "jev_price_assumption": "OpenRouter model page for ~typesafe/jev-latest, checked 2026-10-27: $0.042/M input, $0/M output (same figure as TypeSafe's launch pricing)"
}
```

## spend

```json
{
 "jev_usd_estimate_from_tokens": 3.695,
 "openrouter_usd_reported_by_openrouter": 3.115,
 "anthropic_usd_from_tokens_x_list_price": 141.54,
 "openrouter_usd_set_aside_runs": "see runs/llm_partial_ratelimited (rate-limited and wrong-subset GPT calls, excluded)",
 "bema_u_study": {
  "claude_written_training_material_usd": 12.072,
  "heldout_finance_anthropic_usd": 10.065,
  "heldout_finance_openrouter_usd": 0.341,
  "note": "Jev's held-out finance calls are included in jev_usd_estimate_from_tokens above"
 }
}
```
