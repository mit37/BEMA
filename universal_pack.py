"""
Freeze the BEMA-U training material for the GPU run: write data/universal/*.jsonl.xz
(what the repo ships; the .jsonl files are too big for git) and record every file's
sha256 in data/universal/manifest.json, which gpu_run_universal.py checks before training.

  python3 universal_pack.py
"""
import hashlib
import json
import lzma
import os

D = "data/universal"
FILES = [f"{D}/train.jsonl", f"{D}/val.jsonl", f"{D}/injection_train.jsonl", f"{D}/label_sets.json",
         f"{D}/descriptions.json", "runs/universal/inputs/twitter_fin_topic.jsonl",
         "runs/universal/inputs/twitter_fin_topic_categories.json"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


man = {}
for p in FILES:
    if not os.path.exists(p):
        print(f"missing {p}, skipped")
        continue
    man[p] = {"sha256": sha(p)}
    if p.endswith(".jsonl"):
        man[p]["lines"] = sum(1 for _ in open(p, encoding="utf-8"))
    if p.startswith(D) and p.endswith(".jsonl"):
        with open(p, "rb") as a, lzma.open(p + ".xz", "wb", preset=9) as b:
            b.write(a.read())
        print(f"{p}.xz  {os.path.getsize(p + '.xz') / 1e6:.1f} MB")
json.dump(man, open(f"{D}/manifest.json", "w"), indent=1)
print(f"manifest: {len(man)} files")
