"""
Build the multi-task dataset: three REAL, human-labeled datasets sharing
ONE byte-level BPE vocabulary/tokenizer, so a single encoder trunk can
serve all three typed decision heads (Noul, Choice, Score) from one
shared representation space.

  - Noul  (binary):        SMS Spam Collection (already used in the
                            single-task pipeline) -- is_spam yes/no,
                            human-labeled by the dataset's original
                            curators.
  - Choice (multi-class):  Banking77 (PolyAI-LDN/task-specific-datasets)
                            -- 77 real customer-service intent categories,
                            human-labeled via the original BANKING77 paper
                            (Casanueva et al. 2020, CC-BY 4.0).
  - Score (continuous):    Amazon Fine Food Reviews (McAuley & Leskovec
                            2013, SNAP/Stanford; published on Kaggle by
                            the SNAP team's own account under CC0: Public
                            Domain) -- real customer star ratings (1-5,
                            the reviewer's own rating, not derived or
                            inferred) paired with the reviewer's own
                            review text. Single-text input (Summary +
                            Text), no sentence-pair concatenation needed.

  NOTE: an earlier version of this script used the STS Benchmark
  (sentence-pair similarity) for the Score task. That dataset was
  replaced entirely after a repo review found its underlying sentence
  text mixed sub-sources with unresolved, non-uniform licensing terms
  (a Microsoft Research agreement requirement for some rows, Stack
  Exchange CC-BY-SA 3.0 with per-post attribution for others) -- a real
  compliance gap, not a hypothetical one. Amazon Fine Food Reviews was
  chosen specifically to avoid repeating that mistake: CC0 is a single,
  unambiguous, maximally permissive license (public domain, no
  attribution even required) directly from the dataset's original
  creators' own Kaggle listing, not inferred from a downstream mirror.

IMPORTANT ON PROVENANCE: none of these labels were produced by an LLM.
Spam/ham labels are the original SMS Spam Collection curator labels.
Banking77 categories are the original crowd-sourced intent annotations
(BANKING77's CC-BY 4.0 license was confirmed by reading the LICENSE file
in PolyAI-LDN/task-specific-datasets directly, not just citing the
paper). Amazon Fine Food Reviews' star ratings are each reviewer's own
1-5 rating of the product they reviewed -- real, human-assigned, not
synthesized or inferred. This script only reformats these labels; it
does not relabel or filter based on any model's output. The "not
LLM-generated" claim for all three datasets rests on their publication
history and age (all predate widespread LLM use), not on this repo
independently re-verifying original annotation records. The Amazon
Fine Food Reviews CSV used here (data/raw/amazon_food_reviews.csv) is
the first 10,112 rows of the original ~568,454-row dataset, obtained
from a third-party GitHub mirror (Kaggle itself requires login and is
unreachable from this sandbox) -- content spot-checked against the
well-known first rows of the original dataset to confirm it is genuine,
not fabricated.
"""
import csv
import json
import os
import random
import pickle

from tokenizers import ByteLevelBPETokenizer

random.seed(42)

MAX_LEN = 64
SPECIAL_TOKENS = ["<pad>", "<unk>"]
VOCAB_SIZE = 8000

# ---------------------------------------------------------------------------
# Load each real dataset
# ---------------------------------------------------------------------------

# Noul: SMS Spam Collection (same source as prepare_data.py)
spam_rows = []
with open("data/spam.csv", encoding="latin-1") as f:
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        if len(r) < 2:
            continue
        label, text = r[0], r[1]
        if label not in ("ham", "spam"):
            continue
        spam_rows.append({"text": text, "label": 1 if label == "spam" else 0})
random.shuffle(spam_rows)
n = len(spam_rows)
spam_split = {
    "train": spam_rows[: int(n * 0.7)],
    "val": spam_rows[int(n * 0.7): int(n * 0.85)],
    "test": spam_rows[int(n * 0.85):],
}
print(f"Noul (spam): train={len(spam_split['train'])} val={len(spam_split['val'])} test={len(spam_split['test'])}")

# Choice: Banking77 -- has real train/test files; carve val out of train
with open("data/raw/banking_categories.json") as f:
    categories = json.load(f)
cat2idx = {c: i for i, c in enumerate(categories)}


def load_banking(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({"text": r["text"], "label": cat2idx[r["category"]]})
    return rows


banking_train_full = load_banking("data/raw/banking_train.csv")
banking_test = load_banking("data/raw/banking_test.csv")
random.shuffle(banking_train_full)
n_val = int(len(banking_train_full) * 0.15)
banking_split = {
    "train": banking_train_full[n_val:],
    "val": banking_train_full[:n_val],
    "test": banking_test,
}
print(f"Choice (banking77): train={len(banking_split['train'])} val={len(banking_split['val'])} "
      f"test={len(banking_split['test'])} classes={len(categories)}")


# Score: Amazon Fine Food Reviews (CC0) -- single CSV, no pre-made splits
food_rows = []
with open("data/raw/amazon_food_reviews.csv", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            score = float(r["Score"])
        except (ValueError, KeyError):
            continue
        summary = (r.get("Summary") or "").strip()
        text = (r.get("Text") or "").replace("<br />", " ").strip()
        combined = f"{summary}. {text}" if summary else text
        if not combined:
            continue
        food_rows.append({"text": combined, "label": (score - 1) / 4.0})  # normalize 1-5 -> 0-1
random.shuffle(food_rows)
n = len(food_rows)
score_split = {
    "train": food_rows[: int(n * 0.7)],
    "val": food_rows[int(n * 0.7): int(n * 0.85)],
    "test": food_rows[int(n * 0.85):],
}
print(f"Score (Amazon Fine Food Reviews): train={len(score_split['train'])} "
      f"val={len(score_split['val'])} test={len(score_split['test'])}")

# ---------------------------------------------------------------------------
# Train ONE shared BPE tokenizer on the union of all three TRAIN splits only
# ---------------------------------------------------------------------------
corpus_path = "data/_multitask_corpus.txt"
with open(corpus_path, "w", encoding="utf-8") as f:
    for row in spam_split["train"]:
        f.write(row["text"].replace("\n", " ") + "\n")
    for row in banking_split["train"]:
        f.write(row["text"].replace("\n", " ") + "\n")
    for row in score_split["train"]:
        f.write(row["text"].replace("\n", " ") + "\n")

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=[corpus_path], vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=SPECIAL_TOKENS)
os.remove(corpus_path)

pad_id = tokenizer.token_to_id("<pad>")
assert pad_id == 0
vocab_size = tokenizer.get_vocab_size()
print(f"Shared BPE vocab size: {vocab_size}")
tokenizer.save("data/tokenizer_multitask.json")


def encode(text):
    enc = tokenizer.encode(text)
    ids = enc.ids[:MAX_LEN]
    mask = [1] * len(ids)
    pad_n = MAX_LEN - len(ids)
    ids = ids + [pad_id] * pad_n
    mask = mask + [0] * pad_n
    return ids, mask


def build(rows):
    out = []
    for r in rows:
        ids, mask = encode(r["text"])
        out.append({"ids": ids, "mask": mask, "label": r["label"], "text": r["text"]})
    return out


data = {
    "noul": {k: build(v) for k, v in spam_split.items()},
    "choice": {k: build(v) for k, v in banking_split.items()},
    "score": {k: build(v) for k, v in score_split.items()},
    "num_choice_classes": len(categories),
    "choice_categories": categories,
    "vocab_size": vocab_size,
    "max_len": MAX_LEN,
    "pad_id": pad_id,
}

with open("data/prepared_multitask.pkl", "wb") as f:
    pickle.dump(data, f)

print("Saved data/prepared_multitask.pkl and data/tokenizer_multitask.json")
