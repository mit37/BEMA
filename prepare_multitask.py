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
  - Score (continuous):    STS Benchmark, English portion, via the
                            stsb-multi-mt mirror (Cer et al. 2017 /
                            May 2021 repackaging) -- genuine human
                            similarity judgments in [0, 5], not
                            synthesized. Input is the two sentences joined
                            with a literal <sep> token so a single-text
                            encoder can consume a pair.
                            LICENSE NOTE: this is NOT a single CC-BY-SA
                            4.0 dataset -- the underlying sentence text
                            mixes sub-sources with their own separate
                            terms (Microsoft Research agreement required
                            for MSRpar/MSR-Video; Stack Exchange CC-BY-SA
                            3.0 with per-post/per-author attribution for
                            answers-answers/answers-forums; others).
                            This is a real, unresolved licensing
                            compliance question for the redistributed
                            CSVs here -- see README.md's "Dataset
                            provenance & licensing" section, which flags
                            it explicitly rather than assuming it away.

IMPORTANT ON PROVENANCE: none of these labels were produced by an LLM.
Spam/ham labels are the original SMS Spam Collection curator labels.
Banking77 categories are the original crowd-sourced intent annotations
(BANKING77's CC-BY 4.0 license was confirmed by reading the LICENSE file
in PolyAI-LDN/task-specific-datasets directly, not just citing the
paper). STS-B scores are the original human similarity ratings (averaged
over multiple annotators in the source study). This script only
reformats them; it does not relabel or filter based on any model's
output. The "not LLM-generated" claim for all three datasets rests on
their publication history and age (all predate widespread LLM use), not
on this repo independently re-verifying original annotation records.
"""
import csv
import json
import os
import random
import pickle

from tokenizers import ByteLevelBPETokenizer

random.seed(42)

MAX_LEN = 64
SPECIAL_TOKENS = ["<pad>", "<unk>", "<sep>"]
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


# Score: STS-B English -- has real train/dev/test files
def load_stsb(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) < 3:
                continue
            s1, s2, score = r[0], r[1], r[2]
            try:
                score = float(score)
            except ValueError:
                continue
            rows.append({"text": f"{s1} <sep> {s2}", "label": score / 5.0})  # normalize 0-5 -> 0-1
    return rows


stsb_split = {
    "train": load_stsb("data/raw/stsb_train.csv"),
    "val": load_stsb("data/raw/stsb_dev.csv"),
    "test": load_stsb("data/raw/stsb_test.csv"),
}
print(f"Score (STS-B): train={len(stsb_split['train'])} val={len(stsb_split['val'])} test={len(stsb_split['test'])}")

# ---------------------------------------------------------------------------
# Train ONE shared BPE tokenizer on the union of all three TRAIN splits only
# ---------------------------------------------------------------------------
corpus_path = "data/_multitask_corpus.txt"
with open(corpus_path, "w", encoding="utf-8") as f:
    for row in spam_split["train"]:
        f.write(row["text"].replace("\n", " ") + "\n")
    for row in banking_split["train"]:
        f.write(row["text"].replace("\n", " ") + "\n")
    for row in stsb_split["train"]:
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
    "score": {k: build(v) for k, v in stsb_split.items()},
    "num_choice_classes": len(categories),
    "choice_categories": categories,
    "vocab_size": vocab_size,
    "max_len": MAX_LEN,
    "pad_id": pad_id,
}

with open("data/prepared_multitask.pkl", "wb") as f:
    pickle.dump(data, f)

print("Saved data/prepared_multitask.pkl and data/tokenizer_multitask.json")
