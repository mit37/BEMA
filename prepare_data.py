"""
Prepare the SMS Spam Collection dataset (real, human-labeled ground truth —
5,572 SMS messages labeled ham/spam by the original dataset curators, not
LLM-generated) into train/val/test splits.

Tokenization: byte-level BPE, trained from scratch on the TRAIN split only
(no network access, no pretrained vocab — HuggingFace hub isn't reachable
from this sandbox; see README "Where this stops being a Jev"). This is
still a real improvement over a word-level split vocab: byte-level BPE has
no true OOV/<unk> collapse (arbitrary text always decomposes to bytes) and
captures subword structure (e.g. "u", "ur", "2" txt-speak) that a
whole-word vocab with a frequency cutoff throws away.
"""
import csv
import random
import pickle

from tokenizers import ByteLevelBPETokenizer

random.seed(42)

DATA_PATH = "data/spam.csv"

rows = []
with open(DATA_PATH, encoding="latin-1") as f:
    reader = csv.reader(f)
    header = next(reader)
    for r in reader:
        if len(r) < 2:
            continue
        label, text = r[0], r[1]
        if label not in ("ham", "spam"):
            continue
        rows.append((text, 1 if label == "spam" else 0))

print(f"Total labeled examples: {len(rows)}")
print(f"Spam: {sum(l for _, l in rows)}  Ham: {sum(1 - l for _, l in rows)}")

random.shuffle(rows)
n = len(rows)
n_train = int(n * 0.7)
n_val = int(n * 0.15)
train = rows[:n_train]
val = rows[n_train:n_train + n_val]
test = rows[n_train + n_val:]
print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

# Train a byte-level BPE tokenizer on the TRAIN split only (no leakage,
# no external vocab/weights).
SPECIAL_TOKENS = ["<pad>", "<unk>"]
VOCAB_SIZE = 4000
MAX_LEN = 48

tmp_corpus_path = "data/_train_corpus.txt"
with open(tmp_corpus_path, "w", encoding="utf-8") as f:
    for text, _ in train:
        f.write(text.replace("\n", " ") + "\n")

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=[tmp_corpus_path],
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
)
import os
os.remove(tmp_corpus_path)

pad_id = tokenizer.token_to_id("<pad>")
assert pad_id == 0, f"expected <pad> at id 0, got {pad_id}"
vocab_size = tokenizer.get_vocab_size()
print(f"BPE vocab size: {vocab_size}")

tokenizer.save("data/tokenizer.json")


def encode(text):
    enc = tokenizer.encode(text)
    ids = enc.ids[:MAX_LEN]
    mask = [1] * len(ids)
    pad_n = MAX_LEN - len(ids)
    ids = ids + [pad_id] * pad_n
    mask = mask + [0] * pad_n
    return ids, mask


def build_split(split):
    out = []
    for text, label in split:
        ids, mask = encode(text)
        out.append({"ids": ids, "mask": mask, "label": label, "text": text})
    return out


data = {
    "train": build_split(train),
    "val": build_split(val),
    "test": build_split(test),
    "vocab_size": vocab_size,
    "max_len": MAX_LEN,
    "pad_id": pad_id,
}

with open("data/prepared.pkl", "wb") as f:
    pickle.dump(data, f)

print("Saved data/prepared.pkl and data/tokenizer.json")
