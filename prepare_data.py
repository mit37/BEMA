"""
Prepare the SMS Spam Collection dataset (real, human-labeled ground truth —
5,572 SMS messages labeled ham/spam by the original dataset curators, not
LLM-generated) into train/val/test splits with a simple word-level vocab.
"""
import csv
import re
import random
import json
import pickle

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


def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


# Build vocab from train split only (no leakage)
vocab_counts = {}
for text, _ in train:
    for tok in tokenize(text):
        vocab_counts[tok] = vocab_counts.get(tok, 0) + 1

# Keep tokens seen at least twice
vocab = ["<pad>", "<unk>"] + sorted(
    [w for w, c in vocab_counts.items() if c >= 2],
    key=lambda w: -vocab_counts[w],
)
vocab = vocab[:5000]
stoi = {w: i for i, w in enumerate(vocab)}
print(f"Vocab size: {len(vocab)}")

MAX_LEN = 40


def encode(text):
    toks = tokenize(text)[:MAX_LEN]
    ids = [stoi.get(t, 1) for t in toks]
    ids = ids + [0] * (MAX_LEN - len(ids))
    mask = [1] * len(toks) + [0] * (MAX_LEN - len(toks))
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
    "vocab": vocab,
    "max_len": MAX_LEN,
}

with open("data/prepared.pkl", "wb") as f:
    pickle.dump(data, f)

print("Saved data/prepared.pkl")
