"""
BEMA with a fastText-style shared trunk instead of the transformer.

linear_baseline.py showed that TF-IDF word + character n-grams beat the
small transformer on every task and barely suffer from typos. This model
keeps BEMA's design (one shared state per input, read by fixed-shape
Noul / Choice / Score heads, all three tasks trained jointly) but builds
the state from the mean embedding of hashed word 1-2 grams and character
2-5 grams, the same features the baseline uses.
"""
import re
import zlib

import torch
import torch.nn as nn

WORD_RE = re.compile(r"(?u)\b\w\w+\b")


def ngram_features(text, buckets):
    """Hashed ids of lowercased word 1-2 grams and char 2-5 grams (char_wb style)."""
    words = WORD_RE.findall(text.lower())
    grams = [f"w:{w}" for w in words] + [f"b:{a} {b}" for a, b in zip(words, words[1:])]
    for w in re.findall(r"\S+", text.lower()):
        padded = f" {w} "
        grams += [f"c:{padded[i:i + n]}" for n in range(2, 6) for i in range(len(padded) - n + 1)]
    return [zlib.crc32(g.encode()) % buckets for g in grams] or [0]


def collate(feature_lists):
    """Flatten variable-length feature lists into EmbeddingBag (index, offsets) form."""
    offsets = torch.tensor([0] + [len(f) for f in feature_lists[:-1]]).cumsum(0)
    return torch.tensor([i for f in feature_lists for i in f]), offsets


class NgramEncoder(nn.Module):
    def __init__(self, buckets=2 ** 19, d_model=96, num_choice_classes=None, enable_score=False, dropout=0.15):
        super().__init__()
        self.buckets = buckets
        self.bag = nn.EmbeddingBag(buckets, d_model, mode="mean", sparse=True)
        nn.init.normal_(self.bag.weight, std=0.05)
        self.dropout = nn.Dropout(dropout)
        self.pool_norm = nn.LayerNorm(d_model)

        def head(out):
            return nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, out))

        self.noul_head = head(1)
        self.choice_head = head(num_choice_classes) if num_choice_classes else None
        self.score_head = head(1) if enable_score else None
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)
        self.choice_temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def encode(self, index, offsets):
        return self.pool_norm(self.dropout(self.bag(index, offsets)))

    def featurize(self, texts):
        return collate([ngram_features(t, self.buckets) for t in texts])
