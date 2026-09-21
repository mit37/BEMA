"""
Diagnostic follow-up on the Score head failure (Pearson r=0.29, see
FINDINGS.md section 2). Question: is the shared encoder's pooled
representation actually failing to capture sentence semantics at all, or
does it capture SOME similarity structure that the naive
concatenate-then-mean-pool-then-MLP scoring architecture just isn't
extracting well?

Test: encode sentence1 and sentence2 SEPARATELY through the same trained
shared encoder (model.encode(), no retraining, same weights the
multi-task model already has), then measure the correlation between
plain cosine similarity of the two pooled vectors and the real STS-B
human similarity scores. No new training, no new parameters -- this
isolates representation quality from scoring-head architecture.
"""
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import JevCloneEncoder
from train_multitask import TaskDataset, data, vocab_size, max_len, num_choice_classes, device

model = JevCloneEncoder(
    vocab_size=vocab_size, max_len=max_len,
    num_choice_classes=num_choice_classes, enable_score=True,
).to(device)
model.load_state_dict(torch.load("jev_clone_multitask_calibrated.pt", map_location=device))
model.eval()

# The STS-B rows in data["score"] were encoded as "sentence1 <sep> sentence2"
# as ONE text for the concat-MLP approach. For this diagnostic we need the
# original sentence pairs separately, so re-read the raw STS-B test file
# directly (same file prepare_multitask.py used) rather than the prepared
# concatenated encoding.
import csv
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("data/tokenizer_multitask.json")
pad_id = data["pad_id"]


def encode(text):
    enc = tokenizer.encode(text)
    ids = enc.ids[:max_len]
    mask = [1] * len(ids)
    pad_n = max_len - len(ids)
    ids = ids + [pad_id] * pad_n
    mask = mask + [0] * pad_n
    return ids, mask


pairs = []
with open("data/raw/stsb_test.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    for r in reader:
        if len(r) < 3:
            continue
        s1, s2, score = r[0], r[1], r[2]
        try:
            score = float(score)
        except ValueError:
            continue
        pairs.append((s1, s2, score / 5.0))

print(f"Loaded {len(pairs)} real STS-B test pairs")

cos_sims, true_scores = [], []
with torch.no_grad():
    for s1, s2, score in pairs:
        ids1, mask1 = encode(s1)
        ids2, mask2 = encode(s2)
        ids1_t = torch.tensor([ids1], dtype=torch.long).to(device)
        mask1_t = torch.tensor([mask1], dtype=torch.long).to(device)
        ids2_t = torch.tensor([ids2], dtype=torch.long).to(device)
        mask2_t = torch.tensor([mask2], dtype=torch.long).to(device)

        pooled1 = model.encode(ids1_t, mask1_t)
        pooled2 = model.encode(ids2_t, mask2_t)
        cos = F.cosine_similarity(pooled1, pooled2).item()

        cos_sims.append(cos)
        true_scores.append(score)

cos_t = torch.tensor(cos_sims)
true_t = torch.tensor(true_scores)
pearson_cos = torch.corrcoef(torch.stack([cos_t, true_t]))[0, 1].item()

print(f"\nBi-encoder cosine similarity (NO retraining, same encoder weights) vs real STS-B labels:")
print(f"  Pearson r = {pearson_cos:.4f}")
print(f"\nFor comparison, the trained concat+MLP Score head achieved Pearson r = 0.2941 on this same test set.")

if pearson_cos > 0.35:
    print("\n=> The encoder DOES capture meaningful similarity structure -- the concat+MLP scoring")
    print("   architecture was the bottleneck, not the encoder's representations. A bi-encoder/Siamese")
    print("   scoring approach (encode separately, compare) would likely outperform the current Score head.")
elif pearson_cos < 0.15:
    print("\n=> Cosine similarity of raw pooled states does NOT capture much similarity structure either --")
    print("   this suggests the encoder's mean-pooled representations, not just the scoring head, are the")
    print("   bottleneck. Fixing this needs either supervised contrastive/similarity training of the encoder")
    print("   itself (not just a downstream head), a bigger encoder, or pretrained weights.")
else:
    print("\n=> Cosine similarity captures SOME structure, better or comparable to the trained head, but not")
    print("   strongly. Both the encoder's representation quality and the scoring architecture likely need")
    print("   improvement; this alone doesn't isolate one cause.")
