"""
Measure actual single-example inference latency of the multi-task
jev-clone model on CPU, for the speed/cost claim in the README's "What's
real here" section. This does NOT call any real LLM API (none is
available in this sandbox) -- it only measures this repo's own model,
and the comparison to typical LLM API latency below is a documented
reference figure, not a live measurement, and is labeled as such.
"""
import time
import pickle
import torch
from tokenizers import Tokenizer

from model import JevCloneEncoder

with open("data/prepared_multitask.pkl", "rb") as f:
    data = pickle.load(f)
vocab_size = data["vocab_size"]
max_len = data["max_len"]
pad_id = data["pad_id"]
num_choice_classes = data["num_choice_classes"]
tokenizer = Tokenizer.from_file("data/tokenizer_multitask.json")

device = "cpu"
model = JevCloneEncoder(
    vocab_size=vocab_size, max_len=max_len,
    num_choice_classes=num_choice_classes, enable_score=True,
).to(device)
model.load_state_dict(torch.load("jev_clone_multitask_calibrated.pt", map_location=device))
model.eval()

text = "I still haven't received my new card, I ordered it two weeks ago."


def encode(t):
    enc = tokenizer.encode(t)
    ids = enc.ids[:max_len]
    mask = [1] * len(ids)
    pad_n = max_len - len(ids)
    ids = ids + [pad_id] * pad_n
    mask = mask + [0] * pad_n
    return ids, mask


ids, mask = encode(text)
ids_t = torch.tensor([ids], dtype=torch.long)
mask_t = torch.tensor([mask], dtype=torch.long)

N_WARMUP = 10
N_TRIALS = 200

with torch.no_grad():
    for _ in range(N_WARMUP):
        model.forward_all(ids_t, mask_t)

    latencies = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        model.forward_all(ids_t, mask_t)
        latencies.append((time.perf_counter() - t0) * 1000)

latencies.sort()
p50 = latencies[len(latencies) // 2]
p95 = latencies[int(len(latencies) * 0.95)]
mean = sum(latencies) / len(latencies)

print(f"Single-example CPU inference latency over {N_TRIALS} trials (all 3 typed heads, one forward pass):")
print(f"  mean={mean:.2f}ms  p50={p50:.2f}ms  p95={p95:.2f}ms")
print()
print("Reference point (NOT measured here, no LLM API available in this sandbox):")
print("  Typical LLM API round-trip for a short classification-style prompt is")
print("  commonly reported in the 200ms-2000ms range depending on provider/model/")
print("  load, dominated by network + autoregressive generation, even for short")
print("  outputs. This is a documented industry reference figure, not a live")
print("  side-by-side benchmark against any specific LLM API in this environment.")
print(f"  {mean:.0f}ms vs a 200-2000ms reference band -> roughly {200/mean:.0f}x-{2000/mean:.0f}x")
print("  faster on this specific measurement, IF the reference band is accurate")
print("  for a comparable task -- this is the weakest-evidenced number in this")
print("  repo and should be treated as illustrative, not a validated benchmark.")
