"""
Single-example latency for the local BEMA models, measured with nothing else
running on the machine (the per-model timings written during the busy
training period were inflated by CPU contention and are not reported).

Protocol: the first 300 clean BANKING77 test queries, 20 warm-up passes, then
one query at a time; tokenization is timed separately from the forward pass.
Writes runs/bema/timing_quiet.json.
"""
import json
import os
import pickle
import statistics
import time

import torch
from tokenizers import Tokenizer

import bench_jev as B


def stats(xs):
    xs = sorted(xs)
    return {"median_ms": statistics.median(xs), "p90_ms": xs[int(0.9 * len(xs))]}


def main():
    rows = B.load_inputs("banking77", "clean")[:300]
    texts = [r["text"] for r in rows]
    out = {"_machine": {"cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
                        "load_avg_at_start": os.getloadavg()}}
    data = B.load_data()

    # BEMA v1 (from-scratch encoder, byte-level BPE tokenizer)
    from model import JevCloneEncoder
    tok = Tokenizer.from_file("data/tokenizer_multitask.json")
    m = JevCloneEncoder(vocab_size=data["vocab_size"], max_len=data["max_len"],
                        num_choice_classes=data["num_choice_classes"], enable_score=True)
    m.load_state_dict(torch.load("jev_clone_multitask_calibrated.pt", map_location="cpu"))
    m.eval()

    def enc1(t):
        x = tok.encode(t).ids[:data["max_len"]]
        pad = data["max_len"] - len(x)
        return torch.tensor([x + [data["pad_id"]] * pad]), torch.tensor([[1] * len(x) + [0] * pad])
    fw, tk = [], []
    with torch.no_grad():
        for t in texts[:20]:
            m.forward_all(*enc1(t))
        for t in texts:
            t0 = time.perf_counter(); i, k = enc1(t); t1 = time.perf_counter()
            m.forward_all(i, k); t2 = time.perf_counter()
            tk.append((t1 - t0) * 1000); fw.append((t2 - t1) * 1000)
    out["bema_baseline"] = {**stats(fw), "tokenize_median_ms": statistics.median(tk),
                            "note": "BEMA v1 forward pass, all 3 heads, batch 1, CPU"}

    # BEMA v2 (pretrained MiniLM encoder) and the spelling front end
    import bema_v2 as V
    import bema_spell as S
    path, tok2 = V.load_tok()
    m2 = V.BemaV2(path, data["num_choice_classes"])
    m2.load_state_dict(torch.load(V.CKPT, map_location="cpu"))
    m2.eval()
    speller = S.build_speller("choice")
    fw, tk, sp = [], [], []
    with torch.no_grad():
        for t in texts[:20]:
            m2.pool(*V.enc(tok2, [t]))
        for t in texts:
            t0 = time.perf_counter(); S.correct(speller, t); t1 = time.perf_counter()
            ids, mask = V.enc(tok2, [t]); t2 = time.perf_counter()
            p = m2.pool(ids, mask); m2.noul_head(p); m2.choice_head(p); m2.score_head(p); t3 = time.perf_counter()
            sp.append((t1 - t0) * 1000); tk.append((t2 - t1) * 1000); fw.append((t3 - t2) * 1000)
    out["bema_v2"] = {**stats(fw), "tokenize_median_ms": statistics.median(tk),
                      "note": "BEMA v2 forward pass (MiniLM-L6 + 3 heads), batch 1, CPU"}
    out["bema_v2_spell"] = {"median_ms": statistics.median([a + b for a, b in zip(fw, sp)]),
                            "speller_median_ms": statistics.median(sp),
                            "note": "BEMA v2 forward pass plus the spelling correction step"}
    os.makedirs("runs/bema", exist_ok=True)
    json.dump(out, open("runs/bema/timing_quiet.json", "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
