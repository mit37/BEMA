"""
BEMA v2: the same typed-head design (Noul + Choice + Score on one shared
encoder, one forward pass), but with a PRETRAINED encoder instead of the
from-scratch 3-layer one. The original sandbox could not download pretrained
weights; this one can, so this tests how much of the gap to Jev came from that
constraint.

  Encoder:  sentence-transformers/all-MiniLM-L6-v2 (22M params, Apache-2.0),
            mean pooling, WordPiece tokenizer, max 64 tokens.
  Data:     the exact train/val splits in data/prepared_multitask.pkl (the
            same human labels BEMA v1 used). Test splits are never touched
            until the final evaluation.
  Typos:    each training example is replaced by a 2-swap typo version with
            probability 0.5, re-drawn every time it is sampled (v1's augmented
            model instead added one fixed typo'd duplicate per example). Only
            swap2 is used in training, so the other noise types stay held out.
  Selection and calibration: best epoch by validation score; then one
            temperature per head fit on the clean validation split.

Outputs predictions in the same formats bench_jev.py / redteam_jev.py read,
on the identical frozen inputs (runs/inputs, runs/redteam/inputs):
  runs/bema/bema_v2_<task>_<cond>.jsonl
  runs/redteam/bema/<test>__bema_v2.jsonl

  python3 bema_v2.py train      # writes bema_v2.pt
  python3 bema_v2.py predict    # writes the prediction files + timing
"""
import json
import math
import os
import pickle
import random
import sys
import time

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

import bench_jev as B
from noise import NOISE, perturb

SEED = 1234
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 64
BATCH = 32
EPOCHS = int(os.environ.get("V2_EPOCHS", "3"))
LR = 5e-5
CKPT = os.environ.get("V2_CKPT", "bema_v2.pt")
NAME = "bema_v2"
torch.manual_seed(SEED)
random.seed(SEED)
torch.set_num_threads(max(1, os.cpu_count() or 1))


class BemaV2(nn.Module):
    def __init__(self, path, n_choice):
        super().__init__()
        self.enc = AutoModel.from_pretrained(path)
        h = self.enc.config.hidden_size
        self.noul_head = nn.Linear(h, 1)
        self.choice_head = nn.Linear(h, n_choice)
        self.score_head = nn.Linear(h, 1)
        self.register_buffer("t_noul", torch.ones(1))
        self.register_buffer("t_choice", torch.ones(1))

    def pool(self, ids, mask):
        out = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return (out * m).sum(1) / m.sum(1).clamp(min=1)


def load_tok():
    path = snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.txt", "*.safetensors"])
    return path, AutoTokenizer.from_pretrained(path)


def enc(tok, texts):
    x = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return x["input_ids"], x["attention_mask"]


def batches(rows, rnd, augment):
    rows = rows[:]
    while True:
        rnd.shuffle(rows)
        for i in range(0, len(rows) - BATCH + 1, BATCH):
            chunk = rows[i:i + BATCH]
            texts = [perturb(r["text"], NOISE["swap2"], rnd) if augment and rnd.random() < 0.5 else r["text"]
                     for r in chunk]
            yield texts, [r["label"] for r in chunk]


@torch.no_grad()
def logits_on(model, tok, texts, head):
    model.eval()
    out = []
    for i in range(0, len(texts), 128):
        ids, mask = enc(tok, texts[i:i + 128])
        p = model.pool(ids, mask)
        out.append({"noul": model.noul_head, "choice": model.choice_head, "score": model.score_head}[head](p))
    return torch.cat(out)


def fit_temperature(logits, labels, binary):
    t = torch.zeros(1, requires_grad=True)   # log temperature
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)
    loss_fn = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        z = logits.squeeze(-1) / t.exp() if binary else logits / t.exp()
        loss = loss_fn(z, labels.float() if binary else labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.exp())


def train():
    data = B.load_data()
    path, tok = load_tok()
    model = BemaV2(path, data["num_choice_classes"])
    rnd = random.Random(SEED)
    it_n = batches(data["noul"]["train"], rnd, True)
    it_c = batches(data["choice"]["train"], rnd, True)
    it_s = batches(data["score"]["train"], rnd, True)
    n_pos = sum(r["label"] for r in data["noul"]["train"])
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(data["noul"]["train"]) - n_pos) / n_pos]))
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    steps = int(os.environ.get("V2_MAX_STEPS", "0")) or math.ceil(len(data["choice"]["train"]) / BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps * EPOCHS, pct_start=0.1)
    val = {t: data[t]["val"] for t in ("noul", "choice", "score")}
    best, best_state = -1e9, None
    for ep in range(EPOCHS):
        model.train()
        t0, tot = time.time(), 0.0
        for s in range(steps):
            loss = 0
            for (texts, labels), head in ((next(it_n), "noul"), (next(it_c), "choice"), (next(it_s), "score")):
                ids, mask = enc(tok, texts)
                p = model.pool(ids, mask)
                if head == "noul":
                    loss = loss + bce(model.noul_head(p).squeeze(-1), torch.tensor(labels, dtype=torch.float))
                elif head == "choice":
                    loss = loss + ce(model.choice_head(p), torch.tensor(labels))
                else:
                    loss = loss + mse(torch.sigmoid(model.score_head(p).squeeze(-1)), torch.tensor(labels, dtype=torch.float))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            if s % 50 == 0:
                print(f"  epoch {ep + 1} step {s}/{steps} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
        nv = logits_on(model, tok, [r["text"] for r in val["noul"]], "noul").squeeze(-1)
        cv = logits_on(model, tok, [r["text"] for r in val["choice"]], "choice")
        sv = torch.sigmoid(logits_on(model, tok, [r["text"] for r in val["score"]], "score").squeeze(-1))
        na = ((nv > 0).float() == torch.tensor([r["label"] for r in val["noul"]], dtype=torch.float)).float().mean().item()
        ca = (cv.argmax(-1) == torch.tensor([r["label"] for r in val["choice"]])).float().mean().item()
        mae = (sv - torch.tensor([r["label"] for r in val["score"]], dtype=torch.float)).abs().mean().item()
        comb = na + ca + (1 - mae)
        print(f"Epoch {ep + 1} loss={tot / steps:.4f} noul_val_acc={na:.4f} choice_val_acc={ca:.4f} "
              f"score_val_mae={mae:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if comb > best:
            best, best_state = comb, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    nv = logits_on(model, tok, [r["text"] for r in val["noul"]], "noul")
    cv = logits_on(model, tok, [r["text"] for r in val["choice"]], "choice")
    model.t_noul.fill_(fit_temperature(nv, torch.tensor([r["label"] for r in val["noul"]]), True))
    model.t_choice.fill_(fit_temperature(cv, torch.tensor([r["label"] for r in val["choice"]]), False))
    print(f"Temperatures: noul {model.t_noul.item():.3f}, choice {model.t_choice.item():.3f}")
    torch.save(model.state_dict(), CKPT)
    print(f"Saved {CKPT}")


def predict():
    data = B.load_data()
    path, tok = load_tok()
    model = BemaV2(path, data["num_choice_classes"])
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    cats = B.categories()

    def run(texts, kind):
        if kind == "choice":
            probs = torch.softmax(logits_on(model, tok, texts, "choice") / model.t_choice, -1)
            conf, pred = probs.max(-1)
            return [{"pred": int(p), "conf": float(c)} for p, c in zip(pred, conf)]
        if kind == "noul":
            p = torch.sigmoid(logits_on(model, tok, texts, "noul").squeeze(-1) / model.t_noul)
            return [{"pred": int(q > 0.5), "conf": float(max(q, 1 - q)), "p_yes": float(q)} for q in p]
        s = torch.sigmoid(logits_on(model, tok, texts, "score").squeeze(-1))
        return [{"stars_expected": 1 + 4 * float(v)} for v in s]

    os.makedirs("runs/bema", exist_ok=True)
    for task, spec in B.TASKS.items():
        for cond in spec["conditions"]:
            rows = B.load_inputs(task, cond)
            out = run([r["text"] for r in rows], spec["head"])
            with open(f"runs/bema/{NAME}_{task}_{cond}.jsonl", "w") as f:
                for r, o in zip(rows, out):
                    f.write(json.dumps({"idx": r["idx"], "text_sha256": r["text_sha256"], **o}) + "\n")
            print(f"{task}/{cond}: done")
    os.makedirs("runs/redteam/bema", exist_ok=True)
    for p in sorted(os.listdir("runs/redteam/inputs")):
        name = p[:-len(".jsonl")]
        items = B.read_jsonl(f"runs/redteam/inputs/{p}")
        q = items[0]["qset"]
        kind = {"banking": "choice", "banking_shuffled": "choice", "banking_defended": "choice",
                "spam": "noul", "stars": "score"}.get(q)
        if kind is None:
            continue   # CLINC150 (151-way) and multi-question latency tests: not this model's heads
        out = run([it["state"] for it in items], kind)
        with open(f"runs/redteam/bema/{name}__{NAME}.jsonl", "w") as f:
            for it, o in zip(items, out):
                if kind == "choice":
                    o = {**o, "pred": cats[o["pred"]]}
                f.write(json.dumps({"key": it["key"], "state_sha256": it["state_sha256"], **o}) + "\n")
        print(f"redteam {name}: done")
    # single-example latency, same protocol as bench_jev.py's BEMA timing
    rows = B.load_inputs("banking77", "clean")[:300]
    lat = []
    with torch.no_grad():
        for r in rows[:20]:
            model.pool(*enc(tok, [r["text"]]))
        for r in rows:
            ids, mask = enc(tok, [r["text"]])
            t0 = time.perf_counter()
            p = model.pool(ids, mask)
            model.noul_head(p), model.choice_head(p), model.score_head(p)
            lat.append((time.perf_counter() - t0) * 1000)
    timing = json.load(open("runs/bema/timing.json")) if os.path.exists("runs/bema/timing.json") else {}
    timing[NAME] = {"n": len(lat), "median_ms": sorted(lat)[len(lat) // 2], "p90_ms": sorted(lat)[int(0.9 * len(lat))],
                    "threads": torch.get_num_threads(),
                    "note": "encoder + 3 heads forward pass, batch size 1, CPU, excludes tokenization"}
    json.dump(timing, open("runs/bema/timing.json", "w"), indent=2)
    print("timing:", timing[NAME])


if __name__ == "__main__":
    {"train": train, "predict": predict}[sys.argv[1]]()
