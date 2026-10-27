"""
BEMA zero-shot: can a small open model answer typed questions over category
sets it was never trained on, the way Jev does?

Jev answers from the option names it is given. BEMA v1/v2 have a fixed output
layer and must be retrained for every new set of options. This script tests a
schema-general alternative: embed the text and each option's description with
a pretrained sentence encoder, and pick the closest option (softmax over cosine
similarity with a temperature). New options need no retraining.

Three variants, all evaluated on the identical frozen inputs Jev saw:
  zs_offshelf      all-MiniLM-L6-v2 as downloaded, no training at all;
                   temperature fit on CLINC150's validation split.
  zs_from_clinc    fine-tuned on CLINC150 train (150 intents, different domain),
                   then tested on BANKING77, whose 77 intents it never saw.
  zs_from_banking  fine-tuned on BANKING77 train, then tested on CLINC150
                   (151 options incl. out-of-scope), which it never saw.
Option descriptions are exactly the strings Jev was given (bench_jev.py /
redteam_jev.py `criteria`), so both systems see the same label text.

Fine-tuning uses the human labels of the SOURCE dataset only; the target
dataset's labels are used only for scoring. Temperatures are fit on the SOURCE
validation split, never on the target.

  python3 bema_zeroshot.py train    # writes zs_from_clinc/, zs_from_banking/
  python3 bema_zeroshot.py predict  # writes prediction files + summary
"""
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

import bench_jev as B
import redteam_jev as R

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 4321
MAX_LEN = 64
BATCH = 32
EPOCHS = int(os.environ.get("ZS_EPOCHS", "2"))
LR = 3e-5
torch.manual_seed(SEED)


def base_path():
    return snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.txt", "*.safetensors"])


def load(path):
    return AutoTokenizer.from_pretrained(path), AutoModel.from_pretrained(path)


def embed(tok, model, texts, grad=False):
    out = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i in range(0, len(texts), 128):
            x = tok(texts[i:i + 128], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            h = model(**x).last_hidden_state
            m = x["attention_mask"].unsqueeze(-1).float()
            out.append(F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1))
    return torch.cat(out)


# --- label sets: the exact strings Jev received --------------------------------
def banking_labels():
    crit = B.question_spec("banking77")["intent"]["criteria"]
    return list(crit), [crit[k] for k in crit]


def clinc_labels():
    crit = R.questions_for({"qset": "clinc"})["intent"]["criteria"]
    return list(crit), [crit[k] for k in crit]


def clinc_split(split):
    raw = R.clinc_raw()
    oos = {"train": "oos_train", "val": "oos_val", "test": "oos_test"}[split]
    return [(t, l) for t, l in raw[split]] + [(t, "oos") for t, _ in raw[oos]]


def banking_split(split):
    data = B.load_data()
    cats = data["choice_categories"]
    return [(r["text"], cats[r["label"]]) for r in data["choice"][split]]


def fit_temperature(sims, gold_idx):
    t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(sims * t.exp(), gold_idx)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.exp().detach())   # multiplier on cosine similarity


def train_on(name, rows, keys, descs):
    tok, model = load(base_path())
    model.train()
    idx = {k: i for i, k in enumerate(keys)}
    rows = [(t, idx[l]) for t, l in rows if l in idx]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rnd = random.Random(SEED)
    scale = 20.0   # fixed similarity scale during training (as in sentence-transformers' MNRL)
    steps = int(os.environ.get("ZS_MAX_STEPS", "0")) or len(rows) // BATCH
    for ep in range(EPOCHS):
        rnd.shuffle(rows)
        t0 = time.time()
        for s in range(steps):
            chunk = rows[s * BATCH:(s + 1) * BATCH]
            # Contrast each text against its own option description plus random other
            # options of the source set (64 options per step keeps CPU cost manageable).
            golds = sorted({g for _, g in chunk})
            others = [i for i in range(len(descs)) if i not in set(golds)]
            opts = golds + rnd.sample(others, max(0, min(len(others), 64 - len(golds))))
            pos = {o: j for j, o in enumerate(opts)}
            e_t = embed(tok, model, [t for t, _ in chunk], grad=True)
            e_l = embed(tok, model, [descs[o] for o in opts], grad=True)
            loss = F.cross_entropy(scale * e_t @ e_l.T, torch.tensor([pos[g] for _, g in chunk]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if s % 50 == 0:
                print(f"  {name} epoch {ep + 1} step {s}/{steps} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(name, exist_ok=True)
    model.save_pretrained(name)
    tok.save_pretrained(name)
    print(f"saved {name}/")


def train():
    k, d = clinc_labels()
    train_on("zs_from_clinc", clinc_split("train"), k, d)
    k, d = banking_labels()
    train_on("zs_from_banking", banking_split("train"), k, d)


def temperature_on_source(tok, model, source):
    if source == "clinc":
        keys, descs = clinc_labels()
        rows = clinc_split("val")
    else:
        keys, descs = banking_labels()
        rows = banking_split("val")
    idx = {k: i for i, k in enumerate(keys)}
    sims = embed(tok, model, [t for t, _ in rows]) @ embed(tok, model, descs).T
    return fit_temperature(sims, torch.tensor([idx[l] for _, l in rows]))


def predict():
    # (model path, target -> dataset whose VALIDATION split sets the temperature).
    # The temperature always comes from the other dataset, never the one being scored.
    variants = {"zs_offshelf": (base_path(), {"banking": "clinc", "clinc": "banking"}),
                "zs_from_clinc": ("zs_from_clinc", {"banking": "clinc"}),
                "zs_from_banking": ("zs_from_banking", {"clinc": "banking"})}
    summary = {}
    for name, (path, plan) in variants.items():
        if not os.path.isdir(path):
            print(f"{name}: no model at {path}, skipped")
            continue
        tok, model = load(path)
        model.eval()
        temps = {tgt: temperature_on_source(tok, model, src) for tgt, src in plan.items()}
        summary[name] = {tgt: {"temperature": temps[tgt], "fit_on": f"{src} validation split"}
                         for tgt, src in plan.items()}
        print(f"{name}: {summary[name]}")
        targets = list(plan)

        def choose(texts, keys, descs, temp):
            p = torch.softmax(temp * (embed(tok, model, texts) @ embed(tok, model, descs).T), -1)
            conf, i = p.max(-1)
            return [(keys[j], float(c)) for j, c in zip(i.tolist(), conf)]

        if "banking" in targets:
            keys, descs = banking_labels()
            cats = B.categories()
            for cond in B.TASKS["banking77"]["conditions"]:
                rows = B.load_inputs("banking77", cond)
                out = choose([r["text"] for r in rows], keys, descs, temps["banking"])
                with open(f"runs/bema/{name}_banking77_{cond}.jsonl", "w") as f:
                    for r, (k, c) in zip(rows, out):
                        f.write(json.dumps({"idx": r["idx"], "text_sha256": r["text_sha256"],
                                            "pred": cats.index(k), "conf": c}) + "\n")
            for test in sorted(os.listdir("runs/redteam/inputs")):
                items = B.read_jsonl(f"runs/redteam/inputs/{test}")
                if items[0]["qset"] not in ("banking", "banking_defended", "banking_shuffled"):
                    continue
                out = choose([it["state"] for it in items], keys, descs, temps["banking"])
                with open(f"runs/redteam/bema/{test[:-6]}__{name}.jsonl", "w") as f:
                    for it, (k, c) in zip(items, out):
                        f.write(json.dumps({"key": it["key"], "state_sha256": it["state_sha256"],
                                            "pred": k, "conf": c}) + "\n")
            print(f"{name}: BANKING77 predictions written")
        if "clinc" in targets:
            keys, descs = clinc_labels()
            for test in ("clinc150_clean", "clinc150_swap2", "injection_oos_clinc"):
                items = B.read_jsonl(f"runs/redteam/inputs/{test}.jsonl")
                out = choose([it["state"] for it in items], keys, descs, temps["clinc"])
                with open(f"runs/redteam/bema/{test}__{name}.jsonl", "w") as f:
                    for it, (k, c) in zip(items, out):
                        f.write(json.dumps({"key": it["key"], "state_sha256": it["state_sha256"],
                                            "pred": k, "conf": c}) + "\n")
            print(f"{name}: CLINC150 predictions written")
    json.dump(summary, open("runs/bema/zeroshot_temperatures.json", "w"), indent=2)


if __name__ == "__main__":
    {"train": train, "predict": predict}[sys.argv[1]]()
