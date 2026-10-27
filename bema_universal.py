"""
BEMA-U: one open model that answers "which of these options fits?" for option
sets it has never seen, the way Jev does.

Trained on the pooled datasets from universal_data.py (BANKING77, CLINC150 and
Twitter Financial News Topic held out entirely), it is scored on the exact frozen
inputs and option strings Jev received, so its numbers line up with Jev's.

Two architectures (U_MODE):
  bi     text and each option embedded separately (BGE); score = scaled cosine.
         Fast: options are embedded once per question.
  cross  text and one option read together (a BGE reranker); score = a logit.
         Slower (one pass per option) but usually much more accurate.
Both learn a "none of the other options apply" answer: on U_P_NONE of training
examples the right option is removed from the list and "none" becomes correct,
which is what lets it answer CLINC150's out-of-scope option.

Optional extras, used only when the files exist:
  data/universal/descriptions.json   {set: {label: description}}  (Claude-written,
                                      training label sets only; test options keep
                                      Jev's exact strings)
  data/universal/injection_train.jsonl  planted-note training messages (U_INJECT=1)

  python3 bema_universal.py train
  python3 bema_universal.py predict
Environment: U_MODE, U_MODEL_ID, U_NAME, U_EPOCHS, U_LR, U_SEED, U_MAX_STEPS,
U_INJECT, U_POOL (cls|mean), U_DEVICE, U_LIMIT (dry runs: first N rows per input
file), U_OUT (prediction root, default runs).
"""
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

import bench_jev as B
import redteam_jev as R

MODE = os.environ.get("U_MODE", "bi")
MODEL_ID = os.environ.get("U_MODEL_ID") or ("BAAI/bge-base-en-v1.5" if MODE == "bi" else "BAAI/bge-reranker-base")
NAME = os.environ.get("U_NAME", f"bu_{MODE}")
EPOCHS = int(os.environ.get("U_EPOCHS", "2"))
LR = float(os.environ.get("U_LR", "3e-5"))
SEED = int(os.environ.get("U_SEED", "7"))
MAX_STEPS = int(os.environ.get("U_MAX_STEPS", "0"))
P_NONE = float(os.environ.get("U_P_NONE", "0.15"))
NEG = int(os.environ.get("U_NEG", "15"))           # cross: negatives per example
MICRO = int(os.environ.get("U_MICRO", "8"))        # cross: examples per micro-batch
POOL = os.environ.get("U_POOL", "cls")
LIMIT = int(os.environ.get("U_LIMIT", "0"))
OUT = os.environ.get("U_OUT", "runs")
DEVICE = os.environ.get("U_DEVICE") or ("cuda" if torch.cuda.is_available() else
                                        "mps" if torch.backends.mps.is_available() else "cpu")
BATCH = 32
MAX_LEN = 64
PAIR_LEN = 112
NONE_TEXT = "none of the other options apply"
DATA = "data/universal"
CKPT = f"{NAME}_model"
random.seed(SEED)
torch.manual_seed(SEED)


def autocast():
    if DEVICE == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ------------------------------------------------------------------ the model
class Scorer(nn.Module):
    """score(texts, options) -> logits [len(texts), len(options)]"""

    def __init__(self, path):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(path)
        if MODE == "bi":
            self.enc = AutoModel.from_pretrained(path)
            self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        else:
            self.enc = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1)

    def dev(self):
        return next(self.parameters()).device

    def embed(self, texts):
        x = self.tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(self.dev())
        h = self.enc(**x).last_hidden_state
        if POOL == "cls":
            v = h[:, 0]
        else:
            m = x["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(v.float(), dim=-1)

    def pair_logits(self, texts, options):
        x = self.tok(texts, options, padding=True, truncation="only_first", max_length=PAIR_LEN,
                     return_tensors="pt").to(self.dev())
        return self.enc(**x).logits.squeeze(-1).float()

    def score(self, texts, options, chunk=int(os.environ.get("U_EVAL_CHUNK", "2048"))):
        """Full matrix of logits; used for evaluation."""
        if MODE == "bi":
            t = torch.cat([self.embed(texts[i:i + 256]) for i in range(0, len(texts), 256)])
            o = torch.cat([self.embed(options[i:i + 256]) for i in range(0, len(options), 256)])
            return self.log_scale.exp() * t @ o.T
        out = torch.empty(len(texts), len(options))
        pairs = [(i, j) for i in range(len(texts)) for j in range(len(options))]
        for s in range(0, len(pairs), chunk):
            p = pairs[s:s + chunk]
            lg = self.pair_logits([texts[i] for i, _ in p], [options[j] for _, j in p])
            for (i, j), v in zip(p, lg.tolist()):
                out[i, j] = v
        return out


# ------------------------------------------------------------------ training data
def option_text(desc, set_name, label):
    if set_name in desc and label in desc[set_name]:
        return desc[set_name][label]
    return LS[set_name]["names"][label]


def load_training():
    global LS
    LS = json.load(open(f"{DATA}/label_sets.json"))
    desc = json.load(open(f"{DATA}/descriptions.json")) if os.path.exists(f"{DATA}/descriptions.json") else {}
    rows = read_jsonl(f"{DATA}/train.jsonl")
    if os.environ.get("U_INJECT") == "1" and os.path.exists(f"{DATA}/injection_train.jsonl"):
        rows += read_jsonl(f"{DATA}/injection_train.jsonl")
    by_set = {}
    for r in rows:
        by_set.setdefault(r["set"], []).append(r)
    # Each step shows either the Claude-written descriptions or the plain label names (50/50),
    # so the model works with both; the test sets use Jev's plain names.
    opts = {s: ([option_text(desc, s, l) for l in LS[s]["labels"]], [LS[s]["names"][l] for l in LS[s]["labels"]])
            for s in LS}
    index = {s: {l: i for i, l in enumerate(LS[s]["labels"])} for s in LS}
    return by_set, opts, index, len(rows)


def batches(by_set, rnd, total):
    """Each batch comes from one dataset (so it shares one option list); datasets are
    sampled in proportion to sqrt(size) so the small intent sets are not drowned out."""
    names = sorted(by_set)
    weights = [math.sqrt(len(by_set[n])) for n in names]
    pools = {n: [] for n in names}
    for _ in range(total // BATCH):
        n = rnd.choices(names, weights)[0]
        if len(pools[n]) < BATCH:
            pools[n] += rnd.sample(by_set[n], len(by_set[n]))
        yield n, [pools[n].pop() for _ in range(BATCH)]


def train():
    by_set, opts, index, total = load_training()
    path = snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"])
    model = Scorer(path).to(DEVICE)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    steps_per_epoch = MAX_STEPS or total // BATCH
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.06 * steps_per_epoch * EPOCHS))) *
        max(0.0, 1 - s / (steps_per_epoch * EPOCHS)))
    rnd = random.Random(SEED)
    log = []
    for ep in range(EPOCHS):
        t0 = time.time()
        for step, (s, chunk) in enumerate(batches(by_set, rnd, total)):
            if step >= steps_per_epoch:
                break
            texts = [r["text"] for r in chunk]
            gold = [index[s][r["label"]] for r in chunk]
            none_target = [rnd.random() < P_NONE for _ in chunk]
            olist = list(opts[s][0] if rnd.random() < 0.5 else opts[s][1]) + [NONE_TEXT]
            none_i = len(olist) - 1
            if MODE == "bi":
                with autocast():
                    lg = model.score(texts, olist)                       # [B, K+1]
                    mask = torch.zeros_like(lg, dtype=torch.bool)
                    for i, (g, nt) in enumerate(zip(gold, none_target)):
                        if nt:
                            mask[i, g] = True
                    lg = lg.masked_fill(mask, -1e4)
                    target = torch.tensor([none_i if nt else g for g, nt in zip(gold, none_target)], device=lg.device)
                    loss = F.cross_entropy(lg, target)
            if MODE == "bi":
                opt.zero_grad()
                loss.backward()
            else:
                # Cross mode reads every (text, option) pair, so the batch is processed in
                # micro-batches with gradient accumulation to bound memory.
                opt.zero_grad()
                tot = 0.0
                for m0 in range(0, len(texts), MICRO):
                    pt, po, groups, target = [], [], [], []
                    for t, g, nt in zip(texts[m0:m0 + MICRO], gold[m0:m0 + MICRO], none_target[m0:m0 + MICRO]):
                        others = [j for j in range(len(olist) - 1) if j != g]
                        cand = ([] if nt else [g]) + rnd.sample(others, min(NEG, len(others))) + [none_i]
                        rnd.shuffle(cand)
                        target.append(cand.index(none_i if nt else g))
                        groups.append(len(cand))
                        pt += [t] * len(cand)
                        po += [olist[j] for j in cand]
                    with autocast():
                        flat = model.pair_logits(pt, po)
                    lg = torch.full((len(groups), max(groups)), -1e4, device=flat.device)
                    k = 0
                    for i, n in enumerate(groups):
                        lg[i, :n] = flat[k:k + n]
                        k += n
                    part = F.cross_entropy(lg, torch.tensor(target, device=lg.device)) * len(groups) / len(texts)
                    part.backward()
                    tot += part.item()
                loss = torch.tensor(tot)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 200 == 0:
                msg = f"{NAME} epoch {ep + 1} step {step}/{steps_per_epoch} set {s} loss {loss.item():.3f} ({time.time() - t0:.0f}s)"
                print(msg, flush=True)
                log.append(msg)
        print(f"{NAME} epoch {ep + 1} done in {time.time() - t0:.0f}s", flush=True)
    os.makedirs(CKPT, exist_ok=True)
    model.to("cpu")
    model.enc.save_pretrained(CKPT)
    model.tok.save_pretrained(CKPT)
    if MODE == "bi":
        torch.save({"log_scale": model.log_scale.detach()}, f"{CKPT}/scale.pt")
    json.dump({"model_id": MODEL_ID, "mode": MODE, "epochs": EPOCHS, "lr": LR, "seed": SEED,
               "train_rows": total, "inject": os.environ.get("U_INJECT") == "1",
               "descriptions": os.path.exists(f"{DATA}/descriptions.json"), "p_none": P_NONE},
              open(f"{CKPT}/train_config.json", "w"), indent=1)
    print(f"saved {CKPT}/")


# ------------------------------------------------------------------ evaluation
def load_trained():
    model = Scorer(CKPT)
    if MODE == "bi" and os.path.exists(f"{CKPT}/scale.pt"):
        model.log_scale.data = torch.load(f"{CKPT}/scale.pt")["log_scale"]
    return model.to(DEVICE).eval()


def fit_temperature(model):
    """One temperature, fit on the TRAINING datasets' validation rows (never a test set)."""
    global LS
    LS = json.load(open(f"{DATA}/label_sets.json"))
    desc = json.load(open(f"{DATA}/descriptions.json")) if os.path.exists(f"{DATA}/descriptions.json") else {}
    val = read_jsonl(f"{DATA}/val.jsonl")
    if LIMIT:
        val = val[:LIMIT * 4]
    logits, gold = [], []
    by = {}
    for r in val:
        by.setdefault(r["set"], []).append(r)
    with torch.no_grad(), autocast():
        for s, rows in by.items():
            olist = [option_text(desc, s, l) for l in LS[s]["labels"]]
            lg = model.score([r["text"] for r in rows], olist).float().cpu()
            for row, r in zip(lg, rows):
                logits.append(row)
                gold.append(LS[s]["labels"].index(r["label"]))
    width = max(len(r) for r in logits)
    pad = torch.full((len(logits), width), float("-inf"))
    for i, r in enumerate(logits):
        pad[i, :len(r)] = r
    gold = torch.tensor(gold)
    t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(pad * t.exp(), gold)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.exp().detach())


def choose(model, temp, texts, options):
    with torch.no_grad(), autocast():
        lg = model.score(texts, options).float().cpu()
    p = torch.softmax(lg * temp, -1)
    conf, idx = p.max(-1)
    return idx.tolist(), conf.tolist(), p


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def predict():
    model = load_trained()
    temp = fit_temperature(model)
    print(f"{NAME}: temperature {temp:.3f} (fit on training-set validation rows)", flush=True)
    lim = (lambda xs: xs[:LIMIT]) if LIMIT else (lambda xs: xs)
    bank_crit = B.question_spec("banking77")["intent"]["criteria"]
    bank_keys, bank_opts = list(bank_crit), list(bank_crit.values())
    cats = B.categories()
    t0 = time.time()
    # BANKING77, every typo condition (Jev's exact option strings, no "none" option: Jev had none)
    for cond in B.TASKS["banking77"]["conditions"]:
        rows = lim(B.load_inputs("banking77", cond))
        idx, conf, _ = choose(model, temp, [r["text"] for r in rows], bank_opts)
        write(f"{OUT}/bema/{NAME}_banking77_{cond}.jsonl",
              [{"idx": r["idx"], "text_sha256": r["text_sha256"], "pred": cats.index(bank_keys[i]), "conf": c}
               for r, i, c in zip(rows, idx, conf)])
    # SMS spam as a two-option choice built from Jev's yes/no statement
    spam_opts = ["This SMS message is spam.", "This SMS message is not spam."]
    for cond in B.TASKS["spam"]["conditions"]:
        rows = lim(B.load_inputs("spam", cond))
        _, _, p = choose(model, temp, [r["text"] for r in rows], spam_opts)
        write(f"{OUT}/bema/{NAME}_spam_{cond}.jsonl",
              [{"idx": r["idx"], "text_sha256": r["text_sha256"], "pred": int(q[0] > 0.5),
                "conf": float(max(q[0], 1 - q[0])), "p_yes": float(q[0])} for r, q in zip(rows, p)])
    # red-team inputs
    clinc_crit = R.questions_for({"qset": "clinc"})["intent"]["criteria"]
    for test in sorted(os.listdir("runs/redteam/inputs")):
        items = lim(B.read_jsonl(f"runs/redteam/inputs/{test}"))
        q = items[0]["qset"]
        states = [it["state"] for it in items]
        if q in ("banking", "banking_defended", "banking_shuffled"):
            crit = R.questions_for(items[0])["intent"]["criteria"]
            keys, opts = list(crit), list(crit.values())
            idx, conf, _ = choose(model, temp, states, opts)
            out = [{"key": it["key"], "state_sha256": it["state_sha256"], "pred": keys[i], "conf": c}
                   for it, i, c in zip(items, idx, conf)]
        elif q == "clinc":
            keys, opts = list(clinc_crit), list(clinc_crit.values())
            idx, conf, _ = choose(model, temp, states, opts)
            out = [{"key": it["key"], "state_sha256": it["state_sha256"], "pred": keys[i], "conf": c}
                   for it, i, c in zip(items, idx, conf)]
        elif q == "spam":
            _, _, p = choose(model, temp, states, spam_opts)
            out = [{"key": it["key"], "state_sha256": it["state_sha256"], "pred": int(pp[0] > 0.5),
                    "conf": float(max(pp[0], 1 - pp[0])), "p_yes": float(pp[0])} for it, pp in zip(items, p)]
        else:
            continue
        write(f"{OUT}/redteam/bema/{test[:-6]}__{NAME}.jsonl", out)
    # the new held-out finance test
    tw = lim(read_jsonl("runs/universal/inputs/twitter_fin_topic.jsonl"))
    tcats = json.load(open("runs/universal/inputs/twitter_fin_topic_categories.json"))["categories"]
    idx, conf, _ = choose(model, temp, [r["text"] for r in tw], tcats)
    write(f"{OUT}/universal/bema/{NAME}_twitter_fin_topic.jsonl",
          [{"idx": r["idx"], "text_sha256": r["text_sha256"], "pred": i, "conf": c} for r, i, c in zip(tw, idx, conf)])
    # single-query latency on 200 clean BANKING77 queries (77 options)
    rows = B.load_inputs("banking77", "clean")[:20 if LIMIT else 200]
    lat = []
    with torch.no_grad(), autocast():
        if MODE == "bi":
            o = model.embed(bank_opts)   # options are embedded once per question set
        for r in rows:
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if MODE == "bi":
                (model.embed([r["text"]]) @ o.T).argmax()
            else:
                model.score([r["text"]], bank_opts).argmax()
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t1) * 1000)
    lat.sort()
    info = {"temperature": temp, "device": DEVICE,
            "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
            "latency_median_ms": lat[len(lat) // 2], "latency_p90_ms": lat[int(0.9 * len(lat))],
            "latency_note": "one query at a time, 77 options, includes tokenization",
            "predict_seconds": time.time() - t0, "limit": LIMIT}
    p = f"{OUT}/universal/models.json"
    old = json.load(open(p)) if os.path.exists(p) else {}
    old[NAME] = {**info, **(json.load(open(f"{CKPT}/train_config.json")) if os.path.exists(f"{CKPT}/train_config.json") else {})}
    json.dump(old, open(p, "w"), indent=1)
    print(f"{NAME}: predictions written ({time.time() - t0:.0f}s); latency median {info['latency_median_ms']:.1f} ms", flush=True)


if __name__ == "__main__":
    {"train": train, "predict": predict}[sys.argv[1]]()
