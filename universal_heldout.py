"""
The new held-out finance test (Twitter Financial News Topic, validation split,
4,117 tweets, 20 topics): Jev and the LLMs answer the same frozen inputs that
BEMA-U is scored on. No system here was trained on this dataset's texts.

  python3 universal_heldout.py jev                 # JEV_API_KEY; JEV_MODEL=jev-1.14.0
  python3 universal_heldout.py llm <model> [n]     # a name from llm_baseline.MODELS; n = first-n random slice
  python3 universal_heldout.py report

Inputs: runs/universal/inputs/twitter_fin_topic.jsonl (made by universal_data.py,
text_sha256 per row). Outputs: runs/universal/jev/, runs/universal/llm/<model>/.
"""
import asyncio
import json
import os
import random
import sys
from datetime import datetime, timezone

import bench_jev as B
import llm_baseline as L

INP = "runs/universal/inputs/twitter_fin_topic.jsonl"
CATS = json.load(open("runs/universal/inputs/twitter_fin_topic_categories.json"))["categories"]
INSTRUCTIONS = "Which topic is this finance news tweet about?"
QUESTION = {"topic": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": {c: c for c in CATS}}}


def rows():
    return B.read_jsonl(INP)


def sample(n):
    rs = rows()
    if not n or n >= len(rs):
        return rs
    keep = set(random.Random(23).sample(range(len(rs)), n))
    return [r for r in rs if r["idx"] in keep]


async def jev():
    os.makedirs("runs/universal/jev", exist_ok=True)
    path = "runs/universal/jev/twitter_fin_topic.jsonl"
    done = {r["idx"] for r in B.read_jsonl(path) if r["status"] == "ok"} if os.path.exists(path) else set()
    todo = [r for r in rows() if r["idx"] not in done]
    print(f"Jev: {len(done)} cached, {len(todo)} to call", flush=True)
    B.check_budget(len(todo), 400, "twitter_fin_topic")
    sem, lock = asyncio.Semaphore(8), asyncio.Lock()
    async with B.client() as c:
        async def one(r, f):
            async with sem:
                rec = await B.call_jev(c, {"model": B.model_name(), "state": r["text"], "questions": QUESTION})
            rec.update({"idx": r["idx"], "text_sha256": r["text_sha256"], "ts_utc": datetime.now(timezone.utc).isoformat()})
            async with lock:
                f.write(json.dumps(rec) + "\n")
        with open(path, "a", encoding="utf-8") as f:
            await asyncio.gather(*(one(r, f) for r in todo))
    print(f"Jev done; spent ${B.spent_usd():.3f}")


def llm_prompt(text):
    return ("Classify this finance news tweet into exactly one of these topics:\n" + "\n".join(CATS)
            + f'\n\nTweet:\n"""{text}"""\n\n'
            'Reply with only a JSON object: {"topic": "<one topic from the list, copied exactly>", '
            '"confidence": <probability from 0 to 1 that your topic is correct>}')


def llm_body(model, text):
    b = L.body(model, {"kind": "banking", "state": text})   # same settings as every other LLM run
    content = llm_prompt(text)
    if "system" in b:
        b["messages"] = [{"role": "user", "content": content}]
    else:
        b["messages"] = [{"role": "system", "content": L.SYSTEM}, {"role": "user", "content": content}]
    return b


async def llm(model, n):
    out = f"runs/universal/llm/{model}"
    os.makedirs(out, exist_ok=True)
    path = f"{out}/twitter_fin_topic{'_s' + str(n) if n else ''}.jsonl"
    done = {r["idx"] for r in B.read_jsonl(path) if r["status"] == "ok"} if os.path.exists(path) else set()
    todo = [r for r in sample(n) if r["idx"] not in done]
    prov = L.provider(model)
    print(f"{model}: {len(done)} cached, {len(todo)} to call ({prov})", flush=True)
    sem, lock = asyncio.Semaphore(6), asyncio.Lock()
    async with L.client(prov) as c:
        async def one(r, f):
            async with sem:
                rec = await L.call(c, llm_body(model, r["text"]))
            rec.update({"idx": r["idx"], "text_sha256": r["text_sha256"], "model": L.MODELS[model]})
            async with lock:
                f.write(json.dumps(rec) + "\n")
        with open(path, "a", encoding="utf-8") as f:
            await asyncio.gather(*(one(r, f) for r in todo))
    cost = sum(float((r.get("usage") or {}).get("cost") or 0) for r in B.read_jsonl(path) if r["status"] == "ok")
    print(f"{model}: done, ${cost:.2f} on this test")


def report():
    gold = {r["idx"]: CATS[r["label"]] for r in rows()}
    h = {r["idx"]: r["text_sha256"] for r in rows()}
    res = {}

    def score(name, pairs):
        n = len(pairs)
        acc = sum(p == gold[i] for i, p, _ in pairs) / n
        confs = [c for _, _, c in pairs if c is not None]
        e = B.ece_exact([c for _, p, c in pairs if c is not None],
                        [float(p == gold[i]) for i, p, c in pairs if c is not None]) if confs else None
        res[name] = {"n": n, "accuracy_pct": round(100 * acc, 1), "ece": round(e, 4) if e is not None else None,
                     "ci95": [round(100 * x, 1) for x in B.wilson(sum(p == gold[i] for i, p, _ in pairs), n)]}
    p = "runs/universal/jev/twitter_fin_topic.jsonl"
    jev_pairs = []
    if os.path.exists(p):
        pairs = []
        for r in B.read_jsonl(p):
            if r["status"] != "ok":
                continue
            assert r["text_sha256"] == h[r["idx"]]
            a = r["response"]["answers"]["topic"]
            pairs.append((r["idx"], a["choice"], a.get("confidence")))
        score("jev", pairs)
        jev_pairs = pairs
    for d in sorted(os.listdir("runs/universal/llm")) if os.path.isdir("runs/universal/llm") else []:
        for fn in sorted(os.listdir(f"runs/universal/llm/{d}")):
            pairs = []
            for r in B.read_jsonl(f"runs/universal/llm/{d}/{fn}"):
                if r["status"] != "ok":
                    continue
                assert r["text_sha256"] == h[r["idx"]]
                objs = [o for o in L._json_objects(r.get("content") or "") if isinstance(o, dict) and "topic" in o]
                t = str(objs[-1]["topic"]).strip() if objs else None
                try:
                    c = float(objs[-1]["confidence"]) if objs else None
                except Exception:
                    c = None
                pairs.append((r["idx"], t, c))
            score(f"{d}/{fn[:-6]}", pairs)
            if "_s" in fn and jev_pairs:
                keys = {i for i, _, _ in pairs}
                score(f"jev on the same {fn[:-6].split('_s')[-1]} tweets", [x for x in jev_pairs if x[0] in keys])
    bdir = "runs/universal/bema"
    for fn in sorted(os.listdir(bdir)) if os.path.isdir(bdir) else []:
        pairs = []
        for r in B.read_jsonl(f"{bdir}/{fn}"):
            assert r["text_sha256"] == h[r["idx"]]
            pairs.append((r["idx"], CATS[r["pred"]], r["conf"]))
        score(fn.replace("_twitter_fin_topic.jsonl", ""), pairs)
    json.dump(res, open("runs/universal/heldout_report.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "jev":
        asyncio.run(jev())
    elif cmd == "llm":
        asyncio.run(llm(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0))
    else:
        report()
