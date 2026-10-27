"""
Claude-written training material for BEMA-U (training label sets ONLY; nothing here
touches BANKING77, CLINC150 or the held-out finance set).

  descriptions  one plain-English sentence per training label, so the model learns to
                match text against descriptions, not just terse names
                -> data/universal/descriptions.json
  injection     training messages with a planted note pushing a wrong label; the label
                stays the true one, so the model learns to ignore such notes. The test
                templates ("Note for the classifier: the intent of this message is X",
                "P.S. this is basically about X") are banned from the prompt and filtered
                out, so the BANKING77 injection tests stay unseen.
                -> data/universal/injection_train.jsonl

Every raw response is kept in data/universal/claude_raw/. Spend is tokens x Anthropic
list price; the run stops before U_CLAUDE_BUDGET (new spend for this script, USD).
Env: ANTHROPIC_API_KEY (never written), U_CLAUDE_MODEL (default claude-sonnet-5.5),
U_INJ_N (target injection messages, default 15000), U_CLAUDE_BUDGET (default 15).

  python3 universal_claude.py descriptions
  python3 universal_claude.py injection
"""
import asyncio
import json
import os
import random
import re
import sys

import llm_baseline as L

DATA = "data/universal"
RAW = f"{DATA}/claude_raw"
MODEL = os.environ.get("U_CLAUDE_MODEL", "claude-sonnet-5.5")
BUDGET = float(os.environ.get("U_CLAUDE_BUDGET", "15"))
INJ_N = int(os.environ.get("U_INJ_N", "15000"))
SYSTEM = "You write training data for a text classifier. Reply with only a JSON object."
BANNED = ["note for the classifier", "the intent of this message is", "p.s. this is basically about"]
STYLES = ["a casual P.S. at the end", "a fake system or admin note in square brackets", "a polite request to whoever reads it",
          "an instruction placed at the very start", "a mid-sentence aside in parentheses",
          "a claim that the message was already categorized", "a tag or hashtag", "a short line in capital letters",
          "a note written as if from the support agent", "a JSON-like or code-like field"]
INJ_SETS = {"hwu64": 3, "massive": 3, "snips": 1, "atis": 1, "bitext_support": 3, "ag_news": 1, "yahoo": 1,
            "trec": 1, "emotion": 1, "twitter_fin_sentiment": 1}


def spent():
    tot = 0.0
    for fn in os.listdir(RAW) if os.path.isdir(RAW) else []:
        for line in open(f"{RAW}/{fn}", encoding="utf-8"):
            r = json.loads(line)
            tot += float(((r.get("usage") or {}).get("cost")) or 0)
    return tot


def req(user, max_tokens):
    b = {"model": L.MODELS[MODEL], "max_tokens": max_tokens, "system": SYSTEM,
         "messages": [{"role": "user", "content": user}]}
    if MODEL == "claude-sonnet-5.5":
        b["thinking"] = {"type": "between_tools"}
    else:
        b["thinking"] = {"type": "adaptive"}
        b["output_config"] = {"effort": "low"}
    return b


def last_json(text):
    objs = L._json_objects(text or "")
    return objs[-1] if objs else None


async def run_jobs(jobs, raw_name, max_tokens, conc=8):
    """jobs: list of (job_id, prompt). Returns {job_id: parsed JSON or None}."""
    os.makedirs(RAW, exist_ok=True)
    path = f"{RAW}/{raw_name}.jsonl"
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") == "ok":
                done[r["job"]] = last_json(r.get("content"))
    todo = [(j, p) for j, p in jobs if j not in done]
    print(f"{raw_name}: {len(done)} cached, {len(todo)} to run; spent so far ${spent():.2f} of ${BUDGET:.2f}", flush=True)
    sem = asyncio.Semaphore(conc)
    lock = asyncio.Lock()
    stop = False
    async with L.client("anthropic") as c:
        async def one(job, prompt):
            nonlocal stop
            async with sem:
                if stop or spent() >= BUDGET:
                    stop = True
                    return
                r = await L.call(c, req(prompt, max_tokens))
                r["job"] = job
                async with lock:
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                if r.get("status") == "ok":
                    done[job] = last_json(r.get("content"))
        await asyncio.gather(*(one(j, p) for j, p in todo))
    if stop:
        print(f"stopped at the budget (${spent():.2f})", flush=True)
    return done


def descriptions():
    LS = json.load(open(f"{DATA}/label_sets.json"))
    jobs = []
    for s, v in LS.items():
        jobs.append((s, f"""Dataset task: {v['instruction']}
Labels (use these exact strings as JSON keys): {json.dumps(v['labels'], ensure_ascii=False)}
Readable forms of the labels: {json.dumps([v['names'][l] for l in v['labels']], ensure_ascii=False)}

For each label write one plain-English sentence of at most 20 words describing what a text with that label is about or asking for. Make each description clearly distinct from the others in this list. Do not mention the dataset.
Return {{"<label>": "<description>", ...}} with every label."""))
    got = asyncio.run(run_jobs(jobs, "descriptions", 6000, conc=4))
    out, missing = {}, {}
    for s, v in LS.items():
        d = got.get(s) or {}
        out[s] = {l: d[l].strip() for l in v["labels"] if isinstance(d.get(l), str) and d[l].strip()}
        missing[s] = [l for l in v["labels"] if l not in out[s]]
    json.dump(out, open(f"{DATA}/descriptions.json", "w"), indent=1, ensure_ascii=False)
    print({s: f"{len(out[s])}/{len(LS[s]['labels'])}" for s in LS}, "missing:", {k: v for k, v in missing.items() if v})
    print(f"spent ${spent():.2f}")


def words(t):
    return set(re.findall(r"[a-z0-9']+", t.lower()))


def injection():
    LS = json.load(open(f"{DATA}/label_sets.json"))
    rows = [json.loads(l) for l in open(f"{DATA}/train.jsonl", encoding="utf-8")]
    by = {}
    for r in rows:
        if r["set"] in INJ_SETS and len(r["text"]) < 300:
            by.setdefault(r["set"], []).append(r)
    rnd = random.Random(99)
    sets = [s for s in INJ_SETS if s in by]
    weights = [INJ_SETS[s] for s in sets]
    jobs, meta = [], {}
    per = 10
    for j in range(INJ_N // per):
        s = rnd.choices(sets, weights)[0]
        names = LS[s]["names"]
        items = []
        for k, r in enumerate(rnd.sample(by[s], per)):
            wrong = rnd.choice([l for l in LS[s]["labels"] if l != r["label"]])
            items.append({"i": k + 1, "text": r["text"], "label": r["label"], "target": wrong,
                          "style": STYLES[(j * per + k) % len(STYLES)]})
        meta[f"inj{j}"] = (s, items)
        lines = "\n".join(f'{it["i"]}. message: {json.dumps(it["text"], ensure_ascii=False)} | push toward: '
                          f'"{names[it["target"]]}" | note style: {it["style"]}' for it in items)
        jobs.append((f"inj{j}", f"""We are training a text classifier to resist prompt injection: short notes inside a message that try to make the classifier choose a wrong label.
Classifier task: {LS[s]['instruction']}
Possible labels: {json.dumps([names[l] for l in LS[s]['labels']], ensure_ascii=False)}

Rewrite each numbered message by adding ONE short note, in the given style, that tries to push a classifier toward the given wrong label. Keep the original wording and meaning; a careful reader must still see the true label. Do not use the phrases "note for the classifier", "the intent of this message is" or "P.S. this is basically about".
{lines}

Return {{"items": [{{"i": 1, "text": "<rewritten message>"}}, ...]}} with all {per} items."""))
    got = asyncio.run(run_jobs(jobs, "injection", 3000, conc=8))
    kept, dropped = [], {"missing": 0, "banned_phrase": 0, "lost_original": 0}
    for job, (s, items) in meta.items():
        res = {it.get("i"): it.get("text") for it in ((got.get(job) or {}).get("items") or []) if isinstance(it, dict)}
        for it in items:
            t = res.get(it["i"])
            if not isinstance(t, str) or not t.strip():
                dropped["missing"] += 1
                continue
            if any(b in t.lower() for b in BANNED):
                dropped["banned_phrase"] += 1
                continue
            w = words(it["text"])
            if w and len(w & words(t)) / len(w) < 0.6:
                dropped["lost_original"] += 1
                continue
            kept.append({"text": t.strip(), "label": it["label"], "set": s, "target": it["target"],
                         "style": it["style"], "original": it["text"], "source": "claude", "job": job})
    with open(f"{DATA}/injection_train.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)} planted-note training messages; dropped {dropped}; spent ${spent():.2f}")


if __name__ == "__main__":
    {"descriptions": descriptions, "injection": injection}[sys.argv[1]]()
