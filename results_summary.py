"""
One place for every headline number in the write-up, computed from the raw
per-example files (not copied from other reports), with the file each number
comes from. The paper, slides and infographic quote only runs/results_summary.json.

  python3 results_summary.py      # writes runs/results_summary.json and runs/RESULTS.md
"""
import json
import os
import pickle
import statistics
from datetime import datetime, timezone

import bench_jev as B
import llm_baseline as L
import redteam_jev as R

OUT = {"generated_utc": datetime.now(timezone.utc).isoformat(), "sections": {}}
CATS = B.categories()


def pct(x):
    return None if x is None else round(100 * x, 1)


# ---------------------------------------------------------------- per-key correctness loaders
def jev_core(task, cond):
    rows = {r["idx"]: r for r in B.load_inputs(task, cond)}
    out = {}
    for r in B.read_jsonl(f"runs/jev/{task}_{cond}.jsonl"):
        if r["status"] == "ok":
            p, c, _ = B.parse_answer(task, r["response"])
            out[str(r["idx"])] = (float(p == B.gold(task, rows[r["idx"]]["label"])), c, p)
    return out


def bema_core(model, task, cond):
    rows = {r["idx"]: r for r in B.load_inputs(task, cond)}
    out = {}
    for r in B.read_jsonl(f"runs/bema/{model}_{task}_{cond}.jsonl"):
        p = CATS[r["pred"]] if task == "banking77" else r["pred"]
        out[str(r["idx"])] = (float(p == B.gold(task, rows[r["idx"]]["label"])), r["conf"], p)
    return out


def llm_test(model, test):
    items = {it["key"]: it for it in L.TESTS[test]()}
    out = {}
    for r in B.read_jsonl(f"runs/llm/{model}/{test}.jsonl"):
        if r["status"] != "ok" or r["key"] not in items:
            continue
        it = items[r["key"]]
        pred, conf, valid, _ = L.parse(r["content"], "spam" if it["kind"] == "spam" else "banking")
        out[r["key"]] = (float(bool(valid) and pred == it["gold"]), conf if valid else None, pred if valid else None)
    return out


def redteam_jev(test):
    items = {it["key"]: it for it in B.read_jsonl(f"runs/redteam/inputs/{test}.jsonl")}
    out = {}
    for r in B.read_jsonl(f"runs/redteam/jev/{test}.jsonl"):
        if r["status"] == "ok":
            it = items[r["key"]]
            if it["qset"] in ("banking", "banking_defended", "banking_shuffled", "clinc"):
                p, c, _ = R.parse_choice(r["response"])
            else:
                p, c, _ = R.parse_noul(r["response"])
            out[r["key"]] = (float(p == it["gold"]), c, p)
    return out, items


def redteam_bema(test, model):
    items = {it["key"]: it for it in B.read_jsonl(f"runs/redteam/inputs/{test}.jsonl")}
    out = {}
    for r in B.read_jsonl(f"runs/redteam/bema/{test}__{model}.jsonl"):
        if "pred" in r:
            out[r["key"]] = (float(r["pred"] == items[r["key"]]["gold"]), r["conf"], r["pred"])
    return out


def acc(d, keys=None):
    vals = [v[0] for k, v in d.items() if keys is None or k in keys]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def ece_of(d, keys=None):
    pairs = [(v[1], v[0]) for k, v in d.items() if (keys is None or k in keys) and v[1] is not None]
    return B.ece_exact([c for c, _ in pairs], [a for _, a in pairs]) if pairs else None


def follow(d, items, keys=None):
    vals = [float(v[2] == items[k]["meta"]["target"]) for k, v in d.items() if keys is None or k in keys]
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------- 1. banking typos, full test set
BANK_MODELS = ["jev", "bema_baseline", "bema_typo_augmented", "bema_v2", "bema_v2_spell",
               "bema_v2_full", "bema_v2_full_spell", "bema_v2_full_gpu", "bema_v2_full_gpu_spell", "bema_v3", "bema_v3_spell",
               "zs_offshelf", "zs_from_clinc", "zs_offshelf_bge_base", "zs3_from_clinc",
               "claude-haiku-5.5", "qwen3.8-flash", "claude-sonnet-5.5", "claude-opus-5.5"]
sec = {"source": "runs/jev, runs/bema, runs/llm raw files; n=3,080 BANKING77 test queries", "rows": {}}
for cond in B.TASKS["banking77"]["conditions"]:
    row = {}
    for m in BANK_MODELS:
        if m == "jev":
            d = jev_core("banking77", cond)
        elif m in L.MODELS:
            if not os.path.exists(f"runs/llm/{m}/banking77_{cond}.jsonl"):
                continue
            d = llm_test(m, f"banking77_{cond}")
        else:
            if not os.path.exists(f"runs/bema/{m}_banking77_{cond}.jsonl"):
                continue
            d = bema_core(m, "banking77", cond)
        a, n = acc(d)
        row[m] = {"accuracy_pct": pct(a), "n": n, "ece": round(ece_of(d), 4) if ece_of(d) is not None else None}
    sec["rows"][cond] = row
OUT["sections"]["banking77_typos"] = sec

# ---------------------------------------------------------------- 2. spam
sec = {"source": "n=836 SMS test messages (116 spam); majority-class baseline in 'majority'", "rows": {}}
labels = [r["label"] for r in B.load_inputs("spam", "clean")]
sec["majority_baseline_pct"] = pct(1 - sum(labels) / len(labels))
for cond in B.TASKS["spam"]["conditions"]:
    row = {}
    for m in ["jev", "bema_baseline", "bema_typo_augmented", "bema_v2", "bema_v2_spell", "bema_v2_full",
              "bema_v2_full_spell", "bema_v2_full_gpu", "bema_v2_full_gpu_spell", "bema_v3", "bema_v3_spell",
              "claude-haiku-5.5", "qwen3.8-flash", "claude-sonnet-5.5", "claude-opus-5.5"]:
        if m == "jev":
            d = jev_core("spam", cond)
        elif m in L.MODELS:
            if not os.path.exists(f"runs/llm/{m}/spam_{cond}.jsonl"):
                continue
            d = llm_test(m, f"spam_{cond}")
        else:
            if not os.path.exists(f"runs/bema/{m}_spam_{cond}.jsonl"):
                continue
            d = bema_core(m, "spam", cond)
        a, n = acc(d)
        row[m] = {"accuracy_pct": pct(a), "n": n, "ece": round(ece_of(d), 4) if ece_of(d) is not None else None}
    sec["rows"][cond] = row
OUT["sections"]["spam"] = sec

# ---------------------------------------------------------------- 3. 250-query slice: every model on identical keys
keys = L.subsample_keys()
sec = {"source": "random 250 of the 1,000-query red-team sample (random.Random(11)); identical keys for every model",
       "n_keys": len(keys), "rows": {}}
tests = {"clean": ("banking77", "clean"), "swap2": ("banking77", "swap2"), "injection_explicit": "injection_banking",
         "injection_control": "injection_control_banking", "injection_casual": "injection_casual_banking",
         "injection_casual_defended": "injection_casual_defended_banking"}
for label, t in tests.items():
    row = {}
    for m in ["jev", "bema_v2", "bema_v2_spell", "bema_v2_full", "bema_v2_full_spell", "bema_v3", "bema_v3_spell",
              "claude-haiku-5.5", "qwen3.8-flash", "gpt-6-luna", "gpt-6.1-terra",
              "claude-sonnet-5.5", "claude-opus-5.5", "claude-fable-5.1"]:
        if isinstance(t, tuple):
            if m == "jev":
                d = jev_core(*t)
            elif m in ("gpt-6-luna", "gpt-6.1-terra", "claude-fable-5.1"):
                p = f"runs/llm/{m}/banking77_{t[1]}_s250.jsonl"
                d = llm_test(m, f"banking77_{t[1]}_s250") if os.path.exists(p) else {}
            elif m in L.MODELS:
                d = llm_test(m, f"banking77_{t[1]}")
            else:
                d = bema_core(m, *t) if os.path.exists(f"runs/bema/{m}_{t[0]}_{t[1]}.jsonl") else {}
            items = None
        else:
            items = {it["key"]: it for it in B.read_jsonl(f"runs/redteam/inputs/{t}.jsonl")}
            if m == "jev":
                d, _ = redteam_jev(t)
            elif m in ("gpt-6-luna", "gpt-6.1-terra", "claude-fable-5.1"):
                p = f"runs/llm/{m}/{t}_s250.jsonl"
                d = llm_test(m, f"{t}_s250") if os.path.exists(p) else {}
            elif m in L.MODELS:
                d = llm_test(m, t)
            else:
                # BEMA takes no instructions, so its "defended" answers are its undefended ones;
                # spell-corrected variants were only scored on the typo test sets.
                t2 = None if m.endswith("_spell") else t.replace("_defended", "")
                d = redteam_bema(t2, m) if t2 and os.path.exists(f"runs/redteam/bema/{t2}__{m}.jsonl") else {}
        sub = {k: v for k, v in d.items() if k in keys}
        if not sub:
            continue
        a, n = acc(sub)
        e = {"accuracy_pct": pct(a), "n": n}
        if items and next(iter(items.values()))["meta"].get("target"):
            e["followed_injection_pct"] = pct(follow(sub, items))
        row[m] = e
    sec["rows"][label] = row
OUT["sections"]["slice250_all_models"] = sec

# ---------------------------------------------------------------- 4. injection, 1,000-query sample
sec = {"source": "runs/redteam (Jev, BEMA) and runs/llm (Haiku, Qwen); 1,000 BANKING77 queries", "rows": {}}
for label, t in [("explicit_note_end", "injection_banking"), ("explicit_note_start", "injection_prefix_banking"),
                 ("casual_ps", "injection_casual_banking"), ("neutral_note_control", "injection_control_banking"),
                 ("explicit_note_defended", "injection_defended_banking"),
                 ("casual_ps_defended", "injection_casual_defended_banking")]:
    items = {it["key"]: it for it in B.read_jsonl(f"runs/redteam/inputs/{t}.jsonl")}
    has_target = bool(next(iter(items.values()))["meta"].get("target"))
    row = {}
    for m in ["jev", "bema_baseline", "bema_v2", "bema_v2_full", "bema_v3", "claude-haiku-5.5", "qwen3.8-flash",
              "claude-sonnet-5.5", "claude-opus-5.5"]:
        if m == "jev":
            d, _ = redteam_jev(t)
        elif m in L.MODELS:
            if not os.path.exists(f"runs/llm/{m}/{t}.jsonl"):
                continue
            d = llm_test(m, t)
        else:
            src = t.replace("_defended", "") if "defended" in t else t
            if not os.path.exists(f"runs/redteam/bema/{src}__{m}.jsonl"):
                continue
            d = redteam_bema(src, m)
        if not d:
            continue
        a, n = acc(d)
        e = {"accuracy_pct": pct(a), "n": n, "mean_confidence": round(statistics.mean(v[1] for v in d.values() if v[1] is not None), 3)}
        if has_target:
            e["followed_injection_pct"] = pct(follow(d, items))
        row[m] = e
    sec["rows"][label] = row
sec["note"] = ("BEMA cannot read instructions, so its 'defended' rows are its undefended results; "
               "clean accuracy on these 1,000 queries: see clean_reference.")
sec["clean_reference"] = {m: pct(acc({k: v for k, v in (jev_core("banking77", "clean") if m == "jev" else bema_core(m, "banking77", "clean")).items()
                                      if k in {it["key"] for it in B.read_jsonl("runs/redteam/inputs/injection_banking.jsonl")}})[0])
                          for m in ["jev", "bema_baseline", "bema_v2"]}
OUT["sections"]["injection"] = sec

# ---------------------------------------------------------------- 5. the rest of the red-team suite: copied from the
# regenerated redteam_report.json (recomputed from raw there; independently audited)
rt = json.load(open("runs/redteam_report.json"))["tests"]
OUT["sections"]["redteam_other"] = {"source": "runs/redteam_report.json",
                                    **{k: rt[k] for k in rt if k in (
                                        "injection_spam", "evasion_spam", "ood_spam_on_banking", "gibberish_banking",
                                        "order_shuffle_banking", "repeat_banking_spread", "question_count",
                                        "score_amazon_clean", "score_amazon_swap2", "clinc150_clean", "clinc150_swap2",
                                        "injection_oos_clinc", "latency_sequential_banking")}}
if os.path.exists("clinc150_oos_results.pkl"):
    OUT["sections"]["redteam_other"]["bema_clinc150_threshold_rejection"] = {
        "source": "clinc150_oos_threshold.py (rebuild), tau chosen on validation",
        **{k: v for k, v in pickle.load(open("clinc150_oos_results.pkl", "rb")).items() if not hasattr(v, "shape")}}

# ---------------------------------------------------------------- 6. speed and cost per 1,000 decisions
core = json.load(open("runs/report.json"))
llm = json.load(open("runs/llm_report.json"))
speed = {"jev": {"median_ms_concurrent_core_run": core["jev_latency_ms"]["median"],
                 "median_ms_one_at_a_time": rt["latency_sequential_banking"]["jev"]["median_latency_ms"],
                 "usd_per_1000_banking_decisions": None, "usd_per_1000_spam_decisions": None}}
for task in ("banking77", "spam"):
    toks = [r["response"]["usage"]["input_tokens"] for r in B.read_jsonl(f"runs/jev/{task}_clean.jsonl") if r["status"] == "ok"]
    speed["jev"][f"usd_per_1000_{'banking' if task == 'banking77' else 'spam'}_decisions"] = \
        round(1000 * statistics.mean(toks) * B.PRICE_PER_M_INPUT / 1e6, 4)
for m, T in llm["models"].items():
    b = T.get("banking77_clean") or T.get("banking77_clean_s250")
    s = T.get("spam_clean")
    speed[m] = {"median_ms_one_at_a_time": T["latency_sequential"]["latency_ms"]["median"] if "latency_sequential" in T else None,
                "usd_per_1000_banking_decisions": round(b["cost_per_1000_calls_usd"], 4) if b else None,
                "usd_per_1000_spam_decisions": round(s["cost_per_1000_calls_usd"], 4) if s else None}
if os.path.exists("runs/bema/timing_quiet.json"):
    for m, v in json.load(open("runs/bema/timing_quiet.json")).items():
        if m.startswith("_"):
            speed["_bema_timing_machine"] = v
            continue
        speed[m] = {"median_ms_one_at_a_time": v["median_ms"], "note": v.get("note"),
                    "usd_per_1000_banking_decisions": "local CPU; no API cost"}
OUT["sections"]["speed_cost"] = {"source": "Jev/LLM latency is client-side wall clock from the same cloud machine, "
                                           "including the network round trip; BEMA latency is local CPU forward pass",
                                 "rows": speed, "jev_price_assumption": B.PRICE_SOURCE}

# ---------------------------------------------------------------- 7. spend
OUT["spend"] = {"jev_usd_estimate_from_tokens": round(B.spent_usd(), 3),
                "openrouter_usd_reported_by_openrouter": round(L.spent("openrouter"), 3),
                "anthropic_usd_from_tokens_x_list_price": round(L.spent("anthropic"), 3),
                "openrouter_usd_set_aside_runs": "see runs/llm_partial_ratelimited (rate-limited and wrong-subset GPT calls, excluded)"}

with open("runs/results_summary.json", "w") as f:
    json.dump(OUT, f, indent=2)
lines = ["# Results summary (generated)", "", f"Generated {OUT['generated_utc']} by `python3 results_summary.py` from raw files.", ""]
for name, sec in OUT["sections"].items():
    lines += [f"## {name}", "", "```json", json.dumps(sec, indent=1)[:20000], "```", ""]
lines += ["## spend", "", "```json", json.dumps(OUT["spend"], indent=1), "```"]
open("runs/RESULTS.md", "w").write("\n".join(lines) + "\n")
print("Wrote runs/results_summary.json and runs/RESULTS.md")
