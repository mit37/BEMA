"""
Scores every BEMA-U model (prediction files named *bu_*) against Jev on the frozen
inputs, from the raw files. Writes runs/universal/report.json and prints a table.

Also measures how "unseen" the test categories really are: for every BANKING77,
CLINC150 and finance-topic label, the closest training label (cosine similarity of
off-the-shelf BGE-base embeddings of the names), and accuracy on the test items whose
label has no close training analogue.

  python3 universal_report.py [--root runs]
"""
import glob
import json
import os
import sys

import bench_jev as B
import redteam_jev as R

ROOT = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else "runs"
CLOSE = 0.85   # cosine at or above this = the test label has a close training analogue


def models():
    return sorted({os.path.basename(p)[:-len("_banking77_clean.jsonl")]
                   for p in glob.glob(f"{ROOT}/bema/bu_*_banking77_clean.jsonl")})


def core(m, task, cond):
    rows = {r["idx"]: r for r in B.load_inputs(task, cond)}
    cats = B.categories()
    out = {}
    for r in B.read_jsonl(f"{ROOT}/bema/{m}_{task}_{cond}.jsonl"):
        assert r["text_sha256"] == rows[r["idx"]]["text_sha256"]
        p = cats[r["pred"]] if task == "banking77" else r["pred"]
        out[r["idx"]] = (float(p == B.gold(task, rows[r["idx"]]["label"])), r["conf"], p)
    return out


def jev_core(task, cond):
    rows = {r["idx"]: r for r in B.load_inputs(task, cond)}
    out = {}
    for r in B.read_jsonl(f"runs/jev/{task}_{cond}.jsonl"):
        if r["status"] != "ok":
            continue
        a = list(r["response"]["answers"].values())[0]
        if task == "banking77":
            p = a["choice"]
            c = a["confidence"]
        else:
            q = a["noul"]
            p, c = int(q > 0.5), max(q, 1 - q)
        out[r["idx"]] = (float(p == B.gold(task, rows[r["idx"]]["label"])), c, p)
    return out


def rt(m, test):
    p = f"{ROOT}/redteam/bema/{test}__{m}.jsonl"
    if not os.path.exists(p):
        return None, None
    items = {it["key"]: it for it in B.read_jsonl(f"runs/redteam/inputs/{test}.jsonl")}
    recs = {}
    for r in B.read_jsonl(p):
        assert r["state_sha256"] == items[r["key"]]["state_sha256"]
        recs[r["key"]] = r
    return recs, items


def acc_ece(d):
    v = list(d.values())
    return (round(100 * sum(a for a, _, _ in v) / len(v), 1),
            round(B.ece_exact([c for _, c, _ in v], [a for a, _, _ in v]), 4), len(v))


def model_report(m):
    rep = {"banking77": {}, "spam": {}}
    for cond in B.TASKS["banking77"]["conditions"]:
        a, e, n = acc_ece(core(m, "banking77", cond))
        rep["banking77"][cond] = {"accuracy_pct": a, "ece": e, "n": n}
    jc, mc = jev_core("banking77", "clean"), core(m, "banking77", "clean")
    diffs = [mc[k][0] - jc[k][0] for k in sorted(set(mc) & set(jc))]
    lo, hi = B.boot_ci(diffs)
    rep["banking77_clean_minus_jev"] = {"points": round(100 * sum(diffs) / len(diffs), 1),
                                        "ci95": [round(100 * lo, 1), round(100 * hi, 1)], "n": len(diffs)}
    for cond in B.TASKS["spam"]["conditions"]:
        if os.path.exists(f"{ROOT}/bema/{m}_spam_{cond}.jsonl"):
            a, e, n = acc_ece(core(m, "spam", cond))
            rep["spam"][cond] = {"accuracy_pct": a, "ece": e, "n": n}
    for test in ("clinc150_clean", "clinc150_swap2"):
        recs, items = rt(m, test)
        if not recs:
            continue
        ins = [k for k in recs if items[k]["gold"] != "oos"]
        oos = [k for k in recs if items[k]["gold"] == "oos"]
        if not ins or not oos:
            continue
        rep[test] = {"accuracy_pct": round(100 * sum(recs[k]["pred"] == items[k]["gold"] for k in recs) / len(recs), 1),
                     "in_scope_accuracy_pct": round(100 * sum(recs[k]["pred"] == items[k]["gold"] for k in ins) / len(ins), 1),
                     "oos_recall_pct": round(100 * sum(recs[k]["pred"] == "oos" for k in oos) / len(oos), 1),
                     "false_oos_pct": round(100 * sum(recs[k]["pred"] == "oos" for k in ins) / len(ins), 1),
                     "ece": round(B.ece_exact([recs[k]["conf"] for k in recs],
                                              [float(recs[k]["pred"] == items[k]["gold"]) for k in recs]), 4),
                     "n": len(recs)}
    inj = {}
    for label, test in [("explicit_note_end", "injection_banking"), ("explicit_note_start", "injection_prefix_banking"),
                        ("casual_ps", "injection_casual_banking"), ("neutral_note_control", "injection_control_banking")]:
        recs, items = rt(m, test)
        if not recs:
            continue
        e = {"accuracy_pct": round(100 * sum(recs[k]["pred"] == items[k]["gold"] for k in recs) / len(recs), 1)}
        if items[next(iter(recs))]["meta"].get("target"):
            e["followed_pct"] = round(100 * sum(recs[k]["pred"] == items[k]["meta"]["target"] for k in recs) / len(recs), 1)
        inj[label] = e
    rep["injection_1000"] = inj
    recs, items = rt(m, "injection_oos_clinc")
    if recs:
        rep["injection_oos_clinc_said_oos_pct"] = round(100 * sum(recs[k]["pred"] == "oos" for k in recs) / len(recs), 1)
    for test in ("ood_spam_on_banking", "evasion_spam", "injection_spam"):
        recs, items = rt(m, test)
        if recs:
            rep[test] = {"accuracy_pct": round(100 * sum(recs[k]["pred"] == items[k]["gold"] for k in recs) / len(recs), 1)}
    recs, items = rt(m, "gibberish_banking")
    if recs:
        rep["gibberish_mean_confidence"] = round(sum(r["conf"] for r in recs.values()) / len(recs), 4)
    return rep


def overlap():
    """Closest training label for each test label, by BGE-base cosine of the names."""
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    path = snapshot_download("BAAI/bge-base-en-v1.5", allow_patterns=["*.json", "*.txt", "*.safetensors"])
    tok, mdl = AutoTokenizer.from_pretrained(path), AutoModel.from_pretrained(path).eval()

    def emb(xs):
        with torch.no_grad():
            x = tok(xs, padding=True, truncation=True, max_length=32, return_tensors="pt")
            return F.normalize(mdl(**x).last_hidden_state[:, 0], dim=-1)
    LS = json.load(open("data/universal/label_sets.json"))
    train = sorted({f"{v['names'][l]}" for v in LS.values() for l in v["labels"]})
    te = emb(train)
    tests = {"banking77": list(B.question_spec("banking77")["intent"]["criteria"].items()),
             "clinc150": [(k, v) for k, v in R.questions_for({"qset": "clinc"})["intent"]["criteria"].items() if k != "oos"],
             "twitter_fin_topic": [(c, c) for c in json.load(open("runs/universal/inputs/twitter_fin_topic_categories.json"))["categories"]]}
    out = {}
    for name, labs in tests.items():
        sims = emb([v for _, v in labs]) @ te.T
        best, idx = sims.max(-1)
        out[name] = {k: {"closest_training_label": train[i], "cosine": round(float(b), 3)}
                     for (k, _), b, i in zip(labs, best.tolist(), idx.tolist())}
    return out


def main():
    ms = models()
    rep = {"models": {m: model_report(m) for m in ms}}
    ov = overlap()
    rep["label_overlap"] = {n: {"n_labels": len(v), "n_close": sum(x["cosine"] >= CLOSE for x in v.values()),
                                "threshold": CLOSE, "labels": v} for n, v in ov.items()}
    # accuracy restricted to test items whose label has no close training analogue
    far_bank = {k for k, x in ov["banking77"].items() if x["cosine"] < CLOSE}
    rows = {r["idx"]: r for r in B.load_inputs("banking77", "clean")}
    cats = B.categories()
    sub = {}
    jc = jev_core("banking77", "clean")
    keys = [i for i in rows if cats[rows[i]["label"]] in far_bank]
    sub["jev"] = round(100 * sum(jc[i][0] for i in keys if i in jc) / len(keys), 1)
    for m in ms:
        mc = core(m, "banking77", "clean")
        ks = [i for i in keys if i in mc]
        sub[m] = round(100 * sum(mc[i][0] for i in ks) / len(ks), 1) if ks else None
    rep["banking77_clean_no_close_training_label"] = {"n_items": len(keys), "n_labels": len(far_bank), "accuracy_pct": sub}
    os.makedirs(f"{ROOT}/universal", exist_ok=True)
    json.dump(rep, open(f"{ROOT}/universal/report.json", "w"), indent=1)
    for m in ms:
        r = rep["models"][m]
        print(m, "BANKING77", r["banking77"]["clean"], "vs Jev", r["banking77_clean_minus_jev"],
              "| CLINC", r.get("clinc150_clean"), "| inj", r["injection_1000"])
    print("label overlap:", {n: (v["n_close"], v["n_labels"]) for n, v in rep["label_overlap"].items()})
    print("BANKING77 items with no close training label:", rep["banking77_clean_no_close_training_label"])


if __name__ == "__main__":
    main()
