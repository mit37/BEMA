"""
Training data for BEMA-U: one model that answers "which of these options?" for
option sets it has never seen.

It pools ~14 public labelled text-classification datasets from Hugging Face
(intents, topics, sentiment, spam, toxicity, question types), each with its own
label set. The model learns to match text to a description of each option, so at
test time any new option list works without retraining.

Held out completely (never trained on, and every training text that also
appears in them, case- and whitespace-normalised, is removed):
  BANKING77  (77 banking intents)      -> the frozen inputs Jev already answered
  CLINC150   (150 intents + out of scope) -> the frozen inputs Jev already answered
  Twitter Financial News Topic (20 finance topics) -> a new held-out finance test
So the main comparison with Jev needs no new Jev calls.

  python3 universal_data.py            # writes data/universal/ (+ manifest.json)

Sources are pinned: the dataset's parquet conversion is downloaded and its
revision recorded in data/universal/label_sets.json.
"""
import hashlib
import json
import os
import random
import re
import sys

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

OUT = "data/universal"
CAP = int(os.environ.get("U_CAP", "20000"))       # max training rows per dataset
VAL_PER_SET = 300
SEED = 2026
api = HfApi()


def readable(name):
    name = str(name)
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)      # AddToPlaylist -> Add To Playlist
    name = name.replace("_", " ").replace("+", " and ").strip()
    return re.sub(r"\s+", " ", name).lower() if not name.isupper() else name


def norm(t):
    return " ".join(str(t).lower().split())


def parquet(repo, config, split):
    files = [f for f in api.list_repo_files(repo, repo_type="dataset", revision="refs/convert/parquet")
             if f.startswith(f"{config}/{split}/") and f.endswith(".parquet")]
    if not files:
        sys.exit(f"{repo}: no parquet for {config}/{split}")
    dfs = [pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset", revision="refs/convert/parquet"))
           for f in sorted(files)]
    return pd.concat(dfs, ignore_index=True)


def revision(repo):
    return api.dataset_info(repo, revision="refs/convert/parquet").sha


def intent_names(repo):
    df = parquet(repo, "intents", "intents")
    return {str(r["id"]): r["name"] for _, r in df.iterrows()}


# Each loader returns (rows, labels, instruction): rows are (text, raw_label), split -> rows
def deeppavlov(repo, instruction):
    names = intent_names(repo)
    out = {}
    for split in ("train", "validation", "test"):
        try:
            df = parquet(repo, "default", split)
        except SystemExit:
            continue
        out[split] = [(r["utterance"], names[str(r["label"])]) for _, r in df.iterrows()]
    return out, sorted(set(names.values())), instruction


def simple(repo, config, text_fn, label_fn, splits, instruction, labels=None):
    out = {}
    for split in splits:
        df = parquet(repo, config, split)
        out[split] = [(text_fn(r), label_fn(r)) for _, r in df.iterrows()]
    labs = labels or sorted({l for rows in out.values() for _, l in rows})
    return out, labs, instruction


AG = ["World", "Sports", "Business", "Sci/Tech"]
DBPEDIA = ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation",
           "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"]
YAHOO = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet",
         "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]
EMOTION = ["sadness", "joy", "love", "anger", "fear", "surprise"]
TW_SENT = {0: "Bearish", 1: "Bullish", 2: "Neutral"}                     # dataset card, LABEL_0..2
TW_TOPIC = ["Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt",
            "Dividend", "Earnings", "Energy | Oil", "Financials", "Currencies", "General News | Opinion",
            "Gold | Metals | Materials", "IPO", "Legal | Regulation", "M&A | Investments", "Macro", "Markets",
            "Politics", "Personnel Change", "Stock Commentary", "Stock Movement"]   # dataset card, LABEL_0..19


def clip(t, n=400):
    t = " ".join(str(t).split())
    return t[:n]


def sources():
    S = {}
    S["hwu64"] = ("DeepPavlov/hwu64",) + deeppavlov("DeepPavlov/hwu64", "Which request to a home voice assistant is this?")
    S["massive"] = ("DeepPavlov/massive",) + deeppavlov("DeepPavlov/massive", "Which request to a voice assistant is this?")
    S["snips"] = ("DeepPavlov/snips",) + deeppavlov("DeepPavlov/snips", "Which request to a voice assistant is this?")
    S["atis"] = ("tuetschek/atis",) + simple("tuetschek/atis", "default", lambda r: r["text"], lambda r: r["intent"],
                                             ("train", "test"), "What is this airline-travel query asking for?")
    S["bitext_support"] = ("bitext/Bitext-customer-support-llm-chatbot-training-dataset",) + simple(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset", "default",
        lambda r: r["instruction"], lambda r: r["intent"], ("train",),
        "Which customer-support request is this?")
    S["ag_news"] = ("fancyzhx/ag_news",) + simple("fancyzhx/ag_news", "default", lambda r: clip(r["text"]),
                                                  lambda r: AG[int(r["label"])], ("train", "test"),
                                                  "What is this news article about?", AG)
    S["dbpedia"] = ("fancyzhx/dbpedia_14",) + simple("fancyzhx/dbpedia_14", "dbpedia_14",
                                                     lambda r: clip(f"{r['title']}. {r['content']}"),
                                                     lambda r: DBPEDIA[int(r["label"])], ("train", "test"),
                                                     "What kind of thing does this encyclopedia entry describe?", DBPEDIA)
    S["yahoo"] = ("community-datasets/yahoo_answers_topics",) + simple(
        "community-datasets/yahoo_answers_topics", "yahoo_answers_topics",
        lambda r: clip(f"{r['question_title']} {r['question_content'] or ''}"),
        lambda r: YAHOO[int(r["topic"])], ("train", "test"), "Which topic is this question about?", YAHOO)
    S["emotion"] = ("dair-ai/emotion",) + simple("dair-ai/emotion", "split", lambda r: r["text"],
                                                 lambda r: EMOTION[int(r["label"])], ("train", "validation"),
                                                 "Which emotion does this message express?", EMOTION)
    S["trec"] = ("SetFit/TREC-QC",) + simple("SetFit/TREC-QC", "default", lambda r: r["text"],
                                             lambda r: r["label_text"], ("train", "test"),
                                             "What kind of answer is this question asking for?")
    S["twitter_fin_sentiment"] = ("zeroshot/twitter-financial-news-sentiment",) + simple(
        "zeroshot/twitter-financial-news-sentiment", "default", lambda r: r["text"],
        lambda r: TW_SENT[int(r["label"])], ("train", "validation"),
        "What is the market sentiment of this finance tweet?", list(TW_SENT.values()))
    S["enron_spam"] = ("SetFit/enron_spam",) + simple("SetFit/enron_spam", "default",
                                                      lambda r: clip(f"{r['subject']} {r['message']}"),
                                                      lambda r: "spam" if r["label_text"] == "spam" else
                                                      "not spam (a legitimate email)", ("train", "test"),
                                                      "Is this email spam?")
    S["toxic"] = ("mteb/toxic_conversations_50k",) + simple("mteb/toxic_conversations_50k", "default",
                                                            lambda r: clip(r["text"]), lambda r: r["label_text"],
                                                            ("train", "test"), "Is this comment toxic?")
    S["sst5"] = ("SetFit/sst5",) + simple("SetFit/sst5", "default", lambda r: r["text"], lambda r: r["label_text"],
                                          ("train", "validation"), "What is the sentiment of this movie review?")
    return S


def heldout_texts():
    """Every text from the held-out datasets, normalised, so training can exclude them."""
    sys.path.insert(0, ".")
    import bench_jev as B
    import redteam_jev as R
    held = set()
    d = B.load_data()
    for split in ("train", "val", "test"):
        held |= {norm(r["text"]) for r in d["choice"][split]}
    for split, rows in R.clinc_raw().items():
        held |= {norm(t) for t, _ in rows}
    tw = {}
    for split in ("train", "validation"):
        df = parquet("zeroshot/twitter-financial-news-topic", "default", split)
        tw[split] = [(r["text"], TW_TOPIC[int(r["label"])]) for _, r in df.iterrows()]
        held |= {norm(t) for t, _ in tw[split]}
    return held, tw


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    rnd = random.Random(SEED)
    held, tw = heldout_texts()
    label_sets, train, val, report = {}, [], [], {}
    seen = set()
    for name, (repo, splits, labels, instruction) in sources().items():
        tr = splits.get("train", [])
        ev = splits.get("validation") or splits.get("test") or []
        before = len(tr)
        tr = [(t, l) for t, l in tr if isinstance(t, str) and t.strip() and norm(t) not in held]
        removed_heldout = before - len(tr)
        uniq = []
        for t, l in tr:
            k = norm(t)
            if k in seen:
                continue
            seen.add(k)
            uniq.append((t, l))
        rnd.shuffle(uniq)
        uniq = uniq[:CAP]
        ev = [(t, l) for t, l in ev if isinstance(t, str) and t.strip() and norm(t) not in held and norm(t) not in seen]
        rnd.shuffle(ev)
        ev = ev[:VAL_PER_SET]
        used = sorted({l for _, l in uniq} | {l for _, l in ev})
        label_sets[name] = {"source": repo, "revision": revision(repo), "instruction": instruction,
                            "labels": used, "names": {l: readable(l) for l in used}}
        train += [{"text": t, "label": l, "set": name} for t, l in uniq]
        val += [{"text": t, "label": l, "set": name} for t, l in ev]
        report[name] = {"train_rows": len(uniq), "val_rows": len(ev), "labels": len(used),
                        "removed_as_heldout_overlap": removed_heldout}
        print(f"{name:24s} train {len(uniq):6d}  val {len(ev):4d}  labels {len(used):3d}  "
              f"removed (held-out overlap) {removed_heldout}", flush=True)
    rnd.shuffle(train)
    with open(f"{OUT}/train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{OUT}/val.jsonl", "w") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(label_sets, open(f"{OUT}/label_sets.json", "w"), indent=1, ensure_ascii=False)
    # the new held-out finance test: frozen inputs with fingerprints, like runs/inputs
    os.makedirs("runs/universal/inputs", exist_ok=True)
    with open("runs/universal/inputs/twitter_fin_topic.jsonl", "w") as f:
        for i, (t, l) in enumerate(tw["validation"]):
            f.write(json.dumps({"idx": i, "text": t, "label": TW_TOPIC.index(l),
                                "text_sha256": hashlib.sha256(t.encode()).hexdigest()}, ensure_ascii=False) + "\n")
    json.dump({"categories": TW_TOPIC, "source": "zeroshot/twitter-financial-news-topic",
               "split": "validation", "revision": revision("zeroshot/twitter-financial-news-topic")},
              open("runs/universal/inputs/twitter_fin_topic_categories.json", "w"), indent=1)
    json.dump(report, open(f"{OUT}/report.json", "w"), indent=1)
    man = {p: {"sha256": sha(p), "lines": sum(1 for _ in open(p))} for p in
           [f"{OUT}/train.jsonl", f"{OUT}/val.jsonl", "runs/universal/inputs/twitter_fin_topic.jsonl"]}
    man[f"{OUT}/label_sets.json"] = {"sha256": sha(f"{OUT}/label_sets.json")}
    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
    print(f"train {len(train)} rows over {len(label_sets)} datasets, "
          f"{sum(len(v['labels']) for v in label_sets.values())} labels; val {len(val)}; "
          f"held-out finance test {len(tw['validation'])}")


if __name__ == "__main__":
    main()
