"""
Typo front end for BEMA: correct misspelled words before the model sees them.

Jev holds up under typos far better than BEMA v1 or v2. This tests whether a
cheap, standard pre-processing step closes that gap: SymSpell (MIT license)
spelling correction. A token that is already a real word (in the task's
training text or SymSpell's bundled English dictionary) is never changed;
otherwise it is corrected first against words appearing at least twice in that
task's own TRAINING split, then against the English dictionary. Words of 4 letters or fewer may move by
one edit, longer words by two. Test text is never used to build the
dictionaries.

The model itself is unchanged (bema_v2.pt). Predictions are written as
runs/bema/bema_v2_spell_<task>_<cond>.jsonl, keyed to the ORIGINAL frozen
input (its sha256), and the corrected text the model actually read is saved
alongside in runs/bema/spell_texts/ so every correction can be inspected.

  python3 bema_spell.py
"""
import json
import os
import re
from collections import Counter

import symspellpy
from symspellpy import SymSpell, Verbosity

import bench_jev as B
import bema_v2 as V

TOKEN = re.compile(r"[A-Za-z]+|[^A-Za-z]+")


def build_speller(head):
    """Two tiers: words seen in this task's own training text (domain) first, general English
    second. `head` picks the task: "choice" (BANKING77) or "noul" (SMS spam)."""
    data = B.load_data()
    counts = Counter()
    for head in (head,):
        for r in data[head]["train"]:
            counts.update(w.lower() for w in re.findall(r"[A-Za-z]+", r["text"]))
    domain = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    for w, c in counts.items():
        if c >= 2:   # words seen once in training are often typos themselves
            domain.create_dictionary_entry(w, c)
    english = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    english.load_dictionary(os.path.join(os.path.dirname(symspellpy.__file__), "frequency_dictionary_en_82_765.txt"),
                            term_index=0, count_index=1)
    return domain, english


def correct(speller, text):
    domain, english = speller
    out = []
    for tok in TOKEN.findall(text):
        if not tok.isalpha() or len(tok) < 2:
            out.append(tok)
            continue
        low = tok.lower()
        if domain.lookup(low, Verbosity.TOP, max_edit_distance=0) or \
                english.lookup(low, Verbosity.TOP, max_edit_distance=0):
            out.append(tok)             # already a real word: never "correct" it
            continue
        max_ed = 1 if len(low) <= 4 else 2
        sug = domain.lookup(low, Verbosity.TOP, max_edit_distance=max_ed) or \
            english.lookup(low, Verbosity.TOP, max_edit_distance=max_ed)
        if not sug:
            out.append(tok)
            continue
        fix = sug[0].term
        out.append(fix.capitalize() if tok[0].isupper() else fix)
    return "".join(out)


def main():
    spellers = {"choice": build_speller("choice"), "noul": build_speller("noul")}
    data = B.load_data()
    path, tok = V.load_tok()
    model = V.BemaV2(path, data["num_choice_classes"])
    model.load_state_dict(V.torch.load(V.CKPT, map_location="cpu"))
    model.eval()
    os.makedirs("runs/bema/spell_texts", exist_ok=True)
    for task, spec in B.TASKS.items():
        for cond in spec["conditions"]:
            rows = B.load_inputs(task, cond)
            fixed = [correct(spellers[spec["head"]], r["text"]) for r in rows]
            with open(f"runs/bema/spell_texts/{task}_{cond}.jsonl", "w") as f:
                for r, t in zip(rows, fixed):
                    f.write(json.dumps({"idx": r["idx"], "original": r["text"], "corrected": t}) + "\n")
            if spec["head"] == "choice":
                probs = V.torch.softmax(V.logits_on(model, tok, fixed, "choice") / model.t_choice, -1)
                conf, pred = probs.max(-1)
                out = [{"pred": int(p), "conf": float(c)} for p, c in zip(pred, conf)]
            else:
                p = V.torch.sigmoid(V.logits_on(model, tok, fixed, "noul").squeeze(-1) / model.t_noul)
                out = [{"pred": int(q > 0.5), "conf": float(max(q, 1 - q)), "p_yes": float(q)} for q in p]
            with open(f"runs/bema/bema_v2_spell_{task}_{cond}.jsonl", "w") as f:
                for r, o in zip(rows, out):
                    f.write(json.dumps({"idx": r["idx"], "text_sha256": r["text_sha256"], **o}) + "\n")
            changed = sum(a != r["text"] for a, r in zip(fixed, rows))
            print(f"{task}/{cond}: {changed}/{len(rows)} texts changed by the speller", flush=True)


if __name__ == "__main__":
    main()
