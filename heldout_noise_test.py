"""
Does typo-noise augmentation (augment_and_retrain.py) generalize to typo
types it never saw in training?

The augmented model was trained on exactly one corruption: 2 adjacent-
character swaps. FINDINGS.md flagged that its improvement was only ever
measured against that same corruption. This script evaluates the
baseline and augmented Choice heads (both temperature-calibrated
checkpoints, BANKING77 test split, n=3080) on the training-time noise
plus four held-out noise types:

  swap2      2 adjacent-character swaps          (seen in training)
  swap4      4 adjacent-character swaps          (same kind, heavier)
  keyboard2  2 QWERTY-neighbor substitutions     (held out)
  delete2    2 deleted letters                   (held out)
  insert2    2 inserted random letters           (held out)
  mixed3     1 substitution + 1 deletion + 1 insertion (held out)

Each noise type is generated once with a fixed seed, so both models see
byte-identical perturbed text. Reports accuracy, mean confidence,
overconfidence (mean confidence - accuracy) and ECE per condition.

Usage: python3 heldout_noise_test.py  (needs jev_clone_multitask_calibrated.pt
and jev_clone_multitask_augmented_calibrated.pt)
"""
import pickle
import random

import torch
from tokenizers import Tokenizer

from calib_utils import ece as ece_score
from model import JevCloneEncoder
from noise import HELD_OUT, NOISE, perturb
from train_multitask import data, device, max_len, num_choice_classes, vocab_size

SEED = 99

tokenizer = Tokenizer.from_file("data/tokenizer_multitask.json")
pad_id = data["pad_id"]


def encode_batch(texts):
    ids, masks = [], []
    for t in texts:
        tok = tokenizer.encode(t).ids[:max_len]
        pad = max_len - len(tok)
        ids.append(tok + [pad_id] * pad)
        masks.append([1] * len(tok) + [0] * pad)
    return torch.tensor(ids), torch.tensor(masks)


def perturbed(texts, name):
    """Apply noise type `name` with a fixed per-type seed, so every caller sees identical text."""
    rnd = random.Random(f"{SEED}-{name}")
    return [perturb(t, NOISE[name], rnd) for t in texts]


def load(path):
    m = JevCloneEncoder(vocab_size=vocab_size, max_len=max_len,
                        num_choice_classes=num_choice_classes, enable_score=True).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def evaluate(model, texts, labels):
    ids, mask = encode_batch(texts)
    confs, preds = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            _, conf, probs = model.predict_choice(ids[i:i + 256].to(device), mask[i:i + 256].to(device))
            confs.append(conf.cpu())
            preds.append(probs.argmax(dim=-1).cpu())
    conf, pred = torch.cat(confs), torch.cat(preds)
    correct = (pred == labels).float()
    acc, mean_conf = correct.mean().item(), conf.mean().item()
    return {"acc": acc, "conf": mean_conf, "overconf": mean_conf - acc,
            "ece": ece_score(conf.tolist(), correct.tolist())}


if __name__ == "__main__":
    rows = data["choice"]["test"]
    texts = [r["text"] for r in rows]
    labels = torch.tensor([r["label"] for r in rows])

    conditions = {"clean": texts}
    for name in NOISE:
        conditions[name] = perturbed(texts, name)

    models = {"baseline": load("jev_clone_multitask_calibrated.pt"),
              "augmented": load("jev_clone_multitask_augmented_calibrated.pt")}
    results = {(m, c): evaluate(models[m], conditions[c], labels) for m in models for c in conditions}

    print(f"Choice head (BANKING77 test, n={len(rows)}). Augmented model was trained on swap2 only.\n")
    print(f"{'Noise':<11}{'held out?':<11}{'Baseline acc/conf/ECE':>26}{'Augmented acc/conf/ECE':>28}{'acc gain':>10}")
    for c in conditions:
        b, a = results[("baseline", c)], results[("augmented", c)]
        tag = "yes" if c in HELD_OUT else ("-" if c == "clean" else "no")
        print(f"{c:<11}{tag:<11}"
              f"{b['acc']:>10.4f} {b['conf']:.4f} {b['ece']:.4f}"
              f"{a['acc']:>12.4f} {a['conf']:.4f} {a['ece']:.4f}"
              f"{a['acc'] - b['acc']:>+10.4f}")

    print("\nOverconfidence (mean confidence - accuracy; 0 is perfectly calibrated on average):")
    for c in conditions:
        b, a = results[("baseline", c)], results[("augmented", c)]
        print(f"  {c:<11} baseline {b['overconf']:+.4f}   augmented {a['overconf']:+.4f}")

    print("\nExample perturbations of the first test query:")
    for c in conditions:
        print(f"  {c:<11} {conditions[c][0]!r}")

    with open("heldout_noise_results.pkl", "wb") as f:
        pickle.dump({"n": len(rows), "results": results}, f)
    print("\nSaved heldout_noise_results.pkl")
