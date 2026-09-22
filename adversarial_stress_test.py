"""
Phase 5b: adversarial / ambiguous-input stress test, extending
ood_stress_test.py's cross-domain distribution-shift test with the other
half of the original Phase 5 plan: "inputs that are ambiguous,
adversarially phrased, or meaningfully different in style from the
training distribution."

Three tests, all built from REAL, correctly-labeled data (no synthetic
or LLM-generated labels, per this project's ground-truth rule):

1. REAL HARD NEGATIVES (no perturbation): real, human-written, genuinely
   ham SMS messages from the held-out test split that happen to contain
   classic spam-trigger words ("free", "win", "prize", "urgent", "call
   now", "click", "$", "!!", "winner"). These are not synthetic -- they
   are real messages people actually sent that happen to look
   superficially spam-like. This tests whether the model is keying on
   superficial lexical cues rather than genuine intent.

2. MECHANICAL ADVERSARIAL PERTURBATION (real base text + real label,
   documented mechanical transformation -- NOT a fresh real-world
   sample, weaker evidence than #1 and labeled as such): real,
   correctly-classified spam test messages run through well-documented,
   deterministic filter-evasion transformations that real spammers
   actually use (character spacing, leetspeak substitution, case noise).
   The base text and its label are real; only the surface form is
   mechanically altered. Measures whether calibration degrades
   gracefully (lower confidence on the now-harder input) or stays
   falsely confident while accuracy drops.

3. TYPO ROBUSTNESS ON IN-DOMAIN QUERIES (Choice/banking77): real,
   correctly-classified banking77 test queries with light character-level
   noise (a few random typos), same rationale as #2 -- real query, real
   label, mechanical perturbation, checking whether small realistic noise
   collapses confidence/accuracy disproportionately.

Uses the same calibrated multi-task model as ood_stress_test.py.
"""
import random
import re

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from model import JevCloneEncoder
from train_multitask import TaskDataset, data, vocab_size, max_len, num_choice_classes, device

random.seed(42)

model = JevCloneEncoder(
    vocab_size=vocab_size, max_len=max_len,
    num_choice_classes=num_choice_classes, enable_score=True,
).to(device)
model.load_state_dict(torch.load("jev_clone_multitask_calibrated.pt", map_location=device))
model.eval()

tokenizer = Tokenizer.from_file("data/tokenizer_multitask.json")
pad_id = data["pad_id"]
categories = data["choice_categories"]


def encode(text):
    ids = tokenizer.encode(text).ids[:max_len]
    mask = [1] * len(ids)
    pad_n = max_len - len(ids)
    return ids + [pad_id] * pad_n, mask + [0] * pad_n


# ---------------------------------------------------------------------------
# Test 1: real hard negatives -- real ham messages containing spam-trigger words
# ---------------------------------------------------------------------------
SPAM_TRIGGER_WORDS = ["free", "win", "won", "prize", "urgent", "call now", "click",
                      "$", "!!", "winner", "cash", "claim", "congratulations"]


def looks_spammy(text):
    t = text.lower()
    return any(w in t for w in SPAM_TRIGGER_WORDS)


noul_test_rows = data["noul"]["test"]
hard_ham = [r for r in noul_test_rows if r["label"] == 0 and looks_spammy(r["text"])]
easy_ham = [r for r in noul_test_rows if r["label"] == 0 and not looks_spammy(r["text"])]

print(f"Real ham test messages containing spam-trigger words: {len(hard_ham)} of "
      f"{sum(1 for r in noul_test_rows if r['label'] == 0)} total ham")


def eval_noul_rows(rows):
    if not rows:
        return None, None
    loader = DataLoader(TaskDataset(rows, torch.float), batch_size=64)
    correct, n, confs = 0, 0, []
    with torch.no_grad():
        for ids, mask, label in loader:
            ids, mask = ids.to(device), mask.to(device)
            logit = model.forward(ids, mask) / model.temperature
            prob = torch.sigmoid(logit)
            pred = (prob > 0.5).float().cpu()
            conf = torch.where(pred == 1, prob, 1 - prob).cpu()
            correct += (pred == label).sum().item()
            n += label.size(0)
            confs.extend(conf.tolist())
    return correct / n, sum(confs) / len(confs)


hard_acc, hard_conf = eval_noul_rows(hard_ham)
easy_acc, easy_conf = eval_noul_rows(easy_ham)

print("=" * 70)
print("TEST 1: Real ham messages that superficially look spammy (real hard negatives)")
print("=" * 70)
if hard_acc is not None:
    print(f"Spam-trigger-word ham   n={len(hard_ham):4d}  accuracy={hard_acc:.4f}  mean_confidence={hard_conf:.4f}")
print(f"Other ham               n={len(easy_ham):4d}  accuracy={easy_acc:.4f}  mean_confidence={easy_conf:.4f}")
if hard_acc is not None:
    print(f"Accuracy on lexically-spammy-looking real ham is "
          f"{'LOWER' if hard_acc < easy_acc else 'similar or higher'} than on other ham "
          f"({hard_acc:.4f} vs {easy_acc:.4f}) -- "
          f"{'the model is measurably fooled by surface lexical cues on real messages' if hard_acc < easy_acc - 0.02 else 'the model is not obviously fooled by surface lexical cues alone'}.")

# ---------------------------------------------------------------------------
# Test 2: mechanical adversarial perturbation of real, correctly-classified spam
# ---------------------------------------------------------------------------
LEET_MAP = {"o": "0", "i": "1", "e": "3", "a": "4", "s": "5"}


def perturb_spacing(text):
    """'free' -> 'f r e e' on a random subset of words -- classic filter evasion."""
    words = text.split()
    out = []
    for w in words:
        if len(w) > 3 and random.random() < 0.3:
            out.append(" ".join(list(w)))
        else:
            out.append(w)
    return " ".join(out)


def perturb_leet(text):
    return "".join(LEET_MAP.get(c.lower(), c) if random.random() < 0.4 else c for c in text)


def perturb_case(text):
    return "".join(c.upper() if random.random() < 0.3 else c for c in text)


def perturb(text):
    t = perturb_spacing(text)
    t = perturb_leet(t)
    t = perturb_case(t)
    return t


with torch.no_grad():
    real_spam = [r for r in noul_test_rows if r["label"] == 1]
    real_correct, real_confs = 0, []
    adv_correct, adv_confs = 0, []
    for r in real_spam:
        # Real, unperturbed
        ids, mask = r["ids"], r["mask"]
        ids_t = torch.tensor([ids], dtype=torch.long).to(device)
        mask_t = torch.tensor([mask], dtype=torch.long).to(device)
        logit = model.forward(ids_t, mask_t) / model.temperature
        prob = torch.sigmoid(logit).item()
        pred = prob > 0.5
        real_correct += int(pred == True)
        real_confs.append(prob if pred else 1 - prob)

        # Mechanically perturbed
        adv_text = perturb(r["text"])
        ids2, mask2 = encode(adv_text)
        ids2_t = torch.tensor([ids2], dtype=torch.long).to(device)
        mask2_t = torch.tensor([mask2], dtype=torch.long).to(device)
        logit2 = model.forward(ids2_t, mask2_t) / model.temperature
        prob2 = torch.sigmoid(logit2).item()
        pred2 = prob2 > 0.5
        adv_correct += int(pred2 == True)
        adv_confs.append(prob2 if pred2 else 1 - prob2)

n_spam = len(real_spam)
print("\n" + "=" * 70)
print("TEST 2: Mechanical filter-evasion perturbation of real spam (spacing/leetspeak/case noise)")
print("=" * 70)
print(f"Real spam, unperturbed   n={n_spam}  accuracy={real_correct/n_spam:.4f}  mean_confidence={sum(real_confs)/n_spam:.4f}")
print(f"Same spam, perturbed     n={n_spam}  accuracy={adv_correct/n_spam:.4f}  mean_confidence={sum(adv_confs)/n_spam:.4f}")
print(f"Accuracy drop: {real_correct/n_spam - adv_correct/n_spam:.4f}. "
      f"{'Confidence dropped appropriately alongside accuracy.' if sum(adv_confs)/n_spam < sum(real_confs)/n_spam - 0.02 else 'Confidence did NOT drop proportionally to the accuracy loss -- a calibration gap under adversarial perturbation.'}")

# ---------------------------------------------------------------------------
# Test 3: typo robustness on real, correctly-classified banking77 queries
# ---------------------------------------------------------------------------
def perturb_typo(text, n_typos=2):
    chars = list(text)
    for _ in range(n_typos):
        if len(chars) < 3:
            break
        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]  # adjacent-swap typo
    return "".join(chars)


choice_test_rows = data["choice"]["test"]
with torch.no_grad():
    base_correct, base_confs = 0, []
    typo_correct, typo_confs = 0, []
    for r in choice_test_rows:
        ids_t = torch.tensor([r["ids"]], dtype=torch.long).to(device)
        mask_t = torch.tensor([r["mask"]], dtype=torch.long).to(device)
        pooled = model.encode(ids_t, mask_t)
        logits = model.choice_head(pooled) / model.choice_temperature
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        base_correct += int(pred.item() == r["label"])
        base_confs.append(conf.item())

        typo_text = perturb_typo(r["text"])
        ids2, mask2 = encode(typo_text)
        ids2_t = torch.tensor([ids2], dtype=torch.long).to(device)
        mask2_t = torch.tensor([mask2], dtype=torch.long).to(device)
        pooled2 = model.encode(ids2_t, mask2_t)
        logits2 = model.choice_head(pooled2) / model.choice_temperature
        probs2 = torch.softmax(logits2, dim=-1)
        conf2, pred2 = probs2.max(dim=-1)
        typo_correct += int(pred2.item() == r["label"])
        typo_confs.append(conf2.item())

n_choice = len(choice_test_rows)
print("\n" + "=" * 70)
print("TEST 3: Typo robustness on real banking77 queries (2 adjacent-char swaps per query)")
print("=" * 70)
print(f"Clean queries    n={n_choice}  accuracy={base_correct/n_choice:.4f}  mean_confidence={sum(base_confs)/n_choice:.4f}")
print(f"With typos       n={n_choice}  accuracy={typo_correct/n_choice:.4f}  mean_confidence={sum(typo_confs)/n_choice:.4f}")
print(f"Accuracy drop: {base_correct/n_choice - typo_correct/n_choice:.4f}")
