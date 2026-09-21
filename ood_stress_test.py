"""
Phase 5: stress-test the "cannot hallucinate" / calibration claims under
distribution shift, using REAL data only -- no synthetic or LLM-generated
labels, per the project's ground-truth rule.

Two honest, real-data OOD tests:

1. CROSS-DOMAIN TRANSFER (Noul/spam head): evaluate the spam classifier on
   BANKING77 test utterances. These are real, human-written customer
   service questions ("How do I locate my card?"), never labeled for spam
   detection -- but their provenance (a curated customer-support intent
   dataset) makes "not spam" the only defensible ground truth for every
   example in it: nobody writes a genuine customer-support question as
   unsolicited bulk sales spam. This is a REAL, human-written negative
   set out of the spam model's training distribution, not a synthetic one.

2. CROSS-DOMAIN CONFIDENT-WRONGNESS (Choice/banking77 head): evaluate the
   banking-intent classifier on real SMS spam/ham text (from the Noul
   task's test split), which is NOT a banking query at all. There is no
   correct banking77 category for "WINNER!! You've won a prize" -- so
   there's no accuracy to report here. What we measure instead is whether
   the model still emits a single, structurally valid, *confident* typed
   answer for clearly out-of-scope input. That's the actual "cannot
   hallucinate" claim under stress: the model can never emit a malformed
   answer, but it can -- and does -- confidently pick a wrong-domain label
   for input its schema was never meant to cover. That distinction is the
   finding, not a bug in this script.

Both tests use the SAME calibrated multi-task model, same shared encoder.
"""
import pickle
import torch
from torch.utils.data import DataLoader

from model import JevCloneEncoder
from train_multitask import TaskDataset, data, vocab_size, max_len, num_choice_classes, device

model = JevCloneEncoder(
    vocab_size=vocab_size, max_len=max_len,
    num_choice_classes=num_choice_classes, enable_score=True,
).to(device)
model.load_state_dict(torch.load("jev_clone_multitask_calibrated.pt", map_location=device))
model.eval()

categories = data["choice_categories"]

# ---------------------------------------------------------------------------
# In-distribution baseline (for comparison): Noul on its own real test split
# ---------------------------------------------------------------------------
noul_test_loader = DataLoader(TaskDataset(data["noul"]["test"], torch.float), batch_size=64)
in_dist_correct, in_dist_n = 0, 0
in_dist_confs = []
with torch.no_grad():
    for ids, mask, label in noul_test_loader:
        ids, mask = ids.to(device), mask.to(device)
        logit = model.forward(ids, mask) / model.temperature
        prob = torch.sigmoid(logit)
        pred = (prob > 0.5).float().cpu()
        conf = torch.where(pred == 1, prob, 1 - prob).cpu()
        in_dist_correct += (pred == label).sum().item()
        in_dist_n += label.size(0)
        in_dist_confs.extend(conf.tolist())

print("=" * 70)
print("TEST 1: Noul (spam) head on IN-DISTRIBUTION real test data (baseline)")
print("=" * 70)
print(f"n={in_dist_n}  accuracy={in_dist_correct/in_dist_n:.4f}  mean_confidence={sum(in_dist_confs)/len(in_dist_confs):.4f}")

# ---------------------------------------------------------------------------
# OOD Test 1: Noul (spam) head on real BANKING77 utterances (known-ham by
# dataset construction -- genuine customer support text, never spam)
# ---------------------------------------------------------------------------
banking_all = data["choice"]["test"] + data["choice"]["val"]
banking_loader = DataLoader(TaskDataset(banking_all, torch.long), batch_size=64)

ood_correct, ood_n = 0, 0
ood_confs, false_spam_examples = [], []
with torch.no_grad():
    for ids, mask, label in banking_loader:
        ids, mask = ids.to(device), mask.to(device)
        logit = model.forward(ids, mask) / model.temperature
        prob = torch.sigmoid(logit)
        pred = (prob > 0.5)
        conf = torch.where(pred, prob, 1 - prob).cpu()
        # ground truth here is always "not spam" (label = 0) by dataset construction
        correct = (~pred).cpu()
        ood_correct += correct.sum().item()
        ood_n += pred.size(0)
        ood_confs.extend(conf.tolist())

print("\n" + "=" * 70)
print("TEST 2: Noul (spam) head on OOD real data (BANKING77 utterances, known non-spam)")
print("=" * 70)
print(f"n={ood_n}  accuracy(=1-false_spam_rate)={ood_correct/ood_n:.4f}  "
      f"mean_confidence={sum(ood_confs)/len(ood_confs):.4f}")
print(f"In-distribution accuracy was {in_dist_correct/in_dist_n:.4f} with mean confidence "
      f"{sum(in_dist_confs)/len(in_dist_confs):.4f} -- compare directly above.")

# ---------------------------------------------------------------------------
# OOD Test 2: Choice (banking77) head on real SMS spam/ham text (no correct
# banking77 answer exists for this input -- measures confident-wrongness,
# not accuracy)
# ---------------------------------------------------------------------------
spam_all = data["noul"]["test"] + data["noul"]["val"]
spam_loader = DataLoader(TaskDataset(spam_all, torch.float), batch_size=64)

choice_confs_on_junk = []
with torch.no_grad():
    for ids, mask, label in spam_loader:
        ids, mask = ids.to(device), mask.to(device)
        pooled = model.encode(ids, mask)
        logits = model.choice_head(pooled) / model.choice_temperature
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        choice_confs_on_junk.extend(conf.tolist())

# Use the TEST split, not val -- choice_temperature was fit on val
# (calibrate_multitask.py), so val is not a clean held-out baseline here.
# This mirrors how Test 1's Noul in-distribution baseline correctly uses
# the test split rather than the split its own temperature was fit on.
choice_indist_loader = DataLoader(TaskDataset(data["choice"]["test"], torch.long), batch_size=64)
choice_confs_in_dist = []
with torch.no_grad():
    for ids, mask, label in choice_indist_loader:
        ids, mask = ids.to(device), mask.to(device)
        pooled = model.encode(ids, mask)
        logits = model.choice_head(pooled) / model.choice_temperature
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        choice_confs_in_dist.extend(conf.tolist())

print("\n" + "=" * 70)
print("TEST 3: Choice (banking77) head on OOD real data (SMS text, no valid banking77 answer exists)")
print("=" * 70)
print(f"n={len(choice_confs_on_junk)}  mean_confidence_on_out-of-scope_input={sum(choice_confs_on_junk)/len(choice_confs_on_junk):.4f}")
print(f"For comparison, mean_confidence_on_in-distribution_input={sum(choice_confs_in_dist)/len(choice_confs_in_dist):.4f}")
print("There is no 'correct' banking77 label for SMS text, so no accuracy is reported here.")
print("A model whose confidence collapsed toward uniform (1/77=0.013) on out-of-scope input would be")
print("well-behaved under distribution shift; a model that stays confident is silently wrong instead of")
print("visibly uncertain -- that's the 'cannot hallucinate' claim's actual limit: fixed schema != correct answer.")

with open("ood_results.pkl", "wb") as f:
    pickle.dump({
        "in_dist_acc": in_dist_correct / in_dist_n,
        "in_dist_mean_conf": sum(in_dist_confs) / len(in_dist_confs),
        "ood_noul_acc": ood_correct / ood_n,
        "ood_noul_mean_conf": sum(ood_confs) / len(ood_confs),
        "choice_conf_in_dist": sum(choice_confs_in_dist) / len(choice_confs_in_dist),
        "choice_conf_on_junk": sum(choice_confs_on_junk) / len(choice_confs_on_junk),
    }, f)
print("\nSaved ood_results.pkl")
