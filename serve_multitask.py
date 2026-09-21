"""
Typed decision API for the multi-task model -- demonstrates the actual
architectural claim: ONE encoder forward pass (model.encode()) produces a
pooled state vector that every typed head reads from, so multiple typed
questions about the SAME state are answered in parallel rather than by
three separate models.

Usage:
    python3 serve_multitask.py --demo
    python3 serve_multitask.py noul "Some SMS text"
    python3 serve_multitask.py choice "I still haven't gotten my new card"
    python3 serve_multitask.py score "A dog is running." "A dog is playing."
"""
import sys
import pickle
import json
import torch
from tokenizers import Tokenizer

from model import JevCloneEncoder

with open("data/prepared_multitask.pkl", "rb") as f:
    data = pickle.load(f)
vocab_size = data["vocab_size"]
max_len = data["max_len"]
pad_id = data["pad_id"]
num_choice_classes = data["num_choice_classes"]
categories = data["choice_categories"]
tokenizer = Tokenizer.from_file("data/tokenizer_multitask.json")

device = "cpu"
model = JevCloneEncoder(
    vocab_size=vocab_size, max_len=max_len,
    num_choice_classes=num_choice_classes, enable_score=True,
).to(device)
model.load_state_dict(torch.load("jev_clone_multitask_calibrated.pt", map_location=device))
model.eval()


def encode(text):
    enc = tokenizer.encode(text)
    ids = enc.ids[:max_len]
    mask = [1] * len(ids)
    pad_n = max_len - len(ids)
    ids = ids + [pad_id] * pad_n
    mask = mask + [0] * pad_n
    return ids, mask


def decide_all(text: str) -> dict:
    """
    ONE encoder forward pass -> answers for every typed head that exists
    on this model, read off the same pooled state vector.
    """
    ids, mask = encode(text)
    ids_t = torch.tensor([ids], dtype=torch.long)
    mask_t = torch.tensor([mask], dtype=torch.long)

    with torch.no_grad():
        pooled = model.encode(ids_t, mask_t)  # <-- computed ONCE

        noul_logit = model.noul_head(pooled).squeeze(-1) / model.temperature
        prob_spam = torch.sigmoid(noul_logit)
        is_spam = (prob_spam > 0.5).item()

        choice_logits = model.choice_head(pooled) / model.choice_temperature
        choice_probs = torch.softmax(choice_logits, dim=-1)
        choice_conf, choice_idx = choice_probs.max(dim=-1)

        score = torch.sigmoid(model.score_head(pooled).squeeze(-1))

    return {
        "noul": {"is_spam": bool(is_spam), "confidence": round(max(prob_spam.item(), 1 - prob_spam.item()), 4)},
        "choice": {"category": categories[choice_idx.item()], "confidence": round(choice_conf.item(), 4)},
        "score": round(score.item(), 4),
    }


DEMO_EXAMPLES = [
    "WINNER!! You have been selected to receive a $900 prize reward, call 08712300220 to claim now!",
    "I still haven't received my new card, I ordered it two weeks ago.",
    "A man is playing a large flute. <sep> A man is playing a flute.",
]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        for text in DEMO_EXAMPLES:
            print(json.dumps({"text": text, **decide_all(text)}, indent=None))
    elif len(sys.argv) > 2 and sys.argv[1] == "score":
        text = f"{sys.argv[2]} <sep> {sys.argv[3]}"
        print(json.dumps({"text": text, **decide_all(text)}, indent=2))
    elif len(sys.argv) > 2:
        text = " ".join(sys.argv[2:])
        print(json.dumps({"text": text, **decide_all(text)}, indent=2))
    else:
        print(__doc__)
