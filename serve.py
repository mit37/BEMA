"""
Typed decision API — the Jev-style interface.

Send it state (raw text) and get back a typed, structured decision in one
call: no text generation, no parsing, a fixed output schema, with a
calibrated confidence score. This is the "shape" of what Jev exposes,
built at toy scale.

Usage:
    python3 serve.py "Congratulations! You've won a free cruise, call now!"
    python3 serve.py --demo
"""
import sys
import re
import pickle
import json
import torch

from model import JevCloneEncoder

with open("data/prepared.pkl", "rb") as f:
    data = pickle.load(f)
vocab = data["vocab"]
max_len = data["max_len"]
stoi = {w: i for i, w in enumerate(vocab)}

device = "cpu"
model = JevCloneEncoder(vocab_size=len(vocab), max_len=max_len).to(device)
model.load_state_dict(torch.load("jev_clone_calibrated.pt", map_location=device))
model.eval()


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def encode(text):
    toks = tokenize(text)[:max_len]
    ids = [stoi.get(t, 1) for t in toks]
    ids = ids + [0] * (max_len - len(ids))
    mask = [1] * len(toks) + [0] * (max_len - len(toks))
    return ids, mask


def decide(text: str) -> dict:
    """
    The typed decision call: state (text) -> typed answer.
    No string generation. Fixed schema. Calibrated confidence.
    """
    ids, mask = encode(text)
    ids_t = torch.tensor([ids], dtype=torch.long)
    mask_t = torch.tensor([mask], dtype=torch.long)
    is_spam, confidence, prob_spam = model.predict(ids_t, mask_t)
    return {
        "is_spam": bool(is_spam.item()),      # Noul decision
        "confidence": round(confidence.item(), 4),  # Score, calibrated
        "raw_probability": round(prob_spam.item(), 4),
    }


DEMO_EXAMPLES = [
    "Hey, are we still on for lunch tomorrow?",
    "WINNER!! You have been selected to receive a $900 prize reward, call 08712300220 to claim now!",
    "Can you pick up milk on your way home?",
    "URGENT: Your mobile number has won 2000 pounds in our lucky draw. Text CLAIM to 87121.",
    "Free entry in 2 a wkly comp to win FA Cup final tkts, text FA to 87121",
    "ok see you at 6",
]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        for text in DEMO_EXAMPLES:
            result = decide(text)
            print(json.dumps({"text": text, **result}, indent=None))
    elif len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        result = decide(text)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python3 serve.py \"<text>\"  or  python3 serve.py --demo")
