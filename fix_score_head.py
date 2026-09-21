"""
Follow-up fix to the Score head failure (see FINDINGS.md sec. 2 and
diagnose_score_head.py). diagnose_score_head.py showed untrained cosine
similarity of separately-encoded sentences (r=0.48) already beats the
trained concat+MLP Score head (r=0.29) -- the scoring architecture, not
the encoder, was the bottleneck.

This trains a proper bi-encoder regression head on top of the SAME frozen
shared encoder (no retraining of the trunk -- isolates the scoring-head
architecture as the only variable): encode sentence1 and sentence2
SEPARATELY, build the classic SBERT-style feature vector
[u, v, |u-v|] (Reimers & Gurevych 2019), and fit a small MLP regression
on real STS-B train labels, evaluated on the untouched held-out test
split.
"""
import csv
import pickle
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from model import JevCloneEncoder
from train_multitask import vocab_size, max_len, num_choice_classes, device

torch.manual_seed(42)

model = JevCloneEncoder(
    vocab_size=vocab_size, max_len=max_len,
    num_choice_classes=num_choice_classes, enable_score=True,
).to(device)
model.load_state_dict(torch.load("jev_clone_multitask_calibrated.pt", map_location=device))
model.eval()  # frozen -- we do NOT train the encoder here
for p in model.parameters():
    p.requires_grad = False

tokenizer = Tokenizer.from_file("data/tokenizer_multitask.json")
with open("data/prepared_multitask.pkl", "rb") as f:
    data = pickle.load(f)
pad_id = data["pad_id"]


def encode(text):
    enc = tokenizer.encode(text)
    ids = enc.ids[:max_len]
    mask = [1] * len(ids)
    pad_n = max_len - len(ids)
    ids = ids + [pad_id] * pad_n
    mask = mask + [0] * pad_n
    return ids, mask


def load_pairs(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) < 3:
                continue
            s1, s2, score = r[0], r[1], r[2]
            try:
                score = float(score)
            except ValueError:
                continue
            pairs.append((s1, s2, score / 5.0))
    return pairs


train_pairs = load_pairs("data/raw/stsb_train.csv")
val_pairs = load_pairs("data/raw/stsb_dev.csv")
test_pairs = load_pairs("data/raw/stsb_test.csv")
print(f"train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")


@torch.no_grad()
def pool_batch(pairs):
    """Precompute frozen encoder outputs for all pairs (cheap: no grad)."""
    u_list, v_list, y_list = [], [], []
    for s1, s2, score in pairs:
        ids1, mask1 = encode(s1)
        ids2, mask2 = encode(s2)
        ids1_t = torch.tensor([ids1], dtype=torch.long).to(device)
        mask1_t = torch.tensor([mask1], dtype=torch.long).to(device)
        ids2_t = torch.tensor([ids2], dtype=torch.long).to(device)
        mask2_t = torch.tensor([mask2], dtype=torch.long).to(device)
        u = model.encode(ids1_t, mask1_t).squeeze(0)
        v = model.encode(ids2_t, mask2_t).squeeze(0)
        u_list.append(u)
        v_list.append(v)
        y_list.append(score)
    return torch.stack(u_list), torch.stack(v_list), torch.tensor(y_list, dtype=torch.float)


print("Precomputing frozen encoder outputs for train/val/test...")
u_train, v_train, y_train = pool_batch(train_pairs)
u_val, v_val, y_val = pool_batch(val_pairs)
u_test, v_test, y_test = pool_batch(test_pairs)


def sbert_features(u, v):
    return torch.cat([u, v, (u - v).abs()], dim=-1)


d_model = u_train.shape[-1]
biencoder_head = nn.Sequential(
    nn.Linear(d_model * 3, d_model), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_model, 1)
).to(device)

optimizer = torch.optim.AdamW(biencoder_head.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

feat_train = sbert_features(u_train, v_train).to(device)
feat_val = sbert_features(u_val, v_val).to(device)
feat_test = sbert_features(u_test, v_test).to(device)
y_train_d, y_val_d, y_test_d = y_train.to(device), y_val.to(device), y_test.to(device)

EPOCHS = 60
best_val_mae = float("inf")
best_state = None

for epoch in range(EPOCHS):
    biencoder_head.train()
    optimizer.zero_grad()
    pred = torch.sigmoid(biencoder_head(feat_train).squeeze(-1))
    loss = criterion(pred, y_train_d)
    loss.backward()
    optimizer.step()

    biencoder_head.eval()
    with torch.no_grad():
        val_pred = torch.sigmoid(biencoder_head(feat_val).squeeze(-1))
        val_mae = (val_pred - y_val_d).abs().mean().item()
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_state = {k: v.clone() for k, v in biencoder_head.state_dict().items()}

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:2d}  train_loss={loss.item():.4f}  val_mae={val_mae:.4f}")

biencoder_head.load_state_dict(best_state)
biencoder_head.eval()

with torch.no_grad():
    test_pred = torch.sigmoid(biencoder_head(feat_test).squeeze(-1))
    test_mae = (test_pred - y_test_d).abs().mean().item()
    test_pearson = torch.corrcoef(torch.stack([test_pred.cpu(), y_test]))[0, 1].item()

print("\n" + "=" * 60)
print("Score head architecture comparison (all on the SAME frozen encoder, same held-out test set)")
print("=" * 60)
print(f"{'Approach':40s} {'MAE':>8s} {'Pearson r':>10s}")
print(f"{'Original concat+meanpool+MLP (trained)':40s} {0.2442:>8.4f} {0.2941:>10.4f}")
print(f"{'Untrained cosine sim (no training at all)':40s} {'n/a':>8s} {0.4838:>10.4f}")
print(f"{'Bi-encoder SBERT-features MLP (trained)':40s} {test_mae:>8.4f} {test_pearson:>10.4f}")

torch.save(biencoder_head.state_dict(), "score_biencoder_head.pt")
print("\nSaved score_biencoder_head.pt")
