"""
Self-supervised pretraining of BEMA's encoder, then fine-tuning.

No pretrained encoder can be downloaded here (FINDINGS.md §6), so this
pretrains the shared trunk ourselves with masked-token prediction
(BERT-style MLM) on unlabeled text, then fine-tunes the multitask model
exactly as train_multitask.py does and compares it with the from-scratch
model across the same three training seeds.

Corpus: the TEXT of every training split this repo uses (SMS spam,
BANKING77, Amazon Fine Food Reviews, CLINC150 in-scope + oos), labels
ignored. No validation or test text is used. Long texts are cut into
64-token windows instead of truncated. 5% of windows are held out to pick
the pretraining epoch.

This is small in-domain pretraining (~1.2M tokens), not a web-scale model
like BERT, so it answers a narrower question than FINDINGS.md §6: does
self-supervised pretraining, of the kind possible here, change accuracy,
calibration or typo robustness?

Usage: python3 self_pretrain.py   (resumes from any checkpoint already on disk)
"""
import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn

from calib_utils import ece, fit_temperature
from heldout_noise_test import encode_batch, perturbed, tokenizer
from mlm import chunk_ids, mask_tokens, pad_batch
from model import JevCloneEncoder
from multiseed_sweep import evaluate, get_model
from train_multitask import data, device, max_len, num_choice_classes, train_model, vocab_size

SEEDS = [42, 123, 2024]
TRUNK_PATH = "bema_pretrained_trunk.pt"
PRETRAIN_EPOCHS = 20
BATCH = 64
NOISES = ("swap2", "keyboard2", "mixed3")
TRUNK_PREFIXES = ("embed.", "pos.", "encoder.")


def corpus_windows():
    texts = [r["text"] for task in ("noul", "choice", "score") for r in data[task]["train"]]
    with open("data/raw/clinc150_data_full.json") as f:
        clinc = json.load(f)
    texts += [t for t, _ in clinc["train"]] + [t for t, _ in clinc["oos_train"]]
    windows = []
    for t in texts:
        windows += [w for w in chunk_ids(tokenizer.encode(t).ids, max_len) if len(w) >= 2]
    return texts, windows


def mlm_loss_and_acc(model, head, ids, mask, gen):
    inputs, labels = mask_tokens(ids, mask, vocab_size, gen)
    logits = head(model.token_states(inputs.to(device), mask.to(device)))
    labels = labels.to(device)
    loss = nn.functional.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=-100)
    picked = labels != -100
    acc = (logits.argmax(-1)[picked] == labels[picked]).float().mean().item()
    return loss, acc, int(picked.sum())


def pretrain():
    texts, windows = corpus_windows()
    random.Random(0).shuffle(windows)
    n_val = len(windows) // 20
    val, train = windows[:n_val], windows[n_val:]
    print(f"Pretraining corpus: {len(texts)} texts -> {len(windows)} windows "
          f"({sum(map(len, windows))} tokens); train {len(train)}, held out {len(val)}")

    torch.manual_seed(0)
    model = JevCloneEncoder(vocab_size=vocab_size, max_len=max_len,
                            num_choice_classes=num_choice_classes, enable_score=True).to(device)
    head = nn.Linear(model.embed.embedding_dim, vocab_size).to(device)
    params = list(model.embed.parameters()) + list(model.encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=1e-2)
    steps = PRETRAIN_EPOCHS * ((len(train) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=steps, pct_start=0.06)
    val_gen = torch.Generator().manual_seed(1)
    val_batches = [pad_batch(val[i:i + 256], max_len) for i in range(0, len(val), 256)]
    val_masks = [mask_tokens(ids, mask, vocab_size, val_gen) for ids, mask in val_batches]
    gen = torch.Generator().manual_seed(2)

    best_loss, best_trunk = float("inf"), None
    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        order = torch.randperm(len(train), generator=gen).tolist()
        total = 0.0
        for i in range(0, len(order), BATCH):
            ids, mask = pad_batch([train[j] for j in order[i:i + BATCH]], max_len)
            loss, _, _ = mlm_loss_and_acc(model, head, ids, mask, gen)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            v_loss, v_hits, v_n = 0.0, 0.0, 0
            for (ids, mask), (inputs, labels) in zip(val_batches, val_masks):
                logits = head(model.token_states(inputs.to(device), mask.to(device)))
                lab = labels.to(device)
                picked = lab != -100
                v_loss += nn.functional.cross_entropy(logits[picked], lab[picked], reduction="sum").item()
                v_hits += (logits.argmax(-1)[picked] == lab[picked]).sum().item()
                v_n += int(picked.sum())
        v_loss /= v_n
        print(f"Pretrain epoch {epoch + 1:2d}  train loss {total / ((len(order) + BATCH - 1) // BATCH):.4f}  "
              f"held-out loss {v_loss:.4f}  held-out masked-token acc {v_hits / v_n:.4f}", flush=True)
        if v_loss < best_loss:
            best_loss = v_loss
            best_trunk = {k: v.detach().clone() for k, v in model.state_dict().items()
                          if k.startswith(TRUNK_PREFIXES)}
    torch.save(best_trunk, TRUNK_PATH)
    print(f"Saved {TRUNK_PATH} (best held-out MLM loss {best_loss:.4f})")


def pretrained_factory():
    trunk = torch.load(TRUNK_PATH, map_location="cpu")

    def make():
        model = JevCloneEncoder(vocab_size=vocab_size, max_len=max_len,
                                num_choice_classes=num_choice_classes, enable_score=True)
        missing, unexpected = model.load_state_dict(trunk, strict=False)
        assert not unexpected and all(not k.startswith(TRUNK_PREFIXES) for k in missing)
        return model
    return make


def typo_metrics(model):
    """Choice head, temperature fit on val: clean vs typo'd test accuracy/confidence/ECE."""
    def logits(texts):
        ids, mask = encode_batch(texts)
        with torch.no_grad():
            return torch.cat([model.choice_head(model.encode(ids[i:i + 256].to(device), mask[i:i + 256].to(device))).cpu()
                              for i in range(0, len(texts), 256)])

    model.eval()
    T = fit_temperature(logits([r["text"] for r in data["choice"]["val"]]),
                        torch.tensor([r["label"] for r in data["choice"]["val"]]))
    texts = [r["text"] for r in data["choice"]["test"]]
    y = torch.tensor([r["label"] for r in data["choice"]["test"]])
    out = {}
    for name in ("clean",) + NOISES:
        probs = torch.softmax(logits(texts if name == "clean" else perturbed(texts, name)) / T, -1)
        conf, pred = probs.max(-1)
        correct = (pred == y).float()
        out[f"{name}_acc"] = correct.mean().item()
        out[f"{name}_overconf"] = conf.mean().item() - correct.mean().item()
        out[f"{name}_ece"] = ece(conf.tolist(), correct.tolist())
    for name in NOISES:
        out[f"{name}_drop"] = out["clean_acc"] - out[f"{name}_acc"]
    return out


if __name__ == "__main__":
    if not os.path.exists(TRUNK_PATH):
        pretrain()
    else:
        print(f"Using existing {TRUNK_PATH}")

    results = {}
    for seed in SEEDS:
        path = f"jev_clone_selfpretrained_seed{seed}.pt"
        if os.path.exists(path):
            tuned = JevCloneEncoder(vocab_size=vocab_size, max_len=max_len,
                                    num_choice_classes=num_choice_classes, enable_score=True).to(device)
            tuned.load_state_dict(torch.load(path, map_location=device))
            print(f"[pretrained seed={seed}] loaded {path}")
        else:
            print(f"[pretrained seed={seed}] fine-tuning -> {path}", flush=True)
            tuned = train_model(seed=seed, model_factory=pretrained_factory(), save_path=path)
        for variant, model in (("scratch", get_model("mean", seed)), ("pretrained", tuned)):
            r = evaluate(model)
            r.update(typo_metrics(model))
            results[(variant, seed)] = r
            print(f"[{variant} seed={seed}] " + " ".join(f"{k}={v:.4f}" for k, v in r.items()), flush=True)

    metrics = ["noul_acc", "choice_acc", "choice_ece_cal", "score_mae", "score_r", "clean_acc"]
    metrics += [f"{n}_{k}" for n in NOISES for k in ("acc", "drop", "overconf", "ece")]
    print("\n" + "=" * 86)
    print(f"Test metrics over fine-tuning seeds {SEEDS}: mean +/- sample std (n={len(SEEDS)})")
    print("=" * 86)
    print(f"{'Metric':<20}{'From scratch':>20}{'Self-pretrained':>20}{'Paired diff (pre - scratch)':>28}")
    for m in metrics:
        s = np.array([results[("scratch", k)][m] for k in SEEDS])
        p = np.array([results[("pretrained", k)][m] for k in SEEDS])
        d = p - s
        print(f"{m:<20}{s.mean():>12.4f} +/- {s.std(ddof=1):.4f}{p.mean():>12.4f} +/- {p.std(ddof=1):.4f}"
              f"{d.mean():>18.4f} +/- {d.std(ddof=1):.4f}")

    with open("self_pretrain_results.pkl", "wb") as f:
        pickle.dump({"seeds": SEEDS, "results": results}, f)
    print("\nSaved self_pretrain_results.pkl")
