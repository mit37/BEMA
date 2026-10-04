"""Masked-language-model helpers for self-supervised pretraining of the encoder."""
import torch

MASK_ID = 1  # the tokenizer's <unk>; byte-level BPE never emits it for real text
FIRST_REAL_ID = 2  # ids 0 and 1 are <pad> and <unk>


def chunk_ids(token_ids, max_len):
    """Split one token sequence into consecutive windows of at most max_len."""
    return [token_ids[i:i + max_len] for i in range(0, len(token_ids), max_len)] or [[]]


def pad_batch(windows, max_len, pad_id=0):
    ids = torch.full((len(windows), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(windows), max_len), dtype=torch.long)
    for i, w in enumerate(windows):
        ids[i, :len(w)] = torch.tensor(w, dtype=torch.long)
        mask[i, :len(w)] = 1
    return ids, mask


def mask_tokens(ids, mask, vocab_size, generator, p=0.15):
    """BERT-style corruption: pick p of the real tokens; of those, 80% become
    MASK_ID, 10% a random real token, 10% stay unchanged. Labels are the
    original ids at picked positions and -100 (ignored) everywhere else."""
    picked = (torch.rand(ids.shape, generator=generator) < p) & (mask == 1)
    labels = torch.where(picked, ids, torch.full_like(ids, -100))
    inputs = ids.clone()
    r = torch.rand(ids.shape, generator=generator)
    inputs[picked & (r < 0.8)] = MASK_ID
    swap = picked & (r >= 0.8) & (r < 0.9)
    inputs[swap] = torch.randint(FIRST_REAL_ID, vocab_size, (int(swap.sum()),), generator=generator)
    return inputs, labels
