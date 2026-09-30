import torch

from mlm import MASK_ID, chunk_ids, mask_tokens, pad_batch


def test_chunking_keeps_every_token_in_order():
    ids = list(range(10, 23))
    windows = chunk_ids(ids, 5)
    assert [len(w) for w in windows] == [5, 5, 3]
    assert sum(windows, []) == ids


def test_padding_marks_real_tokens_only():
    ids, mask = pad_batch([[5, 6, 7], [8]], max_len=4)
    assert ids.tolist() == [[5, 6, 7, 0], [8, 0, 0, 0]]
    assert mask.tolist() == [[1, 1, 1, 0], [1, 0, 0, 0]]


def test_masking_only_touches_real_tokens_and_labels_them():
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(2, 100, (64, 32), generator=g)
    mask = torch.zeros_like(ids)
    mask[:, :20] = 1
    inputs, labels = mask_tokens(ids, mask, vocab_size=100, generator=g)
    picked = labels != -100
    assert not picked[mask == 0].any()
    assert torch.equal(labels[picked], ids[picked])
    assert torch.equal(inputs[~picked], ids[~picked])
    rate = picked.sum().item() / mask.sum().item()
    assert 0.12 < rate < 0.18
    masked_share = (inputs[picked] == MASK_ID).float().mean().item()
    assert 0.7 < masked_share < 0.9
