import pytest
import torch

from model import JevCloneEncoder

VOCAB, MAX_LEN, CLASSES = 50, 16, 7


def make(pooling="mean"):
    torch.manual_seed(0)
    model = JevCloneEncoder(vocab_size=VOCAB, max_len=MAX_LEN, num_choice_classes=CLASSES,
                            enable_score=True, pooling=pooling)
    return model.eval()


def batch(lengths):
    ids = torch.zeros(len(lengths), MAX_LEN, dtype=torch.long)
    mask = torch.zeros(len(lengths), MAX_LEN, dtype=torch.long)
    g = torch.Generator().manual_seed(1)
    for i, n in enumerate(lengths):
        ids[i, :n] = torch.randint(1, VOCAB, (n,), generator=g)
        mask[i, :n] = 1
    return ids, mask


@pytest.mark.parametrize("pooling", ["mean", "attn"])
def test_one_forward_pass_answers_every_typed_head(pooling):
    ids, mask = batch([3, 9, 16])
    out = make(pooling).forward_all(ids, mask)
    assert out["noul_logit"].shape == (3,)
    assert out["choice_logits"].shape == (3, CLASSES)
    assert out["score_logit"].shape == (3,)


@pytest.mark.parametrize("pooling", ["mean", "attn"])
def test_typed_outputs_stay_inside_their_declared_ranges(pooling):
    model = make(pooling)
    ids, mask = batch([2, 5, 11, 16])
    is_spam, conf, prob = model.predict(ids, mask)
    assert is_spam.dtype == torch.bool
    assert torch.all((conf >= 0.5) & (conf <= 1)) and torch.all((prob >= 0) & (prob <= 1))

    cls, cconf, probs = model.predict_choice(ids, mask)
    assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-5)
    assert torch.equal(cls, probs.argmax(-1)) and torch.allclose(cconf, probs.max(-1).values)
    assert torch.all((cls >= 0) & (cls < CLASSES))

    score = model.predict_score(ids, mask)
    assert torch.all((score >= 0) & (score <= 1))


@pytest.mark.parametrize("pooling", ["mean", "attn"])
def test_padded_positions_do_not_change_the_answer(pooling):
    model = make(pooling)
    ids, mask = batch([4, 10])
    garbage = ids.clone()
    garbage[mask == 0] = torch.randint(1, VOCAB, (int((mask == 0).sum()),))
    with torch.no_grad():
        assert torch.allclose(model.encode(ids, mask), model.encode(garbage, mask), atol=1e-5)


def test_attention_pooling_adds_only_the_query_layer():
    mean_keys, attn_keys = set(make("mean").state_dict()), set(make("attn").state_dict())
    assert attn_keys - mean_keys == {"attn_query.weight", "attn_query.bias"}
    assert mean_keys <= attn_keys


def test_unknown_pooling_is_rejected():
    with pytest.raises(ValueError):
        JevCloneEncoder(vocab_size=VOCAB, pooling="max")
