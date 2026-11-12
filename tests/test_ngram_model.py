from ngram_model import NgramEncoder, collate, ngram_features


def test_features_are_deterministic_and_in_range():
    a = ngram_features("Card was declined twice", 1000)
    assert a == ngram_features("card WAS declined twice", 1000)
    assert all(0 <= i < 1000 for i in a)


def test_features_include_word_bigrams_and_char_grams():
    one = ngram_features("ab", 2 ** 20)
    # word "ab" + char grams of " ab ": 3 bigrams, 2 trigrams, 1 four-gram
    assert len(one) == 1 + 3 + 2 + 1
    assert len(ngram_features("ab cd", 2 ** 20)) == 2 + 1 + 2 * 6


def test_empty_text_maps_to_one_feature():
    assert ngram_features("", 100) == [0]


def test_collate_offsets_point_at_each_example():
    index, offsets = collate([[1, 2, 3], [4], [5, 6]])
    assert index.tolist() == [1, 2, 3, 4, 5, 6]
    assert offsets.tolist() == [0, 3, 4]


def test_heads_have_fixed_shapes():
    model = NgramEncoder(buckets=512, d_model=16, num_choice_classes=7, enable_score=True).eval()
    h = model.encode(*model.featurize(["hello there", "x", ""]))
    assert h.shape == (3, 16)
    assert model.noul_head(h).shape == (3, 1)
    assert model.choice_head(h).shape == (3, 7)
    assert model.score_head(h).shape == (3, 1)
