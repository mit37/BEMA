import random

import pytest

from noise import NEIGHBORS, NOISE, perturb

TEXT = "How do I locate my card and check the balance?"


def run(name, seed=0):
    return perturb(TEXT, NOISE[name], random.Random(seed))


def test_swaps_only_reorder_characters():
    for name in ("swap2", "swap4"):
        assert sorted(run(name)) == sorted(TEXT)


def test_deletion_and_insertion_change_length_by_two():
    assert len(run("delete2")) == len(TEXT) - 2
    assert len(run("insert2")) == len(TEXT) + 2


def test_keyboard_substitutions_use_neighboring_keys():
    out = run("keyboard2")
    assert len(out) == len(TEXT)
    changed = [(a, b) for a, b in zip(TEXT, out) if a != b]
    assert 1 <= len(changed) <= 2
    for orig, new in changed:
        assert new.lower() in NEIGHBORS[orig.lower()]


def test_neighbor_relation_is_symmetric():
    for key, near in NEIGHBORS.items():
        for other in near:
            assert key in NEIGHBORS[other]


@pytest.mark.parametrize("name", sorted(NOISE))
def test_same_seed_gives_same_perturbation(name):
    assert run(name, seed=3) == run(name, seed=3)
    assert run(name, seed=3) != TEXT
