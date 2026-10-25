import pytest

from captioning.evaluation import tokenize


def test_tokenize():
    assert tokenize("A Dog, running!") == ["a", "dog", "running"]


def test_bleu_perfect_match():
    pytest.importorskip("nltk")
    from captioning.evaluation import bleu

    s = bleu({"a": "a dog runs in the park"}, {"a": ["a dog runs in the park"]})
    assert s["BLEU-1"] == pytest.approx(1.0)
