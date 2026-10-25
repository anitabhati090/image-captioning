from captioning.baseline import Vocabulary


def test_vocabulary_encodes_and_stops_at_eos():
    vocab = Vocabulary(min_freq=1)
    vocab.build(["A dog runs"])
    ids = vocab.encode("a cat runs")
    assert ids[0] == vocab.stoi["<bos>"]
    assert ids[-1] == vocab.stoi["<eos>"]
    assert vocab.decode(ids) == "a <unk> runs"


def test_vocabulary_keeps_reserved_tokens():
    vocab = Vocabulary()
    vocab.build(["the the cat"])
    assert vocab.itos[:4] == ["<pad>", "<bos>", "<eos>", "<unk>"]
