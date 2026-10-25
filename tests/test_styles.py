import pytest

from captioning.styles import (ALT_TEXT_MAX_CHARS, apply_style, clean_caption,
                               extract_hashtags, to_alt_text)


def test_clean_caption_capitalises_and_punctuates():
    assert clean_caption("  a dog running in the park ") == "A dog running in the park."


def test_clean_caption_removes_blip_artifact():
    assert "arafed" not in clean_caption("arafed man riding a bike").lower()


def test_alt_text_strips_filler():
    assert to_alt_text("a picture of a cat on a sofa") == "A cat on a sofa."


def test_alt_text_respects_limit():
    long = "a dog " + "running quickly " * 30
    out = to_alt_text(long)
    assert len(out) <= ALT_TEXT_MAX_CHARS
    assert out.endswith(".")


def test_hashtags_skip_stopwords_and_dedupe():
    tags = extract_hashtags("a dog and a dog sitting on the beach", max_tags=5)
    assert tags == ["#dog", "#beach"]


def test_social_style_has_hashtags():
    assert "#" in apply_style("social", "a woman holding an umbrella in the rain")


def test_unknown_style_raises():
    with pytest.raises(ValueError):
        apply_style("poetic", "a cat")
