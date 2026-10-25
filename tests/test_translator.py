import pytest

from captioning import translator


def test_english_is_passthrough():
    assert translator.translate("hello", "en") == "hello"


def test_empty_text_passthrough():
    assert translator.translate("", "hi") == ""


def test_failure_is_wrapped(monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            pass

        def translate(self, _):
            raise ConnectionError("offline")

    pytest.importorskip("deep_translator")
    monkeypatch.setattr("deep_translator.GoogleTranslator", Boom)
    with pytest.raises(translator.TranslationUnavailable):
        translator.translate("hello", "hi")
