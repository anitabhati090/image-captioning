"""Multilingual translation layer.

Translation is a separate, downstream stage so new languages can be added
without touching the vision-language model.
"""

from __future__ import annotations

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
    "Chinese (Simplified)": "zh-CN",
    "Japanese": "ja",
    "Bengali": "bn",
    "Tamil": "ta",
    "Marathi": "mr",
    "Urdu": "ur",
}


class TranslationUnavailable(RuntimeError):
    """Raised when the translation service cannot be reached."""


def translate(text: str, target_code: str, source_code: str = "en") -> str:
    """Translate ``text`` to ``target_code`` (e.g. 'hi'). Returns text unchanged for English."""
    if not text or target_code == source_code:
        return text
    try:
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source=source_code, target=target_code).translate(text)
    except Exception as exc:  # network errors, rate limits, unsupported code...
        raise TranslationUnavailable(f"Could not translate to '{target_code}': {exc}") from exc
