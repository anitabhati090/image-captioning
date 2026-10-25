"""Multi-style caption engine (pure Python, no ML imports).

One base caption is produced by the VLM; this module reshapes it into the four
output modes described in the dissertation:

    standard | detailed | social | alt_text
"""

from __future__ import annotations

import re

STYLES = {
    "standard": "Concise single-sentence description",
    "detailed": "Longer, richer description",
    "social": "Conversational caption with hashtags",
    "alt_text": "Screen-reader friendly alt-text (WCAG-oriented)",
}

# Decoding settings per style (passed to model.generate()).
GENERATION_KWARGS = {
    "standard": {"max_new_tokens": 30, "num_beams": 3},
    "detailed": {
        "max_new_tokens": 90,
        "min_length": 30,
        "num_beams": 5,
        "repetition_penalty": 1.5,
        "no_repeat_ngram_size": 3,
    },
    "social": {"max_new_tokens": 30, "num_beams": 3},
    "alt_text": {"max_new_tokens": 30, "num_beams": 3},
}

ALT_TEXT_MAX_CHARS = 125  # common practical guideline for concise alt-text

_STOPWORDS = set(
    """a an the of in on at to with and or for from by is are was were be been it its this that
    there their his her them they as into onto over under near next front back behind side top
    sitting standing holding looking walking laying lying wearing playing riding eating
    very some one two three several group picture image photo photograph close up large small
    big little white black""".split()
)

_ALT_PREFIX = re.compile(
    r"^(an?|the)\s+(image|picture|photo|photograph|photography)\s+(of|showing|shows|with)\s+",
    re.IGNORECASE,
)


def clean_caption(text: str) -> str:
    """Normalise whitespace, drop BLIP artefacts, capitalise and end with a full stop."""
    text = re.sub(r"\barafed\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,.")
    if not text:
        return ""
    return text[0].upper() + text[1:] + "."


def extract_hashtags(caption: str, max_tags: int = 5) -> list[str]:
    """Pick up to ``max_tags`` content words from the caption as hashtags."""
    seen: list[str] = []
    for word in re.findall(r"[A-Za-z]+", caption.lower()):
        if len(word) >= 3 and word not in _STOPWORDS and word not in seen:
            seen.append(word)
    return ["#" + w for w in seen[:max_tags]]


def to_alt_text(caption: str, max_chars: int = ALT_TEXT_MAX_CHARS) -> str:
    """Strip filler like 'a picture of', then trim to ``max_chars`` on a word boundary."""
    text = re.sub(r"\barafed\b", "", caption, flags=re.IGNORECASE)
    text = _ALT_PREFIX.sub("", re.sub(r"\s+", " ", text).strip())
    text = clean_caption(text)
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1].rsplit(" ", 1)[0].rstrip(" ,;:.")
    return cut + "."


def to_social(caption: str, max_tags: int = 5) -> str:
    text = clean_caption(caption)
    tags = extract_hashtags(text, max_tags)
    return f"{text} {' '.join(tags)}".strip()


def apply_style(style: str, caption: str) -> str:
    """Turn a raw model caption into the requested style."""
    if style not in STYLES:
        raise ValueError(f"Unknown style '{style}'. Choose from: {', '.join(STYLES)}")
    if style == "alt_text":
        return to_alt_text(caption)
    if style == "social":
        return to_social(caption)
    return clean_caption(caption)  # standard / detailed
