"""End-to-end pipeline: image -> base caption -> style -> translation."""

from __future__ import annotations

import time
from dataclasses import dataclass

from PIL import Image

from .styles import GENERATION_KWARGS, apply_style
from .translator import TranslationUnavailable, translate


@dataclass
class CaptionResult:
    style: str
    language_code: str
    base_caption: str
    styled_caption: str
    translated_caption: str
    latency_s: float
    translation_error: str | None = None


class CaptioningPipeline:
    def __init__(self, model):
        self.model = model

    def run(self, image: Image.Image, style: str = "standard", language_code: str = "en") -> CaptionResult:
        start = time.perf_counter()
        base = self.model.caption(image, **GENERATION_KWARGS[style])
        styled = apply_style(style, base)

        error = None
        try:
            translated = translate(styled, language_code)
        except TranslationUnavailable as exc:
            translated, error = styled, str(exc)

        return CaptionResult(
            style=style,
            language_code=language_code,
            base_caption=base,
            styled_caption=styled,
            translated_caption=translated,
            latency_s=round(time.perf_counter() - start, 2),
            translation_error=error,
        )
