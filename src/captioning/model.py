"""Vision-Language model wrapper (BLIP / BLIP-2 via Hugging Face transformers)."""

from __future__ import annotations

from PIL import Image

from .config import settings


class CaptionModel:
    """Loads a BLIP or BLIP-2 captioning model and generates base captions."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        import torch
        from transformers import (
            Blip2ForConditionalGeneration,
            Blip2Processor,
            BlipForConditionalGeneration,
            BlipProcessor,
        )

        self.model_name = model_name or settings.caption_model
        self.is_blip2 = "blip2" in self.model_name.lower()
        self.device = device or settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (self.is_blip2 and self.device == "cuda") else torch.float32

        proc_cls = Blip2Processor if self.is_blip2 else BlipProcessor
        model_cls = Blip2ForConditionalGeneration if self.is_blip2 else BlipForConditionalGeneration

        self.processor = proc_cls.from_pretrained(self.model_name)
        try:
            # "eager" attention lets us read attention weights for the XAI module.
            self.model = model_cls.from_pretrained(
                self.model_name, torch_dtype=self.dtype, attn_implementation="eager"
            )
        except (TypeError, ValueError):
            self.model = model_cls.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.model.to(self.device).eval()

    def caption(self, image: Image.Image, **gen_kwargs) -> str:
        """Generate one caption for ``image``. ``gen_kwargs`` go to model.generate()."""
        import torch

        gen_kwargs.setdefault("max_new_tokens", 30)
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = inputs.to(self.device, self.dtype) if self.is_blip2 else inputs.to(self.device)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
