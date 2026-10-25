"""Central configuration. Values come from environment variables / a .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    caption_model: str = os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
    clip_model: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    device: str = os.getenv("DEVICE", "")  # "" = auto
    results_dir: Path = ROOT_DIR / "results"
    eval_dir: Path = ROOT_DIR / "data" / "eval"


settings = Settings()

# Models offered in the UI. BLIP-2 needs a GPU with roughly 8 GB+ VRAM.
AVAILABLE_MODELS = [
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
    "Salesforce/blip2-opt-2.7b",
]
