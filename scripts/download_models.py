"""Pre-download model weights so the first app run is fast.  Usage: python scripts/download_models.py"""

import argparse

from captioning.config import settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=settings.caption_model)
    args = ap.parse_args()

    from captioning.model import CaptionModel

    print(f"Downloading {args.model} ...")
    CaptionModel(args.model)
    print(f"Downloading {settings.clip_model} ...")
    from transformers import CLIPModel, CLIPProcessor

    CLIPModel.from_pretrained(settings.clip_model)
    CLIPProcessor.from_pretrained(settings.clip_model)
    print("Done.")


if __name__ == "__main__":
    main()
