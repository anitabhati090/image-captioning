"""Caption-quality metrics: BLEU, METEOR, ROUGE-L, CIDEr and CLIPScore.

Metrics are imported lazily so the rest of the project works even if one
optional dependency is missing.
"""

from __future__ import annotations

import re
from statistics import mean

from PIL import Image

from .config import settings

_CLIP_CACHE = {}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ensure_nltk() -> None:
    import nltk

    for pkg in ("wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)


def bleu(preds: dict[str, str], refs: dict[str, list[str]]) -> dict[str, float]:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    ids = list(preds)
    hyps = [tokenize(preds[i]) for i in ids]
    rfs = [[tokenize(r) for r in refs[i]] for i in ids]
    smooth = SmoothingFunction().method1
    return {
        "BLEU-1": corpus_bleu(rfs, hyps, weights=(1, 0, 0, 0), smoothing_function=smooth),
        "BLEU-4": corpus_bleu(rfs, hyps, weights=(0.25,) * 4, smoothing_function=smooth),
    }


def meteor(preds: dict[str, str], refs: dict[str, list[str]]) -> float:
    from nltk.translate.meteor_score import meteor_score

    ensure_nltk()
    return mean(meteor_score([tokenize(r) for r in refs[i]], tokenize(preds[i])) for i in preds)


def rouge_l(preds: dict[str, str], refs: dict[str, list[str]]) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"])
    return mean(max(scorer.score(r, preds[i])["rougeL"].fmeasure for r in refs[i]) for i in preds)


def cider(preds: dict[str, str], refs: dict[str, list[str]]) -> float:
    """CIDEr needs several images to build document frequencies; use it on a batch."""
    from pycocoevalcap.cider.cider import Cider

    gts = {i: [" ".join(tokenize(r)) for r in refs[i]] for i in preds}
    res = {i: [" ".join(tokenize(preds[i]))] for i in preds}
    score, _ = Cider().compute_score(gts, res)
    return float(score)


def clip_score(images: list[Image.Image], captions: list[str], model_name: str | None = None) -> float:
    """Reference-free CLIPScore = 2.5 * max(cos(image, caption), 0), averaged."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    name = model_name or settings.clip_model
    device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (name, device)
    if key not in _CLIP_CACHE:
        _CLIP_CACHE[key] = (CLIPModel.from_pretrained(name).to(device).eval(), CLIPProcessor.from_pretrained(name))
    model, proc = _CLIP_CACHE[key]
    inputs = proc(text=captions, images=images, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.inference_mode():
        out = model(**inputs)
        img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        cos = (img * txt).sum(dim=-1)
    return float((2.5 * cos.clamp(min=0)).mean())


def compute_all(preds, refs, images=None, include_cider=True) -> dict[str, float]:
    """Run every available metric; failures are reported as NaN-free skips."""
    results: dict[str, float] = {}
    for name, fn in (("BLEU", lambda: bleu(preds, refs)), ("METEOR", lambda: meteor(preds, refs)),
                     ("ROUGE-L", lambda: rouge_l(preds, refs))):
        try:
            out = fn()
            results.update(out) if isinstance(out, dict) else results.__setitem__(name, out)
        except Exception as exc:
            print(f"[evaluation] {name} skipped: {exc}")
    if include_cider:
        try:
            results["CIDEr"] = cider(preds, refs)
        except Exception as exc:
            print(f"[evaluation] CIDEr skipped: {exc}")
    if images is not None:
        try:
            results["CLIPScore"] = clip_score(images, [preds[i] for i in preds])
        except Exception as exc:
            print(f"[evaluation] CLIPScore skipped: {exc}")
    return results
