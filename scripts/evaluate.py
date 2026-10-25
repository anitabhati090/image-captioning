"""Compare BLIP and the trained CNN-LSTM on one held-out annotation set."""
import argparse
import csv
import json
import re

import matplotlib.pyplot as plt
import torch

from captioning import evaluation
from captioning.baseline import BaselineCaptioner
from captioning.config import settings
from captioning.model import CaptionModel
from captioning.styles import GENERATION_KWARGS, STYLES, apply_style
from captioning.utils import load_rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=settings.caption_model)
    parser.add_argument("--baseline", default="results/baseline.pt")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-xai", action="store_true")
    args = parser.parse_args()
    ann_path = settings.eval_dir / "annotations.json"
    if not ann_path.exists():
        raise SystemExit(f"Missing {ann_path}; see data/README.md and prepare held-out annotations first.")
    refs_all = json.loads(ann_path.read_text(encoding="utf-8"))
    names = list(refs_all)[:args.limit or None]
    refs = {name: refs_all[name] for name in names}
    images = [load_rgb(settings.eval_dir / "images" / name) for name in names]
    device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
    models = [("BLIP", CaptionModel(args.model))]
    baseline_path = settings.results_dir / "baseline.pt" if args.baseline == "results/baseline.pt" else __import__("pathlib").Path(args.baseline)
    if baseline_path.exists():
        models.append(("CNN-LSTM", BaselineCaptioner(baseline_path, device)))
    else:
        print(f"[evaluation] CNN-LSTM skipped: checkpoint missing at {baseline_path}; train with scripts/train_baseline.py")
    scores_rows = []
    results = {}
    xai_caption = None
    for model_name, model in models:
        results[model_name] = {}
        for style in STYLES:
            base_captions = {}
            for name, image in zip(names, images):
                kwargs = GENERATION_KWARGS[style] if model_name == "BLIP" else {}
                base_captions[name] = model.caption(image, **kwargs)
                print(f"{model_name} / {style} {name}: {base_captions[name]}")
            preds = {name: apply_style(style, text) for name, text in base_captions.items()}
            row = {"model": model_name, "style": style}
            row.update(evaluation.compute_all(preds, refs, images=images))
            scores_rows.append(row)
            results[model_name][style] = preds
            if model_name == "BLIP" and style == "standard":
                xai_caption = base_captions
            print(f"{model_name} / {style}: { {k: round(v, 4) for k, v in row.items() if isinstance(v, float)} }")
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = settings.results_dir / "metrics.csv"
    metric_names = sorted({k for row in scores_rows for k in row if k not in {"model", "style"}})
    with metrics_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "style", *metric_names])
        writer.writeheader()
        writer.writerows(scores_rows)
    if scores_rows and metric_names:
        fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 4), squeeze=False)
        for ax, metric in zip(axes[0], metric_names):
            for model_name in results:
                subset = [r for r in scores_rows if r["model"] == model_name]
                vals = [r.get(metric, float("nan")) for r in subset]
                ax.plot([r["style"] for r in subset], vals, marker="o", label=model_name)
            ax.set_title(metric); ax.tick_params(axis="x", rotation=30)
        axes[0][0].legend()
        fig.tight_layout(); fig.savefig(settings.results_dir / "comparison.png", dpi=160); plt.close(fig)
    if not args.skip_xai and "BLIP" in results and names:
        from captioning.xai import VisionExplainer
        explainer = VisionExplainer(models[0][1])
        out_dir = settings.results_dir / "xai_examples"
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, (name, image) in enumerate(zip(names[:3], images[:3])):
            caption = xai_caption[name]
            words = re.findall(r"[A-Za-z]+", caption)
            if words:
                _, overlay = explainer.grad_cam(image, caption, target_word=words[0])
                overlay.save(out_dir / f"{idx + 1}_{name}")
    print(f"Saved {metrics_file} and {settings.results_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
