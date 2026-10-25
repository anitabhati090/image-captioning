"""Small helpers shared by the app and scripts."""

from __future__ import annotations

import numpy as np
from PIL import Image


def load_rgb(path_or_file) -> Image.Image:
    """Open an image (path or file-like) and convert to RGB."""
    return Image.open(path_or_file).convert("RGB")


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Blend a [0, 1] heat-map (H x W) over the original image using the 'jet' colormap."""
    from matplotlib import colormaps

    cam = np.asarray(cam, dtype=np.float32)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heat = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC)
    heat_rgb = colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    heat_img = Image.fromarray((heat_rgb * 255).astype(np.uint8))
    return Image.blend(image.convert("RGB"), heat_img, alpha)
