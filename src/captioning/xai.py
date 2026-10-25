"""Explainable-AI module: Grad-CAM and attention rollout for the BLIP vision encoder.

STATUS: first working implementation for BLIP (ViT) models. It still needs to be
validated visually on real images during the Design & Development phase.
BLIP-2 support is planned (its vision tower is wired differently).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .utils import overlay_heatmap


class VisionExplainer:
    def __init__(self, caption_model):
        if caption_model.is_blip2:
            raise NotImplementedError("XAI is implemented for BLIP models; BLIP-2 is a planned extension.")
        self.cm = caption_model

    # ---------------------------------------------------------------- Grad-CAM
    def grad_cam(self, image: Image.Image, caption: str, target_word: str | None = None):
        """Return (heatmap [0,1] ndarray, overlay PIL image).

        The score is the log-likelihood of the caption tokens (or only of
        ``target_word`` if given) given the image. Gradients of that score w.r.t.
        the last ViT block show which patches mattered.
        """
        import torch

        proc, model, dev = self.cm.processor, self.cm.model, self.cm.device
        image = image.convert("RGB")
        inputs = proc(images=image, text=caption, return_tensors="pt").to(dev)
        ids = inputs["input_ids"]

        store: dict = {}

        def fwd_hook(_module, _inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            hs.retain_grad()
            store["act"] = hs

        handle = model.vision_model.encoder.layers[-1].register_forward_hook(fwd_hook)
        try:
            model.zero_grad(set_to_none=True)
            pixel_values = inputs["pixel_values"].clone().requires_grad_(True)
            with torch.enable_grad():
                out = model(
                    pixel_values=pixel_values,
                    input_ids=ids,
                    attention_mask=inputs.get("attention_mask"),
                )
                logits = out.logits[:, :-1, :].float()
                targets = ids[:, 1 : 1 + logits.shape[1]]
                logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                mask = self._token_mask(ids[0, 1 : 1 + logits.shape[1]], target_word).to(logp.device)
                score = (logp[0] * mask).sum()
                score.backward()
        finally:
            handle.remove()

        act = store["act"][0, 1:]  # drop CLS token -> (N, C)
        grad = store["act"].grad[0, 1:]
        weights = grad.mean(dim=0)  # (C,)
        cam = torch.relu((act * weights).sum(dim=-1)).detach().cpu().numpy()
        side = int(np.sqrt(cam.shape[0]))
        cam = cam.reshape(side, side)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, overlay_heatmap(image, cam)

    def _token_mask(self, token_ids, target_word):
        import torch

        tok = self.cm.processor.tokenizer
        ids = token_ids.tolist()
        tokens = tok.convert_ids_to_tokens(ids)
        special = set(tok.all_special_tokens)
        mask = torch.tensor([0.0 if t in special else 1.0 for t in tokens])
        if target_word:
            # Match the word with the model tokenizer so split/subword tokens
            # (e.g. "pupp" + "##y") are attributed together.
            word_ids = tok.encode(target_word.strip(), add_special_tokens=False)
            for start in range(len(ids) - len(word_ids) + 1):
                if word_ids and ids[start : start + len(word_ids)] == word_ids:
                    hit = torch.zeros(len(ids))
                    hit[start : start + len(word_ids)] = 1.0
                    return hit
            # Some tokenizers normalize punctuation/case differently. Fall
            # back to normalized token text, retaining all matching pieces.
            word = target_word.lower().strip().lstrip("#")
            hit = torch.tensor([float(word and word in t.lower().lstrip("#")) for t in tokens])
            if hit.sum() > 0:
                return hit
        return mask if mask.sum() > 0 else torch.ones(len(tokens))

    # ------------------------------------------------------- Attention rollout
    def attention_rollout(self, image: Image.Image):
        """Return (heatmap [0,1] ndarray, overlay PIL image) from ViT self-attention."""
        import torch

        proc, model, dev = self.cm.processor, self.cm.model, self.cm.device
        image = image.convert("RGB")
        inputs = proc(images=image, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.vision_model(pixel_values=inputs["pixel_values"], output_attentions=True)
        attentions = out.attentions
        if not attentions or attentions[0] is None:
            raise RuntimeError("Attention weights unavailable. Load the model with attn_implementation='eager'.")

        n = attentions[0].shape[-1]
        result = torch.eye(n, device=attentions[0].device)
        for att in attentions:
            a = att.mean(dim=1)[0] + torch.eye(n, device=att.device)
            a = a / a.sum(dim=-1, keepdim=True)
            result = a @ result
        cam = result[0, 1:].detach().cpu().numpy()
        side = int(np.sqrt(cam.shape[0]))
        cam = cam.reshape(side, side)
        return cam, overlay_heatmap(image, cam)
