"""CNN-LSTM captioning baseline."""

from pathlib import Path

from .model import Vocabulary, build_model

class BaselineCaptioner:
    """Load a trained checkpoint and expose ``caption(image) -> str``."""

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        import torch
        from torchvision import transforms

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.vocab = Vocabulary()
        self.vocab.itos = saved["vocab"]
        self.vocab.stoi = {word: idx for idx, word in enumerate(self.vocab.itos)}
        self.model = build_model(len(self.vocab.itos)).to(self.device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

    def caption(self, image, **_kwargs) -> str:
        return self.model.caption(self.transform(image.convert("RGB")), self.vocab, self.device)


__all__ = ["BaselineCaptioner", "Vocabulary", "build_model"]
