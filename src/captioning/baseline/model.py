"""Small frozen ResNet-50 and LSTM image captioner."""
from __future__ import annotations

from collections import Counter


class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.itos = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.stoi = {word: i for i, word in enumerate(self.itos)}
        self.min_freq = min_freq

    def build(self, captions: list[str]) -> None:
        words = Counter(w.lower() for c in captions for w in c.split())
        self.itos.extend(w for w, n in words.items() if n >= self.min_freq and w not in self.stoi)
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def encode(self, caption: str) -> list[int]:
        return [self.stoi["<bos>"]] + [self.stoi.get(w.lower(), 3) for w in caption.split()] + [self.stoi["<eos>"]]

    def decode(self, ids: list[int]) -> str:
        words = []
        for idx in ids:
            word = self.itos[idx]
            if word == "<eos>":
                break
            if word not in {"<bos>", "<pad>"}:
                words.append(word)
        return " ".join(words)


def build_model(vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512):
    import torch
    from torch import nn
    from torchvision.models import ResNet50_Weights, resnet50

    class Captioner(nn.Module):
        def __init__(self):
            super().__init__()
            encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.encoder = nn.Sequential(*list(encoder.children())[:-1])
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.project = nn.Linear(2048, embed_dim)
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.output = nn.Linear(hidden_dim, vocab_size)

        def forward(self, images, captions):
            self.encoder.eval()  # Keep frozen batch-normalization statistics fixed.
            features = self.project(self.encoder(images).flatten(1)).unsqueeze(1)
            embeds = self.embedding(captions[:, :-1])
            logits, _ = self.lstm(torch.cat((features, embeds), dim=1))
            return self.output(logits[:, :captions.size(1) - 1])

        @torch.inference_mode()
        def caption(self, image, vocab, device="cpu", max_length=30):
            self.eval()
            x = image.unsqueeze(0).to(device)
            feature = self.project(self.encoder(x).flatten(1)).unsqueeze(1)
            token = torch.tensor([[vocab.stoi["<bos>"]]], device=device)
            state = None
            result = []
            for _ in range(max_length):
                inp = feature if state is None else self.embedding(token[:, -1:])
                out, state = self.lstm(inp, state)
                nxt = self.output(out[:, -1]).argmax(-1)
                result.append(int(nxt.item()))
                if result[-1] == vocab.stoi["<eos>"]:
                    break
                token = torch.cat((token, nxt[:, None]), dim=1)
            return vocab.decode(result)

    return Captioner()
