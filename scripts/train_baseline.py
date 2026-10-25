"""Train the CNN-LSTM baseline on a bounded Flickr8k subset."""
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from captioning.baseline import Vocabulary, build_model


class FlickrDataset(Dataset):
    def __init__(self, root: Path, limit: int):
        caption_file = next(root.rglob("*.txt"), None)
        if caption_file is None:
            raise FileNotFoundError(f"No Flickr8k captions .txt found below {root}; run scripts/download_flickr8k.py")
        rows = []
        with caption_file.open(encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 2:
                    rows.append((row[0].split("#")[0], row[1]))
        chosen = list(dict.fromkeys(name for name, _ in rows))[:limit]
        self.captions = [(name, caption) for name, caption in rows if name in set(chosen)]
        self.root = root
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
        self.vocab = Vocabulary()
        self.vocab.build([caption for _, caption in self.captions])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        name, caption = self.captions[idx]
        path = next(self.root.rglob(name), None)
        if path is None:
            raise FileNotFoundError(name)
        image = self.transform(Image.open(path).convert("RGB"))
        encoded = torch.tensor(self.vocab.encode(caption))
        return image, encoded


def collate(batch):
    from torch.nn.utils.rnn import pad_sequence
    images, captions = zip(*batch)
    return torch.stack(images), pad_sequence(captions, batch_first=True, padding_value=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/flickr8k"))
    parser.add_argument("--limit", type=int, default=500, help="number of unique images; default 500")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results/baseline.pt"))
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = FlickrDataset(args.data, args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    model = build_model(len(ds.vocab.itos)).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images, captions in loader:
            images, captions = images.to(device), captions.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images, captions).reshape(-1, len(ds.vocab.itos)), captions[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / max(len(loader), 1):.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "vocab": ds.vocab.itos, "limit": args.limit}, args.output)
    print(f"Saved {args.output}; unique training images={args.limit} (subset of Flickr8k)")


if __name__ == "__main__":
    main()
